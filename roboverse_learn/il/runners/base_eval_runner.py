
from roboverse_learn.il.configs.base_config import BasePolicyCfg

try:
    from curobo.types.math import Pose
    from metasim.utils.kinematics import get_curobo_models
    from pytorch3d import transforms
except ImportError:
    pass

import sys

import torch
from loguru import logger as log
from metasim.scenario.scenario import ScenarioCfg
from roboverse_learn.il.utils.pytorch_util import dict_apply


class BaseEvalRunner:
    """Base class to run a policy, based on the policyCFG it preprocesses the observation to
    match the policy's input requirements and postprocesses the action into joint space to match the policy's action type
    """

    def __init__(self, runner, scenario: ScenarioCfg, num_envs: int = 1, **kwargs):
        self.num_envs = num_envs
        self.scenario = scenario
        self.policy_cfg: BasePolicyCfg | None = None
        self.step = 0
        self.device = kwargs.get("device", "cuda:0")
        self.task_name = kwargs.get("task_name")
        self._send_vel_target = kwargs.get("send_vel_target", False)
        # Finite-difference velocity feedforward (eval-time, policy-agnostic).
        # Default off => original behaviour unchanged.
        self._fd_vel_target = kwargs.get("fd_vel_target", False)
        self._fd_dt = None
        # Discrete (binary) gripper actuation switch (eval-time, policy-agnostic).
        # Default off => continuous gripper => original behaviour unchanged.
        self._gripper_binarize = kwargs.get("gripper_binarize", False)
        self._gripper_binarize_threshold = kwargs.get("gripper_binarize_threshold", 0.02)
        self._init_policy(runner, **kwargs)
        self.robot_ik = None
        self.curobo_n_dof = None
        self.action_cache = []
        self.velocity_cache = []
        self._pending_velocity_chunk = None
        self.last_velocity = None
        self.last_action_tensor = None
        self.last_call_was_inference = False  # Whether the last get_action() triggered a real model forward pass
        self.__post_init__()

    def _init_policy(self, **kwargs):
        """Method for subclasses to inherit to load in the policy"""
        raise NotImplementedError

    def __post_init__(self):
        if self.policy_cfg.action_config.action_type == "ee":
            *_, self.robot_ik = get_curobo_models(self.scenario.robots[0])
            self.curobo_n_dof = len(self.robot_ik.robot_config.cspace.joint_names)
            self.ee_n_dof = len(self.scenario.robots[0].gripper_open_q)

        if self._fd_vel_target:
            if self.policy_cfg.action_config.temporal_agg:
                raise ValueError(
                    "[FD Velocity] fd_vel_target=True is not supported with "
                    "temporal_agg=True: the temporal-agg branch emits single "
                    "ensembled actions and never produces a velocity chunk, so "
                    "the run would silently apply velocity-mode PD gains while "
                    "sending no velocity target."
                )
            if self._gripper_binarize:
                log.warning("[FD Velocity] gripper_binarize=True: finger position "
                            "targets are snapped to open/close while FD finger "
                            "velocities follow the policy's continuous output "
                            "(same convention as FLASH-G's analytic velocity).")
            # dt between successive cached position targets as executed by the
            # sim: control_dt = decimation * physics_dt, and each cached target
            # is held for action_set_steps env.step() calls (2 for ee, 1 for
            # joint_pos) — mirrors evaluate() in default_runner.py.
            decimation = self.scenario.decimation
            if self.scenario.sim_params.dt is not None:
                physics_dt = self.scenario.sim_params.dt
            else:
                physics_dt = 0.015 / decimation
            action_set_steps = (
                2 if self.policy_cfg.action_config.action_type == "ee" else 1
            )
            self._fd_dt = decimation * physics_dt * action_set_steps
            log.info(f"[FD Velocity] Finite-difference velocity feedforward ENABLED "
                     f"(dt={self._fd_dt:.4f}s, central difference interior / "
                     f"one-sided at chunk ends, in-plan only, "
                     f"send_vel_target={self._send_vel_target})")

        if self.policy_cfg.action_config.temporal_agg:
            self.all_time_actions = torch.zeros(
                [
                    self.num_envs,
                    self.scenario.episode_length,
                    self.scenario.episode_length
                    + self.policy_cfg.action_config.action_chunk_steps,
                    self.policy_cfg.action_config.action_dim,
                ],
                device=self.device,
            )
            self.k = 0.01

    def process_obs(self, obs):
        """
        Processes the observation to be used by the policy, according to the observation the policy is configured to use.
        """
        obs = dict_apply(
            obs,
            lambda x: x.to(device=self.device) if isinstance(x, torch.Tensor) else x,
        )
        obs_dict = {}

        if self.policy_cfg.obs_config.norm_image:
            obs_dict["head_cam"] = obs["rgb"].permute(0, 3, 1, 2) / 255.0
        else:
            obs_dict["head_cam"] = obs["rgb"]
        if self.policy_cfg.obs_config.obs_type == "joint_pos":
            obs_dict["agent_pos"] = obs["joint_qpos"]
        if self.policy_cfg.obs_config.obs_type == "ee":
            robot_ee_state = obs["robot_ee_state"]
            robot_root_state = obs["robot_root_state"]
            robot_pos, robot_quat = robot_root_state[:, 0:3], robot_root_state[:, 3:7]
            curr_ee_pos, curr_ee_quat = robot_ee_state[:, 0:3], robot_ee_state[:, 3:7]
            curr_ee_pos_local = transforms.quaternion_apply(
                transforms.quaternion_invert(robot_quat), curr_ee_pos - robot_pos
            )
            curr_ee_quat_local = transforms.quaternion_multiply(
                transforms.quaternion_invert(robot_quat), curr_ee_quat
            )

            if self.policy_cfg.obs_config.ee_cfg.gripper_rep == "q_pos":
                gripper_state = obs["joint_qpos"][:, -2:]
            else:
                gripper_state = obs["robot_ee_state"][:, -1]

            if self.policy_cfg.obs_config.ee_cfg.rotation_rep == "quaternion":
                curr_ee_rot_local = curr_ee_quat_local
            else:
                curr_ee_rot_local = transforms.matrix_to_euler_angles(
                    transforms.quaternion_to_matrix(curr_ee_quat_local),
                    convention="XYZ",
                )

            obs_dict["agent_pos"] = torch.cat(
                [curr_ee_pos_local, curr_ee_rot_local, gripper_state], dim=1
            )

        if self.policy_cfg.obs_config.obs_padding > 0:
            padding_len = (
                self.policy_cfg.obs_config.obs_padding - obs_dict["agent_pos"].shape[1]
            )
            padding = torch.zeros(self.num_envs, padding_len, device=self.device)
            obs_dict["agent_pos"] = torch.cat([obs_dict["agent_pos"], padding], dim=1)

        assert obs_dict["agent_pos"].shape == (
            self.num_envs,
            self.policy_cfg.obs_config.obs_dim,
        )
        # flush unused keys
        obs_dict = {
            k: v
            for k, v in obs_dict.items()
            if k in self.policy_cfg.obs_config.obs_keys
        }
        return obs_dict

    def action_to_dict(self, curr_action: torch.Tensor, curr_velocity: torch.Tensor | None = None):
        """
        Converts action tensor (and optional velocity tensor) to dict with joint keys.
        """
        action_nested_list = curr_action.tolist()
        vel_nested_list = curr_velocity.tolist() if curr_velocity is not None else None
        sorted_joint_names = sorted(self.scenario.robots[0].joint_limits.keys())
        robot_name = self.scenario.robots[0].name

        actions = []
        for i in range(self.num_envs):
            robot_action = {
                "dof_pos_target": {
                    joint_name: action_nested_list[i][index]
                    for index, joint_name in enumerate(sorted_joint_names)
                }
            }
            if vel_nested_list is not None:
                robot_action["dof_vel_target"] = {
                    joint_name: vel_nested_list[i][index]
                    for index, joint_name in enumerate(sorted_joint_names)
                }
            actions.append({robot_name: robot_action})
        return actions

    def _binarize_gripper(self, curr_action: torch.Tensor) -> torch.Tensor:
        """Discretize the gripper (finger) joint targets of the FINAL qpos action to
        a binary {open, close} command.

        No-op unless eval was launched with ``gripper_binarize=True``. This is an
        eval-time, POLICY-AGNOSTIC post-processing step applied at the shared
        action-apply layer (after temporal aggregation / interpolation, right before
        the action dict is sent to the sim): it snaps ONLY the gripper dims and leaves
        every other dim — and every policy — untouched. Default off => continuous
        gripper => bit-for-bit the original behaviour. Only active for
        ``action_type == 'joint_pos'`` (ee mode already handles the gripper via
        ``ee_cfg.gripper_rep``).

        The gripper indices are resolved by joint NAME (``robot.ee_joint_names``) in the
        same alphabetically-sorted order used by :meth:`action_to_dict`, so this stays
        correct regardless of the action-vector layout. A single open/close decision is
        made per env from the mean finger target (mirroring the ee ``gripper_rep=='strength'``
        rule), then the canonical ``gripper_open_q`` / ``gripper_close_q`` finger vector
        is written back.
        """
        if not self._gripper_binarize:
            return curr_action
        if self.policy_cfg.action_config.action_type != "joint_pos":
            return curr_action

        robot = self.scenario.robots[0]
        ee_joint_names = getattr(robot, "ee_joint_names", None)
        if not ee_joint_names:
            log.warning(
                "[gripper_binarize] robot has no 'ee_joint_names'; skipping gripper "
                "binarization (continuous gripper retained)."
            )
            return curr_action

        sorted_joint_names = sorted(robot.joint_limits.keys())
        grip_idx = [sorted_joint_names.index(n) for n in ee_joint_names]
        open_q = torch.tensor(robot.gripper_open_q, device=curr_action.device, dtype=curr_action.dtype)
        close_q = torch.tensor(robot.gripper_close_q, device=curr_action.device, dtype=curr_action.dtype)
        thr = float(self._gripper_binarize_threshold)

        # one open/close decision per env, from the mean finger target
        is_open = curr_action[:, grip_idx].mean(dim=1, keepdim=True) >= thr  # (num_envs, 1)
        snapped = torch.where(is_open, open_q.unsqueeze(0), close_q.unsqueeze(0))  # (num_envs, n_finger)
        curr_action = curr_action.clone()
        curr_action[:, grip_idx] = snapped
        return curr_action

    def get_temporal_agg_action(self, action_chunk):
        """
        Implements temporal ensembline, as in Aloha ACT. Takes in a current prediction chunk and returns a single ensembled action
        """
        assert action_chunk.shape == (
            self.policy_cfg.action_config.action_chunk_steps,
            self.num_envs,
            self.policy_cfg.action_config.action_dim,
        )

        # Put envs dimension first
        self.all_time_actions[
            :,
            self.step,
            self.step : self.step + self.policy_cfg.action_config.action_chunk_steps,
        ] = action_chunk.transpose(0, 1)

        actions_for_curr_step = self.all_time_actions[:, :, self.step]

        actions_populated = torch.all(
            torch.all(actions_for_curr_step != 0, dim=2), dim=0
        )
        actions_for_curr_step = actions_for_curr_step[:, actions_populated]

        time_indices = torch.arange(
            actions_for_curr_step.shape[1],
            device=actions_for_curr_step.device,
            dtype=torch.float,
        )
        exp_weights = torch.exp(self.k * time_indices)
        exp_weights = exp_weights / exp_weights.sum()

        weighted_actions = actions_for_curr_step * exp_weights.unsqueeze(-1).unsqueeze(
            0
        )

        raw_action = weighted_actions.sum(dim=1)

        return raw_action

    def get_action(self, obs):
        """Returns a single action to be directly executed. For action chunking policies it either uses an previsouly
        predicted action chunk, or if it has exausted all of those actions, it queries the model for a new chunk and returns the first one
        """
        # Always update observation history for policies that need continuous observation history
        # This is critical for action-to-action flow policies like VITA
        processed_obs = self.process_obs(obs)
        self.update_obs(processed_obs)

        if len(self.action_cache) > 0:
            curr_action = self.action_cache.pop(0)
            curr_velocity = self.velocity_cache.pop(0) if self.velocity_cache else None
            self.last_call_was_inference = False
        else:
            action_chunk = self.predict_action(
                None  # Don't pass obs again since we already updated it
            )  # shape: (action_chunk_steps, num_envs, action_dim)
            if self.policy_cfg.action_config.temporal_agg:
                curr_action = self.get_temporal_agg_action(action_chunk)
                curr_action = self.process_action([curr_action], obs)[0]
                curr_velocity = None
            else:
                qpos_action = self.process_action(action_chunk, obs)
                assert (
                    len(qpos_action) == self.policy_cfg.action_config.action_chunk_steps
                ), (
                    f"Expected {self.policy_cfg.action_config.action_chunk_steps} actions, got {len(qpos_action)}"
                )
                # Finite-difference velocity feedforward (opt-in): derive the
                # velocity chunk from the FINAL position-target chunk (after
                # IK / interpolation), then hand it through the same
                # _pending_velocity_chunk -> velocity_cache path used by
                # policy-provided velocities.  Overrides any policy-provided
                # velocity so the velocity SOURCE is the only difference
                # between runs.
                if self._fd_vel_target:
                    self._pending_velocity_chunk = self._finite_diff_velocity(qpos_action)

                self.action_cache = qpos_action
                curr_action = self.action_cache.pop(0)

                if self._pending_velocity_chunk is not None:
                    self.velocity_cache = [v.to(self.device) for v in self._pending_velocity_chunk]
                    self._pending_velocity_chunk = None
                    curr_velocity = self.velocity_cache.pop(0)
                else:
                    curr_velocity = None
            self.last_call_was_inference = True

        self.step += 1
        assert curr_action.shape == (
            self.num_envs,
            len(self.scenario.robots[0].joint_limits.keys()),
        ), (
            f"Expected num_envs X n_dof : {self.num_envs} X {len(self.scenario.robots[0].joint_limits.keys())}, got {curr_action.shape} instead"
        )

        # Optional eval-time binary-gripper actuation (no-op unless enabled).
        # Applied to the FINAL qpos, after temporal-agg/interpolation, so the value
        # sent to the sim (and logged in last_action_tensor) is the binarized command.
        curr_action = self._binarize_gripper(curr_action)

        self.last_velocity = curr_velocity
        self.last_action_tensor = curr_action
        vel_for_sim = curr_velocity if self._send_vel_target else None
        actions = self.action_to_dict(curr_action, vel_for_sim)
        return actions

    def predict_action(self, obs):
        raise NotImplementedError

    def _finite_diff_velocity(self, qpos_chunk):
        """Velocity feedforward via finite differences of the position-target chunk.

        Alignment mirrors the FLASH-G convention (velocity[k] = instantaneous
        velocity of the CURRENT plan at position target k): central differences
        on interior points, one-sided differences at both chunk ends.  The
        previous chunk's tail is deliberately NOT used as a neighbour:
        baselines re-anchor each new chunk to the observed state, so
        differencing across the replan boundary would fold the replan jump
        (erased tracking error) into the feedforward — an artifact FLASH-G's
        plan-consistent analytic velocity does not have.  A single-element
        chunk has no in-plan neighbour and yields zero velocity.

        qpos_chunk: list of (num_envs, n_dof) absolute joint targets [rad]
        returns:    list of (num_envs, n_dof) velocity targets [rad/s]
        """
        q = torch.stack([a.to(self.device) for a in qpos_chunk], dim=0).to(torch.float32)
        n = q.shape[0]
        dt = self._fd_dt
        vel = torch.zeros_like(q)
        if n >= 3:
            vel[1:-1] = (q[2:] - q[:-2]) / (2.0 * dt)
        if n >= 2:
            vel[0] = (q[1] - q[0]) / dt
            vel[-1] = (q[-1] - q[-2]) / dt
        return [vel[i] for i in range(n)]

    def _solve_ik(self, action, curr_ee_pos_local, curr_ee_quat_local, curr_robot_q):
        """Solves IK for the given action end-effector action, in either delta or absolute control"""
        assert action.shape == (
            self.num_envs,
            self.policy_cfg.action_config.action_dim,
        ), (
            f"Expected num_envs X action_dim : {self.num_envs} X {self.policy_cfg.action_config.action_dim}, got {action.shape} instead"
        )
        if self.policy_cfg.action_config.ee_cfg.rotation_rep == "quaternion":
            ee_quat_action = action[:, 3:7]
            quat_norm = torch.norm(ee_quat_action, dim=1, keepdim=True)
            ee_quat_action = ee_quat_action / (quat_norm + 1e-5)
        else:
            ee_quat_action = transforms.matrix_to_quaternion(
                transforms.euler_angles_to_matrix(action[:, 3:6], "XYZ")
            )

        if self.policy_cfg.action_config.delta:
            ee_pos_target = curr_ee_pos_local + action[:, :3]
            ee_quat_target = transforms.quaternion_multiply(
                curr_ee_quat_local, ee_quat_action
            )
        else:
            ee_pos_target = action[:, :3]
            ee_quat_target = ee_quat_action

        # Solve IK
        seed_config = (
            curr_robot_q[:, : self.curobo_n_dof]
            .unsqueeze(1)
            .tile([1, self.robot_ik._num_seeds, 1])
        )
        result = self.robot_ik.solve_batch(
            Pose(ee_pos_target.cuda(0), ee_quat_target.cuda(0)),
            seed_config=seed_config.cuda(0),
        )

        if self.policy_cfg.action_config.ee_cfg.gripper_rep == "strength":
            gripper_pos = 1 - action[:, -1]
            gripper_widths = torch.zeros(
                self.num_envs, self.ee_n_dof, device=self.device
            )
            for i in range(self.num_envs):
                if gripper_pos[i] < 0.5:
                    gripper_widths[i] = torch.tensor(
                        self.scenario.robots[0].gripper_close_q, device=self.device
                    )
                else:
                    gripper_widths[i] = torch.tensor(
                        self.scenario.robots[0].gripper_open_q, device=self.device
                    )
        else:
            gripper_widths = action[:, -self.ee_n_dof :]

        q = curr_robot_q.clone()
        ik_succ = result.success.squeeze(1).to(self.device)
        if (~ik_succ).any():
            log.warning(f"IK failed: {ik_succ}")
            log.info("Trying to POS delta: ", action[:, :3])

        q[ik_succ, : self.curobo_n_dof] = result.solution.to(self.device)[
            ik_succ, 0
        ].clone()
        q[:, -self.ee_n_dof :] = gripper_widths
        return q

    def process_action(self, action_chunk, obs):
        """
        Processes a chunk of actions into joint positions.
        """
        action_chunk = [a.to(self.device) for a in action_chunk]
        for a in action_chunk:
            assert a.shape == (
                self.num_envs,
                self.policy_cfg.action_config.action_dim,
            ), (
                f"Expected num_envs X action_dim : {self.num_envs} X {self.policy_cfg.action_config.action_dim}, got {a.shape} instead"
            )
        if self.policy_cfg.action_config.action_type == "joint_pos":
            qpos_action_chunk = action_chunk
        elif self.policy_cfg.action_config.action_type == "ee":
            qpos_action_chunk = []
            robot_ee_state = obs["robot_ee_state"].to(self.device)
            robot_root_state = obs["robot_root_state"].to(self.device)
            robot_pos, robot_quat = robot_root_state[:, 0:3], robot_root_state[:, 3:7]
            curr_ee_pos, curr_ee_quat = robot_ee_state[:, 0:3], robot_ee_state[:, 3:7]
            curr_ee_pos_local = transforms.quaternion_apply(
                transforms.quaternion_invert(robot_quat), curr_ee_pos - robot_pos
            )
            curr_ee_quat_local = transforms.quaternion_multiply(
                transforms.quaternion_invert(robot_quat), curr_ee_quat
            )
            curr_robot_q = obs["joint_qpos"].to(self.device)
            for action in action_chunk:
                target_qpos = self._solve_ik(
                    action, curr_ee_pos_local, curr_ee_quat_local, curr_robot_q
                )
                qpos_action_chunk.append(target_qpos)

        if self.policy_cfg.action_config.interpolate_chunk:
            return self._interpolate_chunk(
                obs["joint_qpos"].to(self.device), qpos_action_chunk
            )
        else:
            return qpos_action_chunk

    def _interpolate_chunk(self, curr_qpos, qpos_action_chunk):
        """Smoothly interpolates between the current state and final predicted action of the chunk"""
        last_action = qpos_action_chunk[-1]
        assert curr_qpos.shape == last_action.shape, (
            f"Expected {curr_qpos.shape} and {last_action.shape} to be the same, got {curr_qpos.shape} and {last_action.shape} instead"
        )

        return [
            curr_qpos
            + (last_action - curr_qpos)
            * (i + 1)
            / self.policy_cfg.action_config.action_chunk_steps
            for i in range(self.policy_cfg.action_config.action_chunk_steps)
        ]

    def update_obs(self, current_obs):
        """Update observation history. Override in subclass if needed."""
        pass  # Default implementation does nothing; subclasses should override

    def reset(self):
        self.action_cache = []
        self.velocity_cache = []
        self._pending_velocity_chunk = None
        self.last_velocity = None
        self.last_action_tensor = None
        self.step = 0
