from collections import deque

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from roboverse_learn.il.configs.base_config import DiffusionPolicyCfg
from roboverse_learn.il.runners.base_eval_runner import BaseEvalRunner
from roboverse_learn.il.runners.default_runner import DefaultRunner

# Keys removed from ActionCfg/ObsCfg in newer code versions.  Old checkpoints
# still carry them in their saved config; strip before passing to from_dict so
# the dataclass update doesn't KeyError on unknown attributes.
_REMOVED_ACTION_CFG_KEYS = ("temporal_ensemble", "temporal_ensemble_decay")


def _drop_removed_keys(cfg_section, removed_keys):
    """Return a plain dict copy of cfg_section with removed_keys stripped."""
    d = OmegaConf.to_container(cfg_section, resolve=True) if OmegaConf.is_config(cfg_section) else dict(cfg_section)
    for k in removed_keys:
        d.pop(k, None)
    return d


class DefaultEvalRunner(BaseEvalRunner):
    """Runner for a diffusion policy, loads in a workspace and policy from checkpoint, and overrides some of the
    PolicyCFG attributes to match how the policy was trained
    """

    def _init_policy(self, default_runner: DefaultRunner, **kwargs):
        from loguru import logger as log
        self.task_name = kwargs.get("task_name")
        payload = torch.load(open(kwargs["checkpoint_path"], "rb"), pickle_module=dill)
        cfg = payload["cfg"]

        # Save inference-only overrides from the CURRENT YAML before the model
        # gets replaced by the checkpoint rebuild.  These parameters only affect
        # inference behaviour and should always honour the user's live config.
        # Training-bound params (poly_order, hidden_dim, basis_type, etc.) must
        # come from the checkpoint to stay compatible with saved weights.
        _INFERENCE_ONLY_ATTRS = ["num_inference_steps", "flash_solver"]
        _yaml_overrides = {}
        for attr in _INFERENCE_ONLY_ATTRS:
            val = getattr(default_runner.model, attr, None)
            if val is not None:
                _yaml_overrides[attr] = val

        # Keep a reference to the initial normalizer (needed for any rebuild).
        initial_normalizer = default_runner.model.normalizer

        # Rebuild model from checkpoint's config if architecture differs,
        # so that poly_order, hidden_dim, etc. always match the checkpoint.
        import copy
        try:
            log.info("=" * 60)
            log.info("[Eval] Rebuilding model from CHECKPOINT config...")
            ckpt_model = hydra.utils.instantiate(cfg.policy_config)
            ckpt_model.set_normalizer(initial_normalizer)
            default_runner.model = ckpt_model
            if cfg.train_config.training_params.use_ema:
                default_runner.ema_model = copy.deepcopy(ckpt_model)
            log.info("[Eval] Model rebuilt from checkpoint config successfully.")
            log.info("=" * 60)
        except Exception as e:
            log.warning(f"Could not rebuild model from checkpoint config: {e}. Using current model.")

        try:
            # strict=False: allow missing non-learned buffers (e.g. anchor
            # projection matrices that are now always registered but may be
            # absent in older checkpoints).  These buffers are deterministic
            # — already correctly computed by __init__ — so missing keys
            # from the checkpoint are harmless.
            default_runner.load_payload(payload, exclude_keys=None, include_keys=None, strict=False)
        except RuntimeError as e:
            if "size mismatch" not in str(e):
                raise
            log.warning("[Eval] Config/weight shape mismatch detected — checkpoint "
                        "was likely saved during resumed training with stale config. "
                        "Auto-recovering correct architecture from weights...")
            cfg = self._recover_cfg_from_weights(cfg, payload, initial_normalizer, log)
            ckpt_model = hydra.utils.instantiate(cfg.policy_config)
            ckpt_model.set_normalizer(initial_normalizer)
            default_runner.model = ckpt_model
            if cfg.train_config.training_params.use_ema:
                default_runner.ema_model = copy.deepcopy(ckpt_model)
            default_runner.load_payload(payload, exclude_keys=None, include_keys=None, strict=False)
            log.info("[Eval] Auto-recovery successful.")

        # get policy from workspace
        policy = default_runner.model
        if cfg.train_config.training_params.use_ema:
            policy = default_runner.ema_model

        device = torch.device(self.device)
        policy.to(device)
        policy.eval()
        self.policy = policy

        # Apply inference-only overrides from the current YAML.
        # The checkpoint rebuild uses the saved config which may predate these
        # parameters (falling back to __init__ defaults).  The user's YAML
        # should always take precedence for inference-only switches.
        for attr, val in _yaml_overrides.items():
            if hasattr(self.policy, attr):
                setattr(self.policy, attr, val)
                log.info(f"[Eval] {attr} = {val} (overridden from current YAML)")

        # ---- FLASH C_h robustness probe (eval-time only, default off) ----
        _hn_sigma = float(kwargs.get("eval_history_noise_std", 0.0) or 0.0)
        if _hn_sigma > 0.0:
            if hasattr(self.policy, "eval_history_noise_std"):
                self.policy.eval_history_noise_std = _hn_sigma
                self.policy.eval_history_noise_seed = int(
                    kwargs.get("eval_history_noise_seed", 0) or 0)
                log.info(f"[Eval] C_h perturbation ENABLED: eval_history_noise_std="
                         f"{_hn_sigma} (seed={self.policy.eval_history_noise_seed}, "
                         f"dedicated generator — global RNG untouched)")
            else:
                log.warning(f"[Eval] eval_history_noise_std={_hn_sigma} ignored: "
                            f"policy {type(self.policy).__name__} has no "
                            f"history-coefficient flow start (FLASH only).")
        self.yaml_cfg = cfg
        self.policy_cfg = DiffusionPolicyCfg()

        if "policy_runner" in cfg.eval_config:
            self.policy_cfg.obs_config.from_dict(cfg.eval_config.policy_runner.obs)
            self.policy_cfg.action_config.from_dict(
                _drop_removed_keys(cfg.eval_config.policy_runner.action, _REMOVED_ACTION_CFG_KEYS)
            )
            # [image-only ablation] shape_meta will throw "Key 'agent_pos' is not in struct" when directly use after removing agent_pose
            # Proprioception is removed from global condition; but process_obs is still inserted to the joint vectors and asserts that the dimensionality == obs_dim,
            # so when using the “image-only” option, the system reverts to the number of joints (= the action dimension). Once agent_pos is restored, the program automatically reverts to the if branch without the need for manual intervention.
            if 'agent_pos' in cfg.shape_meta.obs:
                self.policy_cfg.obs_config.obs_dim = cfg.shape_meta.obs.agent_pos.shape[0]
            else:
                self.policy_cfg.obs_config.obs_dim = cfg.shape_meta.action.shape[0]
            self.policy_cfg.action_config.action_dim = cfg.shape_meta.action.shape[0]

        # Use n_action_steps from model (policy attribute) if available, else from checkpoint cfg
        # This handles cases where checkpoint was saved with old config but model uses new settings
        if hasattr(self.policy, 'n_action_steps'):
            self.policy_cfg.action_config.action_chunk_steps = self.policy.n_action_steps
        else:
            self.policy_cfg.action_config.action_chunk_steps = cfg.n_action_steps
        # [image-only ablation] shape_meta will throw "Key 'agent_pos' is not in struct" when directly use after removing agent_pose
        # Proprioception is removed from global condition; but process_obs is still inserted to the joint vectors and asserts that the dimensionality == obs_dim,
        # so when using the “image-only” option, the system reverts to the number of joints (= the action dimension). Once agent_pos is restored, the program automatically reverts to the if branch without the need for manual intervention.
        if 'agent_pos' in cfg.shape_meta.obs:
            self.policy_cfg.obs_config.obs_dim = cfg.shape_meta.obs.agent_pos.shape[0]
        else:
            self.policy_cfg.obs_config.obs_dim = cfg.shape_meta.action.shape[0]
        self.policy_cfg.action_config.action_dim = cfg.shape_meta.action.shape[0]

        # Use n_obs_steps from model if available
        if hasattr(self.policy, 'n_obs_steps'):
            self.obs = deque(maxlen=self.policy.n_obs_steps + 1)
        else:
            self.obs = deque(maxlen=cfg.n_obs_steps + 1)
        self.env = None

        # === FLASH-G Continuous-Time Resampling ===
        #
        # Training and inference operate at different frequencies:
        #   - Training data: expert_freq / downsample_ratio  (e.g. 30/1=30 Hz, 30/4=7.5 Hz)
        #   - Simulator control: 1/control_dt                (e.g. 66.67 Hz)
        #
        # FLASH-G's polynomial q(s) is continuous on s ∈ [0,1], so we can evaluate
        # it at any frequency. At inference we resample at the simulator's native
        # control frequency, regardless of what frequency the training data used.
        #
        # Key formulas:
        #   expert_dt   = downsample_ratio / expert_freq     (time between training frames)
        #   control_dt  = decimation × physics_dt             (simulator control period)
        #   n_dense     = round(n_action_steps × expert_dt / control_dt)
        #   obs_subsample_rate = round(expert_dt / control_dt)
        self._vel_time_scale = 1.0
        self._continuous_resample = False
        self._obs_subsample_rate = 1

        if hasattr(self.policy, 'horizon') and hasattr(self.policy, 'basis_type'):
            H = self.policy.horizon
            n_act = self.policy.n_action_steps
            n_obs = self.policy.n_obs_steps

            # --- Simulator control frequency ---
            decimation = self.scenario.decimation
            if self.scenario.sim_params.dt is not None:
                physics_dt = self.scenario.sim_params.dt
            else:
                physics_dt = 0.015 / decimation
            control_dt = decimation * physics_dt
            self._control_dt = control_dt

            # --- Expert data frequency ---
            # The base recording frequency is the simulator's control frequency
            # (1/control_dt ≈ 66.67 Hz).  The zarr metadata field raw_data_freq_hz
            # is unreliable (some pipelines write 30 Hz there, but the actual
            # per-frame dt in Isaac Lab demos is always control_dt).
            # downsample_ratio then divides this base frequency.
            expert_freq = 1.0 / control_dt
            downsample_ratio = int(kwargs.get("downsample_ratio", 1))
            expert_dt = downsample_ratio / expert_freq

            # When fit_pad > 0, the polynomial was fitted on H_ext = H + 2*fit_pad
            # points with s ∈ [0,1]. The middle H points have different s-values.
            fit_pad = getattr(self.policy, 'fit_pad', 0)
            poly_prefix = getattr(self.policy, 'poly_prefix', None)

            poly_suffix = getattr(self.policy, 'poly_suffix', 0)
            if poly_prefix is not None:
                H_poly = poly_prefix + n_act + poly_suffix
                anchor_in_poly = poly_prefix
            else:
                H_poly = H
                anchor_in_poly = n_obs - 1

            H_poly_ext = H_poly + 2 * fit_pad
            T_expert = (H_poly_ext - 1) * expert_dt

            obs_subsample = max(1, round(expert_dt / control_dt))
            self._obs_subsample_rate = obs_subsample
            self.obs = deque(maxlen=n_obs * obs_subsample + 1)

            s_start = (fit_pad + anchor_in_poly) / (H_poly_ext - 1)
            s_end = (fit_pad + anchor_in_poly + n_act - 1) / (H_poly_ext - 1)
            n_dense = max(2, round(n_act * expert_dt / control_dt))

            if n_dense > n_act:
                s_dense = torch.linspace(float(s_start), float(s_end), n_dense,
                                         dtype=torch.float64)
                basis = self.policy.basis_type
                if basis == "legendre":
                    eval_mat, d1_mat, _ = type(self.policy)._build_legendre_basis(
                        s_dense, self.policy.poly_order)
                elif basis == "power":
                    eval_mat, d1_mat, _ = type(self.policy)._build_power_basis(
                        s_dense, self.policy.poly_order)
                else:
                    raise ValueError(f"Unknown basis_type: {basis}")

                device = torch.device(self.device)
                self._dense_eval_mat = eval_mat.float().to(device)
                self._dense_d1_mat = d1_mat.float().to(device)
                self._T_expert = T_expert
                self._continuous_resample = True
                self.policy_cfg.action_config.action_chunk_steps = n_dense
                self._vel_time_scale = 1.0

                log.info(f"[FLASH-G Resample] ENABLED  (n_dense={n_dense} > n_act={n_act})")
            else:
                self._vel_time_scale = 1.0 / T_expert if T_expert > 0 else 1.0
                log.info(f"[FLASH-G Resample] SKIPPED  (n_dense={n_dense} ≈ n_act={n_act})")

            log.info(f"[FLASH-G] expert_freq={expert_freq:.1f}Hz, ds={downsample_ratio}, "
                     f"expert_dt={expert_dt:.4f}s, control_dt={control_dt:.4f}s, "
                     f"T_expert={T_expert:.4f}s, obs_subsample={obs_subsample}")

        elif hasattr(self.policy, 'horizon'):
            H = self.policy.horizon
            decimation = self.scenario.decimation
            if self.scenario.sim_params.dt is not None:
                physics_dt = self.scenario.sim_params.dt
            else:
                physics_dt = 0.015 / decimation
            control_dt = decimation * physics_dt
            T_physical = (H - 1) * control_dt
            if T_physical > 0:
                self._vel_time_scale = 1.0 / T_physical
            log.info(f"[Non-FLASH-G Velocity] physics_dt={physics_dt:.6f}s, "
                     f"control_dt={control_dt:.4f}s, vel_scale={self._vel_time_scale:.2f}x")

    def _stack_last_n_obs(self, all_obs, n_steps):
        assert len(all_obs) > 0
        all_obs = list(all_obs)
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
            result = np.swapaxes(
                result, 0, 1
            )  # Policy expects (Batch_size, n_steps, ...)
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros(
                (n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype
            )
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
            result = result.transpose(0, 1)  # Policy expects (Batch_size, n_steps, ...)
        else:
            raise RuntimeError(f"Unsupported obs type {type(all_obs[0])}")
        return result

    def reset(self):
        self.obs.clear()
        super().reset()

    def update_obs(self, current_obs):
        self.obs.append(current_obs)

    def _get_n_steps_obs(self):
        assert len(self.obs) > 0, "no observation is recorded, please update obs first"

        all_obs = list(self.obs)
        R = self._obs_subsample_rate

        if R > 1 and len(all_obs) >= R:
            subsampled = all_obs[::-R][::-1]
        else:
            subsampled = all_obs

        result = dict()
        for key in subsampled[0].keys():
            result[key] = self._stack_last_n_obs(
                [obs[key] for obs in subsampled], self.yaml_cfg.n_obs_steps
            )

        return result

    def predict_action(self, observaton=None):
        if observaton is not None:
            self.obs.append(observaton)  # update
        obs = self._get_n_steps_obs()

        with torch.no_grad():
            result = self.policy.predict_action(obs)

            if self._continuous_resample and "coefficients" in result:
                action_chunk, vel_chunk = self._resample_polynomial(
                    result["coefficients"])
                if "delta_q" in result:
                    n_dense = action_chunk.shape[0]
                    dev = action_chunk.device
                    w = torch.linspace(1.0, 0.0, n_dense, device=dev).view(-1, 1, 1)
                    dq = result["delta_q"].unsqueeze(0)   # (1, B, Da)
                    action_chunk = action_chunk + dq * w
                    T_chunk = max((n_dense - 1) * self._control_dt, 1e-6)
                    vel_chunk = vel_chunk + dq * (-1.0 / T_chunk)
                self._pending_velocity_chunk = vel_chunk
            else:
                action_chunk = result["action"].detach().to(torch.float32)
                action_chunk = action_chunk.transpose(0, 1)

                if "velocity_action" in result:
                    vel_chunk = result["velocity_action"].detach().to(torch.float32)
                    vel_chunk = vel_chunk * self._vel_time_scale
                    self._pending_velocity_chunk = vel_chunk.transpose(0, 1)
                else:
                    self._pending_velocity_chunk = None

        return action_chunk

    @staticmethod
    def _recover_cfg_from_weights(cfg, payload, initial_normalizer, log):
        """Fix a checkpoint whose saved config doesn't match its weights.

        This happens when a checkpoint was saved during resumed training before
        the self.cfg fix (the resumed run wrote the current YAML defaults
        instead of the original training config).  We infer the correct
        architecture (horizon, poly_order, n_obs_steps, n_action_steps) from
        the state-dict tensor shapes and patch ``cfg`` in-place.
        """
        from omegaconf import open_dict

        model_sd = payload["state_dicts"]["model"]
        H = int(model_sd["_time_points"].shape[0])
        C = int(model_sd["_eval_matrix"].shape[1])
        poly_order = C - 1

        # obs_feature_dim is independent of H / poly_order, so we can
        # instantiate the obs_encoder from the (possibly wrong) config.
        obs_encoder = hydra.utils.instantiate(cfg.policy_config.obs_encoder)
        obs_feature_dim = obs_encoder.output_shape()[0]
        cond_in = int(model_sd["model.cond_embed.weight"].shape[1])
        n_obs_steps = cond_in // obs_feature_dim
        n_action_steps = H - n_obs_steps

        log.info(f"[Eval] Inferred from weights: horizon={H}, poly_order={poly_order}, "
                 f"n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps} "
                 f"(obs_feature_dim={obs_feature_dim})")

        with open_dict(cfg):
            cfg.horizon = H
            cfg.n_obs_steps = n_obs_steps
            cfg.n_action_steps = n_action_steps
            cfg.policy_config.horizon = H
            cfg.policy_config.poly_order = poly_order
            cfg.policy_config.n_obs_steps = n_obs_steps
            cfg.policy_config.n_action_steps = n_action_steps

        return cfg

    @staticmethod
    def _read_expert_freq(cfg, fallback: float = 66.67):
        """Read raw_data_freq_hz from the zarr metadata saved in the checkpoint config.

        Falls back to *fallback* Hz (default: simulator control frequency ≈ 66.67 Hz)
        if the zarr is unavailable or doesn't contain the field.
        """
        import os
        try:
            zarr_path = cfg.dataset_config.zarr_path
            if os.path.exists(zarr_path):
                import zarr
                z = zarr.open(zarr_path, "r")
                freq = z["meta"].attrs.get("raw_data_freq_hz", None)
                if freq is not None and freq > 0:
                    return float(freq)
        except Exception:
            pass
        return fallback

    def _resample_polynomial(self, ncoeff):
        """Resample FLASH-G's polynomial at the simulator's native control frequency.

        Instead of using the default H-point evaluation, evaluate the polynomial
        on a denser s-grid whose spacing matches control_dt. This lets the
        trajectory play at its true physical pace regardless of the training
        data frequency.

        Args:
            ncoeff: (B, C, Da) normalized polynomial coefficients from FLASH-G.

        Returns:
            action_chunk: (n_dense, B, Da) position targets at control_dt spacing.
            vel_chunk:    (n_dense, B, Da) velocity targets in rad/s.
        """
        raw_coeff = ncoeff * self.policy._coeff_scale  # (B, C, Da)

        traj_norm = torch.einsum(
            "hc,bcd->bhd", self._dense_eval_mat, raw_coeff)  # (B, n_dense, Da)

        action_dense = self.policy.normalizer["action"].unnormalize(traj_norm)

        vel_norm = torch.einsum(
            "hc,bcd->bhd", self._dense_d1_mat, raw_coeff) / self._T_expert
        scale = self.policy.normalizer["action"].params_dict["scale"].view(1, 1, -1)
        vel_dense = vel_norm / scale  # (B, n_dense, Da) in rad/s

        action_chunk = action_dense.detach().to(torch.float32).transpose(0, 1)
        vel_chunk = vel_dense.detach().to(torch.float32).transpose(0, 1)
        return action_chunk, vel_chunk
