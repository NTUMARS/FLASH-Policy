from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import imageio
import numpy as np
import rootutils
import torch
from loguru import logger as log
from PIL import Image
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from metasim.utils.kinematics import get_curobo_models
from metasim.task.registry import get_task_class

# Try to import randomization components
try:
    from metasim.randomization import DomainRandomizationManager, DRConfig
    RANDOMIZATION_AVAILABLE = True
except ImportError as e:
    log.warning(f"Domain randomization not available: {e}")
    RANDOMIZATION_AVAILABLE = False



def images_to_video(images, video_path, frame_size=(1920, 1080), fps=30):
    if not images:
        print("No images found in the specified directory!")
        return

    writer = imageio.get_writer(video_path, fps=fps)

    for image in images:
        if image.shape[1] > frame_size[0] or image.shape[0] > frame_size[1]:
            print("Warning: frame size is smaller than the one of the images.")
            print("Images will be resized to match frame size.")
            image = np.array(Image.fromarray(image).resize(frame_size))

        writer.append_data(image)

    writer.close()
    print("Video created successfully!")


def ensure_clean_state(handler, expected_state=None):
    """Ensure environment is in clean initial state with intelligent validation."""
    prev_state = None
    stable_count = 0
    max_steps = 10
    min_steps = 2

    for step in range(max_steps):
        handler.simulate()
        current_state = handler.get_states()

        if step >= min_steps:
            if prev_state is not None:
                is_stable = True
                if hasattr(current_state, "objects") and hasattr(prev_state, "objects"):
                    for obj_name, obj_state in current_state.objects.items():
                        if obj_name in prev_state.objects:
                            curr_dof = getattr(obj_state, "dof_pos", None)
                            prev_dof = getattr(prev_state.objects[obj_name], "dof_pos", None)
                            if curr_dof is not None and prev_dof is not None:
                                if not torch.allclose(curr_dof, prev_dof, atol=1e-5):
                                    is_stable = False
                                    break

                if is_stable and expected_state is not None:
                    is_correct_state = _validate_state_correctness(current_state, expected_state)
                    if not is_correct_state:
                        log.debug(f"State stable but incorrect at step {step}, continuing simulation...")
                        stable_count = 0
                        is_stable = False

                if is_stable:
                    stable_count += 1
                    if stable_count >= 2:
                        break
                else:
                    stable_count = 0

            prev_state = current_state

    if expected_state is not None:
        final_state = handler.get_states()
        is_final_correct = _validate_state_correctness(final_state, expected_state)
        if not is_final_correct:
            log.warning(f"State validation failed after {max_steps} steps - reset may not have taken full effect")

    handler.get_states()


def _validate_state_correctness(current_state, expected_state):
    """Validate that current state matches expected initial state for critical objects."""
    if not hasattr(current_state, "objects") or not hasattr(expected_state, "objects"):
        return True

    critical_objects = []
    for obj_name, expected_obj in expected_state.objects.items():
        if hasattr(expected_obj, "dof_pos") and getattr(expected_obj, "dof_pos", None) is not None:
            critical_objects.append(obj_name)

    if not critical_objects:
        return True

    tolerance = 5e-3

    for obj_name in critical_objects:
        if obj_name not in current_state.objects:
            continue

        expected_obj = expected_state.objects[obj_name]
        current_obj = current_state.objects[obj_name]

        expected_dof = getattr(expected_obj, "dof_pos", None)
        current_dof = getattr(current_obj, "dof_pos", None)

        if expected_dof is not None and current_dof is not None:
            if not torch.allclose(current_dof, expected_dof, atol=tolerance):
                diff = torch.abs(current_dof - expected_dof).max().item()
                log.debug(f"DOF mismatch for {obj_name}: max diff = {diff:.6f} (tolerance = {tolerance})")
                return False

    return True


def str2bool(v):
    """argparse bool cannot use type=bool (bool('False') is True). See default_runner / common argparse usage."""
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in ("yes", "true", "t", "1", "y"):
        return True
    if s in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--robot", type=str, default="franka")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument(
        "--sim",
        type=str,
        default="isaacsim",
        choices=["isaacsim", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"],
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="openvla",
        choices=["act"],
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="openvla/openvla-7b",
    )
    parser.add_argument(
        "--temporal_agg",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="True/False (default False; consistent with explicit True in training/scripts)",
    )

    parser.add_argument(
        "--headless",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="True/False (default True, headless simulation)",
    )
    parser.add_argument(
        "--num_eval",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=400,
    )

    # Domain Randomization options
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Randomization level: 0=None, 1=Scene+Material, 2=+Light, 3=+Camera"
    )
    parser.add_argument(
        "--scene_mode",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Scene mode: 0=Manual, 1=USD Table, 2=USD Scene, 3=Full USD"
    )
    parser.add_argument(
        "--randomization_seed",
        type=int,
        default=None,
        help="Seed for reproducible randomization. If None, uses random seed"
    )
    parser.add_argument(
        "--downsample_ratio",
        type=int,
        default=1,
        help="Training data downsample ratio (for eval dir naming; align with il_run.sh / default_runner)",
    )
    parser.add_argument(
        "--eval_ckpt_name",
        type=str,
        default="policy_last",
        help="Checkpoint filename without .ckpt suffix (e.g. 'step_10000', 'policy_last', 'policy_best')",
    )

    # ACT model architecture parameters (must match training config)
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension of transformer")
    parser.add_argument("--dim_feedforward", type=int, default=3200, help="Feedforward dimension of transformer")
    parser.add_argument("--enc_layers", type=int, default=4, help="Number of encoder layers")
    parser.add_argument("--dec_layers", type=int, default=7, help="Number of decoder layers")
    parser.add_argument("--nheads", type=int, default=8, help="Number of attention heads")

    args = parser.parse_args()
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    args = parse_args()
    num_envs: int = args.num_envs

    import numpy as np
    import torch

    from metasim.scenario.cameras import PinholeCameraCfg
    from metasim.scenario.lights import DiskLightCfg, SphereLightCfg
    from metasim.utils.demo_util import get_traj
    from metasim.utils.setup_util import get_robot

    task_cls = get_task_class(args.task)

    if args.task in {"stack_cube", "pick_cube", "pick_butter"}:
        dp_camera = True
    else:
        dp_camera = args.task != "close_box"

    is_libero_dataset = "libero_90" in args.task

    if is_libero_dataset:
        dp_pos = (2.0, 0.0, 2)
    elif dp_camera:
        dp_pos = (1.0, 0.0, 0.75)
    else:
        dp_pos = (1.5, 0.0, 1.5)

    camera = PinholeCameraCfg(
        name="camera",
        data_types=["rgb", "depth"],
        width=256,
        height=256,
        pos=dp_pos,
        look_at=(0.0, 0.0, 0.0),
    )

    # Lighting setup (same logic as collect_demo.py)
    # Determine intensity based on render mode (if available)
    render_mode = getattr(args, 'render_mode', 'raytracing')
    if render_mode == "pathtracing":
        ceiling_main = 18000.0
        ceiling_corners = 8000.0
    else:
        ceiling_main = 12000.0
        ceiling_corners = 5000.0

    lights = [
        DiskLightCfg(
            name="ceiling_main",
            intensity=ceiling_main,
            color=(1.0, 1.0, 1.0),
            radius=1.2,
            pos=(0.0, 0.0, 2.8),
            rot=(0.7071, 0.0, 0.0, 0.7071),
        ),
        SphereLightCfg(
            name="ceiling_ne", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(1.0, 1.0, 2.5)
        ),
        SphereLightCfg(
            name="ceiling_nw", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(-1.0, 1.0, 2.5)
        ),
        SphereLightCfg(
            name="ceiling_sw", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(-1.0, -1.0, 2.5)
        ),
        SphereLightCfg(
            name="ceiling_se", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(1.0, -1.0, 2.5)
        ),
    ]

    scenario = task_cls.scenario.update(
        robots=[args.robot],
        simulator=args.sim,
        num_envs=args.num_envs,
        headless=args.headless,
        lights=lights,
        cameras=[camera]
    )

    tic = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = task_cls(scenario, device=device)
    robot = get_robot(args.robot)
    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")

    ## Data
    tic = time.time()
    assert os.path.exists(env.traj_filepath), (
        f"Trajectory file: {env.traj_filepath} does not exist."
    )
    init_states, all_actions, all_states = get_traj(env.traj_filepath, robot, env.handler)
    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")

    # Initialize Domain Randomization Manager
    if not RANDOMIZATION_AVAILABLE:
        log.warning("Randomization components not available!")
        raise ImportError("Domain Randomization not available. Please check installation.")

    # Determine render mode from args
    render_mode = getattr(args, 'render_mode', 'raytracing')

    # Create render config for DR
    from dataclasses import dataclass
    @dataclass
    class SimpleRenderCfg:
        mode: str = render_mode

    randomization_manager = DomainRandomizationManager(
        config=DRConfig(
            level=args.level,
            scene_mode=args.scene_mode,
            randomization_seed=args.randomization_seed,
        ),
        scenario=scenario,
        handler=env.handler,
        init_states=init_states,
        render_cfg=SimpleRenderCfg(mode=render_mode)
    )

    if args.algo == "act":
        state_dim = 9
        franka_state_dim = 9
        lr_backbone = 1e-5
        backbone = "resnet18"
        # Use command line args for model architecture (must match training config)
        enc_layers = args.enc_layers
        dec_layers = args.dec_layers
        nheads = args.nheads
        camera_names = ["front"]
        kl_weight = 10
        # chunk_size = args.chunk_size
        hidden_dim = args.hidden_dim
        batch_size = 8
        dim_feedforward = args.dim_feedforward
        lr = 1e-5
        act_ckpt_name = f"{args.eval_ckpt_name}.ckpt"
        policy_config = {
            "lr": lr,
            "num_queries": args.chunk_size,
            "kl_weight": kl_weight,
            "hidden_dim": hidden_dim,
            "dim_feedforward": dim_feedforward,
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
            "state_dim": state_dim,
        }

        import pickle

        from roboverse_learn.il.policies.act.policy import ACTPolicy

        ckpt_path = os.path.join(args.ckpt_path, act_ckpt_name)
        policy = ACTPolicy(policy_config)
        loading_status = policy.load_state_dict(torch.load(ckpt_path))
        print(loading_status)
        policy.cuda()
        policy.eval()
        print(f"Loaded: {ckpt_path}")
        stats_path = os.path.join(args.ckpt_path, "dataset_stats.pkl")
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

        def pre_process(s_qpos):
           # return (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
            return (s_qpos - stats["state_mean"]) / stats["state_std"]


        def post_process(a):
            return a * stats["action_std"] + stats["action_mean"]

        query_frequency = policy_config["num_queries"]
        if args.temporal_agg:
            query_frequency = 1
            num_queries = policy_config["num_queries"]
        max_timesteps = env.max_episode_steps
        max_timesteps = int(max_timesteps * 1)

    import datetime, pathlib, re
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Extract short ckpt tag from filename: policy_last.ckpt→last, step_1250.ckpt→1250, policy_best.ckpt→best
    ckpt_tag = os.path.splitext(act_ckpt_name)[0]  # remove .ckpt
    ckpt_tag = re.sub(r'^policy_', '', ckpt_tag)    # policy_last → last
    ckpt_tag = re.sub(r'^step_', '', ckpt_tag)      # step_1250 → 1250
    ds_ratio = getattr(args, "downsample_ratio", 1)
    dr_level = args.level
    ds_dr = f"_ds{ds_ratio}_dr{dr_level}" if dr_level > 0 else f"_ds{ds_ratio}"
    # Align with default_runner: {tag}_ds{ds}[_dr{dr}].ckpt_{time}
    eval_dir_name = f"{ckpt_tag}{ds_dr}.ckpt_{time_str}"
    base_eval_dir = pathlib.Path(f"il_outputs/{args.algo}/{args.task}/eval/{args.task}/{args.algo}/{args.robot}/{eval_dir_name}")
    base_eval_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Eval output dir: {base_eval_dir}")

    ## cuRobo controller (commented out - not needed for ACT joint control)
    # *_, robot_ik = get_curobo_models(scenario.robots[0])
    # curobo_n_dof = len(robot_ik.robot_config.cspace.joint_names)
    # ee_n_dof = len(scenario.robots[0].gripper_open_q)

    ## Reset before first step
    TotalSuccess = 0
    num_eval: int = args.num_eval
    all_inference_times = []  # Collect inference times from all steps
    demo_avg_inference_times = []  # Collect average inference time for each demo
    success_episode_durations = []  # Episode wall-clock durations for successful episodes (ms)
    success_episode_total_infer_times = []  # Total inference time per successful episode (ms)

    for i in range(num_eval):
        demo_idx = i

        # Apply domain randomization before reset
        log.info(f"[ACT Eval] Episode {i}: Applying DR for demo_idx={demo_idx}")
        randomization_manager.apply_randomization(demo_idx=demo_idx, is_initial=(i == 0))
        randomization_manager.update_positions_to_table(demo_idx=demo_idx, env_id=0)
        randomization_manager.update_camera_look_at(env_id=0)
        randomization_manager.apply_camera_randomization()

        tic = time.time()
        obs, extras = env.reset(states=[init_states[demo_idx]])
        toc = time.time()
        log.trace(f"Time to reset: {toc - tic:.2f}s")

        # Ensure environment stabilizes after reset
        ensure_clean_state(env.handler, expected_state=init_states[demo_idx])

        # Reset episode step counter after stabilization
        if hasattr(env, "_episode_steps"):
            env._episode_steps[0] = 0

        log.debug(f"Env: {i}")

        step = 0
        MaxStep = 800
        SuccessOnce = [False] * num_envs
        SuccessEnd = [False] * num_envs
        TimeOut = [False] * num_envs
        image_list = []
        inference_times = []  # Track inference times for this demo

        # act specific
        if args.temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()

        episode_start_time = time.time()
        with torch.no_grad():
            while step < MaxStep:
                # log.debug(f"Step {step}")
                robot_joint_limits = scenario.robots[0].joint_limits

                image_list.append(np.array(obs.cameras['camera'].rgb.cpu())[0])

                qpos_numpy = np.array(obs.robots['franka'].joint_pos.cpu())
                # qpos_numpy = np.array(obs["joint_qpos"])
                qpos = pre_process(qpos_numpy)
                # qpos = np.concatenate([qpos, np.zeros((qpos.shape[0], 14 - qpos.shape[1]))], axis=1)
                qpos = torch.from_numpy(qpos).float().cuda()
                qpos_history[:, step] = qpos
                curr_image = np.array(obs.cameras['camera'].rgb.cpu()).transpose(0, 3, 1, 2)
                # cur_image = np.stack([curr_image, curr_image], axis=0)
                curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
                # breakpoint()
                # Compute targets

                if step % query_frequency == 0:
                    # Measure inference time
                    torch.cuda.synchronize()
                    infer_start = time.time()
                    all_actions = policy(qpos, curr_image)
                    torch.cuda.synchronize()
                    infer_end = time.time()
                    infer_time = (infer_end - infer_start) * 1000  # Convert to ms
                    inference_times.append(infer_time)
                    log.debug(f"Step {step} inference time: {infer_time:.2f} ms")
                if args.temporal_agg:
                    all_time_actions[[step], step : step + num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, step]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, step % query_frequency]

                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                action = action[:franka_state_dim]
                action = torch.tensor(action, dtype=torch.float32, device="cpu")

                # IK solver expects original joint order, but state uses alphabetical order
                reorder_idx = env.handler.get_joint_reindex(args.robot)
                inverse_reorder_idx = [reorder_idx.index(i) for i in range(len(reorder_idx))]
                actions = action[inverse_reorder_idx]
                inner_actions = {"dof_pos_target": dict(zip(scenario.robots[0].joint_limits.keys(), actions))}
                # Format: actions[env_id][robot_name][action_type]
                actions = [{"franka": inner_actions}]
                #log.debug(f"Actions: {actions}")
                # log.debug(f"Action: {actions}")
                obs, reward, success, time_out, extras = env.step(actions)
                env.handler.refresh_render()
                # print(reward, success, time_out)

                # eval
                # if success[0]:
                #     TotalSuccess += 1
                #     print(f"Env {i} Success")
                if success[0] and not SuccessOnce[0]:
                    TotalSuccess += 1
                    SuccessOnce[0] = True
                    print(f"Env {i} Success")
                    break

                SuccessOnce = [SuccessOnce[i] or success[i] for i in range(num_envs)]
                TimeOut = [TimeOut[i] or time_out[i] for i in range(num_envs)]
                for TimeOutIndex in range(num_envs):
                    if TimeOut[TimeOutIndex]:
                        SuccessEnd[TimeOutIndex] = False
                if all(TimeOut):
                    print("All time out")
                    break

                step += 1

            episode_end_time = time.time()
            episode_duration_ms = (episode_end_time - episode_start_time) * 1000
            episode_total_infer_time_ms = sum(inference_times)

            log.debug(f"TotalSuccess: {TotalSuccess} | Success Rate: {TotalSuccess / (i + 1):.4f}")
            images_to_video(image_list, str(base_eval_dir / f"{i}.mp4"))

            # Collect durations for successful episodes
            if SuccessOnce[0]:
                success_episode_durations.append(episode_duration_ms)
                success_episode_total_infer_times.append(episode_total_infer_time_ms)

            # Save inference time statistics for this demo
            if inference_times:
                avg_time = sum(inference_times) / len(inference_times)
                min_time = min(inference_times)
                max_time = max(inference_times)

                # Collect for overall statistics
                all_inference_times.extend(inference_times)
                demo_avg_inference_times.append(avg_time)

                timing_file = str(base_eval_dir / f"{i}_timing.txt")
                with open(timing_file, "w") as f:
                    f.write(f"Demo {i} Inference Time Statistics\n")
                    f.write(f"{'='*40}\n")
                    f.write(f"Total inference calls: {len(inference_times)}\n")
                    f.write(f"Average inference time: {avg_time:.2f} ms\n")
                    f.write(f"Min inference time: {min_time:.2f} ms\n")
                    f.write(f"Max inference time: {max_time:.2f} ms\n")
                    f.write(f"{'='*40}\n")
                    f.write(f"Success: {SuccessOnce[0]}\n")
                    f.write(f"\n--- Episode Duration ---\n")
                    f.write(f"Episode Duration: {episode_duration_ms:.2f}ms\n")
                    f.write(f"Episode Total Inference Time: {episode_total_infer_time_ms:.2f}ms\n")
                    f.write(f"\nPer-step inference times (ms):\n")
                    for idx, t in enumerate(inference_times):
                        f.write(f"  Step {idx * query_frequency}: {t:.2f} ms\n")

                log.info(f"Demo {i}: Avg={avg_time:.2f}ms, Min={min_time:.2f}ms, Max={max_time:.2f}ms")

    success_rate = TotalSuccess / num_eval
    print("Success Rate: ", success_rate)

    # Calculate overall inference time statistics
    overall_total_steps = len(all_inference_times)
    overall_avg_inference_time = sum(all_inference_times) / overall_total_steps if overall_total_steps > 0 else 0
    overall_min_inference_time = min(all_inference_times) if all_inference_times else 0
    overall_max_inference_time = max(all_inference_times) if all_inference_times else 0
    
    # Calculate STD of demo-level average inference times
    num_demos_evaluated = len(demo_avg_inference_times)
    if num_demos_evaluated > 1:
        demo_avg_mean = sum(demo_avg_inference_times) / num_demos_evaluated
        demo_avg_variance = sum((x - demo_avg_mean) ** 2 for x in demo_avg_inference_times) / (num_demos_evaluated - 1)
        demo_avg_std = demo_avg_variance ** 0.5
    else:
        demo_avg_std = 0.0
    
    # For ACT, inference is only measured at actual forward passes (step % query_frequency == 0),
    # so per-inference stats are identical to overall stats. Output them for format consistency.
    overall_total_inferences = overall_total_steps  # ACT only records actual inferences
    overall_avg_per_inference_time = overall_avg_inference_time
    overall_min_per_inference_time = overall_min_inference_time
    overall_max_per_inference_time = overall_max_inference_time

    if len(demo_avg_inference_times) > 1:
        pi_std = demo_avg_std  # same data for ACT
    else:
        pi_std = 0.0

    # Calculate average episode duration and total inference time for successful episodes only
    num_success_episodes = len(success_episode_durations)
    avg_success_episode_duration = sum(success_episode_durations) / num_success_episodes if num_success_episodes > 0 else 0
    avg_success_episode_total_infer = sum(success_episode_total_infer_times) / num_success_episodes if num_success_episodes > 0 else 0

    log.info(f"FINAL RESULTS: Average Success Rate = {success_rate:.4f}")
    log.info(f"FINAL RESULTS: Overall Avg Inference Time = {overall_avg_inference_time:.2f}ms (STD across demos: {demo_avg_std:.2f}ms), "
             f"Min: {overall_min_inference_time:.2f}ms, Max: {overall_max_inference_time:.2f}ms, "
             f"Total Steps: {overall_total_steps}")
    log.info(f"FINAL RESULTS: Overall Avg Per-Inference Time = {overall_avg_per_inference_time:.2f}ms (STD across demos: {pi_std:.2f}ms), "
             f"Min: {overall_min_per_inference_time:.2f}ms, Max: {overall_max_per_inference_time:.2f}ms, "
             f"Total Actual Inferences: {overall_total_inferences}")

    result_file = str(base_eval_dir / "00_final_stats.txt")
    with open(result_file, "w") as f:
        f.write(f"=== Success Statistics ===\n")
        f.write(f"Total Success: {TotalSuccess}\n")
        f.write(f"Total Evaluated: {num_eval}\n")
        f.write(f"Average Success Rate: {success_rate:.4f}\n")
        f.write(f"\n=== Domain Randomization ===\n")
        f.write(f"Domain Randomization Level: {args.level}\n")
        f.write(f"Domain Randomization Scene Mode: {args.scene_mode}\n")
        f.write(f"Domain Randomization Seed: {args.randomization_seed}\n")
        f.write(f"\n=== Overall Inference Time Statistics ===\n")
        f.write(f"Total Inference Steps: {overall_total_steps}\n")
        f.write(f"Number of Demos Evaluated: {num_demos_evaluated}\n")
        f.write(f"Average Inference Time: {overall_avg_inference_time:.2f}ms\n")
        f.write(f"STD of Demo Avg Inference Time: {demo_avg_std:.2f}ms\n")
        f.write(f"Min Inference Time: {overall_min_inference_time:.2f}ms\n")
        f.write(f"Max Inference Time: {overall_max_inference_time:.2f}ms\n")
        f.write(f"\n=== Overall Per-Inference Time Statistics (model forward pass only) ===\n")
        f.write(f"Total Actual Inferences: {overall_total_inferences}\n")
        f.write(f"Number of Demos Evaluated: {num_demos_evaluated}\n")
        f.write(f"Average Per-Inference Time: {overall_avg_per_inference_time:.2f}ms\n")
        f.write(f"STD of Demo Avg Per-Inference Time: {pi_std:.2f}ms\n")
        f.write(f"Min Per-Inference Time: {overall_min_per_inference_time:.2f}ms\n")
        f.write(f"Max Per-Inference Time: {overall_max_per_inference_time:.2f}ms\n")
        f.write(f"\n=== Successful Episodes Duration (successful episodes only) ===\n")
        f.write(f"Number of Successful Episodes: {num_success_episodes}\n")
        f.write(f"Avg Episode Duration (success only): {avg_success_episode_duration:.2f}ms\n")
        f.write(f"\n=== Successful Episodes Total Inference Time (successful episodes only) ===\n")
        f.write(f"Number of Successful Episodes: {num_success_episodes}\n")
        f.write(f"Avg Episode Total Inference Time (success only): {avg_success_episode_total_infer:.2f}ms\n")

    # Rename eval dir: embed success rate and avg inference time after first path segment
    # e.g. "1250_ds4.ckpt_2026-..." → "1250_sr86_3.80ms_ds4.ckpt_2026-..."
    try:
        infer_tag = f"{avg_success_episode_total_infer:.2f}ms"
        dir_name = base_eval_dir.name
        ckpt_prefix = dir_name.split(".ckpt_", 1)
        if len(ckpt_prefix) == 2 and "_sr" not in ckpt_prefix[0]:
            sr_tag = "100" if success_rate >= 1.0 else f"{int(success_rate * 100):02d}"
            prefix = ckpt_prefix[0]
            parts = prefix.split("_", 1)
            if len(parts) == 2:
                prefix = f"{parts[0]}_sr{sr_tag}_{infer_tag}_{parts[1]}"
            else:
                prefix = f"{parts[0]}_sr{sr_tag}_{infer_tag}"
            new_name = f"{prefix}.ckpt_{ckpt_prefix[1]}"
            new_dir = base_eval_dir.parent / new_name
            base_eval_dir.rename(new_dir)
            log.info(f"Eval dir renamed: {dir_name} -> {new_name}")
    except Exception as e:
        log.warning(f"Failed to rename eval dir: {e}")

    # Isaac Sim often blocks on env.close(); same as default_runner.evaluate: os._exit releases GPU after timeout
    # Isaac Sim worker orphans are cleaned up by batch_run.sh's setsid + PGID kill.
    def _force_exit():
        log.info("Force-exiting to release GPU memory (env.close timed out).")
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    _exit_timer = threading.Timer(30.0, _force_exit)
    _exit_timer.daemon = True
    _exit_timer.start()
    try:
        env.close()
    finally:
        _exit_timer.cancel()
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
