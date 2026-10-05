"""CorridorPush closed-loop evaluation (README.md).

Standalone lightweight loader (does NOT reuse DefaultEvalRunner._init_policy —
that path is metasim-coupled): payload cfg -> hydra instantiate ->
set_normalizer(dataset) -> load model/ema state_dict.

Outputs per run dir: 00_eval_config.txt, 00_final_stats.txt, optional mp4s.

  python -m roboverse_learn.il.push2d.push2d_eval \
      --checkpoint .../10000.ckpt --zarr .../cp.zarr --out .../eval_run \
      --levels 2 --eps 50
"""

import argparse
from collections import deque
from pathlib import Path

import dill
import hydra
import numpy as np
import torch

from roboverse_learn.il.push2d import push2d_viz as viz
from roboverse_learn.il.push2d.corridor_geometry import WORLD
from roboverse_learn.il.push2d.corridor_push_env import CorridorPushEnv
from roboverse_learn.il.utils.eval_config_report import build_eval_config_lines


def load_dataset_normalizer(zarr_path: str):
    """Fit the training normalizer without allocating CUDA-pinned batches."""
    from roboverse_learn.il.utils.replay_buffer import ReplayBuffer
    from roboverse_learn.il.utils.normalizer import LinearNormalizer
    from roboverse_learn.il.utils.normalize_util import get_image_range_normalizer

    replay = ReplayBuffer.copy_from_path(zarr_path, keys=["state", "action"])
    normalizer = LinearNormalizer()
    normalizer.fit(data={"action": replay["action"], "agent_pos": replay["state"]},
                   last_n_dims=1, mode="limits")
    for key in ("head_cam", "front_cam", "left_cam", "right_cam"):
        normalizer[key] = get_image_range_normalizer()
    return normalizer


def load_policy(ckpt_path: str, zarr_path: str, device: str):
    with open(ckpt_path, "rb") as checkpoint:
        payload = torch.load(checkpoint, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy_config)
    normalizer = load_dataset_normalizer(zarr_path)
    policy.set_normalizer(normalizer)
    sd = payload["state_dicts"]
    key = "ema_model" if cfg.train_config.training_params.use_ema and "ema_model" in sd else "model"
    missing, unexpected = policy.load_state_dict(sd[key], strict=False)
    print(f"[push2d_eval] loaded '{key}' (missing={len(missing)}, unexpected={len(unexpected)})")
    # Normal checkpoints carry their training normalizer; reconstruct only when absent.
    if "head_cam" not in policy.normalizer.params_dict:
        print("[push2d_eval] WARNING: ckpt carried no normalizer — rebuilt from dataset")
        policy.set_normalizer(normalizer)
    policy.to(device).eval()
    return policy, cfg


@torch.no_grad()
def run(args):
    if args.levels != [2] or args.eps < 1:
        raise ValueError("Published CorridorPush evaluation requires level 2 and positive episodes")
    if getattr(args, "torch_seed", None) is not None:
        torch.manual_seed(args.torch_seed)
        np.random.seed(args.torch_seed % 2**32)
    device = args.device
    policy, cfg = load_policy(args.checkpoint, args.zarr, device)
    if getattr(args, "num_inference_steps", None) is not None:
        # Apply the requested inference setting for this evaluation.
        policy.num_inference_steps = args.num_inference_steps
        print(f"[push2d_eval] num_inference_steps overridden to {args.num_inference_steps}")
    n_obs = policy.n_obs_steps
    n_act = policy.n_action_steps
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stats, jerk_by_level, max_steps_by_level = {}, {}, {}
    env_meta = {}
    for lv in args.levels:
        succ, jerks = 0, []
        for ep in range(args.eps):
            env = CorridorPushEnv(level=lv, seed=args.seed + ep, max_steps=getattr(args, "max_steps", None))
            max_steps_by_level[lv] = env.max_steps
            if not env_meta:  # control-side constants are level-independent
                env_meta = {"control_hz": env.control_hz, "sim_hz": int(round(1 / env.dt)),
                            "n_substeps": env.n_substeps, "kp": env.kp, "v_max": env.v_max,
                            "damping": env.space.damping, "success_x": env.geom.success_x,
                            "world": int(WORLD), "render_size": env.render_size}
            obs = env.reset()
            frames = deque(maxlen=n_obs)
            pos = deque(maxlen=n_obs)
            for _ in range(n_obs):  # bootstrap history with the initial frame
                frames.append(np.moveaxis(obs["image"], -1, 0).astype(np.float32) / 255.0)
                pos.append(obs["agent_pos"])
            done, info, acts = False, {"success": False}, []
            viz_on = not args.no_video
            video = [] if viz_on else None
            puck_xy = [tuple(env.puck.position)] if viz_on else None
            agent_xy = [tuple(env.agent.position)] if viz_on else None
            while not done:
                od = {"head_cam": torch.from_numpy(np.stack(frames))[None].to(device),
                      "agent_pos": torch.from_numpy(np.stack(pos))[None].to(device)}
                action = policy.predict_action(od)["action"][0].cpu().numpy()
                for a in action[:n_act]:
                    if done:
                        break
                    obs, done, info = env.step(a)
                    acts.append(a)
                    frames.append(np.moveaxis(obs["image"], -1, 0).astype(np.float32) / 255.0)
                    pos.append(obs["agent_pos"])
                    if viz_on:
                        import cv2
                        f = cv2.resize(env.render(None), (viz.VIDEO_SIZE, viz.VIDEO_SIZE),
                                       interpolation=cv2.INTER_AREA)
                        video.append(viz.annotate(f, f"eval  level {lv}  ep {ep}  t={env.t}"))
                        puck_xy.append(tuple(env.puck.position))
                        agent_xy.append(tuple(env.agent.position))
            succ += int(info["success"])
            a = np.asarray(acts)
            if len(a) >= 3:
                jerks.append(float(np.sqrt(np.mean((a[2:] - 2 * a[1:-1] + a[:-2]) ** 2))))
            if viz_on:
                # Each rollout produces a video and a trajectory overview;
                # the outcome is encoded in the filename.
                viz.save_episode_visuals(out / "rollouts", f"lv{lv}_ep{ep:03d}",
                                         info["success"], video, puck_xy, agent_xy,
                                         env.geom, f"level {lv}  ep {ep}")
        stats[lv] = succ / args.eps
        jerk_by_level[lv] = float(np.mean(jerks)) if jerks else float("nan")
        print(f"[push2d_eval] level {lv}: {succ}/{args.eps} = {stats[lv]:.1%}  "
              f"jerk={jerk_by_level[lv]:.2f}px")

    # ---- Model architecture + FLASH/gate blocks come from build_eval_config_lines
    #      (populated from the live policy — these are the meaningful "which ckpt am
    #      I evaluating" fields).  The metasim joint-velocity-control knobs genuinely
    #      don't apply to a 2D position-PD env, so they're marked n/a(2D); the REAL
    #      task thresholds go in the dedicated Push2D block below.
    eval_params = {k: "n/a(2D env)" for k in
                   ("send_vel_target", "velocity_pd_gains", "velocity_kp_scale",
                    "velocity_kd_scale", "downsample_ratio")}
    # max_step varies per level (failure threshold); summarize the range here and
    # give the full per-level table in the Push2D block.
    _ms = [max_steps_by_level[lv] for lv in sorted(max_steps_by_level)]
    eval_params["max_step"] = (f"per-level {min(_ms)}..{max(_ms)} (default 60+16*level); "
                               f"see Push2D block")
    lines = build_eval_config_lines(policy, eval_params, dr_level="n/a(2D env)")

    lines.append("  --- Push2D Task Parameters ---")
    lines.append(f"  push2d_levels             = {sorted(stats)}")
    lines.append(f"  push2d_eps_per_level      = {args.eps}")
    lines.append(f"  torch_seed                = {getattr(args, 'torch_seed', None)}")
    lines.append(f"  push2d_eval_seed          = {args.seed}  (eval-only: seeds "
                 f"per-episode init jitter; NOT the training seed; same across all arms)")
    lines.append(f"  success_criterion         = puck.x >= {env_meta['success_x']:.0f} "
                 f"(world width {env_meta['world']})")
    lines.append("  max_steps_per_level (failure threshold; reaching it counts as failure):")
    for lv in sorted(max_steps_by_level):
        lines.append(f"      level {lv}              = {max_steps_by_level[lv]} action-steps")
    lines.append(f"  control_hz                = {env_meta['control_hz']} "
                 f"({env_meta['n_substeps']} physics substeps per action)")
    lines.append(f"  sim_hz                    = {env_meta['sim_hz']}")
    lines.append(f"  pusher_pd_kp              = {env_meta['kp']}")
    lines.append(f"  pusher_pd_v_max           = {env_meta['v_max']}")
    lines.append(f"  space_damping             = {env_meta['damping']}")
    lines.append(f"  obs_render_size           = {env_meta['render_size']}")
    (out / "00_eval_config.txt").write_text("\n".join(lines) + "\n")

    total = float(np.mean(list(stats.values())))
    mean_jerk = float(np.nanmean(list(jerk_by_level.values())))
    body = ["CorridorPush eval",
            "(action_jerk_rms_px = RMS of the second action difference; "
            "lower is smoother; compare methods within the same level)"] + [
        f"level {lv}: success_rate = {stats[lv]:.3f}   "
        f"action_jerk_rms_px = {jerk_by_level[lv]:.2f}"
        for lv in sorted(stats)
    ] + [f"mean_success = {total:.3f}", f"mean_action_jerk_rms_px = {mean_jerk:.2f}"]
    (out / "00_final_stats.txt").write_text("\n".join(body) + "\n")
    import json
    (out / "00_final_stats.json").write_text(json.dumps({
        "checkpoint": args.checkpoint, "eps_per_level": args.eps, "seed": args.seed,
        "levels": {str(lv): {"success_rate": stats[lv],
                             "action_jerk_rms_px": jerk_by_level[lv]}
                   for lv in sorted(stats)},
        "mean_success": total, "mean_action_jerk_rms_px": mean_jerk,
    }, indent=1))

    # ---- Add result metrics to the directory name after evaluation ----
    # 5010.ckpt_2026-07-25_12-16-03 -> 5010.ckpt_sr33_step1_jk12.3_2026-07-25_12-16-03
    # sr=mean success rate (%) / step=num_inference_steps from the loaded policy
    # (the effective value for this evaluation) / jk=mean action jerk in pixels.
    # Apply only to il_run's "<name>.ckpt_<timestamp>" directories. Preserve
    # custom --out names and keep the operation idempotent.
    out = rename_with_metrics(out, total, mean_jerk,
                              getattr(policy, "num_inference_steps", None))
    print(f"[push2d_eval] results -> {out}")


def rename_with_metrics(out: Path, mean_success: float, mean_jerk: float,
                        n_infer_steps=None) -> Path:
    import re
    name = out.name
    if re.search(r"_sr\d+_(step\S+_)?jk", name):   # already tagged (idempotent)
        return out
    m = re.match(r"^(.+\.ckpt_)(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$", name)
    if not m:
        return out                            # custom out dir: keep user's name
    jk = "nan" if np.isnan(mean_jerk) else f"{mean_jerk:.1f}"
    step = f"step{n_infer_steps}" if n_infer_steps is not None else "stepNA"
    new = out.with_name(
        f"{m.group(1)}sr{round(mean_success * 100)}_{step}_jk{jk}_{m.group(2)}")
    try:
        out.rename(new)
        return new
    except OSError as e:                      # e.g. name collision — keep original
        print(f"[push2d_eval] WARNING: rename failed ({e}); keeping {out}")
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--zarr", required=True, help="dataset zarr (normalizer source)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", type=int, nargs="+", choices=[2], default=[2])
    ap.add_argument("--eps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--torch-seed", type=int, default=None,
                    help="Optional sampler RNG seed; historical evaluations seeded only the environment")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--num-inference-steps", type=int, default=None,
                    help="Override policy.num_inference_steps at eval time "
                         "(inference-only; None = use checkpoint value)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--video", action="store_true",
                    help="deprecated no-op: per-rollout visuals are ON by default")
    ap.add_argument("--no-video", action="store_true",
                    help="disable per-rollout mp4 + trajectory PNG generation")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
