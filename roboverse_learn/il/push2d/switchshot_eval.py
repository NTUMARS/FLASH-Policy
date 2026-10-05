"""Closed-loop ForkReach evaluation (historical SwitchShot N/lv4).

Reports success rate, RMS second-order action differences, successful goal
counts, and commitment counts over all episodes (including timeouts). Mode
balance is min(top,bottom)/max(top,bottom) for the latter counts.
"""

import argparse
import json
import re
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

from roboverse_learn.il.push2d import push2d_viz as viz
from roboverse_learn.il.push2d.push2d_eval import load_policy
from roboverse_learn.il.push2d.switchshot_env import SwitchShotEnv, draw_switch_scene
from roboverse_learn.il.push2d.switchshot_geometry import CENTER_Y, GOAL_DY
from roboverse_learn.il.utils.eval_config_report import build_eval_config_lines


def commit_side(subject_y, band=GOAL_DY / 2.0):
    """Classify final agent position for all episodes, including failures.

    A top/bottom commitment crosses half the center-to-goal distance. A
    middle outcome has not crossed either threshold and is reported separately.
    """
    if subject_y < CENTER_Y - band:
        return "top"
    if subject_y > CENTER_Y + band:
        return "bottom"
    return "middle"


def validate_task_metadata(zarr_path):
    """Accept historical ForkReach sidecars and reject another task's data."""
    path = Path(zarr_path)
    sidecar = path.with_name(path.stem + "_episodes_meta.json")
    if not sidecar.exists():
        return
    params = json.loads(sidecar.read_text()).get("params", {})
    if params.get("variant", "N") != "N" or set(params.get("levels", [4])) != {4}:
        raise ValueError("ForkReach evaluation requires variant N / level 4 data")
    if int(params.get("downsample_ratio", 1)) != 1:
        raise ValueError("ForkReach retains the original 5 Hz control protocol")


@torch.no_grad()
def run(args):
    if args.variant != "N" or args.levels != [4]:
        raise ValueError("Only ForkReach (SwitchShot variant N, level 4) is retained")
    if args.eps < 1:
        raise ValueError("--eps must be positive")
    validate_task_metadata(args.zarr)
    device = args.device
    policy, cfg = load_policy(args.checkpoint, args.zarr, device)
    if args.num_inference_steps is not None:
        policy.num_inference_steps = args.num_inference_steps
    n_obs = policy.n_obs_steps
    k = args.execute_steps
    if not 1 <= k <= policy.n_action_steps:
        raise ValueError("--execute-steps must be in [1, n_action_steps]")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res, mode_hits, commit_hits = Counter(), Counter(), Counter()
    jerks = []
    for ep in range(args.eps):
        env = SwitchShotEnv(level=4, variant="N", seed=args.seed + ep)
        obs = env.reset()
        frames, pos = deque(maxlen=n_obs), deque(maxlen=n_obs)
        for _ in range(n_obs):
            frames.append(np.moveaxis(obs["image"], -1, 0).astype(np.float32) / 255.0)
            pos.append(obs["agent_pos"])
        done, info, acts = False, {}, []
        viz_on = not args.no_video
        video = [] if viz_on else None
        agent_xy = [tuple(env.agent.position)]
        chunks, replan_ts, colors_seq = [], [], []
        while not done:
            od = {"head_cam": torch.from_numpy(np.stack(frames))[None].to(device),
                  "agent_pos": torch.from_numpy(np.stack(pos))[None].to(device)}
            action = policy.predict_action(od)["action"][0].cpu().numpy()
            chunks.append(action.copy())
            replan_ts.append(env.t)
            for a in action[:k]:
                if done:
                    break
                obs, done, info = env.step(a)
                acts.append(a)
                agent_xy.append(tuple(env.agent.position))
                c = env.visible_colors()
                colors_seq.append(f"{c['top']}|{c['bottom']}")
                frames.append(np.moveaxis(obs["image"], -1, 0).astype(np.float32) / 255.0)
                pos.append(obs["agent_pos"])
                if viz_on:
                    import cv2
                    f = cv2.resize(env.render(None), (viz.VIDEO_SIZE, viz.VIDEO_SIZE),
                                   interpolation=cv2.INTER_AREA)
                    video.append(viz.annotate(f, f"eval L4 ep{ep} t={env.t} k={k}"))
        if not args.no_log:
            logd = out / "rollout_logs"
            logd.mkdir(parents=True, exist_ok=True)
            sc = {kk: (list(vv) if isinstance(vv, tuple) else vv)
                  for kk, vv in info["schedule"].items()}
            np.savez_compressed(
                logd / f"lv4_ep{ep:03d}.npz",
                # Keep the historical puck_pos key for existing log consumers;
                # in variant N the terminal subject has always been the agent.
                puck_pos=np.asarray(agent_xy, dtype=np.float32), subject="agent",
                agent_pos=np.asarray(agent_xy, dtype=np.float32),
                actions=np.asarray(acts, dtype=np.float32),
                pred_chunks=np.asarray(chunks, dtype=np.float32),
                replan_t=np.asarray(replan_ts, dtype=np.int32),
                colors=np.asarray(colors_seq), schedule=json.dumps(sc),
                terminal=str(info["fail_reason"] or "success"),
                seed=args.seed + ep, level=4,
                force_initial="None", force_switch="None",
                execute_steps=k, downsample_ratio=1)
        res[info["fail_reason"] or "success"] += 1
        a = np.asarray(acts)
        if len(a) >= 3:
            jerks.append(float(np.sqrt(np.mean((a[2:] - 2 * a[1:-1] + a[:-2]) ** 2))))
        py = env.agent.position.y
        if info.get("success"):
            mode_hits["top" if py < CENTER_Y else "bottom"] += 1
        commit_hits[commit_side(py)] += 1
        if viz_on:
            colors = env.visible_colors()
            viz.save_episode_visuals(
                out / "rollouts", f"lv4_ep{ep:03d}", info.get("success"),
                video, agent_xy, agent_xy, env.geom, f"L4 ep{ep} k={k}",
                draw_scene_fn=lambda g, _c=colors: draw_switch_scene(g, _c),
                crop_band=None)
    print(f"[switchshot_eval] L4: {dict(res)}")

    eval_params = {"send_vel_target": "n/a(2D env)",
                   "downsample_ratio": "1 (original ForkReach protocol)",
                   "max_step": "120"}
    lines = build_eval_config_lines(policy, eval_params, dr_level="n/a(2D env)")
    lines += ["  --- ForkReach Task Parameters ---",
              "  variant                   = N",
              "  levels                    = [4]",
              f"  eps_per_level             = {args.eps}",
              f"  eval_seed                 = {args.seed} (eval-only; same across arms)",
              f"  execute_steps_k           = {k}  (replanning interval; chunk={policy.n_action_steps})",
              f"  num_inference_steps       = {policy.num_inference_steps}",
              "  control_period            = 0.2 s/action",
              "  rules: success = agent inside either always-green goal; timeout = fail",
              "  mode_balance = min(top,bottom)/max(top,bottom), all committed episodes"]
    (out / "00_eval_config.txt").write_text("\n".join(lines) + "\n")
    sr = res["success"] / args.eps
    jerk = float(np.mean(jerks)) if jerks else None
    top, bot, mid = commit_hits["top"], commit_hits["bottom"], commit_hits["middle"]
    top_fraction = top / (top + bot) if top + bot else None
    balance = min(top, bot) / max(top, bot) if max(top, bot) else 0.0
    stats = {"success_rate": sr, "timeout_rate": res["timeout"] / args.eps,
             "action_jerk_rms_px": jerk}
    body = [f"ForkReach eval (variant N, level 4, k={k})",
            "level 4: " + "  ".join(f"{kk}={vv:.3f}" if isinstance(vv, float)
                                      else f"{kk}={vv}" for kk, vv in stats.items()),
            f"L4 mode_split (successes only): {dict(mode_hits)}",
            f"L4 commit_split (all {args.eps} eps): top={top} bottom={bot} middle={mid}",
            f"L4 top_fraction = {top_fraction}",
            f"L4 mode_balance = {balance:.3f} (min/max of committed top,bottom; 1=balanced)",
            f"mean_success = {sr:.3f}", f"mean_jerk_px = {jerk}"]
    (out / "00_final_stats.txt").write_text("\n".join(body) + "\n")
    (out / "00_final_stats.json").write_text(json.dumps(
        {"checkpoint": args.checkpoint, "variant": "N", "eps": args.eps,
         "seed": args.seed, "execute_steps": k, "downsample_ratio": 1,
         "levels": {"4": stats}, "L4_mode_split": dict(mode_hits),
         "L4_commit_split": dict(commit_hits), "L4_top_fraction": top_fraction,
         "L4_mode_balance": balance, "mean_success": sr}, indent=1))
    out = rename_with_metrics(out, sr, nis=policy.num_inference_steps)
    print(f"[switchshot_eval] results -> {out}")


def rename_with_metrics(out: Path, sr, nis=None) -> Path:
    """Idempotent original N/lv4 directory format: checkpoint_srXX_stepN_time."""
    if re.search(r"_sr\d+_step(?:\d+|na)_", out.name):
        return out
    m = re.match(r"^(.+\.ckpt_)(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$", out.name)
    if not m:
        return out
    step = "na" if nis is None else str(int(nis))
    new = out.with_name(f"{m.group(1)}sr{round(sr*100)}_step{step}_{m.group(2)}")
    try:
        out.rename(new)
        return new
    except OSError as e:
        print(f"[switchshot_eval] WARNING: rename failed ({e})")
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=["N"], default="N")
    ap.add_argument("--levels", type=int, nargs="+", choices=[4], default=[4])
    ap.add_argument("--eps", type=int, default=48)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--execute-steps", type=int, default=8)
    ap.add_argument("--num-inference-steps", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-log", action="store_true",
                    help="disable per-episode npz rollout logs (enabled by default)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
