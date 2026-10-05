"""CorridorPush visualization: per-rollout mp4 videos + trajectory overview PNGs.

Two uses:
1) Library helpers shared with push2d_eval (write_video / trajectory_overlay /
   annotate) — eval auto-generates visuals for every rollout.
2) CLI for EXPERT demonstrations, either fresh rollouts or exact replays of the
   episodes inside a generated dataset (via its *_episodes_meta.json sidecar,
   which records every episode's seeds):

  # Replay the first two training episodes at each difficulty in high resolution.
  python -m roboverse_learn.il.push2d.push2d_viz \
      --sidecar /path/corridor_push_lv2_100_episodes_meta.json \
      --per-level 2 --out /path/viz_expert

  # Or run several expert episodes without a dataset.
  python -m roboverse_learn.il.push2d.push2d_viz --levels 2 --eps 2 --out /path/viz_expert
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from roboverse_learn.il.push2d.corridor_push_env import CorridorPushEnv, draw_static_scene
from roboverse_learn.il.push2d.scripted_expert import ScriptedExpert

VIDEO_SIZE = 384
FPS = 10  # 2x real time (control_hz=5)


def annotate(frame: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    cv2.putText(frame, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(frame, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


def write_video(frames, path, fps: int = FPS):
    """H.264 via imageio/ffmpeg — same as the repo's metasim/LIBERO eval videos
    (cv2's 'mp4v' codec is unplayable in many players/IDE viewers).
    Frames arrive BGR (cv2-native); imageio wants RGB."""
    import imageio
    imageio.mimsave(str(path), [f[:, :, ::-1] for f in frames], fps=fps, quality=8)


def trajectory_overlay(geom, puck_xy, agent_xy, success: bool | None = None,
                       title: str | None = None, *, draw_scene_fn=None,
                       crop_band="corridor") -> np.ndarray:
    """Single overview image: scene + full puck path (red) + pusher path (orange),
    both brightening over time; start markers hollow.  BGR uint8 (512x512).

    Keyword-only extension points (defaults == original CorridorPush behavior):
      draw_scene_fn: geom -> BGR canvas (default: corridor draw_static_scene)
      crop_band    : "corridor" = crop to corridor band (original), None = full
                     frame, or (y0, y1) custom crop.
    """
    c = (draw_scene_fn or draw_static_scene)(geom)

    def draw_path(pts, base):  # base: BGR
        pts = np.asarray(pts)
        n = len(pts)
        for i in range(1, n):
            f = 0.35 + 0.65 * i / max(n - 1, 1)          # time gradient
            col = tuple(int(b * f) for b in base)
            p0 = tuple(np.round(pts[i - 1]).astype(int))
            p1 = tuple(np.round(pts[i]).astype(int))
            cv2.line(c, p0, p1, col, 2, cv2.LINE_AA)
        cv2.circle(c, tuple(np.round(pts[0]).astype(int)), 5, base, 1, cv2.LINE_AA)
        cv2.circle(c, tuple(np.round(pts[-1]).astype(int)), 4, base, -1, cv2.LINE_AA)

    draw_path(agent_xy, (255, 160, 80))   # pusher: light blue (BGR)
    draw_path(puck_xy, (80, 80, 255))     # puck: red
    if crop_band == "corridor":
        # crop to the corridor band -> paper-friendly wide banner (original)
        from roboverse_learn.il.push2d.corridor_geometry import CENTER_Y, HALF_W, WALL_T
        y0 = max(0, int(CENTER_Y - HALF_W - WALL_T - 34))
        y1 = min(c.shape[0], int(CENTER_Y + HALF_W + WALL_T + 34))
        c = c[y0:y1].copy()
    elif crop_band is not None:
        y0, y1 = crop_band
        c = c[int(y0):int(y1)].copy()
    if title:
        annotate(c, title)
    if success is not None:
        tag, col = ("SUCCESS", (90, 220, 90)) if success else ("FAIL", (80, 80, 255))
        cv2.putText(c, tag, (390, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(c, tag, (390, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    return c


def record_expert_episode(level: int, env_seed: int, expert_seed: int,
                          noise_std: float = 1.0):
    """Roll one expert episode, recording hi-res frames + both trajectories."""
    env = CorridorPushEnv(level=level, seed=env_seed)
    expert = ScriptedExpert(env.geom, noise_std=noise_std, seed=expert_seed)
    obs = env.reset()
    frames, puck_xy, agent_xy = [], [], []
    done, info = False, {"success": False}
    while not done:
        puck_xy.append(tuple(env.puck.position))
        agent_xy.append(tuple(env.agent.position))
        f = cv2.resize(env.render(None), (VIDEO_SIZE, VIDEO_SIZE),
                       interpolation=cv2.INTER_AREA)
        frames.append(annotate(f, f"expert  level {level}  t={env.t}"))
        action = expert.act(env.puck.position, obs["agent_pos"])
        obs, done, info = env.step(action)
    puck_xy.append(tuple(env.puck.position))
    agent_xy.append(tuple(env.agent.position))
    return info["success"], frames, puck_xy, agent_xy, env.geom


def save_episode_visuals(out_dir: Path, stem: str, success, frames,
                         puck_xy, agent_xy, geom, title: str, **overlay_kwargs):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "success" if success else "fail"
    write_video(frames, out_dir / f"{stem}_{tag}.mp4")
    overlay = trajectory_overlay(geom, puck_xy, agent_xy, success=success,
                                 title=title, **overlay_kwargs)
    cv2.imwrite(str(out_dir / f"{stem}_{tag}_path.png"), overlay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sidecar", help="dataset *_episodes_meta.json: replay the "
                                      "exact recorded episodes (same seeds)")
    ap.add_argument("--per-level", type=int, default=2,
                    help="with --sidecar: episodes to replay per level "
                         "(-1 = ALL recorded episodes of each level)")
    ap.add_argument("--levels", type=int, nargs="+", choices=[2], default=[2])
    ap.add_argument("--eps", type=int, default=2, help="fresh expert episodes/level")
    ap.add_argument("--noise-std", type=float, default=1.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = []  # (level, env_seed, expert_seed, stem)
    if args.sidecar:
        meta = json.loads(Path(args.sidecar).read_text())
        assert "variant" not in meta.get("params", {}), (
            "This is a SwitchShot sidecar, but this tool only replays "
            "CorridorPush. Use python -m roboverse_learn.il.push2d.switchshot_viz")
        args.noise_std = meta["params"].get("noise_std", args.noise_std)
        cap = float("inf") if args.per_level < 0 else args.per_level
        count = {}
        for i, e in enumerate(meta["episodes"]):
            lv = e["level"]
            if count.get(lv, 0) >= cap or lv not in args.levels:
                continue
            count[lv] = count.get(lv, 0) + 1
            jobs.append((lv, e["seed"], e["seed"] + 1, f"demo_lv{lv}_ep{i:04d}"))
    else:
        for lv in args.levels:
            for k in range(args.eps):
                s = 77000 + lv * 100 + k
                jobs.append((lv, s, s + 1, f"expert_lv{lv}_ep{k:02d}"))

    for lv, es, xs, stem in jobs:
        success, frames, puck_xy, agent_xy, geom = record_expert_episode(
            lv, es, xs, noise_std=args.noise_std)
        save_episode_visuals(out, stem, success, frames, puck_xy, agent_xy,
                             geom, f"level {lv}")
        print(f"[push2d_viz] {stem}: {'success' if success else 'FAIL'} "
              f"({len(frames)} steps) -> {out}")


if __name__ == "__main__":
    main()
