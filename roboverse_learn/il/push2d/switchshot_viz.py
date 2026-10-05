"""ForkReach (SwitchShot N/lv4) expert videos and trajectory overview PNGs.

Exact replay of recorded dataset episodes via the *_episodes_meta.json sidecar
(same env seed / expert seed / mode_bit / noise), re-rendered in high res.
Do NOT feed SwitchShot sidecars to push2d_viz (that CLI replays CorridorPush).

  # Replay the first two recorded episodes.
  python -m roboverse_learn.il.push2d.switchshot_viz \
      --sidecar data_policy/switchshot_N_lv4_100_episodes_meta.json \
      --per-level 2 --out outputs/forkreach_expert
  # Or generate fresh expert episodes.
  python -m roboverse_learn.il.push2d.switchshot_viz --eps 2 --out outputs/forkreach_expert
"""

import argparse
import json
from pathlib import Path

import cv2

from roboverse_learn.il.push2d import push2d_viz as viz
from roboverse_learn.il.push2d.switchshot_env import SwitchShotEnv, draw_switch_scene
from roboverse_learn.il.push2d.switchshot_expert import expert_for


def record_episode(level, variant, env_seed, expert_seed, mode_bit, noise_std):
    env = SwitchShotEnv(level=level, variant=variant, seed=env_seed)
    ex = expert_for(variant, env.geom, mode_bit=mode_bit, noise_std=noise_std,
                    seed=expert_seed)
    env.reset()
    frames, puck_xy, agent_xy = [], [], []
    done, info = False, {}
    while not done:
        puck_xy.append(tuple(env.subject.position))
        agent_xy.append(tuple(env.agent.position))
        c = env.visible_colors()
        f = cv2.resize(env.render(None), (viz.VIDEO_SIZE, viz.VIDEO_SIZE),
                       interpolation=cv2.INTER_AREA)
        frames.append(viz.annotate(
            f, f"expert L{level} t={env.t} [{c['top'][:1]}|{c['bottom'][:1]}]"))
        _, done, info = env.step(ex.act(env))
    puck_xy.append(tuple(env.subject.position))
    agent_xy.append(tuple(env.agent.position))
    # terminal-state frame: obs above are recorded PRE-step, so without this
    # the video ends one action step before the agent reaches the goal.
    # Hold a few copies so the outcome is actually visible.
    c = env.visible_colors()
    f = cv2.resize(env.render(None), (viz.VIDEO_SIZE, viz.VIDEO_SIZE),
                   interpolation=cv2.INTER_AREA)
    tag = "SUCCESS" if info.get("success") else str(info.get("fail_reason")).upper()
    frames.extend([viz.annotate(f, f"expert L{level} t={env.t} {tag}")] * 5)
    return info, frames, puck_xy, agent_xy, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sidecar", help="dataset *_episodes_meta.json (exact replay)")
    ap.add_argument("--per-level", type=int, default=2,
                    help="with --sidecar: episodes per level (-1 = ALL)")
    ap.add_argument("--levels", type=int, nargs="+", choices=[4], default=[4],
                    help="ForkReach uses level 4")
    ap.add_argument("--eps", type=int, default=2, help="fresh episodes/level")
    ap.add_argument("--variant", choices=["N"], default="N")
    ap.add_argument("--noise-std", type=float, default=1.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = []  # (level, variant, env_seed, expert_seed, mode_bit, noise, stem)
    if args.sidecar:
        meta = json.loads(Path(args.sidecar).read_text())
        params = meta["params"]
        if params.get("variant") != "N" or set(params.get("levels", [])) != {4}:
            raise SystemExit(
                "[switchshot_viz] expected a ForkReach sidecar (variant N, levels [4])")
        if params.get("derived") or int(params.get("downsample_ratio", 1)) != 1:
            raise SystemExit("[switchshot_viz] exact replay requires original demonstration frames")
        cap = float("inf") if args.per_level < 0 else args.per_level
        count = {}
        for i, e in enumerate(meta["episodes"]):
            lv = e["level"]
            if lv != 4 or e.get("variant", "N") != "N":
                raise SystemExit("[switchshot_viz] sidecar contains an unsupported task")
            if count.get(lv, 0) >= cap:
                continue
            count[lv] = count.get(lv, 0) + 1
            jobs.append((lv, e["variant"], e["seed"], e["seed"] + 7,
                         e.get("expert_chosen_mode"), params["noise_std"],
                         f"demo_lv{lv}_ep{i:04d}"))
    else:
        for lv in args.levels:
            for k in range(args.eps):
                s = 88000 + lv * 100 + k
                mode = ["top", "bottom"][k % 2]
                jobs.append((lv, args.variant, s, s + 7, mode, args.noise_std,
                             f"expert_lv{lv}_ep{k:02d}"))

    if not jobs:
        raise SystemExit("[switchshot_viz] No episodes matched; check that "
                         "--levels matches the levels in the sidecar")
    for lv, variant, es, xs, mode, noise, stem in jobs:
        info, frames, puck_xy, agent_xy, env = record_episode(
            lv, variant, es, xs, mode, noise)
        colors = env.visible_colors()   # end-of-episode signal state
        viz.save_episode_visuals(
            out, stem, info.get("success"), frames, puck_xy, agent_xy, env.geom,
            f"L{lv}" + (f" mode={mode}" if mode else ""),
            draw_scene_fn=lambda g, _c=colors: draw_switch_scene(g, _c),
            crop_band=None)
        tag = "success" if info.get("success") else info.get("fail_reason")
        print(f"[switchshot_viz] {stem}: {tag} ({len(frames)} steps) -> {out}")


if __name__ == "__main__":
    main()
