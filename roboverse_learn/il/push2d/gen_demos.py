"""Generate CorridorPush demonstrations into a RobotImageDataset-compatible zarr.

Writes via ReplayBuffer.add_episode (guarantees /data + /meta/episode_ends
layout), head_camera stored CHW uint8 (training fast path does no moveaxis),
episode order shuffled across levels (val mask quirk: last episode becomes the
val set), per-episode level recorded in a JSON sidecar.

Usage (with the project dependencies installed):
  python -m roboverse_learn.il.push2d.gen_demos \
      --out /path/corridor_push.zarr --eps-per-level 150 --levels 2
"""

import argparse
import json
from pathlib import Path

import numpy as np

from roboverse_learn.il.push2d.corridor_push_env import CorridorPushEnv
from roboverse_learn.il.push2d.scripted_expert import ScriptedExpert
from roboverse_learn.il.utils.replay_buffer import ReplayBuffer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", type=int, nargs="+", choices=[2], default=[2])
    ap.add_argument("--eps-per-level", type=int, default=150)
    ap.add_argument("--render-size", type=int, default=96)
    ap.add_argument("--noise-std", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-retry", type=int, default=3)
    args = ap.parse_args()

    est_gb = (len(args.levels) * args.eps_per_level * 45
              * 3 * args.render_size ** 2) / 1e9
    print(f"[gen_demos] levels={args.levels} eps/level={args.eps_per_level} "
          f"render={args.render_size} -> rough RAM/zarr estimate ~{est_gb:.1f} GB")

    jobs = [(lv, i) for lv in args.levels for i in range(args.eps_per_level)]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(jobs)

    rb = ReplayBuffer.create_empty_numpy()
    meta, attempts = [], {lv: [0, 0] for lv in args.levels}  # lv -> [succ, tries]
    for lv, i in jobs:
        ok = False
        for r in range(args.max_retry):
            ep_seed = args.seed * 1_000_000 + lv * 10_000 + i * 10 + r
            env = CorridorPushEnv(level=lv, render_size=args.render_size,
                                  seed=ep_seed)
            expert = ScriptedExpert(env.geom, noise_std=args.noise_std,
                                    seed=ep_seed + 1)
            attempts[lv][1] += 1
            success, frames, states, actions = expert.rollout(env)
            if success:
                rb.add_episode({"head_camera": frames, "state": states,
                                "action": actions})
                meta.append({"level": lv, "seed": ep_seed, "len": len(frames)})
                attempts[lv][0] += 1
                ok = True
                break
        if not ok:
            print(f"[gen_demos] WARNING: level {lv} episode {i} failed "
                  f"{args.max_retry} retries — skipped")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rb.save_to_path(str(out))
    sidecar = out.with_name(out.stem + "_episodes_meta.json")
    sidecar.write_text(json.dumps({
        "episodes": meta,
        "params": {k: getattr(args, k) for k in
                   ("levels", "eps_per_level", "render_size", "noise_std", "seed")},
    }, indent=1))

    print(f"[gen_demos] wrote {rb.n_episodes} episodes / {rb.n_steps} steps -> {out}")
    print(f"[gen_demos] sidecar -> {sidecar}")
    for lv in args.levels:
        s, t = attempts[lv]
        rate = s / max(t, 1)
        flag = "" if rate >= 0.95 else "  <-- BELOW 95% ACCEPTANCE"
        print(f"[gen_demos] level {lv}: {s}/{t} expert success ({rate:.1%}){flag}")


if __name__ == "__main__":
    main()
