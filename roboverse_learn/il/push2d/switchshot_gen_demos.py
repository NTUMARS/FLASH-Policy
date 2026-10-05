"""Generate ForkReach demonstrations (historical SwitchShot N/lv4).

Every pair uses one environment seed with opposite expert modes: identical
initial observations admit two correct continuations. Both halves are retried
and accepted together so a failed rollout cannot break the paired protocol.

    python -m roboverse_learn.il.push2d.switchshot_gen_demos \\
        --out data_policy/switchshot_N_lv4_100.zarr --eps-per-level 100
"""

import argparse
import json
from pathlib import Path

import numpy as np

from roboverse_learn.il.push2d.switchshot_env import SwitchShotEnv
from roboverse_learn.il.push2d.switchshot_expert import expert_for
from roboverse_learn.il.utils.replay_buffer import ReplayBuffer


def collect_pair(env_seed, pair, render_size=96, noise_std=1.0, max_retry=4):
    """Return two successful opposite-goal episodes sharing the same reset."""
    for retry in range(max_retry):
        seed = env_seed + retry
        halves = []
        for mode in ("top", "bottom"):
            env = SwitchShotEnv(level=4, variant="N", render_size=render_size, seed=seed)
            ex = expert_for("N", env.geom, mode_bit=mode, noise_std=noise_std, seed=seed + 7)
            success, info, frames, states, actions = ex.rollout(env)
            if not success:
                break
            sc = info["schedule"]
            meta = {"level": 4, "seed": seed, "variant": "N",
                    "s0": sc["s0"], "s1": sc["s1"], "lock": sc["lock"],
                    "initial": sc["initial"], "final": sc["final"],
                    "valid_goal_set": list(sc["valid_goals"]),
                    "expert_chosen_mode": mode, "pair": pair,
                    "retried": retry, "len": len(frames)}
            halves.append(({"head_camera": frames, "state": states, "action": actions}, meta))
        if len(halves) == 2:
            return halves
    raise RuntimeError(f"ForkReach pair {pair}, seed {env_seed} failed {max_retry} retries")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=["N"], default="N")
    ap.add_argument("--levels", type=int, nargs="+", choices=[4], default=[4])
    ap.add_argument("--eps-per-level", type=int, default=100)
    ap.add_argument("--render-size", type=int, default=96)
    ap.add_argument("--noise-std", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-retry", type=int, default=4)
    args = ap.parse_args()
    if args.levels != [4]:
        ap.error("--levels must be exactly 4")
    if args.eps_per_level < 2 or args.eps_per_level % 2:
        ap.error("--eps-per-level must be a positive even count for matched pairs")
    if args.max_retry < 1:
        ap.error("--max-retry must be positive")

    episodes = []
    for pair in range(args.eps_per_level // 2):
        # Preserve the audited generator's original N/lv4 seed convention.
        env_seed = args.seed * 1_000_000 + 4 * 20_000 + pair * 20
        episodes.extend(collect_pair(env_seed, pair, args.render_size,
                                     args.noise_std, args.max_retry))
    # With successful first attempts this yields the original shuffled order.
    np.random.default_rng(args.seed).shuffle(episodes)
    rb = ReplayBuffer.create_empty_numpy()
    meta = []
    for episode, episode_meta in episodes:
        rb.add_episode(episode)
        meta.append(episode_meta)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rb.save_to_path(str(out))
    sidecar = out.with_name(out.stem + "_episodes_meta.json")
    sidecar.write_text(json.dumps({"episodes": meta, "params": vars(args)}, indent=1))
    print(f"[switchshot_gen] wrote {rb.n_episodes} episodes / {rb.n_steps} steps -> {out}")
    print(f"[switchshot_gen] L4 pairs complete: {len(episodes)//2}/{args.eps_per_level//2}")


if __name__ == "__main__":
    main()
