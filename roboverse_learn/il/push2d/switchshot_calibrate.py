"""Zero-training calibration of retained ForkReach (SwitchShot N/lv4).

Checks navigation success, opposite-goal pairing from identical observations,
expert action spectrum, and visibility of both green goal regions at 96px.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from roboverse_learn.il.push2d.push2d_spectrum import fit_error
from roboverse_learn.il.push2d.switchshot_env import SwitchShotEnv
from roboverse_learn.il.push2d.switchshot_expert import expert_for


def gate1_expert(eps, variant="N"):
    outcomes = Counter()
    for ep in range(eps):
        env = SwitchShotEnv(level=4, variant=variant, seed=41000 + ep)
        mode = ["top", "bottom"][ep % 2]
        ex = expert_for(variant, env.geom, mode_bit=mode, seed=42000 + ep)
        success, info, *_ = ex.rollout(env, record=False)
        outcomes[info["fail_reason"] or "success"] += 1
    rate = outcomes["success"] / eps
    ok = rate >= 0.95
    return ok, ["== Gate 1: expert success ==",
                f"  L4: {dict(outcomes)} success={rate:.1%}" + ("  <-- FAIL" if not ok else "")]


def gate2_pairs(eps, variant="N"):
    ok = True
    for ep in range(eps):
        trajectories = []
        for mode in ("top", "bottom"):
            env = SwitchShotEnv(level=4, variant=variant, seed=43000 + ep)
            ex = expert_for(variant, env.geom, mode_bit=mode, seed=44000 + ep)
            success, info, frames, states, actions = ex.rollout(env)
            gx, gy = env.geom.goals[mode]
            ok &= success and np.hypot(env.agent.position.x - gx, env.agent.position.y - gy) <= env.geom.goal_r
            trajectories.append((frames[0], states[0], actions[0]))
        top, bottom = trajectories
        ok &= np.array_equal(top[0], bottom[0]) and np.array_equal(top[1], bottom[1])
        ok &= bool(top[2][1] < bottom[2][1])
    return ok, ["== Gate 2: paired opposite modes ==",
                f"  {eps} pairs: identical initial observations, both chosen goals reachable"
                + ("  <-- FAIL" if not ok else "")]


def gate3_spectrum(eps, variant="N"):
    e6s, e9s = [], []
    for ep in range(min(eps, 10)):
        env = SwitchShotEnv(level=4, variant=variant, seed=45000 + ep)
        mode = ["top", "bottom"][ep % 2]
        ex = expert_for(variant, env.geom, mode_bit=mode, seed=46000 + ep)
        success, info, frames, states, actions = ex.rollout(env)
        if not success:
            continue
        a = actions / 256.0 - 1.0
        for i in range(0, len(a) - 12, 2):
            e6s.append(fit_error(a[i:i + 12], 6))
            e9s.append(fit_error(a[i:i + 12], 9))
    e6px = float(np.sqrt(np.mean(e6s)) * 256) if e6s else float("nan")
    e9px = float(np.sqrt(np.mean(e9s)) * 256) if e9s else float("nan")
    ok = np.isfinite(e6px) and e6px <= 10.0
    return ok, ["== Gate 3: expert action spectrum ==",
                f"  L4: e6={e6px:.1f}px e9={e9px:.1f}px" + ("  <-- FAIL" if not ok else "")]


def gate4_visibility(variant="N"):
    env = SwitchShotEnv(level=4, variant=variant, seed=47000)
    obs = env.reset()
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    image = np.moveaxis(obs["image"][:, :, ::-1], -1, 0).astype(np.float32) / 255.0
    feats = (image - mean) / std
    yy, xx = np.ogrid[:96, :96]
    ok, lines = True, ["== Gate 4: both goal regions visible at 96px =="]
    for side, (gx, gy) in env.geom.goals.items():
        scale = 96 / 512.0
        cx, cy, radius = int(gx * scale), int(gy * scale), max(2, int(env.geom.goal_r * scale))
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
        goal = feats[:, mask].mean(axis=1)
        background = (np.full(3, 24 / 255.0) - mean[:, 0, 0]) / std[:, 0, 0]
        contrast = float(np.sqrt(np.mean((goal - background) ** 2)))
        visible = contrast >= 0.3
        ok &= visible
        lines.append(f"  {side}: goal/background contrast={contrast:.2f} over {int(mask.sum())}px"
                     + ("  <-- FAIL" if not visible else ""))
    return ok, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eps", type=int, default=30)
    ap.add_argument("--variant", choices=["N"], default="N")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.eps < 1:
        ap.error("--eps must be positive")
    all_ok, report = True, [f"ForkReach calibration (variant N, level 4, eps={args.eps})"]
    for check in (lambda: gate1_expert(args.eps, args.variant),
                  lambda: gate2_pairs(args.eps, args.variant),
                  lambda: gate3_spectrum(args.eps, args.variant),
                  lambda: gate4_visibility(args.variant)):
        ok, lines = check()
        all_ok &= ok
        report += lines
    report.append(f"== OVERALL: {'PASS' if all_ok else 'FAIL'} ==")
    text = "\n".join(report) + "\n"
    print(text, end="")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
