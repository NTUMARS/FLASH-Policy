"""CorridorPush calibration (PLAN.md §0/§5): spectrum + oracle closed-loop gates.

Spectrum mode: per-level fit-error-vs-order curves measured on REAL training
windows (RobotImageDataset + SequenceSampler incl. fit_pad extension), in the
flash preprocessing space (action normalizer + poly-region slice).

Oracle mode  : representation-level closed-loop gate — a perfect expert whose
per-chunk action sequence is projected onto a degree-K polynomial before
execution.  K6-oracle failing while K9-oracle succeeds proves task failure is
caused by the representation order, independent of learning.

  python -m roboverse_learn.il.push2d.push2d_spectrum --zarr .../cp.zarr
  python -m roboverse_learn.il.push2d.push2d_spectrum --oracle 6 9 --levels 0 3 4 --eps 20
"""

import argparse
import json
from pathlib import Path

import numpy as np


def legendre_basis(n_pts: int, degree: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n_pts)
    x = 2 * s - 1
    P = np.zeros((n_pts, degree + 1))
    P[:, 0] = 1.0
    if degree >= 1:
        P[:, 1] = x
    for n in range(1, degree):
        P[:, n + 1] = ((2 * n + 1) * x * P[:, n] - n * P[:, n - 1]) / (n + 1)
    return P


def fit_error(seq: np.ndarray, degree: int) -> float:
    S = legendre_basis(len(seq), degree)
    coef, *_ = np.linalg.lstsq(S, seq, rcond=None)
    return float(np.mean((S @ coef - seq) ** 2))


def spectrum(args):
    import torch
    from roboverse_learn.il.datasets.robot_image_dataset import RobotImageDataset

    fp = args.fit_pad
    horizon_ext = args.horizon + 2 * fp
    ds = RobotImageDataset(args.zarr, horizon=horizon_ext,
                           pad_before=args.n_obs - 1 + fp,
                           pad_after=args.horizon - args.n_obs - 1 + fp + 2,
                           batch_size=1, val_ratio=0.0)
    sidecar = Path(args.zarr).with_name(Path(args.zarr).stem + "_episodes_meta.json")
    ep_levels = [e["level"] for e in json.loads(sidecar.read_text())["episodes"]]
    ep_ends = np.asarray(ds.replay_buffer.episode_ends)
    norm = ds.get_normalizer()["action"]
    poly_start = args.n_obs - 1 - 1  # n_obs-1-poly_prefix
    H_ext = 10 + 2 * fp              # H_poly + 2*fit_pad

    per_level = {}
    for i in range(len(ds.sampler)):
        buf_start = int(ds.sampler.indices[i][0])
        ep = int(np.searchsorted(ep_ends, buf_start, side="right"))
        lv = ep_levels[ep]
        sample = ds.sampler.sample_sequence(i)
        act = norm.normalize(torch.from_numpy(sample["action"].astype(np.float32))).numpy()
        win = act[poly_start:poly_start + H_ext]
        rec = per_level.setdefault(lv, {k: [] for k in range(3, 10)})
        for k in range(3, 10):
            rec[k].append(fit_error(win, k))

    lines, records = ["level |   e6(px RMS)   e9(px RMS)   e6/e9 | verdict"], []
    scale = 256.0  # normalized [-1,1] -> px (world 512)
    for lv in sorted(per_level):
        e6 = float(np.mean(per_level[lv][6]))
        e9 = float(np.mean(per_level[lv][9]))
        r = e6 / max(e9, 1e-12)
        verdict = "K6-hostile" if r >= 5 and np.sqrt(e6) * scale > 8.0 else "smooth/mild"
        lines.append(f"  {lv}   | {e6:.2e}({np.sqrt(e6)*scale:5.1f}) "
                     f"{e9:.2e}({np.sqrt(e9)*scale:5.1f})  {r:6.1f} | {verdict}")
        records.append({"level": lv, "verdict": verdict, "e6_px_rms": np.sqrt(e6) * scale,
                        "e9_px_rms": np.sqrt(e9) * scale, "ratio_e6_e9": r,
                        "e_k_mean": {k: float(np.mean(per_level[lv][k])) for k in range(3, 10)}})
    print("\n".join(lines))
    return lines, records


def oracle(args):
    from roboverse_learn.il.push2d.corridor_push_env import CorridorPushEnv
    from roboverse_learn.il.push2d.scripted_expert import ScriptedExpert

    n_virtual, n_exec, n_hist = 10, 8, 2
    lines, records = [], []
    for lv in args.levels:
        for K in args.oracle:
            S = legendre_basis(n_hist + n_virtual, K)
            proj = S @ np.linalg.lstsq(S, np.eye(n_hist + n_virtual), rcond=None)[0]
            succ = 0
            for ep in range(args.eps):
                env = CorridorPushEnv(level=lv, seed=9000 + ep)
                ex = ScriptedExpert(env.geom, noise_std=0.0, seed=9500 + ep)
                obs = env.reset()
                hist = [np.asarray(obs["agent_pos"], dtype=np.float64)] * n_hist
                done, info = False, {"success": False}
                while not done:
                    snap = env.snapshot()
                    virtual = []
                    vobs, vdone = obs, False
                    for _ in range(n_virtual):
                        a = ex.act(env.puck.position, vobs["agent_pos"])
                        virtual.append(np.asarray(a, dtype=np.float64))
                        if not vdone:
                            vobs, vdone, _ = env.step(a)
                    env.restore(snap)
                    seq = np.stack(hist + virtual)          # (12, 2)
                    fitted = proj @ seq                     # degree-K projection
                    for a in fitted[n_hist:n_hist + n_exec]:
                        if done:
                            break
                        obs, done, info = env.step(a.astype(np.float32))
                        hist = [hist[-1], a]
                succ += int(info["success"])
            line = f"oracle level {lv} K={K}: success {succ}/{args.eps}"
            print(line)
            lines.append(line)
            records.append({"level": lv, "K": K, "success": succ, "eps": args.eps})
    return lines, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr")
    ap.add_argument("--fit-pad", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--n-obs", type=int, default=8)
    ap.add_argument("--oracle", type=int, nargs="*")
    ap.add_argument("--levels", type=int, nargs="+", choices=[2], default=[2])
    ap.add_argument("--eps", type=int, default=20)
    ap.add_argument("--out", help="report path prefix (default: alongside the zarr)")
    args = ap.parse_args()

    all_lines, report = [], {}
    if args.oracle:
        lines, rec = oracle(args)
        all_lines += ["== oracle closed-loop =="] + lines
        report["oracle"] = rec
    if args.zarr:
        lines, rec = spectrum(args)
        all_lines += ["== fit-error spectrum =="] + lines
        report["spectrum"] = rec

    # ---- persist the calibration report (txt for humans, json for plotting) ----
    if args.out:
        prefix = Path(args.out)
    elif args.zarr:
        prefix = Path(args.zarr).with_name(Path(args.zarr).stem + "_calibration")
    else:
        prefix = Path("push2d_calibration")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(str(prefix) + ".txt").write_text("\n".join(all_lines) + "\n")
    Path(str(prefix) + ".json").write_text(json.dumps(report, indent=1))
    print(f"[push2d_spectrum] report saved -> {prefix}.txt / .json")


if __name__ == "__main__":
    main()
