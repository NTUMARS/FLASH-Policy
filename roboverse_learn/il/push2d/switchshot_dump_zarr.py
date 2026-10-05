"""Render stored ForkReach (SwitchShot N/lv4) frames without expert replay.

  python -m roboverse_learn.il.push2d.switchshot_dump_zarr \
      --zarr data_policy/switchshot_N_lv4_100.zarr \
      --out outputs/forkreach_stored --per-level 2
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import zarr

from roboverse_learn.il.push2d.push2d_viz import write_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-level", type=int, default=2, help="-1 = all")
    ap.add_argument("--levels", type=int, nargs="+", choices=[4], default=[4])
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--scale", type=int, default=5, help="96px*scale nearest-neighbor upscaling")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    g = zarr.open(args.zarr, "r")
    frames_all = g["data/head_camera"]          # (N,3,96,96) uint8 CHW BGR
    ee = g["meta/episode_ends"][:]
    starts = np.concatenate([[0], ee[:-1]])
    sidecar = Path(args.zarr).with_name(Path(args.zarr).stem + "_episodes_meta.json")
    sidecar_data = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    eps_meta = sidecar_data.get("episodes") or [{}] * len(ee)
    if sidecar_data:
        params = sidecar_data.get("params", {})
        if params.get("variant") != "N" or set(params.get("levels", [])) != {4}:
            raise SystemExit("[switchshot_dump] expected ForkReach data (variant N, levels [4])")

    cap = float("inf") if args.per_level < 0 else args.per_level
    count, n_out = {}, 0
    for k in range(len(ee)):
        lv = eps_meta[k].get("level", 4)
        if args.levels is not None and lv not in args.levels:
            continue
        if count.get(lv, 0) >= cap:
            continue
        count[lv] = count.get(lv, 0) + 1
        f = frames_all[starts[k]:ee[k]]         # CHW BGR
        size = f.shape[-1] * args.scale
        stem = f"dump_lv{lv}_ep{k:04d}"
        # Use H.264 through push2d_viz.write_video, as elsewhere in the repo.
        # OpenCV's mp4v is MPEG-4 Part 2, which VS Code and Chromium cannot
        # decode even though VLC can.
        clip = []
        for t in range(len(f)):
            img = cv2.resize(np.moveaxis(f[t], 0, -1), (size, size),
                             interpolation=cv2.INTER_NEAREST)
            cv2.putText(img, f"L{lv} ep{k} frame {t}/{len(f)-1} (stored)",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            clip.append(img)
        write_video(clip, out / f"{stem}.mp4", fps=args.fps)
        cv2.imwrite(str(out / f"{stem}_last.png"),
                    cv2.resize(np.moveaxis(f[-1], 0, -1), (size, size),
                               interpolation=cv2.INTER_NEAREST))
        n_out += 1
    if n_out == 0:
        raise SystemExit("[switchshot_dump] No episodes matched; check --levels")
    print(f"[switchshot_dump] {n_out} eps -> {out}")


if __name__ == "__main__":
    main()
