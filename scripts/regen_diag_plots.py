#!/usr/bin/env python3
"""Regenerate diagnostic plots from saved .npz files.

Usage:
    # Process all .npz files in a directory:
    python scripts/regen_diag_plots.py /path/to/npz_data/

    # Process a single .npz file:
    python scripts/regen_diag_plots.py /path/to/npz_data/diag_data_demo33.npz

    # Specify a custom policy name for the legend:
    python scripts/regen_diag_plots.py /path/to/npz_data/ --policy-name flash_g

    # Disable fixed y-limits on the error column:
    python scripts/regen_diag_plots.py /path/to/npz_data/ --no-fixed-ylim

Output images are saved to a sibling directory named "regen_plots/" next to the
input directory (or next to the parent directory of a single file).
"""

import argparse
import pathlib
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from roboverse_learn.il.utils.visualization import (
    BASELINE_ERROR_YLIM,
    make_combined_diag_fig,
)

DEMO_RE = re.compile(r"diag_data_demo(\d+)\.npz$")


def process_npz(npz_path: pathlib.Path, out_dir: pathlib.Path, *,
                policy_name: str, error_ylims):
    d = np.load(str(npz_path))
    ctrl_dt = float(d["control_dt"])

    pos_des = d["pos_desired"] if "pos_desired" in d else None
    pos_act = d["pos_actual"] if "pos_actual" in d else None
    vel_des = d["vel_desired"] if "vel_desired" in d else None
    vel_act = d["vel_actual"] if "vel_actual" in d else None

    if pos_des is None and vel_des is None:
        print(f"  [SKIP] {npz_path.name}: no position or velocity data")
        return

    if pos_des is not None:
        n_joints = pos_des.shape[1]
    elif vel_des is not None:
        n_joints = vel_des.shape[1]
    else:
        return

    m = DEMO_RE.search(npz_path.name)
    demo_idx = int(m.group(1)) if m else 0

    fig = make_combined_diag_fig(
        pos_des, pos_act,
        vel_des, vel_act,
        ctrl_dt, demo_idx, n_joints,
        policy_name=policy_name,
        error_ylims=error_ylims,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"diag_demo{demo_idx}.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate diagnostic plots from .npz data files.")
    parser.add_argument("input", type=str,
                        help="Path to a directory of .npz files or a single .npz file.")
    parser.add_argument("--policy-name", type=str, default="policy",
                        help="Label shown in the legend for the desired curve "
                             "(default: 'policy').")
    parser.add_argument("--no-fixed-ylim", action="store_true",
                        help="Disable fixed y-limits on the error column.")
    args = parser.parse_args()

    inp = pathlib.Path(args.input).resolve()
    error_ylims = None if args.no_fixed_ylim else BASELINE_ERROR_YLIM

    if inp.is_file():
        npz_files = [inp]
        out_dir = inp.parent.parent / "regen_plots"
    elif inp.is_dir():
        npz_files = sorted(inp.glob("diag_data_demo*.npz"))
        out_dir = inp.parent / "regen_plots"
    else:
        print(f"Error: {inp} does not exist.", file=sys.stderr)
        sys.exit(1)

    if not npz_files:
        print(f"No diag_data_demo*.npz files found in {inp}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(npz_files)} npz file(s). Output -> {out_dir}/")
    for f in npz_files:
        process_npz(f, out_dir, policy_name=args.policy_name,
                    error_ylims=error_ylims)
    print("Done.")


if __name__ == "__main__":
    main()
