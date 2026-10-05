"""Visualise how Legendre polynomial fitting + receding-horizon stitching
reproduces the *expert* trajectory.

Supports an **extended-window** mode (--fit_pad N): fit the polynomial to
H + 2*N time-steps but only keep the middle H steps.  This tests whether
giving the least-squares fit more context improves velocity continuity at
chunk boundaries.

Supports a **spline** mode (--spline): fit a global C2-continuous cubic
interpolating spline to each episode, then project each window's spline
segment onto the Legendre basis via dense sampling.

Usage:
    # Original (H=16, no padding)
    python scripts/visualize_expert_poly_fit.py \
        --zarr data_policy/stack_cubeFrankaL0_obs:joint_pos_act:joint_pos_ds4_100.zarr \
        --episodes 0 1 2 --output_dir il_outputs/expert_poly_diag

    # Extended window: fit 32 steps, keep middle 16
    python scripts/visualize_expert_poly_fit.py \
        --zarr data_policy/stack_cubeFrankaL0_obs:joint_pos_act:joint_pos_ds4_100.zarr \
        --episodes 0 1 2 --output_dir il_outputs/expert_poly_diag --fit_pad 8

    # Side-by-side comparison (original vs extended, on one figure)
    python scripts/visualize_expert_poly_fit.py \
        --zarr data_policy/stack_cubeFrankaL0_obs:joint_pos_act:joint_pos_ds4_100.zarr \
        --episodes 0 1 2 --output_dir il_outputs/expert_poly_diag --compare

    # Spline-projected comparison (original vs spline, on one figure)
    python scripts/visualize_expert_poly_fit.py \
        --zarr data_policy/stack_cubeFrankaL0_obs:joint_pos_act:joint_pos_ds4_100.zarr \
        --episodes 0 1 2 --output_dir il_outputs/expert_poly_diag --spline
"""

import argparse
import glob
import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from roboverse_learn.il.utils.spline_projection import (
    fit_episode_splines,
    _project_segment_to_legendre,
)


# ── Legendre basis (mirrored from flash_g_policy.py) ──────────────

def build_legendre_basis(s: torch.Tensor, degree: int):
    x = 2 * s - 1
    H = len(s)
    C = degree + 1
    P = torch.zeros(H, C, dtype=s.dtype)
    dP = torch.zeros(H, C, dtype=s.dtype)
    P[:, 0] = 1.0
    if C > 1:
        P[:, 1] = x
        dP[:, 1] = 1.0
    for n in range(1, C - 1):
        P[:, n + 1] = ((2 * n + 1) * x * P[:, n] - n * P[:, n - 1]) / (n + 1)
        dP[:, n + 1] = ((2 * n + 1) * (P[:, n] + x * dP[:, n]) - n * dP[:, n - 1]) / (n + 1)
    return P, 2 * dP


def make_fit_matrices(H, poly_order):
    """Return eval_mat (H,C), d1_mat (H,C), lstsq_mat (C,H)."""
    s = torch.linspace(0.0, 1.0, H, dtype=torch.float64)
    E, D1 = build_legendre_basis(s, poly_order)
    E, D1 = E.numpy(), D1.numpy()
    StS = E.T @ E
    L = np.linalg.solve(StS, E.T)
    return E, D1, L


def make_dense_eval_matrices(H, H_ext, fit_pad, poly_order, dense_factor=10):
    """Return dense evaluation matrices for the middle H portion of the H_ext window."""
    n_dense = (H - 1) * dense_factor + 1
    s_ext = torch.linspace(0.0, 1.0, H_ext, dtype=torch.float64)
    s_mid = s_ext[fit_pad:fit_pad + H] if fit_pad > 0 else s_ext
    s_dense = torch.linspace(s_mid[0].item(), s_mid[-1].item(), n_dense, dtype=torch.float64)
    E, D1 = build_legendre_basis(s_dense, poly_order)
    return E.numpy(), D1.numpy(), n_dense


def build_constraint_correction(eval_mat, constraint_indices):
    """Pre-compute the correction matrix for C0 constrained least-squares.

    Returns (A, correction) where A is (K, C) and correction is (C, K).
    """
    A = eval_mat[constraint_indices, :]                         # (K, C)
    StS = eval_mat.T @ eval_mat                                 # (C, C)
    StS_inv = np.linalg.solve(StS, np.eye(StS.shape[0]))       # (C, C)
    StS_inv_AT = StS_inv @ A.T                                  # (C, K)
    M = A @ StS_inv_AT                                          # (K, K)
    correction = StS_inv_AT @ np.linalg.inv(M)                  # (C, K)
    return A, correction


def build_c1_constraint(eval_mat, d1_mat, pos_indices):
    """Pre-compute correction matrix for C1 constraints (position + velocity).

    4 constraints: position and velocity at each index in pos_indices.
    Returns (A_pos, A_vel, correction) where correction is (C, 4).
    """
    A_pos = eval_mat[pos_indices, :]                            # (2, C)
    A_vel = d1_mat[pos_indices, :]                              # (2, C)
    A = np.vstack([A_pos, A_vel])                               # (4, C)
    StS = eval_mat.T @ eval_mat
    StS_inv = np.linalg.solve(StS, np.eye(StS.shape[0]))
    StS_inv_AT = StS_inv @ A.T                                  # (C, 4)
    M = A @ StS_inv_AT                                          # (4, 4)
    correction = StS_inv_AT @ np.linalg.inv(M)                  # (C, 4)
    return A_pos, A_vel, correction


# ── Stitching engine ─────────────────────────────────────────────────────

def stitch_episode(ep_action, H, To, Ta, poly_order, ctrl_dt, fit_pad=0,
                   constrained=False, dense_factor=0, poly_prefix=None,
                   poly_suffix=0):
    """Run receding-horizon polynomial stitching on one episode.

    If fit_pad > 0, each chunk is fit to (H + 2*fit_pad) expert steps,
    but only the middle H steps are kept.

    If constrained=True, each polynomial is pinned to exactly pass through
    the expert at anchor and last-action indices (C0 boundary constraint).

    If dense_factor > 0, also returns dense curve segments for plotting.

    Returns (stitched_pos, stitched_vel, boundaries, dense_segs).
    dense_segs is a list of (t_arr, pos_arr, vel_arr) or None.
    """
    Da = ep_action.shape[1]
    L = len(ep_action)

    if poly_prefix is not None:
        H_poly = poly_prefix + Ta + poly_suffix
        anchor_idx = poly_prefix
        poly_offset = To - 1 - poly_prefix
    else:
        H_poly = H
        anchor_idx = To - 1
        poly_offset = 0

    H_ext = H_poly + 2 * fit_pad
    n_eval = H_poly + (1 if poly_prefix is not None and fit_pad > 0 else 0)
    eval_ext, d1_ext, lstsq_ext = make_fit_matrices(H_ext, poly_order)

    eval_core, d1_core, lstsq_core = make_fit_matrices(H_poly, poly_order) if fit_pad > 0 else (None, None, None)

    bc_anchor_i = fit_pad + anchor_idx
    bc_overlap_i = fit_pad + anchor_idx + Ta

    bc_A_pos = bc_A_vel = bc_correction = None
    if constrained:
        bc_A_pos, bc_A_vel, bc_correction = build_c1_constraint(
            eval_ext, d1_ext, [bc_anchor_i, bc_overlap_i])

    eval_dense = d1_dense = n_dense_pts = None
    if dense_factor > 0:
        eval_dense, d1_dense, n_dense_pts = make_dense_eval_matrices(
            n_eval, H_ext, fit_pad, poly_order, dense_factor)

    pad_before = (To - 1) + fit_pad
    pad_after = (Ta - 1) + fit_pad
    padded = np.concatenate([
        np.tile(ep_action[0:1], (pad_before, 1)),
        ep_action,
        np.tile(ep_action[-1:], (pad_after, 1)),
    ], axis=0)

    stitched_pos = np.full((L, Da), np.nan)
    stitched_vel = np.full((L, Da), np.nan)
    boundaries = []
    overlap_gaps_pos = []
    overlap_gaps_vel = []
    dense_segs = [] if dense_factor > 0 else None

    T_ext = (H_ext - 1) * ctrl_dt
    prev_overlap_pos = None
    prev_overlap_vel = None
    first_chunk = True

    t_cursor = 0
    while t_cursor < L:
        orig_win_start = t_cursor + pad_before - (To - 1)
        poly_win_start = orig_win_start + poly_offset
        ext_win_start = poly_win_start - fit_pad

        if ext_win_start < 0 or ext_win_start + H_ext > len(padded):
            break

        chunk_ext = padded[ext_win_start: ext_win_start + H_ext]
        coeff = lstsq_ext @ chunk_ext

        if bc_A_pos is not None:
            cur_pos = bc_A_pos @ coeff                                 # (2, Da)
            tgt_pos = chunk_ext[[bc_anchor_i, bc_overlap_i], :]        # (2, Da)
            cur_vel = bc_A_vel @ coeff                                 # (2, Da)
            vel_anchor = (chunk_ext[bc_anchor_i + 1] - chunk_ext[bc_anchor_i - 1]) * (H_ext - 1) / 2
            if bc_overlap_i + 1 < H_ext:
                vel_overlap = (chunk_ext[bc_overlap_i + 1] - chunk_ext[bc_overlap_i - 1]) * (H_ext - 1) / 2
            elif bc_overlap_i >= 2:
                vel_overlap = (3 * chunk_ext[bc_overlap_i] - 4 * chunk_ext[bc_overlap_i - 1] + chunk_ext[bc_overlap_i - 2]) * (H_ext - 1) / 2
            else:
                vel_overlap = (chunk_ext[bc_overlap_i] - chunk_ext[bc_overlap_i - 1]) * (H_ext - 1)
            tgt_vel = np.stack([vel_anchor, vel_overlap])
            residual = np.vstack([cur_pos - tgt_pos, cur_vel - tgt_vel])
            coeff = coeff - bc_correction @ residual

            if lstsq_core is not None:
                traj_core = eval_ext[fit_pad:fit_pad + H_poly, :] @ coeff
                coeff = lstsq_core @ traj_core

        if constrained and lstsq_core is not None:
            T_core = (H_poly - 1) * ctrl_dt if H_poly > 1 else 1.0
            recon_mid = eval_core @ coeff
            vel_mid = d1_core @ coeff / T_core
        else:
            recon_ext = eval_ext @ coeff
            vel_ext = d1_ext @ coeff / T_ext
            recon_mid = recon_ext[fit_pad:fit_pad + n_eval]
            vel_mid = vel_ext[fit_pad:fit_pad + n_eval]

        anchor_pos_val = recon_mid[anchor_idx]
        anchor_vel_val = vel_mid[anchor_idx]
        if prev_overlap_pos is not None:
            overlap_gaps_pos.append(np.abs(anchor_pos_val - prev_overlap_pos))
            overlap_gaps_vel.append(np.abs(anchor_vel_val - prev_overlap_vel))
        if not constrained or lstsq_core is None:
            if bc_overlap_i < H_ext:
                prev_overlap_pos = recon_ext[bc_overlap_i].copy()
                prev_overlap_vel = vel_ext[bc_overlap_i].copy()
            else:
                prev_overlap_pos = recon_mid[-1].copy()
                prev_overlap_vel = vel_mid[-1].copy()
        else:
            prev_overlap_pos = recon_mid[-1].copy()
            prev_overlap_vel = vel_mid[-1].copy()

        act_start = anchor_idx
        n_write = min(Ta, L - t_cursor)
        stitched_pos[t_cursor:t_cursor + n_write] = recon_mid[act_start:act_start + n_write]
        stitched_vel[t_cursor:t_cursor + n_write] = vel_mid[act_start:act_start + n_write]

        if dense_segs is not None:
            poly_phys_start = t_cursor - anchor_idx
            if constrained and lstsq_core is not None:
                n_pts_d = (H_poly - 1) * dense_factor + 1
                t_chunk = poly_phys_start + np.linspace(0, H_poly - 1, n_pts_d)
                s_d = np.linspace(0, 1, n_pts_d)
                ed, dd = build_legendre_basis(torch.from_numpy(s_d).double(), poly_order)
                pos_d = ed.numpy() @ coeff
                T_core = (H_poly - 1) * ctrl_dt if H_poly > 1 else 1.0
                vel_d = dd.numpy() @ coeff / T_core
            else:
                t_chunk = poly_phys_start + np.linspace(0, n_eval - 1, n_dense_pts)
                pos_d = eval_dense @ coeff
                vel_d = d1_dense @ coeff / T_ext
                n_pts_d = n_dense_pts
            ws = 0 if first_chunk else anchor_idx
            dws = ws * dense_factor
            dwe = n_pts_d
            dense_segs.append((t_chunk[dws:dwe], pos_d[dws:dwe], vel_d[dws:dwe]))

        if t_cursor > 0:
            boundaries.append(t_cursor)
        first_chunk = False
        t_cursor += Ta

    ogp = np.array(overlap_gaps_pos) if overlap_gaps_pos else np.zeros((0, Da))
    ogv = np.array(overlap_gaps_vel) if overlap_gaps_vel else np.zeros((0, Da))
    return stitched_pos, stitched_vel, np.array(boundaries), dense_segs, ogp, ogv


def stitch_episode_spline(ep_action, H, To, Ta, poly_order, ctrl_dt, fit_pad=0,
                          constrained=False, dense_factor=0):
    """Run receding-horizon stitching using global-spline-projected Legendre.

    Returns (stitched_pos, stitched_vel, boundaries, dense_segs).
    """
    Da = ep_action.shape[1]
    L = len(ep_action)
    anchor_idx = To - 1

    H_ext = H + 2 * fit_pad
    eval_ext, d1_ext, _ = make_fit_matrices(H_ext, poly_order)

    bc_A_pos = bc_A_vel = bc_correction = None
    bc_anchor_idx = fit_pad + anchor_idx
    bc_overlap_idx = fit_pad + anchor_idx + Ta
    if constrained:
        bc_A_pos, bc_A_vel, bc_correction = build_c1_constraint(
            eval_ext, d1_ext, [bc_anchor_idx, bc_overlap_idx])

    eval_dense = d1_dense = n_dense_pts = None
    if dense_factor > 0:
        eval_dense, d1_dense, n_dense_pts = make_dense_eval_matrices(
            H, H_ext, fit_pad, poly_order, dense_factor)

    ep_starts = np.array([0])
    ep_ends = np.array([L])
    splines = fit_episode_splines(ep_action.astype(np.float64), ep_starts, ep_ends)
    spline = splines[0]
    if spline is None:
        z = np.zeros((0, Da))
        return np.full((L, Da), np.nan), np.full((L, Da), np.nan), np.array([]), None, z, z

    T_ext = (H_ext - 1) * ctrl_dt

    stitched_pos = np.full((L, Da), np.nan)
    stitched_vel = np.full((L, Da), np.nan)
    boundaries = []
    overlap_gaps_pos = []
    overlap_gaps_vel = []
    dense_segs = [] if dense_factor > 0 else None
    prev_overlap_pos = None
    prev_overlap_vel = None
    first_chunk = True

    pad_before = (To - 1) + fit_pad
    pad_after = (Ta - 1) + fit_pad
    padded = np.concatenate([
        np.tile(ep_action[0:1], (pad_before, 1)),
        ep_action,
        np.tile(ep_action[-1:], (pad_after, 1)),
    ], axis=0) if constrained else None

    t_cursor = 0
    while t_cursor < L:
        base_win_start = t_cursor - anchor_idx
        ext_win_start = base_win_start - fit_pad
        ext_win_end = ext_win_start + H_ext - 1

        t_lo = float(ext_win_start)
        t_hi = float(ext_win_end)

        coeff = _project_segment_to_legendre(
            spline, t_lo, t_hi, poly_order, 200,
            clip_lo=0.0, clip_hi=float(L - 1),
        )

        if bc_A_pos is not None:
            orig_win_start = t_cursor + pad_before - anchor_idx
            ext_ws = orig_win_start - fit_pad
            chunk_ext = padded[ext_ws: ext_ws + H_ext]
            bc_oi = bc_overlap_idx
            cur_pos = bc_A_pos @ coeff
            tgt_pos = chunk_ext[[bc_anchor_idx, bc_oi], :]
            cur_vel = bc_A_vel @ coeff
            vel_anchor = (chunk_ext[bc_anchor_idx + 1] - chunk_ext[bc_anchor_idx - 1]) * (H_ext - 1) / 2
            if bc_oi + 1 < H_ext:
                vel_overlap = (chunk_ext[bc_oi + 1] - chunk_ext[bc_oi - 1]) * (H_ext - 1) / 2
            elif bc_oi >= 2:
                vel_overlap = (3 * chunk_ext[bc_oi] - 4 * chunk_ext[bc_oi - 1] + chunk_ext[bc_oi - 2]) * (H_ext - 1) / 2
            else:
                vel_overlap = (chunk_ext[bc_oi] - chunk_ext[bc_oi - 1]) * (H_ext - 1)
            tgt_vel = np.stack([vel_anchor, vel_overlap])
            residual = np.vstack([cur_pos - tgt_pos, cur_vel - tgt_vel])
            coeff = coeff - bc_correction @ residual

        recon_ext = eval_ext @ coeff
        vel_ext = d1_ext @ coeff / T_ext

        mid_start = fit_pad
        recon_mid = recon_ext[mid_start:mid_start + H]
        vel_mid = vel_ext[mid_start:mid_start + H]

        anchor_pos_val = recon_mid[anchor_idx]
        anchor_vel_val = vel_mid[anchor_idx]
        if prev_overlap_pos is not None:
            overlap_gaps_pos.append(np.abs(anchor_pos_val - prev_overlap_pos))
            overlap_gaps_vel.append(np.abs(anchor_vel_val - prev_overlap_vel))
        overlap_idx_in_mid = anchor_idx + Ta
        if overlap_idx_in_mid < H:
            prev_overlap_pos = recon_mid[overlap_idx_in_mid].copy()
            prev_overlap_vel = vel_mid[overlap_idx_in_mid].copy()
        else:
            prev_overlap_pos = recon_ext[mid_start + overlap_idx_in_mid].copy()
            prev_overlap_vel = vel_ext[mid_start + overlap_idx_in_mid].copy()

        act_start = anchor_idx
        n_write = min(Ta, L - t_cursor)
        stitched_pos[t_cursor:t_cursor + n_write] = recon_mid[act_start:act_start + n_write]
        stitched_vel[t_cursor:t_cursor + n_write] = vel_mid[act_start:act_start + n_write]

        if dense_segs is not None:
            chunk_start = t_cursor - anchor_idx
            t_chunk = chunk_start + np.linspace(0, H - 1, n_dense_pts)
            pos_d = eval_dense @ coeff
            vel_d = d1_dense @ coeff / T_ext
            ws = 0 if first_chunk else anchor_idx
            dws = ws * dense_factor
            dwe = n_dense_pts
            dense_segs.append((t_chunk[dws:dwe], pos_d[dws:dwe], vel_d[dws:dwe]))

        if t_cursor > 0:
            boundaries.append(t_cursor)
        first_chunk = False
        t_cursor += Ta

    ogp = np.array(overlap_gaps_pos) if overlap_gaps_pos else np.zeros((0, Da))
    ogv = np.array(overlap_gaps_vel) if overlap_gaps_vel else np.zeros((0, Da))
    return stitched_pos, stitched_vel, np.array(boundaries), dense_segs, ogp, ogv


# ── Plotting helpers ─────────────────────────────────────────────────────

def _load_eval_ylims(npz_dir: str, n_joints: int):
    result = {}
    for f in glob.glob(f"{npz_dir}/diag_data_demo*.npz"):
        m = re.search(r"demo(\d+)", f)
        if not m:
            continue
        idx = int(m.group(1))
        d = np.load(f)
        pos_des, pos_act = d["pos_desired"], d["pos_actual"]
        vel_des = d["vel_desired"]
        vel_act = d["vel_actual"]
        pos_err = pos_des - pos_act
        margin = 0.05
        plims, vlims, elims = [], [], []
        for j in range(n_joints):
            plo = min(pos_des[:, j].min(), pos_act[:, j].min())
            phi = max(pos_des[:, j].max(), pos_act[:, j].max())
            pad = (phi - plo) * margin
            plims.append((plo - pad, phi + pad))
            vlo = min(vel_des[:, j].min(), vel_act[:, j].min())
            vhi = max(vel_des[:, j].max(), vel_act[:, j].max())
            pad = (vhi - vlo) * margin
            vlims.append((vlo - pad, vhi + pad))
            elo, ehi = pos_err[:, j].min(), pos_err[:, j].max()
            pad = (ehi - elo) * margin
            elims.append((elo - pad, ehi + pad))
        result[idx] = {"pos": plims, "vel": vlims, "err": elims}
    print(f"[INFO] Loaded eval y-limits for {len(result)} demos")
    return result


_log_lines = []


def _boundary_stats(stitched, boundaries, n_j):
    jumps = []
    for b in boundaries:
        if 0 < b < len(stitched):
            jumps.append(np.abs(stitched[b, :n_j] - stitched[b - 1, :n_j]))
    if jumps:
        jumps = np.array(jumps)
        return jumps.max(axis=0), jumps.mean(axis=0)
    return np.zeros(n_j), np.zeros(n_j)


# ── Comparison plot (original vs extended, same figure) ──────────────────

def plot_compare(ep_idx, L, ctrl_dt, expert, expert_vel,
                 pos_orig, vel_orig, bnd_orig,
                 pos_ext, vel_ext, bnd_ext,
                 n_j, poly_order, ext_poly_order, H, To, Ta, fit_pad, out_dir,
                 method_label="Extended", file_tag="compare",
                 dense_orig=None, dense_ext=None):

    fig, axes = plt.subplots(n_j, 3, figsize=(7 * 3, 2.6 * n_j),
                             sharex=True, squeeze=False)
    t = np.arange(L) * ctrl_dt
    c_exp = "#1f77b4"
    c_orig = "#ff7f0e"
    c_ext = "#2ca02c"
    c_bnd = "#cccccc"

    err_orig = pos_orig[:L] - expert[:L]
    err_ext = pos_ext[:L] - expert[:L]

    if method_label == "Extended":
        ext_pos_label = f"Extended (order {ext_poly_order}, fit {H+2*fit_pad})"
    else:
        ext_pos_label = f"{method_label} (order {ext_poly_order})"

    for j in range(n_j):
        # ── Position (dense segments if available, else sparse) ──
        ax = axes[j, 0]
        ax.plot(t, expert[:L, j], lw=1.2, color=c_exp, label="Expert")
        if dense_orig:
            for k, (st, sp, _) in enumerate(dense_orig):
                ax.plot(st * ctrl_dt, sp[:, j], lw=1.0, color=c_orig, alpha=0.8,
                        label=f"Original (order {poly_order}, H={H})" if k == 0 else None)
        else:
            ax.plot(t, pos_orig[:L, j], lw=1.0, color=c_orig, alpha=0.8,
                    label=f"Original (order {poly_order}, H={H})")
        if dense_ext:
            for k, (st, sp, _) in enumerate(dense_ext):
                ax.plot(st * ctrl_dt, sp[:, j], lw=1.0, color=c_ext, alpha=0.8,
                        ls="--", label=ext_pos_label if k == 0 else None)
        else:
            ax.plot(t, pos_ext[:L, j], lw=1.0, color=c_ext, alpha=0.8,
                    ls="--", label=ext_pos_label)
        for b in bnd_orig:
            ax.axvline(b * ctrl_dt, color=c_bnd, lw=0.5, ls="--", alpha=0.4)
        ax.set_ylabel(f"J{j+1}", fontsize=8)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_title("Position (rad)", fontsize=10)
            ax.legend(fontsize=6, loc="upper right")

        # ── Velocity (dense segments if available, else sparse) ──
        ax = axes[j, 1]
        ax.plot(t, expert_vel[:L, j], lw=1.0, color=c_exp, alpha=0.6,
                label="Expert (finite diff)")
        if dense_orig:
            for k, (st, _, sv) in enumerate(dense_orig):
                ax.plot(st * ctrl_dt, sv[:, j], lw=1.0, color=c_orig, alpha=0.8,
                        label="Original dq/dt" if k == 0 else None)
        else:
            ax.plot(t, vel_orig[:L, j], lw=1.0, color=c_orig, alpha=0.8,
                    label="Original dq/dt")
        if dense_ext:
            for k, (st, _, sv) in enumerate(dense_ext):
                ax.plot(st * ctrl_dt, sv[:, j], lw=1.0, color=c_ext, alpha=0.8,
                        ls="--", label=f"{method_label} dq/dt" if k == 0 else None)
        else:
            ax.plot(t, vel_ext[:L, j], lw=1.0, color=c_ext, alpha=0.8,
                    ls="--", label=f"{method_label} dq/dt")
        for b in bnd_orig:
            ax.axvline(b * ctrl_dt, color=c_bnd, lw=0.5, ls="--", alpha=0.4)
        ax.set_ylabel(f"J{j+1}", fontsize=8)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_title("Velocity (rad/s)", fontsize=10)
            ax.legend(fontsize=6, loc="upper right")

        # ── Position Error (always sparse — only at integer timesteps) ──
        ax = axes[j, 2]
        ax.plot(t, err_orig[:L, j], lw=0.9, color=c_orig, alpha=0.8,
                label="Original err")
        ax.plot(t, err_ext[:L, j], lw=0.9, color=c_ext, alpha=0.8,
                ls="--", label=f"{method_label} err")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        for b in bnd_orig:
            ax.axvline(b * ctrl_dt, color=c_bnd, lw=0.5, ls="--", alpha=0.4)
        ax.set_ylabel(f"J{j+1}", fontsize=8)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_title("Position Error (recon \u2212 expert)", fontsize=10)
            ax.legend(fontsize=6, loc="upper right")

    for c in range(3):
        axes[-1, c].set_xlabel("Time (s)", fontsize=9)

    max_orig, mean_orig = _boundary_stats(pos_orig, bnd_orig, n_j)
    max_ext, mean_ext = _boundary_stats(pos_ext, bnd_ext, n_j)

    if method_label == "Extended":
        vs_desc = f"Extended (order {ext_poly_order}, fit {H+2*fit_pad}, keep middle {H})"
    else:
        vs_desc = f"{method_label} (order {ext_poly_order})"

    fig.suptitle(
        f"Polynomial Fit Comparison \u2014 Demo {ep_idx}\n"
        f"Original (order {poly_order}, H={H}) vs "
        f"{vs_desc}  |  "
        f"To={To}, Ta={Ta}\n"
        f"Max boundary jump:  Original={max_orig.max():.5f} rad   "
        f"{method_label}={max_ext.max():.5f} rad",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    path = out_dir / f"{file_tag}_ep{ep_idx}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  \u2192 {path}")
    _log_lines.append(f"  \u2192 {path}")


def plot_compare_zoom(ep_idx, L, ctrl_dt, expert, expert_vel,
                      pos_orig, vel_orig, bnd_orig,
                      pos_ext, vel_ext, n_j, fit_pad, H, out_dir,
                      method_label="Extended", file_tag="compare"):
    if len(bnd_orig) < 2:
        return
    zoom_b = bnd_orig[1]
    half = 6
    lo = max(0, zoom_b - half)
    hi = min(L, zoom_b + half)
    sl = slice(lo, hi)
    t_z = np.arange(lo, hi) * ctrl_dt
    c_exp, c_orig, c_ext, c_bnd = "#1f77b4", "#ff7f0e", "#2ca02c", "#cccccc"

    fig, axes = plt.subplots(n_j, 2, figsize=(14, 2.5 * n_j), squeeze=False)
    for j in range(n_j):
        ax = axes[j, 0]
        ax.plot(t_z, expert[sl, j], "o-", ms=4, lw=1.2, color=c_exp, label="Expert")
        ax.plot(t_z, pos_orig[sl, j], "s-", ms=3, lw=1.0, color=c_orig, alpha=0.8,
                label="Original")
        ax.plot(t_z, pos_ext[sl, j], "^-", ms=3, lw=1.0, color=c_ext, alpha=0.8,
                label=method_label)
        ax.axvline(zoom_b * ctrl_dt, color=c_bnd, lw=1, ls="--", alpha=0.6)
        ax.set_ylabel(f"J{j+1}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_title(f"Position ZOOM — boundary t={zoom_b*ctrl_dt:.2f}s", fontsize=10)
            ax.legend(fontsize=7)

        ax = axes[j, 1]
        ax.plot(t_z, expert_vel[sl, j], "o-", ms=4, lw=1.0, color=c_exp, alpha=0.6,
                label="Expert vel")
        ax.plot(t_z, vel_orig[sl, j], "s-", ms=3, lw=1.0, color=c_orig, alpha=0.8,
                label="Original dq/dt")
        ax.plot(t_z, vel_ext[sl, j], "^-", ms=3, lw=1.0, color=c_ext, alpha=0.8,
                label=f"{method_label} dq/dt")
        ax.axvline(zoom_b * ctrl_dt, color=c_bnd, lw=1, ls="--", alpha=0.6)
        ax.set_ylabel(f"J{j+1}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_title(f"Velocity ZOOM — boundary t={zoom_b*ctrl_dt:.2f}s", fontsize=10)
            ax.legend(fontsize=7)

    for c in range(2):
        axes[-1, c].set_xlabel("Time (s)", fontsize=9)

    if method_label == "Extended":
        vs_desc = f"Extended (fit {H+2*fit_pad})"
    else:
        vs_desc = method_label

    fig.suptitle(
        f"Boundary Zoom Comparison \u2014 Demo {ep_idx}\n"
        f"Original (H={H}) vs {vs_desc}",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    path = out_dir / f"{file_tag}_ep{ep_idx}_zoom.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  \u2192 {path}")
    _log_lines.append(f"  \u2192 {path}")


# ── Single-method plot (unchanged from before) ──────────────────────────

def plot_single(ep_idx, L, ctrl_dt, expert, expert_vel,
                recon_pos, recon_vel, boundaries,
                n_j, poly_order, H, To, Ta, fit_pad,
                pos_ylims, vel_ylims, err_ylims, out_dir):

    fit_err = recon_pos[:L] - expert[:L]
    fig, axes = plt.subplots(n_j, 3, figsize=(7 * 3, 2.2 * n_j),
                             sharex=True, squeeze=False)
    t = np.arange(L) * ctrl_dt
    c_des, c_act, c_err, c_bnd = "#1f77b4", "#ff7f0e", "#d62728", "#bbbbbb"
    tag = f"pad{fit_pad}" if fit_pad else "orig"

    for j in range(n_j):
        ax = axes[j, 0]
        ax.plot(t, expert[:L, j], lw=1.1, color=c_des, label="Expert trajectory")
        ax.plot(t, recon_pos[:L, j], lw=1.1, color=c_act, alpha=0.85,
                label="Poly recon (stitched)")
        for b in boundaries:
            ax.axvline(b * ctrl_dt, color=c_bnd, lw=0.6, ls="--", alpha=0.5)
        ax.set_ylabel(f"J{j+1}", fontsize=8); ax.grid(True, alpha=0.3)
        if pos_ylims: ax.set_ylim(pos_ylims[j])
        if j == 0:
            ax.set_title("Position (rad)", fontsize=10)
            ax.legend(fontsize=6, loc="upper right")

        ax = axes[j, 1]
        ax.plot(t, expert_vel[:L, j], lw=1.1, color=c_des, label="Expert (finite diff)")
        ax.plot(t, recon_vel[:L, j], lw=1.1, color=c_act, alpha=0.85,
                label="Poly dq/dt (stitched)")
        for b in boundaries:
            ax.axvline(b * ctrl_dt, color=c_bnd, lw=0.6, ls="--", alpha=0.5)
        ax.set_ylabel(f"J{j+1}", fontsize=8); ax.grid(True, alpha=0.3)
        if vel_ylims: ax.set_ylim(vel_ylims[j])
        if j == 0:
            ax.set_title("Velocity (rad/s)", fontsize=10)
            ax.legend(fontsize=6, loc="upper right")

        ax = axes[j, 2]
        ax.plot(t, fit_err[:L, j], lw=1.0, color=c_err, alpha=0.85,
                label="pos err (recon − expert)")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        for b in boundaries:
            ax.axvline(b * ctrl_dt, color=c_bnd, lw=0.6, ls="--", alpha=0.5)
        ax.set_ylabel(f"J{j+1}", fontsize=8); ax.grid(True, alpha=0.3)
        if err_ylims: ax.set_ylim(err_ylims[j])
        if j == 0:
            ax.set_title("Position Error (recon \u2212 expert)", fontsize=10)
            ax.legend(fontsize=6, loc="upper right")

    for c in range(3):
        axes[-1, c].set_xlabel("Time (s)", fontsize=9)

    H_fit = H + 2 * fit_pad if fit_pad else H
    fig.suptitle(
        f"Expert Poly Fit \u2014 Demo {ep_idx}  [{tag}]\n"
        f"(Legendre order={poly_order}, fit_window={H_fit}, keep={H}, "
        f"To={To}, Ta={Ta})",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    path = out_dir / f"expert_poly_fit_ep{ep_idx}_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  \u2192 {path}")
    _log_lines.append(f"  \u2192 {path}")


# ── Main ─────────────────────────────────────────────────────────────────

def _natural_step_stats(expert, boundaries, n_j):
    """Compute the expert's natural step |expert[b] - expert[b-1]| at boundaries."""
    jumps = []
    for b in boundaries:
        if 0 < b < len(expert):
            jumps.append(np.abs(expert[b, :n_j] - expert[b - 1, :n_j]))
    if jumps:
        jumps = np.array(jumps)
        return jumps.max(axis=0), jumps.mean(axis=0)
    return np.zeros(n_j), np.zeros(n_j)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr", required=True)
    p.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--output_dir", default="il_outputs/expert_poly_diag")
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--n_obs_steps", type=int, default=8)
    p.add_argument("--n_action_steps", type=int, default=8)
    p.add_argument("--poly_order", type=int, default=7)
    p.add_argument("--n_joints", type=int, default=7)
    p.add_argument("--downsample_ratio", type=int, default=4)
    p.add_argument("--base_control_dt", type=float, default=0.015)
    p.add_argument("--fit_pad", type=int, default=0,
                   help="Extra steps to add on each side when fitting "
                        "(0 = original, 8 = fit 32 steps keep middle 16)")
    p.add_argument("--compare", action="store_true",
                   help="Generate side-by-side comparison of original vs "
                        "extended (fit_pad=8) on the same figure")
    p.add_argument("--ext_poly_order", type=int, default=None,
                   help="Polynomial order for the extended window only "
                        "(default: same as --poly_order)")
    p.add_argument("--eval_npz_dir", type=str, default=None)
    p.add_argument("--spline", action="store_true",
                   help="Generate side-by-side comparison of original vs "
                        "global-spline-projected Legendre fit")
    p.add_argument("--constrained", action="store_true",
                   help="Generate side-by-side comparison of original vs "
                        "boundary-constrained (C0 pinning at anchor & last action)")
    p.add_argument("--dense", type=int, default=10,
                   help="Dense oversampling factor for smooth curve plotting "
                        "(0 = sparse integer-step only, 10 = default)")
    p.add_argument("--poly_prefix", type=int, default=None,
                   help="Reduced polynomial: covers n_action_steps + poly_prefix + poly_suffix. "
                        "1 = prefix of 1 extra obs point. None = full horizon.")
    p.add_argument("--poly_suffix", type=int, default=0,
                   help="Extra points AFTER action steps (only when poly_prefix is set). "
                        "1 = cover next window's anchor for C1 continuity.")
    args = p.parse_args()

    import datetime

    H = args.horizon
    To = args.n_obs_steps
    Ta = args.n_action_steps
    ctrl_dt = args.base_control_dt * args.downsample_ratio
    n_j = args.n_joints
    df = args.dense

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path(args.output_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    _log_lines.clear()

    def log(msg):
        print(msg)
        _log_lines.append(msg)

    z = zarr.open(args.zarr, "r")
    actions = z["data"]["action"][:]
    ep_ends = z["meta"]["episode_ends"][:]
    ep_starts = np.concatenate([[0], ep_ends[:-1]])

    eval_ylims = None
    if args.eval_npz_dir:
        eval_ylims = _load_eval_ylims(args.eval_npz_dir, n_j)

    log(f"[INFO] {len(ep_ends)} episodes, Da={actions.shape[1]}, "
        f"H={H}, To={To}, Ta={Ta}, poly={args.poly_order}, "
        f"ctrl_dt={ctrl_dt:.4f}s, fit_pad={args.fit_pad}, dense={df}")

    for ep_idx in args.episodes:
        if ep_idx >= len(ep_ends):
            log(f"[WARN] Episode {ep_idx} out of range, skip."); continue
        s, e = int(ep_starts[ep_idx]), int(ep_ends[ep_idx])
        ep_action = actions[s:e]
        L = len(ep_action)
        expert_vel = np.gradient(ep_action, ctrl_dt, axis=0)
        log(f"\n── Episode {ep_idx}  ({L} steps, {L*ctrl_dt:.2f}s) ──")

        if args.constrained:
            # ── Side-by-side: original vs boundary-constrained ──
            # When --spline is also set, the constrained method uses
            # spline-projected coefficients as base (before C1 correction).
            use_spline = args.spline
            if use_spline:
                method_tag = "Constr+Spline"
                file_tag = "constr_spline_compare"
            else:
                method_tag = "Training GT" if args.fit_pad > 0 else "Constrained"
                file_tag = "constrained_compare"

            pos_o, vel_o, bnd_o, dense_o, ogp_o, ogv_o = stitch_episode(
                ep_action, H, To, Ta, args.poly_order, ctrl_dt,
                fit_pad=args.fit_pad, constrained=False, dense_factor=df,
                poly_prefix=args.poly_prefix, poly_suffix=args.poly_suffix)
            if use_spline:
                pos_c, vel_c, bnd_c, dense_c, ogp_c, ogv_c = stitch_episode_spline(
                    ep_action, H, To, Ta, args.poly_order, ctrl_dt,
                    fit_pad=args.fit_pad, constrained=True, dense_factor=df)
            else:
                pos_c, vel_c, bnd_c, dense_c, ogp_c, ogv_c = stitch_episode(
                    ep_action, H, To, Ta, args.poly_order, ctrl_dt,
                    fit_pad=args.fit_pad, constrained=True, dense_factor=df,
                    poly_prefix=args.poly_prefix, poly_suffix=args.poly_suffix)

            max_o, mean_o = _boundary_stats(pos_o, bnd_o, n_j)
            max_c, mean_c = _boundary_stats(pos_c, bnd_c, n_j)
            nat_max, nat_mean = _natural_step_stats(ep_action, bnd_o, n_j)
            log(f"  [Natural]       max step = {nat_max.max():.5f} rad, "
                f"mean = {nat_mean.mean():.5f} rad")
            log(f"  [Original]      max boundary jump = {max_o.max():.5f} rad, "
                f"mean = {mean_o.mean():.5f} rad")
            log(f"  [{method_tag}]   max boundary jump = {max_c.max():.5f} rad, "
                f"mean = {mean_c.mean():.5f} rad")

            _, vmean_o = _boundary_stats(vel_o, bnd_o, n_j)
            _, vmean_c = _boundary_stats(vel_c, bnd_c, n_j)
            log(f"  [Original]      mean vel jump = {vmean_o.mean():.3f} rad/s")
            log(f"  [{method_tag}]   mean vel jump = {vmean_c.mean():.3f} rad/s")

            if len(ogp_o) > 0:
                log(f"  [Original]      overlap pos gap: max={ogp_o[:,:n_j].max():.5f}, "
                    f"mean={ogp_o[:,:n_j].mean():.5f} rad")
            if len(ogp_c) > 0:
                log(f"  [{method_tag}]   overlap pos gap: max={ogp_c[:,:n_j].max():.5f}, "
                    f"mean={ogp_c[:,:n_j].mean():.5f} rad")
            if len(ogv_o) > 0:
                log(f"  [Original]      overlap vel gap: max={ogv_o[:,:n_j].max():.3f}, "
                    f"mean={ogv_o[:,:n_j].mean():.3f} rad/s")
            if len(ogv_c) > 0:
                log(f"  [{method_tag}]   overlap vel gap: max={ogv_c[:,:n_j].max():.3f}, "
                    f"mean={ogv_c[:,:n_j].mean():.3f} rad/s")

            plot_compare(ep_idx, L, ctrl_dt, ep_action, expert_vel,
                         pos_o, vel_o, bnd_o, pos_c, vel_c, bnd_c,
                         n_j, args.poly_order, args.poly_order, H, To, Ta,
                         args.fit_pad, out_dir,
                         method_label=method_tag, file_tag=file_tag,
                         dense_orig=dense_o, dense_ext=dense_c)
            plot_compare_zoom(ep_idx, L, ctrl_dt, ep_action, expert_vel,
                              pos_o, vel_o, bnd_o, pos_c, vel_c,
                              n_j, args.fit_pad, H, out_dir,
                              method_label=method_tag, file_tag=file_tag)

        elif args.spline:
            # ── Side-by-side: original vs spline-projected ──
            pos_o, vel_o, bnd_o, dense_o, _, _ = stitch_episode(
                ep_action, H, To, Ta, args.poly_order, ctrl_dt,
                fit_pad=args.fit_pad, dense_factor=df,
                poly_prefix=args.poly_prefix, poly_suffix=args.poly_suffix)
            pos_sp, vel_sp, bnd_sp, dense_sp, _, _ = stitch_episode_spline(
                ep_action, H, To, Ta, args.poly_order, ctrl_dt,
                fit_pad=args.fit_pad, dense_factor=df)

            max_o, mean_o = _boundary_stats(pos_o, bnd_o, n_j)
            max_sp, mean_sp = _boundary_stats(pos_sp, bnd_sp, n_j)
            nat_max, nat_mean = _natural_step_stats(ep_action, bnd_o, n_j)
            log(f"  [Natural]   max step = {nat_max.max():.5f} rad, "
                f"mean = {nat_mean.mean():.5f} rad")
            log(f"  [Original]  max boundary jump = {max_o.max():.5f} rad, "
                f"mean = {mean_o.mean():.5f} rad")
            log(f"  [Spline]    max boundary jump = {max_sp.max():.5f} rad, "
                f"mean = {mean_sp.mean():.5f} rad")

            _, vmean_o = _boundary_stats(vel_o, bnd_o, n_j)
            _, vmean_sp = _boundary_stats(vel_sp, bnd_sp, n_j)
            log(f"  [Original]  mean vel jump = {vmean_o.mean():.3f} rad/s")
            log(f"  [Spline]    mean vel jump = {vmean_sp.mean():.3f} rad/s")

            plot_compare(ep_idx, L, ctrl_dt, ep_action, expert_vel,
                         pos_o, vel_o, bnd_o, pos_sp, vel_sp, bnd_sp,
                         n_j, args.poly_order, args.poly_order, H, To, Ta,
                         args.fit_pad, out_dir,
                         method_label="Spline", file_tag="spline_compare",
                         dense_orig=dense_o, dense_ext=dense_sp)
            plot_compare_zoom(ep_idx, L, ctrl_dt, ep_action, expert_vel,
                              pos_o, vel_o, bnd_o, pos_sp, vel_sp,
                              n_j, args.fit_pad, H, out_dir,
                              method_label="Spline", file_tag="spline_compare")

        elif args.compare:
            # ── Side-by-side: original vs extended ──
            fit_pad = args.fit_pad if args.fit_pad > 0 else 8
            ext_order = args.ext_poly_order if args.ext_poly_order else args.poly_order
            pos_o, vel_o, bnd_o, dense_o, _, _ = stitch_episode(
                ep_action, H, To, Ta, args.poly_order, ctrl_dt,
                fit_pad=0, dense_factor=df,
                poly_prefix=args.poly_prefix, poly_suffix=args.poly_suffix)
            pos_e, vel_e, bnd_e, dense_e, _, _ = stitch_episode(
                ep_action, H, To, Ta, ext_order, ctrl_dt,
                fit_pad=fit_pad, dense_factor=df,
                poly_prefix=args.poly_prefix, poly_suffix=args.poly_suffix)

            max_o, mean_o = _boundary_stats(pos_o, bnd_o, n_j)
            max_e, mean_e = _boundary_stats(pos_e, bnd_e, n_j)
            nat_max, nat_mean = _natural_step_stats(ep_action, bnd_o, n_j)
            log(f"  [Natural]   max step = {nat_max.max():.5f} rad, "
                f"mean = {nat_mean.mean():.5f} rad")
            log(f"  [Original]  max boundary jump = {max_o.max():.5f} rad, "
                f"mean = {mean_o.mean():.5f} rad")
            log(f"  [Extended]  max boundary jump = {max_e.max():.5f} rad, "
                f"mean = {mean_e.mean():.5f} rad")

            _, vmean_o = _boundary_stats(vel_o, bnd_o, n_j)
            _, vmean_e = _boundary_stats(vel_e, bnd_e, n_j)
            log(f"  [Original]  mean vel jump = {vmean_o.mean():.3f} rad/s")
            log(f"  [Extended]  mean vel jump = {vmean_e.mean():.3f} rad/s")

            plot_compare(ep_idx, L, ctrl_dt, ep_action, expert_vel,
                         pos_o, vel_o, bnd_o, pos_e, vel_e, bnd_e,
                         n_j, args.poly_order, ext_order, H, To, Ta,
                         fit_pad, out_dir,
                         dense_orig=dense_o, dense_ext=dense_e)
            plot_compare_zoom(ep_idx, L, ctrl_dt, ep_action, expert_vel,
                              pos_o, vel_o, bnd_o, pos_e, vel_e,
                              n_j, fit_pad, H, out_dir)
        else:
            # ── Single method ──
            pos, vel, bnd, _, _, _ = stitch_episode(
                ep_action, H, To, Ta, args.poly_order, ctrl_dt,
                fit_pad=args.fit_pad, poly_prefix=args.poly_prefix,
                poly_suffix=args.poly_suffix)

            max_j, mean_j = _boundary_stats(pos, bnd, n_j)
            log(f"  max boundary jump = {max_j.max():.5f} rad, "
                f"mean = {mean_j.mean():.5f} rad")

            pyl = vyl = eyl = None
            if eval_ylims and ep_idx in eval_ylims:
                pyl = eval_ylims[ep_idx]["pos"]
                vyl = eval_ylims[ep_idx]["vel"]
                eyl = eval_ylims[ep_idx]["err"]

            plot_single(ep_idx, L, ctrl_dt, ep_action, expert_vel,
                        pos, vel, bnd, n_j, args.poly_order, H, To, Ta,
                        args.fit_pad, pyl, vyl, eyl, out_dir)

    log_path = out_dir / "stats.txt"
    log_path.write_text("\n".join(_log_lines) + "\n", encoding="utf-8")
    log(f"\n[DONE] All plots + stats \u2192 {out_dir}/")


if __name__ == "__main__":
    main()
