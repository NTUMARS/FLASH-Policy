import copy
import datetime
import math
import os
import pathlib
import random
import time
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import dill
import hydra
import imageio.v2 as iio
import numpy as np
import torch
import tqdm
import wandb
from loguru import logger as log

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot
from metasim.task.registry import get_task_class
from metasim.randomization import DomainRandomizationManager, DRConfig
from roboverse_learn.il.utils.ema_model import EMAModel
from roboverse_learn.il.runners.base_runner import BaseRunner
from roboverse_learn.il.utils.json_logger import JsonLogger
from roboverse_learn.il.utils.lr_scheduler import get_scheduler
from roboverse_learn.il.utils.pytorch_util import optimizer_to
from roboverse_learn.il.utils.visualization import plot_all_latent_visualizations

RANDOMIZATION_AVAILABLE = True

# Velocity-mode damping (Kd) for Franka joints.
# Derived from FLASH-G diagnostic data: Kd is set so that even at peak
# vel_target the velocity-driven torque stays comparable to the
# position-driven torque, avoiding the destabilising spikes caused by
# the original Kd (up to 1e4) when vel_target != 0.
#
# These are *base* values; the eval-arg ``velocity_kd_scale`` is applied
# on top as a uniform multiplier for easy tuning.
#
# Original Kd ➜ New Kd  |  Reasoning
# ──────────────────────────────────────────────────────────────────
# J1  10 000  ➜  200    |  vel p95=0.25 → torque 50  (was 2 500)
# J2   1 000  ➜   20    |  vel p95=0.19 → torque 3.8 (was 190)
# J3   5 000  ➜  100    |  vel p95=0.70 → torque 70  (was 3 500)
# J4  10 000  ➜  200    |  vel p95=0.81 → torque 162 (was 8 140)
# J5      50  ➜   10    |  vel p95=0.92 → torque 9.2 (was 46)
# J6      50  ➜   10    |  vel p95=0.69 → torque 6.9 (was 35)
# J7      50  ➜   20    |  vel p95=0.59 → torque 12  (was 30)
VELOCITY_MODE_KP: dict[str, float] = {
    "panda_joint1": 100000,
    "panda_joint2": 10000,
    "panda_joint3": 100000,
    "panda_joint4": 100000,
    "panda_joint5": 400,
    "panda_joint6": 250,
    "panda_joint7": 800,
}

VELOCITY_MODE_KD: dict[str, float] = {
    "panda_joint1": 10000,
    "panda_joint2": 1000,
    "panda_joint3": 5000,
    "panda_joint4": 10000,
    "panda_joint5": 50,
    "panda_joint6": 50,
    "panda_joint7": 50,
}

# Per-joint position-error y-axis limits for diagnostic plots.
# Computed from 150 demos across 3 baseline models (a2a, fm_dit, fm_unet)
# on the close_box task, with 10 % padding on each side.
from roboverse_learn.il.utils.visualization import (
    BASELINE_ERROR_YLIM,
    make_combined_diag_fig,
)

def _get_or_compute_task_ylim(dataset, output_dir, horizon=16):
    """Get per-joint y-axis limits for poly-fit diagnostic error plots.

    On first call for a task, computes worst-case fitting errors by fitting
    expert trajectories with poly_order=3 (lowest reasonable order → largest
    errors) across ALL episodes.  Result is saved to the task-level directory
    as ``poly_diag_ylim.json`` and reused by subsequent runs of the same task.
    """
    import json
    task_dir = pathlib.Path(output_dir).parent
    ylim_path = task_dir / "poly_diag_ylim.json"

    if ylim_path.exists():
        with open(ylim_path) as f:
            data = json.load(f)
        log.info(f"[poly diag] loaded ylim from {ylim_path}")
        return data["pos_err_ylim"], data["vel_err_ylim"]

    log.info("[poly diag] computing per-joint ylim (first run for this task) ...")

    replay = dataset.sampler.replay_buffer
    ep_ends = replay.episode_ends[:]
    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    actions = replay["action"]
    Da = actions.shape[1]
    H = horizon

    baseline_order = 3
    from roboverse_learn.il.policies.flash.flash_g_policy import FlashGDiTImagePolicy
    build_basis = FlashGDiTImagePolicy._build_legendre_basis
    s = torch.linspace(0, 1, H, dtype=torch.float64)
    ev, d1, _ = build_basis(s, baseline_order)
    lstsq = torch.linalg.solve((ev.T @ ev), ev.T)

    n_obs = max(H // 2, 1)
    n_act = H - n_obs
    step_size = n_act

    pos_absmax = np.zeros(Da)
    vel_absmax = np.zeros(Da)

    for ep_idx in range(len(ep_ends)):
        si, ei = int(ep_starts[ep_idx]), int(ep_ends[ep_idx])
        ep_action = actions[si:ei]
        L = len(ep_action)
        expert_vel = np.gradient(ep_action, axis=0)

        for cs in range(0, L - n_obs, step_size):
            ce = min(cs + H, L)
            if ce - cs < H:
                break
            chunk = torch.from_numpy(ep_action[cs:ce]).double()
            coeff = lstsq @ chunk
            recon = (ev @ coeff).numpy()
            vel_recon = (d1 @ coeff).numpy()
            expert_chunk = ep_action[cs:ce]
            vel_chunk = expert_vel[cs:ce]
            for j in range(Da):
                pos_absmax[j] = max(pos_absmax[j], np.max(np.abs(recon[:, j] - expert_chunk[:, j])))
                vel_absmax[j] = max(vel_absmax[j], np.max(np.abs(vel_recon[:, j] - vel_chunk[:, j])))

    pos_err_ylim = []
    vel_err_ylim = []
    for j in range(Da):
        pm = round(float(pos_absmax[j] * 1.15), 6)
        vm = round(float(vel_absmax[j] * 1.15), 4)
        pos_err_ylim.append((-pm, pm))
        vel_err_ylim.append((-vm, vm))

    task_dir.mkdir(parents=True, exist_ok=True)
    with open(ylim_path, "w") as f:
        json.dump({"pos_err_ylim": pos_err_ylim, "vel_err_ylim": vel_err_ylim}, f, indent=2)
    log.info(f"[poly diag] saved ylim to {ylim_path}")
    return pos_err_ylim, vel_err_ylim


def ensure_clean_state(handler, expected_state=None):
    """Ensure environment is in clean initial state with intelligent validation."""
    prev_state = None
    stable_count = 0
    max_steps = 10
    min_steps = 2

    for step in range(max_steps):
        handler.simulate()
        current_state = handler.get_states()

        if step >= min_steps:
            if prev_state is not None:
                is_stable = True
                if hasattr(current_state, "objects") and hasattr(prev_state, "objects"):
                    for obj_name, obj_state in current_state.objects.items():
                        if obj_name in prev_state.objects:
                            curr_dof = getattr(obj_state, "dof_pos", None)
                            prev_dof = getattr(prev_state.objects[obj_name], "dof_pos", None)
                            if curr_dof is not None and prev_dof is not None:
                                if not torch.allclose(curr_dof, prev_dof, atol=1e-5):
                                    is_stable = False
                                    break

                if is_stable and expected_state is not None:
                    is_correct_state = _validate_state_correctness(current_state, expected_state)
                    if not is_correct_state:
                        log.debug(f"State stable but incorrect at step {step}, continuing simulation...")
                        stable_count = 0
                        is_stable = False

                if is_stable:
                    stable_count += 1
                    if stable_count >= 2:
                        break
                else:
                    stable_count = 0

            prev_state = current_state

    if expected_state is not None:
        final_state = handler.get_states()
        is_final_correct = _validate_state_correctness(final_state, expected_state)
        if not is_final_correct:
            log.warning(f"State validation failed after {max_steps} steps - reset may not have taken full effect")

    handler.get_states()


def _validate_state_correctness(current_state, expected_state):
    """Validate that current state matches expected initial state for critical objects."""
    if not hasattr(current_state, "objects") or not hasattr(expected_state, "objects"):
        return True

    critical_objects = []
    for obj_name, expected_obj in expected_state.objects.items():
        if hasattr(expected_obj, "dof_pos") and getattr(expected_obj, "dof_pos", None) is not None:
            critical_objects.append(obj_name)

    if not critical_objects:
        return True

    tolerance = 5e-3

    for obj_name in critical_objects:
        if obj_name not in current_state.objects:
            continue

        expected_obj = expected_state.objects[obj_name]
        current_obj = current_state.objects[obj_name]

        expected_dof = getattr(expected_obj, "dof_pos", None)
        current_dof = getattr(current_obj, "dof_pos", None)

        if expected_dof is not None and current_dof is not None:
            if not torch.allclose(current_dof, expected_dof, atol=tolerance):
                diff = torch.abs(current_dof - expected_dof).max().item()
                log.debug(f"DOF mismatch for {obj_name}: max diff = {diff:.6f} (tolerance = {tolerance})")
                return False

    return True


def _generate_poly_diagnostic(model, dataset, output_dir, num_episodes=5,
                              pos_err_ylim=None, vel_err_ylim=None):
    """Generate a diagnostic plot comparing expert trajectory vs poly-fitted training data.

    When fit_pad > 0, also shows the extended-window fit for comparison.
    When spline_gt is enabled, also shows the spline-projected fit.
    Saves one PNG + one NPZ per episode to {output_dir}/expert_poly_diag/.
    Columns: Position | Velocity | Position Error | Velocity Error
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit_pad = getattr(model, 'fit_pad', 0)
    has_ext = fit_pad > 0
    has_spline = getattr(model, 'spline_gt', False)
    H_full = model.horizon
    poly_order = model.poly_order
    basis_type = model.basis_type

    poly_prefix_val = getattr(model, 'poly_prefix', None)
    if poly_prefix_val is not None:
        H = model._H_poly
        _poly_offset = model._poly_nf_start
        _anchor_idx = poly_prefix_val
    else:
        H = H_full
        _poly_offset = 0
        _anchor_idx = model.n_obs_steps - 1
    H_ext = H + 2 * fit_pad

    build_basis = (type(model)._build_legendre_basis if basis_type == "legendre"
                   else type(model)._build_power_basis)

    # Sparse basis (integer timesteps — for npz data / error computation)
    s_std = torch.linspace(0, 1, H, dtype=torch.float64)
    eval_std, d1_std, _ = build_basis(s_std, poly_order)
    StS_std = eval_std.T @ eval_std
    lstsq_std = torch.linalg.solve(StS_std, eval_std.T)

    # Dense basis (smooth curves for plotting)
    DENSE = 10
    n_dense = (H - 1) * DENSE + 1
    s_std_dense = torch.linspace(0, 1, n_dense, dtype=torch.float64)
    eval_std_d, d1_std_d, _ = build_basis(s_std_dense, poly_order)

    if has_ext:
        s_ext = torch.linspace(0, 1, H_ext, dtype=torch.float64)
        eval_ext_full, _, _ = build_basis(s_ext, poly_order)
        StS_ext = eval_ext_full.T @ eval_ext_full
        lstsq_ext = torch.linalg.solve(StS_ext, eval_ext_full.T)
        n_mid_eval = H + (1 if poly_prefix_val is not None else 0)
        s_mid = s_ext[fit_pad:fit_pad + n_mid_eval]
        eval_mid, d1_mid, _ = build_basis(s_mid, poly_order)
        n_dense_mid = (n_mid_eval - 1) * DENSE + 1
        s_mid_dense = torch.linspace(s_mid[0].item(), s_mid[-1].item(),
                                     n_dense_mid, dtype=torch.float64)
        eval_mid_d, d1_mid_d, _ = build_basis(s_mid_dense, poly_order)

    # Boundary constraint setup
    has_bc = getattr(model, 'boundary_constraint', False)
    bc_eval_pos = bc_eval_vel = bc_corr = bc_idx = None
    if has_bc:
        bc_eval_pos = model._bc_eval_pos.double().cpu()
        bc_eval_vel = model._bc_eval_vel.double().cpu()
        bc_corr = model._bc_correction.double().cpu()
        bc_idx = model._bc_pos_indices.cpu()

    # Spline projection setup
    ep_splines = None
    if has_spline:
        from roboverse_learn.il.utils.spline_projection import (
            fit_episode_splines, _project_segment_to_legendre,
        )

    replay = dataset.sampler.replay_buffer
    ep_ends = replay.episode_ends[:]
    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    actions = replay["action"]
    Da = actions.shape[1]
    n_j = Da

    if has_spline:
        ep_splines = fit_episode_splines(
            actions[:].astype(np.float64), ep_starts, ep_ends)

    out_dir = pathlib.Path(output_dir) / "expert_poly_diag"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx in range(min(num_episodes, len(ep_ends))):
        s, e = int(ep_starts[ep_idx]), int(ep_ends[ep_idx])
        ep_action = actions[s:e]
        L = len(ep_action)
        n_obs = model.n_obs_steps
        n_act = model.n_action_steps
        step_size = n_act

        # Sparse arrays (integer timesteps) for npz / error columns
        pos_std_all = np.full((L, Da), np.nan)
        vel_std_all = np.full((L, Da), np.nan)
        pos_ext_all = np.full((L, Da), np.nan) if has_ext else None
        vel_ext_all = np.full((L, Da), np.nan) if has_ext else None
        pos_spl_all = np.full((L, Da), np.nan) if has_spline else None
        vel_spl_all = np.full((L, Da), np.nan) if has_spline else None
        pos_gt_all = np.full((L, Da), np.nan) if has_bc else None
        vel_gt_all = np.full((L, Da), np.nan) if has_bc else None

        # Dense segments for smooth plotting: list of (t_arr, pos_arr, vel_arr)
        std_segs = []
        ext_segs = []
        spl_segs = []
        gt_segs = []

        boundaries = []
        first_chunk = True
        for chunk_start in range(0, L - n_obs, step_size):
            if chunk_start + _poly_offset + H > L:
                break
            poly_start = chunk_start + _poly_offset
            chunk = torch.from_numpy(ep_action[poly_start:poly_start + H]).double()

            coeff_s = lstsq_std @ chunk
            recon_s = (eval_std @ coeff_s).numpy()
            vel_s = (d1_std @ coeff_s).numpy()

            if has_ext:
                ext_start = poly_start - fit_pad
                ext_end = poly_start + H + fit_pad
                pb = max(0, -ext_start)
                pa = max(0, ext_end - L)
                actual_start = max(0, ext_start)
                actual_end = min(L, ext_end)
                chunk_ext_np = ep_action[actual_start:actual_end]
                if pb > 0:
                    chunk_ext_np = np.concatenate(
                        [np.tile(chunk_ext_np[0:1], (pb, 1)), chunk_ext_np], axis=0)
                if pa > 0:
                    chunk_ext_np = np.concatenate(
                        [chunk_ext_np, np.tile(chunk_ext_np[-1:], (pa, 1))], axis=0)
                chunk_ext = torch.from_numpy(chunk_ext_np[:H_ext]).double()
                coeff_e = lstsq_ext @ chunk_ext
                recon_e = (eval_mid @ coeff_e).numpy()
                vel_e = (d1_mid @ coeff_e).numpy()

            coeff_sp = None
            if has_spline and ep_splines[ep_idx] is not None:
                spline = ep_splines[ep_idx]
                t_lo = float(poly_start - fit_pad)
                t_hi = float(poly_start + H - 1 + fit_pad)
                if t_hi > t_lo:
                    coeff_sp_np = _project_segment_to_legendre(
                        spline, t_lo, t_hi, poly_order, 200,
                        clip_lo=0.0, clip_hi=float(L - 1))
                    coeff_sp = torch.from_numpy(coeff_sp_np).double()

            # Compute GT (boundary-constrained + re-projected) coefficients
            # This is the ACTUAL training target for flow matching:
            # 1. Fit on extended data → 2. BC correction → 3. Re-project to core
            coeff_gt = None
            if has_bc:
                base = coeff_e if has_ext else coeff_s
                data = chunk_ext if has_ext else chunk
                H_local = data.shape[0]
                cur_p = bc_eval_pos @ base
                tgt_p = data[bc_idx, :]
                cur_v = bc_eval_vel @ base
                tgt_v_parts = []
                for ki in range(len(bc_idx)):
                    ii = bc_idx[ki].item()
                    if ii + 1 < H_local:
                        tgt_v_parts.append((data[ii + 1, :] - data[ii - 1, :]) * (H_local - 1) / 2)
                    elif ii >= 2:
                        tgt_v_parts.append((3 * data[ii, :] - 4 * data[ii - 1, :] + data[ii - 2, :]) * (H_local - 1) / 2)
                    else:
                        tgt_v_parts.append((data[ii, :] - data[ii - 1, :]) * (H_local - 1))
                tgt_v = torch.stack(tgt_v_parts)
                residual = torch.cat([cur_p - tgt_p, cur_v - tgt_v], dim=0)
                coeff_gt = base - bc_corr @ residual

                if has_ext:
                    traj_core = eval_mid[:H, :] @ coeff_gt
                    coeff_gt = lstsq_std @ traj_core

            if first_chunk:
                ws = 0
                first_chunk = False
            else:
                ws = _anchor_idx
            write_end = min(ws + (H - ws), H)
            dst_start = poly_start + ws
            dst_end = poly_start + write_end
            n_write = dst_end - dst_start

            # Sparse (integer timesteps)
            pos_std_all[dst_start:dst_end] = recon_s[ws:ws + n_write]
            vel_std_all[dst_start:dst_end] = vel_s[ws:ws + n_write]
            if has_ext:
                pos_ext_all[dst_start:dst_end] = recon_e[ws:ws + n_write]
                vel_ext_all[dst_start:dst_end] = vel_e[ws:ws + n_write]
            if coeff_sp is not None:
                recon_sp = (eval_std @ coeff_sp).numpy()
                vel_sp = (d1_std @ coeff_sp).numpy()
                pos_spl_all[dst_start:dst_end] = recon_sp[ws:ws + n_write]
                vel_spl_all[dst_start:dst_end] = vel_sp[ws:ws + n_write]
            if coeff_gt is not None:
                recon_gt = (eval_std @ coeff_gt).numpy()
                vel_gt = (d1_std @ coeff_gt).numpy()
                pos_gt_all[dst_start:dst_end] = recon_gt[ws:ws + n_write]
                vel_gt_all[dst_start:dst_end] = vel_gt[ws:ws + n_write]

            # Dense (smooth curves) — clipped so consecutive segments exactly
            # touch with no gap and no overlap (avoids visual "duplication").
            # Core curves (Std/GT/Spline) span H pts; Ext spans n_mid_eval pts.
            dc = max(0, H - 1 - step_size) * DENSE if ws > 0 else 0
            de = max(0, n_mid_eval - 1 - step_size) * DENSE if (has_ext and ws > 0) else 0
            t_chunk_dense = poly_start + np.linspace(0, H - 1, n_dense)

            pos_s_dense = (eval_std_d @ coeff_s).numpy()
            vel_s_dense = (d1_std_d @ coeff_s).numpy()
            std_segs.append((t_chunk_dense[dc:], pos_s_dense[dc:], vel_s_dense[dc:]))
            if has_ext:
                t_ext_dense = poly_start + np.linspace(0, n_mid_eval - 1, n_dense_mid)
                pos_e_dense = (eval_mid_d @ coeff_e).numpy()
                vel_e_dense = (d1_mid_d @ coeff_e).numpy()
                ext_segs.append((t_ext_dense[de:], pos_e_dense[de:], vel_e_dense[de:]))
            if coeff_sp is not None:
                pos_sp_dense = (eval_std_d @ coeff_sp).numpy()
                vel_sp_dense = (d1_std_d @ coeff_sp).numpy()
                spl_segs.append((t_chunk_dense[dc:], pos_sp_dense[dc:], vel_sp_dense[dc:]))
            if coeff_gt is not None:
                pos_gt_dense = (eval_std_d @ coeff_gt).numpy()
                vel_gt_dense = (d1_std_d @ coeff_gt).numpy()
                gt_segs.append((t_chunk_dense[dc:], pos_gt_dense[dc:], vel_gt_dense[dc:]))

            if ws > 0:
                boundaries.append(dst_start)

        expert = ep_action[:L]
        expert_vel = np.gradient(ep_action, axis=0)[:L]
        t = np.arange(L)

        fig, axes = plt.subplots(n_j, 4, figsize=(28, 2.6 * n_j), squeeze=False)
        parts = [f"Episode {ep_idx}"]
        if poly_prefix_val is not None:
            poly_suffix_val = getattr(model, 'poly_suffix', 0)
            parts.append(f"poly_prefix={poly_prefix_val}, poly_suffix={poly_suffix_val} (H_poly={H})")
        if has_ext:
            parts.append(f"fit_pad={fit_pad} (H_ext={H_ext})")
        if has_spline:
            parts.append("spline_gt")
        if has_bc:
            parts.append("boundary_constraint")
        parts.append(f"poly_order={poly_order}")
        fig.suptitle(" — ".join(parts), fontsize=13, y=1.0)

        expert_color = "#2ca02c"
        has_others = has_ext or has_spline or has_bc
        std_label = "Std fit" if has_others else "Poly fit"
        std_vel_label = "Std dq/ds" if has_others else "Poly dq/ds"

        for j in range(n_j):
            # Position (dense curves)
            ax = axes[j, 0]
            ax.plot(t, expert[:, j], lw=1.2, color=expert_color, alpha=0.7, label="Expert")
            for k, (st, sp, _) in enumerate(std_segs):
                ax.plot(st, sp[:, j], lw=1, color="tab:blue", alpha=0.8,
                        label=std_label if k == 0 else None)
            if has_ext:
                for k, (st, sp, _) in enumerate(ext_segs):
                    ax.plot(st, sp[:, j], lw=1, color="tab:orange", alpha=0.8, ls="--",
                            label="Ext fit" if k == 0 else None)
            if has_spline:
                for k, (st, sp, _) in enumerate(spl_segs):
                    ax.plot(st, sp[:, j], lw=1, color="tab:red", alpha=0.8, ls="-.",
                            label="Spline proj" if k == 0 else None)
            if has_bc:
                for k, (st, sp, _) in enumerate(gt_segs):
                    ax.plot(st, sp[:, j], lw=1.3, color="tab:purple", alpha=0.9, ls="-",
                            label="Training GT" if k == 0 else None)
            for b in boundaries:
                ax.axvline(b, color="red", lw=0.5, ls="--", alpha=0.4)
            ax.set_ylabel(f"J{j+1} position", fontsize=8)
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_title("Position (rad)", fontsize=10)
                ax.legend(fontsize=6)
            if j == n_j - 1:
                ax.set_xlabel("Timestep", fontsize=9)

            # Velocity (dense curves)
            ax = axes[j, 1]
            ax.plot(t, expert_vel[:, j], lw=1.2, color=expert_color, alpha=0.7,
                    label="Expert (finite diff)")
            for k, (st, _, sv) in enumerate(std_segs):
                ax.plot(st, sv[:, j], lw=1, color="tab:blue", alpha=0.8,
                        label=std_vel_label if k == 0 else None)
            if has_ext:
                for k, (st, _, sv) in enumerate(ext_segs):
                    ax.plot(st, sv[:, j], lw=1, color="tab:orange", alpha=0.8, ls="--",
                            label="Ext dq/ds" if k == 0 else None)
            if has_spline:
                for k, (st, _, sv) in enumerate(spl_segs):
                    ax.plot(st, sv[:, j], lw=1, color="tab:red", alpha=0.8, ls="-.",
                            label="Spline dq/ds" if k == 0 else None)
            if has_bc:
                for k, (st, _, sv) in enumerate(gt_segs):
                    ax.plot(st, sv[:, j], lw=1.3, color="tab:purple", alpha=0.9, ls="-",
                            label="GT dq/ds" if k == 0 else None)
            for b in boundaries:
                ax.axvline(b, color="red", lw=0.5, ls="--", alpha=0.4)
            ax.set_ylabel(f"J{j+1} velocity", fontsize=8)
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_title("Velocity (rad/step)", fontsize=10)
                ax.legend(fontsize=6)
            if j == n_j - 1:
                ax.set_xlabel("Timestep", fontsize=9)

            # Position error (sparse)
            ax = axes[j, 2]
            pos_err_std = pos_std_all[:, j] - expert[:, j]
            ax.plot(t, pos_err_std, lw=1, color="tab:blue", alpha=0.8, label="Std")
            if has_ext:
                pos_err_ext = pos_ext_all[:, j] - expert[:, j]
                ax.plot(t, pos_err_ext, lw=1, color="tab:orange", alpha=0.8, ls="--", label="Ext")
            if has_spline and pos_spl_all is not None:
                pos_err_spl = pos_spl_all[:, j] - expert[:, j]
                ax.plot(t, pos_err_spl, lw=1, color="tab:red", alpha=0.8, ls="-.", label="Spline")
            if has_bc and pos_gt_all is not None:
                pos_err_gt = pos_gt_all[:, j] - expert[:, j]
                ax.plot(t, pos_err_gt, lw=1, color="tab:purple", alpha=0.8, label="GT")
            for b in boundaries:
                ax.axvline(b, color="red", lw=0.5, ls="--", alpha=0.4)
            if pos_err_ylim and j < len(pos_err_ylim):
                ax.set_ylim(pos_err_ylim[j])
            ax.set_ylabel(f"J{j+1} error", fontsize=8)
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_title("Position Error (rad)", fontsize=10)
                ax.legend(fontsize=6)
            if j == n_j - 1:
                ax.set_xlabel("Timestep", fontsize=9)

            # Velocity error (sparse)
            ax = axes[j, 3]
            vel_err_std = vel_std_all[:, j] - expert_vel[:, j]
            ax.plot(t, vel_err_std, lw=1, color="tab:blue", alpha=0.8, label="Std")
            if has_ext:
                vel_err_ext = vel_ext_all[:, j] - expert_vel[:, j]
                ax.plot(t, vel_err_ext, lw=1, color="tab:orange", alpha=0.8, ls="--", label="Ext")
            if has_spline and vel_spl_all is not None:
                vel_err_spl = vel_spl_all[:, j] - expert_vel[:, j]
                ax.plot(t, vel_err_spl, lw=1, color="tab:red", alpha=0.8, ls="-.", label="Spline")
            if has_bc and vel_gt_all is not None:
                vel_err_gt = vel_gt_all[:, j] - expert_vel[:, j]
                ax.plot(t, vel_err_gt, lw=1, color="tab:purple", alpha=0.8, label="GT")
            for b in boundaries:
                ax.axvline(b, color="red", lw=0.5, ls="--", alpha=0.4)
            if vel_err_ylim and j < len(vel_err_ylim):
                ax.set_ylim(vel_err_ylim[j])
            ax.set_ylabel(f"J{j+1} error", fontsize=8)
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_title("Velocity Error (rad/step)", fontsize=10)
                ax.legend(fontsize=6)
            if j == n_j - 1:
                ax.set_xlabel("Timestep", fontsize=9)

        fig.tight_layout()
        path = out_dir / f"fit_pad_diag_ep{ep_idx}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        save_kw = dict(
            expert=expert, expert_vel=expert_vel,
            pos_std=pos_std_all, vel_std=vel_std_all,
            boundaries=np.array(boundaries),
            fit_pad=fit_pad, H=H, H_full=H_full, poly_order=poly_order,
        )
        if has_ext:
            save_kw.update(pos_ext=pos_ext_all, vel_ext=vel_ext_all, H_ext=H_ext)
        if has_spline and pos_spl_all is not None:
            save_kw.update(pos_spl=pos_spl_all, vel_spl=vel_spl_all)
        if has_bc and pos_gt_all is not None:
            save_kw.update(pos_gt=pos_gt_all, vel_gt=vel_gt_all)
        npz_path = out_dir / f"fit_pad_diag_ep{ep_idx}.npz"
        np.savez_compressed(str(npz_path), **save_kw)
        log.info(f"[poly diagnostic] → {path}  data → {npz_path}")


# ===== OOD Target object mapping =====
# Eval OOD: Translate the manipulated target object radially away from the robot base. args.ood_shift (in meters)
# Each task provides candidate names (sorted by priority). During execution, the system first attempts a precise name match, followed by a substring match. If no match is found, it prints the available keys and proceeds to the next task.
OOD_TARGET_OBJECTS = {
    "close_box": ("box",),
    "pick_cube": ("cube",),
    "stack_cube": ("cube",),
    "libero_90.kitchen_scene1_open_bottom_drawer": ("wooden_cabinet", "cabinet", "drawer"),
    "libero_90.kitchen_scene1_put_the_black_bowl_on_the_plate": ("akita_black_bowl", "black_bowl", "bowl"),
}
OOD_TOKENS = ("cube", "box", "bowl", "cabinet", "drawer")


def _apply_ood_shift(state, task_name, shift, log_once=False):
    import math
    if not (isinstance(state, dict) and "objects" in state and state["objects"]):
        return None
    objs = state["objects"]
    keys = list(objs.keys())
    target = None
    for cand in OOD_TARGET_OBJECTS.get(task_name, ()):
        if cand in objs:
            target = cand
            break
    if target is None:
        for cand in list(OOD_TARGET_OBJECTS.get(task_name, ())) + list(OOD_TOKENS):
            hit = [k for k in keys if cand.lower() in k.lower()]
            if hit:
                target = hit[0]
                break
    if target is None and len(keys) == 1:
        target = keys[0]
    if target is None:
        if log_once:
            log.warning(f"[OOD] task={task_name}: no target object matched; available={keys}; skip")
        return None
    base_x, base_y = 0.0, 0.0
    if "robots" in state and state["robots"]:
        rp = list(state["robots"].values())[0].get("pos", None)
        if rp is not None:
            base_x, base_y = float(rp[0]), float(rp[1])
    pos = objs[target]["pos"]
    ox, oy = float(pos[0]), float(pos[1])
    dx, dy = ox - base_x, oy - base_y
    norm = math.hypot(dx, dy)
    ux, uy = (dx / norm, dy / norm) if norm > 1e-6 else (1.0, 0.0)
    nx, ny = ox + shift * ux, oy + shift * uy
    try:
        pos[0] = nx
        pos[1] = ny
    except (TypeError, RuntimeError):
        newpos = list(pos)
        newpos[0], newpos[1] = nx, ny
        objs[target]["pos"] = newpos
    if log_once:
        log.info(f"[OOD] task={task_name} shift={shift}m target={target} pos ({ox:.3f},{oy:.3f})->({nx:.3f},{ny:.3f}) dir=({ux:.2f},{uy:.2f})")
    return target


class DefaultRunner(BaseRunner):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.train_config.training_params.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model (NOTE: during evaluation, this initial model will be
        # rebuilt from the checkpoint's config in DefaultEvalRunner._init_policy)
        if cfg.eval_enable and not cfg.train_enable:
            log.info("[Init] Creating initial model from current config (will be replaced by checkpoint config during eval)...")
        self.model = hydra.utils.instantiate(cfg.policy_config)
        self.policy_name = cfg.policy_name

        self.ema_model = None
        if cfg.train_config.training_params.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.train_config.optimizer, params=self.model.parameters()
        )

        # configure training state
        self.global_step = 0
        self.epoch = 0

        self.eval_args = hydra.utils.instantiate(cfg.eval_config.eval_args)

    def train(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training — always use the checkpoint's saved config to
        # rebuild the model, optimizer, and dataset so that architecture
        # params (poly_order, horizon, hidden_dim, …) are guaranteed to
        # match the saved weights.  Only num_epochs and zarr_path are
        # preserved from the user's current command-line / YAML.
        if cfg.train_config.training_params.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path is not None and lastest_ckpt_path.is_file():
                log.info(f"[Resume] Loading checkpoint {lastest_ckpt_path}")
                payload = torch.load(
                    lastest_ckpt_path.open("rb"), pickle_module=dill)
                ckpt_cfg = payload["cfg"]

                log.info("[Resume] Rebuilding model / optimizer from "
                         "CHECKPOINT config ...")
                self.model = hydra.utils.instantiate(ckpt_cfg.policy_config)
                if ckpt_cfg.train_config.training_params.use_ema:
                    self.ema_model = copy.deepcopy(self.model)
                else:
                    self.ema_model = None
                self.optimizer = hydra.utils.instantiate(
                    ckpt_cfg.train_config.optimizer,
                    params=self.model.parameters())

                user_cfg = cfg
                cfg = copy.deepcopy(ckpt_cfg)
                cfg.train_config.training_params.num_epochs = (
                    user_cfg.train_config.training_params.num_epochs)
                cfg.train_config.training_params.resume = True
                cfg.dataset_config.zarr_path = (
                    user_cfg.dataset_config.zarr_path)
                for _key in ("num_steps", "checkpoint_every_n_steps"):
                    _val = user_cfg.train_config.training_params.get(_key, None)
                    if _val is not None:
                        cfg.train_config.training_params[_key] = _val
                log.info("[Resume] Checkpoint config applied.  "
                         f"policy: poly_order={ckpt_cfg.policy_config.get('poly_order','?')}, "
                         f"horizon={ckpt_cfg.policy_config.get('horizon','?')}, "
                         f"n_obs_steps={ckpt_cfg.policy_config.get('n_obs_steps','?')}")

                self.load_payload(payload)
                self.epoch += 1
                self.global_step += 1
                log.info(f"[Resume] Continuing from epoch {self.epoch}")

                # Persist the checkpoint's config so save_checkpoint writes correct
                # architecture info (horizon, poly_order, …) into future checkpoints.
                self.cfg = copy.deepcopy(cfg)

        # When fit_pad > 0, extend the dataset's horizon and padding so that
        # each training sample provides extra context around the base window.
        fit_pad = getattr(self.model, 'fit_pad', 0)
        if fit_pad > 0:
            cfg.dataset_config.horizon = cfg.dataset_config.horizon + 2 * fit_pad
            cfg.dataset_config.pad_before = cfg.dataset_config.pad_before + fit_pad
            cfg.dataset_config.pad_after = cfg.dataset_config.pad_after + fit_pad
            log.info(f"[fit_pad={fit_pad}] dataset horizon → {cfg.dataset_config.horizon}, "
                     f"pad_before → {cfg.dataset_config.pad_before}, "
                     f"pad_after → {cfg.dataset_config.pad_after}")

        # configure dataset
        dataset = hydra.utils.instantiate(cfg.dataset_config)
        train_dataloader = create_dataloader(dataset, **cfg.train_config.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        # get_validation_dataset() does a shallow copy of the training dataset, so
        # val_dataset.batch_size / buffers still have the training batch_size (e.g. 32).
        # The val_dataloader may use a different batch_size (e.g. 4 for high
        # downsample_ratio), so we rebuild the buffers to match.
        val_bs = cfg.train_config.val_dataloader.batch_size
        if val_bs != val_dataset.batch_size:
            seq_len = val_dataset.sampler.sequence_length
            val_dataset.batch_size = val_bs
            val_dataset.buffers = {
                k: np.zeros((val_bs, seq_len, *v.shape[1:]), dtype=v.dtype)
                for k, v in val_dataset.sampler.replay_buffer.items()
            }
            # torch.from_numpy shares memory with the numpy array, so writes
            # to self.buffers[k] in __getitem__ are reflected in the tensor.
            # Do NOT chain .pin_memory() here — it would create a separate
            # copy and break the shared-memory contract.
            val_dataset.buffers_torch = {
                k: torch.from_numpy(v) for k, v in val_dataset.buffers.items()
            }
            for v in val_dataset.buffers_torch.values():
                v.pin_memory()
        val_dataloader = create_dataloader(
            val_dataset, **cfg.train_config.val_dataloader
        )

        self.model.set_normalizer(normalizer)
        if cfg.train_config.training_params.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # Pre-compute spline-projected Legendre coefficients when enabled
        if getattr(self.model, 'spline_gt', False):
            from roboverse_learn.il.utils.spline_projection import precompute_all_spline_coefficients
            coeff_scale = self.model._coeff_scale.squeeze().cpu().numpy()  # (C,)
            base_horizon = self.model.horizon
            _spline_kw = dict(
                poly_order=self.model.poly_order,
                fit_pad=fit_pad,
                horizon=base_horizon,
                coeff_scale=coeff_scale,
            )
            log.info("[spline_gt] Pre-computing spline-projected Legendre "
                     "coefficients for training set ...")
            train_coeffs = precompute_all_spline_coefficients(
                replay_buffer=dataset.replay_buffer,
                normalizer=normalizer,
                sampler_indices=dataset.sampler.indices,
                **_spline_kw,
            )
            dataset.set_spline_coefficients(train_coeffs)
            log.info(f"[spline_gt] Training set: {train_coeffs.shape}")

            log.info("[spline_gt] Pre-computing for validation set ...")
            val_coeffs = precompute_all_spline_coefficients(
                replay_buffer=val_dataset.replay_buffer,
                normalizer=normalizer,
                sampler_indices=val_dataset.sampler.indices,
                **_spline_kw,
            )
            val_dataset.set_spline_coefficients(val_coeffs)
            log.info(f"[spline_gt] Validation set: {val_coeffs.shape}")

        # Generate diagnostic plot: expert trajectory vs poly-fitted training data
        try:
            _pos_ylim, _vel_ylim = _get_or_compute_task_ylim(
                dataset, self.output_dir, horizon=self.model.horizon)
            _generate_poly_diagnostic(
                self.model, dataset, self.output_dir, num_episodes=5,
                pos_err_ylim=_pos_ylim, vel_err_ylim=_vel_ylim)
        except Exception as e:
            log.warning(f"[poly diagnostic] plot failed: {e}")

        # configure lr scheduler
        # Steps per epoch is the smaller of the full dataloader and max_train_steps.
        max_steps_cfg = cfg.train_config.training_params.max_train_steps
        steps_per_epoch = len(train_dataloader)
        if max_steps_cfg is not None and max_steps_cfg < steps_per_epoch:
            steps_per_epoch = max_steps_cfg
        grad_accum = cfg.train_config.training_params.gradient_accumulate_every

        # num_steps mode: convert total steps → num_epochs
        num_steps_cfg = cfg.train_config.training_params.get("num_steps", 0) or 0
        if num_steps_cfg > 0:
            computed_epochs = math.ceil(num_steps_cfg / steps_per_epoch)
            actual_steps = computed_epochs * steps_per_epoch
            log.info(f"[num_steps mode] {num_steps_cfg} steps requested → "
                     f"{computed_epochs} epochs × {steps_per_epoch} steps/epoch "
                     f"= {actual_steps} actual steps")
            cfg.train_config.training_params.num_epochs = computed_epochs

        num_new_epochs = cfg.train_config.training_params.num_epochs
        total_new_steps = (steps_per_epoch * num_new_epochs) // grad_accum

        # On resume the optimizer is restored from checkpoint, but we build a
        # FRESH scheduler over the new num_epochs so that the full warmup +
        # cosine decay cycle runs again.  Without this, global_step would
        # already exceed total_training_steps and the LR would stay at 0.
        # lr_total_steps > 0 decouples the scheduler horizon from the actual run
        # length (e.g. to reproduce the de-facto constant-LR regime of the legacy
        # num_epochs=10000 baseline runs while still auto-stopping at num_steps).
        lr_total_steps = cfg.train_config.training_params.get("lr_total_steps", 0) or 0
        scheduler_horizon = lr_total_steps if lr_total_steps > 0 else total_new_steps
        lr_scheduler = get_scheduler(
            cfg.train_config.training_params.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.train_config.training_params.lr_warmup_steps,
            num_training_steps=scheduler_horizon,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=-1,
        )
        log.info(f"[LR Scheduler] steps_per_epoch={steps_per_epoch} "
                 f"(dataloader={len(train_dataloader)}, max_train_steps={max_steps_cfg}), "
                 f"total_training_steps={total_new_steps}, "
                 f"scheduler_horizon={scheduler_horizon}, "
                 f"lr_warmup_steps={cfg.train_config.training_params.lr_warmup_steps}")

        # optional grad clipping (0 = disabled, identical to all legacy runs)
        grad_clip_norm = cfg.train_config.training_params.get("grad_clip_norm", 0) or 0
        if grad_clip_norm > 0:
            log.info(f"[Grad Clip] clip_grad_norm_ enabled with max_norm={grad_clip_norm}")

        # configure ema
        ema: EMAModel | None = None
        if cfg.train_config.training_params.use_ema:
            ema = hydra.utils.instantiate(cfg.train_config.ema, model=self.ema_model)

        wandb_run = None

        # configure logging
        if cfg.logging.mode == "online":
            # Truncate tags to max 64 characters (wandb limit)
            logging_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
            if "tags" in logging_cfg and logging_cfg["tags"]:
                logging_cfg["tags"] = [tag[:64] if len(tag) > 64 else tag for tag in logging_cfg["tags"]]

            # If resuming, reuse the saved WandB run ID so the curve continues on the same run.
            wandb_id_path = pathlib.Path(self.output_dir) / "wandb_run_id.txt"
            if cfg.train_config.training_params.resume and wandb_id_path.exists():
                saved_id = wandb_id_path.read_text().strip()
                if saved_id:
                    logging_cfg["id"] = saved_id
                    logging_cfg["resume"] = "allow"
                    log.info(f"[WandB] Resuming run id={saved_id}")

            try:
                wandb_run = wandb.init(
                    dir=str(self.output_dir),
                    config=OmegaConf.to_container(cfg, resolve=True),
                    **logging_cfg,
                )
            except Exception as e:
                log.warning(f"[WandB] Failed to resume run: {e}. "
                            "Starting a fresh run instead.")
                logging_cfg.pop("id", None)
                logging_cfg.pop("resume", None)
                wandb_run = wandb.init(
                    dir=str(self.output_dir),
                    config=OmegaConf.to_container(cfg, resolve=True),
                    **logging_cfg,
                )
            wandb.config.update(
                {"output_dir": self.output_dir},
                allow_val_change=True,
            )

            # Persist the run ID so future resumes can reconnect to this run.
            wandb_id_path.write_text(wandb_run.id)

            # On resume, WandB may have logged steps beyond our checkpoint's
            # global_step (e.g. from a previous crashed run).  Advance
            # global_step so we never try to log at an already-consumed step.
            if cfg.train_config.training_params.resume and wandb_run.step > self.global_step:
                log.info(f"[WandB] Advancing global_step {self.global_step} -> {wandb_run.step} to match WandB history")
                self.global_step = wandb_run.step

        # device transfer
        device = torch.device(cfg.train_config.training_params.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.train_config.training_params.debug:
            cfg.train_config.training_params.num_epochs = 2
            cfg.train_config.training_params.max_train_steps = 3
            cfg.train_config.training_params.max_val_steps = 3
            cfg.train_config.training_params.rollout_every = 1
            cfg.train_config.training_params.checkpoint_every = 1
            cfg.train_config.training_params.val_every = 1
            cfg.train_config.training_params.sample_every = 1

        # training loop
        ckpt_every_n_steps = cfg.train_config.training_params.get(
            "checkpoint_every_n_steps", 0) or 0
        if ckpt_every_n_steps > 0:
            log.info(f"[Checkpoint] Step-based mode: save every {ckpt_every_n_steps} global steps")
        else:
            log.info(f"[Checkpoint] Epoch-based mode: save every "
                     f"{cfg.train_config.training_params.checkpoint_every} epochs")

        start_epoch = self.epoch
        target_epoch = start_epoch + cfg.train_config.training_params.num_epochs
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.train_config.training_params.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.train_config.training_params.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.train_config.training_params.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dataset.postprocess(batch, device)
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            raw_loss = self.model.compute_loss(batch)
                        loss = (
                            raw_loss
                            / cfg.train_config.training_params.gradient_accumulate_every
                        )
                        loss.backward()

                        # step optimizer
                        grad_norm_cpu = None
                        if (
                            self.global_step
                            % cfg.train_config.training_params.gradient_accumulate_every
                            == 0
                        ):
                            # optional grad clipping (official MIP recipe: 10);
                            # disabled (0) by default -> identical to all legacy runs
                            if grad_clip_norm > 0:
                                grad_norm_cpu = torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), grad_clip_norm
                                ).item()
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.train_config.training_params.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }
                        if grad_norm_cpu is not None:
                            step_log["grad_norm"] = grad_norm_cpu
                        # policies may expose per-term loss components (e.g. MIP's
                        # draft/refine terms) for logging via this optional attr
                        loss_components = getattr(self.model, "last_loss_components", None)
                        if loss_components:
                            step_log.update(loss_components)

                        is_last_batch = batch_idx == (steps_per_epoch - 1)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            if wandb_run is not None:
                                wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                            # step-based checkpoint (mid-epoch)
                            if (ckpt_every_n_steps > 0
                                    and self.global_step % ckpt_every_n_steps == 0):
                                self.save_checkpoint(
                                    cfg.checkpoint.save_root_dir
                                    + f"/checkpoints/{self.global_step}.ckpt"
                                )

                        if (
                            cfg.train_config.training_params.max_train_steps is not None
                        ) and batch_idx >= (
                            cfg.train_config.training_params.max_train_steps - 1
                        ):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.train_config.training_params.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                # if (self.epoch % cfg.train_config.training_params.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     # log all
                #     step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.train_config.training_params.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.train_config.training_params.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dataset.postprocess(batch, device)
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (
                                    cfg.train_config.training_params.max_val_steps
                                    is not None
                                ) and batch_idx >= (
                                    cfg.train_config.training_params.max_val_steps - 1
                                ):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log["val_loss"] = val_loss

                # Latent space visualization for A2A policy
                if hasattr(policy, 'get_latents_for_visualization'):
                    try:
                        with torch.no_grad():
                            # Collect latents from multiple validation batches
                            all_history_latents = []
                            all_future_latents = []
                            max_samples = 500  # Limit samples for t-SNE performance
                            first_batch = None
                            
                            for batch_idx, batch in enumerate(val_dataloader):
                                batch = dataset.postprocess(batch, device)
                                if first_batch is None:
                                    first_batch = batch  # Save for trajectory visualization
                                history_latents, future_latents = policy.get_latents_for_visualization(batch)
                                all_history_latents.append(history_latents.cpu())
                                all_future_latents.append(future_latents.cpu())
                                
                                if sum(h.shape[0] for h in all_history_latents) >= max_samples:
                                    break
                            
                            # Concatenate all collected latents
                            history_latents = torch.cat(all_history_latents, dim=0)[:max_samples]
                            future_latents = torch.cat(all_future_latents, dim=0)[:max_samples]
                            
                            # Get flow trajectories for visualization (uses model's num_sampling_steps)
                            trajectories = None
                            trajectory_targets = None
                            if hasattr(policy, 'get_flow_trajectories') and first_batch is not None:
                                trajectories, trajectory_targets = policy.get_flow_trajectories(
                                    first_batch, n_samples=5
                                )
                            
                            # Generate all visualizations
                            viz_dir = pathlib.Path(self.output_dir) / "latent_viz"
                            viz_results = plot_all_latent_visualizations(
                                history_latents=history_latents,
                                future_latents=future_latents,
                                epoch=self.epoch + 1,
                                save_dir=str(viz_dir),
                                trajectories=trajectories,
                                trajectory_targets=trajectory_targets,
                            )
                            log.info(f"Saved latent visualizations to {viz_dir}")
                            log.info(f"  Avg t-SNE Distance: {viz_results['avg_tsne_distance']:.2f}")
                            
                            # Log metrics to wandb
                            wandb_metrics = {
                                "latent/avg_tsne_distance": viz_results['avg_tsne_distance'],
                            }
                            if 'flow_end_to_target_dist' in viz_results:
                                wandb_metrics["latent/flow_end_to_target_dist"] = viz_results['flow_end_to_target_dist']
                            if wandb_run is not None:
                                wandb_run.log(wandb_metrics, step=self.global_step)
                    except Exception as e:
                        log.warning(f"Failed to generate latent visualization: {e}")

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.train_config.training_params.sample_every) == 0:
                    with torch.no_grad():
                        batch = train_sampling_batch
                        obs_dict = batch["obs"]
                        gt_action = batch["action"]

                        # When fit_pad > 0, the batch contains H_ext-length sequences.
                        # predict_action expects obs starting at the model's obs window,
                        # not the pre-padding region.  Slice to the middle portion.
                        fp = getattr(policy, 'fit_pad', 0)
                        if fp > 0:
                            from roboverse_learn.il.utils.pytorch_util import dict_apply as _da
                            obs_dict = _da(obs_dict, lambda x: x[:, fp:, ...])
                            gt_action = gt_action[:, fp:fp + policy.horizon, :]

                        result = policy.predict_action(obs_dict)
                        pred_action = result["action_pred"]
                        
                        pred_len = pred_action.shape[1]
                        gt_len = gt_action.shape[1]
                        if pred_len != gt_len:
                            n_obs_steps = gt_len - pred_len + 1
                            start_idx = n_obs_steps - 1
                            gt_action = gt_action[:, start_idx:start_idx + pred_len, :]
                        
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # checkpoint (epoch-based, only when step-based mode is off)
                if ckpt_every_n_steps <= 0:
                    if (
                        (self.epoch + 1) % cfg.train_config.training_params.checkpoint_every
                    ) == 0 or self.epoch + 1 >= target_epoch:
                        self.save_checkpoint(
                            cfg.checkpoint.save_root_dir
                            + f"/checkpoints/{self.epoch + 1}.ckpt"
                        )

                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                json_logger.log(step_log)
                if wandb_run is not None:
                    wandb_run.log(step_log, step=self.global_step)
                self.global_step += 1

                # step-based checkpoint for the last batch of this epoch
                if ckpt_every_n_steps > 0:
                    is_last_epoch = (self.epoch + 1 >= target_epoch)
                    if self.global_step % ckpt_every_n_steps == 0 or is_last_epoch:
                        self.save_checkpoint(
                            cfg.checkpoint.save_root_dir
                            + f"/checkpoints/{self.global_step}.ckpt"
                        )

                self.epoch += 1

        # Wait for the last checkpoint-saving thread to finish so the file
        # is ready before evaluate() tries to load it.
        if hasattr(self, "_saving_thread") and self._saving_thread is not None:
            self._saving_thread.join()

        # Explicitly finish the wandb run after training is complete.
        # Without this, wandb relies on an atexit handler to call finish().
        # During evaluation, huge stdout output causes wandb's sender to back up
        # ("flowcontrol: backed up"), and the atexit handler may not complete
        # before the process exits — leaving the run as "crashed" with missing metrics (e.g. lr).
        if wandb_run is not None:
            wandb_run.finish()

    def evaluate(self, ckpt_path=None):
        args = self.eval_args

        num_envs: int = args.num_envs
        log.info(f"Using GPU device: {args.gpu_id}")
        task_cls = get_task_class(args.task)

        # Camera configuration
        if args.task in {"stack_cube", "pick_cube", "pick_butter"}:
            dp_camera = True
        else:
            dp_camera = args.task != "close_box"

        is_libero_dataset = "libero_90" in args.task

        if is_libero_dataset:
            dp_pos = (2.0, 0.0, 2)
        elif dp_camera:
            dp_pos = (1.0, 0.0, 0.75)
        else:
            dp_pos = (1.5, 0.0, 1.5)

        camera = PinholeCameraCfg(
            name="camera0",
            data_types=["rgb", "depth"],
            width=256,
            height=256,
            pos=dp_pos,
            look_at=(0.0, 0.0, 0.0),
        )

        # Lighting setup
        render_mode = getattr(args, 'render_mode', 'raytracing')
        if render_mode == "pathtracing":
            ceiling_main = 18000.0
            ceiling_corners = 8000.0
        else:
            ceiling_main = 12000.0
            ceiling_corners = 5000.0

        from metasim.scenario.lights import DiskLightCfg, SphereLightCfg
        lights = [
            DiskLightCfg(
                name="ceiling_main",
                intensity=ceiling_main,
                color=(1.0, 1.0, 1.0),
                radius=1.2,
                pos=(0.0, 0.0, 2.8),
                rot=(0.7071, 0.0, 0.0, 0.7071),
            ),
            SphereLightCfg(
                name="ceiling_ne", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(1.0, 1.0, 2.5)
            ),
            SphereLightCfg(
                name="ceiling_nw", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(-1.0, 1.0, 2.5)
            ),
            SphereLightCfg(
                name="ceiling_sw", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(-1.0, -1.0, 2.5)
            ),
            SphereLightCfg(
                name="ceiling_se", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(1.0, -1.0, 2.5)
            ),
        ]

        scenario = task_cls.scenario.update(
            robots=[args.robot],
            simulator=args.sim,
            num_envs=args.num_envs,
            headless=args.headless,
            lights=lights,
            cameras=[camera]
        )

        has_velocity = hasattr(self.model, "_compute_velocity")
        # A policy is velocity-capable either natively (FLASH family,
        # _compute_velocity) or via finite differences of its own position
        # targets (fd_vel_target=True, any policy).
        fd_vel = getattr(args, "fd_vel_target", False)
        vel_capable = has_velocity or fd_vel
        if fd_vel:
            log.info("[FD Velocity] fd_vel_target=True — velocity targets from "
                     "finite differences of the position-target chunk"
                     + (" (REPLACES the policy's analytic velocity)." if has_velocity else "."))

        if getattr(args, "velocity_pd_gains", False) and not vel_capable:
            log.warning("[Velocity PD] velocity_pd_gains=True ignored for "
                        f"policy '{self.policy_name}' (no _compute_velocity method).")
        if getattr(args, "velocity_pd_gains", False) and vel_capable and scenario.robots:
            kp_scale = getattr(args, "velocity_kp_scale", 1.0)
            kd_scale = getattr(args, "velocity_kd_scale", 1.0)
            robot_cfg = scenario.robots[0]
            log.info(f"[Velocity PD] Applying gain overrides  "
                     f"(Kp scale={kp_scale:.2f}, Kd scale={kd_scale:.2f}):")
            log.info(f"  {'Joint':<16} {'Kp_orig':>10} {'Kp_new':>10}   "
                     f"{'Kd_orig':>10} {'Kd_new':>10}")
            for jname in VELOCITY_MODE_KP:
                act = robot_cfg.actuators.get(jname)
                if act is None:
                    continue
                old_kp, old_kd = act.stiffness, act.damping
                new_kp = VELOCITY_MODE_KP[jname] * kp_scale
                new_kd = VELOCITY_MODE_KD[jname] * kd_scale
                act.stiffness = new_kp
                act.damping = new_kd
                log.info(f"  {jname:<16} {old_kp:>10.0f} {new_kp:>10.1f}   "
                         f"{old_kd:>10.0f} {new_kd:>10.1f}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tic = time.time()
        env = task_cls(scenario, device=device)
        robot = get_robot(args.robot)

        # Domain Randomization configuration
        dr_level = getattr(args, 'level', 0)
        dr_scene_mode = getattr(args, 'scene_mode', 0)
        dr_seed = getattr(args, 'randomization_seed', None)

        if not RANDOMIZATION_AVAILABLE:
            if dr_level > 0:
                log.warning("Domain randomization requested but not available!")
            randomization_manager = None
        else:
            from dataclasses import dataclass as dc

            @dc
            class SimpleRenderCfg:
                mode: str = render_mode

            randomization_manager = DomainRandomizationManager(
                config=DRConfig(
                    level=dr_level,
                    scene_mode=dr_scene_mode,
                    randomization_seed=dr_seed,
                ),
                scenario=scenario,
                handler=env.handler,
                init_states=None,
                render_cfg=SimpleRenderCfg(mode=render_mode)
            )
            if dr_level > 0:
                log.info(f"Domain Randomization enabled: level={dr_level}, scene_mode={dr_scene_mode}, seed={dr_seed}")
            else:
                log.info("Domain Randomization disabled (level=0)")

        toc = time.time()
        log.trace(f"Time to launch: {toc - tic:.2f}s")

        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint = self.get_checkpoint_path()
        if ckpt_path is not None and not pathlib.Path(ckpt_path).is_file():
            log.warning(f"Specified checkpoint not found: {ckpt_path}. "
                        f"Falling back to latest checkpoint.")
            ckpt_path = None
        checkpoint = ckpt_path if ckpt_path is not None else checkpoint
        if checkpoint is None:
            raise ValueError(
                "No checkpoint found, please provide a valid checkpoint path."
            )
        args.checkpoint_path = pathlib.Path(checkpoint)
        n_steps = getattr(self.model, 'num_inference_steps', None)
        ds_ratio = getattr(args, "downsample_ratio", 1)
        # Eval dir: {stem}_ds{ds}[_dr{dr}][_step{n}].ckpt_... — dr omitted when 0
        ckpt_stem = pathlib.Path(args.checkpoint_path.name).stem
        ds_dr = f"_ds{ds_ratio}_dr{dr_level}" if dr_level > 0 else f"_ds{ds_ratio}"
        if n_steps is not None:
            ckpt_base = f"{ckpt_stem}{ds_dr}_step{n_steps}.ckpt"
        else:
            ckpt_base = f"{ckpt_stem}{ds_dr}.ckpt"
        # C_h-perturbation sweeps get a _hn{sigma} tag so per-sigma eval dirs stay
        # distinguishable. Appended right before ".ckpt" (after _step{n}) so the
        # existing name handling — split on ".ckpt_", _sr insertion after the
        # first "_" segment — is untouched; sigma=0 keeps names byte-identical.
        _hn_sigma = float(getattr(args, "eval_history_noise_std", 0.0) or 0.0)
        if _hn_sigma > 0.0:
            ckpt_base = ckpt_base[:-len(".ckpt")] + f"_hn{_hn_sigma}.ckpt"
        ckpt_name = ckpt_base + "_" + time_str
        ckpt_name = f"{args.task}/{self.policy_name}/{args.robot}/{ckpt_name}"

        from roboverse_learn.il.runners.default_eval_runner import DefaultEvalRunner

        send_vel = getattr(args, "send_vel_target", False) and vel_capable
        if getattr(args, "send_vel_target", False) and not vel_capable:
            log.warning("[Velocity] send_vel_target=True ignored for "
                        f"policy '{self.policy_name}' (no _compute_velocity method "
                        "and fd_vel_target=False).")

        policyRunner = DefaultEvalRunner(
            self,
            scenario=scenario,
            num_envs=num_envs,
            checkpoint_path=args.checkpoint_path,
            device=f"cuda:{args.gpu_id}",
            task_name=args.task,
            subset=args.subset,
            downsample_ratio=ds_ratio,
            send_vel_target=send_vel,
            fd_vel_target=fd_vel,
            gripper_binarize=getattr(args, "gripper_binarize", False),
            gripper_binarize_threshold=getattr(args, "gripper_binarize_threshold", 0.02),
            eval_history_noise_std=float(getattr(args, "eval_history_noise_std", 0.0) or 0.0),
            eval_history_noise_seed=int(getattr(args, "eval_history_noise_seed", 0) or 0),
        )

        # Capture actual PD gains for end-of-eval logging
        robot_cfg = scenario.robots[0] if scenario.robots else None
        eval_params = {
            "send_vel_target": send_vel,
            "fd_vel_target": fd_vel,
            "velocity_pd_gains": getattr(args, "velocity_pd_gains", False),
            "velocity_kp_scale": getattr(args, "velocity_kp_scale", 1.0),
            "velocity_kd_scale": getattr(args, "velocity_kd_scale", 1.0),
            "downsample_ratio": ds_ratio,
            "max_step": args.max_step,
        }
        if robot_cfg is not None:
            eval_params["joint_gains"] = {
                jname: {"Kp": act.stiffness, "Kd": act.damping}
                for jname, act in sorted(robot_cfg.actuators.items())
            }

        action_set_steps = (
            2 if policyRunner.policy_cfg.action_config.action_type == "ee" else 1
        )
        # Data
        tic = time.time()
        assert os.path.exists(env.traj_filepath), (
            f"Trajectory file: {env.traj_filepath} does not exist."
        )
        init_states, all_actions, all_states = get_traj(env.traj_filepath, robot, env.handler)
        num_demos = len(init_states)
        # ===== OOD: Moves the target object radially outward (away from the robot base) according to args.ood_shift =====
        _ood_shift = float(getattr(args, "ood_shift", 0.0) or 0.0)
        if _ood_shift != 0.0:
            for _i, _st in enumerate(init_states):
                _apply_ood_shift(_st, args.task, _ood_shift, log_once=(_i == 0))
            log.info(f"[OOD] applied ood_shift={_ood_shift}m to {len(init_states)} init_states (radial away from base)")
        toc = time.time()
        log.trace(f"Time to load data: {toc - tic:.2f}s")

        # Update DR manager with init_states
        if randomization_manager is not None:
            randomization_manager.init_states = init_states
            randomization_manager.original_positions = {}
            for demo_idx, init_state in enumerate(init_states):
                demo_key = f"demo_{demo_idx}"
                randomization_manager.original_positions[demo_key] = {}

                if "objects" in init_state:
                    for obj_name, obj_state in init_state["objects"].items():
                        randomization_manager.original_positions[demo_key][f"obj_{obj_name}"] = {
                            "x": float(obj_state["pos"][0]),
                            "y": float(obj_state["pos"][1]),
                            "z": float(obj_state["pos"][2]),
                        }

                if "robots" in init_state:
                    for robot_name, robot_state in init_state["robots"].items():
                        randomization_manager.original_positions[demo_key][f"robot_{robot_name}"] = {
                            "x": float(robot_state["pos"][0]),
                            "y": float(robot_state["pos"][1]),
                            "z": float(robot_state["pos"][2]),
                        }

        total_success = 0
        total_completed = 0
        all_inference_times = []  # Collect inference times from all steps
        demo_avg_inference_times = []  # Collect average inference time for each demo
        all_per_inference_times = []  # Collect times only for actual model forward passes
        demo_avg_per_inference_times = []  # Collect per-inference average for each demo
        success_episode_durations = []  # Episode wall-clock durations for successful episodes (ms)
        success_episode_total_infer_times = []  # Total inference time per successful episode (ms)
        
        if args.max_demo is None:
            max_demos = args.task_id_range_high - args.task_id_range_low
        else:
            max_demos = args.max_demo
        max_demos = min(max_demos, num_demos)

        # ---- Write 00_eval_config.txt BEFORE running demos ----
        base_eval_dir = pathlib.Path(self.output_dir).joinpath("eval", ckpt_name)
        base_eval_dir.mkdir(parents=True, exist_ok=True)

        config_lines = []
        config_lines.append("=" * 70)
        config_lines.append("[Eval Config] Parameters used in THIS evaluation run:")
        config_lines.append(f"  send_vel_target     = {eval_params['send_vel_target']}")
        config_lines.append(f"  fd_vel_target       = {eval_params['fd_vel_target']}")
        config_lines.append(f"  velocity_pd_gains   = {eval_params['velocity_pd_gains']}")
        config_lines.append(f"  velocity_kp_scale   = {eval_params['velocity_kp_scale']}")
        config_lines.append(f"  velocity_kd_scale   = {eval_params['velocity_kd_scale']}")
        config_lines.append(f"  downsample_ratio    = {eval_params['downsample_ratio']}")
        config_lines.append(f"  dr_level_eval       = {dr_level}")
        config_lines.append(f"  max_step            = {eval_params['max_step']}")
        policy_obj = policyRunner.policy
        config_lines.append(f"  fit_pad             = {getattr(policy_obj, 'fit_pad', 0)}")
        config_lines.append(f"  boundary_constraint = {getattr(policy_obj, 'boundary_constraint', False)}")
        config_lines.append(f"  poly_prefix         = {getattr(policy_obj, 'poly_prefix', None)}")
        config_lines.append(f"  poly_suffix         = {getattr(policy_obj, 'poly_suffix', 0)}")
        config_lines.append(f"  H_poly              = {getattr(policy_obj, '_H_poly', 'N/A')}")
        config_lines.append(f"  --- Model Architecture (from checkpoint) ---")
        config_lines.append(f"  poly_order          = {getattr(policy_obj, 'poly_order', 'N/A')}")
        config_lines.append(f"  basis_type          = {getattr(policy_obj, 'basis_type', 'N/A')}")
        config_lines.append(f"  horizon             = {getattr(policy_obj, 'horizon', 'N/A')}")
        config_lines.append(f"  n_obs_steps         = {getattr(policy_obj, 'n_obs_steps', 'N/A')}")
        config_lines.append(f"  n_action_steps      = {getattr(policy_obj, 'n_action_steps', 'N/A')}")
        config_lines.append(f"  num_inference_steps = {getattr(policy_obj, 'num_inference_steps', 'N/A')}")
        if hasattr(policy_obj, 'flash_solver'):
            config_lines.append(f"  --- FLASH Parameters ---")
            config_lines.append(f"  flash_solver        = {policy_obj.flash_solver}")
            config_lines.append(f"  history_reg_lambda  = {policy_obj.history_reg_lambda} (from checkpoint)")
            config_lines.append(f"  history_noise_std   = {policy_obj.history_noise_std} (from checkpoint)")
            config_lines.append(f"  eval_history_noise_std = "
                                f"{getattr(policy_obj, 'eval_history_noise_std', 0.0)} "
                                f"(EVAL-TIME C_h perturbation, this run only)")
        if "joint_gains" in eval_params:
            config_lines.append(f"  {'Joint':<22} {'Kp':>12} {'Kd':>12}")
            for jname, gains in eval_params["joint_gains"].items():
                config_lines.append(
                    f"  {jname:<22} {gains['Kp']:>12.1f} {gains['Kd']:>12.1f}")
        config_lines.append("=" * 70)

        for line in config_lines:
            log.info(line)
        with open(base_eval_dir.joinpath("00_eval_config.txt"), "w") as f:
            f.write("\n".join(config_lines) + "\n")

        for demo_start_idx in range(
            args.task_id_range_low, args.task_id_range_low + max_demos, num_envs
        ):
            demo_end_idx = min(demo_start_idx + num_envs, num_demos)
            current_demo_idxs = list(range(demo_start_idx, demo_end_idx))

            # Apply domain randomization before reset
            if randomization_manager is not None and dr_level > 0:
                for env_id, demo_idx in enumerate(current_demo_idxs):
                    log.info(f"[DP Eval] Episode {demo_idx}: Applying DR")
                    randomization_manager.apply_randomization(
                        demo_idx=demo_idx, is_initial=(demo_start_idx == args.task_id_range_low))
                    randomization_manager.update_positions_to_table(demo_idx=demo_idx, env_id=env_id)
                    randomization_manager.update_camera_look_at(env_id=env_id)
                    randomization_manager.apply_camera_randomization()

            tic = time.time()
            obs, extras = env.reset(states=init_states[demo_start_idx:demo_end_idx])
            toc = time.time()
            log.trace(f"Time to reset: {toc - tic:.2f}s")

            # Ensure environment stabilizes after reset
            if randomization_manager is not None and dr_level > 0:
                ensure_clean_state(env.handler)

                if hasattr(env, "_episode_steps"):
                    for env_id in range(num_envs):
                        env._episode_steps[env_id] = 0

            policyRunner.reset()

            step = 0
            MaxStep = args.max_step
            SuccessOnce = [False] * num_envs
            TimeOut = [False] * num_envs
            images_list = []
            inference_times = []  # Record inference time for each step
            per_inference_times = []  # Record time only for actual model forward passes
            vel_diag_desired = []
            vel_diag_actual = []
            pos_diag_desired = []
            pos_diag_actual = []
            print(policyRunner.policy_cfg)

            episode_start_time = time.time()
            while step < MaxStep:
                new_obs = {
                    "rgb": obs.cameras["camera0"].rgb,
                    "joint_qpos": obs.robots[args.robot].joint_pos,
                }

                images_list.append(np.array(new_obs["rgb"].cpu()))
                
                # Measure inference time (synchronize GPU to ensure accurate timing)
                torch.cuda.synchronize()
                inference_start = time.time()
                action = policyRunner.get_action(new_obs)
                torch.cuda.synchronize()
                inference_end = time.time()
                inference_time_ms = (inference_end - inference_start) * 1000
                inference_times.append(inference_time_ms)
                if policyRunner.last_call_was_inference:
                    per_inference_times.append(inference_time_ms)

                log.debug(f"Step {step} | Inference time: {inference_time_ms:.2f}ms")

                for round_i in range(action_set_steps):
                    obs, reward, success, time_out, extras = env.step(action)

                # Record diagnostics (env 0 only to keep data small)
                if policyRunner.last_action_tensor is not None:
                    pos_diag_desired.append(policyRunner.last_action_tensor[0].cpu().numpy())
                    pos_diag_actual.append(obs.robots[args.robot].joint_pos[0].cpu().numpy())
                if policyRunner.last_velocity is not None:
                    vel_diag_desired.append(policyRunner.last_velocity[0].cpu().numpy())
                    vel_diag_actual.append(obs.robots[args.robot].joint_vel[0].cpu().numpy())

                # eval
                SuccessOnce = [SuccessOnce[i] or success[i] for i in range(num_envs)]
                TimeOut = [TimeOut[i] or time_out[i] for i in range(num_envs)]
                step += 1
                if all(SuccessOnce):
                    break

            episode_end_time = time.time()
            episode_duration_ms = (episode_end_time - episode_start_time) * 1000

            # Calculate inference time statistics
            total_steps = len(inference_times)
            avg_inference_time = sum(inference_times) / total_steps if total_steps > 0 else 0
            min_inference_time = min(inference_times) if inference_times else 0
            max_inference_time = max(inference_times) if inference_times else 0
            
            log.info(f"Demo {demo_start_idx}-{demo_end_idx}: Avg inference time: {avg_inference_time:.2f}ms, "
                     f"Min: {min_inference_time:.2f}ms, Max: {max_inference_time:.2f}ms, Total steps: {total_steps}")
            
            # Collect inference times for overall statistics
            all_inference_times.extend(inference_times)
            demo_avg_inference_times.append(avg_inference_time)  # Store demo-level average

            # Calculate per-inference time statistics (only actual model forward passes)
            num_actual_inferences = len(per_inference_times)
            avg_per_inference_time = sum(per_inference_times) / num_actual_inferences if num_actual_inferences > 0 else 0
            log.info(f"Demo {demo_start_idx}-{demo_end_idx}: Avg per-inference time: {avg_per_inference_time:.2f}ms, "
                     f"Actual inferences: {num_actual_inferences}/{total_steps} steps")
            all_per_inference_times.extend(per_inference_times)
            demo_avg_per_inference_times.append(avg_per_inference_time)

            # Episode-level total inference time (sum of actual model forward passes)
            episode_total_infer_time_ms = sum(per_inference_times)

            # --- Trajectory & velocity diagnostic ---
            diag_dir = pathlib.Path(self.output_dir).joinpath("eval", ckpt_name, "vel_diag")
            diag_dir.mkdir(parents=True, exist_ok=True)

            _dec = scenario.decimation
            _pdt = scenario.sim_params.dt if scenario.sim_params.dt is not None else 0.015 / _dec
            ctrl_dt = _dec * _pdt

            has_vel = len(vel_diag_desired) > 0
            has_pos = len(pos_diag_desired) > 0
            n_joints = 0

            if has_vel:
                vel_des_arr = np.array(vel_diag_desired)
                vel_act_arr = np.array(vel_diag_actual)
                n_joints = vel_des_arr.shape[1]
            if has_pos:
                pos_des_arr = np.array(pos_diag_desired)
                pos_act_arr = np.array(pos_diag_actual)
                n_joints = pos_des_arr.shape[1]

            if has_vel or has_pos:
                jnames = [f"joint_{j+1}" for j in range(n_joints)]
                log.info("=" * 60)
                log.info(f"[Trajectory Diagnostic] Demo {demo_start_idx}  (control_dt={ctrl_dt*1000:.1f} ms)")

                if has_vel:
                    vd = vel_des_arr; va = vel_act_arr
                    log.info(f"  --- Velocity (rad/s) ---")
                    log.info(f"  {'Joint':<10}  {'MAE':>10}  {'RMSE':>10}  {'|des| mean':>10}  {'|act| mean':>10}")
                    for j in range(n_joints):
                        mae  = np.abs(vd[:, j] - va[:, j]).mean()
                        rmse = np.sqrt(((vd[:, j] - va[:, j]) ** 2).mean())
                        log.info(f"  {jnames[j]:<10}  {mae:>10.4f}  {rmse:>10.4f}  "
                                 f"{np.abs(vd[:, j]).mean():>10.4f}  {np.abs(va[:, j]).mean():>10.4f}")

                if has_pos:
                    pd = pos_des_arr; pa = pos_act_arr
                    log.info(f"  --- Position (rad) ---")
                    log.info(f"  {'Joint':<10}  {'MAE':>10}  {'RMSE':>10}  {'|des| mean':>10}  {'|act| mean':>10}")
                    for j in range(n_joints):
                        mae  = np.abs(pd[:, j] - pa[:, j]).mean()
                        rmse = np.sqrt(((pd[:, j] - pa[:, j]) ** 2).mean())
                        log.info(f"  {jnames[j]:<10}  {mae:>10.4f}  {rmse:>10.4f}  "
                                 f"{np.abs(pd[:, j]).mean():>10.4f}  {np.abs(pa[:, j]).mean():>10.4f}")
                log.info("=" * 60)

                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    _err_ylims = None
                    if getattr(args, "fixed_error_ylim", True):
                        _eval_ylim_path = pathlib.Path("il_outputs/eval_diag_ylim") / f"{args.task}.json"
                        if _eval_ylim_path.exists():
                            import json as _json
                            with open(_eval_ylim_path) as _f:
                                _err_ylims = _json.load(_f)["pos_err_ylim"]
                        else:
                            _err_ylims = BASELINE_ERROR_YLIM

                    fig_all = make_combined_diag_fig(
                        pd if has_pos else None,
                        pa if has_pos else None,
                        vd if has_vel else None,
                        va if has_vel else None,
                        ctrl_dt, demo_start_idx, n_joints,
                        policy_name=self.policy_name,
                        error_ylims=_err_ylims,
                    )
                    combined_path = diag_dir / f"diag_demo{demo_start_idx}.png"
                    fig_all.savefig(str(combined_path), dpi=150)
                    plt.close(fig_all)
                    log.info(f"[Diagnostic] Combined plot -> {combined_path}")

                    npz_dir = diag_dir / "npz_data"
                    npz_dir.mkdir(parents=True, exist_ok=True)
                    save_kw = {"control_dt": ctrl_dt}
                    if has_vel:
                        save_kw["vel_desired"] = vel_des_arr; save_kw["vel_actual"] = vel_act_arr
                    if has_pos:
                        save_kw["pos_desired"] = pos_des_arr; save_kw["pos_actual"] = pos_act_arr
                    np.savez(str(npz_dir / f"diag_data_demo{demo_start_idx}.npz"), **save_kw)

                except ImportError:
                    log.warning("[Diagnostic] matplotlib not available, skipping plots.")

            SuccessEnd = success.tolist()
            total_success += SuccessOnce.count(True)
            total_completed += len(SuccessOnce)
            for i, demo_idx in enumerate(range(demo_start_idx, demo_end_idx)):
                demo_idx_str = str(demo_idx).zfill(4)
                if i % args.save_video_freq == 0:
                    iio.mimwrite(
                        str(base_eval_dir.joinpath(f"{demo_idx}.mp4")),
                        [images[i] for images in images_list],
                    )
                with open(base_eval_dir.joinpath(f"{demo_idx_str}.txt"), "w") as f:
                    f.write(f"Demo Index: {demo_idx}\n")
                    f.write(f"Num Envs: {num_envs}\n")
                    f.write(f"SuccessOnce: {SuccessOnce[i]}\n")
                    f.write(f"SuccessEnd: {SuccessEnd[i]}\n")
                    f.write(f"TimeOut: {TimeOut[i]}\n")
                    f.write(f"Domain Randomization Level: {dr_level}\n")
                    f.write(f"Domain Randomization Scene Mode: {dr_scene_mode}\n")
                    f.write(f"Domain Randomization Seed: {dr_seed}\n")
                    f.write(
                        f"Cumulative Average Success Rate: {total_success / total_completed:.4f}\n"
                    )
                    # Add inference time statistics
                    f.write(f"\n--- Inference Time Statistics ---\n")
                    f.write(f"Total Steps: {total_steps}\n")
                    f.write(f"Average Inference Time: {avg_inference_time:.2f}ms\n")
                    f.write(f"Min Inference Time: {min_inference_time:.2f}ms\n")
                    f.write(f"Max Inference Time: {max_inference_time:.2f}ms\n")
                    f.write(f"\n--- Per-Inference Time Statistics (model forward pass only) ---\n")
                    f.write(f"Actual Inferences: {num_actual_inferences}\n")
                    f.write(f"Average Per-Inference Time: {avg_per_inference_time:.2f}ms\n")
                    f.write(f"\n--- Episode Duration ---\n")
                    f.write(f"Episode Duration: {episode_duration_ms:.2f}ms\n")
                    f.write(f"Episode Total Inference Time: {episode_total_infer_time_ms:.2f}ms\n")

            # Collect durations for successful episodes
            for i in range(len(SuccessOnce)):
                if SuccessOnce[i]:
                    success_episode_durations.append(episode_duration_ms)
                    success_episode_total_infer_times.append(episode_total_infer_time_ms)

            log.info("Demo Indices: ", range(demo_start_idx, demo_end_idx))
            log.info("Num Envs: ", num_envs)
            log.info(f"SuccessOnce: {SuccessOnce}")
            log.info(f"SuccessEnd: {SuccessEnd}")
            log.info(f"TimeOut: {TimeOut}")
        # Calculate overall inference time statistics
        overall_total_steps = len(all_inference_times)
        overall_avg_inference_time = sum(all_inference_times) / overall_total_steps if overall_total_steps > 0 else 0
        overall_min_inference_time = min(all_inference_times) if all_inference_times else 0
        overall_max_inference_time = max(all_inference_times) if all_inference_times else 0
        
        # Calculate STD of demo-level average inference times
        num_demos_evaluated = len(demo_avg_inference_times)
        if num_demos_evaluated > 1:
            demo_avg_mean = sum(demo_avg_inference_times) / num_demos_evaluated
            demo_avg_variance = sum((x - demo_avg_mean) ** 2 for x in demo_avg_inference_times) / (num_demos_evaluated - 1)
            demo_avg_std = demo_avg_variance ** 0.5
        else:
            demo_avg_std = 0.0
        
        # Calculate overall per-inference time statistics (actual model forward passes only)
        overall_total_inferences = len(all_per_inference_times)
        overall_avg_per_inference_time = sum(all_per_inference_times) / overall_total_inferences if overall_total_inferences > 0 else 0
        overall_min_per_inference_time = min(all_per_inference_times) if all_per_inference_times else 0
        overall_max_per_inference_time = max(all_per_inference_times) if all_per_inference_times else 0

        # Calculate STD of demo-level average per-inference times
        if len(demo_avg_per_inference_times) > 1:
            pi_mean = sum(demo_avg_per_inference_times) / len(demo_avg_per_inference_times)
            pi_variance = sum((x - pi_mean) ** 2 for x in demo_avg_per_inference_times) / (len(demo_avg_per_inference_times) - 1)
            pi_std = pi_variance ** 0.5
        else:
            pi_std = 0.0

        # Calculate average episode duration and total inference time for successful episodes only
        num_success_episodes = len(success_episode_durations)
        avg_success_episode_duration = sum(success_episode_durations) / num_success_episodes if num_success_episodes > 0 else 0
        avg_success_episode_total_infer = sum(success_episode_total_infer_times) / num_success_episodes if num_success_episodes > 0 else 0

        log.info(f"FINAL RESULTS: Average Success Rate = {total_success / total_completed:.4f}")
        log.info(f"FINAL RESULTS: Overall Avg Inference Time = {overall_avg_inference_time:.2f}ms (STD across demos: {demo_avg_std:.2f}ms), "
                 f"Min: {overall_min_inference_time:.2f}ms, Max: {overall_max_inference_time:.2f}ms, "
                 f"Total Steps: {overall_total_steps}")
        log.info(f"FINAL RESULTS: Overall Avg Per-Inference Time = {overall_avg_per_inference_time:.2f}ms (STD across demos: {pi_std:.2f}ms), "
                 f"Min: {overall_min_per_inference_time:.2f}ms, Max: {overall_max_per_inference_time:.2f}ms, "
                 f"Total Actual Inferences: {overall_total_inferences}")
        
        with open(base_eval_dir.joinpath("00_final_stats.txt"), "w") as f:
            f.write(f"=== Success Statistics ===\n")
            f.write(f"Total Success: {total_success}\n")
            f.write(f"Total Completed: {total_completed}\n")
            f.write(f"Average Success Rate: {total_success / total_completed:.4f}\n")
            f.write(f"\n=== Domain Randomization ===\n")
            f.write(f"Domain Randomization Level: {dr_level}\n")
            f.write(f"Domain Randomization Scene Mode: {dr_scene_mode}\n")
            f.write(f"Domain Randomization Seed: {dr_seed}\n")
            f.write(f"\n=== Overall Inference Time Statistics ===\n")
            f.write(f"Total Inference Steps: {overall_total_steps}\n")
            f.write(f"Number of Demos Evaluated: {num_demos_evaluated}\n")
            f.write(f"Average Inference Time: {overall_avg_inference_time:.2f}ms\n")
            f.write(f"STD of Demo Avg Inference Time: {demo_avg_std:.2f}ms\n")
            f.write(f"Min Inference Time: {overall_min_inference_time:.2f}ms\n")
            f.write(f"Max Inference Time: {overall_max_inference_time:.2f}ms\n")
            f.write(f"\n=== Overall Per-Inference Time Statistics (model forward pass only) ===\n")
            f.write(f"Total Actual Inferences: {overall_total_inferences}\n")
            f.write(f"Number of Demos Evaluated: {num_demos_evaluated}\n")
            f.write(f"Average Per-Inference Time: {overall_avg_per_inference_time:.2f}ms\n")
            f.write(f"STD of Demo Avg Per-Inference Time: {pi_std:.2f}ms\n")
            f.write(f"Min Per-Inference Time: {overall_min_per_inference_time:.2f}ms\n")
            f.write(f"Max Per-Inference Time: {overall_max_per_inference_time:.2f}ms\n")
            f.write(f"\n=== Successful Episodes Duration (successful episodes only) ===\n")
            f.write(f"Number of Successful Episodes: {num_success_episodes}\n")
            f.write(f"Avg Episode Duration (success only): {avg_success_episode_duration:.2f}ms\n")
            f.write(f"\n=== Successful Episodes Total Inference Time (successful episodes only) ===\n")
            f.write(f"Number of Successful Episodes: {num_success_episodes}\n")
            f.write(f"Avg Episode Total Inference Time (success only): {avg_success_episode_total_infer:.2f}ms\n")

        # Rename eval directory: embed success rate and avg inference time after checkpoint stem
        # e.g. "15000_ds4_step6.ckpt_2026-..." -> "15000_sr86_3.80ms_ds4_step6.ckpt_..."
        try:
            sr = total_success / total_completed
            infer_tag = f"{avg_success_episode_total_infer:.2f}ms"
            dir_name = base_eval_dir.name
            ckpt_prefix = dir_name.split(".ckpt_", 1)
            if len(ckpt_prefix) == 2 and "_sr" not in ckpt_prefix[0]:
                sr_tag = "100" if sr >= 1.0 else f"{int(sr * 100):02d}"
                # Runs with finite-difference velocity feedforward carry an
                # explicit "fdvel" tag so they are distinguishable on disk.
                fd_tag = "_fdvel" if fd_vel else ""
                prefix = ckpt_prefix[0]
                # Insert _sr{tag}[_fdvel]_{infer_tag} right after the first underscore-separated segment
                parts = prefix.split("_", 1)
                if len(parts) == 2:
                    prefix = f"{parts[0]}_sr{sr_tag}{fd_tag}_{infer_tag}_{parts[1]}"
                else:
                    prefix = f"{parts[0]}_sr{sr_tag}{fd_tag}_{infer_tag}"
                new_name = f"{prefix}.ckpt_{ckpt_prefix[1]}"
                new_dir = base_eval_dir.parent / new_name
                base_eval_dir.rename(new_dir)
                log.info(f"Eval dir renamed: {dir_name} -> {new_name}")
        except Exception as e:
            log.warning(f"Failed to rename eval dir: {e}")

        # env.close() often hangs due to Isaac Sim background threads.
        # Schedule a forced exit so the process releases GPU memory even
        # if env.close() never returns.
        import threading, sys
        def _force_exit():
            log.info("Force-exiting to release GPU memory (env.close timed out).")
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except Exception:
                pass
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        _exit_timer = threading.Timer(30.0, _force_exit)
        _exit_timer.daemon = True
        _exit_timer.start()

        env.close()

        _exit_timer.cancel()

    def run(
        self,
        train=None,
        eval=None,
        ckpt_path=None,
    ):
        train = self.cfg.train_enable
        eval = self.cfg.eval_enable
        # Always use eval_path if provided (respects num_epochs setting)
        ckpt_path = self.cfg.eval_path
        if train:
            self.train()
        if eval:
            self.evaluate(ckpt_path=ckpt_path)


class BatchSampler:
    def __init__(
        self,
        data_size: int,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = True,
    ):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batch = data_size // batch_size
        self.discard = data_size - batch_size * self.num_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None

    def __iter__(self):
        if self.shuffle:
            perm = self.rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
        if self.discard > 0:
            perm = perm[: -self.discard]
        perm = perm.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i]

    def __len__(self):
        return self.num_batch


def create_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int = 0,
):
    # print("create_dataloader_batch_size", batch_size)
    batch_sampler = BatchSampler(
        len(dataset), batch_size, shuffle=shuffle, seed=seed, drop_last=True
    )

    def collate(x):
        assert len(x) == 1
        return x[0]

    dataloader = DataLoader(
        dataset,
        collate_fn=collate,
        sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
    )
    return dataloader


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = DefaultRunner(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
