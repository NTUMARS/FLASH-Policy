"""Global-spline-projected Legendre coefficients for FLASH-G training.

Instead of fitting each sliding window independently via least-squares on H
discrete points, this module:

  1. Fits a C2-continuous cubic interpolating spline to the *entire* episode.
  2. For each sliding window, densely samples the spline segment.
  3. Projects the dense samples onto the Legendre basis via least-squares,
     producing coefficients that are globally consistent across windows.

This eliminates the boundary discontinuities that arise when adjacent windows
are fitted independently.
"""

from typing import List, Optional

import numpy as np
import torch
from scipy.interpolate import CubicSpline


# ── Legendre basis (mirrors flash_g_policy._build_legendre_basis) ──

def _build_legendre_basis_np(s: np.ndarray, degree: int) -> np.ndarray:
    """Build shifted Legendre basis on s in [0, 1].  Returns eval_mat (H, C)."""
    x = 2.0 * s - 1.0
    H = len(s)
    C = degree + 1
    P = np.zeros((H, C), dtype=np.float64)
    P[:, 0] = 1.0
    if C > 1:
        P[:, 1] = x
    for n in range(1, C - 1):
        P[:, n + 1] = ((2 * n + 1) * x * P[:, n] - n * P[:, n - 1]) / (n + 1)
    return P


# ── Spline fitting ──────────────────────────────────────────────────────

def fit_episode_splines(
    actions: np.ndarray,
    ep_starts: np.ndarray,
    ep_ends: np.ndarray,
) -> List[Optional[CubicSpline]]:
    """Fit a CubicSpline per episode over all action dimensions jointly.

    Args:
        actions: (T_total, Da) – the full concatenated action array.
        ep_starts: (num_episodes,) – start index of each episode in *actions*.
        ep_ends:   (num_episodes,) – end index (exclusive) of each episode.

    Returns:
        List of CubicSpline objects (one per episode).
        Each spline maps integer timestep t in [0, ep_len-1] → (Da,).
    """
    splines: List[Optional[CubicSpline]] = []
    for s, e in zip(ep_starts, ep_ends):
        ep_len = int(e - s)
        if ep_len < 4:
            splines.append(None)
            continue
        t = np.arange(ep_len, dtype=np.float64)
        ep_data = actions[int(s):int(e)].astype(np.float64)
        splines.append(CubicSpline(t, ep_data, bc_type="not-a-knot"))
    return splines


# ── Spline → Legendre projection ────────────────────────────────────────

def _project_segment_to_legendre(
    spline: CubicSpline,
    t_start: float,
    t_end: float,
    poly_order: int,
    n_dense: int,
    clip_lo: float | None = None,
    clip_hi: float | None = None,
) -> np.ndarray:
    """Sample *spline* densely on [t_start, t_end] and fit Legendre coeffs.

    When *clip_lo* / *clip_hi* are given, the spline evaluation times are
    clipped to that range (constant extrapolation, same effect as the
    edge-padding used by the standard least-squares fitting).

    Returns:
        raw_coeff: (C, Da) – Legendre polynomial coefficients.
    """
    t_dense = np.linspace(t_start, t_end, n_dense, dtype=np.float64)
    if clip_lo is not None or clip_hi is not None:
        t_dense = np.clip(
            t_dense,
            clip_lo if clip_lo is not None else t_dense[0],
            clip_hi if clip_hi is not None else t_dense[-1],
        )
    samples = spline(t_dense)  # (n_dense, Da)

    s_dense = np.linspace(0.0, 1.0, n_dense, dtype=np.float64)
    E = _build_legendre_basis_np(s_dense, poly_order)  # (n_dense, C)

    StS = E.T @ E                                # (C, C)
    raw_coeff = np.linalg.solve(StS, E.T @ samples)  # (C, Da)
    return raw_coeff


# ── Batch pre-computation ────────────────────────────────────────────────

def _find_episode_for_index(buffer_start: int, ep_ends: np.ndarray) -> int:
    """Return the episode index that contains *buffer_start*."""
    return int(np.searchsorted(ep_ends, buffer_start, side="right"))


def precompute_all_spline_coefficients(
    replay_buffer,
    normalizer,
    sampler_indices: np.ndarray,
    poly_order: int,
    fit_pad: int,
    horizon: int,
    coeff_scale: np.ndarray,
    n_dense: int = 200,
) -> np.ndarray:
    """Pre-compute spline-projected, normalised Legendre coefficients.

    The coefficient normalisation exactly matches the training pipeline:
        norm_coeff = raw_coeff / coeff_scale

    where *coeff_scale* is ``model._coeff_scale`` squeezed to shape (C,).

    Args:
        replay_buffer: The dataset's replay buffer (with ``episode_ends``
            and ``"action"`` array).
        normalizer: A fitted ``LinearNormalizer`` (must have ``"action"`` key).
        sampler_indices: (N, 4) array produced by ``create_indices``.
        poly_order: Legendre polynomial degree.
        fit_pad: Extended-window padding (0 = standard).
        horizon: Base horizon H (before fit_pad extension).
        coeff_scale: (C,) array – per-coefficient scale factors.
        n_dense: Number of dense sample points for spline projection.

    Returns:
        norm_coeffs: (N, C, Da) float32 array of normalised coefficients.
    """
    ep_ends = replay_buffer.episode_ends[:]
    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    actions_raw = replay_buffer["action"][:]              # (T_total, Da)

    action_norm = normalizer["action"]
    actions_normed = action_norm.normalize(
        torch.from_numpy(actions_raw).float()
    ).detach().numpy().astype(np.float64)                 # (T_total, Da)

    splines = fit_episode_splines(actions_normed, ep_starts, ep_ends)

    N = len(sampler_indices)
    C = poly_order + 1
    Da = actions_normed.shape[1]
    H_ext = horizon + 2 * fit_pad                         # extended window
    coeff_scale_2d = coeff_scale[:, None]                 # (C, 1)

    norm_coeffs = np.zeros((N, C, Da), dtype=np.float32)

    for i in range(N):
        buf_start = int(sampler_indices[i, 0])
        ep_idx = _find_episode_for_index(buf_start, ep_ends)
        ep_start = int(ep_starts[ep_idx])
        ep_end = int(ep_ends[ep_idx])
        ep_len = ep_end - ep_start

        spline = splines[ep_idx]
        if spline is None:
            continue

        # The window in absolute buffer coords → local episode coords
        local_start = buf_start - ep_start
        # The sample may have been padded by the sampler; reconstruct the
        # original window boundaries that *would* include H_ext steps.
        sample_start_idx = int(sampler_indices[i, 2])
        pad_leading = sample_start_idx
        win_local_start = local_start - pad_leading

        # Use the full (unclamped) window range so s ∈ [0,1] maps
        # consistently to H_ext time steps.  Clip the spline evaluation
        # to the episode domain (constant extrapolation, like edge-padding).
        t_start = float(win_local_start)
        t_end = float(win_local_start + H_ext - 1)

        if t_end <= t_start:
            continue

        raw_coeff = _project_segment_to_legendre(
            spline, t_start, t_end, poly_order, n_dense,
            clip_lo=0.0, clip_hi=float(ep_len - 1),
        )
        norm_coeffs[i] = (raw_coeff / coeff_scale_2d).astype(np.float32)

    return norm_coeffs
