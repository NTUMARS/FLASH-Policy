"""
Visualization utilities for imitation learning policies.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from scipy import stats
import pathlib
from typing import Optional


# ---------------------------------------------------------------------------
# Baseline y-limits for the position-error column so that plots from
# different models / checkpoints are visually comparable.
# ---------------------------------------------------------------------------
BASELINE_ERROR_YLIM: list[tuple[float, float]] = [
    (-0.043, 0.043),   # J1  panda_finger_joint1  (alphabetical order)
    (-0.034, 0.031),   # J2  panda_finger_joint2
    (-0.708, 0.618),   # J3  panda_joint1
    (-0.793, 0.525),   # J4  panda_joint2
    (-0.485, 0.607),   # J5  panda_joint3
    (-1.085, 0.263),   # J6  panda_joint4
    (-0.590, 0.438),   # J7  panda_joint5
    (-0.700, 0.700),   # J8  panda_joint6
    (-1.200, 1.200),   # J9  panda_joint7
]


def make_combined_diag_fig(
    pos_des: Optional[np.ndarray],
    pos_act: Optional[np.ndarray],
    vel_des: Optional[np.ndarray],
    vel_act: Optional[np.ndarray],
    ctrl_dt: float,
    demo_idx: int,
    n_joints: int,
    policy_name: str = "unknown",
    error_ylims: Optional[list[tuple[float, float]]] = None,
):
    """Create a combined diagnostic figure (Position / Velocity / Position-Error).

    Returns a *matplotlib.figure.Figure* that the caller should save and close.
    """
    has_p = pos_des is not None
    has_v = vel_des is not None
    cols: list = []
    if has_p:
        cols.append(("Position (rad)", pos_des, pos_act))
    if has_v:
        cols.append(("Velocity (rad/s)", vel_des, vel_act))
    cols.append(None)  # placeholder for the error column

    ncols = len(cols)
    fig, axes = plt.subplots(
        n_joints, ncols,
        figsize=(7 * ncols, 2.2 * n_joints),
        sharex=True, squeeze=False,
    )

    color_des = "#1f77b4"
    color_act = "#ff7f0e"
    color_err_pos = "#d62728"

    desired_label = f"desired ({policy_name})"

    col_idx = 0
    err_data_list: list = []

    for label, des, act in (c for c in cols if c is not None):
        t = np.arange(len(des)) * ctrl_dt
        err = des[:, :n_joints] - act[:, :n_joints]
        err_data_list.append((label, t, err))
        for j in range(n_joints):
            ax = axes[j, col_idx]
            ax.plot(t, des[:, j], label=desired_label,
                    linewidth=1.1, color=color_des)
            ax.plot(t, act[:, j], label="actual (sim)",
                    linewidth=1.1, color=color_act, alpha=0.85)
            ax.set_ylabel(f"J{j+1}", fontsize=8)
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_title(label, fontsize=10)
                ax.legend(fontsize=6, loc="upper right")
        axes[-1, col_idx].set_xlabel("Time (s)", fontsize=9)
        col_idx += 1

    # --- Position error column ---
    err_col = col_idx
    pos_err_entry = next(
        ((lbl, t, err) for lbl, t, err in err_data_list if "Position" in lbl),
        None,
    )
    if pos_err_entry is None and err_data_list:
        pos_err_entry = err_data_list[0]
    for j in range(n_joints):
        ax = axes[j, err_col]
        if pos_err_entry is not None:
            lbl, t, err = pos_err_entry
            ax.plot(t, err[:, j], label="pos err",
                    linewidth=1.0, color=color_err_pos, alpha=0.85)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel(f"J{j+1}", fontsize=8)
        ax.grid(True, alpha=0.3)
        if error_ylims is not None and j < len(error_ylims):
            ax.set_ylim(error_ylims[j])
        if j == 0:
            ax.set_title("Position Error (desired \u2212 actual)", fontsize=10)
            ax.legend(fontsize=6, loc="upper right")
    axes[-1, err_col].set_xlabel("Time (s)", fontsize=9)

    fig.suptitle(
        f"Trajectory Diagnostic \u2014 Demo {demo_idx}",
        fontsize=13, y=1.0,
    )
    fig.tight_layout()
    return fig


def get_truncated_viridis(minval=0.0, maxval=0.7, n=256):
    """
    Get a truncated viridis colormap that excludes the bright yellow portion.
    
    Args:
        minval: Start of the colormap range (0.0 = dark purple)
        maxval: End of the colormap range (1.0 = bright yellow, 0.7 = teal/cyan)
        n: Number of colors
    
    Returns:
        Truncated colormap (dark purple → teal, no yellow)
    """
    viridis = plt.cm.get_cmap('viridis')
    colors = viridis(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list('truncated_viridis', colors)


def plot_latent_paired_tsne(
    history_latents: np.ndarray,
    future_latents: np.ndarray,
    epoch: int,
    save_path: str,
    perplexity: int = 30,
    random_state: int = 42,
    fixed_colorbar_max: float = 3,
):
    """
    Generate t-SNE visualization with lines connecting paired history-future latents.
    
    Args:
        history_latents: (N, D) array of history state latents
        future_latents: (N, D) array of future action latents
        epoch: Current training epoch (for title)
        save_path: Path to save the figure
        fixed_colorbar_max: Fixed maximum value for colorbar (default: 60)
    """
    # Set font (DejaVu Sans is available on most Linux systems)
    # plt.rcParams['font.family'] = 'Times New Roman'
    
    # Ensure inputs are numpy arrays
    if hasattr(history_latents, 'cpu'):
        history_latents = history_latents.cpu().numpy()
    if hasattr(future_latents, 'cpu'):
        future_latents = future_latents.cpu().numpy()
    
    n_samples = history_latents.shape[0]
    
    # Calculate distance in ORIGINAL latent space (consistent across epochs)
    latent_space_distances = np.linalg.norm(history_latents - future_latents, axis=1)
    avg_latent_distance = np.mean(latent_space_distances)
    
    # Combine latents for joint t-SNE
    combined_latents = np.concatenate([history_latents, future_latents], axis=0)
    
    # Adjust perplexity
    effective_perplexity = min(perplexity, n_samples - 1, combined_latents.shape[0] // 3)
    effective_perplexity = max(5, effective_perplexity)
    
    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=random_state,
    )
    embedded = tsne.fit_transform(combined_latents)
    
    # Split back into history and future
    history_embedded = embedded[:n_samples]
    future_embedded = embedded[n_samples:]
    
    # Create figure (larger to accommodate 2x font size)
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Create truncated viridis colormap (exclude yellow part: use 0-0.7 range)
    viridis_truncated = get_truncated_viridis(minval=0.0, maxval=0.7)
    
    # Draw lines connecting paired points (with color based on LATENT SPACE distance)
    for i in range(n_samples):
        # Color: dark purple (close) to green (far), using fixed range
        normalized_dist = min(latent_space_distances[i] / fixed_colorbar_max, 1.0)
        color = viridis_truncated(normalized_dist)  # Dark for short, bright for long
        ax.plot(
            [history_embedded[i, 0], future_embedded[i, 0]],
            [history_embedded[i, 1], future_embedded[i, 1]],
            c=color, alpha=0.7, linewidth=1.5, zorder=1
        )
    
    # Plot points on top (larger size)
    ax.scatter(
        history_embedded[:, 0], history_embedded[:, 1],
        c='#3498db', alpha=0.9, s=120, label='History Latents',
        edgecolors='white', linewidths=1.0, zorder=2
    )
    ax.scatter(
        future_embedded[:, 0], future_embedded[:, 1],
        c='#e74c3c', alpha=0.9, s=120, label='Future Latents',
        edgecolors='white', linewidths=1.0, zorder=2
    )
    
    # Styling (2x larger fonts, title unchanged)
    # ax.set_xlabel('t-SNE Dimension 1', fontsize=56)
    # ax.set_ylabel('t-SNE Dimension 2', fontsize=56)
    # ax.set_title(f'A2A Paired Latent Space (Epoch {epoch})', fontsize=32)
    ax.legend(loc='lower right', fontsize=48, markerscale=2.0)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=44)
    
    # Add annotations (2x larger font) - use LATENT SPACE distance for consistency
    ax.text(
        0.02, 0.98, f'Latent Space Distance = {avg_latent_distance:.2f}',
        transform=ax.transAxes, fontsize=48,
        verticalalignment='top', # fontfamily='Times New Roman',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
    )
    
    # Add colorbar for line distances (FIXED range, full height, truncated viridis)
    sm = plt.cm.ScalarMappable(cmap=viridis_truncated, norm=plt.Normalize(0, fixed_colorbar_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=1.0, pad=0.02, aspect=30)
    cbar.set_label('Latent Distance', fontsize=48)
    cbar.ax.tick_params(labelsize=36)
    
    # Ensure save directory exists
    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return str(save_path), avg_latent_distance


def plot_density_contour(
    history_latents: np.ndarray,
    future_latents: np.ndarray,
    epoch: int,
    save_path: str,
    perplexity: int = 30,
    random_state: int = 42,
):
    """
    Generate density contour plot showing overlap of history and future latent distributions.
    
    Args:
        history_latents: (N, D) array of history state latents
        future_latents: (N, D) array of future action latents
        epoch: Current training epoch (for title)
        save_path: Path to save the figure
    """
    # Ensure inputs are numpy arrays
    if hasattr(history_latents, 'cpu'):
        history_latents = history_latents.cpu().numpy()
    if hasattr(future_latents, 'cpu'):
        future_latents = future_latents.cpu().numpy()
    
    n_samples = history_latents.shape[0]
    
    # Combine latents for joint t-SNE (same embedding for fair comparison)
    combined_latents = np.concatenate([history_latents, future_latents], axis=0)
    
    # Adjust perplexity
    effective_perplexity = min(perplexity, n_samples - 1, combined_latents.shape[0] // 3)
    effective_perplexity = max(5, effective_perplexity)
    
    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=random_state,
    )
    embedded = tsne.fit_transform(combined_latents)
    
    # Split back into history and future
    history_embedded = embedded[:n_samples]
    future_embedded = embedded[n_samples:]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create grid for density estimation
    x_min = min(history_embedded[:, 0].min(), future_embedded[:, 0].min()) - 1
    x_max = max(history_embedded[:, 0].max(), future_embedded[:, 0].max()) + 1
    y_min = min(history_embedded[:, 1].min(), future_embedded[:, 1].min()) - 1
    y_max = max(history_embedded[:, 1].max(), future_embedded[:, 1].max()) + 1
    
    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # Compute KDE for history
    try:
        history_kernel = stats.gaussian_kde(history_embedded.T)
        history_density = np.reshape(history_kernel(positions).T, xx.shape)
    except np.linalg.LinAlgError:
        history_density = np.zeros(xx.shape)
    
    # Compute KDE for future
    try:
        future_kernel = stats.gaussian_kde(future_embedded.T)
        future_density = np.reshape(future_kernel(positions).T, xx.shape)
    except np.linalg.LinAlgError:
        future_density = np.zeros(xx.shape)
    
    # Normalize densities
    if history_density.max() > 0:
        history_density = history_density / history_density.max()
    if future_density.max() > 0:
        future_density = future_density / future_density.max()
    
    # Plot density contours
    levels = np.linspace(0.1, 1.0, 6)
    
    # History contours (blue)
    cs_history = ax.contour(xx, yy, history_density, levels=levels, 
                            colors='#3498db', linewidths=2, alpha=0.8)
    ax.contourf(xx, yy, history_density, levels=levels, 
                colors=['#3498db'], alpha=0.15)
    
    # Future contours (red)
    cs_future = ax.contour(xx, yy, future_density, levels=levels, 
                           colors='#e74c3c', linewidths=2, alpha=0.8)
    ax.contourf(xx, yy, future_density, levels=levels, 
                colors=['#e74c3c'], alpha=0.15)
    
    # Add scatter points with low alpha for reference
    ax.scatter(history_embedded[:, 0], history_embedded[:, 1],
               c='#3498db', alpha=0.3, s=15, label='History Latents')
    ax.scatter(future_embedded[:, 0], future_embedded[:, 1],
               c='#e74c3c', alpha=0.3, s=15, label='Future Latents')
    
    # Calculate overlap metric (intersection over union of density regions)
    threshold = 0.2
    history_mask = history_density > threshold
    future_mask = future_density > threshold
    intersection = np.sum(history_mask & future_mask)
    union = np.sum(history_mask | future_mask)
    overlap_ratio = intersection / union if union > 0 else 0
    
    # Styling
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    # ax.set_title(f'A2A Latent Distribution Density (Epoch {epoch})\nContour overlap indicates distribution similarity', 
    #              fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add overlap annotation
    ax.text(
        0.02, 0.98, f'N = {n_samples} samples\nDistribution Overlap = {overlap_ratio:.1%}',
        transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    )
    
    # Ensure save directory exists
    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return str(save_path), overlap_ratio


def plot_flow_trajectories(
    trajectories: list,
    future_latents: np.ndarray,
    epoch: int,
    save_path: str,
    perplexity: int = 30,
    random_state: int = 42,
):
    """
    Plot flow trajectories from history to future latents.
    
    Args:
        trajectories: List of (num_steps+1, latent_dim) arrays, one per sample
        future_latents: (n_samples, latent_dim) - ground truth targets
        epoch: Current training epoch
        save_path: Path to save the figure
    """
    # Set font (DejaVu Sans is available on most Linux systems)
    # plt.rcParams['font.family'] = 'Times New Roman'
    
    n_samples = len(trajectories)
    num_steps = trajectories[0].shape[0] - 1  # Exclude initial point
    latent_dim = trajectories[0].shape[1]
    
    # Calculate distance in ORIGINAL latent space (not t-SNE space)
    # This gives a consistent metric across epochs
    flow_end_latents = np.array([traj[-1] for traj in trajectories])  # (n_samples, latent_dim)
    latent_space_dist = np.mean(np.linalg.norm(flow_end_latents - future_latents, axis=1))
    
    # Combine all trajectory points and targets for joint t-SNE
    all_points = []
    point_labels = []  # 0=start, 1=intermediate, 2=end, 3=target
    
    for i, traj in enumerate(trajectories):
        for step, point in enumerate(traj):
            all_points.append(point)
            if step == 0:
                point_labels.append(0)  # Start (history)
            elif step == len(traj) - 1:
                point_labels.append(2)  # End (flow result)
            else:
                point_labels.append(1)  # Intermediate
    
    # Add ground truth targets
    for target in future_latents:
        all_points.append(target)
        point_labels.append(3)  # Target (ground truth)
    
    all_points = np.array(all_points)
    point_labels = np.array(point_labels)
    
    # Run t-SNE
    effective_perplexity = min(perplexity, len(all_points) // 3)
    effective_perplexity = max(5, effective_perplexity)
    
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=random_state,
    )
    embedded = tsne.fit_transform(all_points)
    
    # Create figure (larger to accommodate 2x font size)
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Colors for trajectory lines (different color per sample)
    colors = plt.cm.Set1(np.linspace(0, 1, n_samples))
    
    # Plot trajectories
    idx = 0
    for i in range(n_samples):
        traj_len = len(trajectories[i])
        traj_embedded = embedded[idx:idx + traj_len]
        
        # Draw trajectory line with arrows (thicker)
        for j in range(traj_len - 1):
            ax.annotate('', 
                xy=traj_embedded[j + 1], 
                xytext=traj_embedded[j],
                arrowprops=dict(arrowstyle='->', color=colors[i], lw=2.5, alpha=0.8))
        
        # Plot intermediate points (larger)
        if traj_len > 2:
            ax.scatter(traj_embedded[1:-1, 0], traj_embedded[1:-1, 1],
                       c=[colors[i]], s=80, alpha=0.6, marker='o')
        
        idx += traj_len
    
    # Plot start points (history) - larger
    start_mask = point_labels == 0
    ax.scatter(embedded[start_mask, 0], embedded[start_mask, 1],
               c='#3498db', s=300, marker='o', label='Start (History)',
               edgecolors='black', linewidths=2.0, zorder=5)
    
    # Plot end points (flow result) - larger
    end_mask = point_labels == 2
    ax.scatter(embedded[end_mask, 0], embedded[end_mask, 1],
               c='#27ae60', s=300, marker='s', label='End (Flow Result)',
               edgecolors='black', linewidths=2.0, zorder=5)
    
    # Plot target points (ground truth) - larger
    target_mask = point_labels == 3
    ax.scatter(embedded[target_mask, 0], embedded[target_mask, 1],
               c='#e74c3c', s=400, marker='*', label='Target (Ground Truth)',
               edgecolors='black', linewidths=1.5, zorder=5)
    
    # Draw lines from flow end to ground truth target
    end_embedded = embedded[end_mask]
    target_embedded = embedded[target_mask]
    for i in range(n_samples):
        ax.plot([end_embedded[i, 0], target_embedded[i, 0]],
                [end_embedded[i, 1], target_embedded[i, 1]],
                'k--', alpha=0.4, linewidth=2)
    
    # Styling (2x larger fonts, title unchanged)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=56)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=56)
    ax.set_title(f'A2A Flow Trajectories (Epoch {epoch})', fontsize=32)
    ax.legend(loc='lower right', fontsize=48, markerscale=1.5)
    ax.tick_params(axis='both', labelsize=44)
    ax.grid(True, alpha=0.3)
    
    # Add annotation (2x larger font) - use LATENT SPACE distance, not t-SNE distance
    ax.text(
        0.02, 0.98, 
        f'{n_samples} samples, {num_steps} flow steps\n'
        f'Latent Space Distance = {latent_space_dist:.2f}',
        transform=ax.transAxes, fontsize=48,
        verticalalignment='top', # fontfamily='Times New Roman',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    )
    
    # Ensure save directory exists
    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return str(save_path), latent_space_dist


def plot_all_latent_visualizations(
    history_latents: np.ndarray,
    future_latents: np.ndarray,
    epoch: int,
    save_dir: str,
    perplexity: int = 30,
    random_state: int = 42,
    trajectories: list | None = None,
    trajectory_targets: np.ndarray | None = None,
):
    """
    Generate all visualization types for A2A latent analysis.
    
    Args:
        history_latents: (N, D) array of history state latents
        future_latents: (N, D) array of future action latents
        epoch: Current training epoch
        save_dir: Directory to save figures
        trajectories: Optional list of flow trajectories for visualization
        trajectory_targets: Optional ground truth targets for trajectories
        
    Returns:
        Dictionary with paths to all generated figures and metrics
    """
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Paired t-SNE with connecting lines
    paired_path = save_dir / f"epoch_{epoch:02d}_paired_tsne.png"
    _, avg_distance = plot_latent_paired_tsne(
        history_latents, future_latents, epoch, str(paired_path),
        perplexity=perplexity, random_state=random_state
    )
    results['paired_tsne'] = str(paired_path)
    results['avg_tsne_distance'] = avg_distance
    
    # 2. Flow trajectories (if provided)
    if trajectories is not None and trajectory_targets is not None:
        traj_path = save_dir / f"epoch_{epoch:02d}_flow_trajectories.png"
        _, end_to_target_dist = plot_flow_trajectories(
            trajectories, trajectory_targets, epoch, str(traj_path),
            perplexity=perplexity, random_state=random_state
        )
        results['flow_trajectories'] = str(traj_path)
        results['flow_end_to_target_dist'] = end_to_target_dist
    
    return results
