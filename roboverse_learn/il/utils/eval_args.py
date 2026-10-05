from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from loguru import logger as log


@dataclass
class Args:
    task: str
    """Task name"""
    robot: str = "franka"
    """Robot name"""
    num_envs: int = 1
    """Number of parallel environments, find a proper number for best performance on your machine"""
    sim: Literal["isaaclab", "mujoco", "isaacgym"] = "isaacsim"
    """Simulator backend"""
    max_demo: int | None = None
    """Maximum number of demos to collect, None for all demos"""
    headless: bool = True
    """Run in headless mode"""
    table: bool = True
    """Try to add a table"""
    task_id_range_low: int = 0
    """Low end of the task id range"""
    task_id_range_high: int = 1000
    """High end of the task id range"""
    subset: str = "pickcube_l0"
    """Subset your ckpt trained on"""
    action_set_steps: int = 1
    """Number of steps to take for each action set"""
    save_video_freq: int = 1
    """Frequency of saving videos"""
    max_step: int = 250
    """Maximum number of steps to collect"""
    gpu_id: int = 0
    """GPU ID to use"""
    downsample_ratio: int = 1
    """Training data downsample ratio. Must match the ratio used in data2zarr_dp.py.
    FLASH-G always resamples its polynomial at the simulator's native control frequency
    (66.67 Hz), decoupled from the training data frequency (30 Hz / downsample_ratio)."""

    # Domain Randomization options
    level: Literal[0, 1, 2, 3] = 0
    """Randomization level: 0=None, 1=Scene+Material, 2=+Light, 3=+Camera"""
    scene_mode: Literal[0, 1, 2, 3] = 0
    """Scene mode: 0=Manual, 1=USD Table, 2=USD Scene, 3=Full USD"""
    randomization_seed: int | None = None
    """Seed for reproducible randomization. If None, uses random seed"""

    # Velocity-feedforward PD gain tuning
    velocity_pd_gains: bool = False
    """Enable velocity-mode PD gain overrides (custom Kp/Kd for each joint).
    When True, applies VELOCITY_MODE_KP/KD from default_runner.py, scaled by
    velocity_kp_scale and velocity_kd_scale respectively."""
    send_vel_target: bool = False
    """Send FLASH-G's velocity targets to the simulator's PD controller.
    When True, vel_target is included in the action dict and passed to
    set_joint_velocity_target().  When False, velocity is still computed
    for diagnostic plots but NOT sent to the robot."""
    velocity_kp_scale: float = 1.0
    """Uniform multiplier for velocity-mode Kp values (>1 = stiffer position
    tracking, <1 = softer).  Only takes effect when velocity_pd_gains=True."""
    velocity_kd_scale: float = 1.0
    """Uniform multiplier for velocity-mode Kd values (>1 = more damping,
    <1 = less).  Only takes effect when velocity_pd_gains=True."""
    fd_vel_target: bool = False
    """Finite-difference velocity feedforward (policy-agnostic).  When True,
    the eval runner derives a velocity chunk from the FINAL position-target
    chunk (central differences on interior points, one-sided differences at
    both chunk ends — strictly within the current plan, never across the
    replan boundary; dt = control period between successive targets) and
    feeds it through the SAME dof_vel_target pipeline used for FLASH-G's
    analytic velocity.  Works for any chunked policy; for FLASH-family
    policies it REPLACES the analytic velocity (velocity-source ablation).
    Whether the velocity is actually sent to the sim is still gated by
    send_vel_target, and PD gain overrides are still gated by
    velocity_pd_gains. NOT supported with temporal_agg=True (raises).
    Default False => original behaviour unchanged."""

    # Error plot visualization
    fixed_error_ylim: bool = True
    """Use fixed per-joint y-axis limits on error diagnostic plots
    (based on baseline model statistics) for cross-model comparison."""

    # Out-of-distribution (OOD) evaluation
    ood_shift: float = 0.0  # OOD intensity: The distance (in meters) that the target object moves radially away from the robot base, 0 = close

    # FLASH C_h robustness probe — inference-time perturbation of the history-fitted coefficients
    eval_history_noise_std: float = 0.0
    """Std of Gaussian noise added to the history polynomial coefficients C_h at
    INFERENCE time (normalized-coefficient units — directly comparable to the
    training-time history_noise_std). FLASH policy only (perturbs the flow start);
    other policies log a warning and ignore it. 0.0 (default) = disabled =>
    bit-for-bit original behaviour. Noise comes from a DEDICATED torch.Generator,
    so the global RNG stream (env / domain randomization) is untouched and runs
    with different sigma see identical environment randomness.
    Enable with: --override "eval_config.eval_args.eval_history_noise_std=0.5"."""
    eval_history_noise_seed: int = 0
    """Seed for the dedicated generator behind eval_history_noise_std."""

    # Discrete (binary) gripper actuation — discontinuity stress-test
    gripper_binarize: bool = False
    """Discretize the gripper at eval time. When True, the final gripper (finger)
    joint targets are snapped to fully-open / fully-closed by a threshold, i.e. the
    gripper is driven by a binary {open, close} signal instead of the policy's
    continuous value. When False (default) the behaviour is UNCHANGED (continuous
    gripper), so the continuous baseline is bit-for-bit the original path. This is an
    eval-time, policy-agnostic post-processing step applied at the shared action-apply
    layer (same rule for every policy); only active for action_type='joint_pos' (ee
    mode already handles the gripper via ee_cfg.gripper_rep)."""
    gripper_binarize_threshold: float = 0.02
    """Midpoint threshold (in raw finger-joint units; franka finger range [0.0, 0.04])
    used when gripper_binarize=True: mean finger target >= threshold -> open
    (robot.gripper_open_q) else close (robot.gripper_close_q). Only takes effect when
    gripper_binarize=True."""

    def __post_init__(self):
        log.info(f"Args: {self}")
