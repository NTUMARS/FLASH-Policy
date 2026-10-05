"""FLASH-G: Gaussian-to-polynomial flow matching with a DiT backbone.

Expert actions are normalized, fitted by least squares and, when enabled,
corrected with KKT position and velocity constraints. Coefficient rows are then
scale-normalized. The constraints enforce C1 continuity of expert training
targets; independently predicted chunks are not projected at inference.

Training uses coefficient-space flow-matching MSE from a standard Gaussian
source. Inference transports Gaussian noise and decodes the
result into positions and analytic derivatives. The evaluation runner supplies
the physical-time scaling for velocity commands.

The supplied flash_g.yaml uses Legendre degree 6 (7 coefficient tokens), fit
padding 1 and C1 target constraints. Legendre and power bases are both supported.
"""

from typing import Dict, Optional, Tuple  # Type hints for function signatures

import torch                               # PyTorch core library
import torch.nn.functional as F            # Functional API (MSE loss, etc.)
from einops import reduce                  # Einstein notation for tensor reductions

# --- Project-specific imports ---
from roboverse_learn.il.utils.models.flow_net import FlowTransformer          # DiT backbone: Transformer with AdaLN for flow matching
from roboverse_learn.il.utils.vision.multi_image_obs_encoder import MultiImageObsEncoder  # Encodes RGB images + low-dim proprioception into feature vectors
from roboverse_learn.il.utils.normalizer import LinearNormalizer               # Min-max normalizer that maps data to [-1, 1]
from roboverse_learn.il.utils.pytorch_util import dict_apply                   # Applies a function to all values in a dict
from roboverse_learn.il.policies.base_image_policy import BaseImagePolicy      # Abstract base class with predict_action() and set_normalizer()


class FlashGDiTImagePolicy(BaseImagePolicy):
    """
    FLASH-G policy with DiT (Diffusion Transformer) backbone.

    Uses Flow Matching to generate polynomial coefficients that define smooth trajectories.
    Supports both Legendre (recommended) and power polynomial bases.

    The polynomial q(s) = Σ_j a_j B_j(s)  (s ∈ [0,1]) provides:
      - Position  q(s)   by direct evaluation
      - Velocity  v(t) = q'(s) / T   via first derivative
      - Acceleration a(t) = q''(s) / T²  via second derivative

    where T is the effective execution time.

    Key architectural difference from standard Flow Matching:
      Standard FM: model predicts flow vectors in TRAJECTORY space (B, H, Da)
      FLASH-G FM:     model predicts flow vectors in COEFFICIENT space (B, C, Da)
      where C = poly_order + 1 << H, so the generation space is much smaller.
    """

    def __init__(
        self,
        shape_meta: dict,                       # Dictionary describing observation and action shapes, e.g. {"action": {"shape": [9]}, "obs": {...}}
        obs_encoder: MultiImageObsEncoder,      # Neural network that encodes multi-modal observations (images + proprioception) into a feature vector
        horizon,                                # Total trajectory length in timesteps (e.g. 16), includes both observation and action steps
        n_action_steps,                         # Number of future action steps to execute (e.g. 8), model predicts this many steps into the future
        n_obs_steps,                            # Number of past observation steps used as context (e.g. 8), fed to the obs_encoder
        num_inference_steps=None,               # Number of ODE solver steps during inference (e.g. 10), more steps = more accurate but slower
        obs_as_global_cond=True,                # If True, observation features are used as global conditioning (via AdaLN), not as inpainting tokens
        diffusion_step_embed_dim=256,           # Dimension of the sinusoidal time embedding for the flow matching timestep τ ∈ [0,1]
        hidden_dim=512,                         # Hidden dimension of the DiT Transformer (width of each layer)
        num_layers=4,                           # Number of AdaLN Transformer blocks stacked sequentially
        num_heads=8,                            # Number of attention heads in multi-head self-attention
        mlp_ratio=4.0,                          # MLP expansion ratio: the feedforward hidden dim = hidden_dim * mlp_ratio
        dropout=0.1,                            # Dropout probability applied in attention and MLP layers
        # ---- FLASH-G-specific hyperparameters ----
        poly_order: int = 3,                    # Degree of the polynomial (3=cubic → 4 coefficients per action dimension)
        basis_type: str = "legendre",           # Which polynomial basis to use: "legendre" (orthogonal, well-conditioned) or "power" (standard s^n basis)
        fit_pad: int = 0,                       # Extended-window fitting: pad fit_pad steps on each side when fitting expert trajectories (0 = disabled)
        boundary_constraint: bool = False,      # KKT C1 constraints on expert training targets at both execution anchors
        poly_prefix: int | None = None,         # Extra timesteps before action points (None=full horizon, 0=action only, N=action+N history)
        poly_suffix: int = 0,                   # Extra timesteps after action points (only effective when poly_prefix is not None)
        **kwargs,                               # Catch-all for any extra keyword arguments (passed through to conditional_sample)
    ):
        super().__init__()  # Initialize nn.Module base class (registers parameters, buffers, etc.)

        # FLASH-G requires global conditioning because inpainting in polynomial coefficient space
        # has no clear physical meaning (unlike trajectory space where you can fix observed positions)
        assert obs_as_global_cond, (
            "FLASH-G only supports obs_as_global_cond=True. "
            "Inpainting in polynomial-coefficient space is not meaningful."
        )

        # ---- Parse action shape from metadata ----
        action_shape = shape_meta["action"]["shape"]    # e.g. [9] for Franka (7 joints + 2 gripper fingers)
        assert len(action_shape) == 1                   # Only support 1D action vectors (no spatial actions)
        action_dim = action_shape[0]                    # Da = 9: the dimensionality of each action vector

        # Get the output feature dimension of the observation encoder
        # This is the concatenation of: ResNet18 image features + raw proprioception (agent_pos)
        obs_feature_dim = obs_encoder.output_shape()[0]  # e.g. 512 (ResNet18) + 9 (joint_pos) = 521

        # ---- Store FLASH-G-specific parameters ----
        self.poly_order = poly_order          # e.g. 3 for cubic polynomial
        self.num_coeff = poly_order + 1       # Number of polynomial coefficients = degree + 1 (e.g. 4 for cubic)
        self.basis_type = basis_type          # "legendre" or "power"
        self.fit_pad = fit_pad                       # Extended-window fitting pad (0 = standard)
        self.boundary_constraint = boundary_constraint  # KKT C1 constraints on training targets
        self.poly_prefix = poly_prefix
        self.poly_suffix = poly_suffix

        # ==================================================================
        # Build polynomial basis matrices
        #
        # When poly_prefix is set (e.g. 1), the polynomial covers a REDUCED
        # region: n_action_steps + poly_prefix + poly_suffix points.  This
        # focuses the polynomial expressiveness on the action region while
        # keeping a few extra points for velocity continuity at the anchor.
        #
        # When poly_prefix is None, the full horizon is used (original FLASH-G
        # behavior) — the polynomial covers all n_obs + n_action points.
        #
        # When fit_pad > 0 (extended-window fitting):
        #   - Fitting is done on H_poly_ext = H_poly + 2*fit_pad points
        #   - Evaluation / derivatives are computed at the MIDDLE H_poly points
        # ==================================================================

        C = poly_order + 1

        if poly_prefix is not None:
            H_poly = poly_prefix + n_action_steps + poly_suffix
            self._anchor_idx = poly_prefix
            self._poly_nf_start = n_obs_steps - 1 - poly_prefix
        else:
            H_poly = horizon
            self._anchor_idx = n_obs_steps - 1
            self._poly_nf_start = 0

        self._H_poly = H_poly
        H_poly_ext = H_poly + 2 * fit_pad

        # Extended time grid for fitting
        s_ext = torch.linspace(0.0, 1.0, H_poly_ext, dtype=torch.float64)
        # Middle time points for evaluation / derivatives / inference
        s_mid = s_ext[fit_pad: fit_pad + H_poly] if fit_pad > 0 else s_ext

        build_basis = (self._build_legendre_basis if basis_type == "legendre"
                       else self._build_power_basis if basis_type == "power"
                       else None)
        if build_basis is None:
            raise ValueError(f"Unknown basis_type: {basis_type}. Use 'legendre' or 'power'.")

        # Matrices for the extended window (used for fitting during training)
        eval_ext, d1_ext, _ = build_basis(s_ext, poly_order)
        StS = eval_ext.T @ eval_ext
        lstsq_mat = torch.linalg.solve(StS, eval_ext.T)   # (C, H_ext)

        # Matrices at the middle H time points (used for evaluation / derivatives)
        eval_mat, d1_mat, d2_mat = build_basis(s_mid, poly_order)

        # Coefficient normalization scale (derived from the FITTING matrix)
        coeff_scale = torch.norm(lstsq_mat, dim=1).clamp(min=1e-8)  # (C,)

        # Diagnostics
        cond_number = torch.linalg.cond(StS).item()
        if poly_prefix is not None:
            pp_desc = f" (poly_prefix={poly_prefix}, poly_suffix={poly_suffix})"
        else:
            pp_desc = f" (full horizon={horizon})"
        print(f"[FLASH-G] basis={basis_type}, degree={poly_order}, H_poly={H_poly}{pp_desc}"
              f"{f', fit_pad={fit_pad} (H_poly_ext={H_poly_ext})' if fit_pad else ''}")
        print(f"[FLASH-G] cond(S^TS)={cond_number:.2e}")
        print(f"[FLASH-G] coeff scales: {[f'{x:.4f}' for x in coeff_scale.tolist()]}")

        # Register buffers
        self.register_buffer("_eval_matrix", eval_mat.float())
        self.register_buffer("_d1_matrix", d1_mat.float())
        self.register_buffer("_d2_matrix", d2_mat.float())
        self.register_buffer("_lstsq_matrix", lstsq_mat.float())
        self.register_buffer("_coeff_scale",
                             coeff_scale.float().view(1, -1, 1))
        self.register_buffer("_time_points", s_mid.float())

        # ---- Boundary constraint matrices (C1: position + velocity at anchor & overlap) ----
        # overlap = the first time step of the NEXT window = anchor + Ta
        # Corrected expert targets satisfy poly_k(overlap) = poly_{k+1}(anchor).
        # Inference outputs are not projected onto these training constraints.
        if boundary_constraint:
            bc_anchor = fit_pad + self._anchor_idx
            bc_overlap = fit_pad + self._anchor_idx + n_action_steps
            if bc_overlap >= H_poly_ext:
                print(f"[FLASH-G] WARNING: bc_overlap={bc_overlap} >= H_poly_ext={H_poly_ext}. "
                      f"Increase fit_pad for valid boundary constraint.")
            elif bc_overlap == H_poly_ext - 1:
                print(f"[FLASH-G] WARNING: bc_overlap at polynomial edge (index {bc_overlap} of "
                      f"{H_poly_ext}). Velocity uses backward diff. Consider fit_pad += 1.")
            bc_pos_indices = torch.tensor([bc_anchor, bc_overlap], dtype=torch.long)
            A_pos = eval_ext[bc_pos_indices.tolist(), :]                        # (2, C)
            A_vel = d1_ext[bc_pos_indices.tolist(), :]                          # (2, C)
            A = torch.cat([A_pos, A_vel], dim=0)                               # (4, C)
            StS_inv = torch.linalg.solve(StS, torch.eye(StS.shape[0], dtype=torch.float64))
            StS_inv_AT = StS_inv @ A.T                                         # (C, 4)
            M = A @ StS_inv_AT                                                 # (4, 4)
            bc_correction = StS_inv_AT @ torch.linalg.inv(M)                   # (C, 4)
            self.register_buffer("_bc_eval_pos", A_pos.float())                # (2, C)
            self.register_buffer("_bc_eval_vel", A_vel.float())                # (2, C)
            self.register_buffer("_bc_correction", bc_correction.float())      # (C, 4)
            self.register_buffer("_bc_pos_indices", bc_pos_indices)            # (2,) long
            print(f"[FLASH-G] boundary_constraint=C1, pin indices={bc_pos_indices.tolist()} "
                  f"in H_poly_ext={H_poly_ext} (4 constraints: 2 pos + 2 vel)")

        self._num_pinned = 0
        self._num_coeff_model = poly_order + 1

        # ==================================================================
        # Flow Matching backbone (DiT) — operates in NORMALIZED coefficient space
        #
        # The DiT processes a sequence of C tokens (one per polynomial coefficient),
        # each of dimension Da (action_dim). Self-attention allows coefficients
        # to exchange information (e.g., c_3 and c_0 can interact to ensure smoothness).
        #
        # Conditioning:
        #   - Flow timestep τ ∈ [0,1]: embedded via sinusoidal + MLP, injected through AdaLN
        #   - Observation features: embedded via linear projection, injected through AdaLN
        #
        # Input:  (B, C, Da) — noisy/interpolated polynomial coefficients at flow time τ
        # Output: (B, C, Da) — predicted velocity field (direction to move in coefficient space)
        # ==================================================================
        input_dim = action_dim                           # Each token has Da dimensions (one coefficient vector per basis function)
        global_cond_dim = obs_feature_dim * n_obs_steps  # Total conditioning dimension: obs features concatenated across all observation timesteps

        model = FlowTransformer(
            input_dim=input_dim,               # Token dimension = action_dim (e.g. 9)
            condition_dim=global_cond_dim,      # Global condition dimension (e.g. 521 * 8 = 4168)
            hidden_dim=hidden_dim,             # Transformer hidden width (e.g. 256)
            output_dim=input_dim,              # Output same dimension as input (velocity field in coefficient space)
            num_layers=num_layers,             # Number of stacked AdaLN Transformer blocks (e.g. 3)
            num_heads=num_heads,               # Attention heads (e.g. 8)
            mlp_ratio=mlp_ratio,               # Feedforward expansion ratio (e.g. 4.0 → FF dim = 256*4 = 1024)
            dropout=dropout,                   # Dropout rate for regularization (e.g. 0.1)
            time_embed_dim=diffusion_step_embed_dim,  # Dimension for flow timestep embedding (e.g. 256)
        )

        # ---- Store all remaining attributes ----
        self.obs_encoder = obs_encoder         # Image + proprioception encoder (ResNet18 + concat)
        self.model = model                     # DiT backbone for flow matching in coefficient space
        self.normalizer = LinearNormalizer()   # Action/observation normalizer (fitted on training data, maps to [-1,1])
        self.horizon = horizon                 # Total trajectory length (e.g. 16 timesteps)
        self.obs_feature_dim = obs_feature_dim # Output dimension of obs_encoder
        self.action_dim = action_dim           # Action dimensionality (e.g. 9 for Franka)
        self.n_action_steps = n_action_steps   # Number of future steps to execute (e.g. 8)
        self.n_obs_steps = n_obs_steps         # Number of past observation steps (e.g. 8)
        self.obs_as_global_cond = obs_as_global_cond  # Always True for FLASH-G
        self.kwargs = kwargs                   # Extra kwargs passed through to conditional_sample
        self.num_inference_steps = num_inference_steps  # ODE solver steps at inference (e.g. 10)

    # ==================================================================
    #  Basis construction (static methods, called once during __init__)
    #
    #  These methods build the evaluation matrices for polynomial bases.
    #  Each returns three matrices of shape (H, C):
    #    eval_mat: B_j(s_i)    — basis function values
    #    d1_mat:   B_j'(s_i)   — first derivatives
    #    d2_mat:   B_j''(s_i)  — second derivatives
    #  where i indexes time points and j indexes basis functions.
    # ==================================================================

    @staticmethod
    def _build_legendre_basis(
        s: torch.Tensor, degree: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build shifted Legendre polynomial basis on s ∈ [0, 1].

        Legendre polynomials P_n(x) are orthogonal on [-1, 1], meaning:
            ∫_{-1}^{1} P_i(x) P_j(x) dx = 0  for i ≠ j

        This orthogonality makes the Gram matrix S^T·S nearly diagonal,
        resulting in a condition number close to 1 (vs. 10^4+ for power basis).

        We use the substitution x = 2s - 1 to map [0, 1] → [-1, 1],
        creating "shifted" Legendre polynomials P̃_n(s) = P_n(2s-1).

        The chain rule gives us derivatives w.r.t. s:
            d/ds P̃_n(s) = 2 · P_n'(2s-1)      (factor of 2 from dx/ds)
            d²/ds² P̃_n(s) = 4 · P_n''(2s-1)    (factor of 4 = 2²)

        The Bonnet recurrence relation builds P_n from P_{n-1} and P_{n-2}:
            (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
        We differentiate this recurrence to get P_n'(x) and P_n''(x).

        Args:
            s: (H,) normalized time points in [0, 1]
            degree: polynomial degree (e.g. 3 for cubic)

        Returns:
            eval_mat: (H, C) — P̃_0(s), P̃_1(s), ..., P̃_n(s) at each time point
            d1_mat:   (H, C) — first derivatives d/ds P̃_j(s)
            d2_mat:   (H, C) — second derivatives d²/ds² P̃_j(s)
        """
        x = 2 * s - 1  # Map s ∈ [0,1] to x ∈ [-1,1] where standard Legendre polynomials are defined
        H = len(s)      # Number of time points (= horizon, e.g. 16)
        C = degree + 1   # Number of basis functions (= number of coefficients, e.g. 4 for cubic)

        # Allocate matrices for P_n(x), P_n'(x), P_n''(x) using the three-term recurrence
        P = torch.zeros(H, C, dtype=s.dtype)      # P[i, j] = P_j(x_i): Legendre polynomial values
        dP = torch.zeros(H, C, dtype=s.dtype)      # dP[i, j] = P_j'(x_i): first derivatives w.r.t. x
        d2P = torch.zeros(H, C, dtype=s.dtype)     # d2P[i, j] = P_j''(x_i): second derivatives w.r.t. x

        # Base case: P_0(x) = 1 (constant polynomial)
        P[:, 0] = 1.0     # P_0(x) = 1 everywhere
        dP[:, 0] = 0.0    # Derivative of constant is 0
        d2P[:, 0] = 0.0   # Second derivative of constant is 0

        # Base case: P_1(x) = x (linear polynomial)
        if C > 1:
            P[:, 1] = x       # P_1(x) = x
            dP[:, 1] = 1.0    # P_1'(x) = 1
            d2P[:, 1] = 0.0   # P_1''(x) = 0

        # Build higher-order polynomials using the Bonnet recurrence relation
        for n in range(1, C - 1):
            # Bonnet's recurrence: (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
            # This computes P_{n+1} from P_n and P_{n-1}
            P[:, n + 1] = (
                (2 * n + 1) * x * P[:, n] - n * P[:, n - 1]
            ) / (n + 1)

            # Differentiate the recurrence to get the first derivative:
            # (n+1) P_{n+1}'(x) = (2n+1) [P_n(x) + x P_n'(x)] - n P_{n-1}'(x)
            # This uses the product rule on the x·P_n(x) term: d/dx[x·P_n] = P_n + x·P_n'
            dP[:, n + 1] = (
                (2 * n + 1) * (P[:, n] + x * dP[:, n]) - n * dP[:, n - 1]
            ) / (n + 1)

            # Differentiate again to get the second derivative:
            # (n+1) P_{n+1}''(x) = (2n+1) [2P_n'(x) + x P_n''(x)] - n P_{n-1}''(x)
            # This uses the product rule on x·P_n'(x): d/dx[x·P_n'] = P_n' + x·P_n''
            # Combined with d/dx[P_n] = P_n', we get 2P_n' + x·P_n''
            d2P[:, n + 1] = (
                (2 * n + 1) * (2 * dP[:, n] + x * d2P[:, n]) - n * d2P[:, n - 1]
            ) / (n + 1)

        # Apply chain rule to convert derivatives from x-space to s-space:
        # Since x = 2s - 1, we have dx/ds = 2, so:
        #   d/ds f(x(s)) = f'(x) · dx/ds = 2 · f'(x)
        #   d²/ds² f(x(s)) = f''(x) · (dx/ds)² = 4 · f''(x)
        eval_mat = P           # Evaluation matrix: no chain rule needed for function values
        d1_mat = 2 * dP        # First derivative w.r.t. s: multiply by dx/ds = 2
        d2_mat = 4 * d2P       # Second derivative w.r.t. s: multiply by (dx/ds)² = 4

        return eval_mat, d1_mat, d2_mat

    @staticmethod
    def _build_power_basis(
        s: torch.Tensor, degree: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build standard power polynomial basis [s^n, s^{n-1}, ..., s, 1].

        This is the "naive" polynomial basis. While intuitive, it suffers from
        poor numerical conditioning: the Gram matrix S^T·S resembles a Hilbert
        matrix, with condition number growing exponentially with degree.

        For degree 3 (cubic): condition number ≈ 1.1×10^4
        For degree 5:         condition number ≈ 1.0×10^7

        Kept for comparison and backward compatibility. Use Legendre for production.

        Column ordering: [s^n, s^{n-1}, ..., s^1, s^0] (highest power first)

        Args:
            s: (H,) normalized time points in [0, 1]
            degree: polynomial degree

        Returns:
            eval_mat: (H, C) — [s_i^n, s_i^{n-1}, ..., 1] at each time point
            d1_mat:   (H, C) — first derivatives [n·s^{n-1}, (n-1)·s^{n-2}, ..., 1, 0]
            d2_mat:   (H, C) — second derivatives [n(n-1)·s^{n-2}, ..., 0, 0]
        """
        # Power indices in descending order: [n, n-1, ..., 1, 0]
        powers = torch.arange(degree, -1, -1, dtype=s.dtype)  # e.g. [3, 2, 1, 0] for cubic

        # Evaluation matrix: eval_mat[i, j] = s_i^{powers[j]}
        # s.unsqueeze(-1) → (H, 1), powers.unsqueeze(0) → (1, C), broadcasting → (H, C)
        eval_mat = s.unsqueeze(-1).pow(powers.unsqueeze(0))  # (H, C)

        # First derivative: d/ds s^p = p · s^{p-1}
        d1_powers = (powers - 1).clamp(min=0)  # Exponents for derivative (clamp to avoid s^{-1})
        d1_mat = powers.unsqueeze(0) * s.unsqueeze(-1).pow(d1_powers.unsqueeze(0))  # p · s^{p-1}
        d1_mat[:, powers < 1] = 0  # d/ds(s^0) = d/ds(1) = 0: zero out the constant term column

        # Second derivative: d²/ds² s^p = p(p-1) · s^{p-2}
        d2_powers = (powers - 2).clamp(min=0)  # Exponents for second derivative
        d2_coeff = powers * (powers - 1)        # Coefficient: p(p-1)
        d2_mat = d2_coeff.unsqueeze(0) * s.unsqueeze(-1).pow(d2_powers.unsqueeze(0))  # p(p-1) · s^{p-2}
        d2_mat[:, powers < 2] = 0  # Zero out constant (p=0) and linear (p=1) columns

        return eval_mat, d1_mat, d2_mat

    # ==================================================================
    #  Polynomial operations
    #
    #  All methods below work with NORMALIZED coefficients.
    #  "Normalized" means divided by _coeff_scale, so all coefficient
    #  dimensions have approximately unit variance. This is the space
    #  in which Flow Matching operates.
    #
    #  The conversion between raw and normalized coefficients:
    #    normalized = raw / _coeff_scale
    #    raw = normalized * _coeff_scale
    # ==================================================================

    def _trajectory_to_coefficients(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Fit a polynomial to a trajectory via batch least-squares,
        returning NORMALIZED coefficients.

        Args:
            trajectory: (B, H_ext, Da) — trajectory points (H_ext = H + 2*fit_pad)

        Returns:
            (B, C, Da) — normalized polynomial coefficients
        """
        # Batch matrix multiply: for each batch and action dim, multiply fitting matrix (C,H) with trajectory (H,)
        # einsum "ch,bhd->bcd": C coefficients = L(C,H) @ trajectory(B,H,Da) for each batch b and dim d
        raw_coeff = torch.einsum("ch,bhd->bcd", self._lstsq_matrix, trajectory)
        # Divide by analytical scale so all coefficient dimensions have ~unit variance
        # _coeff_scale shape is (1, C, 1), broadcasts with (B, C, Da)
        return raw_coeff / self._coeff_scale

    def _apply_boundary_constraint(
        self,
        norm_coeff: torch.Tensor,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """Apply KKT C1 constraints to expert targets at both execution anchors.

        Four equality constraints pin two positions and two derivatives.
        Derivative targets use finite differences of the expert samples.
        This correction is applied during training, not to inference outputs.

        Args:
            norm_coeff: (B, C, Da) — normalized polynomial coefficients
            trajectory:  (B, H_ext, Da) — expert trajectory (normalized actions)

        Returns:
            (B, C, Da) — constrained normalized coefficients
        """
        raw = norm_coeff * self._coeff_scale                           # (B, C, Da)
        idx = self._bc_pos_indices                                     # (2,)
        H_ext = trajectory.shape[1]

        # Position residual
        cur_pos = torch.einsum("kc,bcd->bkd", self._bc_eval_pos, raw) # (B, 2, Da)
        tgt_pos = trajectory[:, idx, :]                                # (B, 2, Da)

        # Velocity residual (finite difference in s-space)
        # Use centered diff when possible; at the window edge, use 2nd-order
        # backward diff (3-point formula, O(Δ²) accurate — same order as
        # centered diff) so adjacent windows get consistent velocity targets.
        tgt_vel_parts = []
        for k in range(len(idx)):
            i = idx[k]
            if i + 1 < H_ext:
                v = (trajectory[:, i + 1, :] - trajectory[:, i - 1, :]) * (H_ext - 1) / 2
            elif i >= 2:
                v = (3 * trajectory[:, i, :] - 4 * trajectory[:, i - 1, :] + trajectory[:, i - 2, :]) * (H_ext - 1) / 2
            else:
                v = (trajectory[:, i, :] - trajectory[:, i - 1, :]) * (H_ext - 1)
            tgt_vel_parts.append(v)
        tgt_vel = torch.stack(tgt_vel_parts, dim=1)  # (B, 2, Da)
        cur_vel = torch.einsum("kc,bcd->bkd", self._bc_eval_vel, raw) # (B, 2, Da)

        # Concatenate: (B, 4, Da)
        residual = torch.cat([cur_pos - tgt_pos, cur_vel - tgt_vel], dim=1)
        correction = torch.einsum("ck,bkd->bcd", self._bc_correction, residual)
        return (raw - correction) / self._coeff_scale

    def _coefficients_to_trajectory(
        self, norm_coeff: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate polynomial at all time points from NORMALIZED coefficients.

        This reconstructs the trajectory from polynomial coefficients:
            q(s_i) = Σ_j eval_mat[i,j] * raw_coeff[j]

        Args:
            norm_coeff: (B, C, Da) — normalized polynomial coefficients

        Returns:
            (B, H, Da) — reconstructed trajectory in normalized action space
        """
        # First, undo the coefficient normalization to get raw coefficients
        raw_coeff = norm_coeff * self._coeff_scale  # (B, C, Da)
        # Matrix multiply: for each batch and action dim, evaluate polynomial at all H time points
        # einsum "hc,bcd->bhd": trajectory(B,H,Da) = eval_matrix(H,C) @ raw_coeff(B,C,Da)
        return torch.einsum("hc,bcd->bhd", self._eval_matrix, raw_coeff)

    def _compute_velocity(
        self,
        norm_coeff: torch.Tensor,
        total_time: float = 1.0,
    ) -> torch.Tensor:
        """Compute velocity at all time points from NORMALIZED coefficients.
            
        Mathematical derivation:
            q(s) = Σ_j B_j(s) · c_j           (polynomial in normalized time s ∈ [0,1])
            dq/ds = Σ_j B_j'(s) · c_j         (derivative w.r.t. normalized time)
            v(t) = dq/dt = dq/ds · ds/dt       (chain rule to physical time)
                 = dq/ds · (1/T)               (since s = t/T, so ds/dt = 1/T)

        Args:
            norm_coeff: (B, C, Da) — normalized polynomial coefficients
            total_time: effective execution time T

        Returns:
            (B, H, Da) — velocity at each time point, in normalized action space per unit time

        Note on `total_time`:
        - If total_time == 1.0 (default), this returns the normalized derivative dq/ds.
        - If total_time == T_physical, this returns the true physical velocity dq/dt in rad/s.
        
        For Embodied AI tasks, it is highly recommended to let the Runner handle the 
        physical time scaling (via Continuous-Time Resampling) rather than setting it here.
        
        """
        raw_coeff = norm_coeff * self._coeff_scale  # Undo normalization: (B, C, Da)
        # Evaluate first derivative at all time points using pre-computed derivative matrix
        # d1_matrix[i,j] = B_j'(s_i), so dq/ds at s_i = Σ_j d1_matrix[i,j] * c_j
        return torch.einsum("hc,bcd->bhd", self._d1_matrix, raw_coeff) / total_time

    def _compute_acceleration(
        self,
        norm_coeff: torch.Tensor,
        total_time: float = 1.0,
    ) -> torch.Tensor:
        """Compute acceleration at all time points from NORMALIZED coefficients.

        Mathematical derivation:
            a(t) = d²q/dt² = d²q/ds² · (ds/dt)²    (chain rule, second derivative)
                 = d²q/ds² · (1/T²)                 (since ds/dt = 1/T)

        Args:
            norm_coeff: (B, C, Da) — normalized polynomial coefficients
            total_time: effective execution time T

        Returns:
            (B, H, Da) — acceleration at each time point, in normalized action space per unit time²
        """
        raw_coeff = norm_coeff * self._coeff_scale  # Undo normalization
        # Evaluate second derivative at all time points using pre-computed derivative matrix
        return torch.einsum("hc,bcd->bhd", self._d2_matrix, raw_coeff) / (total_time ** 2)

    # ==================================================================
    #  Inference
    # ==================================================================

    def conditional_sample(
        self,
        condition_data: torch.Tensor,    # (B, C, Da): conditioning data (zeros for FLASH-G — no inpainting)
        condition_mask: torch.Tensor,     # (B, C, Da): boolean mask (all False for FLASH-G — no inpainting)
        local_cond=None,                  # Not used in FLASH-G (would be per-token conditioning)
        global_cond=None,                 # (B, global_cond_dim): observation features for conditioning
        generator=None,                   # Optional torch.Generator for reproducible sampling
        **kwargs,                         # Extra arguments (unused)
    ) -> torch.Tensor:
        """Sample polynomial coefficients via ODE integration.

        Flow Matching defines a probability path from noise distribution p_0 = N(0,I)
        to data distribution p_1 = p_data via the interpolation:
            x_τ = (1-τ)·noise + τ·data,  τ ∈ [0, 1]

        The model learns the velocity field v(x_τ, τ) = data - noise that transports
        noise to data along this path. At inference time, we integrate this ODE:
            dx/dτ = v(x_τ, τ)

        Args:
            condition_data: (B, C, Da) — data to enforce where condition_mask is True (zeros for FLASH-G)
            condition_mask:  (B, C, Da) — boolean mask indicating which values to keep fixed (all False for FLASH-G)
            global_cond:    (B, D) — observation features for global conditioning

        Returns:
            (B, C, Da) — sampled normalized polynomial coefficients
        """
        model = self.model  # The DiT backbone

        # Step 1: Start from pure Gaussian noise in normalized coefficient space
        # Shape matches condition_data: (B, C, Da) where C=num_coeff, Da=action_dim
        trajectory = torch.randn(
            size=condition_data.shape,     # e.g. (32, 4, 9) for batch=32, cubic polynomial, 9 DOF
            dtype=condition_data.dtype,    # float32
            device=condition_data.device,  # GPU
            generator=generator,           # Optional: for reproducible random numbers
        )

        # Integrate the ODE from τ=0 (noise) to τ=1 (data).
        # Create evenly-spaced flow timesteps: [0, 1/N, 2/N, ..., 1]
        time_steps = torch.linspace(0, 1.0, self.num_inference_steps + 1)  # e.g. [0, 0.1, 0.2, ..., 1.0] for N=10

        for i in range(self.num_inference_steps):
            # Enforce conditioning (for FLASH-G, condition_mask is all False, so this is a no-op)
            # For other models, this would fix observed positions/velocities
            trajectory[condition_mask] = condition_data[condition_mask]

            # Compute timestep boundaries for this integration step
            t_start = time_steps[i].view(1).expand(trajectory.shape[0]).to(self.device)      # τ_i, expanded to batch size
            t_end = time_steps[i + 1].view(1).expand(trajectory.shape[0]).to(self.device)    # τ_{i+1}
            dt = (t_end - t_start).view(-1, 1, 1)  # Step size, reshaped to (B, 1, 1) for broadcasting with (B, C, Da)
            t_mid = t_start + (t_end - t_start) / 2

            v_start = model(
                trajectory, t_start, local_cond=local_cond, global_cond=global_cond
            )  # v(x_τ, τ): predicted velocity field at current state and time

            trajectory_mid = trajectory + v_start * dt / 2  # x_mid = x_τ + (dt/2) · v(x_τ, τ)

            v_mid = model(
                trajectory_mid, t_mid, local_cond=local_cond, global_cond=global_cond
            )
            trajectory = trajectory + dt * v_mid  # x_{τ+dt} = x_τ + dt · v(x_mid, τ + dt/2)

        # Final conditioning enforcement (no-op for FLASH-G)
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory  # (B, C, Da): sampled normalized polynomial coefficients

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],     # Observation dictionary: {"head_cam": (B,T,3,H,W), "agent_pos": (B,T,9)}
    ) -> Dict[str, torch.Tensor]:
        """Full inference pipeline: observation → polynomial coefficients → trajectory + dynamics.

        This is the main method called during evaluation/deployment.

        Pipeline:
            1. Normalize observations (images + proprioception)
            2. Encode observations into feature vector via obs_encoder
            3. Sample polynomial coefficients via Flow Matching ODE
            4. Evaluate polynomial → reconstruct trajectory
            5. Compute velocity and acceleration via analytical derivatives
            6. Extract the executable action slice

        Args:
            obs_dict: observation dictionary with keys like "head_cam", "agent_pos"

        Returns:
            dict with:
                action        (B, n_action_steps, Da) — the executable action chunk sent to controller
                action_pred   (B, H, Da) — full predicted trajectory over entire horizon
                velocity      (B, H, Da) — v(t) in real (unnormalized) action space
                acceleration  (B, H, Da) — a(t) in real (unnormalized) action space
                coefficients  (B, C, Da) — normalized polynomial coefficients (for debugging)
        """
        assert "past_action" not in obs_dict  # Past action conditioning not implemented

        # ---- Step 1: Normalize observations to [-1, 1] ----
        nobs = self.normalizer.normalize(obs_dict)  # Normalize each obs key independently
        value = next(iter(nobs.values()))            # Get any tensor to read batch size
        B = value.shape[0]                           # Batch size
        Da = self.action_dim                         # Action dimension (e.g. 9)
        To = self.n_obs_steps                        # Number of observation steps to use (e.g. 8)
        device, dtype = self.device, self.dtype      # GPU device and float32 dtype

        # ---- Step 2: Encode observations ----
        # Take the first To timesteps of each observation, then flatten time into batch dimension
        # From (B, T, ...) → (B*To, ...) so the encoder processes each timestep independently
        this_nobs = dict_apply(
            nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)  # (B*To, obs_feature_dim): encode each frame
        # Reshape back and concatenate all timesteps: (B, To*obs_feature_dim)
        # This becomes the global condition for the DiT
        global_cond = nobs_features.reshape(B, -1)

        # ---- Step 3: Prepare conditioning & sample coefficients ----
        cond_data = torch.zeros(B, self.num_coeff, Da, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # ---- Step 4: Flow Matching → sample polynomial coefficients ----
        ncoeff_sampled = self.conditional_sample(
            cond_data, cond_mask,
            local_cond=None, global_cond=global_cond,
            **self.kwargs,
        )  # (B, C, Da)

        ncoeff_pred = ncoeff_sampled

        # ---- Step 5: Evaluate polynomial → reconstruct trajectory ----
        naction_pred = self._coefficients_to_trajectory(ncoeff_pred)  # (B, H, Da) in [-1,1]

        # ---- Step 6: Unnormalize to real action space ----
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # ---- Step 7: Compute velocity and acceleration ----
        nvel = self._compute_velocity(ncoeff_pred)
        nacce = self._compute_acceleration(ncoeff_pred)
        scale = self.normalizer["action"].params_dict["scale"].view(1, 1, -1)
        velocity = nvel / scale
        acceleration = nacce / scale

        # ---- Step 8: Extract the executable action chunk ----
        start = self._anchor_idx
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        velocity_action = velocity[:, start:end]

        result = {
            "action": action,
            "action_pred": action_pred,
            "velocity": velocity,
            "velocity_action": velocity_action,
            "acceleration": acceleration,
            "coefficients": ncoeff_pred,
        }
        return result

    # ==================================================================
    #  Training
    # ==================================================================

    def set_normalizer(self, normalizer: LinearNormalizer):
        """Load the fitted normalizer from the training dataset.

        Called by the training runner after computing normalizer statistics
        from the full dataset. The normalizer maps actions to [-1, 1] based
        on observed min/max values.
        """
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch: Dict) -> torch.Tensor:
        """Compute coefficient-space flow-matching MSE from Gaussian noise.

        This is the core training method, called once per batch by the training loop.

        Pipeline:
            1. Normalize expert trajectory to [-1, 1]           → (B, H, Da)
            2. Least-squares polynomial fit → normalized coeffs → (B, C, Da)
            3. Sample random flow timestep τ ~ Uniform(0, 1)
            4. Create interpolated coefficients: x_τ = (1-τ)·noise + τ·target
            5. Predict velocity field: v_pred = model(x_τ, τ, obs_condition)
            6. FM loss: ||v_pred - (target - noise)||²
        Args:
            batch: dict with "obs" (observation dict) and "action" (B, H, Da) expert trajectories

        Returns:
            scalar flow-matching loss tensor
        """
        assert "valid_mask" not in batch  # Valid mask not supported

        # ---- Step 1: Normalize inputs ----
        nobs = self.normalizer.normalize(batch["obs"])
        nactions_full = self.normalizer["action"].normalize(batch["action"])
        B = nactions_full.shape[0]

        fp = self.fit_pad
        poly_start = self._poly_nf_start
        H_poly_ext = self._H_poly + 2 * fp
        poly_traj = nactions_full[:, poly_start:poly_start + H_poly_ext, :]

        # ---- Step 2: Fit polynomial to expert trajectory ----
        if "target_coeff" in batch:
            target_coeff = batch["target_coeff"]
        else:
            target_coeff = self._trajectory_to_coefficients(poly_traj)

        if self.boundary_constraint:
            target_coeff = self._apply_boundary_constraint(target_coeff, poly_traj)

        # ---- Step 3: Encode observations for conditioning ----
        # When fit_pad > 0, observations start at index fit_pad in the extended sequence
        obs_start = fp
        this_nobs = dict_apply(
            nobs, lambda x: x[:, obs_start: obs_start + self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)       # (B*To, obs_feature_dim)
        global_cond = nobs_features.reshape(B, -1)        # (B, To*obs_feature_dim): global conditioning vector

        # ---- Step 4: Flow Matching forward process ----
        noise = torch.randn_like(target_coeff)
        timesteps = torch.rand(B, device=target_coeff.device)
        t_exp = timesteps.view(-1, 1, 1)

        coeff_inter = (1 - t_exp) * noise + t_exp * target_coeff
        vector_fm_true = target_coeff - noise

        # ---- Step 5: Model prediction ----
        vector_fm_pred = self.model(
            coeff_inter, timesteps, local_cond=None, global_cond=global_cond
        )

        # ---- Step 6: Flow Matching loss ----
        loss_fm = F.mse_loss(vector_fm_pred, vector_fm_true, reduction="none")
        loss_fm = reduce(loss_fm, "b ... -> b (...)", "mean").mean()

        return loss_fm  # Scalar tensor, backpropagated by the training loop

    # ==================================================================
    #  Diagnostics (not used in training loop, useful for debugging)
    # ==================================================================

    @torch.no_grad()
    def compute_poly_fit_error(self, batch: Dict) -> float:
        """Measure how well the polynomial approximates the expert trajectory."""
        nactions_full = self.normalizer["action"].normalize(batch["action"])
        fp = self.fit_pad
        poly_start = self._poly_nf_start
        H_poly_ext = self._H_poly + 2 * fp
        poly_traj = nactions_full[:, poly_start:poly_start + H_poly_ext, :]
        nactions_mid = poly_traj[:, fp:fp + self._H_poly, :] if fp > 0 else poly_traj
        coeff = self._trajectory_to_coefficients(poly_traj)
        recon = self._coefficients_to_trajectory(coeff)
        return F.mse_loss(recon, nactions_mid).item()

    @torch.no_grad()
    def get_coeff_statistics(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Get statistics of the NORMALIZED coefficient distribution."""
        nactions_full = self.normalizer["action"].normalize(batch["action"])
        fp = self.fit_pad
        poly_start = self._poly_nf_start
        H_poly_ext = self._H_poly + 2 * fp
        poly_traj = nactions_full[:, poly_start:poly_start + H_poly_ext, :]
        coeff = self._trajectory_to_coefficients(poly_traj)
        return {
            "mean": coeff.mean(dim=0),
            "std": coeff.std(dim=0),
            "min": coeff.min(dim=0).values,
            "max": coeff.max(dim=0).values,
        }
