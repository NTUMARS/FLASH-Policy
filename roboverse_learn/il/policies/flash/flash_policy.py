"""FLASH: history-to-future polynomial flow matching with a DiT backbone.

Expert actions are normalized, fitted by least squares and, when enabled,
corrected with KKT position and velocity constraints. Coefficient rows are then
scale-normalized. The constraints enforce C1 continuity of expert training
targets; independently predicted chunks are not projected at inference.

Measured proprioceptive history is fitted with Tikhonov regularization in the
same coefficient space. Training transports the noisy history coefficients to
the expert coefficients using flow-matching MSE and a weighted one-step
coefficient consistency loss.

The supplied flash.yaml uses degree 6, fit padding 1 and C1 target constraints.
Inference decodes the generated coefficients
into positions and analytic derivatives; the evaluation runner supplies the
physical-time scaling for velocity commands.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import reduce

from roboverse_learn.il.utils.models.flow_net import FlowTransformer
from roboverse_learn.il.utils.vision.multi_image_obs_encoder import MultiImageObsEncoder
from roboverse_learn.il.utils.normalizer import LinearNormalizer
from roboverse_learn.il.utils.pytorch_util import dict_apply
from roboverse_learn.il.policies.base_image_policy import BaseImagePolicy


class FlashDiTImagePolicy(BaseImagePolicy):
    """
    FLASH (Polynomial-to-Polynomial) Flow Matching Policy with DiT backbone.

    Combines FLASH-G's polynomial coefficient generation with A2A's informative
    flow-start strategy. Instead of transporting Gaussian noise → target coefficients,
    FLASH transports history polynomial coefficients → target polynomial coefficients.

    History is fitted with Tikhonov regularization from the last n_obs_steps
    measured states. Training adds a coefficient consistency loss to the
    flow-matching objective.
    """

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        # ---- FLASH-G-inherited hyperparameters ----
        poly_order: int = 3,
        basis_type: str = "legendre",
        fit_pad: int = 0,
        boundary_constraint: bool = False,
        # ---- FLASH-specific hyperparameters ----
        history_noise_std: float = 0.0,
        history_reg_lambda: float = 1e-2,
        consistency_weight: float = 1.0,
        flash_solver: str = "euler",
        poly_prefix: int | None = None,
        poly_suffix: int = 0,
        **kwargs,
    ):
        super().__init__()

        assert obs_as_global_cond, (
            "FLASH only supports obs_as_global_cond=True. "
            "Inpainting in polynomial-coefficient space is not meaningful."
        )

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]

        # ---- FLASH-G parameters ----
        self.poly_order = poly_order
        self.num_coeff = poly_order + 1
        self.basis_type = basis_type
        self._anchor_idx = n_obs_steps - 1
        self.fit_pad = fit_pad
        self.boundary_constraint = boundary_constraint

        # ---- FLASH parameters ----
        self.history_noise_std = history_noise_std
        self.history_reg_lambda = history_reg_lambda
        self.consistency_weight = consistency_weight
        self.flash_solver = flash_solver
        self.poly_prefix = poly_prefix
        self.poly_suffix = poly_suffix

        # Eval-time C_h robustness probe: std of Gaussian noise injected into the
        # history coefficients at INFERENCE ONLY (never during training; distinct
        # from history_noise_std, which is training-only). Not a ctor arg — set
        # post-hoc by the eval runner; 0.0 = off. Declared here so the eval
        # runner can detect support via hasattr().
        self.eval_history_noise_std = 0.0
        self.eval_history_noise_seed = 0
        self._eval_history_noise_gen = None

        # ==================================================================
        # Build polynomial basis matrices
        #
        # When poly_prefix is set (e.g. 1), the polynomial covers a REDUCED
        # region: n_action_steps + poly_prefix points.  This focuses the
        # polynomial expressiveness on the action region while keeping a few
        # extra observation points for velocity continuity at the anchor.
        #
        # When poly_prefix is None, the full horizon is used (original FLASH-G
        # behavior) — the polynomial covers all n_obs + n_action points.
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

        build_basis = (self._build_legendre_basis if basis_type == "legendre"
                       else self._build_power_basis if basis_type == "power"
                       else None)
        if build_basis is None:
            raise ValueError(f"Unknown basis_type: {basis_type}. Use 'legendre' or 'power'.")

        # ==================================================================
        # Basis matrices — ORIGINAL approach (no re-projection)
        #
        # _lstsq_matrix: (C, H_poly_ext) — fitted on extended s-grid
        # _eval_matrix:  (H_poly, C) — evaluated at middle s-values
        # ==================================================================

        s_ext = torch.linspace(0.0, 1.0, H_poly_ext, dtype=torch.float64)
        s_mid = s_ext[fit_pad: fit_pad + H_poly] if fit_pad > 0 else s_ext

        eval_ext, d1_ext, _ = build_basis(s_ext, poly_order)
        StS = eval_ext.T @ eval_ext
        lstsq_mat = torch.linalg.solve(StS, eval_ext.T)

        eval_mat, d1_mat, d2_mat = build_basis(s_mid, poly_order)

        coeff_scale = torch.norm(lstsq_mat, dim=1).clamp(min=1e-8)

        cond_number = torch.linalg.cond(StS).item()
        if poly_prefix is not None:
            pp_desc = f" (poly_prefix={poly_prefix}, poly_suffix={poly_suffix})"
        else:
            pp_desc = f" (full horizon={horizon})"
        print(f"[FLASH] basis={basis_type}, degree={poly_order}, H_poly={H_poly}{pp_desc}"
              f"{f', fit_pad={fit_pad} (H_poly_ext={H_poly_ext})' if fit_pad else ''}")
        print(f"[FLASH] cond(S^TS)={cond_number:.2e}")
        print(f"[FLASH] coeff scales: {[f'{x:.4f}' for x in coeff_scale.tolist()]}")

        self.register_buffer("_eval_matrix", eval_mat.float())
        self.register_buffer("_d1_matrix", d1_mat.float())
        self.register_buffer("_d2_matrix", d2_mat.float())
        self.register_buffer("_lstsq_matrix", lstsq_mat.float())
        self.register_buffer("_coeff_scale",
                             coeff_scale.float().view(1, -1, 1))
        self.register_buffer("_time_points", s_mid.float())

        # ==================================================================
        # FLASH: History polynomial fitting
        #
        # poly_prefix=None: Tikhonov on target's s-grid (original, 98% pick_cube)
        # poly_prefix>=0:   subset of target's s-grid (first n_obs rows of eval_mat)
        # ==================================================================

        if poly_prefix is None:
            S_hist = eval_mat[:n_obs_steps, :].double()
            StS_hist = S_hist.T @ S_hist
            hist_cond_raw = torch.linalg.cond(StS_hist).item()
            StS_reg = StS_hist + history_reg_lambda * torch.eye(C, dtype=torch.float64)
            history_lstsq = torch.linalg.solve(StS_reg, S_hist.T).float()
            hist_cond_final = torch.linalg.cond(StS_reg).item()
            self.register_buffer("_history_lstsq_matrix", history_lstsq)
            print(f"[FLASH] history fit (shared grid + Tikhonov): {n_obs_steps} pts → {C} coeffs, "
                  f"cond(S^TS) raw={hist_cond_raw:.2e} → regularized={hist_cond_final:.2e} (λ={history_reg_lambda})")
        else:
            if H_poly == n_obs_steps:
                S_hist = eval_mat.double() if fit_pad > 0 else eval_ext.double()
            else:
                S_hist = eval_mat[:n_obs_steps, :].double()
            StS_hist = S_hist.T @ S_hist
            hist_cond_raw = torch.linalg.cond(StS_hist).item()
            if history_reg_lambda > 0:
                StS_hist = StS_hist + history_reg_lambda * torch.eye(C, dtype=torch.float64)
            hist_cond_final = torch.linalg.cond(StS_hist).item()
            history_lstsq = torch.linalg.solve(StS_hist, S_hist.T).float()
            self.register_buffer("_history_lstsq_matrix", history_lstsq)
            n_hist_pts = S_hist.shape[0]
            print(f"[FLASH] history fit: {n_hist_pts}/{H_poly} pts, "
                  f"cond={hist_cond_raw:.2e}"
                  f"{f' → {hist_cond_final:.2e} (λ={history_reg_lambda})' if history_reg_lambda > 0 else ''}")
        print(f"[FLASH] history_noise_std={history_noise_std}, solver={flash_solver}")

        # ---- Boundary constraint matrices ----
        if boundary_constraint:
            bc_anchor = fit_pad + self._anchor_idx
            bc_overlap = fit_pad + self._anchor_idx + n_action_steps
            if bc_overlap >= H_poly_ext:
                print(f"[FLASH] WARNING: bc_overlap={bc_overlap} >= H_poly_ext={H_poly_ext}. "
                      f"Increase fit_pad for valid boundary constraint.")
            elif bc_overlap == H_poly_ext - 1:
                print(f"[FLASH] WARNING: bc_overlap at polynomial edge (index {bc_overlap} of "
                      f"{H_poly_ext}). Velocity uses backward diff. Consider fit_pad += 1.")
            bc_pos_indices = torch.tensor([bc_anchor, bc_overlap], dtype=torch.long)
            A_pos = eval_ext[bc_pos_indices.tolist(), :]
            A_vel = d1_ext[bc_pos_indices.tolist(), :]
            A = torch.cat([A_pos, A_vel], dim=0)
            StS_inv = torch.linalg.solve(StS, torch.eye(StS.shape[0], dtype=torch.float64))
            StS_inv_AT = StS_inv @ A.T
            M = A @ StS_inv_AT
            bc_correction = StS_inv_AT @ torch.linalg.inv(M)
            self.register_buffer("_bc_eval_pos", A_pos.float())
            self.register_buffer("_bc_eval_vel", A_vel.float())
            self.register_buffer("_bc_correction", bc_correction.float())
            self.register_buffer("_bc_pos_indices", bc_pos_indices)
            print(f"[FLASH] boundary_constraint=C1, pin indices={bc_pos_indices.tolist()} "
                  f"in H_poly_ext={H_poly_ext} (4 constraints: 2 pos + 2 vel)")

        self._num_pinned = 0
        self._num_coeff_model = poly_order + 1

        # ==================================================================
        # Flow Matching backbone (DiT) — operates in NORMALIZED coefficient space
        # ==================================================================
        input_dim = action_dim
        global_cond_dim = obs_feature_dim * n_obs_steps

        model = FlowTransformer(
            input_dim=input_dim,
            condition_dim=global_cond_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            time_embed_dim=diffusion_step_embed_dim,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        self.num_inference_steps = num_inference_steps

    # ==================================================================
    #  Basis construction (identical to FLASH-G)
    # ==================================================================

    @staticmethod
    def _build_legendre_basis(
        s: torch.Tensor, degree: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build shifted Legendre polynomial basis on s in [0, 1]."""
        x = 2 * s - 1
        H = len(s)
        C = degree + 1

        P = torch.zeros(H, C, dtype=s.dtype)
        dP = torch.zeros(H, C, dtype=s.dtype)
        d2P = torch.zeros(H, C, dtype=s.dtype)

        P[:, 0] = 1.0
        dP[:, 0] = 0.0
        d2P[:, 0] = 0.0

        if C > 1:
            P[:, 1] = x
            dP[:, 1] = 1.0
            d2P[:, 1] = 0.0

        for n in range(1, C - 1):
            P[:, n + 1] = (
                (2 * n + 1) * x * P[:, n] - n * P[:, n - 1]
            ) / (n + 1)
            dP[:, n + 1] = (
                (2 * n + 1) * (P[:, n] + x * dP[:, n]) - n * dP[:, n - 1]
            ) / (n + 1)
            d2P[:, n + 1] = (
                (2 * n + 1) * (2 * dP[:, n] + x * d2P[:, n]) - n * d2P[:, n - 1]
            ) / (n + 1)

        eval_mat = P
        d1_mat = 2 * dP
        d2_mat = 4 * d2P

        return eval_mat, d1_mat, d2_mat

    @staticmethod
    def _build_power_basis(
        s: torch.Tensor, degree: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build standard power polynomial basis [s^n, s^{n-1}, ..., s, 1]."""
        powers = torch.arange(degree, -1, -1, dtype=s.dtype)
        eval_mat = s.unsqueeze(-1).pow(powers.unsqueeze(0))

        d1_powers = (powers - 1).clamp(min=0)
        d1_mat = powers.unsqueeze(0) * s.unsqueeze(-1).pow(d1_powers.unsqueeze(0))
        d1_mat[:, powers < 1] = 0

        d2_powers = (powers - 2).clamp(min=0)
        d2_coeff = powers * (powers - 1)
        d2_mat = d2_coeff.unsqueeze(0) * s.unsqueeze(-1).pow(d2_powers.unsqueeze(0))
        d2_mat[:, powers < 2] = 0

        return eval_mat, d1_mat, d2_mat

    # ==================================================================
    #  Polynomial operations
    # ==================================================================

    def _trajectory_to_coefficients(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Fit polynomial to a full trajectory via batch least-squares.

        Args:
            trajectory: (B, H_ext, Da)

        Returns:
            (B, C, Da) — normalized polynomial coefficients
        """
        raw_coeff = torch.einsum("ch,bhd->bcd", self._lstsq_matrix, trajectory)
        return raw_coeff / self._coeff_scale

    def _history_to_coefficients(self, history_states: torch.Tensor) -> torch.Tensor:
        """Fit polynomial to history states.

        Args:
            history_states: (B, n_obs_steps, Da) or (B, H_poly_ext, Da) if shared

        Returns:
            (B, C, Da) — normalized polynomial coefficients
        """
        raw_coeff = torch.einsum("ch,bhd->bcd", self._history_lstsq_matrix, history_states)
        return raw_coeff / self._coeff_scale

    def _apply_boundary_constraint(
        self,
        norm_coeff: torch.Tensor,
        trajectory: torch.Tensor,
        coeff_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply KKT C1 constraints to expert targets at both execution anchors.

        Derivative targets use finite differences of the expert samples.
        This correction is applied during training, not to inference outputs.
        """
        if coeff_scale is None:
            coeff_scale = self._coeff_scale
        raw = norm_coeff * coeff_scale
        idx = self._bc_pos_indices
        H_ext = trajectory.shape[1]

        cur_pos = torch.einsum("kc,bcd->bkd", self._bc_eval_pos, raw)
        tgt_pos = trajectory[:, idx, :]

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
        tgt_vel = torch.stack(tgt_vel_parts, dim=1)
        cur_vel = torch.einsum("kc,bcd->bkd", self._bc_eval_vel, raw)

        residual = torch.cat([cur_pos - tgt_pos, cur_vel - tgt_vel], dim=1)
        correction = torch.einsum("ck,bkd->bcd", self._bc_correction, residual)
        return (raw - correction) / coeff_scale

    def _coefficients_to_trajectory(
        self, norm_coeff: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate polynomial at all time points from NORMALIZED coefficients.

        Args:
            norm_coeff: (B, C, Da)

        Returns:
            (B, H, Da) — reconstructed trajectory
        """
        raw_coeff = norm_coeff * self._coeff_scale
        return torch.einsum("hc,bcd->bhd", self._eval_matrix, raw_coeff)

    def _compute_velocity(
        self,
        norm_coeff: torch.Tensor,
        total_time: float = 1.0,
    ) -> torch.Tensor:
        """Compute velocity at all time points."""
        raw_coeff = norm_coeff * self._coeff_scale
        return torch.einsum("hc,bcd->bhd", self._d1_matrix, raw_coeff) / total_time

    def _compute_acceleration(
        self,
        norm_coeff: torch.Tensor,
        total_time: float = 1.0,
    ) -> torch.Tensor:
        """Compute acceleration at all time points."""
        raw_coeff = norm_coeff * self._coeff_scale
        return torch.einsum("hc,bcd->bhd", self._d2_matrix, raw_coeff) / (total_time ** 2)

    # ==================================================================
    #  Inference
    # ==================================================================

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        local_cond=None,
        global_cond=None,
        generator=None,
        start_coeff=None,
        **kwargs,
    ) -> torch.Tensor:
        """Sample polynomial coefficients via ODE integration.

        FLASH difference from FLASH-G: when start_coeff is provided, the ODE starts
        from the history polynomial instead of Gaussian noise, dramatically
        reducing the required number of integration steps.

        Args:
            condition_data: (B, C, Da) — conditioning data (zeros for FLASH)
            condition_mask:  (B, C, Da) — boolean mask (all False for FLASH)
            global_cond:    (B, D) — observation features
            start_coeff:    (B, C, Da) — history polynomial coefficients (FLASH flow source).
                            If None, falls back to Gaussian noise (standard FLASH-G behavior).

        Returns:
            (B, C, Da) — sampled normalized polynomial coefficients
        """
        model = self.model

        if start_coeff is not None:
            trajectory = start_coeff.clone()
        else:
            trajectory = torch.randn(
                size=condition_data.shape,
                dtype=condition_data.dtype,
                device=condition_data.device,
                generator=generator,
            )

        time_steps = torch.linspace(0, 1.0, self.num_inference_steps + 1)

        for i in range(self.num_inference_steps):
            trajectory[condition_mask] = condition_data[condition_mask]

            t_start = time_steps[i].view(1).expand(trajectory.shape[0]).to(self.device)
            t_end = time_steps[i + 1].view(1).expand(trajectory.shape[0]).to(self.device)
            dt = (t_end - t_start).view(-1, 1, 1)

            if self.flash_solver == "euler":
                v = model(
                    trajectory, t_start, local_cond=local_cond, global_cond=global_cond
                )
                trajectory = trajectory + dt * v
            else:
                t_mid = t_start + (t_end - t_start) / 2
                v_start = model(
                    trajectory, t_start, local_cond=local_cond, global_cond=global_cond
                )
                trajectory_mid = trajectory + v_start * dt / 2
                v_mid = model(
                    trajectory_mid, t_mid, local_cond=local_cond, global_cond=global_cond
                )
                trajectory = trajectory + dt * v_mid

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Full inference pipeline: observation → history polynomial → flow → trajectory.

        FLASH pipeline:
            1. Normalize observations
            2. Encode observations into feature vector
            3. Fit history polynomial to past n_obs_steps states (FLASH-specific!)
            4. Flow Matching ODE: history_coeff → target_coeff (fewer steps)
            5. Evaluate polynomial → trajectory + velocity + acceleration
            6. Extract executable action slice
        """
        assert "past_action" not in obs_dict

        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B = value.shape[0]
        Da = self.action_dim
        To = self.n_obs_steps
        device, dtype = self.device, self.dtype

        # ---- Encode observations ----
        this_nobs = dict_apply(
            nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(B, -1)

        # ---- FLASH: Fit history polynomial to past states ----
        q_history_phys = obs_dict["agent_pos"][:, :To, :].to(device)
        q_history_norm = self.normalizer["action"].normalize(q_history_phys)
        history_coeff = self._history_to_coefficients(q_history_norm)

        # ---- Eval-time C_h perturbation (robustness probe; off unless the eval
        # runner sets eval_history_noise_std > 0). Same coefficient space and
        # units as the training-time history_noise_std. Drawn from a DEDICATED
        # generator so the global RNG stream — and thus env / domain-random
        # draws — is identical across sigma values in a sweep.
        sigma = float(self.eval_history_noise_std or 0.0)
        if sigma > 0.0 and not self.training:
            gen = self._eval_history_noise_gen
            if gen is None or gen.device != history_coeff.device:
                gen = torch.Generator(device=history_coeff.device)
                gen.manual_seed(int(self.eval_history_noise_seed))
                self._eval_history_noise_gen = gen
            history_coeff = history_coeff + sigma * torch.randn(
                history_coeff.shape, generator=gen,
                device=history_coeff.device, dtype=history_coeff.dtype)

        # ---- Prepare conditioning ----
        cond_data = torch.zeros(B, self.num_coeff, Da, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # ---- Flow Matching: history_coeff → target_coeff ----
        ncoeff_sampled = self.conditional_sample(
            cond_data, cond_mask,
            local_cond=None, global_cond=global_cond,
            start_coeff=history_coeff,
            **self.kwargs,
        )

        ncoeff_pred = ncoeff_sampled

        # ---- Evaluate polynomial ----
        naction_pred = self._coefficients_to_trajectory(ncoeff_pred)
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # ---- Velocity and acceleration ----
        nvel = self._compute_velocity(ncoeff_pred)
        nacce = self._compute_acceleration(ncoeff_pred)
        scale = self.normalizer["action"].params_dict["scale"].view(1, 1, -1)
        velocity = nvel / scale
        acceleration = nacce / scale

        # ---- Extract action chunk ----
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
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch: Dict) -> torch.Tensor:
        """Compute FM MSE plus weighted one-step coefficient consistency MSE.

        FLASH difference from FLASH-G:
          - Flow source (x_0) = history polynomial coefficients (not Gaussian noise)
          - FM velocity target = target_coeff - history_coeff (not target_coeff - noise)
          - Optional Gaussian noise added to history coefficients for robustness
        """
        assert "valid_mask" not in batch

        # ---- Normalize inputs ----
        nobs = self.normalizer.normalize(batch["obs"])
        nactions_full = self.normalizer["action"].normalize(batch["action"])
        B = nactions_full.shape[0]

        fp = self.fit_pad
        poly_start = self._poly_nf_start
        H_poly_ext = self._H_poly + 2 * fp
        poly_traj = nactions_full[:, poly_start:poly_start + H_poly_ext, :]

        # ---- Fit target polynomial (original approach, no re-projection) ----
        if "target_coeff" in batch:
            target_coeff = batch["target_coeff"]
        else:
            target_coeff = self._trajectory_to_coefficients(poly_traj)

        if self.boundary_constraint:
            target_coeff = self._apply_boundary_constraint(target_coeff, poly_traj)

        # ---- FLASH: Fit history polynomial (original approach) ----
        history_raw = batch["obs"]["agent_pos"][:, fp:fp + self.n_obs_steps, :].to(nactions_full.device)
        history_actions = self.normalizer["action"].normalize(history_raw)
        history_coeff = self._history_to_coefficients(history_actions)

        if self.training and self.history_noise_std > 0:
            history_coeff = history_coeff + torch.randn_like(history_coeff) * self.history_noise_std

        # ---- Encode observations ----
        obs_start = fp
        this_nobs = dict_apply(
            nobs, lambda x: x[:, obs_start: obs_start + self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(B, -1)

        # ---- FLASH Flow Matching: history_coeff → target_coeff ----
        timesteps = torch.rand(B, device=target_coeff.device)
        t_exp = timesteps.view(-1, 1, 1)

        coeff_inter = (1 - t_exp) * history_coeff + t_exp * target_coeff
        vector_fm_true = target_coeff - history_coeff

        # ---- Model prediction ----
        vector_fm_pred = self.model(
            coeff_inter, timesteps, local_cond=None, global_cond=global_cond
        )

        # ---- FM loss ----
        loss_fm = F.mse_loss(vector_fm_pred, vector_fm_true, reduction="none")
        loss_fm = reduce(loss_fm, "b ... -> b (...)", "mean").mean()

        loss = loss_fm

        # ---- Consistency loss ----
        # Compare the training prediction with the target coefficients.
        if self.consistency_weight > 0:
            t_zero = torch.zeros(B, device=target_coeff.device)
            v_at_zero = self.model(
                history_coeff, t_zero, local_cond=None, global_cond=global_cond
            )
            predicted_coeff = history_coeff + v_at_zero
            consistency_loss = F.mse_loss(predicted_coeff, target_coeff)
            loss = loss + self.consistency_weight * consistency_loss

        return loss

    # ==================================================================
    #  Diagnostics
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

    @torch.no_grad()
    def compute_flash_distance(self, batch: Dict) -> Dict[str, float]:
        """Measure transport distance between history and target coefficients."""
        nactions_full = self.normalizer["action"].normalize(batch["action"])
        fp = self.fit_pad

        poly_start = self._poly_nf_start
        H_poly_ext = self._H_poly + 2 * fp
        poly_traj = nactions_full[:, poly_start:poly_start + H_poly_ext, :]
        target_coeff = self._trajectory_to_coefficients(poly_traj)

        history_actions = nactions_full[:, fp:fp + self.n_obs_steps, :]
        history_coeff = self._history_to_coefficients(history_actions)

        flash_dist = (target_coeff - history_coeff).norm(dim=(1, 2)).mean().item()
        target_norm = target_coeff.norm(dim=(1, 2)).mean().item()
        noise_expected = (self.num_coeff * self.action_dim) ** 0.5

        return {
            "flash_distance": flash_dist,
            "noise_expected_distance": noise_expected + target_norm,
            "reduction_ratio": flash_dist / (noise_expected + target_norm),
        }
