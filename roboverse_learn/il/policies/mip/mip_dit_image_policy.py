from typing import Dict

import torch
import torch.nn.functional as F

from roboverse_learn.il.utils.models.flow_net import FlowTransformer
from roboverse_learn.il.utils.vision.multi_image_obs_encoder import MultiImageObsEncoder
from roboverse_learn.il.utils.normalizer import LinearNormalizer
from roboverse_learn.il.utils.pytorch_util import dict_apply
from roboverse_learn.il.policies.base_image_policy import BaseImagePolicy


def _get_norm(x: torch.Tensor, norm_type: str) -> torch.Tensor:
    # ported verbatim from official mip/losses.py::get_norm
    if norm_type == "l2":
        # squared L2 (no sqrt)
        return torch.sum(x * x, dim=-1)
    elif norm_type == "l1":
        return torch.sum(torch.abs(x), dim=-1)
    elif norm_type == "smooth_l1":
        # per-element smooth L1, then sum over last dim
        return torch.sum(
            F.smooth_l1_loss(x, torch.zeros_like(x), reduction="none"), dim=-1
        )
    else:
        raise NotImplementedError(f"Norm type {norm_type} not implemented.")


class MIPDiTImagePolicy(BaseImagePolicy):
    """Minimal Iterative Policy (MIP) baseline.

    Ported from the official implementation of "Much Ado About Noising:
    Dispelling the Myths of Generative Robotic Control" (arXiv 2512.01809):
    mip/losses.py::mip_loss and mip/samplers.py::mip_sampler (the simplified
    form, which is the official default). Everything except the MIP objective
    (obs encoder, DiT backbone, horizon/steps, normalizer) is kept identical
    to fm_dit for fair comparison.

    Training (both terms regress the GT action a, t* = t_two_step = 0.9):
        pred_0 = net(x=0,              t=0,  obs)
        pred_1 = net(x=a + (1-t*)*z,   t=t*, obs),  z ~ N(0, I)
        loss = loss_scale * mean( |pred_0 - a|^2 / t*^2
                                + |pred_1 - a|^2 / (1-t*)^2 )
    """

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=2,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        t_two_step=0.9,
        loss_scale=0.1,
        norm_type="l2",
        **kwargs,
    ):
        super().__init__()

        # MIP port only supports global conditioning (the pipeline default);
        # in this mode fm_dit's inpainting mask is a no-op, so dropping it is
        # numerically identical.
        assert obs_as_global_cond, "MIPDiTImagePolicy only supports obs_as_global_cond=True"

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

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
            # RoPE cache length must be >= transformer sequence length (action horizon).
            max_seq_len=horizon,
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
        self.t_two_step = t_two_step
        self.loss_scale = loss_scale
        self.norm_type = norm_type
        self.kwargs = kwargs

        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: flat dict of obs keys (head_cam (B,To,C,H,W), agent_pos (B,To,Da))
        result: must include "action" key
        """
        assert "past_action" not in obs_dict  # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B = value.shape[0]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        # condition through global feature; encode obs once, reuse for both steps
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(B, -1)

        # ---- official mip_sampler (simplified form), mip/samplers.py:120 ----
        s = torch.zeros(B, device=device)
        t = torch.full((B,), self.t_two_step, device=device, dtype=torch.float32)
        act_0 = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        act_pred = self.model(act_0, s, local_cond=None, global_cond=global_cond)
        if self.num_inference_steps != 1:
            act_pred = self.model(act_pred, t, local_cond=None, global_cond=global_cond)

        # unnormalize prediction
        naction_pred = act_pred[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])  # B * horizon * action_dim
        batch_size = nactions.shape[0]

        # condition through global feature (identical to fm_dit)
        this_nobs = dict_apply(nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(batch_size, -1)

        act = nactions
        device = act.device

        # ---- official mip_loss (simplified form), mip/losses.py:181 ----
        s = torch.zeros(batch_size, device=device)
        t = torch.full((batch_size,), self.t_two_step, device=device, dtype=torch.float32)
        # major difference compared to tsd: remove stochasticity in input
        act_0 = torch.zeros_like(act)
        noise = torch.empty_like(act).normal_(0, 1)
        act_t = act + (1 - self.t_two_step) * noise

        act_pred_0 = self.model(act_0, s, local_cond=None, global_cond=global_cond)
        act_pred_1 = self.model(act_t, t, local_cond=None, global_cond=global_cond)

        loss0 = _get_norm((act_pred_0 - act) / self.t_two_step, self.norm_type)
        loss1 = _get_norm((act_pred_1 - act) / (1 - self.t_two_step), self.norm_type)
        loss = self.loss_scale * torch.mean(loss0 + loss1)
        # expose per-term components for the trainer's optional logging
        # (draft = regression step, refine = denoising step; both loss_scale-weighted)
        self.last_loss_components = {
            "mip_loss_draft": self.loss_scale * loss0.detach().mean().item(),
            "mip_loss_refine": self.loss_scale * loss1.detach().mean().item(),
        }
        return loss
