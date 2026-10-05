# roboverse_learn/il/policies/act/act_image_policy.py
"""ACT adapter for the shared image-policy training and 2D evaluation pipeline.

Uses the current frame, the shared position/action normalizer, and ACT's
ImageNet image normalization. The generic runner owns optimization and EMA.
"""
import argparse
from typing import Dict

import torch
import torchvision.transforms as transforms
from torch.nn import functional as F

from roboverse_learn.il.policies.act.detr.main import get_args_parser
from roboverse_learn.il.policies.act.detr.models import build_ACT_model
from roboverse_learn.il.policies.act.policy import kl_divergence
from roboverse_learn.il.policies.base_image_policy import BaseImagePolicy
from roboverse_learn.il.utils.normalizer import LinearNormalizer
from roboverse_learn.il.utils.kwargs_guard import reject_unknown_kwargs


def _build_act_model_no_cli(overrides: dict):
    """Build DETRVAE without reading ``sys.argv``.

    ``build_ACT_model_and_optimizer`` (detr/main.py:74-94) parses the real
    process arguments, while ``--task_name`` is required by ``get_args_parser``
    (main.py:62). Hydra's ``key=value`` command line does not provide it, so
    argparse would call ``sys.exit``. Pass an explicit argument list to
    ``parse_known_args`` to obtain the default Namespace, then apply overrides
    with ``setattr`` (matching main.py:78-79). This function only builds the
    model: ``DefaultRunner.to(device)`` manages device placement and
    ``train_config`` owns the optimizer. The legacy Franka path through
    ``build_ACT_model_and_optimizer`` remains unchanged.
    """
    parser = argparse.ArgumentParser(parents=[get_args_parser()], add_help=False)
    args, _ = parser.parse_known_args(args=["--task_name", "hydra_wrapped"])
    for k, v in overrides.items():
        setattr(args, k, v)
    return build_ACT_model(args)


class ACTImagePolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        horizon,
        n_action_steps,
        n_obs_steps,
        chunk_size=8,
        kl_weight=10,
        hidden_dim=256,
        dim_feedforward=1024,
        enc_layers=2,
        dec_layers=4,
        nheads=4,
        dropout=0.1,
        backbone="resnet18",
        position_embedding="sine",
        camera_names=("head_cam",),
        **kwargs,
    ):
        super().__init__()
        # Reject unknown constructor arguments immediately so misspelled
        # overrides, YAML/signature drift, and obsolete checkpoint keys cannot
        # disappear silently (see utils/kwargs_guard.py).
        reject_unknown_kwargs(self, kwargs)
        action_dim = shape_meta["action"]["shape"][0]
        agent_pos_dim = shape_meta["obs"]["agent_pos"]["shape"][0]
        # DETRVAE's action_head and encoder_*_proj share one state_dim
        # (detr_vae.py:53,70-71); there is no separate action_dim.
        assert action_dim == agent_pos_dim, (
            f"DETRVAE requires state_dim == action_dim; "
            f"got agent_pos={agent_pos_dim} vs action={action_dim}"
        )
        assert n_action_steps <= chunk_size, (
            f"n_action_steps({n_action_steps}) cannot exceed chunk_size({chunk_size})"
        )
        assert (n_obs_steps - 1) + chunk_size <= horizon, (
            f"Supervision window exceeds the horizon: (n_obs_steps-1)+chunk_size="
            f"{(n_obs_steps - 1) + chunk_size} > horizon({horizon})"
        )
        # This implementation creates one camera slot with unsqueeze(1) in
        # _current_frame. DETRVAE indexes images using camera_names
        # (detr_vae.py:121-125), so reject multiple cameras until
        # _current_frame supports them.
        assert tuple(camera_names) == ("head_cam",), (
            f"ACTImagePolicy currently supports only one camera ('head_cam',); "
            f"got {tuple(camera_names)}"
        )

        self.model = _build_act_model_no_cli(dict(
            state_dim=action_dim,
            num_queries=chunk_size,
            camera_names=list(camera_names),
            hidden_dim=hidden_dim,
            dim_feedforward=dim_feedforward,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            nheads=nheads,
            dropout=dropout,
            backbone=backbone,
            position_embedding=position_embedding,
        ))
        self.normalizer = LinearNormalizer()
        # Match the ImageNet normalization in ACTPolicy.__call__
        # (policy.py:19-21). Inputs are in [0, 1]: the data path only divides
        # by 255 and does not apply head_cam's [-1, 1] normalizer.
        self.image_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.chunk_size = chunk_size
        self.kl_weight = kl_weight
        # The following attributes are retained for reporting. The ACT block in
        # eval_config_report reads them directly so every policy YAML parameter
        # can be recovered from the policy instance.
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.nheads = nheads
        self.dropout = dropout
        self.position_embedding = position_embedding
        self.camera_names = tuple(camera_names)
        self.backbone_name = backbone
        self.kwargs = kwargs

    @property
    def num_inference_steps(self):
        return 1

    @num_inference_steps.setter
    def num_inference_steps(self, value):
        if value not in (None, 1):
            print(
                f"[ACTImagePolicy] Ignoring num_inference_steps={value}"
            )

    # ---------- Shared pipeline interface ----------
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _current_frame(self, obs_dict: Dict[str, torch.Tensor]):
        """Select the observation frame at index n_obs_steps-1."""
        t0 = self.n_obs_steps - 1
        nagent = self.normalizer["agent_pos"].normalize(obs_dict["agent_pos"])
        qpos = nagent[:, t0]                                   # (B, D)
        image = obs_dict["head_cam"][:, t0].unsqueeze(1)       # (B, K=1, 3, H, W), [0,1]
        image = self.image_normalize(image)
        return qpos, image

    def compute_loss(self, batch):
        """Compute the action reconstruction and KL losses."""
        qpos, image = self._current_frame(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        t0 = self.n_obs_steps - 1
        target = nactions[:, t0 : t0 + self.chunk_size]        # (B, chunk, D) = [7:15]
        is_pad = torch.zeros(
            target.shape[:2], dtype=torch.bool, device=target.device
        )
        a_hat, _, (mu, logvar) = self.model(qpos, image, None, target, is_pad)
        total_kld, _, _ = kl_divergence(mu.float(), logvar.float())
        all_l1 = F.l1_loss(target, a_hat, reduction="none")
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        return l1 + total_kld[0] * self.kl_weight

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Match ACTPolicy.__call__ inference and the shared output interface.

        The output chunk starts at the current frame, matching the temporal
        alignment of fm_dit's execution slice ``action_pred[:, To-1 :
        To-1+n_action_steps]`` (fm_dit_image_policy.py:288-291).
        ``action_pred`` is also consumed by the runner's periodic sampling
        diagnostics (default_runner.py requires this key).
        """
        qpos, image = self._current_frame(obs_dict)
        a_hat, _, (_, _) = self.model(qpos, image, None)       # (B, chunk, D), normalized
        action_pred = self.normalizer["action"].unnormalize(a_hat)
        action = action_pred[:, : self.n_action_steps]
        return {"action": action, "action_pred": action_pred}
