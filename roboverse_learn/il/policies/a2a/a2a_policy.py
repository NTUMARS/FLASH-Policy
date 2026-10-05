"""
Action-to-Action Flow Matching Policy (A2A)

A flow matching policy that directly transforms history action distributions
to future action distributions, conditioned on visual observations.

Architecture:
    History States [s_{t-n+1}, ..., s_t] --encode--> history_latents (x_0)
    Visual Obs [img_{t-n+1}, ..., img_t] --encode--> obs_latents (condition)
    
    Flow Matching: x_0 --flow(condition)--> x_1 (future_action_latents)
    
    x_1 --decode--> Future Actions [a_t, a_{t+1}, ..., a_{t+k}]
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from roboverse_learn.il.utils.normalizer import LinearNormalizer
from roboverse_learn.il.utils.pytorch_util import dict_apply
from roboverse_learn.il.policies.base_image_policy import BaseImagePolicy

from roboverse_learn.il.utils.models.flow_net import SimpleFlowNet
from roboverse_learn.il.policies.a2a.action_ae import CNNActionEncoder, SimpleActionDecoder
from roboverse_learn.il.utils.vision.multi_image_obs_encoder import MultiImageObsEncoder
from roboverse_learn.il.utils.flow.flow_matchers import TorchFlowMatcher


class A2AImagePolicy(BaseImagePolicy):
    """
    Action-to-Action Flow Matching Policy.
    
    Flow Matching from history states to future actions, conditioned on visual observations.
    
    - Flow START: History states (past n_obs_steps proprioceptive states)
    - Flow TARGET: Future actions (next n_action_steps)
    - CONDITION: Visual observation latents
    """

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        # A2A specific params
        flow_net,
        flow_matcher: TorchFlowMatcher,
        decode_flow_latents=True,
        consistency_weight=1.0,
        enc_contrastive_weight=1e-4,
        flow_contrastive_weight=0.0,
        latent_dim=512,
        action_ae=None,
        **kwargs,
    ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        self.decode_flow_latents = decode_flow_latents
        self.consistency_weight = consistency_weight
        self.enc_contrastive_weight = enc_contrastive_weight
        self.flow_contrastive_weight = flow_contrastive_weight
        self.latent_dim = latent_dim
        self.num_sampling_steps = flow_matcher.num_sampling_steps

        self.flow_matcher = flow_matcher
        self.action_ae = action_ae

        # Visual observation encoder
        self.obs_encoder = obs_encoder
        self.obs_projector = nn.Linear(
            obs_feature_dim * n_obs_steps,
            latent_dim
        )

        # Flow network with condition support
        self.flow_net = SimpleFlowNet(
            input_dim=latent_dim,
            hidden_dim=flow_net.hidden_dim,
            output_dim=latent_dim,
            num_layers=flow_net.num_layers,
            mlp_ratio=flow_net.mlp_ratio,
            dropout=flow_net.dropout,
            condition_dim=latent_dim,  # obs_latents as condition
        )
        
        # History state encoder
        # Note: n_obs_steps must be >= 8 for CNN with 3 layers (stride=2 each)
        self.history_action_encoder = CNNActionEncoder(
            pred_horizon=n_obs_steps,  # History length
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=action_ae.net.enc_hidden_dim,
        )
        
        # Future action encoder/decoder
        # Data structure with horizon=16, n_obs_steps=8:
        #   Time:      t-7  t-6  ...  t   t+1  ...  t+8
        #   state:     s0   s1   ... s7   s8   ... s15   (16 frames)
        #   action:    a0   a1   ... a7   a8   ... a15   (16 frames)
        #
        # History: state[0:8] = [s_{t-7}, ..., s_t]  (8 frames)
        # Future:  action[7:15] = [a_t, ..., a_{t+7}]  (8 frames from current time)
        future_horizon = n_action_steps  # 8 future actions
        self.future_horizon = future_horizon
        
        self.action_encoder = CNNActionEncoder(
            pred_horizon=future_horizon,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=action_ae.net.enc_hidden_dim,
        )
        self.action_decoder = SimpleActionDecoder(
            dec_hidden_dim=action_ae.net.dec_hidden_dim,
            latent_dim=latent_dim,
            pred_horizon=future_horizon,
            action_dim=action_dim,
            num_layers=action_ae.net.num_layers,
            dropout=action_ae.net.dropout,
        )

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]

        # Encode visual observations
        # reshape B, T, ... to B*T for image encoding
        this_nobs = dict_apply(nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(batch_size, -1)
        obs_latents = self.obs_projector(nobs_features)

        # Encode history states (8 frames ending at current time)
        history_states = nobs["agent_pos"][:, :self.n_obs_steps, :]  # (B, 8, D)
        history_latents = self.history_action_encoder(history_states)
        
        # Encode future actions (8 frames starting from current time)
        future_start = self.n_obs_steps - 1  # = 7 (current time index)
        future_end = future_start + self.n_action_steps  # = 7 + 8 = 15
        future_actions = nactions[:, future_start:future_end, :]  # (B, 8, D)
        future_action_latents = self.action_encoder(future_actions)
        
        # Flow matching: history_latents -> future_action_latents, conditioned on obs_latents
        flow_loss, metrics = self.flow_matcher.compute_loss(
            self.flow_net,
            target=future_action_latents,
            start=history_latents,  # History states as flow source
            global_cond=obs_latents,  # Visual observations as condition
        )
        
        loss = flow_loss
        metrics['flow_loss'] = flow_loss.item()

        # Encoder contrastive loss
        if self.enc_contrastive_weight > 0:
            image_features = obs_latents.view(batch_size, -1)
            action_features = future_action_latents.view(batch_size, -1)
            contrastive_loss = self._compute_contrastive_loss(image_features, action_features)
            loss += self.enc_contrastive_weight * contrastive_loss
            metrics['enc_contrastive_loss'] = contrastive_loss.item()

        # Flow latent decoding
        if self.decode_flow_latents:
            # Sample with history states as start and obs as condition
            action_latents_pred = self.flow_matcher.sample(
                self.flow_net,
                shape=(batch_size, self.latent_dim),
                device=obs_latents.device,
                start=history_latents,
                num_steps=self.num_sampling_steps,
                global_cond=obs_latents,
            )

            if self.consistency_weight > 0:
                consistency_loss = F.mse_loss(action_latents_pred, future_action_latents)
                loss += self.consistency_weight * consistency_loss
                metrics['consistency_loss'] = consistency_loss.item()

            if self.flow_contrastive_weight > 0:
                image_features = obs_latents.view(batch_size, -1)
                action_features = action_latents_pred.view(batch_size, -1)
                contrastive_loss = self._compute_contrastive_loss(image_features, action_features)
                loss += self.flow_contrastive_weight * contrastive_loss
                metrics['flow_contrastive_loss'] = contrastive_loss.item()

            if self.action_ae["flow_recon_weight"] > 0:
                actions_recon = self.action_decoder(action_latents_pred)
                action_recon_loss = F.l1_loss(actions_recon, future_actions)
                metrics['flow_action_recon_loss'] = action_recon_loss.item()
                loss += self.action_ae["flow_recon_weight"] * action_recon_loss
        else:
            action_latents_pred = future_action_latents

        # Encoder reconstruction losses
        if self.action_ae["enc_recon_weight"] > 0:
            actions_recon = self.action_decoder(future_action_latents)
            action_recon_loss = F.l1_loss(actions_recon, future_actions)
            metrics['enc_action_recon_loss'] = action_recon_loss.item()
            loss += self.action_ae["enc_recon_weight"] * action_recon_loss

        return loss

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B = value.shape[0]

        # Encode visual observations
        # reshape B, T, ... to B*T for image encoding
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(B, -1)
        obs_latents = self.obs_projector(nobs_features)

        # Encode history states
        # These are the past n_obs_steps states: [s_{t-n+1}, ..., s_t]
        history_states = nobs["agent_pos"][:, :self.n_obs_steps, :]  # (B, n_obs_steps, D)
        history_latents = self.history_action_encoder(history_states)
        
        # Run flow sampling: history -> future, conditioned on visual obs
        action_latents_pred = self.flow_matcher.sample(
            self.flow_net,
            shape=(B, self.latent_dim),
            device=obs_latents.device,
            num_steps=self.num_sampling_steps,
            start=history_latents,  # History states as flow source
            global_cond=obs_latents,  # Visual observations as condition
            return_traces=False
        )

        with torch.no_grad():
            action_pred = self.action_decoder(action_latents_pred)

        # unnormalize prediction
        action_pred = self.normalizer["action"].unnormalize(action_pred)

        # action_pred is [a_t, a_{t+1}, ..., a_{t+7}] (8 frames)
        # All frames are future actions starting from current time
        action = action_pred[:, :self.n_action_steps]  # (B, 8, D)

        result = {"action": action, "action_pred": action_pred}
        return result

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    @torch.no_grad()
    def get_latents_for_visualization(self, batch):
        """
        Extract history and future latents for t-SNE visualization.
        
        Returns:
            history_latents: (B, latent_dim) - encoded from past states
            future_latents: (B, latent_dim) - encoded from future actions
        """
        # normalize input
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        
        # Encode history states (past n_obs_steps states)
        history_states = nobs["agent_pos"][:, :self.n_obs_steps, :]  # (B, 8, D)
        history_latents = self.history_action_encoder(history_states)
        
        # Encode future actions (next n_action_steps actions)
        future_start = self.n_obs_steps - 1
        future_end = future_start + self.n_action_steps
        future_actions = nactions[:, future_start:future_end, :]  # (B, 8, D)
        future_latents = self.action_encoder(future_actions)
        
        return history_latents, future_latents

    @torch.no_grad()
    def get_flow_trajectories(self, batch, num_steps=None, n_samples=5):
        """
        Get flow trajectories for visualization.
        
        Args:
            batch: Input batch
            num_steps: Number of flow steps to sample (default: use self.num_sampling_steps)
            n_samples: Number of sample trajectories to return
            
        Returns:
            trajectories: List of (num_steps+1, latent_dim) arrays for each sample
            future_latents: (n_samples, latent_dim) - ground truth targets
        """
        # Use configured num_sampling_steps if not specified
        if num_steps is None:
            num_steps = self.num_sampling_steps
        
        # normalize input
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        
        # Only use first n_samples
        n_samples = min(n_samples, batch_size)
        
        # Encode visual observations
        this_nobs = dict_apply(nobs, lambda x: x[:n_samples, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(n_samples, -1)
        obs_latents = self.obs_projector(nobs_features)
        
        # Encode history states
        history_states = nobs["agent_pos"][:n_samples, :self.n_obs_steps, :]
        history_latents = self.history_action_encoder(history_states)
        
        # Encode future actions (ground truth target)
        future_start = self.n_obs_steps - 1
        future_end = future_start + self.n_action_steps
        future_actions = nactions[:n_samples, future_start:future_end, :]
        future_latents = self.action_encoder(future_actions)
        
        # Run flow sampling with traces
        _, (traj_history, _) = self.flow_matcher.sample(
            self.flow_net,
            shape=(n_samples, self.latent_dim),
            device=obs_latents.device,
            num_steps=num_steps,
            start=history_latents,
            global_cond=obs_latents,
            return_traces=True
        )
        
        # Convert to numpy arrays (ensure all tensors are on CPU first)
        traj_history_cpu = []
        for t in traj_history:
            if hasattr(t, 'cpu'):
                traj_history_cpu.append(t.cpu())
            else:
                traj_history_cpu.append(torch.tensor(t))
        
        traj_stacked = torch.stack(traj_history_cpu, dim=0)  # (num_steps+1, n_samples, latent_dim)
        trajectories = [traj_stacked[:, i, :].numpy() for i in range(n_samples)]
        future_latents_np = future_latents.cpu().numpy()
        
        return trajectories, future_latents_np

    @staticmethod
    def _compute_contrastive_loss(image_features, action_features, temperature=0.07):
        """Contrastive loss between image and action features (InfoNCE)"""
        batch_size = image_features.size(0)
        image_features = F.normalize(image_features, dim=1)
        action_features = F.normalize(action_features, dim=1)

        # Compute similarity matrix
        logits = torch.matmul(image_features, action_features.T) / temperature

        # Symmetric contrastive loss (image-to-action + action-to-image)
        labels = torch.arange(batch_size, device=logits.device)
        loss_i2a = F.cross_entropy(logits, labels)
        loss_a2i = F.cross_entropy(logits.T, labels)

        return (loss_i2a + loss_a2i) / 2
