# roboverse_learn/il/policies/act/test_act_image_policy.py
"""Tests for the ACTImagePolicy adapter used by the shared Push2D pipeline.

These tests cover the shared ``compute_loss``, ``predict_action``, and
``set_normalizer`` interfaces; bypassing argparse (pytest's argv does not
contain ``--task_name``); CPU construction without the unconditional
``.cuda()`` in detr/main.py:82; temporal alignment at frame 7 with supervision
from ``actions[7:15]``; checkpoint round trips; and CUDA bf16 autocast (skipped
when CUDA is unavailable).
"""
import copy

import pytest
import torch

from roboverse_learn.il.policies.act.act_image_policy import ACTImagePolicy
from roboverse_learn.il.utils.normalize_util import get_image_range_normalizer
from roboverse_learn.il.utils.normalizer import LinearNormalizer

SHAPE_META = {
    "obs": {
        "head_cam": {"shape": [3, 96, 96], "type": "rgb"},
        "agent_pos": {"shape": [2], "type": "low_dim"},
    },
    "action": {"shape": [2]},
}


def _tiny_policy():
    # Use a small model for speed while preserving the real structural
    # dimensions (chunk, n_obs, and horizon).
    return ACTImagePolicy(
        shape_meta=SHAPE_META, horizon=16, n_action_steps=8, n_obs_steps=8,
        chunk_size=8, kl_weight=10, hidden_dim=64, dim_feedforward=128,
        enc_layers=1, dec_layers=1, nheads=2,
    )


def _fit_normalizer():
    n = LinearNormalizer()
    data = {
        "action": torch.rand(100, 2) * 512.0,
        "agent_pos": torch.rand(100, 2) * 512.0,
    }
    n.fit(data=data, last_n_dims=1, mode="limits")
    n["head_cam"] = get_image_range_normalizer()
    return n


def _train_batch(B=2, T=16):
    return {
        "obs": {
            "head_cam": torch.rand(B, T, 3, 96, 96),
            "agent_pos": torch.rand(B, T, 2) * 512.0,
        },
        "action": torch.rand(B, T, 2) * 512.0,
    }


def test_cpu_construction_without_cli_args():
    """Construct without reading sys.argv or calling .cuda()."""
    policy = _tiny_policy()
    assert next(policy.parameters()).device.type == "cpu"


def test_required_attrs():
    policy = _tiny_policy()
    assert policy.n_obs_steps == 8
    assert policy.n_action_steps == 8
    assert policy.horizon == 16
    assert policy.num_inference_steps == 1
    assert policy.chunk_size == 8
    assert policy.kl_weight == 10


def test_num_inference_steps_override_ignored():
    """Check handling of an unsupported inference setting."""
    policy = _tiny_policy()
    policy.num_inference_steps = 10
    assert policy.num_inference_steps == 1


def test_compute_loss_scalar_and_backward():
    policy = _tiny_policy()
    policy.set_normalizer(_fit_normalizer())
    loss = policy.compute_loss(_train_batch())
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()  # Gradients propagate through the model.


def test_time_alignment_frame7_and_target_7_15():
    """Verify frame-7 qpos and the normalized action target at [7:15]."""
    policy = _tiny_policy()
    normalizer = _fit_normalizer()
    policy.set_normalizer(normalizer)

    captured = {}

    class SpyModel(torch.nn.Module):
        num_queries = 8

        def forward(self, qpos, image, env_state, actions=None, is_pad=None):
            captured["qpos"] = qpos
            captured["actions"] = actions
            B = qpos.shape[0]
            a_hat = torch.zeros(B, 8, 2, requires_grad=True)
            mu = torch.zeros(B, 32)
            logvar = torch.zeros(B, 32)
            return a_hat, None, (mu, logvar)

    policy.model = SpyModel()
    batch = _train_batch(B=2, T=16)
    policy.compute_loss(batch)

    expect_qpos = normalizer["agent_pos"].normalize(batch["obs"]["agent_pos"])[:, 7]
    expect_target = normalizer["action"].normalize(batch["action"])[:, 7:15]
    torch.testing.assert_close(captured["qpos"], expect_qpos)
    torch.testing.assert_close(captured["actions"], expect_target)


def test_predict_action_eval_shape():
    """Evaluate with T == n_obs_steps == 8, as in switchshot_eval's deque."""
    policy = _tiny_policy()
    policy.set_normalizer(_fit_normalizer())
    od = {
        "head_cam": torch.rand(1, 8, 3, 96, 96),
        "agent_pos": torch.rand(1, 8, 2) * 512.0,
    }
    with torch.no_grad():
        result = policy.predict_action(od)
    assert result["action"].shape == (1, 8, 2)
    assert result["action_pred"].shape == (1, 8, 2)
    # Unnormalized actions should return to the pixel-coordinate scale.
    assert result["action"].abs().max() < 1024.0


def test_predict_action_training_horizon():
    """Accept T=horizon=16 observations from periodic runner sampling."""
    policy = _tiny_policy()
    policy.set_normalizer(_fit_normalizer())
    od = {
        "head_cam": torch.rand(2, 16, 3, 96, 96),
        "agent_pos": torch.rand(2, 16, 2) * 512.0,
    }
    with torch.no_grad():
        result = policy.predict_action(od)
    assert result["action"].shape == (2, 8, 2)


def test_state_dict_roundtrip_preserves_normalizer():
    """Preserve normalizer values through a checkpoint state-dict round trip."""
    policy = _tiny_policy()
    policy.set_normalizer(_fit_normalizer())
    sd = policy.state_dict()

    fresh = _tiny_policy()
    missing, unexpected = fresh.load_state_dict(sd, strict=False)
    assert not missing and not unexpected
    assert "head_cam" in fresh.normalizer.params_dict  # Guard used by push2d_eval.py:43
    for key in ("action", "agent_pos"):
        a = policy.normalizer.params_dict[key]["scale"]
        b = fresh.normalizer.params_dict[key]["scale"]
        torch.testing.assert_close(a, b)


def test_state_action_dim_mismatch_raises():
    bad = {
        "obs": {
            "head_cam": {"shape": [3, 96, 96], "type": "rgb"},
            "agent_pos": {"shape": [3], "type": "low_dim"},  # != action dim 2
        },
        "action": {"shape": [2]},
    }
    with pytest.raises(AssertionError, match="state_dim"):
        ACTImagePolicy(shape_meta=bad, horizon=16, n_action_steps=8, n_obs_steps=8)


def test_chunk_vs_horizon_asserts():
    with pytest.raises(AssertionError):
        ACTImagePolicy(shape_meta=SHAPE_META, horizon=16, n_action_steps=9,
                       n_obs_steps=8, chunk_size=8)  # n_action_steps > chunk
    with pytest.raises(AssertionError):
        ACTImagePolicy(shape_meta=SHAPE_META, horizon=10, n_action_steps=8,
                       n_obs_steps=8, chunk_size=8)  # 7+8 > 10


def test_single_camera_only():
    with pytest.raises(AssertionError, match="head_cam"):
        ACTImagePolicy(shape_meta=SHAPE_META, horizon=16, n_action_steps=8,
                       n_obs_steps=8, camera_names=("head_cam", "front_cam"))


def test_ema_deepcopy_compatible():
    """Support DefaultRunner's deepcopy when use_ema=True (:649-651)."""
    policy = _tiny_policy()
    policy.set_normalizer(_fit_normalizer())
    ema = copy.deepcopy(policy)
    assert ema is not policy


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bf16_autocast_finite():
    """Reproduce DefaultRunner's bf16 autocast training sequence.

    Match the runner order: call ``set_normalizer`` (:759) before
    ``.to(device)`` (:921). ``load_state_dict`` reconstructs ``params_dict``
    with CPU tensors, which are moved to CUDA by the later module-wide transfer.
    """
    policy = _tiny_policy()
    policy.set_normalizer(_fit_normalizer())
    policy = policy.cuda()
    batch = _train_batch()
    batch = {
        "obs": {k: v.cuda() for k, v in batch["obs"].items()},
        "action": batch["action"].cuda(),
    }
    for _ in range(3):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)
        loss.backward()
        grads_finite = all(
            torch.isfinite(p.grad).all()
            for p in policy.parameters() if p.grad is not None
        )
        assert grads_finite
        policy.zero_grad()
