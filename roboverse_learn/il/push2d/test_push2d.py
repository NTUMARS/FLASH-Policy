"""Public CorridorPush geometry, data contract and open-loop evaluation tests."""

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from roboverse_learn.il.push2d.corridor_geometry import (
    HALF_W, LEVEL_PERIOD, PUCK_R, build_corridor,
)


def test_published_corridor_geometry():
    assert LEVEL_PERIOD == {2: 100.0}
    geometry = build_corridor(2)
    tips = np.array([tooth[2][0] for tooth in geometry.teeth])
    np.testing.assert_allclose(np.diff(tips), 50.0)
    throat_width = 2 * HALF_W - geometry.protrusion
    assert throat_width == 41.0
    assert (throat_width - 2 * PUCK_R) / 2 == 8.5
    assert np.all(np.diff(geometry.path[:, 0]) >= 0)


@pytest.mark.parametrize("level", [0, 1, 3, 4, 5])
def test_unpublished_corridor_level_rejected(level):
    with pytest.raises(AssertionError, match="unknown level"):
        build_corridor(level)


@pytest.mark.parametrize("seed", [50, 51, 52])
def test_level2_expert_succeeds(seed):
    from roboverse_learn.il.push2d.corridor_push_env import CorridorPushEnv
    from roboverse_learn.il.push2d.scripted_expert import ScriptedExpert

    env = CorridorPushEnv(level=2, seed=seed)
    success, frames, states, actions = ScriptedExpert(env.geom, seed=seed + 10).rollout(env)
    assert success
    assert frames.dtype == np.uint8 and frames.shape[1:] == (3, 96, 96)
    assert len(frames) == len(states) == len(actions) >= 18
    assert states.shape[1:] == actions.shape[1:] == (2,)


def test_zarr_training_contract_and_cpu_normalizer(tmp_path, monkeypatch):
    from roboverse_learn.il.datasets.robot_image_dataset import RobotImageDataset
    from roboverse_learn.il.push2d.push2d_eval import load_dataset_normalizer

    out = tmp_path / "test.zarr"
    subprocess.run([sys.executable, "-m", "roboverse_learn.il.push2d.gen_demos",
                    "--out", str(out), "--eps-per-level", "2", "--levels", "2",
                    "--seed", "3"], check=True)
    # The shared CUDA trainer pins its batch buffers; this data-contract check
    # also runs on CPU-only machines and does not change the production dataset.
    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    dataset = RobotImageDataset(str(out), horizon=18, pad_before=8, pad_after=8,
                                batch_size=2, val_ratio=0.0)
    batch = dataset.postprocess(dataset[np.arange(2)], torch.device("cpu"))
    assert batch["obs"]["head_cam"].shape == (2, 18, 3, 96, 96)
    assert batch["obs"]["agent_pos"].shape == batch["action"].shape == (2, 18, 2)
    assert 0 <= batch["obs"]["head_cam"].min() <= batch["obs"]["head_cam"].max() <= 1
    expected = dataset.get_normalizer().state_dict()
    actual = load_dataset_normalizer(str(out)).state_dict()
    assert expected.keys() == actual.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_eval_executes_whole_chunks_and_computes_jerk(tmp_path, monkeypatch):
    from roboverse_learn.il.push2d import push2d_eval

    class Policy:
        n_obs_steps = 2
        n_action_steps = 8
        num_inference_steps = 2

        def __init__(self):
            self.calls = 0
            self.actions = []

        def predict_action(self, obs):
            assert obs["head_cam"].shape == (1, 2, 3, 96, 96)
            assert obs["agent_pos"].shape == (1, 2, 2)
            index = np.arange(self.calls * 8, (self.calls + 1) * 8)
            action = np.stack([100 + index, 240 + 0.01 * index ** 2], axis=-1).astype(np.float32)
            self.calls += 1
            self.actions.extend(action)
            return {"action": torch.from_numpy(action)[None]}

    policy = Policy()
    monkeypatch.setattr(push2d_eval, "load_policy", lambda *args: (policy, None))
    monkeypatch.setattr(push2d_eval, "build_eval_config_lines", lambda *args, **kwargs: [])
    args = SimpleNamespace(levels=[2], eps=1, seed=10000, device="cpu",
                           checkpoint="test.ckpt", zarr="unused", out=str(tmp_path),
                           no_video=True, max_steps=18, torch_seed=42)
    push2d_eval.run(args)
    assert policy.calls == 3  # 8 + 8 + 2 actions until the episode limit.
    actions = np.array(policy.actions[:18])
    expected_jerk = np.sqrt(np.mean(np.diff(actions, n=2, axis=0) ** 2))
    stats = json.loads((tmp_path / "00_final_stats.json").read_text())
    assert stats["levels"]["2"]["success_rate"] == 0
    assert stats["levels"]["2"]["action_jerk_rms_px"] == pytest.approx(expected_jerk)
