"""ForkReach preservation checks; run with the project dependencies installed."""

import json
from argparse import Namespace

import numpy as np
import pytest
import torch

from roboverse_learn.il.push2d.switchshot_env import SwitchShotEnv
from roboverse_learn.il.push2d.switchshot_expert import expert_for
from roboverse_learn.il.push2d.switchshot_geometry import (
    CENTER_Y, GOAL_DY, LEVELS, build_switch, draw_schedule, signal_colors,
)


def test_only_forkreach_retained():
    assert LEVELS == (4,)
    assert build_switch().goals == {"top": (470.0, 164.0), "bottom": (470.0, 348.0)}
    for variant in ("P", "S"):
        with pytest.raises(ValueError, match="Only ForkReach"):
            build_switch(variant)
        with pytest.raises(ValueError, match="Only ForkReach"):
            SwitchShotEnv(4, variant=variant)
    for level in (0, 1, 2, 3, 5, 6, 7, 8, 9, 10):
        with pytest.raises(ValueError, match="Only ForkReach"):
            SwitchShotEnv(level)
        with pytest.raises(ValueError, match="Only ForkReach"):
            draw_schedule(level, np.random.default_rng(0))


def test_historical_seed_physics_and_action_regression():
    """Golden original N/lv4 start/actions guard both otherwise-unused RNG draws."""
    env = SwitchShotEnv(seed=99)
    obs = env.reset()
    np.testing.assert_array_equal(obs["agent_pos"], np.array([42.8331184387207, 256.6894226074219], dtype=np.float32))
    ex = expert_for("N", env.geom, mode_bit="top", seed=106)
    action = ex.act(env)
    np.testing.assert_array_equal(action, np.array([66.55546569824219, 249.6416473388672], dtype=np.float32))
    obs, done, info = env.step(action)
    np.testing.assert_array_equal(obs["agent_pos"], np.array([66.24793243408203, 249.73301696777344], dtype=np.float32))
    assert not done
    assert env.n_substeps == 20 and env.dt == 0.01 and env.max_steps == 120
    assert env.puck is None and env.subject is env.agent
    assert obs["image"].dtype == np.uint8 and obs["image"].shape == (96, 96, 3)


def test_both_goals_always_green():
    env = SwitchShotEnv(seed=99)
    env.reset()
    assert env.schedule == dict(s0=0, s1=None, lock=0, initial=None, final=None,
                                valid_goals=("top", "bottom"))
    for t in (0, 10, 119):
        assert signal_colors(env.schedule, t) == {"top": "green", "bottom": "green"}
    full = env.render(None)
    for gx, gy in env.geom.goals.values():
        np.testing.assert_array_equal(full[int(gy), int(gx)], [60, 200, 60])


@pytest.mark.parametrize("mode", ["top", "bottom"])
def test_goal_entry_latched_and_correct_mode(mode):
    env = SwitchShotEnv(seed=1)
    env.reset()
    env.agent.position = env.geom.goals[mode]
    env.space.reindex_shapes_for_body(env.agent)
    obs, done, info = env.step(np.asarray(env.agent.position))
    assert done and info["success"] and info["fail_reason"] is None
    final_t = env.t
    again, done, info = env.step([0, 0])
    assert done and info["success"] and env.t == final_t
    np.testing.assert_array_equal(again["agent_pos"], obs["agent_pos"])


def test_timeout_and_snapshot_restore():
    env = SwitchShotEnv(max_steps=2, seed=99)
    obs = env.reset()
    snap = env.snapshot()
    env.step(obs["agent_pos"])
    _, done, info = env.step(obs["agent_pos"])
    assert done and not info["success"] and info["fail_reason"] == "timeout"
    env.restore(snap)
    assert env.t == 0 and env.terminal is None
    np.testing.assert_array_equal(env._obs()["image"], obs["image"])
    np.testing.assert_array_equal(env._obs()["agent_pos"], obs["agent_pos"])


def test_opposite_modes_pair_identical_initial_observations():
    from roboverse_learn.il.push2d.switchshot_gen_demos import collect_pair
    for seed in (70, 99, 80000):
        (top, top_meta), (bottom, bottom_meta) = collect_pair(seed, pair=0)
        np.testing.assert_array_equal(top["head_camera"][0], bottom["head_camera"][0])
        np.testing.assert_array_equal(top["state"][0], bottom["state"][0])
        assert top_meta["seed"] == bottom_meta["seed"] == seed
        assert top_meta["expert_chosen_mode"] == "top"
        assert bottom_meta["expert_chosen_mode"] == "bottom"
        assert top["action"][0, 1] < bottom["action"][0, 1]
        assert top["state"][-1, 1] < CENTER_Y < bottom["state"][-1, 1]
        assert top["head_camera"].shape[1:] == (3, 96, 96)


def test_pair_retry_is_atomic(monkeypatch):
    from roboverse_learn.il.push2d import switchshot_gen_demos as gen
    real_expert_for = gen.expert_for
    failed = False

    def fail_first_bottom(variant, geom, **kw):
        nonlocal failed
        ex = real_expert_for(variant, geom, **kw)
        if kw["mode_bit"] == "bottom" and not failed:
            failed = True
            real_rollout = ex.rollout

            def rollout(env):
                success, info, frames, states, actions = real_rollout(env)
                return False, info, frames, states, actions

            ex.rollout = rollout
        return ex

    monkeypatch.setattr(gen, "expert_for", fail_first_bottom)
    (top, tm), (bottom, bm) = gen.collect_pair(80000, pair=5)
    assert tm["seed"] == bm["seed"] == 80001
    assert tm["retried"] == bm["retried"] == 1
    np.testing.assert_array_equal(top["head_camera"][0], bottom["head_camera"][0])
    np.testing.assert_array_equal(top["state"][0], bottom["state"][0])


def test_commit_side_boundaries():
    from roboverse_learn.il.push2d.switchshot_eval import commit_side
    band = GOAL_DY / 2.0
    assert commit_side(CENTER_Y - GOAL_DY) == "top"
    assert commit_side(CENTER_Y + GOAL_DY) == "bottom"
    assert commit_side(CENTER_Y) == "middle"
    assert commit_side(CENTER_Y - band) == "middle"
    assert commit_side(CENTER_Y + band) == "middle"
    assert commit_side(CENTER_Y - band - 1) == "top"
    assert commit_side(CENTER_Y + band + 1) == "bottom"


def test_rename_original_forkreach_format(tmp_path):
    from roboverse_learn.il.push2d.switchshot_eval import rename_with_metrics
    d = tmp_path / "3000.ckpt_2026-07-27_15-37-21"
    d.mkdir()
    out = rename_with_metrics(d, sr=0.96, nis=6)
    assert out.name == "3000.ckpt_sr96_step6_2026-07-27_15-37-21"
    assert rename_with_metrics(out, sr=0.96, nis=6) == out


def test_historical_data_metadata_rejected_only_when_task_differs(tmp_path):
    from roboverse_learn.il.push2d.switchshot_eval import validate_task_metadata
    zarr = tmp_path / "task.zarr"
    sidecar = tmp_path / "task_episodes_meta.json"
    sidecar.write_text(json.dumps({"params": {"variant": "N", "levels": [4], "downsample_ratio": 1}}))
    validate_task_metadata(zarr)
    for params in ({"variant": "P", "levels": [4]}, {"variant": "N", "levels": [3, 4]},
                   {"variant": "N", "levels": [4], "downsample_ratio": 4}):
        sidecar.write_text(json.dumps({"params": params}))
        with pytest.raises(ValueError):
            validate_task_metadata(zarr)


def test_eval_balance_counts_committed_timeout(monkeypatch, tmp_path):
    """The failed bottom episode contributes to balance, preserving rebuttal metric."""
    from roboverse_learn.il.push2d import switchshot_eval as ev
    initial_third = SwitchShotEnv(seed=10002).reset()["agent_pos"]

    class Policy:
        n_obs_steps = 2
        n_action_steps = 8
        num_inference_steps = 2

        def predict_action(self, observations):
            initial = observations["agent_pos"][0, 0].numpy()
            if np.array_equal(initial, initial_third):
                target = [float(initial[0]), 400.0]
            else:
                # The two preceding episodes take opposite successful goals.
                first = SwitchShotEnv(seed=10000).reset()["agent_pos"]
                target = [470.0, 164.0 if np.array_equal(initial, first) else 348.0]
            return {"action": torch.tensor([[target] * 8], dtype=torch.float32)}

    episode_index = 0

    def env_factory(**kw):
        nonlocal episode_index
        env = SwitchShotEnv(**kw, max_steps=2 if episode_index == 2 else 120)
        episode_index += 1
        return env

    monkeypatch.setattr(ev, "SwitchShotEnv", env_factory)
    monkeypatch.setattr(ev, "load_policy", lambda *a: (Policy(), {}))
    monkeypatch.setattr(ev, "build_eval_config_lines", lambda *a, **kw: [])
    args = Namespace(variant="N", levels=[4], eps=3, zarr=str(tmp_path / "task.zarr"),
                     device="cpu", checkpoint="fake.ckpt", num_inference_steps=None,
                     execute_steps=8, out=str(tmp_path / "eval"), seed=10000,
                     no_video=True, no_log=False)
    ev.run(args)
    stats = json.loads((tmp_path / "eval/00_final_stats.json").read_text())
    assert stats["mean_success"] == pytest.approx(2 / 3)
    assert stats["L4_mode_split"] == {"top": 1, "bottom": 1}
    assert stats["L4_commit_split"] == {"top": 1, "bottom": 2}
    assert stats["L4_mode_balance"] == 0.5
    assert stats["levels"]["4"]["action_jerk_rms_px"] == 0.0
    log = np.load(tmp_path / "eval/rollout_logs/lv4_ep002.npz")
    assert str(log["terminal"]) == "timeout" and str(log["subject"]) == "agent"
    np.testing.assert_array_equal(log["puck_pos"], log["agent_pos"])
