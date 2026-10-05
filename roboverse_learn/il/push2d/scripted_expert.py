"""Scripted expert for CorridorPush: pure waypoint-following pushing (README.md).

The weave slope is bounded by geometry (<=50deg), so the pusher can stay behind
the puck for the whole episode — no side-swap maneuvers needed.
"""

import numpy as np

from roboverse_learn.il.push2d.corridor_geometry import AGENT_R, PUCK_R


class ScriptedExpert:
    def __init__(self, geom, noise_std: float = 1.0, advance: float = 20.0,
                 lookahead: float = 22.0, overlap: float = 2.0,
                 seed: int | None = None):
        self.geom = geom
        self.noise_std = noise_std
        self.advance = advance
        self.lookahead = lookahead
        self.overlap = overlap
        self.rng = np.random.default_rng(seed)

    def act(self, puck_pos, agent_pos):
        puck = np.asarray(puck_pos, dtype=np.float64)
        agent = np.asarray(agent_pos, dtype=np.float64)
        wp = self.geom.path_target(puck[0], self.lookahead)
        d = wp - puck
        n = np.linalg.norm(d)
        d = d / n if n > 1e-6 else np.array([1.0, 0.0])
        r_sum = PUCK_R + AGENT_R
        behind = puck - d * (r_sum - self.overlap)
        if np.linalg.norm(behind - agent) > 14.0:
            target = behind                      # regain contact position first
        else:
            target = behind + d * self.advance   # push along the weave
        target = target + self.rng.normal(0.0, self.noise_std, size=2)
        return target.astype(np.float32)

    def rollout(self, env, record: bool = True, min_len: int = 20):
        """Run one episode.  Returns (success, frames CHW uint8, states, actions).

        After success, keeps recording gentle hold steps until `min_len` frames
        so every episode is at least one full training window (horizon 16 +
        2*fit_pad) long — avoids padding-dominated sampler windows on the
        short smooth levels.
        """
        obs = env.reset()
        frames, states, actions = [], [], []
        done, info = False, {"success": False}
        while (not done) or (record and info.get("success") and len(frames) < min_len):
            if info.get("success"):
                # Post-success HOLD: command the current agent position.
                # Previously act() kept steering here, and path_target's
                # lookahead clips backwards past the goal — the recorded
                # "hold" steps actually pushed the puck back toward the start,
                # producing incorrect post-success demonstration tails.
                action = np.asarray(obs["agent_pos"], dtype=np.float32).copy()
            else:
                action = self.act(env.puck.position, obs["agent_pos"])
            if record:
                frames.append(np.moveaxis(obs["image"], -1, 0))  # HWC -> CHW
                states.append(obs["agent_pos"].copy())
                actions.append(action.copy())
            obs, done_step, info = env.step(action)
            done = done_step and not (info.get("success") and record and len(frames) < min_len)
        if record:
            return (info["success"], np.asarray(frames, dtype=np.uint8),
                    np.asarray(states, dtype=np.float32),
                    np.asarray(actions, dtype=np.float32))
        return info["success"], None, None, None
