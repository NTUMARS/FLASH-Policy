"""ForkReach navigation expert and matched opposite-goal demonstrations."""

import numpy as np

from roboverse_learn.il.push2d.switchshot_geometry import STAGING_X


class NavExpert:
    """Approach the chosen goal through a lane-alignment waypoint.

    Set mode_bit to top or bottom for demonstrations. Its RNG is independent
    of the environment RNG, so matched modes share exactly the same reset.
    """

    def __init__(self, geom, mode_bit: str | None = None, noise_std: float = 1.0,
                 advance: float = 24.0, seed: int | None = None):
        if geom.variant != "N":
            raise ValueError("Only ForkReach (SwitchShot variant N) is retained")
        if mode_bit not in (None, "top", "bottom"):
            raise ValueError("mode_bit must be top, bottom, or None")
        self.geom = geom
        self.mode_bit = mode_bit
        self.noise_std = noise_std
        self.advance = advance
        self.rng = np.random.default_rng(seed)

    def _waypoint(self, env):
        # Preserve the historical action-level coin draw when no mode is forced.
        side = self.mode_bit or ("top" if self.rng.random() < 0.5 else "bottom")
        return self._go_waypoint(env, side), "go"

    def _go_waypoint(self, env, side):
        gx, gy = self.geom.goals[side]
        agent = np.asarray(env.agent.position, float)
        if abs(agent[1] - gy) > self.geom.goal_r + 14.0:
            return np.asarray([STAGING_X, gy], float)
        return np.asarray([gx, gy], float)

    def act(self, env):
        agent = np.asarray(env.agent.position, float)
        wp, phase = self._waypoint(env)
        d = wp - agent
        n = np.linalg.norm(d)
        step = d / n * min(self.advance, n) if n > 1e-6 else np.zeros(2)
        return (agent + step + self.rng.normal(0.0, self.noise_std, 2)).astype(np.float32)

    def rollout(self, env, record: bool = True, min_len: int = 0):
        obs = env.reset()
        frames, states, actions = [], [], []
        done, info = False, {"success": False}
        while not done:
            action = self.act(env)
            if record:
                frames.append(np.moveaxis(obs["image"], -1, 0))
                states.append(obs["agent_pos"].copy())
                actions.append(action.copy())
            obs, done, info = env.step(action)
        if record:
            # No padding by default; the sequence sampler handles boundaries.
            while info.get("success") and len(frames) < min_len and frames:
                frames.append(frames[-1].copy())
                states.append(states[-1].copy())
                actions.append(actions[-1].copy())
            return (info["success"], info, np.asarray(frames, dtype=np.uint8),
                    np.asarray(states, dtype=np.float32),
                    np.asarray(actions, dtype=np.float32))
        return info["success"], info, None, None, None


def expert_for(variant: str, geom, **kw):
    if variant != "N":
        raise ValueError("Only ForkReach (SwitchShot variant N) is retained")
    return NavExpert(geom, **kw)
