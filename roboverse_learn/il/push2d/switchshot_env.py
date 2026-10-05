"""ForkReach environment, retained as SwitchShot variant N / level 4.

A navigation agent reaches either of two always-green goal regions. Goal
entry is checked at every physics substep and latched until reset.
"""

import numpy as np
import pymunk

from roboverse_learn.il.push2d.switchshot_geometry import (
    AGENT_R, WORLD, build_switch, draw_schedule, signal_colors,
)

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("switchshot_env needs opencv-python (cv2)") from e

COLORS = {"green": (60, 200, 60)}  # BGR


def draw_switch_scene(geom, colors: dict) -> np.ndarray:
    c = np.full((int(WORLD), int(WORLD), 3), 24, dtype=np.uint8)
    for cx, cy, w, h in geom.walls:
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
        cv2.rectangle(c, (x0, y0), (int(x0 + w), int(y0 + h)), (190, 190, 190), -1)
    for side, (gx, gy) in geom.goals.items():
        cv2.circle(c, (int(gx), int(gy)), int(geom.goal_r), COLORS[colors[side]], -1)
    return c


class SwitchShotEnv:
    def __init__(self, level: int = 4, variant: str = "N", render_size: int = 96,
                 control_hz: int = 5, sim_hz: int = 100,
                 max_steps: int | None = None, seed: int | None = None):
        if level != 4:
            raise ValueError("Only ForkReach (SwitchShot variant N, level 4) is retained")
        self.geom = build_switch(variant)
        self.level = level
        self.variant = variant
        self.render_size = render_size
        self.control_hz = control_hz
        self.n_substeps = sim_hz // control_hz
        self.dt = 1.0 / sim_hz
        self.max_steps = max_steps if max_steps is not None else 120
        self.rng = np.random.default_rng(seed)
        self.kp = 20.0
        self.v_max = 350.0
        self._build_space()

    def _build_space(self):
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = 0.05
        for cx, cy, w, h in self.geom.walls:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = (cx, cy)
            shape = pymunk.Poly.create_box(body, (w, h))
            shape.friction = 0.5
            shape.elasticity = 0.0
            self.space.add(body, shape)
        self.puck = None  # historical navigation API; no puck body is created
        self.agent = pymunk.Body(mass=5.0, moment=pymunk.moment_for_circle(5.0, 0, AGENT_R))
        ash = pymunk.Circle(self.agent, AGENT_R)
        ash.friction = 0.6
        self.space.add(self.agent, ash)

    @property
    def subject(self):
        return self.agent

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.schedule = draw_schedule(self.level, self.rng)
        # Historical N/lv4 consumed this unused puck jitter before agent jitter.
        # Keep its RNG consumption to reproduce existing data and evaluations.
        self.rng.uniform(-4.0, 4.0, size=2)
        ja = self.rng.uniform(-3.0, 3.0, size=2)
        self.agent.position = tuple(np.asarray(self.geom.agent_start) + ja)
        self.agent.velocity = (0.0, 0.0)
        self.agent.angular_velocity = 0.0
        self.agent.angle = 0.0
        self.t = 0
        self.terminal = None
        return self._obs()

    def visible_colors(self):
        return signal_colors(self.schedule, self.t)

    def _check_terminal_substep(self):
        p = np.asarray(self.agent.position)
        for gx, gy in self.geom.goals.values():
            if np.hypot(p[0] - gx, p[1] - gy) <= self.geom.goal_r:
                self.terminal = "success"
                return True
        return False

    def step(self, action):
        if self.terminal is not None:
            return self._obs(), True, self._info()
        target = np.clip(np.asarray(action, dtype=np.float64), 0.0, WORLD)
        for _ in range(self.n_substeps):
            err = target - np.asarray(self.agent.position)
            vel = self.kp * err
            sp = np.linalg.norm(vel)
            if sp > self.v_max:
                vel = vel * (self.v_max / sp)
            self.agent.velocity = tuple(vel)
            self.space.step(self.dt)
            if self._check_terminal_substep():
                self.agent.velocity = (0.0, 0.0)
                break
        self.t += 1
        if self.terminal is None and self.t >= self.max_steps:
            self.terminal = "timeout"
        done = self.terminal is not None
        return self._obs(), done, self._info()

    def _info(self):
        return {"success": self.terminal == "success",
                "fail_reason": None if self.terminal in (None, "success") else self.terminal,
                "schedule": dict(self.schedule)}

    def snapshot(self):
        b = self.agent
        return {"agent": (tuple(b.position), tuple(b.velocity), b.angle, b.angular_velocity),
                "t": self.t, "terminal": self.terminal}

    def restore(self, snap):
        pos, vel, ang, avel = snap["agent"]
        self.agent.position, self.agent.velocity = pos, vel
        self.agent.angle, self.agent.angular_velocity = ang, avel
        self.t = snap["t"]
        self.terminal = snap["terminal"]
        self.space.reindex_shapes_for_body(self.agent)

    def _obs(self):
        return {"image": self.render(self.render_size),
                "agent_pos": np.asarray(self.agent.position, dtype=np.float32)}

    def render(self, size: int | None = None):
        c = draw_switch_scene(self.geom, self.visible_colors())
        ax, ay = self.agent.position
        cv2.circle(c, (int(ax), int(ay)), int(AGENT_R), (220, 120, 60), -1)
        if size is not None and size != c.shape[0]:
            c = cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)
        return c
