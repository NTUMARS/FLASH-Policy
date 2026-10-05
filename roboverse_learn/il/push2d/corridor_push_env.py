"""CorridorPush environment: pymunk physics + cv2 rendering (README.md).

Physics uses pymunk and rendering uses OpenCV. This module does not import
utils/pymunk_util, which imports pygame for the legacy renderer.

Action   : (2,) float32 — pusher target position in world px, executed by a
           velocity-PD over `sim_hz/control_hz` substeps.
Obs      : {"image": (H, W, 3) uint8, "agent_pos": (2,) float32}
Snapshot : full pose+velocity of the two dynamic bodies (oracle replay support).
"""

import numpy as np
import pymunk

from roboverse_learn.il.push2d.corridor_geometry import (
    AGENT_R, PUCK_R, WORLD, CorridorGeometry, build_corridor,
)

# Collision categories. Every shape carries a distinct non-zero category while
# masks stay ALL by default, so all pairs still collide exactly as before —
# these bits exist only so the recovery FSM can mask out the single
# agent<->puck pair during an open-loop retreat (set_agent_puck_collision).
_CAT_STATIC = 0b001      # walls + teeth
_CAT_PUCK = 0b010
_CAT_AGENT = 0b100

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("corridor_push_env needs opencv-python (cv2)") from e


def draw_static_scene(geom: CorridorGeometry) -> np.ndarray:
    """Walls + teeth + goal on a fresh 512 canvas (BGR).  Shared by the env's
    observation renderer and push2d_viz — must stay pixel-identical to keep
    already-generated datasets consistent with eval-time observations."""
    c = np.full((int(WORLD), int(WORLD), 3), 24, dtype=np.uint8)
    for cx, cy, w, h in geom.walls:
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
        cv2.rectangle(c, (x0, y0), (int(x0 + w), int(y0 + h)), (190, 190, 190), -1)
    for tri in geom.teeth:
        pts = np.asarray(tri, dtype=np.int32)
        cv2.fillPoly(c, [pts], (190, 190, 190))
    gx, gy = geom.goal_center
    cv2.circle(c, (int(gx), int(gy)), int(geom.goal_radius), (60, 170, 60), -1)
    return c


class CorridorPushEnv:
    def __init__(self, level: int, render_size: int = 96,
                 control_hz: int = 5, sim_hz: int = 100,
                 max_steps: int | None = None, seed: int | None = None):
        # control_hz=5 (20 substeps/action): each action spans ~20px of expert
        # progress, packing ~2-4 weave extrema into an 8-step chunk at high
        # levels — the regime where K6 fitting error (13-15px RMS) exceeds the
        # tooth-throat clearance (~8.5px) while K9 stays well inside (3-5px).
        # Verified by the inline spectrum check (README.md).
        self.geom: CorridorGeometry = build_corridor(level)
        self.level = level
        self.render_size = render_size
        self.control_hz = control_hz
        self.n_substeps = sim_hz // control_hz
        self.dt = 1.0 / sim_hz
        self.max_steps = max_steps if max_steps is not None else 60 + 16 * level
        self.rng = np.random.default_rng(seed)
        self.kp = 20.0
        self.v_max = 350.0
        self._build_space()

    # ------------------------------------------------------------------ setup
    def _build_space(self):
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = 0.05  # strong velocity decay: quasi-static pushing
        for cx, cy, w, h in self.geom.walls:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = (cx, cy)
            shape = pymunk.Poly.create_box(body, (w, h))
            shape.friction = 0.5
            shape.filter = pymunk.ShapeFilter(categories=_CAT_STATIC)
            self.space.add(body, shape)
        for tri in self.geom.teeth:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            shape = pymunk.Poly(body, [tuple(p) for p in tri])
            shape.friction = 0.5
            shape.filter = pymunk.ShapeFilter(categories=_CAT_STATIC)
            self.space.add(body, shape)

        self.puck = pymunk.Body(mass=0.5, moment=pymunk.moment_for_circle(0.5, 0, PUCK_R))
        self.puck_shape = pymunk.Circle(self.puck, PUCK_R)
        self.puck_shape.friction = 0.6
        self.puck_shape.filter = pymunk.ShapeFilter(categories=_CAT_PUCK)
        self.space.add(self.puck, self.puck_shape)

        self.agent = pymunk.Body(mass=5.0, moment=pymunk.moment_for_circle(5.0, 0, AGENT_R))
        self.agent_shape = pymunk.Circle(self.agent, AGENT_R)
        self.agent_shape.friction = 0.6
        self.agent_shape.filter = pymunk.ShapeFilter(categories=_CAT_AGENT)
        self.space.add(self.agent, self.agent_shape)
        self.agent_puck_collision = True

    # ------------------------------------------------- retreat pass-through
    def set_agent_puck_collision(self, enabled: bool):
        """Toggle ONLY the agent<->puck contact; walls and teeth are untouched.

        Used by the recovery FSM so an open-loop retreat cannot shove the puck
        backwards on its way out — the pusher retraces its own past targets, and
        the puck has moved since, so the return path can run straight through it.

        Implemented with pymunk's category/mask filter rather than by removing
        the shape: removal would drop the puck's wall contacts too and let it
        tunnel out of the corridor. Masks default to ALL, and every shape keeps a
        non-zero category, so `enabled=True` is exactly the historic behaviour
        (every pair still passes the filter) — verified byte-identical.

        Deployability note: on a real arm this is not "phasing through matter",
        it is the 2D stand-in for lifting the end-effector over the object before
        retracting — the 3D counterpart is a legal motion.
        """
        mask = pymunk.ShapeFilter.ALL_MASKS()
        if not enabled:
            mask &= ~_CAT_PUCK
        self.agent_shape.filter = pymunk.ShapeFilter(categories=_CAT_AGENT, mask=mask)
        self.agent_puck_collision = bool(enabled)

    def agent_puck_overlap(self) -> float:
        """Penetration depth (px) between agent and puck; <=0 means separated.

        Restoring contact while the two overlap makes pymunk resolve the overlap
        with a large impulse, which would look like the retreat "kicked" the
        puck. Callers use this to detect (and report) that case.
        """
        d = (np.asarray(self.agent.position, dtype=np.float64)
             - np.asarray(self.puck.position, dtype=np.float64))
        return float(PUCK_R + AGENT_R - np.linalg.norm(d))

    # ------------------------------------------------------------------- api
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        jp = self.rng.uniform(-4.0, 4.0, size=2)
        ja = self.rng.uniform(-3.0, 3.0, size=2)
        self.puck.position = tuple(np.asarray(self.geom.puck_start) + jp)
        self.agent.position = tuple(np.asarray(self.geom.agent_start) + ja)
        for b in (self.puck, self.agent):
            b.velocity = (0.0, 0.0)
            b.angular_velocity = 0.0
            b.angle = 0.0
        self.t = 0
        return self._obs()

    def step(self, action):
        target = np.clip(np.asarray(action, dtype=np.float64), 0.0, WORLD)
        for _ in range(self.n_substeps):
            err = target - np.asarray(self.agent.position)
            vel = self.kp * err
            speed = np.linalg.norm(vel)
            if speed > self.v_max:
                vel = vel * (self.v_max / speed)
            self.agent.velocity = tuple(vel)
            self.space.step(self.dt)
        self.t += 1
        success = self.puck.position.x >= self.geom.success_x
        done = success or self.t >= self.max_steps
        return self._obs(), done, {"success": bool(success)}

    # --------------------------------------------------- snapshot for oracle
    def snapshot(self):
        out = {}
        for name, b in (("puck", self.puck), ("agent", self.agent)):
            out[name] = (tuple(b.position), tuple(b.velocity), b.angle, b.angular_velocity)
        out["t"] = self.t
        return out

    def restore(self, snap):
        for name, b in (("puck", self.puck), ("agent", self.agent)):
            pos, vel, ang, avel = snap[name]
            b.position, b.velocity, b.angle, b.angular_velocity = pos, vel, ang, avel
        self.t = snap["t"]
        self.space.reindex_shapes_for_body(self.puck)
        self.space.reindex_shapes_for_body(self.agent)

    # ---------------------------------------------------------------- render
    def _obs(self):
        return {
            "image": self.render(self.render_size),
            "agent_pos": np.asarray(self.agent.position, dtype=np.float32),
        }

    def render(self, size: int | None = None):
        c = draw_static_scene(self.geom)
        px, py = self.puck.position
        cv2.circle(c, (int(px), int(py)), int(PUCK_R), (70, 70, 220), -1)
        ax, ay = self.agent.position
        cv2.circle(c, (int(ax), int(ay)), int(AGENT_R), (220, 120, 60), -1)
        if size is not None and size != c.shape[0]:
            c = cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)
        return c
