"""CorridorPush geometry: sawtooth weave corridor (README.md).

A horizontal corridor whose two walls grow interleaved triangular teeth.
The puck must weave between tooth throats with period P (the frequency dial).
Straight-line paths are blocked by construction: tooth protrusion is solved
from  protrusion = (corridor_width - 2*puck_r + block_eps) / 2  so that no
horizontal line clears both tooth rows by the puck radius.

Environment and expert share this single geometry source.
"""

from dataclasses import dataclass, field

import numpy as np

WORLD = 512.0
CENTER_Y = 256.0
HALF_W = 32.0          # corridor half width (width 64)
WALL_T = 10.0
PUCK_R = 12.0
AGENT_R = 8.0
BLOCK_EPS = 6.0        # extra straight-line blocking margin (px)
X_LEFT = 16.0          # corridor inner span
X_RIGHT = 496.0
TEETH_X0 = 110.0       # keep start region clear
TEETH_X1 = 420.0       # keep goal region clear
GOAL_X = 458.0
GOAL_R = 22.0
SUCCESS_X = 448.0      # puck center beyond this = success

# Published level -> tooth period P in px
LEVEL_PERIOD = {2: 100.0}  # Published CorridorPush stress test
MAX_SLOPE_DEG = 50.0   # expert steerability bound for the weave path


@dataclass
class CorridorGeometry:
    level: int
    period: float | None
    protrusion: float
    walls: list = field(default_factory=list)   # axis-aligned rects (cx, cy, w, h)
    teeth: list = field(default_factory=list)   # triangles [(x,y)*3], CCW
    path: np.ndarray | None = None              # dense weave path samples (M, 2), x-monotone
    puck_start: tuple = (0.0, 0.0)
    agent_start: tuple = (0.0, 0.0)
    goal_center: tuple = (GOAL_X, CENTER_Y)
    goal_radius: float = GOAL_R
    success_x: float = SUCCESS_X

    def path_target(self, x: float, lookahead: float) -> np.ndarray:
        """Point on the weave path `lookahead` px ahead (in x) of position x."""
        xt = np.clip(x + lookahead, self.path[0, 0], self.path[-1, 0])
        i = int(np.searchsorted(self.path[:, 0], xt))
        i = min(max(i, 1), len(self.path) - 1)
        p0, p1 = self.path[i - 1], self.path[i]
        t = 0.0 if p1[0] == p0[0] else (xt - p0[0]) / (p1[0] - p0[0])
        return p0 + t * (p1 - p0)


def _weave_vertices(period: float | None, protrusion: float):
    """Polyline vertices of the weave path: throat centers + inter-tooth midpoints."""
    if period is None:
        return [(X_LEFT + 20.0, CENTER_Y), (X_RIGHT - 20.0, CENTER_Y)]
    y_bot_inner = CENTER_Y - HALF_W
    y_top_inner = CENTER_Y + HALF_W
    verts = [(X_LEFT + 20.0, CENTER_Y)]
    xs = np.arange(TEETH_X0, TEETH_X1 + 1e-6, period / 2.0)
    for k, x in enumerate(xs):
        bottom = (k % 2 == 0)
        if bottom:  # tooth from bottom wall -> throat above its tip
            tip_y = y_bot_inner + protrusion
            throat_c = tip_y + (y_top_inner - tip_y) / 2.0
        else:       # tooth from top wall -> throat below its tip
            tip_y = y_top_inner - protrusion
            throat_c = tip_y - (tip_y - y_bot_inner) / 2.0
        verts.append((float(x), float(throat_c)))
    verts.append((X_RIGHT - 20.0, CENTER_Y))
    return verts


def build_corridor(level: int) -> CorridorGeometry:
    assert level in LEVEL_PERIOD, f"unknown level {level}"
    period = LEVEL_PERIOD[level]
    protrusion = (2 * HALF_W - 2 * PUCK_R + BLOCK_EPS) / 2.0  # = 23.0 by default

    g = CorridorGeometry(level=level, period=period, protrusion=protrusion)

    # ---- walls: top / bottom / left cap / right cap ----
    y_bot = CENTER_Y - HALF_W
    y_top = CENTER_Y + HALF_W
    span_w = X_RIGHT - X_LEFT + 2 * WALL_T
    cx = (X_LEFT + X_RIGHT) / 2.0
    g.walls = [
        (cx, y_bot - WALL_T / 2.0, span_w, WALL_T),
        (cx, y_top + WALL_T / 2.0, span_w, WALL_T),
        (X_LEFT - WALL_T / 2.0, CENTER_Y, WALL_T, 2 * HALF_W + 2 * WALL_T),
        (X_RIGHT + WALL_T / 2.0, CENTER_Y, WALL_T, 2 * HALF_W + 2 * WALL_T),
    ]

    # ---- teeth ----
    if period is not None:
        base_w = 0.9 * period / 2.0
        xs = np.arange(TEETH_X0, TEETH_X1 + 1e-6, period / 2.0)
        for k, x in enumerate(xs):
            if k % 2 == 0:  # from bottom wall, apex up
                g.teeth.append([(x - base_w / 2, y_bot), (x + base_w / 2, y_bot),
                                (x, y_bot + protrusion)])
            else:           # from top wall, apex down
                g.teeth.append([(x + base_w / 2, y_top), (x - base_w / 2, y_top),
                                (x, y_top - protrusion)])

    # ---- weave path (dense, x-monotone) ----
    verts = np.asarray(_weave_vertices(period, protrusion), dtype=np.float64)
    dense = [verts[0]]
    for a, b in zip(verts[:-1], verts[1:]):
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / 4.0)))
        for t in np.linspace(0, 1, n + 1)[1:]:
            dense.append(a + t * (b - a))
    g.path = np.asarray(dense)

    g.puck_start = (X_LEFT + 44.0, CENTER_Y)
    g.agent_start = (X_LEFT + 44.0 - (PUCK_R + AGENT_R + 4.0), CENTER_Y)

    _check_feasibility(g)
    return g


def _check_feasibility(g: CorridorGeometry):
    """Fail fast if the geometry violates the design contracts (README.md)."""
    if g.period is None:
        return
    y_bot_inner = CENTER_Y - HALF_W
    y_top_inner = CENTER_Y + HALF_W
    tip_bot = y_bot_inner + g.protrusion    # bottom-tooth apex y
    tip_top = y_top_inner - g.protrusion    # top-tooth apex y
    # 1) straight horizontal path blocked: need clear band
    #    [tip_bot + PUCK_R, tip_top - PUCK_R] to be empty.
    lo, hi = tip_bot + PUCK_R, tip_top - PUCK_R
    assert lo >= hi + 1.0, (
        f"straight path NOT blocked (band {lo:.1f}..{hi:.1f}); increase protrusion")
    # 2) throat wide enough for puck and for the trailing agent
    throat = 2 * HALF_W - g.protrusion
    assert throat >= 2 * PUCK_R + 6.0, f"throat {throat:.1f} too narrow for puck"
    assert throat >= 2 * AGENT_R + 4.0, f"throat {throat:.1f} too narrow for agent"
    # 3) weave slope steerable by a trailing pusher
    p2p = abs((tip_bot + (y_top_inner - tip_bot) / 2.0)
              - (tip_top - (tip_top - y_bot_inner) / 2.0))
    slope_deg = np.degrees(np.arctan2(p2p, g.period / 2.0))
    assert slope_deg <= MAX_SLOPE_DEG, (
        f"weave slope {slope_deg:.1f}deg > {MAX_SLOPE_DEG}deg; widen period or shrink protrusion")
