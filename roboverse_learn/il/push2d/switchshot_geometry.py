"""ForkReach geometry, retained under its historical SwitchShot N/lv4 name."""

from dataclasses import dataclass, field

import numpy as np

WORLD = 512.0
CENTER_Y = 256.0
WALL_T = 10.0
AGENT_R = 16.0
GOAL_X = 470.0
GOAL_DY = 92.0
GOAL_R = 26.0
STAGING_X = 330.0
X_MIN, X_MAX = WALL_T, WORLD - WALL_T
Y_MIN, Y_MAX = WALL_T, WORLD - WALL_T
LEVELS = (4,)


@dataclass
class SwitchGeometry:
    variant: str
    walls: list = field(default_factory=list)
    goals: dict = field(default_factory=dict)
    goal_r: float = GOAL_R
    agent_start: tuple = (40.0, CENTER_Y)


def build_switch(variant: str = "N") -> SwitchGeometry:
    if variant != "N":
        raise ValueError("Only ForkReach (SwitchShot variant N, level 4) is retained")
    g = SwitchGeometry(variant=variant)
    g.goals = {"top": (GOAL_X, CENTER_Y - GOAL_DY),
               "bottom": (GOAL_X, CENTER_Y + GOAL_DY)}
    cx = WORLD / 2.0
    g.walls = [
        (cx, WALL_T / 2.0, WORLD, WALL_T),
        (cx, WORLD - WALL_T / 2.0, WORLD, WALL_T),
        (WALL_T / 2.0, cx, WALL_T, WORLD),
        (WORLD - WALL_T / 2.0, cx, WALL_T, WORLD),
    ]
    for x, y in g.goals.values():
        assert X_MIN + GOAL_R < x < X_MAX - 2
        assert Y_MIN + GOAL_R < y < Y_MAX - GOAL_R
    return g


def draw_schedule(level: int, rng: np.random.Generator) -> dict:
    """Both goals are valid from reset; preserve the original seed stream."""
    if level != 4:
        raise ValueError("Only ForkReach (SwitchShot variant N, level 4) is retained")
    # The original lv4 implementation drew an unused initial side before
    # selecting both goals. Keep that draw so old seeds reproduce old starts.
    rng.integers(2)
    return dict(s0=0, s1=None, lock=0, initial=None, final=None,
                valid_goals=("top", "bottom"))


def signal_colors(schedule: dict, t: int) -> dict:
    """The two goal discs stay green throughout every ForkReach episode."""
    return {"top": "green", "bottom": "green"}
