from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from typing import Callable

import numpy as np


class Cell(IntFlag):
    """Bit flags used in :class:`Observation.vision`."""

    EMPTY = 0
    WALL = 1
    TARGET = 2
    CARGO = 4


@dataclass(frozen=True)
class Observation:
    """Local information available to one agent for one turn.

    ``energy`` is the battery level after automatic recharge at the beginning
    of the turn.

    ``cell_position`` is the fractional ``(x, y)`` position inside the
    current grid cell. Absolute world coordinates are not exposed.

    ``vision`` is a square array of :class:`Cell` bit masks centered on the
    current grid cell. Its shape reveals the scenario's vision radius.

    ``pheromones`` has shape ``(n_channels, 3, 3)``. The center entry of each
    channel is the concentration in the current cell.

    ``on_cargo`` is true exactly when the point agent lies strictly inside one
    cargo at the beginning of the turn and can therefore push that cargo.
    """

    energy: float
    cell_position: tuple[float, float]
    vision: np.ndarray
    pheromones: np.ndarray
    on_cargo: bool


@dataclass(frozen=True)
class Action:
    """Requested action for one turn, in execution order.

    ``push`` is a non-negative scalar. If the agent is on a cargo, the force
    points automatically from the agent towards that cargo's center.

    ``move`` is the requested continuous displacement ``(dx, dy)``.

    ``pheromones`` contains one signed value per available channel. Negative
    values request uptake from the current cell before push/move; positive
    values request deposition at the final cell after movement. Leaving it
    empty means no pheromone action on any channel.
    """

    push: float = 0.0
    move: tuple[float, float] = (0.0, 0.0)
    pheromones: tuple[float, ...] = ()


@dataclass(frozen=True)
class Result:
    success: bool
    turns: int


class RuleError(RuntimeError):
    """Raised when ``rule`` returns an invalid action or memory state."""


RuleFunction = Callable[[Observation, np.ndarray], tuple[Action, np.ndarray]]
