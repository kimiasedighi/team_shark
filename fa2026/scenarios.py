from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Goal(str, Enum):
    """Scenario success condition."""

    AGENTS = "agents"
    CARGOES = "cargoes"
    BOTH = "both"


@dataclass(frozen=True, repr=False)
class _Scenario:
    """Engine-owned scenario data.

    The object is intentionally private. Exact scenario parameters are engine
    data rather than part of the local rule interface.
    """

    name: str
    walls: np.ndarray
    agent_positions: np.ndarray
    cargo_positions: np.ndarray
    cargo_sizes: np.ndarray
    target_mask: np.ndarray
    goal: Goal
    vision_radius: int
    battery_capacity: float
    recharge: float
    cargo_mobility: float
    memory_size: int
    pheromone_decays: tuple[float, ...]
    seed: int

    @property
    def grid_shape(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.walls.shape)

    @property
    def n_agents(self) -> int:
        return int(self.agent_positions.shape[0])

    @property
    def n_cargoes(self) -> int:
        return int(self.cargo_positions.shape[0])

    @property
    def n_pheromones(self) -> int:
        return len(self.pheromone_decays)

    def __repr__(self) -> str:
        return f"Scenario(name={self.name!r}, seed={self.seed}, goal={self.goal.value!r})"


def _freeze(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    array.setflags(write=False)
    return array


def _boundary_walls(height: int, width: int) -> np.ndarray:
    walls = np.zeros((height, width), dtype=bool)
    walls[0, :] = True
    walls[-1, :] = True
    walls[:, 0] = True
    walls[:, -1] = True
    return walls


def _inside_square(point: np.ndarray, center: np.ndarray, side: float) -> bool:
    half = side / 2.0
    return bool(
        abs(float(point[0] - center[0])) < half
        and abs(float(point[1] - center[1])) < half
    )


def _square_bounds(center: np.ndarray, side: float) -> tuple[float, float, float, float]:
    half = float(side) / 2.0
    x, y = float(center[0]), float(center[1])
    return x - half, x + half, y - half, y + half


def _squares_overlap_strict(
    center_a: np.ndarray,
    side_a: float,
    center_b: np.ndarray,
    side_b: float,
) -> bool:
    ax0, ax1, ay0, ay1 = _square_bounds(center_a, side_a)
    bx0, bx1, by0, by1 = _square_bounds(center_b, side_b)
    return bool(
        min(ax1, bx1) - max(ax0, bx0) > 1e-10
        and min(ay1, by1) - max(ay0, by0) > 1e-10
    )


def _square_overlaps_mask(center: np.ndarray, side: float, mask: np.ndarray) -> bool:
    xmin, xmax, ymin, ymax = _square_bounds(center, side)
    h, w = mask.shape
    if xmin < 0.0 or ymin < 0.0 or xmax > w or ymax > h:
        return True

    ix0 = max(0, int(np.floor(xmin)))
    ix1 = min(w - 1, int(np.floor(np.nextafter(xmax, -np.inf))))
    iy0 = max(0, int(np.floor(ymin)))
    iy1 = min(h - 1, int(np.floor(np.nextafter(ymax, -np.inf))))
    for iy in range(iy0, iy1 + 1):
        for ix in range(ix0, ix1 + 1):
            if not mask[iy, ix]:
                continue
            if (
                min(xmax, ix + 1.0) - max(xmin, float(ix)) > 1e-10
                and min(ymax, iy + 1.0) - max(ymin, float(iy)) > 1e-10
            ):
                return True
    return False


def _random_target_mask(
    walls: np.ndarray,
    rng: np.random.Generator,
    *,
    size: tuple[int, int],
    at_boundary: bool = False,
    forbidden_mask: np.ndarray | None = None,
    max_attempts: int = 10_000,
) -> np.ndarray:
    """Place one legal rectangular target reproducibly.

    ``size`` is ``(height, width)`` in grid cells. With ``at_boundary=True``
    the target is placed directly adjacent to one of the four boundary walls,
    never on top of the wall itself.
    """

    walls = np.asarray(walls, dtype=bool)
    h, w = walls.shape
    th, tw = (int(size[0]), int(size[1]))
    if th < 1 or tw < 1 or th > h - 2 or tw > w - 2:
        raise ValueError("target size does not fit inside the accessible map")
    if forbidden_mask is not None and np.asarray(forbidden_mask).shape != walls.shape:
        raise ValueError("forbidden_mask must match walls")

    for _ in range(max_attempts):
        if at_boundary:
            side = int(rng.integers(0, 4))
            if side == 0:  # left
                x0 = 1
                y0 = int(rng.integers(1, h - th))
            elif side == 1:  # right
                x0 = w - 1 - tw
                y0 = int(rng.integers(1, h - th))
            elif side == 2:  # bottom
                x0 = int(rng.integers(1, w - tw))
                y0 = 1
            else:  # top
                x0 = int(rng.integers(1, w - tw))
                y0 = h - 1 - th
        else:
            x0 = int(rng.integers(1, w - tw))
            y0 = int(rng.integers(1, h - th))

        target = np.zeros_like(walls, dtype=bool)
        target[y0 : y0 + th, x0 : x0 + tw] = True
        if np.any(target & walls):
            continue
        if forbidden_mask is not None and np.any(target & np.asarray(forbidden_mask, dtype=bool)):
            continue
        return target

    raise RuntimeError("could not place a legal target rectangle")


def _random_boundary_target_mask(
    walls: np.ndarray,
    rng: np.random.Generator,
    *,
    depth: int,
    length: int,
) -> np.ndarray:
    """Place a rectangular target against a randomly chosen arena boundary.

    ``depth`` measures how far the target extends inward from the boundary and
    ``length`` measures its span along the boundary.  The rectangle is rotated
    automatically for horizontal versus vertical boundaries.
    """

    walls = np.asarray(walls, dtype=bool)
    h, w = walls.shape
    depth, length = int(depth), int(length)
    if depth < 1 or length < 1:
        raise ValueError("target depth and length must be positive")
    side = int(rng.integers(0, 4))
    target = np.zeros_like(walls, dtype=bool)

    if side in (0, 1):  # left / right
        if depth > w - 2 or length > h - 2:
            raise ValueError("target does not fit inside accessible map")
        y0 = int(rng.integers(1, h - length))
        x0 = 1 if side == 0 else w - 1 - depth
        target[y0 : y0 + length, x0 : x0 + depth] = True
    else:  # bottom / top
        if depth > h - 2 or length > w - 2:
            raise ValueError("target does not fit inside accessible map")
        x0 = int(rng.integers(1, w - length))
        y0 = 1 if side == 2 else h - 1 - depth
        target[y0 : y0 + depth, x0 : x0 + length] = True

    if np.any(target & walls):
        raise RuntimeError("boundary target unexpectedly overlaps a wall")
    return target


def _random_cargo_positions(
    cargo_sizes: np.ndarray,
    walls: np.ndarray,
    rng: np.random.Generator,
    *,
    forbidden_mask: np.ndarray | None = None,
    center_region: tuple[float, float, float, float] | None = None,
    max_attempts_per_cargo: int = 20_000,
) -> np.ndarray:
    """Place non-overlapping square cargoes reproducibly in free space.

    ``center_region`` optionally restricts cargo centers to
    ``(xmin, xmax, ymin, ymax)`` before the cargo-size margin is applied.
    """

    cargo_sizes = np.asarray(cargo_sizes, dtype=float)
    walls = np.asarray(walls, dtype=bool)
    h, w = walls.shape
    if cargo_sizes.size == 0:
        return np.empty((0, 2), dtype=float)
    if forbidden_mask is not None and np.asarray(forbidden_mask).shape != walls.shape:
        raise ValueError("forbidden_mask must match walls")

    positions: list[np.ndarray] = []
    for side in cargo_sizes:
        half = float(side) / 2.0
        if center_region is None:
            rx0, rx1, ry0, ry1 = 1.0, w - 1.0, 1.0, h - 1.0
        else:
            rx0, rx1, ry0, ry1 = map(float, center_region)
        lo_x = max(1.0 + half + 1e-6, rx0)
        hi_x = min(w - 1.0 - half - 1e-6, rx1)
        lo_y = max(1.0 + half + 1e-6, ry0)
        hi_y = min(h - 1.0 - half - 1e-6, ry1)
        if not (lo_x < hi_x and lo_y < hi_y):
            raise ValueError("cargo does not fit inside the accessible map")

        for _ in range(max_attempts_per_cargo):
            center = np.array(
                [rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)],
                dtype=float,
            )
            if _square_overlaps_mask(center, float(side), walls):
                continue
            if forbidden_mask is not None and _square_overlaps_mask(
                center, float(side), np.asarray(forbidden_mask, dtype=bool)
            ):
                continue
            if any(
                _squares_overlap_strict(center, float(side), old, float(old_side))
                for old, old_side in zip(positions, cargo_sizes[: len(positions)])
            ):
                continue
            positions.append(center)
            break
        else:
            raise RuntimeError("could not place all cargoes legally")

    return np.asarray(positions, dtype=float).reshape((-1, 2))


def _random_agent_positions(
    n_agents: int,
    walls: np.ndarray,
    cargo_positions: np.ndarray,
    cargo_sizes: np.ndarray,
    *,
    rng: np.random.Generator,
    excluded_mask: np.ndarray | None = None,
    region: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Reproducibly distribute agents in accessible free space.

    ``region`` optionally restricts the sampled continuous coordinates to
    ``(xmin, xmax, ymin, ymax)``.  It is useful for teaching scenarios that
    deliberately start the swarm in a loose central population.
    """

    height, width = walls.shape
    if region is None:
        xmin, xmax, ymin, ymax = 1.05, width - 1.05, 1.05, height - 1.05
    else:
        xmin, xmax, ymin, ymax = map(float, region)
        xmin = max(1.05, xmin)
        xmax = min(width - 1.05, xmax)
        ymin = max(1.05, ymin)
        ymax = min(height - 1.05, ymax)
        if not (xmin < xmax and ymin < ymax):
            raise ValueError("agent sampling region is empty")

    positions: list[tuple[float, float]] = []

    while len(positions) < n_agents:
        point = np.array(
            [rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)],
            dtype=float,
        )
        ix, iy = int(point[0]), int(point[1])
        if walls[iy, ix]:
            continue
        if excluded_mask is not None and excluded_mask[iy, ix]:
            continue
        if any(
            _inside_square(point, cargo_positions[i], float(cargo_sizes[i]))
            for i in range(len(cargo_sizes))
        ):
            continue
        positions.append((float(point[0]), float(point[1])))

    return np.asarray(positions, dtype=float)


def _gather(seed: int = 3101) -> _Scenario:
    """Teaching scenario 1: explore, communicate, and gather all agents."""

    rng = np.random.default_rng(seed)
    height, width = 48, 64
    walls = _boundary_walls(height, width)
    target = _random_boundary_target_mask(walls, rng, depth=5, length=12)

    cargo_positions = np.empty((0, 2), dtype=float)
    cargo_sizes = np.empty((0,), dtype=float)
    # Start as a loose central population.  The exact positions and the target
    # boundary/offset all vary with the same seed.
    agent_positions = _random_agent_positions(
        100,
        walls,
        cargo_positions,
        cargo_sizes,
        rng=rng,
        excluded_mask=target,
        region=(22.0, 42.0, 17.0, 31.0),
    )

    return _Scenario(
        name="gather",
        walls=_freeze(walls),
        agent_positions=_freeze(agent_positions),
        cargo_positions=_freeze(cargo_positions),
        cargo_sizes=_freeze(cargo_sizes),
        target_mask=_freeze(target),
        goal=Goal.AGENTS,
        vision_radius=4,
        battery_capacity=24.0,
        recharge=0.45,
        cargo_mobility=0.10,  # irrelevant here, but kept within the common schema
        memory_size=8,
        pheromone_decays=(0.010,),
        seed=int(seed),
    )


def _transport(seed: int = 3102) -> _Scenario:
    """Teaching scenario 2: recruit agents for slow collective transport."""

    rng = np.random.default_rng(seed)
    height, width = 34, 42
    walls = _boundary_walls(height, width)
    target = _random_boundary_target_mask(walls, rng, depth=8, length=10)

    cargo_sizes = np.array([2.0], dtype=float)
    cargo_positions = _random_cargo_positions(
        cargo_sizes,
        walls,
        rng,
        forbidden_mask=target,
        center_region=(18.0, 24.0, 14.0, 20.0),
    )
    agent_positions = _random_agent_positions(
        100,
        walls,
        cargo_positions,
        cargo_sizes,
        rng=rng,
        excluded_mask=target,
    )

    return _Scenario(
        name="transport",
        walls=_freeze(walls),
        agent_positions=_freeze(agent_positions),
        cargo_positions=_freeze(cargo_positions),
        cargo_sizes=_freeze(cargo_sizes),
        target_mask=_freeze(target),
        goal=Goal.CARGOES,
        vision_radius=16,
        battery_capacity=30.0,
        recharge=0.85,
        cargo_mobility=0.05,
        memory_size=8,
        pheromone_decays=(0.08,),
        seed=int(seed),
    )


def _communicate(seed: int = 3103) -> _Scenario:
    """Teaching scenario 3: discover target information and transport cargo."""

    rng = np.random.default_rng(seed)
    height, width = 48, 64
    walls = _boundary_walls(height, width)
    target = _random_boundary_target_mask(walls, rng, depth=5, length=10)

    cargo_sizes = np.array([2.0], dtype=float)
    cargo_positions = _random_cargo_positions(
        cargo_sizes,
        walls,
        rng,
        forbidden_mask=target,
        center_region=(22.0, 42.0, 16.0, 32.0),
    )
    agent_positions = _random_agent_positions(
        140,
        walls,
        cargo_positions,
        cargo_sizes,
        rng=rng,
        excluded_mask=target,
        region=(20.0, 44.0, 14.0, 34.0),
    )

    return _Scenario(
        name="communicate",
        walls=_freeze(walls),
        agent_positions=_freeze(agent_positions),
        cargo_positions=_freeze(cargo_positions),
        cargo_sizes=_freeze(cargo_sizes),
        target_mask=_freeze(target),
        goal=Goal.CARGOES,
        vision_radius=4,
        battery_capacity=32.0,
        recharge=0.70,
        cargo_mobility=0.05,
        memory_size=8,
        pheromone_decays=(0.005,),
        seed=int(seed),
    )


_DEFAULT_SEEDS = {
    "gather": 3101,
    "transport": 3102,
    "communicate": 3103,
}


def load_scenario(name: str, *, seed: int | None = None) -> _Scenario:
    """Load one of the public teaching scenarios.

    ``seed`` controls every randomized initial-geometry choice made by the
    scenario generator. Omitting it uses the scenario's deterministic default.
    Student notebooks normally call :func:`fa2026.run` rather than this
    lower-level helper.
    """

    normalized = name.strip().lower()
    if normalized not in _DEFAULT_SEEDS:
        available = ", ".join(repr(k) for k in _DEFAULT_SEEDS)
        raise KeyError(f"unknown scenario {name!r}; available scenarios: {available}")

    actual_seed = _DEFAULT_SEEDS[normalized] if seed is None else int(seed)
    if normalized == "gather":
        return _gather(actual_seed)
    if normalized == "transport":
        return _transport(actual_seed)
    return _communicate(actual_seed)

_SCENARIO_SUMMARIES = {
    "gather": {
        "title": "Search and gather",
        "goal": "All 100 agent points must be inside the target simultaneously.",
        "target": "5 cells deep x 12 cells along one arena boundary; side and offset are randomized.",
        "agents": "100 agents sampled in a loose central region.",
        "cargo": "None.",
        "randomized": "target side/position and all agent positions",
        "main_idea": "exploration and collective guidance",
    },
    "transport": {
        "title": "Cooperative transport",
        "goal": "The single square cargo must lie fully inside the target.",
        "target": "8 cells deep x 10 cells along one arena boundary; side and offset are randomized.",
        "agents": "100 agents sampled throughout accessible free space.",
        "cargo": "One 2 x 2 square cargo, initialized near the center with randomized position.",
        "randomized": "target side/position, cargo position, and all agent positions",
        "main_idea": "recruitment, positioning, and collective pushing",
    },
    "communicate": {
        "title": "Search, communicate, transport",
        "goal": "The single square cargo must lie fully inside the target.",
        "target": "5 cells deep x 10 cells along one arena boundary; side and offset are randomized.",
        "agents": "140 agents sampled in a broad central region.",
        "cargo": "One 2 x 2 square cargo, initialized in the central region with randomized position.",
        "randomized": "target side/position, cargo position, and all agent positions",
        "main_idea": "combine exploration, communication, and transport",
    },
}


def info(name: str | None = None) -> None:
    """Print the public specification of the introductory scenarios.

    ``info()`` lists the available scenarios. ``info(name)`` prints the fixed
    rules and parameters for one scenario. It intentionally does not reveal
    the realized coordinates produced by a particular seed and does not
    expose the internal mutable simulation state.
    """

    if name is None:
        print("Available scenarios:\n")
        for scenario_name in _DEFAULT_SEEDS:
            summary = _SCENARIO_SUMMARIES[scenario_name]
            print(f"  {scenario_name:<12} {summary['main_idea']}")
        print("\nUse fa.info(\"gather\") (or another name) for full parameters.")
        return None

    normalized = str(name).strip().lower()
    if normalized not in _DEFAULT_SEEDS:
        available = ", ".join(repr(k) for k in _DEFAULT_SEEDS)
        raise KeyError(f"unknown scenario {name!r}; available scenarios: {available}")

    scenario = load_scenario(normalized)
    summary = _SCENARIO_SUMMARIES[normalized]
    height, width = scenario.grid_shape

    print(f"Scenario: {normalized} — {summary['title']}\n")
    print("Goal")
    print(f"  {summary['goal']}")
    print("  Score: turns until success\n")

    print("World")
    print(f"  Arena: {width} x {height} cells (width x height)")
    print(f"  Agents: {scenario.n_agents}")
    print(f"  Agent initialization: {summary['agents']}")
    print(f"  Cargoes: {scenario.n_cargoes}")
    print(f"  Cargo setup: {summary['cargo']}")
    if scenario.n_cargoes:
        print(f"  Cargo mobility mu: {scenario.cargo_mobility:g}")
        print("  Cargo displacement: (mu / cargo area) * summed push force")
    print(f"  Target: {summary['target']}")
    print(f"  Randomized by seed: {summary['randomized']}\n")

    print("Agent resources")
    print(f"  Vision radius: {scenario.vision_radius} cells "
          f"({2 * scenario.vision_radius + 1} x {2 * scenario.vision_radius + 1} local patch)")
    print(f"  Memory: {scenario.memory_size} floats per agent")
    print(f"  Battery capacity: {scenario.battery_capacity:g}")
    print(f"  Recharge per turn: {scenario.recharge:g}\n")

    print("Pheromones")
    print(f"  Channels: {scenario.n_pheromones}")
    for channel, decay in enumerate(scenario.pheromone_decays):
        print(f"  Channel {channel} decay fraction per turn: {decay:g}")
    print("  Sensing: local 3 x 3 concentration patch per channel\n")

    print("Action rules and energy")
    print("  Push: finite p >= 0; cost p^2; effective only while the agent is inside a cargo")
    print("        direction is automatic: from the agent toward that cargo's center")
    print("  Move: finite (dx, dy); cost dx^2 + dy^2; walls stop motion")
    print("  Pheromone q > 0: deposit at the final cell; cost q / decay")
    print("  Pheromone q < 0: request uptake at the old cell; absorbed amount recovers amount / decay")
    print("  There is no separate hard magnitude cap on push or move; energy affordability is the main limit.")
    print("  Uptake is processed before affordability. If the outgoing push + move + positive")
    print("  deposits are still unaffordable, that whole outgoing package is rejected; absorbed")
    print("  energy and returned memory are retained.")
    return None

