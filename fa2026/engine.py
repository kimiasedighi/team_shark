from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .api import Action, Cell, Observation, Result, RuleError, RuleFunction
from .scenarios import Goal, _Scenario

_EPS = 1e-12
_GEOM_EPS = 1e-10
_EVENT_TOL = 1e-10


@dataclass
class _AgentState:
    position: np.ndarray
    energy: float
    memory: np.ndarray


@dataclass(frozen=True)
class _PreparedAction:
    push: float
    move: np.ndarray
    pheromones: np.ndarray
    outgoing_cost: float

    @property
    def uptake_request(self) -> np.ndarray:
        return np.maximum(-self.pheromones, 0.0)

    @property
    def deposit(self) -> np.ndarray:
        return np.maximum(self.pheromones, 0.0)


class Engine:
    """Internal simulation engine. Student code normally uses :func:`run`."""

    def __init__(self, scenario: _Scenario, rule: RuleFunction):
        self.scenario = scenario
        self.rule_fn = rule
        self._validate_scenario()

        self.walls = np.asarray(scenario.walls, dtype=bool).copy()
        self.target_mask = np.asarray(scenario.target_mask, dtype=bool).copy()
        self.cargo_positions = np.asarray(scenario.cargo_positions, dtype=float).copy()
        self.cargo_sizes = np.asarray(scenario.cargo_sizes, dtype=float).copy()
        self.pheromone_decays = np.asarray(scenario.pheromone_decays, dtype=float)
        self.pheromone_fields = np.zeros(
            (scenario.n_pheromones, *scenario.grid_shape), dtype=float
        )

        self._static_cells = np.zeros(scenario.grid_shape, dtype=np.uint8)
        self._static_cells[self.walls] |= int(Cell.WALL)
        self._static_cells[self.target_mask] |= int(Cell.TARGET)

        ys, xs = np.nonzero(self.walls)
        self._wall_centers = np.column_stack((xs + 0.5, ys + 0.5)).astype(float)

        self.agents = [
            _AgentState(
                position=np.asarray(position, dtype=float).copy(),
                energy=float(scenario.battery_capacity),
                memory=np.zeros(scenario.memory_size, dtype=float),
            )
            for position in np.asarray(scenario.agent_positions, dtype=float)
        ]
        self.turn = 0

    # ------------------------------------------------------------------
    # Validation

    def _validate_scenario(self) -> None:
        s = self.scenario
        h, w = s.grid_shape
        if not (12 <= h <= 160 and 12 <= w <= 160):
            raise ValueError("map dimensions must each lie in [12, 160]")
        if not (4 <= s.n_agents <= 512):
            raise ValueError("number of agents must lie in [4, 512]")
        if not (0 <= s.n_cargoes <= 10):
            raise ValueError("number of cargoes must lie in [0, 10]")
        if not isinstance(s.goal, Goal):
            raise ValueError("scenario goal must be a Goal value")
        if s.goal in (Goal.CARGOES, Goal.BOTH) and s.n_cargoes == 0:
            raise ValueError("cargo-based goals require at least one cargo")
        if not (1 <= s.vision_radius <= 16):
            raise ValueError("vision_radius must lie in [1, 16]")
        if not (1.0 <= s.battery_capacity <= 256.0):
            raise ValueError("battery_capacity must lie in [1, 256]")
        if not (0.001 * s.battery_capacity <= s.recharge <= s.battery_capacity):
            raise ValueError("recharge must lie in [0.001, 1] times battery_capacity")
        if not (0.01 <= s.cargo_mobility <= 2.0):
            raise ValueError("cargo_mobility must lie in [0.01, 2]")
        if not (0 <= s.memory_size <= 256):
            raise ValueError("memory_size must lie in [0, 256]")
        if not (0 <= s.n_pheromones <= 8):
            raise ValueError("number of pheromone channels must lie in [0, 8]")
        if any(not (0.001 <= float(d) <= 1.0) for d in s.pheromone_decays):
            raise ValueError("pheromone decay values must lie in [0.001, 1]")

        walls = np.asarray(s.walls, dtype=bool)
        target = np.asarray(s.target_mask, dtype=bool)
        agent_positions = np.asarray(s.agent_positions, dtype=float)
        cargo_positions = np.asarray(s.cargo_positions, dtype=float)
        cargo_sizes = np.asarray(s.cargo_sizes, dtype=float)

        if walls.shape != (h, w):
            raise ValueError(f"walls must have shape {(h, w)}")
        if target.shape != (h, w):
            raise ValueError(f"target_mask must have shape {(h, w)}")
        if agent_positions.shape != (s.n_agents, 2):
            raise ValueError(f"agent_positions must have shape {(s.n_agents, 2)}")
        if cargo_positions.shape != (s.n_cargoes, 2):
            raise ValueError(f"cargo_positions must have shape {(s.n_cargoes, 2)}")
        if cargo_sizes.shape != (s.n_cargoes,):
            raise ValueError(f"cargo_sizes must have shape {(s.n_cargoes,)}")
        if not np.all(np.isfinite(agent_positions)):
            raise ValueError("agent positions must be finite")
        if not np.all(np.isfinite(cargo_positions)):
            raise ValueError("cargo positions must be finite")
        if not np.all(np.isfinite(cargo_sizes)):
            raise ValueError("cargo sizes must be finite")
        if np.any(cargo_sizes < 1.0) or np.any(cargo_sizes > 8.0):
            raise ValueError("cargo side lengths must lie in [1, 8]")

        if not np.any(target):
            raise ValueError("target_mask must contain one rectangular target")
        if np.any(walls & target):
            raise ValueError("target may not overlap walls")
        ys, xs = np.nonzero(target)
        y0, y1 = int(np.min(ys)), int(np.max(ys))
        x0, x1 = int(np.min(xs)), int(np.max(xs))
        rectangle = target[y0 : y1 + 1, x0 : x1 + 1]
        if not np.all(rectangle) or int(np.count_nonzero(target)) != rectangle.size:
            raise ValueError("target_mask must describe one axis-aligned rectangle")

        for position in agent_positions:
            if self._point_in_wall_static(position, walls):
                raise ValueError("an initial agent position lies inside a wall")

        for i, (position, size) in enumerate(zip(cargo_positions, cargo_sizes)):
            if self._box_overlaps_wall_static(position, float(size), walls):
                raise ValueError(f"initial cargo {i} overlaps a wall or leaves the map")

        for i in range(s.n_cargoes):
            for j in range(i + 1, s.n_cargoes):
                if self._boxes_overlap_strict(
                    cargo_positions[i], cargo_sizes[i], cargo_positions[j], cargo_sizes[j]
                ):
                    raise ValueError("initial cargoes may not overlap one another")

    # ------------------------------------------------------------------
    # Public-ish state used by visualization/tests

    @property
    def agent_positions(self) -> np.ndarray:
        return np.array([a.position for a in self.agents], dtype=float)

    @property
    def agent_energies(self) -> np.ndarray:
        return np.array([a.energy for a in self.agents], dtype=float)

    # ------------------------------------------------------------------
    # Basic geometry

    @staticmethod
    def _point_in_wall_static(position: np.ndarray, walls: np.ndarray) -> bool:
        x, y = float(position[0]), float(position[1])
        h, w = walls.shape
        if x < 0.0 or y < 0.0 or x >= w or y >= h:
            return True
        return bool(walls[int(math.floor(y)), int(math.floor(x))])

    def _point_in_wall(self, position: np.ndarray) -> bool:
        return self._point_in_wall_static(position, self.walls)

    def _cell_is_wall(self, ix: int, iy: int) -> bool:
        h, w = self.walls.shape
        return ix < 0 or iy < 0 or ix >= w or iy >= h or bool(self.walls[iy, ix])

    @staticmethod
    def _box_bounds(center: np.ndarray, size: float) -> tuple[float, float, float, float]:
        half = float(size) / 2.0
        x, y = float(center[0]), float(center[1])
        return x - half, x + half, y - half, y + half

    @classmethod
    def _boxes_overlap_strict(
        cls,
        center_a: np.ndarray,
        size_a: float,
        center_b: np.ndarray,
        size_b: float,
    ) -> bool:
        ax0, ax1, ay0, ay1 = cls._box_bounds(center_a, float(size_a))
        bx0, bx1, by0, by1 = cls._box_bounds(center_b, float(size_b))
        return bool(
            min(ax1, bx1) - max(ax0, bx0) > _GEOM_EPS
            and min(ay1, by1) - max(ay0, by0) > _GEOM_EPS
        )

    @classmethod
    def _box_overlaps_wall_static(
        cls, center: np.ndarray, size: float, walls: np.ndarray
    ) -> bool:
        xmin, xmax, ymin, ymax = cls._box_bounds(center, size)
        h, w = walls.shape
        if xmin < -_GEOM_EPS or ymin < -_GEOM_EPS or xmax > w + _GEOM_EPS or ymax > h + _GEOM_EPS:
            return True

        ix0 = max(0, int(math.floor(xmin)))
        ix1 = min(w - 1, int(math.floor(np.nextafter(xmax, -math.inf))))
        iy0 = max(0, int(math.floor(ymin)))
        iy1 = min(h - 1, int(math.floor(np.nextafter(ymax, -math.inf))))
        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                if not walls[iy, ix]:
                    continue
                if (
                    min(xmax, ix + 1.0) - max(xmin, float(ix)) > _GEOM_EPS
                    and min(ymax, iy + 1.0) - max(ymin, float(iy)) > _GEOM_EPS
                ):
                    return True
        return False

    def _cargo_index_at_point(self, position: np.ndarray) -> int | None:
        """Cargo strictly containing a point, or None.

        Valid scenarios never contain overlapping cargo interiors, so at most
        one cargo can match. Boundaries deliberately do not count as on-cargo.
        """

        x, y = float(position[0]), float(position[1])
        found: int | None = None
        for i, (center, size) in enumerate(zip(self.cargo_positions, self.cargo_sizes)):
            half = float(size) / 2.0
            if (
                abs(x - float(center[0])) < half - _GEOM_EPS
                and abs(y - float(center[1])) < half - _GEOM_EPS
            ):
                if found is not None:
                    raise RuntimeError("agent lies inside overlapping cargoes")
                found = i
        return found

    def _move_agent_until_wall(self, start: np.ndarray, displacement: np.ndarray) -> np.ndarray:
        """Move a point along a segment, stopping at the first crossed wall cell."""

        dx, dy = float(displacement[0]), float(displacement[1])
        if abs(dx) <= _EPS and abs(dy) <= _EPS:
            return start.copy()

        x, y = float(start[0]), float(start[1])
        ix, iy = int(math.floor(x)), int(math.floor(y))
        step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
        step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)

        if step_x:
            boundary_x = ix + 1 if step_x > 0 else ix
            t_max_x = (boundary_x - x) / dx
            t_delta_x = 1.0 / abs(dx)
        else:
            t_max_x = math.inf
            t_delta_x = math.inf

        if step_y:
            boundary_y = iy + 1 if step_y > 0 else iy
            t_max_y = (boundary_y - y) / dy
            t_delta_y = 1.0 / abs(dy)
        else:
            t_max_y = math.inf
            t_delta_y = math.inf

        def stop_at(t_hit: float) -> np.ndarray:
            # Retreat by a tiny *spatial* margin. Merely taking the previous
            # floating-point value of ``t_hit`` is not sufficient: after the
            # multiplication/addition it can still round exactly onto the next
            # (wall) cell boundary, from where a later turn could escape the
            # arena.
            speed_inf = max(abs(dx), abs(dy), _EPS)
            t_margin = 10.0 * _GEOM_EPS / speed_inf
            t_safe = max(0.0, t_hit - t_margin)
            point = start + t_safe * displacement
            if self._point_in_wall(point):
                # Defensive fallback for extreme floating-point cases. The
                # beginning-of-turn position is guaranteed to be valid.
                return start.copy()
            return point

        while True:
            t_next = min(t_max_x, t_max_y)
            if t_next > 1.0:
                return start + displacement

            if abs(t_max_x - t_max_y) <= 1e-14:
                candidates: list[tuple[int, int]] = []
                if step_x:
                    candidates.append((ix + step_x, iy))
                if step_y:
                    candidates.append((ix, iy + step_y))
                if step_x and step_y:
                    candidates.append((ix + step_x, iy + step_y))
                if any(self._cell_is_wall(cx, cy) for cx, cy in candidates):
                    return stop_at(t_next)
                ix += step_x
                iy += step_y
                t_max_x += t_delta_x
                t_max_y += t_delta_y
            elif t_max_x < t_max_y:
                next_ix = ix + step_x
                if self._cell_is_wall(next_ix, iy):
                    return stop_at(t_max_x)
                ix = next_ix
                t_max_x += t_delta_x
            else:
                next_iy = iy + step_y
                if self._cell_is_wall(ix, next_iy):
                    return stop_at(t_max_y)
                iy = next_iy
                t_max_y += t_delta_y

    # ------------------------------------------------------------------
    # Swept cargo collisions

    @staticmethod
    def _axis_overlap_interval(relative_pos: float, relative_velocity: float, half_sum: float):
        """Open time interval in which two 1-D intervals strictly overlap."""

        if abs(relative_velocity) <= _EPS:
            if abs(relative_pos) < half_sum - _GEOM_EPS:
                return -math.inf, math.inf
            return None
        a = (-half_sum - relative_pos) / relative_velocity
        b = (half_sum - relative_pos) / relative_velocity
        return (min(a, b), max(a, b))

    @classmethod
    def _swept_box_collision_time(
        cls,
        center_a: np.ndarray,
        size_a: float,
        velocity_a: np.ndarray,
        center_b: np.ndarray,
        size_b: float,
        velocity_b: np.ndarray,
        horizon: float,
    ) -> float | None:
        """First contact time before strict AABB overlap would begin."""

        relative = np.asarray(center_b, dtype=float) - np.asarray(center_a, dtype=float)
        relative_velocity = np.asarray(velocity_b, dtype=float) - np.asarray(velocity_a, dtype=float)
        half_sum = (float(size_a) + float(size_b)) / 2.0

        ix = cls._axis_overlap_interval(relative[0], relative_velocity[0], half_sum)
        iy = cls._axis_overlap_interval(relative[1], relative_velocity[1], half_sum)
        if ix is None or iy is None:
            return None

        entry = max(ix[0], iy[0], 0.0)
        exit_time = min(ix[1], iy[1], horizon)
        # There must be a non-zero time interval after contact during which the
        # boxes would overlap. This rejects touching boxes moving apart/parallel.
        if entry < exit_time - _EVENT_TOL and entry <= horizon + _EVENT_TOL:
            return max(0.0, min(float(entry), horizon))
        return None

    def _world_boundary_collision_time(
        self, center: np.ndarray, size: float, velocity: np.ndarray, horizon: float
    ) -> float | None:
        half = float(size) / 2.0
        x, y = float(center[0]), float(center[1])
        vx, vy = float(velocity[0]), float(velocity[1])
        h, w = self.scenario.grid_shape
        candidates: list[float] = []

        if vx < -_EPS:
            candidates.append((half - x) / vx)
        elif vx > _EPS:
            candidates.append((w - half - x) / vx)
        if vy < -_EPS:
            candidates.append((half - y) / vy)
        elif vy > _EPS:
            candidates.append((h - half - y) / vy)

        valid = [t for t in candidates if -_EVENT_TOL <= t <= horizon + _EVENT_TOL]
        if not valid:
            return None
        return max(0.0, min(valid))

    def _wall_collision_time(
        self, center: np.ndarray, size: float, velocity: np.ndarray, horizon: float
    ) -> float | None:
        best = self._world_boundary_collision_time(center, size, velocity, horizon)
        if self._wall_centers.size == 0:
            return best

        # Only wall cells intersecting the swept cargo bounding box can matter.
        # This keeps exact collision semantics while avoiding a scan of every
        # wall cell for every cargo on every turn.
        end = np.asarray(center, dtype=float) + np.asarray(velocity, dtype=float) * horizon
        margin = float(size) / 2.0 + 0.5 + _GEOM_EPS
        lo = np.minimum(center, end) - margin
        hi = np.maximum(center, end) + margin
        candidates = self._wall_centers[
            (self._wall_centers[:, 0] >= lo[0])
            & (self._wall_centers[:, 0] <= hi[0])
            & (self._wall_centers[:, 1] >= lo[1])
            & (self._wall_centers[:, 1] <= hi[1])
        ]

        zero = np.zeros(2, dtype=float)
        for wall_center in candidates:
            t = self._swept_box_collision_time(
                center, size, velocity, wall_center, 1.0, zero, horizon
            )
            if t is not None and (best is None or t < best):
                best = t
        return best

    def _move_cargoes_simultaneously(self, displacements: np.ndarray) -> None:
        """Move cargoes over t in [0,1], stopping objects at first collisions."""

        velocities = np.asarray(displacements, dtype=float).copy()
        positions = self.cargo_positions.copy()
        active = np.linalg.norm(velocities, axis=1) > _EPS
        t_global = 0.0

        while t_global < 1.0 - _EVENT_TOL and np.any(active):
            horizon = 1.0 - t_global
            events: list[tuple[float, tuple[int, ...]]] = []

            for i in np.flatnonzero(active):
                t = self._wall_collision_time(
                    positions[i], float(self.cargo_sizes[i]), velocities[i], horizon
                )
                if t is not None:
                    events.append((t, (int(i),)))

            n = len(positions)
            for i in range(n):
                for j in range(i + 1, n):
                    if not (active[i] or active[j]):
                        continue
                    vi = velocities[i] if active[i] else np.zeros(2, dtype=float)
                    vj = velocities[j] if active[j] else np.zeros(2, dtype=float)
                    t = self._swept_box_collision_time(
                        positions[i],
                        float(self.cargo_sizes[i]),
                        vi,
                        positions[j],
                        float(self.cargo_sizes[j]),
                        vj,
                        horizon,
                    )
                    if t is not None:
                        events.append((t, (i, j)))

            if not events:
                positions[active] += velocities[active] * horizon
                t_global = 1.0
                break

            earliest = min(event[0] for event in events)
            earliest = max(0.0, min(float(earliest), horizon))
            if earliest > 0.0:
                positions[active] += velocities[active] * earliest
                t_global += earliest

            involved: set[int] = set()
            for event_time, indices in events:
                if abs(event_time - earliest) <= _EVENT_TOL:
                    involved.update(indices)

            # Only currently moving cargoes need state changes. Marking both
            # members of a cargo-cargo event is intentional: both stop.
            changed = False
            for index in involved:
                if active[index]:
                    active[index] = False
                    changed = True

            if not changed:
                # Numerical safety: avoid a zero-time loop without changing the
                # intended collision semantics.
                tiny = min(horizon, 1e-12)
                positions[active] += velocities[active] * tiny
                t_global += tiny

        self.cargo_positions = positions

    # ------------------------------------------------------------------
    # Observation

    def _cargo_overlaps_cell(self, cargo_index: int, ix: int, iy: int) -> bool:
        xmin, xmax, ymin, ymax = self._box_bounds(
            self.cargo_positions[cargo_index], float(self.cargo_sizes[cargo_index])
        )
        return bool(
            min(xmax, ix + 1.0) - max(xmin, float(ix)) > _GEOM_EPS
            and min(ymax, iy + 1.0) - max(ymin, float(iy)) > _GEOM_EPS
        )

    def _visible_cell_map(self) -> np.ndarray:
        cells = self._static_cells.copy()
        h, w = self.scenario.grid_shape
        for cargo_index, (center, size) in enumerate(
            zip(self.cargo_positions, self.cargo_sizes)
        ):
            xmin, xmax, ymin, ymax = self._box_bounds(center, float(size))
            ix0 = max(0, int(math.floor(xmin)))
            ix1 = min(w - 1, int(math.floor(np.nextafter(xmax, -math.inf))))
            iy0 = max(0, int(math.floor(ymin)))
            iy1 = min(h - 1, int(math.floor(np.nextafter(ymax, -math.inf))))
            for iy in range(iy0, iy1 + 1):
                for ix in range(ix0, ix1 + 1):
                    if self._cargo_overlaps_cell(cargo_index, ix, iy):
                        cells[iy, ix] |= int(Cell.CARGO)
        return cells

    def _observation_for(
        self,
        agent: _AgentState,
        padded_cells: np.ndarray | None = None,
        padded_pheromones: np.ndarray | None = None,
    ) -> Observation:
        x, y = float(agent.position[0]), float(agent.position[1])
        cx, cy = int(math.floor(x)), int(math.floor(y))
        radius = self.scenario.vision_radius
        size = 2 * radius + 1

        if padded_cells is None:
            padded_cells = np.pad(
                self._visible_cell_map(),
                radius,
                mode="constant",
                constant_values=int(Cell.WALL),
            )
        vision = padded_cells[cy : cy + size, cx : cx + size].copy()
        vision.setflags(write=False)

        n_channels = self.scenario.n_pheromones
        if n_channels == 0:
            pheromones = np.zeros((0, 3, 3), dtype=float)
        else:
            if padded_pheromones is None:
                padded_pheromones = np.pad(
                    self.pheromone_fields,
                    ((0, 0), (1, 1), (1, 1)),
                    mode="constant",
                )
            pheromones = padded_pheromones[:, cy : cy + 3, cx : cx + 3].copy()
        pheromones.setflags(write=False)

        return Observation(
            energy=float(agent.energy),
            cell_position=(x - math.floor(x), y - math.floor(y)),
            vision=vision,
            pheromones=pheromones,
            on_cargo=self._cargo_index_at_point(agent.position) is not None,
        )

    # ------------------------------------------------------------------
    # Rule actions and pheromones

    def _prepare_action(self, action: Action) -> _PreparedAction:
        if not isinstance(action, Action):
            raise RuleError("rule(...) must return fa.Action")

        try:
            push = float(action.push)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuleError("Action.push must be one finite non-negative number") from exc
        if not math.isfinite(push) or push < 0.0:
            raise RuleError("Action.push must be finite and non-negative")

        try:
            move = np.asarray(action.move, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuleError("Action.move must contain two finite numbers") from exc
        if move.shape != (2,) or not np.all(np.isfinite(move)):
            raise RuleError("Action.move must contain exactly two finite numbers")

        try:
            pheromones = np.asarray(action.pheromones, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuleError("Action.pheromones must be a finite vector") from exc
        expected = (self.scenario.n_pheromones,)
        if pheromones.shape == (0,) and self.scenario.n_pheromones > 0:
            # Convenient default: omitting ``pheromones`` means no signal.
            pheromones = np.zeros(self.scenario.n_pheromones, dtype=float)
        elif pheromones.shape != expected:
            raise RuleError(
                f"Action.pheromones must have shape {expected}, got {pheromones.shape}"
            )
        if not np.all(np.isfinite(pheromones)):
            raise RuleError("Action.pheromones values must be finite")

        # Very large but still finite values can overflow the implied energy
        # request. Treat those as invalid rule output rather than letting
        # infinities/NaNs enter the uptake or affordability calculations.
        with np.errstate(over="ignore", invalid="ignore"):
            pheromone_energy_magnitudes = (
                np.abs(pheromones) / self.pheromone_decays
                if pheromones.size
                else np.zeros(0, dtype=float)
            )
        if not np.all(np.isfinite(pheromone_energy_magnitudes)):
            raise RuleError("requested pheromone action is too large")

        deposit = np.maximum(pheromones, 0.0)
        with np.errstate(over="ignore", invalid="ignore"):
            outgoing_cost = push * push + float(np.dot(move, move))
            if deposit.size:
                outgoing_cost += float(np.sum(deposit / self.pheromone_decays))
        if not math.isfinite(outgoing_cost):
            raise RuleError("requested action is too large")

        return _PreparedAction(
            push=push,
            move=move.copy(),
            pheromones=pheromones.copy(),
            outgoing_cost=max(0.0, outgoing_cost),
        )

    def _process_uptake(self, actions: list[_PreparedAction]) -> np.ndarray:
        """Apply simultaneous uptake and return actual concentration per agent/channel."""

        n_agents = len(self.agents)
        n_channels = self.scenario.n_pheromones
        actual = np.zeros((n_agents, n_channels), dtype=float)
        if n_channels == 0:
            return actual

        effective = np.zeros_like(actual)
        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            request = action.uptake_request
            if not np.any(request > 0.0):
                continue
            room = max(0.0, self.scenario.battery_capacity - agent.energy)
            if room <= _EPS:
                continue
            requested_energy = float(np.sum(request / self.pheromone_decays))
            if requested_energy <= _EPS:
                continue
            scale = min(1.0, room / requested_energy)
            effective[i] = request * scale

        # Competition is local to a cell and channel. Allocation is proportional
        # to effective requests, making the result independent of agent loop order.
        groups: dict[tuple[int, int, int], list[int]] = {}
        for i, (agent, request) in enumerate(zip(self.agents, effective)):
            if not np.any(request > 0.0):
                continue
            ix = int(math.floor(float(agent.position[0])))
            iy = int(math.floor(float(agent.position[1])))
            for channel in np.flatnonzero(request > 0.0):
                groups.setdefault((ix, iy, int(channel)), []).append(i)

        for (ix, iy, channel), indices in groups.items():
            available = float(self.pheromone_fields[channel, iy, ix])
            if available <= _EPS:
                continue
            total_request = float(sum(effective[i, channel] for i in indices))
            if total_request <= _EPS:
                continue
            scale = min(1.0, available / total_request)
            removed = 0.0
            for i in indices:
                amount = effective[i, channel] * scale
                actual[i, channel] = amount
                removed += amount
            self.pheromone_fields[channel, iy, ix] = max(0.0, available - removed)

        for i, agent in enumerate(self.agents):
            if n_channels:
                recovered = float(np.sum(actual[i] / self.pheromone_decays))
                agent.energy = min(self.scenario.battery_capacity, agent.energy + recovered)

        return actual

    def _deposit_at(self, position: np.ndarray, amounts: np.ndarray, grid: np.ndarray) -> None:
        if amounts.size == 0 or not np.any(amounts > 0.0):
            return
        ix = int(math.floor(float(position[0])))
        iy = int(math.floor(float(position[1])))
        h, w = self.scenario.grid_shape
        if 0 <= ix < w and 0 <= iy < h and not self.walls[iy, ix]:
            grid[:, iy, ix] += amounts

    # ------------------------------------------------------------------
    # Success

    def _cargo_inside_target(self, cargo_index: int) -> bool:
        xmin, xmax, ymin, ymax = self._box_bounds(
            self.cargo_positions[cargo_index], float(self.cargo_sizes[cargo_index])
        )
        h, w = self.scenario.grid_shape
        if xmin < -_GEOM_EPS or ymin < -_GEOM_EPS or xmax > w + _GEOM_EPS or ymax > h + _GEOM_EPS:
            return False

        ix0 = max(0, int(math.floor(xmin)))
        ix1 = min(w - 1, int(math.floor(np.nextafter(xmax, -math.inf))))
        iy0 = max(0, int(math.floor(ymin)))
        iy1 = min(h - 1, int(math.floor(np.nextafter(ymax, -math.inf))))
        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                if not self._cargo_overlaps_cell(cargo_index, ix, iy):
                    continue
                if not self.target_mask[iy, ix]:
                    return False
        return True

    def _all_cargoes_inside_target(self) -> bool:
        return all(self._cargo_inside_target(i) for i in range(self.scenario.n_cargoes))

    def _agent_inside_target(self, position: np.ndarray) -> bool:
        x, y = float(position[0]), float(position[1])
        h, w = self.scenario.grid_shape
        if x < 0.0 or y < 0.0 or x >= w or y >= h:
            return False
        return bool(self.target_mask[int(math.floor(y)), int(math.floor(x))])

    def _all_agents_inside_target(self) -> bool:
        return all(self._agent_inside_target(agent.position) for agent in self.agents)

    def _is_success(self) -> bool:
        goal = self.scenario.goal
        if goal is Goal.AGENTS:
            return self._all_agents_inside_target()
        if goal is Goal.CARGOES:
            return self._all_cargoes_inside_target()
        if goal is Goal.BOTH:
            return self._all_agents_inside_target() and self._all_cargoes_inside_target()
        raise RuntimeError(f"unsupported scenario goal: {goal!r}")

    # ------------------------------------------------------------------
    # One turn

    def step(self) -> bool:
        """Execute one complete turn and return whether the task is solved."""

        # 1) Recharge.
        for agent in self.agents:
            agent.energy = min(
                self.scenario.battery_capacity,
                agent.energy + self.scenario.recharge,
            )

        # 2) Observe a shared beginning-of-turn world.
        radius = self.scenario.vision_radius
        padded_cells = np.pad(
            self._visible_cell_map(),
            radius,
            mode="constant",
            constant_values=int(Cell.WALL),
        )
        padded_pheromones = (
            np.pad(
                self.pheromone_fields,
                ((0, 0), (1, 1), (1, 1)),
                mode="constant",
            )
            if self.scenario.n_pheromones
            else None
        )

        actions: list[_PreparedAction] = []
        new_memories: list[np.ndarray] = []

        # 3-4) Evaluate and validate all local rules before mutating anything
        # except the turn-start recharge.
        for agent in self.agents:
            observation = self._observation_for(agent, padded_cells, padded_pheromones)
            memory_view = agent.memory.copy()
            memory_view.setflags(write=False)
            returned = self.rule_fn(observation, memory_view)
            if not isinstance(returned, tuple) or len(returned) != 2:
                raise RuleError("rule(...) must return (Action, new_memory)")
            action, new_memory = returned
            actions.append(self._prepare_action(action))

            try:
                memory = np.asarray(new_memory, dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuleError("new_memory must be a finite numeric vector") from exc
            expected = (self.scenario.memory_size,)
            if memory.shape != expected:
                raise RuleError(
                    f"new_memory must have shape {expected}, got {memory.shape}"
                )
            if not np.all(np.isfinite(memory)):
                raise RuleError("new_memory values must be finite")
            new_memories.append(memory.copy())

        # 5) Uptake from old/current cells. Harvested energy remains even if the
        # subsequent outgoing package is unaffordable.
        self._process_uptake(actions)

        # 6) Outgoing action is all-or-nothing. No automatic scaling.
        allowed = np.zeros(len(self.agents), dtype=bool)
        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            if action.outgoing_cost <= agent.energy + _EPS:
                allowed[i] = True
                agent.energy = max(0.0, agent.energy - min(action.outgoing_cost, agent.energy))

        # 7) Push from CURRENT positions. Force direction is automatic.
        forces = np.zeros((self.scenario.n_cargoes, 2), dtype=float)
        for ok, agent, action in zip(allowed, self.agents, actions):
            if not ok or action.push <= 0.0:
                continue
            cargo_index = self._cargo_index_at_point(agent.position)
            if cargo_index is None:
                continue
            direction = self.cargo_positions[cargo_index] - agent.position
            length = float(np.linalg.norm(direction))
            if length > _EPS:
                forces[cargo_index] += action.push * direction / length

        # 8) Cargoes move simultaneously. Mobility falls inversely with area.
        displacements = np.zeros_like(self.cargo_positions)
        for i, size in enumerate(self.cargo_sizes):
            area = float(size) ** 2
            displacements[i] = self.scenario.cargo_mobility * forces[i] / area
        self._move_cargoes_simultaneously(displacements)

        # 9) Agents move. Cargoes and other agents do not block them.
        for ok, agent, action in zip(allowed, self.agents, actions):
            if ok:
                agent.position = self._move_agent_until_wall(agent.position, action.move)

        # 10) Remaining old pheromone decays; valid positive requests are then
        # deposited at final positions and become visible next turn.
        if self.scenario.n_pheromones:
            self.pheromone_fields *= (1.0 - self.pheromone_decays)[:, None, None]
            self.pheromone_fields[:, self.walls] = 0.0
            deposits = np.zeros_like(self.pheromone_fields)
            for ok, agent, action in zip(allowed, self.agents, actions):
                if ok:
                    self._deposit_at(agent.position, action.deposit, deposits)
            self.pheromone_fields += deposits

        # 11) Memory always persists, including after an unaffordable outgoing
        # request, so the rule can learn from failed attempts.
        for agent, memory in zip(self.agents, new_memories):
            agent.memory = memory

        # 12) Success.
        self.turn += 1
        return self._is_success()


def execute(
    scenario: _Scenario,
    rule: RuleFunction,
    *,
    visualize: bool = True,
    delay: float = 0.02,
    draw_every: int = 1,
    max_turns: int | None = None,
) -> Result:
    """Run a scenario until success, interruption, or an optional turn limit.

    ``max_turns`` is not a game rule. Leaving it as ``None`` runs until the
    scenario is solved or the user interrupts execution. A finite value is
    useful for automated tests, benchmarks, and deliberately bounded trials.
    Display settings do not affect game state.
    """

    if draw_every < 1:
        raise ValueError("draw_every must be at least 1")
    if max_turns is not None and max_turns < 1:
        raise ValueError("max_turns must be positive or None")
    if delay < 0:
        raise ValueError("delay must be non-negative")

    engine = Engine(scenario, rule)
    visualizer = None
    if visualize:
        from .visualization import Visualizer

        visualizer = Visualizer(engine)
        visualizer.draw(delay=delay)

    try:
        if engine._is_success():
            return Result(success=True, turns=0)

        while max_turns is None or engine.turn < max_turns:
            success = engine.step()
            if visualizer is not None and (engine.turn % draw_every == 0 or success):
                visualizer.draw(delay=delay)
            if success:
                return Result(success=True, turns=engine.turn)

        return Result(success=False, turns=engine.turn)
    except KeyboardInterrupt:
        # In notebooks this makes "stop this unsuccessful experiment" a normal,
        # traceback-free interaction while preserving the number of turns run.
        return Result(success=False, turns=engine.turn)
    finally:
        if visualizer is not None:
            visualizer.finish()


def run(
    scenario: str,
    rule: RuleFunction,
    *,
    seed: int | None = None,
    visualize: bool = True,
    delay: float = 0.02,
    draw_every: int = 1,
    max_turns: int | None = None,
) -> Result:
    """Run a named scenario with one local rule.

    This is the normal student-facing entry point. ``seed`` controls all
    randomized initial-geometry choices made by the named scenario generator;
    omitting it uses that scenario's built-in deterministic default.
    """

    from .scenarios import load_scenario

    loaded = load_scenario(scenario, seed=seed)
    return execute(
        loaded,
        rule,
        visualize=visualize,
        delay=delay,
        draw_every=draw_every,
        max_turns=max_turns,
    )
