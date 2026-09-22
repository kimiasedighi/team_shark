from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import fa2026 as fa
from fa2026.api import Action, Cell, RuleError
from fa2026.engine import Engine, execute
from fa2026.scenarios import (
    Goal,
    _boundary_walls,
    _random_agent_positions,
    _random_cargo_positions,
    _random_target_mask,
    load_scenario,
)


def _copy_scenario(base, **changes):
    values = dict(
        walls=np.array(base.walls, copy=True),
        agent_positions=np.array(base.agent_positions, copy=True),
        cargo_positions=np.array(base.cargo_positions, copy=True),
        cargo_sizes=np.array(base.cargo_sizes, copy=True),
        target_mask=np.array(base.target_mask, copy=True),
    )
    values.update(changes)
    return replace(base, **values)


def _zeros(obs):
    return (0.0,) * obs.pheromones.shape[0]


def stationary_rule(obs, memory):
    return Action(pheromones=_zeros(obs)), memory


def test_transport_loads_and_initial_battery_is_full():
    scenario = load_scenario("transport")
    engine = Engine(scenario, stationary_rule)
    assert np.allclose(engine.agent_energies, scenario.battery_capacity)
    assert engine.turn == 0


def test_zero_memory_and_zero_pheromone_channels_are_allowed():
    base = load_scenario("transport")
    scenario = _copy_scenario(base, memory_size=0, pheromone_decays=())

    def rule(obs, memory):
        assert memory.shape == (0,)
        assert obs.pheromones.shape == (0, 3, 3)
        return Action(), np.zeros(0)

    engine = Engine(scenario, rule)
    engine.step()
    assert engine.pheromone_fields.shape == (0, *scenario.grid_shape)


def test_observation_structure_and_read_only_arrays():
    scenario = load_scenario("transport")
    engine = Engine(scenario, stationary_rule)
    obs = engine._observation_for(engine.agents[0])
    r = scenario.vision_radius
    assert obs.vision.shape == (2 * r + 1, 2 * r + 1)
    assert obs.pheromones.shape == (scenario.n_pheromones, 3, 3)
    assert len(obs.cell_position) == 2
    assert 0.0 <= obs.cell_position[0] < 1.0
    assert 0.0 <= obs.cell_position[1] < 1.0
    assert isinstance(obs.on_cargo, bool)
    assert obs.vision.flags.writeable is False
    assert obs.pheromones.flags.writeable is False


def test_on_cargo_is_strict_interior():
    scenario = load_scenario("transport")
    engine = Engine(scenario, stationary_rule)
    center = engine.cargo_positions[0]
    half = engine.cargo_sizes[0] / 2
    assert engine._cargo_index_at_point(center + np.array([-0.5 * half, 0.0])) == 0
    assert engine._cargo_index_at_point(center + np.array([half, 0.0])) is None


@pytest.mark.parametrize(
    "action, match",
    [
        (Action(push=-1.0, pheromones=(0.0,)), "push"),
        (Action(push=np.nan, pheromones=(0.0,)), "push"),
        (Action(move=(np.inf, 0.0), pheromones=(0.0,)), "move"),
        (Action(pheromones=(0.0, 0.0)), "pheromones"),
        (Action(pheromones=(np.nan,)), "pheromones"),
    ],
)
def test_invalid_actions_raise_controller_error(action, match):
    scenario = load_scenario("transport")

    def rule(obs, memory):
        return action, memory

    engine = Engine(scenario, rule)
    with pytest.raises(RuleError, match=match):
        engine.step()


def test_omitted_pheromones_mean_no_pheromone_action():
    scenario = load_scenario("transport")

    def rule(obs, memory):
        return Action(move=(0.0, 0.0)), memory

    engine = Engine(scenario, rule)
    engine.step()
    assert np.allclose(engine.pheromone_fields, 0.0)


def test_invalid_memory_raises_controller_error():
    scenario = load_scenario("transport")

    def rule(obs, memory):
        return Action(pheromones=(0.0,)), np.zeros(1)

    with pytest.raises(RuleError, match="new_memory"):
        Engine(scenario, rule).step()


def test_recharge_happens_before_observation_and_caps_at_capacity():
    base = load_scenario("transport")
    scenario = _copy_scenario(base, battery_capacity=10.0, recharge=2.0)
    seen = []

    def rule(obs, memory):
        seen.append(obs.energy)
        return Action(pheromones=(0.0,)), memory

    engine = Engine(scenario, rule)
    engine.agents[0].energy = 3.0
    engine.step()
    assert seen[0] == pytest.approx(5.0)
    assert max(seen[1:]) == pytest.approx(10.0)


def test_uptake_remains_when_outgoing_action_is_unaffordable_and_memory_persists():
    base = load_scenario("transport")
    scenario = _copy_scenario(base, battery_capacity=10.0, recharge=0.01, memory_size=2)
    position = np.array([5.2, 5.2])
    positions = np.array(scenario.agent_positions, copy=True)
    positions[:] = np.array([8.2, 8.2])
    positions[0] = position
    scenario = _copy_scenario(scenario, agent_positions=positions)

    def rule(obs, memory):
        m = memory.copy()
        m[0] += 1.0
        if obs.cell_position[0] == pytest.approx(position[0] % 1):
            return Action(move=(10.0, 0.0), pheromones=(-1.0,)), m
        return Action(pheromones=(0.0,)), m

    engine = Engine(scenario, rule)
    engine.agents[0].energy = 1.0
    engine.pheromone_fields[0, 5, 5] = 1.0  # d=.08 -> 12.5 energy, battery caps uptake
    before = engine.agents[0].position.copy()
    engine.step()
    assert np.allclose(engine.agents[0].position, before)  # move cost 100 rejected
    assert engine.agents[0].energy == pytest.approx(10.0)
    assert engine.agents[0].memory[0] == pytest.approx(1.0)
    assert engine.pheromone_fields[0, 5, 5] < (1.0 - scenario.pheromone_decays[0])


def test_uptake_is_limited_by_empty_battery_capacity_and_excess_remains():
    base = load_scenario("transport")
    scenario = _copy_scenario(base, battery_capacity=10.0, recharge=0.01)
    positions = np.array(scenario.agent_positions, copy=True)
    positions[0] = np.array([5.2, 5.2])
    scenario = _copy_scenario(scenario, agent_positions=positions)

    def rule(obs, memory):
        if obs.cell_position[0] == pytest.approx(0.2) and obs.cell_position[1] == pytest.approx(0.2):
            return Action(pheromones=(-10.0,)), memory
        return Action(pheromones=(0.0,)), memory

    engine = Engine(scenario, rule)
    engine.agents[0].energy = 9.0
    engine.pheromone_fields[0, 5, 5] = 10.0
    engine.step()
    # After recharge there is 0.99 energy room => concentration 0.99*d absorbed.
    absorbed = 0.99 * scenario.pheromone_decays[0]
    assert engine.agents[0].energy == pytest.approx(10.0)
    expected_after_decay = (10.0 - absorbed) * (1.0 - scenario.pheromone_decays[0])
    assert engine.pheromone_fields[0, 5, 5] == pytest.approx(expected_after_decay)


def test_competing_uptake_is_proportional_and_order_independent():
    base = load_scenario("transport")
    scenario = _copy_scenario(base, battery_capacity=20.0, recharge=0.02)
    positions = np.array(scenario.agent_positions, copy=True)
    positions[:] = np.array([5.2, 5.2])
    scenario = _copy_scenario(scenario, agent_positions=positions)

    def rule(obs, memory):
        return Action(pheromones=(-1.0,)), memory

    engine = Engine(scenario, rule)
    for agent in engine.agents:
        agent.energy = 0.0
    engine.pheromone_fields[0, 5, 5] = 1.0
    engine.step()
    # All 80 agents make equal requests, so each gets 1/80 concentration.
    expected_gain = (1.0 / scenario.n_agents) / scenario.pheromone_decays[0]
    assert np.allclose(engine.agent_energies, scenario.recharge + expected_gain)
    assert engine.pheromone_fields[0, 5, 5] == pytest.approx(0.0)


def test_pheromone_uptake_old_cell_decay_then_deposit_new_cell():
    base = load_scenario("transport")
    scenario = _copy_scenario(base, pheromone_decays=(0.25, 0.5), battery_capacity=30.0, recharge=0.03)
    positions = np.array(scenario.agent_positions, copy=True)
    positions[:] = np.array([8.2, 8.2])
    positions[0] = np.array([5.2, 5.2])
    scenario = _copy_scenario(scenario, agent_positions=positions)

    def rule(obs, memory):
        if obs.cell_position[0] == pytest.approx(0.2) and obs.cell_position[1] == pytest.approx(0.2):
            return Action(move=(1.0, 0.0), pheromones=(-0.4, 1.0)), memory
        return Action(pheromones=(0.0, 0.0)), memory

    engine = Engine(scenario, rule)
    engine.agents[0].energy = 20.0
    engine.pheromone_fields[0, 5, 5] = 1.0
    engine.step()
    assert engine.pheromone_fields[0, 5, 5] == pytest.approx((1.0 - 0.4) * 0.75)
    assert engine.pheromone_fields[1, 5, 6] == pytest.approx(1.0)  # new deposit not decayed
    assert engine.pheromone_fields[1, 5, 5] == pytest.approx(0.0)


def test_affordable_but_useless_push_still_costs_energy():
    scenario = load_scenario("transport")

    def rule(obs, memory):
        return Action(push=1.0, pheromones=(0.0,)), memory

    engine = Engine(scenario, rule)
    before_cargo = engine.cargo_positions.copy()
    engine.step()
    assert np.allclose(engine.cargo_positions, before_cargo)
    assert np.allclose(engine.agent_energies, scenario.battery_capacity - 1.0)


def test_unaffordable_outgoing_action_is_rejected_without_outgoing_energy_loss():
    base = load_scenario("transport")
    scenario = _copy_scenario(base, battery_capacity=5.0, recharge=0.005)

    def rule(obs, memory):
        return Action(move=(3.0, 0.0), pheromones=(0.0,)), memory  # cost 9

    engine = Engine(scenario, rule)
    before = engine.agent_positions.copy()
    engine.step()
    assert np.allclose(engine.agent_positions, before)
    assert np.allclose(engine.agent_energies, 5.0)  # started full; recharge caps


def test_scalar_push_points_toward_cargo_center():
    base = load_scenario("transport")
    positions = np.array(base.agent_positions, copy=True)
    center = np.array(base.cargo_positions[0])
    positions[:] = np.array([5.2, 5.2])
    positions[0] = center + np.array([-0.6, 0.0])
    scenario = _copy_scenario(base, agent_positions=positions)

    def rule(obs, memory):
        return Action(push=1.0 if obs.on_cargo else 0.0, pheromones=(0.0,)), memory

    engine = Engine(scenario, rule)
    before = engine.cargo_positions[0].copy()
    engine.step()
    assert engine.cargo_positions[0, 0] > before[0]
    assert engine.cargo_positions[0, 1] == pytest.approx(before[1])


def test_push_from_exact_cargo_center_has_no_force_but_costs_energy():
    base = load_scenario("transport")
    positions = np.array(base.agent_positions, copy=True)
    positions[:] = np.array([5.2, 5.2])
    positions[0] = np.array(base.cargo_positions[0])
    scenario = _copy_scenario(base, agent_positions=positions)

    def rule(obs, memory):
        return Action(push=2.0 if obs.on_cargo else 0.0, pheromones=(0.0,)), memory

    engine = Engine(scenario, rule)
    before = engine.cargo_positions.copy()
    engine.step()
    assert np.allclose(engine.cargo_positions, before)
    assert engine.agents[0].energy == pytest.approx(base.battery_capacity - 4.0)


def test_cargo_mobility_scales_inverse_with_area():
    base = load_scenario("transport")
    cargo_positions = np.array([[10.0, 10.0], [30.0, 22.0]])
    cargo_sizes = np.array([2.0, 4.0])
    positions = np.array(base.agent_positions, copy=True)
    positions[:] = np.array([5.2, 5.2])
    positions[0] = np.array([9.4, 10.0])   # inside 2x2 cargo, pushes +x
    positions[1] = np.array([29.0, 22.0]) # inside 4x4 cargo, pushes +x
    scenario = _copy_scenario(
        base,
        cargo_positions=cargo_positions,
        cargo_sizes=cargo_sizes,
        agent_positions=positions,
        cargo_mobility=0.4,
    )

    def rule(obs, memory):
        return Action(push=1.0 if obs.on_cargo else 0.0, pheromones=(0.0,)), memory

    engine = Engine(scenario, rule)
    before = engine.cargo_positions.copy()
    engine.step()
    dx = engine.cargo_positions[:, 0] - before[:, 0]
    assert dx[0] > 0 and dx[1] > 0
    assert dx[0] / dx[1] == pytest.approx(4.0)


def test_agent_move_cannot_tunnel_through_wall():
    base = load_scenario("transport")
    walls = np.array(base.walls, copy=True)
    walls[1:-1, 10] = True
    positions = np.array(base.agent_positions, copy=True)
    positions[:] = np.array([2.5, 2.5])
    scenario = _copy_scenario(base, walls=walls, agent_positions=positions, battery_capacity=256.0, recharge=0.256)

    def rule(obs, memory):
        return Action(move=(12.0, 0.0), pheromones=(0.0,)), memory

    engine = Engine(scenario, rule)
    engine.step()
    assert np.all(engine.agent_positions[:, 0] < 10.0)
    assert not any(engine._point_in_wall(p) for p in engine.agent_positions)


def test_cargo_stops_at_wall_without_tunneling():
    base = load_scenario("transport")
    walls = np.array(base.walls, copy=True)
    walls[8:14, 18] = True
    cargo_positions = np.array([[14.0, 10.0]])
    cargo_sizes = np.array([2.0])
    positions = np.array(base.agent_positions, copy=True)
    positions[:] = np.array([5.2, 5.2])
    positions[0] = np.array([13.4, 10.0])
    scenario = _copy_scenario(
        base,
        walls=walls,
        cargo_positions=cargo_positions,
        cargo_sizes=cargo_sizes,
        agent_positions=positions,
        battery_capacity=256.0,
        recharge=0.256,
        cargo_mobility=2.0,
    )

    def rule(obs, memory):
        return Action(push=10.0 if obs.on_cargo else 0.0, pheromones=(0.0,)), memory

    engine = Engine(scenario, rule)
    engine.step()
    # Wall cell begins at x=18; cargo half-size is 1, so center cannot pass 17.
    assert engine.cargo_positions[0, 0] <= 17.0 + 1e-8
    assert not engine._box_overlaps_wall_static(engine.cargo_positions[0], 2.0, engine.walls)


def test_two_moving_cargoes_stop_at_same_collision_time():
    base = load_scenario("transport")
    scenario = _copy_scenario(
        base,
        cargo_positions=np.array([[10.0, 10.0], [14.0, 10.0]]),
        cargo_sizes=np.array([2.0, 2.0]),
    )
    engine = Engine(scenario, stationary_rule)
    engine._move_cargoes_simultaneously(np.array([[4.0, 0.0], [-2.0, 0.0]]))
    # Relative closing speed 6, initial gap between faces 2 => collision at t=1/3.
    assert np.allclose(engine.cargo_positions[0], [11.3333333333333, 10.0])
    assert np.allclose(engine.cargo_positions[1], [13.3333333333333, 10.0])


def test_unrelated_cargo_continues_after_other_cargoes_collide():
    base = load_scenario("transport")
    scenario = _copy_scenario(
        base,
        cargo_positions=np.array([[8.0, 8.0], [12.0, 8.0], [25.0, 25.0]]),
        cargo_sizes=np.array([2.0, 2.0, 2.0]),
    )
    engine = Engine(scenario, stationary_rule)
    engine._move_cargoes_simultaneously(
        np.array([[4.0, 0.0], [-2.0, 0.0], [0.0, 3.0]])
    )
    assert np.allclose(engine.cargo_positions[2], [25.0, 28.0])
    assert engine.cargo_positions[1, 0] - engine.cargo_positions[0, 0] == pytest.approx(2.0)


def test_moving_cargo_can_hit_already_stopped_cargo():
    base = load_scenario("transport")
    scenario = _copy_scenario(
        base,
        cargo_positions=np.array([[8.0, 8.0], [12.0, 8.0]]),
        cargo_sizes=np.array([2.0, 2.0]),
    )
    engine = Engine(scenario, stationary_rule)
    engine._move_cargoes_simultaneously(np.array([[6.0, 0.0], [0.0, 0.0]]))
    assert np.allclose(engine.cargo_positions[0], [10.0, 8.0])
    assert np.allclose(engine.cargo_positions[1], [12.0, 8.0])


def test_success_requires_all_cargoes_inside_target_simultaneously():
    base = load_scenario("transport")
    target = np.zeros(base.grid_shape, dtype=bool)
    target[5:15, 28:40] = True
    scenario = _copy_scenario(
        base,
        target_mask=target,
        cargo_positions=np.array([[32.0, 9.0], [37.0, 9.0]]),
        cargo_sizes=np.array([2.0, 3.0]),
    )
    engine = Engine(scenario, stationary_rule)
    assert engine._all_cargoes_inside_target()
    engine.cargo_positions[1] = np.array([20.0, 9.0])
    assert not engine._all_cargoes_inside_target()


def test_multiple_cargoes_are_visible_as_same_cargo_flag():
    base = load_scenario("transport")
    scenario = _copy_scenario(
        base,
        cargo_positions=np.array([[20.0, 18.0], [27.0, 18.0]]),
        cargo_sizes=np.array([2.0, 2.0]),
    )
    engine = Engine(scenario, stationary_rule)
    cells = engine._visible_cell_map()
    assert np.any((cells & int(Cell.CARGO)) != 0)
    assert set(np.unique(cells & int(Cell.CARGO))).issubset({0, int(Cell.CARGO)})


def test_scenario_rejects_initial_overlapping_cargoes():
    base = load_scenario("transport")
    scenario = _copy_scenario(
        base,
        cargo_positions=np.array([[10.0, 10.0], [10.5, 10.0]]),
        cargo_sizes=np.array([2.0, 2.0]),
    )
    with pytest.raises(ValueError, match="overlap"):
        Engine(scenario, stationary_rule)


def test_scenario_rejects_non_rectangular_target():
    base = load_scenario("transport")
    target = np.zeros(base.grid_shape, dtype=bool)
    target[5:8, 5:8] = True
    target[6, 6] = False
    scenario = _copy_scenario(base, target_mask=target)
    with pytest.raises(ValueError, match="rectangle"):
        Engine(scenario, stationary_rule)


def test_repeated_runs_are_deterministic():
    scenario = load_scenario("transport")

    def rule(obs, memory):
        m = memory.copy()
        if m.size:
            m[0] += 1.0
        return Action(move=(0.17, -0.11), pheromones=(0.0,)), m

    a = Engine(scenario, rule)
    b = Engine(scenario, rule)
    for _ in range(25):
        a.step()
        b.step()
    assert np.array_equal(a.agent_positions, b.agent_positions)
    assert np.array_equal(a.agent_energies, b.agent_energies)
    assert np.array_equal(a.cargo_positions, b.cargo_positions)
    assert np.array_equal(a.pheromone_fields, b.pheromone_fields)


def test_execute_rejects_invalid_display_options():
    scenario = load_scenario("transport")
    with pytest.raises(ValueError):
        execute(scenario, stationary_rule, visualize=False, delay=-1)
    with pytest.raises(ValueError):
        execute(scenario, stationary_rule, visualize=False, draw_every=0)
    with pytest.raises(ValueError):
        execute(scenario, stationary_rule, visualize=False, max_turns=0)


def test_push_happens_before_move_so_moving_into_cargo_does_not_push_same_turn():
    base = load_scenario("transport")
    center = np.array(base.cargo_positions[0])
    positions = np.array(base.agent_positions, copy=True)
    positions[:] = np.array([5.2, 5.2])
    positions[0] = center + np.array([-1.4, 0.0])  # outside 2x2 cargo
    scenario = _copy_scenario(base, agent_positions=positions)

    def rule(obs, memory):
        return Action(push=1.0, move=(0.6, 0.0), pheromones=(0.0,)), memory

    engine = Engine(scenario, rule)
    before = engine.cargo_positions.copy()
    engine.step()
    assert np.allclose(engine.cargo_positions, before)
    assert engine._cargo_index_at_point(engine.agents[0].position) == 0


def test_agent_can_push_then_follow_cargo_in_same_turn():
    base = load_scenario("transport")
    center = np.array(base.cargo_positions[0])
    positions = np.array(base.agent_positions, copy=True)
    positions[:] = np.array([5.2, 5.2])
    positions[0] = center + np.array([-0.6, 0.0])
    scenario = _copy_scenario(base, agent_positions=positions)

    def rule(obs, memory):
        if obs.on_cargo:
            return Action(push=1.0, move=(0.3, 0.0), pheromones=(0.0,)), memory
        return Action(pheromones=(0.0,)), memory

    engine = Engine(scenario, rule)
    old_agent = engine.agents[0].position.copy()
    old_cargo = engine.cargo_positions[0].copy()
    engine.step()
    assert engine.cargo_positions[0, 0] > old_cargo[0]
    assert engine.agents[0].position[0] == pytest.approx(old_agent[0] + 0.3)


def test_touching_cargoes_may_move_apart():
    base = load_scenario("transport")
    scenario = _copy_scenario(
        base,
        cargo_positions=np.array([[10.0, 10.0], [12.0, 10.0]]),
        cargo_sizes=np.array([2.0, 2.0]),
    )
    engine = Engine(scenario, stationary_rule)
    engine._move_cargoes_simultaneously(np.array([[-1.0, 0.0], [1.0, 0.0]]))
    assert np.allclose(engine.cargo_positions, [[9.0, 10.0], [13.0, 10.0]])


def test_cargo_touching_wall_can_move_away():
    base = load_scenario("transport")
    walls = np.array(base.walls, copy=True)
    # Boundary wall occupies x in [0,1]; a 2x2 cargo centered at x=2 touches it.
    scenario = _copy_scenario(
        base,
        walls=walls,
        cargo_positions=np.array([[2.0, 10.0]]),
        cargo_sizes=np.array([2.0]),
    )
    engine = Engine(scenario, stationary_rule)
    engine._move_cargoes_simultaneously(np.array([[1.0, 0.0]]))
    assert np.allclose(engine.cargo_positions[0], [3.0, 10.0])


def test_extreme_finite_negative_pheromone_request_is_rejected_cleanly():
    scenario = load_scenario("transport")

    def rule(obs, memory):
        return Action(pheromones=(-1e308,)), memory

    with pytest.raises(RuleError, match="too large"):
        Engine(scenario, rule).step()


def test_loaded_scenario_arrays_are_read_only():
    scenario = load_scenario("communicate")
    assert not scenario.walls.flags.writeable
    assert not scenario.agent_positions.flags.writeable
    assert not scenario.cargo_positions.flags.writeable
    assert not scenario.cargo_sizes.flags.writeable
    assert not scenario.target_mask.flags.writeable


def test_outgoing_energy_cost_adds_push_move_and_deposition():
    base = load_scenario("transport")
    positions = np.array(base.agent_positions, copy=True)
    positions[:] = np.array([5.2, 5.2])
    scenario = _copy_scenario(base, agent_positions=positions)
    decay = scenario.pheromone_decays[0]
    deposit = 0.5 * decay  # exactly 0.5 energy

    def rule(obs, memory):
        return Action(push=1.0, move=(0.3, 0.4), pheromones=(deposit,)), memory

    engine = Engine(scenario, rule)
    engine.step()
    expected_cost = 1.0**2 + (0.3**2 + 0.4**2) + 0.5
    assert np.allclose(engine.agent_energies, scenario.battery_capacity - expected_cost)


def test_public_interface_is_one_namespace_and_run_works():
    assert fa.Action is Action
    assert fa.Cell is Cell
    assert fa.RuleError is RuleError
    assert callable(fa.run)
    # Lower-level helpers are intentionally not part of the normal package API.
    assert "execute" not in fa.__all__
    assert "load_scenario" not in fa.__all__

    def rule(obs, memory):
        return fa.Action(pheromones=(0.0,)), memory

    result = fa.run("transport", rule, visualize=False, max_turns=2)
    assert result == fa.Result(success=False, turns=2)


def test_default_seed_matches_explicit_default_seed():
    a = load_scenario("transport")
    b = load_scenario("transport", seed=3102)
    assert a.seed == b.seed == 3102
    assert np.array_equal(a.agent_positions, b.agent_positions)


def test_seed_override_is_reproducible_and_changes_initial_geometry():
    a = load_scenario("transport", seed=1234)
    b = load_scenario("transport", seed=1234)
    c = load_scenario("transport", seed=1235)
    assert np.array_equal(a.agent_positions, b.agent_positions)
    assert not np.array_equal(a.agent_positions, c.agent_positions)
    # Seed changes randomized geometry, not the fixed scenario mechanics.
    assert np.array_equal(a.walls, c.walls)
    assert not np.array_equal(a.cargo_positions, c.cargo_positions)
    assert not np.array_equal(a.target_mask, c.target_mask)
    assert a.vision_radius == c.vision_radius
    assert a.battery_capacity == c.battery_capacity
    assert a.recharge == c.recharge
    assert a.cargo_mobility == c.cargo_mobility
    assert a.pheromone_decays == c.pheromone_decays


def test_agent_cannot_escape_through_boundary_wall_after_repeated_moves():
    scenario = load_scenario("gather", seed=1)

    def rule(obs, memory):
        return Action(move=(0.55, 0.0), pheromones=(0.0,)), memory

    engine = Engine(scenario, rule)
    for _ in range(200):
        engine.step()
    h, w = scenario.grid_shape
    assert np.all(engine.agent_positions[:, 0] >= 1.0)
    assert np.all(engine.agent_positions[:, 0] < w - 1.0)
    assert np.all(engine.agent_positions[:, 1] >= 1.0)
    assert np.all(engine.agent_positions[:, 1] < h - 1.0)
    assert not any(engine._point_in_wall(p) for p in engine.agent_positions)


def _positions_in_target(scenario):
    ys, xs = np.nonzero(scenario.target_mask)
    assert len(xs) > 0
    point = np.array([float(xs[0]) + 0.5, float(ys[0]) + 0.5])
    return np.repeat(point[None, :], scenario.n_agents, axis=0)


def test_goal_agents_requires_all_agents_inside_target():
    base = load_scenario("transport")
    scenario = _copy_scenario(
        base,
        goal=Goal.AGENTS,
        cargo_positions=np.empty((0, 2), dtype=float),
        cargo_sizes=np.empty((0,), dtype=float),
        agent_positions=_positions_in_target(base),
    )
    engine = Engine(scenario, stationary_rule)
    assert engine._is_success()
    assert engine._all_agents_inside_target()

    engine.agents[0].position = np.array([2.5, 2.5])
    assert not engine._is_success()


def test_goal_cargoes_ignores_agents_and_requires_all_cargoes():
    base = load_scenario("transport")
    ys, xs = np.nonzero(base.target_mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cargo_positions = np.array(
        [
            [x0 + 2.0, y0 + 2.0],
            [x1 - 2.0, y1 - 2.0],
        ],
        dtype=float,
    )
    scenario = _copy_scenario(
        base,
        goal=Goal.CARGOES,
        cargo_positions=cargo_positions,
        cargo_sizes=np.array([2.0, 2.0]),
    )
    engine = Engine(scenario, stationary_rule)
    assert engine._is_success()

    # Agent locations are irrelevant for this goal.
    engine.agents[0].position = np.array([2.5, 2.5])
    assert engine._is_success()

    engine.cargo_positions[1] = np.array([10.0, 10.0])
    assert not engine._is_success()


def test_goal_both_requires_agents_and_cargoes():
    base = load_scenario("transport")
    ys, xs = np.nonzero(base.target_mask)
    cargo = np.array([[float(xs.min()) + 2.0, float(ys.min()) + 2.0]])
    scenario = _copy_scenario(
        base,
        goal=Goal.BOTH,
        cargo_positions=cargo,
        cargo_sizes=np.array([2.0]),
        agent_positions=_positions_in_target(base),
    )
    engine = Engine(scenario, stationary_rule)
    assert engine._is_success()

    engine.agents[0].position = np.array([2.5, 2.5])
    assert not engine._is_success()
    engine.agents[0].position = _positions_in_target(base)[0].copy()
    engine.cargo_positions[0] = np.array([10.0, 10.0])
    assert not engine._is_success()


def test_zero_cargoes_are_allowed_for_agent_goal():
    base = load_scenario("transport")
    scenario = _copy_scenario(
        base,
        goal=Goal.AGENTS,
        cargo_positions=np.empty((0, 2), dtype=float),
        cargo_sizes=np.empty((0,), dtype=float),
    )
    engine = Engine(scenario, stationary_rule)
    assert scenario.n_cargoes == 0
    assert engine.cargo_positions.shape == (0, 2)
    assert engine.cargo_sizes.shape == (0,)
    assert not np.any(engine._visible_cell_map() & int(Cell.CARGO))


def test_zero_cargoes_are_rejected_for_cargo_based_goals():
    base = load_scenario("transport")
    for goal in (Goal.CARGOES, Goal.BOTH):
        scenario = _copy_scenario(
            base,
            goal=goal,
            cargo_positions=np.empty((0, 2), dtype=float),
            cargo_sizes=np.empty((0,), dtype=float),
        )
        with pytest.raises(ValueError, match="require at least one cargo"):
            Engine(scenario, stationary_rule)


def _randomized_geometry(seed):
    rng = np.random.default_rng(seed)
    walls = _boundary_walls(30, 42)
    target = _random_target_mask(
        walls,
        rng,
        size=(6, 7),
        at_boundary=True,
    )
    cargo_sizes = np.array([2.0, 3.0], dtype=float)
    cargo_positions = _random_cargo_positions(
        cargo_sizes,
        walls,
        rng,
        forbidden_mask=target,
    )
    agent_positions = _random_agent_positions(
        24,
        walls,
        cargo_positions,
        cargo_sizes,
        rng=rng,
        excluded_mask=target,
    )
    return walls, target, cargo_positions, cargo_sizes, agent_positions


def test_randomized_geometry_is_reproducible_for_same_seed():
    a = _randomized_geometry(12345)
    b = _randomized_geometry(12345)
    for aa, bb in zip(a, b):
        assert np.array_equal(aa, bb)


def test_randomized_geometry_changes_with_seed_and_is_legal():
    base = load_scenario("transport")
    a = _randomized_geometry(12345)
    b = _randomized_geometry(12346)
    assert not np.array_equal(a[1], b[1]) or not np.array_equal(a[2], b[2])
    walls, target, cargo_positions, cargo_sizes, agent_positions = a

    scenario = _copy_scenario(
        base,
        walls=walls,
        target_mask=target,
        cargo_positions=cargo_positions,
        cargo_sizes=cargo_sizes,
        agent_positions=agent_positions,
        goal=Goal.CARGOES,
    )
    # Engine validation is the authoritative legality check.
    Engine(scenario, stationary_rule)


def test_execute_accepts_none_max_turns_and_initial_success_returns_immediately():
    base = load_scenario("transport")
    scenario = _copy_scenario(
        base,
        goal=Goal.AGENTS,
        cargo_positions=np.empty((0, 2), dtype=float),
        cargo_sizes=np.empty((0,), dtype=float),
        agent_positions=_positions_in_target(base),
    )
    result = execute(scenario, stationary_rule, visualize=False, max_turns=None)
    assert result == fa.Result(success=True, turns=0)


def _target_visible_from_position(scenario, position):
    ix, iy = int(np.floor(position[0])), int(np.floor(position[1]))
    ys, xs = np.nonzero(scenario.target_mask)
    return bool(
        np.any(
            (np.abs(xs - ix) <= scenario.vision_radius)
            & (np.abs(ys - iy) <= scenario.vision_radius)
        )
    )


def test_teaching_scenarios_have_intended_goals_and_resources():
    gather = load_scenario("gather")
    transport = load_scenario("transport")
    communicate = load_scenario("communicate")

    assert gather.goal is Goal.AGENTS
    assert gather.n_cargoes == 0
    assert gather.n_pheromones == 1
    assert gather.memory_size == 8

    for scenario in (transport, communicate):
        assert scenario.goal is Goal.CARGOES
        assert scenario.n_cargoes == 1
        assert scenario.cargo_sizes[0] == pytest.approx(2.0)
        assert scenario.cargo_mobility == pytest.approx(0.05)
        assert scenario.n_pheromones == 1
        assert scenario.memory_size == 8


def test_teaching_scenario_seed_reproduces_all_randomized_geometry():
    for name in ("gather", "transport", "communicate"):
        a = load_scenario(name, seed=4321)
        b = load_scenario(name, seed=4321)
        c = load_scenario(name, seed=4322)

        assert np.array_equal(a.target_mask, b.target_mask)
        assert np.array_equal(a.cargo_positions, b.cargo_positions)
        assert np.array_equal(a.agent_positions, b.agent_positions)

        assert not np.array_equal(a.target_mask, c.target_mask)
        assert not np.array_equal(a.agent_positions, c.agent_positions)
        if a.n_cargoes:
            assert not np.array_equal(a.cargo_positions, c.cargo_positions)


def test_gather_starts_centrally_and_target_is_initially_hidden():
    scenario = load_scenario("gather", seed=7)
    assert np.all((scenario.agent_positions[:, 0] >= 22.0) & (scenario.agent_positions[:, 0] <= 42.0))
    assert np.all((scenario.agent_positions[:, 1] >= 17.0) & (scenario.agent_positions[:, 1] <= 31.0))
    assert not any(
        _target_visible_from_position(scenario, p) for p in scenario.agent_positions
    )


def test_transport_target_is_visible_from_cargo_for_many_seeds():
    # Scenario 2 is intentionally about finding/recruiting around the cargo,
    # not about communicating a hidden target direction.
    for seed in range(1, 101):
        scenario = load_scenario("transport", seed=seed)
        assert _target_visible_from_position(scenario, scenario.cargo_positions[0])


def test_communicate_target_is_hidden_from_cargo_and_initial_agents():
    # Scenario 3 deliberately removes the direct target cue at the cargo.
    for seed in range(1, 51):
        scenario = load_scenario("communicate", seed=seed)
        assert not _target_visible_from_position(scenario, scenario.cargo_positions[0])
        assert not any(
            _target_visible_from_position(scenario, p) for p in scenario.agent_positions
        )


def test_info_lists_public_scenarios(capsys):
    assert fa.info() is None
    out = capsys.readouterr().out
    for name in ("gather", "transport", "communicate"):
        assert name in out


def test_info_reports_rules_without_realized_coordinates(capsys):
    assert fa.info("transport") is None
    out = capsys.readouterr().out
    assert "Scenario: transport" in out
    assert "Cargo mobility mu: 0.05" in out
    assert "Vision radius: 16" in out
    assert "Battery capacity: 30" in out
    assert "Recharge per turn: 0.85" in out
    assert "Channel 0 decay fraction per turn: 0.08" in out
    assert "cost p^2" in out
    assert "cost dx^2 + dy^2" in out
    assert "cost q / decay" in out
    assert "Randomized by seed" in out
    assert "cargo position" in out
    # Public info describes the distribution/specification, not one seed's coordinates.
    scenario = load_scenario("transport")
    coordinate_strings = [f"{float(v):.6f}" for v in scenario.cargo_positions.ravel()]
    assert not any(value in out for value in coordinate_strings)


def test_info_unknown_scenario_matches_run_lookup_behavior():
    with pytest.raises(KeyError, match="available scenarios"):
        fa.info("does-not-exist")

