import math
from types import SimpleNamespace

import numpy as np
import pytest

from config import DEFAULT_ENV_CONFIG
from env.geometry import DirectedParkingSlot, wrap_to_pi
from env.hybrid_astar_reward import OptionalHybridAStarReward
from env.reward import LocalParkingReward


# ---------------------------------------------------------------------------
#  Existing tests (updated for PBRS interface)
# ---------------------------------------------------------------------------

def test_overlap_and_heading_reward_only_improve_history_best():
    reward_model = LocalParkingReward(DEFAULT_ENV_CONFIG)
    reward_model.reset(initial_distance=5.0, initial_overlap=0.2, initial_heading_error=1.0)
    _, improved = reward_model.compute(
        front_overlap=0.4,
        distance_to_goal=4.0,
        heading_error=0.8,
        step_count=1,
    )
    _, regressed = reward_model.compute(
        front_overlap=0.3,
        distance_to_goal=4.2,
        heading_error=0.9,
        step_count=2,
    )
    assert improved["iou_improvement"] == 0.2
    assert improved["heading_improvement"] > 0.0
    assert regressed["iou_improvement"] == 0.0
    assert regressed["heading_improvement"] == 0.0


def test_distance_reward_matches_hope_initial_distance_formula():
    reward_model = LocalParkingReward(
        replace_config(DEFAULT_ENV_CONFIG, distance_d_min=1.0)
    )
    reward_model.reset(
        initial_distance=10.0,
        initial_overlap=0.0,
        initial_heading_error=0.0,
    )

    _, closer = reward_model.compute(
        front_overlap=0.0,
        distance_to_goal=6.0,
        heading_error=0.0,
        step_count=1,
    )
    _, unchanged = reward_model.compute(
        front_overlap=0.0,
        distance_to_goal=10.0,
        heading_error=0.0,
        step_count=2,
    )
    _, farther = reward_model.compute(
        front_overlap=0.0,
        distance_to_goal=12.0,
        heading_error=0.0,
        step_count=3,
    )

    assert closer["distance"] == pytest.approx(0.4)
    assert unchanged["distance"] == pytest.approx(0.0)
    assert farther["distance"] == pytest.approx(-0.2)


def test_distance_reward_uses_configured_d_min():
    reward_model = LocalParkingReward(
        replace_config(DEFAULT_ENV_CONFIG, distance_d_min=2.0)
    )
    reward_model.reset(
        initial_distance=0.5,
        initial_overlap=0.0,
        initial_heading_error=0.0,
    )
    _, components = reward_model.compute(
        front_overlap=0.0,
        distance_to_goal=0.25,
        heading_error=0.0,
        step_count=1,
    )
    assert components["distance"] == pytest.approx(0.125)


def test_hybrid_failure_keeps_base_reward_terms():
    class FailingPlanner:
        def plan_with_cost(self, scene, state, slot):
            return _no_path_result()

        def plan(self, scene, state, slot):
            return None

    guide = OptionalHybridAStarReward(planner=FailingPlanner())
    state = SimpleNamespace(x_front=0.0, y_front=0.0, theta_front=0.0)
    slot = DirectedParkingSlot(1.0, 0.0, 0.0, 4.0, 2.0)
    assert guide.reset(scene=None, state=state, slot=slot) is False
    assert guide.valid is False
    assert guide._fallback_used is True

    # step should return a finite reward (not 0.0 in the old "inert" sense,
    # though with fallback it may be small; the key is that it is finite
    # and base reward terms still work)
    hybrid_reward, hybrid_info = guide.step(0.0, 0.0, 0.0)
    assert isinstance(hybrid_reward, float)
    assert abs(hybrid_reward) < 2.0
    assert hybrid_info["planner_valid"] is False
    assert hybrid_info["planner_fallback_used"] is True

    reward_model = LocalParkingReward(DEFAULT_ENV_CONFIG)
    reward_model.reset(initial_distance=5.0, initial_overlap=0.0, initial_heading_error=1.0)
    total, components = reward_model.compute(
        front_overlap=0.2,
        distance_to_goal=4.0,
        heading_error=0.8,
        step_count=10,
        hybrid_reward=hybrid_reward,
    )
    assert components["iou_improvement"] > 0.0
    assert components["distance"] > 0.0
    assert components["heading_improvement"] > 0.0
    assert components["time"] < 0.0
    assert total != 0.0


# ---------------------------------------------------------------------------
#  New PBRS tests
# ---------------------------------------------------------------------------

def test_unidirectional_heading_reverse_is_error():
    """theta_goal and theta_goal+pi are NOT equivalent; reverse gives ~pi error."""
    slot = DirectedParkingSlot(x_goal=0.0, y_goal=0.0, theta_goal=0.0,
                               front_body_length=4.0, front_body_width=2.0)

    class FailPlanner:
        def plan_with_cost(self, scene, state, slot):
            return _no_path_result()
        def plan(self, scene, state, slot):
            return None

    guide = OptionalHybridAStarReward(planner=FailPlanner())
    state = SimpleNamespace(x_front=0.0, y_front=0.0, theta_front=math.pi)
    guide.reset(scene=None, state=state, slot=slot)

    # heading error should be ~pi, not ~0
    _, info = guide.step(0.0, 0.0, math.pi)
    heading_err = info["goal_heading_error_single_direction"]
    assert heading_err == pytest.approx(math.pi, abs=1e-4)
    assert heading_err > math.pi - 1e-4


def test_goal_check_fails_reverse_heading():
    """Planner must reject a state that is at the correct position but reversed heading."""
    from planning.passenger_hybrid_astar import PassengerHybridAStar

    # Single-step simple scene where the start is right at the goal but reversed
    class MiniScene:
        world_bounds = (-40.0, -40.0, 40.0, 40.0)
        resolution = 1.0
        def is_occupied_world(self, x, y):
            return False

    scene = MiniScene()
    slot = DirectedParkingSlot(x_goal=0.0, y_goal=0.0, theta_goal=0.0,
                               front_body_length=4.0, front_body_width=2.0)

    def _make_state(theta_f):
        return SimpleNamespace(x_front=0.0, y_front=0.0, theta_front=theta_f)

    planner = PassengerHybridAStar(
        max_expansions=100, goal_pos_tol=0.8, goal_heading_tol_deg=15.0,
        # minimal footprint samples to keep test fast
        front_half_length=0.1, front_half_width=0.1, footprint_samples=1,
    )

    # Correct heading — should succeed quickly
    r1 = planner.plan_with_cost(scene, _make_state(0.0), slot)
    assert r1.valid, "planner should reach goal with correct heading"
    assert r1.reason == "success"

    # Reversed heading — must NOT succeed (unidirectional)
    r2 = planner.plan_with_cost(scene, _make_state(math.pi), slot)
    assert not r2.valid, "planner must reject reversed heading (unidirectional goal)"
    assert "no_path" in r2.reason or "max" in r2.reason


def test_planner_fallback_returns_finite_j():
    class FailPlanner:
        def plan_with_cost(self, scene, state, slot):
            return _no_path_result()
        def plan(self, scene, state, slot):
            return None

    guide = OptionalHybridAStarReward(planner=FailPlanner(),
                                       failure_bias=3.0, max_cost=80.0)
    state = SimpleNamespace(x_front=5.0, y_front=0.0, theta_front=0.3)
    slot = DirectedParkingSlot(0.0, 0.0, 0.0, 4.0, 2.0)
    guide.reset(scene=None, state=state, slot=slot)

    assert guide.valid is False
    assert guide._fallback_used is True

    # step should compute a finite J (not 0, not inf)
    _, info = guide.step(3.0, 1.0, 0.1)
    J = info["planner_cost"]
    assert 0.0 < J <= 80.0, "fallback J should be finite positive, got {}".format(J)

    # Closer to goal should yield lower J
    _, info2 = guide.step(0.5, 0.0, 0.05)
    J2 = info2["planner_cost"]
    assert J2 < J, "J should decrease when closer to target (J={}, J2={})".format(J, J2)


def test_j_state_decreases_approaching_on_path():
    """J_state must decrease when moving along the planner path toward the goal."""
    # Build a simple straight path using a mini-planner
    class MiniPlanner:
        def plan_with_cost(self, scene, state, slot):
            import numpy as np
            # Build a straight path from start toward goal
            heading = math.atan2(
                slot.y_goal - state.y_front,
                slot.x_goal - state.x_front,
            )
            dx = slot.x_goal - state.x_front
            dy = slot.y_goal - state.y_front
            dist = math.hypot(dx, dy)
            n = max(2, int(dist / 0.5) + 1)
            xs = np.linspace(state.x_front, slot.x_goal, n)
            ys = np.linspace(state.y_front, slot.y_goal, n)
            thetas = np.full(n, heading)
            path = np.stack([xs, ys, thetas], axis=1).astype(np.float32)
            from planning.passenger_hybrid_astar import PlannerResult
            return PlannerResult(
                valid=True, path=path,
                cost=float(dist), goal_cost=0.0,
                reason="success", expanded_nodes=10,
            )
        def plan(self, scene, state, slot):
            return self.plan_with_cost(scene, state, slot).path

    guide = OptionalHybridAStarReward(
        planner=MiniPlanner(),
        cost_scale=25.0,
        lateral_residual_weight=0.5,
        goal_heading_weight=1.0,
    )
    state = SimpleNamespace(x_front=8.0, y_front=0.0, theta_front=0.0)
    slot = DirectedParkingSlot(0.0, 0.0, 0.0, 4.0, 2.0)
    guide.reset(scene=None, state=state, slot=slot)
    assert guide.valid is True

    J_start = guide._previous_J
    # Move forward 1.0 towards goal at heading 0 (reverse = π)
    _, info = guide.step(7.0, 0.0, 0.0)
    J_mid = info["planner_cost"]
    assert J_mid < J_start, "J should decrease when moving toward goal"


def test_j_state_increases_with_lateral_deviation():
    """J_state must increase when moving perpendicularly away from the path."""
    class MiniPlanner:
        def plan_with_cost(self, scene, state, slot):
            import numpy as np
            heading = 0.0
            path = np.array([
                [8.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ], dtype=np.float32)
            from planning.passenger_hybrid_astar import PlannerResult
            return PlannerResult(
                valid=True, path=path,
                cost=8.0, goal_cost=0.0,
                reason="success", expanded_nodes=3,
            )
        def plan(self, scene, state, slot):
            return self.plan_with_cost(scene, state, slot).path

    guide = OptionalHybridAStarReward(
        planner=MiniPlanner(),
        cost_scale=25.0,
        lateral_residual_weight=0.5,
        goal_heading_weight=1.0,
    )
    state = SimpleNamespace(x_front=4.0, y_front=0.0, theta_front=0.0)
    slot = DirectedParkingSlot(0.0, 0.0, 0.0, 4.0, 2.0)
    guide.reset(scene=None, state=state, slot=slot)

    # Step laterally away from the path (y=2.0 is far off a y=0 path)
    _, info = guide.step(4.0, 2.0, 0.0)
    J_lateral = info["planner_cost"]

    # Moving another step farther from path should increase J
    _, info2 = guide.step(4.0, 4.0, 0.0)
    J_lateral2 = info2["planner_cost"]
    assert J_lateral2 > J_lateral, \
        "J should increase when moving laterally away from path"


def test_potential_reward_positive_when_j_decreases():
    """PBRS reward > 0 when J(s') < J(s)."""
    class MiniPlanner:
        def plan_with_cost(self, scene, state, slot):
            import numpy as np
            path = np.array([
                [8.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ], dtype=np.float32)
            from planning.passenger_hybrid_astar import PlannerResult
            return PlannerResult(
                valid=True, path=path,
                cost=8.0, goal_cost=0.0,
                reason="success", expanded_nodes=3,
            )
        def plan(self, scene, state, slot):
            return self.plan_with_cost(scene, state, slot).path

    # Use a coef that is clearly positive so potential reward stands out
    guide = OptionalHybridAStarReward(
        planner=MiniPlanner(),
        gamma=0.98, cost_scale=25.0, potential_coef=1.0, potential_clip=10.0,
        lateral_residual_weight=0.0, goal_heading_weight=0.0,
    )
    state = SimpleNamespace(x_front=8.0, y_front=0.0, theta_front=0.0)
    slot = DirectedParkingSlot(0.0, 0.0, 0.0, 4.0, 2.0)
    guide.reset(scene=None, state=state, slot=slot)

    # Move toward goal along path — J should decrease, reward > 0
    reward, info = guide.step(7.0, 0.0, 0.0)
    assert info["planner_cost"] < guide._previous_J + 1e-9  # not strictly needed but sanity
    assert reward > 0.0, "PBRS reward must be > 0 when approaching goal (got {})".format(reward)


def test_potential_reward_negative_when_j_increases():
    """PBRS reward < 0 when J(s') > J(s)."""
    class MiniPlanner:
        def plan_with_cost(self, scene, state, slot):
            import numpy as np
            path = np.array([
                [4.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ], dtype=np.float32)
            from planning.passenger_hybrid_astar import PlannerResult
            return PlannerResult(
                valid=True, path=path,
                cost=4.0, goal_cost=0.0,
                reason="success", expanded_nodes=3,
            )
        def plan(self, scene, state, slot):
            return self.plan_with_cost(scene, state, slot).path

    guide = OptionalHybridAStarReward(
        planner=MiniPlanner(),
        gamma=0.98, cost_scale=25.0, potential_coef=1.0, potential_clip=10.0,
        lateral_residual_weight=0.0, goal_heading_weight=0.0,
    )
    state = SimpleNamespace(x_front=2.0, y_front=0.0, theta_front=0.0)
    slot = DirectedParkingSlot(0.0, 0.0, 0.0, 4.0, 2.0)
    guide.reset(scene=None, state=state, slot=slot)

    # Move away from goal — J should increase, reward < 0
    reward, info = guide.step(4.0, 0.0, 0.0)
    assert reward < 0.0, "PBRS reward must be < 0 when retreating (got {})".format(reward)


def test_legacy_xy_reward_inactive_in_default_config():
    """Default legacy weights are 0.0; step_legacy returns 0 reward."""
    import numpy as np
    class MiniPlanner:
        def plan_with_cost(self, scene, state, slot):
            path = np.array([
                [4.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ], dtype=np.float32)
            from planning.passenger_hybrid_astar import PlannerResult
            return PlannerResult(
                valid=True, path=path,
                cost=4.0, goal_cost=0.0,
                reason="success", expanded_nodes=2,
            )
        def plan(self, scene, state, slot):
            return self.plan_with_cost(scene, state, slot).path

    guide = OptionalHybridAStarReward(planner=MiniPlanner())
    state = SimpleNamespace(x_front=4.0, y_front=0.0, theta_front=0.0)
    slot = DirectedParkingSlot(0.0, 0.0, 0.0, 4.0, 2.0)
    guide.reset(scene=None, state=state, slot=slot)

    legacy_reward, legacy_info = guide.step_legacy(2.0, 0.0)
    assert legacy_reward == pytest.approx(0.0, abs=1e-6), \
        "legacy XY reward should be 0 when weights are 0"
    assert legacy_info["hybrid_astar_valid"] is True
    assert legacy_info["hybrid_astar_progress"] > 0.0


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def replace_config(base, **overrides):
    from dataclasses import replace
    return replace(base, **overrides)


def _no_path_result():
    from planning.passenger_hybrid_astar import PlannerResult
    return PlannerResult(valid=False, path=None, cost=0.0, goal_cost=0.0,
                         reason="no_path", expanded_nodes=5)
