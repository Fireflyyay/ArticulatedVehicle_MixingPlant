from dataclasses import replace
import math

import numpy as np
import pytest

from config import DEFAULT_ENV_CONFIG
from env.geometry import DirectedParkingSlot, wrap_to_pi
from env.local_parking_env import LocalParkingEnv
from env.rs_potential import RSPlanResult, RSPotentialOracle, RSPotentialPlanner
from env.vehicle import ArticulatedState
from planning.passenger_hybrid_astar import PlannerResult
from planning.reeds_shepp import generate_reeds_shepp_paths


class RecordingRSPlanner:
    def __init__(self, valid=True, reason="success"):
        self.valid = bool(valid)
        self.reason = str(reason)
        self.calls = 0

    def plan(self, scene, state, slot):
        self.calls += 1
        if not self.valid:
            return RSPlanResult(
                valid=False,
                path=None,
                reason=self.reason,
                total_time_ms=0.1,
            )
        path = np.asarray(
            [
                [state.x_front, state.y_front, state.theta_front],
                [
                    0.5 * (state.x_front + slot.x_goal),
                    0.5 * (state.y_front + slot.y_goal),
                    slot.theta_goal,
                ],
                [slot.x_goal, slot.y_goal, slot.theta_goal],
            ],
            dtype=np.float64,
        )
        return RSPlanResult(
            valid=True,
            path=path,
            reason="success",
            total_length=float(
                math.hypot(state.x_front - slot.x_goal, state.y_front - slot.y_goal)
            ),
            candidate_count=2,
            checked_candidates=1,
            collision_checks=3,
            sample_count=3,
            total_time_ms=0.1,
        )


class StraightHybridPlanner:
    def __init__(self):
        self.calls = 0

    def plan_with_cost(self, scene, state, slot):
        self.calls += 1
        path = np.asarray(
            [
                [state.x_front, state.y_front, state.theta_front],
                [slot.x_goal, slot.y_goal, slot.theta_goal],
            ],
            dtype=np.float32,
        )
        return PlannerResult(
            valid=True,
            path=path,
            cost=float(
                math.hypot(state.x_front - slot.x_goal, state.y_front - slot.y_goal)
            ),
            goal_cost=0.0,
            reason="success",
            expanded_nodes=2,
        )


def _state(x, y=0.0, theta=0.0):
    return ArticulatedState(
        x_front=float(x),
        y_front=float(y),
        theta_front=float(theta),
        theta_rear=float(theta),
    )


def _slot():
    return DirectedParkingSlot(
        x_goal=0.0,
        y_goal=0.0,
        theta_goal=0.0,
        front_body_length=4.45,
        front_body_width=3.016,
    )


def test_rs_disabled_preserves_hybrid_reward_path(synthetic_action_mask):
    hybrid = StraightHybridPlanner()
    env = LocalParkingEnv(
        config=replace(
            DEFAULT_ENV_CONFIG,
            rs_potential_enabled=False,
            use_hybrid_astar=True,
        ),
        action_mask=synthetic_action_mask,
        hybrid_planner=hybrid,
        seed=2,
    )
    env.reset()
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))

    assert info["planner_source"] == "hybrid_astar"
    assert info["rs_latched"] is False
    assert info["reward_components"]["rs_potential"] == 0.0
    assert info["reward_components"]["planner"] == pytest.approx(
        info["reward_components"]["hybrid_astar"]
    )


def test_outside_d_rs_does_not_call_planner():
    planner = RecordingRSPlanner()
    oracle = RSPotentialOracle(planner=planner, d_rs=10.0)
    reward, info = oracle.step(None, _state(12.5), _state(12.0), _slot())

    assert planner.calls == 0
    assert reward == 0.0
    assert info["rs_fail_reason"] == "outside_d_rs"
    assert info["rs_attempt_count"] == 0


def test_collision_free_rs_path_latches_and_planner_is_not_called_again():
    planner = RecordingRSPlanner()
    oracle = RSPotentialOracle(planner=planner, d_rs=10.0)

    _, first_info = oracle.step(None, _state(9.0), _state(8.0), _slot())
    _, second_info = oracle.step(None, _state(8.0), _state(7.0), _slot())

    assert first_info["rs_latched"] is True
    assert first_info["rs_success_count"] == 1
    assert second_info["rs_planner_disabled_after_latch"] is True
    assert planner.calls == 1


def test_rs_latch_suppresses_hybrid_reward_on_same_transition(
    synthetic_action_mask,
):
    hybrid = StraightHybridPlanner()
    rs_planner = RecordingRSPlanner()
    env = LocalParkingEnv(
        config=replace(
            DEFAULT_ENV_CONFIG,
            rs_potential_enabled=True,
            rs_potential_d_rs=10.0,
            use_hybrid_astar=True,
        ),
        action_mask=synthetic_action_mask,
        hybrid_planner=hybrid,
        rs_planner=rs_planner,
        seed=4,
    )
    env.reset()
    axis = np.asarray(
        [math.cos(env.slot.theta_goal), math.sin(env.slot.theta_goal)]
    )
    center = np.asarray(env.slot.center) - 5.0 * axis
    env.state = ArticulatedState(
        x_front=float(center[0]),
        y_front=float(center[1]),
        theta_front=float(env.slot.theta_goal),
        theta_rear=float(env.slot.theta_goal),
    )
    env._update_sensors_and_mask()

    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))

    assert info["rs_latched"] is True
    assert info["planner_source"] == "rs"
    assert info["hybrid_astar_suppressed_by_rs"] is True
    assert info["reward_components"]["hybrid_astar"] == 0.0
    assert info["reward_components"]["planner"] == pytest.approx(info["rs_reward"])
    assert hybrid.calls == 1


def test_colliding_rs_candidates_do_not_latch_or_reward():
    class AlwaysOccupied:
        def _is_rectangle_occupied(self, scene, x, y, theta):
            return True

    planner = RSPotentialPlanner(
        collision_checker=AlwaysOccupied(),
        turning_radius=6.4,
        candidate_limit=2,
        sample_step=0.3,
    )
    result = planner.plan(None, _state(5.0), _slot())
    oracle = RSPotentialOracle(planner=RecordingRSPlanner(False, "collision"))
    reward, info = oracle.step(None, _state(6.0), _state(5.0), _slot())

    assert result.valid is False
    assert result.reason == "collision"
    assert info["rs_latched"] is False
    assert info["rs_fail_reason"] == "collision"
    assert reward == 0.0


def test_rs_failure_has_no_fallback_planner_reward_without_hybrid(
    synthetic_action_mask,
):
    env = LocalParkingEnv(
        config=replace(
            DEFAULT_ENV_CONFIG,
            rs_potential_enabled=True,
            rs_potential_d_rs=10.0,
            use_hybrid_astar=False,
        ),
        action_mask=synthetic_action_mask,
        rs_planner=RecordingRSPlanner(False, "collision"),
        seed=4,
    )
    env.reset()
    axis = np.asarray(
        [math.cos(env.slot.theta_goal), math.sin(env.slot.theta_goal)]
    )
    center = np.asarray(env.slot.center) - 5.0 * axis
    env.state = ArticulatedState(
        x_front=float(center[0]),
        y_front=float(center[1]),
        theta_front=float(env.slot.theta_goal),
        theta_rear=float(env.slot.theta_goal),
    )
    env._update_sensors_and_mask()
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))

    assert info["rs_fail_reason"] == "collision"
    assert info["planner_source"] == "none"
    assert info["reward_components"]["planner"] == 0.0
    assert info["planner_potential_reward"] == 0.0


def test_success_rejects_reverse_and_heading_over_15_degrees(
    synthetic_action_mask,
):
    env = LocalParkingEnv(
        config=replace(
            DEFAULT_ENV_CONFIG,
            rs_potential_enabled=False,
            use_hybrid_astar=False,
            success_overlap=0.0,
        ),
        action_mask=synthetic_action_mask,
        seed=6,
    )
    env.reset()

    for heading_error in (math.pi, math.radians(15.1)):
        env.state = ArticulatedState(
            env.slot.x_goal,
            env.slot.y_goal,
            wrap_to_pi(env.slot.theta_goal + heading_error),
            env.slot.theta_goal,
        )
        env._update_sensors_and_mask()
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert info["success"] is False


def test_rs_reward_is_transition_level_potential_difference():
    planner = RecordingRSPlanner()
    oracle = RSPotentialOracle(
        planner=planner,
        d_rs=10.0,
        gamma=0.98,
        cost_scale=10.0,
        potential_coef=1.0,
        potential_clip=10.0,
        lateral_weight=0.0,
        heading_weight=0.0,
    )
    oracle.step(None, _state(9.0), _state(8.0), _slot())
    previous_cost = float(oracle.rs_last_cost)
    reward, info = oracle.step(None, _state(8.0), _state(7.0), _slot())
    current_cost = float(info["rs_cost"])
    expected = 0.98 * (-current_cost / 10.0) - (-previous_cost / 10.0)

    assert reward == pytest.approx(expected)
    assert reward > 0.0
    assert info["rs_remaining_length"] < previous_cost


def test_rs_candidates_include_extended_scs_family_and_directed_endpoint():
    goal = (-8.0, -8.0, -7.0 * math.pi / 8.0)
    paths = generate_reeds_shepp_paths(
        start=(0.0, 0.0, 0.0),
        goal=goal,
        turning_radius=6.4,
        sample_step=0.3,
    )
    scs_paths = [
        path
        for path in paths
        if path.segment_types in (["S", "L", "S"], ["S", "R", "S"])
    ]

    assert scs_paths
    endpoint = scs_paths[0].poses[-1]
    assert endpoint[:2] == pytest.approx(goal[:2], abs=1e-6)
    assert abs(wrap_to_pi(endpoint[2] - goal[2])) <= 1e-6
