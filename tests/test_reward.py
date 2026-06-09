from types import SimpleNamespace
from dataclasses import replace

import pytest

from config import DEFAULT_ENV_CONFIG
from env.geometry import DirectedParkingSlot
from env.hybrid_astar_reward import OptionalHybridAStarReward
from env.reward import LocalParkingReward


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
        replace(DEFAULT_ENV_CONFIG, distance_d_min=1.0)
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
        replace(DEFAULT_ENV_CONFIG, distance_d_min=2.0)
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
        def plan(self, scene, state, slot):
            return None

    guide = OptionalHybridAStarReward(planner=FailingPlanner())
    state = SimpleNamespace(x_front=0.0, y_front=0.0)
    slot = DirectedParkingSlot(1.0, 0.0, 0.0, 4.0, 2.0)
    assert guide.reset(scene=None, state=state, slot=slot) is False
    hybrid_reward, hybrid_info = guide.step(0.0, 0.0)
    assert hybrid_reward == 0.0
    assert hybrid_info["hybrid_astar_valid"] is False

    reward_model = LocalParkingReward(DEFAULT_ENV_CONFIG)
    reward_model.reset(initial_distance=5.0, initial_overlap=0.0, initial_heading_error=1.0)
    total, components = reward_model.compute(
        front_overlap=0.2,
        distance_to_goal=4.0,
        heading_error=0.8,
        step_count=10,
        hybrid_reward=0.0,
    )
    assert components["hybrid_astar"] == 0.0
    assert components["iou_improvement"] > 0.0
    assert components["distance"] > 0.0
    assert components["heading_improvement"] > 0.0
    assert components["time"] < 0.0
    assert total != 0.0
