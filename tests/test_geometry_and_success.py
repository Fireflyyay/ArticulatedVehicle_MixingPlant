from dataclasses import replace
import math

import numpy as np

from config import DEFAULT_ENV_CONFIG
from env.geometry import DirectedParkingSlot, oriented_box, overlap_ratio
from env.local_parking_env import LocalParkingEnv
from env.vehicle import ArticulatedState


def test_directed_parking_slot_transform():
    slot = DirectedParkingSlot(1.0, 2.0, math.pi / 2.0, 4.0, 2.0)
    error = slot.position_error_in_slot_frame(1.0, 3.0)
    assert np.allclose(error, [1.0, 0.0], atol=1e-6)
    corners = slot.target_corners_in_ego_frame(1.0, 2.0, math.pi / 2.0)
    assert corners.shape == (4, 2)
    assert np.allclose(np.sort(np.abs(corners[:, 0])), [2.0, 2.0, 2.0, 2.0])


def test_front_body_overlap_geometry():
    target = oriented_box((0.0, 0.0), 0.0, 4.0, 2.0)
    same = oriented_box((0.0, 0.0), 0.0, 4.0, 2.0)
    half = oriented_box((2.0, 0.0), 0.0, 4.0, 2.0)
    disjoint = oriented_box((10.0, 0.0), 0.0, 4.0, 2.0)
    assert np.isclose(overlap_ratio(same, target), 1.0)
    assert np.isclose(overlap_ratio(half, target), 0.5)
    assert np.isclose(overlap_ratio(disjoint, target), 0.0)


def test_success_uses_front_overlap_and_wrapped_heading(synthetic_action_mask):
    env = LocalParkingEnv(
        config=replace(DEFAULT_ENV_CONFIG, curriculum_stage=1),
        action_mask=synthetic_action_mask,
        seed=4,
    )
    env.reset()
    env.state = ArticulatedState(
        env.slot.x_goal,
        env.slot.y_goal,
        env.slot.theta_goal + 2.0 * math.pi,
        env.slot.theta_goal,
    )
    env._update_sensors_and_mask()
    _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert info["front_overlap"] >= 0.80
    assert info["heading_error_deg"] <= 10.0
    assert "rear_body_overlap" in info
    assert "rear_heading_error_deg" in info
