from types import SimpleNamespace
import math

import numpy as np
import pytest

from config import DEFAULT_VEHICLE_PARAMS
from dataclasses import replace
from config import DEFAULT_ENV_CONFIG
from env.geometry import oriented_box
from env.lidar import DualBodyLidar
from env.local_parking_env import LocalParkingEnv, ResetInitialStateError
from env.mixing_plant_scene import (
    CachedScenePool,
    TASK_FAMILIES,
    generate_cached_mixing_plant_scene,
)
from env.vehicle import ArticulatedState, ArticulatedVehicleModel


class _ConstantActionMask:
    feature_dim = 22
    min_safe_ratio = 1e-3

    def __init__(self, value):
        self.value = float(value)

    def compute_mask(self, phi, front_lidar_m, rear_lidar_m):
        return np.full((2, 11), self.value, dtype=np.float32)


def _edges(polygons):
    result = []
    for polygon in polygons:
        coords = np.asarray(polygon.exterior.coords)
        result.extend(np.stack([coords[:-1], coords[1:]], axis=1))
    return np.asarray(result)


def test_front_rear_lidar_frames_dimensions_and_normalization():
    p = DEFAULT_VEHICLE_PARAMS
    model = ArticulatedVehicleModel(p)
    state = ArticulatedState(0.0, 0.0, 0.0, math.pi / 2.0)
    rear_center = model.rear_center(state)
    obstacles = [
        oriented_box((6.0, 0.0), 0.0, 1.0, 1.0),
        oriented_box((rear_center[0], rear_center[1] + 6.0), 0.0, 1.0, 1.0),
    ]
    scene = SimpleNamespace(obstacle_edges=_edges(obstacles))
    lidar = DualBodyLidar(p)
    front, rear = lidar.observe(state, model, scene, normalize=True)
    assert front.shape == (54,)
    assert rear.shape == (54,)
    assert np.all((front >= 0.0) & (front <= 1.0))
    assert np.all((rear >= 0.0) & (rear <= 1.0))
    assert np.isclose(front[0] * p.lidar_range, 5.5, atol=0.1)
    assert np.isclose(rear[0] * p.lidar_range, 5.5, atol=0.1)


def test_rule_carved_scene_is_cached_and_deterministic():
    first = generate_cached_mixing_plant_scene(stage=3, seed=7)
    second = generate_cached_mixing_plant_scene(stage=3, seed=7)
    assert first is second
    assert (
        first.metadata["generation_mode"]
        == "blocked_grid_then_constructive_corridor_and_bay_carve"
    )
    assert np.array_equal(first.occupancy_grid, second.occupancy_grid)
    assert 0.0 < first.metadata["free_ratio"] < 1.0
    assert first.world_bounds == (-40.0, -40.0, 40.0, 40.0)
    assert first.occupancy_grid.shape == (80, 80)
    assert float(first.metadata["corridor_width"]) >= 12.0
    assert len(first.parking_bays) >= 2


def test_different_seeds_produce_distinct_parameterized_layouts():
    scenes = [
        generate_cached_mixing_plant_scene(
            stage=3,
            seed=100 + index,
            task_family=TASK_FAMILIES[index % len(TASK_FAMILIES)],
        )
        for index in range(18)
    ]
    grid_signatures = {scene.occupancy_grid.tobytes() for scene in scenes}
    slot_poses = {
        (
            round(scene.slot.x_goal, 3),
            round(scene.slot.y_goal, 3),
            round(scene.slot.theta_goal, 3),
        )
        for scene in scenes
    }
    assert len(grid_signatures) == len(scenes)
    assert len(slot_poses) >= 12
    assert {scene.metadata["task_family"] for scene in scenes} == set(TASK_FAMILIES)
    assert {scene.metadata["goal_orientation_mode"] for scene in scenes} == {
        "head_in",
        "parallel",
    }


def test_scene_pool_uses_explicit_family_schedule_and_derived_seeds():
    schedule = ("parallel_rev", "head_in", "parallel_rev", "parallel_fwd")
    first = CachedScenePool(
        stage=1,
        pool_size=8,
        base_seed=23,
        family_schedule=schedule,
    )
    second = CachedScenePool(
        stage=1,
        pool_size=8,
        base_seed=23,
        family_schedule=schedule,
    )
    families = [first.get(index).metadata["task_family"] for index in range(8)]
    seeds = [first.get(index).metadata["seed"] for index in range(8)]
    assert families == list(schedule) * 2
    assert seeds == [second.get(index).metadata["seed"] for index in range(8)]
    assert seeds != [23 + index for index in range(8)]
    assert families.count("parallel_rev") == 4
    assert families.count("head_in") == 2
    assert families.count("parallel_fwd") == 2


def test_scene_pool_expands_default_schedule_to_balanced_family_counts():
    pool = CachedScenePool(
        stage=1,
        pool_size=16,
        base_seed=31,
        family_schedule=TASK_FAMILIES,
    )
    families = [pool.get(index).metadata["task_family"] for index in range(pool.pool_size)]
    assert pool.pool_size == 18
    assert families.count("head_in") == 6
    assert families.count("parallel_fwd") == 6
    assert families.count("parallel_rev") == 6


def test_target_slot_is_inside_bay_with_supported_orientation_modes():
    model = ArticulatedVehicleModel(DEFAULT_VEHICLE_PARAMS)
    modes = set()
    for index, task_family in enumerate(TASK_FAMILIES * 6):
        scene = generate_cached_mixing_plant_scene(
            stage=3,
            seed=200 + index,
            task_family=task_family,
        )
        modes.add(scene.metadata["goal_orientation_mode"])
        assert scene.metadata["task_family"] == task_family
        assert not scene.metadata["nominal_target_collision"]
        assert scene.metadata["nominal_target_front_in_bay"]
        assert scene.metadata["nominal_target_rear_in_bay"]
        assert scene.metadata["success_neighborhood_feasible_count"] > 0
        assert scene.metadata["clearance_bucket"] in {
            "tight",
            "narrow",
            "normal",
            "open",
        }
        assert scene.metadata["approach_side_bucket"] in {"left_bay", "right_bay"}
        assert scene.metadata["reverse_required_bucket"] in {
            "required",
            "not_required",
        }
        goal = scene.slot
        target_state = ArticulatedState(
            goal.x_goal,
            goal.y_goal,
            goal.theta_goal,
            goal.theta_goal,
        )
        front_box, rear_box = model.body_boxes(target_state)
        assert scene.target_bay.polygon.covers(front_box)
        assert scene.target_bay.polygon.covers(rear_box)
        assert not scene.prepared_obstacles.intersects(front_box)
        assert not scene.prepared_obstacles.intersects(rear_box)

        goal_direction = np.asarray(
            [math.cos(goal.theta_goal), math.sin(goal.theta_goal)]
        )
        corridor_direction = np.asarray(
            [
                math.cos(scene.target_bay.corridor_heading),
                math.sin(scene.target_bay.corridor_heading),
            ]
        )
        inward_direction = np.asarray(
            [
                math.cos(scene.target_bay.inward_heading),
                math.sin(scene.target_bay.inward_heading),
            ]
        )
        if scene.metadata["goal_orientation_mode"] == "parallel":
            assert abs(float(np.dot(goal_direction, corridor_direction))) > 0.999
        else:
            assert float(np.dot(goal_direction, inward_direction)) > 0.999
    assert modes == {"head_in", "parallel"}


def test_target_bay_mouth_connects_to_main_corridor():
    for seed in range(16):
        scene = generate_cached_mixing_plant_scene(stage=3, seed=seed)
        mouth = np.asarray(scene.target_bay.mouth_center)
        inward = np.asarray(
            [
                math.cos(scene.target_bay.inward_heading),
                math.sin(scene.target_bay.inward_heading),
            ]
        )
        assert not scene.is_occupied_world(*(mouth - 0.5 * inward))
        assert not scene.is_occupied_world(*(mouth + 0.5 * inward))


def test_parking_bays_do_not_overlap_each_other():
    for index in range(16):
        scene = generate_cached_mixing_plant_scene(
            stage=3,
            seed=index,
            task_family=TASK_FAMILIES[index % len(TASK_FAMILIES)],
        )
        for index, first in enumerate(scene.parking_bays):
            for second in scene.parking_bays[index + 1 :]:
                assert first.polygon.intersection(second.polygon).area <= 1e-8


def test_recovery_samples_are_diverse_near_obstacles_and_articulated(
    synthetic_action_mask,
):
    env = LocalParkingEnv(
        config=replace(DEFAULT_ENV_CONFIG, curriculum_stage=4),
        action_mask=synthetic_action_mask,
        seed=10,
    )
    samples = []
    state_signatures = set()
    collisions = []
    for _ in range(16):
        _, info = env.reset()
        samples.append(info)
        state_signatures.add(tuple(np.round(env.state.as_array()[:4], 3)))
        collisions.append(env._state_collides(env.state))
    assert all(item["scenario_type"] == "recovery" for item in samples)
    assert all(item["min_lidar_distance"] <= 2.2 for item in samples)
    assert all(abs(item["phi"]) >= math.radians(18.0) for item in samples)
    assert not any(collisions)
    assert len(state_signatures) >= 12


def test_near_goal_initial_states_cover_distance_heading_lateral_and_phi(
    synthetic_action_mask,
):
    env = LocalParkingEnv(
        config=replace(DEFAULT_ENV_CONFIG, curriculum_stage=2),
        action_mask=synthetic_action_mask,
        seed=31,
    )
    states = []
    for _ in range(32):
        _, info = env.reset()
        position_error = env.slot.position_error_in_slot_frame(
            env.state.x_front,
            env.state.y_front,
        )
        states.append(
            (
                info["distance_to_goal"],
                abs(info["heading_error_deg"]),
                abs(float(position_error[1])),
                abs(math.degrees(env.state.phi)),
                tuple(np.round(env.state.as_array()[:4], 3)),
            )
        )

    assert len({item[4] for item in states}) == len(states)
    assert max(item[0] for item in states) - min(item[0] for item in states) > 4.0
    assert max(item[1] for item in states) > 45.0
    assert max(item[2] for item in states) > 2.0
    assert max(item[3] for item in states) > 10.0


def test_parallel_reverse_initial_curriculum_expands_from_warmup(
    synthetic_action_mask,
):
    env = LocalParkingEnv(
        config=replace(
            DEFAULT_ENV_CONFIG,
            curriculum_stage=1,
            scene_pool_size=1,
            scene_family_schedule=("parallel_rev",),
            parallel_rev_curriculum_episodes=4,
        ),
        action_mask=synthetic_action_mask,
        seed=17,
    )
    _, first_info = env.reset(seed=17)
    assert first_info["task_family"] == "parallel_rev"
    assert first_info["scenario_type"] == "parallel_rev_warmup"
    assert np.isclose(first_info["parallel_rev_curriculum_progress"], 0.0)
    assert np.isclose(first_info["initial_distance_min"], 4.0)
    assert np.isclose(first_info["initial_distance_max"], 8.0)
    assert first_info["initial_lateral_range"] <= 1.25
    assert np.isclose(first_info["initial_heading_range_deg"], 20.0)
    assert np.isclose(first_info["initial_phi_range_deg"], 8.0)

    last_info = first_info
    for _ in range(4):
        _, last_info = env.reset()
    assert np.isclose(last_info["parallel_rev_curriculum_progress"], 1.0)
    assert np.isclose(last_info["initial_distance_min"], 8.0)
    assert np.isclose(last_info["initial_distance_max"], 15.0)
    assert np.isclose(last_info["initial_heading_range_deg"], 45.0)
    assert np.isclose(last_info["initial_phi_range_deg"], 12.0)


def test_parallel_reverse_curriculum_counts_actual_reverse_samples(
    synthetic_action_mask,
):
    env = LocalParkingEnv(
        config=replace(
            DEFAULT_ENV_CONFIG,
            curriculum_stage=1,
            scene_pool_size=2,
            scene_family_schedule=("head_in", "parallel_rev"),
            parallel_rev_curriculum_episodes=4,
        ),
        action_mask=synthetic_action_mask,
        seed=23,
    )
    _, head_info = env.reset(seed=23)
    assert head_info["task_family"] == "head_in"

    _, rev_info = env.reset()
    assert rev_info["task_family"] == "parallel_rev"
    assert np.isclose(rev_info["parallel_rev_curriculum_progress"], 0.0)
    assert rev_info["parallel_rev_curriculum_sample_count"] == 0
    assert rev_info["parallel_rev_curriculum_sample_count_after"] == 1


def test_reset_rejects_collision_free_state_without_executable_mask():
    env = LocalParkingEnv(
        config=replace(
            DEFAULT_ENV_CONFIG,
            curriculum_stage=1,
            scene_pool_size=1,
            scene_family_schedule=("head_in",),
            initial_sampling_attempts=2,
            reset_scene_retry_count=2,
        ),
        action_mask=_ConstantActionMask(0.0),
        seed=29,
    )
    with pytest.raises(RuntimeError, match="reset viability"):
        env.reset(seed=29)


def test_reset_replaces_scene_seed_after_recovery_sampling_failure(
    synthetic_action_mask,
    monkeypatch,
):
    env = LocalParkingEnv(
        config=replace(
            DEFAULT_ENV_CONFIG,
            curriculum_stage=4,
            scene_pool_size=1,
            scene_family_schedule=("parallel_rev",),
            reset_scene_retry_count=3,
        ),
        action_mask=synthetic_action_mask,
        seed=37,
    )
    failed_seed = int(env.scene_pool.get(0).metadata["seed"])
    original_sample_initial_state = env._sample_initial_state
    calls = {"count": 0}

    def fail_once_then_sample():
        if calls["count"] == 0:
            calls["count"] += 1
            raise ResetInitialStateError(
                "no reset-viable near-obstacle recovery state for scene seed {}".format(
                    env.scene.metadata["seed"]
                )
            )
        return original_sample_initial_state()

    monkeypatch.setattr(env, "_sample_initial_state", fail_once_then_sample)

    _, info = env.reset(seed=37)

    assert calls["count"] == 1
    assert info["task_family"] == "parallel_rev"
    assert info["reset_scene_retry_count"] == 1
    assert info["reset_scene_last_failed_seed"] == failed_seed
    assert info["scene_seed"] != failed_seed
    assert info["reset_scene_success_seed"] == info["scene_seed"]
