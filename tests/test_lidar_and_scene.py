from types import SimpleNamespace
import math

import numpy as np
import pytest
from shapely.errors import GEOSException
from shapely.geometry import Point
from shapely.ops import unary_union

from config import DEFAULT_SCENE_CONFIG, DEFAULT_VEHICLE_PARAMS
from dataclasses import replace
from config import DEFAULT_ENV_CONFIG
from env.geometry import oriented_box, wrap_to_pi
from env.lidar import DualBodyLidar
from env.local_parking_env import LocalParkingEnv, ResetInitialStateError
from env.mixing_plant_scene import (
    CachedScenePool,
    TASK_FAMILIES,
    derive_scene_seed,
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
    assert float(first.metadata["corridor_width"]) == 5.0
    assert len(first.parking_bays) >= 2
    assert first.metadata["constructed_obstacle_feature_count"] > 0
    assert first.metadata["reset_geometry_candidate_count"] > 0


def test_stage4_scene_keeps_narrow_corridor_obstacle_categories_and_reset_audit():
    assert DEFAULT_SCENE_CONFIG.corridor_width_by_stage[3] == 5.0
    assert DEFAULT_SCENE_CONFIG.noncritical_obstacle_count_by_stage[3] == 7
    for seed in range(8):
        scene = generate_cached_mixing_plant_scene(stage=4, seed=600 + seed)
        labels = set(scene.metadata["constructed_obstacle_labels"])
        assert float(scene.metadata["corridor_width"]) == 5.0
        assert 0 < scene.metadata["constructed_obstacle_feature_count"] <= 7
        assert scene.metadata["constructed_wall_feature_count"] >= 0
        assert labels
        assert scene.metadata["topology_variant"] in {
            "straight_main",
            "t_branch",
            "double_t",
            "short_dead_end",
            "offset_parallel_aisle",
            "dogleg",
            "bulb_turnaround",
            "chicane",
        }
        assert scene.metadata["local_complexity_variant"]
        assert scene.metadata["success_neighborhood_feasible_count"] > 0
        assert scene.metadata["reset_geometry_candidate_count"] > 0
        assert scene.metadata["reset_geometry_recovery_band_count"] > 0


def _in_target_bay_approach_band(scene, state):
    bay = scene.target_bay
    mouth = np.asarray(bay.mouth_center, dtype=np.float64)
    corridor_axis = np.asarray(
        [math.cos(bay.corridor_heading), math.sin(bay.corridor_heading)],
        dtype=np.float64,
    )
    inward_axis = np.asarray(
        [math.cos(bay.inward_heading), math.sin(bay.inward_heading)],
        dtype=np.float64,
    )
    delta = np.asarray((state.x_front, state.y_front), dtype=np.float64) - mouth
    along = abs(float(np.dot(delta, corridor_axis)))
    inward = float(np.dot(delta, inward_axis))
    bay_half_width = 0.5 * float(DEFAULT_SCENE_CONFIG.head_in_bay_width)
    corridor_width = float(scene.metadata["corridor_width"])
    return bool(
        along
        <= bay_half_width + float(DEFAULT_ENV_CONFIG.stage_lateral_ranges[3]) + 0.75
        and -corridor_width - 0.75
        <= inward
        <= float(DEFAULT_SCENE_CONFIG.head_in_bay_depth) + 0.75
    )


def test_stage4_structured_recovery_candidates_stay_at_target_bay_approach(
    synthetic_action_mask,
):
    env = LocalParkingEnv(
        config=replace(DEFAULT_ENV_CONFIG, curriculum_stage=4, scene_pool_size=1),
        action_mask=synthetic_action_mask,
        seed=10,
    )
    for seed in (32690671, 4009347464, 600, 601, 602, 603):
        scene = generate_cached_mixing_plant_scene(stage=4, seed=seed)
        env.scene = scene
        env.slot = scene.slot
        axis = np.asarray(
            [math.cos(scene.slot.theta_goal), math.sin(scene.slot.theta_goal)],
            dtype=np.float64,
        )
        normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
        bank = env._build_reset_candidate_bank(4, scene.slot, axis, normal)
        candidates = bank["candidates"]
        assert candidates
        assert all(
            _in_target_bay_approach_band(scene, candidate["state"])
            for candidate in candidates
        )


def test_different_seeds_produce_distinct_parameterized_layouts():
    scenes = [
        generate_cached_mixing_plant_scene(
            stage=3,
            seed=100 + index,
            task_family="head_in",
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
    assert {scene.metadata["task_family"] for scene in scenes} == {"head_in"}
    assert {scene.metadata["goal_orientation_mode"] for scene in scenes} == {"head_in"}
    assert len({scene.metadata["obstacle_layout_variant"] for scene in scenes}) > 1
    assert len({scene.metadata["topology_variant"] for scene in scenes}) > 1
    assert len({scene.metadata["local_complexity_variant"] for scene in scenes}) > 1


def test_default_scene_topology_variants_cover_multiple_seeded_layouts():
    scenes = [
        generate_cached_mixing_plant_scene(
            stage=3,
            seed=900 + index,
            task_family="head_in",
        )
        for index in range(32)
    ]
    variants = {scene.metadata["topology_variant"] for scene in scenes}
    assert len(variants) >= 4
    assert all(scene.metadata["task_family"] == "head_in" for scene in scenes)


def test_scene_pool_uses_explicit_family_schedule_and_derived_seeds():
    schedule = ("head_in",)
    first = CachedScenePool(
        stage=1,
        pool_size=4,
        base_seed=23,
        family_schedule=schedule,
    )
    second = CachedScenePool(
        stage=1,
        pool_size=4,
        base_seed=23,
        family_schedule=schedule,
    )
    families = [first.get(index).metadata["task_family"] for index in range(4)]
    seeds = [first.get(index).metadata["seed"] for index in range(4)]
    assert families == ["head_in"] * 4
    assert seeds == [second.get(index).metadata["seed"] for index in range(4)]
    assert seeds != [23 + index for index in range(4)]


def test_scene_pool_expands_default_schedule_to_balanced_family_counts():
    pool = CachedScenePool(
        stage=1,
        pool_size=16,
        base_seed=31,
        family_schedule=TASK_FAMILIES,
    )
    families = [pool.get(index).metadata["task_family"] for index in range(pool.pool_size)]
    assert pool.pool_size == 16
    assert families.count("head_in") == 16


def test_rule_scene_generators_are_seed_reproducible():
    for scene_type in (
        "mixing_station_bay_corridor",
        "loading_truck_rectangle_space",
    ):
        cfg = replace(DEFAULT_SCENE_CONFIG, scene_type=scene_type)
        first = generate_cached_mixing_plant_scene(
            stage=1,
            seed=101,
            scene_config=cfg,
            task_family="head_in",
        )
        second = generate_cached_mixing_plant_scene(
            stage=1,
            seed=101,
            scene_config=cfg,
            task_family="head_in",
        )
        assert np.array_equal(first.occupancy_grid, second.occupancy_grid)
        assert first.world_bounds == second.world_bounds
        assert (
            round(first.slot.x_goal, 6),
            round(first.slot.y_goal, 6),
            round(first.slot.theta_goal, 6),
        ) == (
            round(second.slot.x_goal, 6),
            round(second.slot.y_goal, 6),
            round(second.slot.theta_goal, 6),
        )
        assert first.metadata["scene_type"] == scene_type


def test_mixing_station_bay_corridor_geometry_and_reset(synthetic_action_mask):
    cfg = replace(
        DEFAULT_SCENE_CONFIG,
        scene_type="mixing_station_bay_corridor",
        target_bay_sampling_mode="fixed",
        fixed_target_bay_index=1,
        initial_spawn_mode="bay",
        fixed_initial_bay_index=2,
    )
    scene = generate_cached_mixing_plant_scene(
        stage=1,
        seed=202,
        scene_config=cfg,
        task_family="head_in",
    )
    assert scene.metadata["bay_count"] == cfg.bay_count
    assert scene.metadata["partition_wall_count"] == cfg.bay_count - 1
    assert scene.metadata["bottom_wall_exists"] is True
    assert scene.metadata["corridor_ends_open"] is True
    assert scene.metadata["corridor_end_wall_count"] == 0
    assert scene.metadata["corridor_outer_wall_exists"] is True
    assert float(scene.metadata["corridor_width"]) == cfg.corridor_width
    assert scene.metadata["target_bay_index"] == 1
    assert scene.metadata["requested_initial_spawn_mode"] == "bay"
    assert scene.metadata["initial_spawn_mode"] == "corridor"
    assert {
        str(candidate[0]) for candidate in scene.metadata["initial_pose_candidates"]
    } == {"corridor"}

    target_state = ArticulatedState(
        scene.slot.x_goal,
        scene.slot.y_goal,
        scene.slot.theta_goal,
        scene.slot.theta_goal,
    )
    model = ArticulatedVehicleModel(DEFAULT_VEHICLE_PARAMS)
    front_box, rear_box = model.body_boxes(target_state)
    assert scene.target_bay.polygon.covers(front_box)
    assert scene.target_bay.polygon.covers(rear_box)
    target_direction = np.asarray(
        [math.cos(scene.slot.theta_goal), math.sin(scene.slot.theta_goal)]
    )
    inward = np.asarray(
        [math.cos(scene.target_bay.inward_heading), math.sin(scene.target_bay.inward_heading)]
    )
    assert float(np.dot(target_direction, inward)) > 0.999

    corridor_y0 = float(scene.metadata["corridor_region_bounds"][1])
    wall_count = int(scene.metadata["wall_obstacle_count"])
    assert wall_count == 4 + cfg.bay_count - 1
    assert len(scene.obstacle_polygons) == wall_count + scene.metadata["parked_vehicle_count"]
    for wall in scene.obstacle_polygons[2:wall_count]:
        assert wall.bounds[3] <= corridor_y0 + 1e-8
    for parked_vehicle in scene.obstacle_polygons[wall_count:]:
        assert parked_vehicle.bounds[3] <= corridor_y0 + 1e-8

    env = LocalParkingEnv(
        config=replace(DEFAULT_ENV_CONFIG, scene_pool_size=1),
        scene_config=cfg,
        action_mask=synthetic_action_mask,
        seed=202,
    )
    _, info = env.reset(seed=202)
    assert info["scene_type"] == "mixing_station_bay_corridor"
    assert info["requested_initial_spawn_mode"] == "bay"
    assert info["initial_spawn_mode"] == "corridor"
    assert info["initial_spawn_region"] == "corridor"
    assert info["initial_bay_index"] == -1
    assert info["reset_feasible_mask_available"] is True
    assert not info["initial_collision"]


def test_mixing_station_parked_vehicles_stay_in_non_target_bays():
    cfg = replace(
        DEFAULT_SCENE_CONFIG,
        scene_type="mixing_station_bay_corridor",
        target_bay_sampling_mode="fixed",
        fixed_target_bay_index=0,
        bay_parked_vehicle_count_range=(4, 4),
    )
    scene = generate_cached_mixing_plant_scene(
        stage=1,
        seed=707,
        scene_config=cfg,
        task_family="head_in",
    )
    wall_count = int(scene.metadata["wall_obstacle_count"])
    parked_count = int(scene.metadata["parked_vehicle_count"])
    parked_vehicles = scene.obstacle_polygons[wall_count : wall_count + parked_count]
    wall_union = unary_union(scene.obstacle_polygons[:wall_count])
    assert parked_count > 0
    assert len(parked_vehicles) == parked_count
    assert len(scene.metadata["parked_vehicle_labels"]) == parked_count
    assert len(scene.metadata["parked_vehicle_headings"]) == parked_count

    max_heading_error = math.radians(cfg.bay_parked_vehicle_heading_noise_deg) + 1e-8
    for vehicle, label, heading in zip(
        parked_vehicles,
        scene.metadata["parked_vehicle_labels"],
        scene.metadata["parked_vehicle_headings"],
    ):
        bay_index = int(str(label).split("_")[-1])
        bay = scene.parking_bays[bay_index]
        assert bay is not scene.target_bay
        assert bay.polygon.covers(vehicle)
        assert not vehicle.intersects(wall_union)
        assert not vehicle.intersects(scene.target_bay.polygon)
        assert abs(wrap_to_pi(float(heading) - bay.inward_heading)) <= max_heading_error

    for index, first in enumerate(parked_vehicles):
        for second in parked_vehicles[index + 1 :]:
            assert not first.intersects(second)
            assert first.distance(second) >= cfg.bay_parked_vehicle_pair_spacing - 1e-8


def test_loading_truck_rectangle_space_geometry_and_obstacle_exclusion(
    synthetic_action_mask,
):
    cfg = replace(
        DEFAULT_SCENE_CONFIG,
        scene_type="loading_truck_rectangle_space",
        discrete_obstacle_count=4,
    )
    scene = generate_cached_mixing_plant_scene(
        stage=1,
        seed=303,
        scene_config=cfg,
        task_family="head_in",
    )
    assert scene.metadata["boundary_wall_count"] == 0
    assert scene.metadata["constructed_wall_feature_count"] == 0
    assert len(scene.obstacle_polygons) >= 1
    assert len(scene.obstacle_polygons) == 1 + scene.metadata["discrete_obstacle_count"]
    assert scene.metadata["truck_in_front"] is True
    assert scene.metadata["truck_perpendicular"] is True
    assert scene.metadata["discrete_obstacle_count"] <= cfg.discrete_obstacle_count

    initial_xys = tuple(
        np.asarray((candidate[2], candidate[3]), dtype=np.float64)
        for candidate in scene.metadata["initial_pose_candidates"]
    )
    target_xy = np.asarray(scene.slot.center, dtype=np.float64)
    truck = scene.obstacle_polygons[0]
    initial_radius = float(cfg.obstacle_exclusion_radius_around_initial)
    target_radius = float(cfg.obstacle_exclusion_radius_around_target)
    truck_radius = float(cfg.obstacle_exclusion_radius_around_truck)
    target_exclusion = Point(float(target_xy[0]), float(target_xy[1])).buffer(
        target_radius,
        resolution=16,
    )
    for obstacle in scene.obstacle_polygons[1:]:
        center = np.asarray(obstacle.centroid.coords[0], dtype=np.float64)
        assert all(
            float(np.linalg.norm(center - initial_xy)) >= initial_radius
            for initial_xy in initial_xys
        )
        assert float(np.linalg.norm(center - target_xy)) >= target_radius
        assert not obstacle.intersects(target_exclusion)
        assert obstacle.distance(truck) >= truck_radius - 1e-8

    env = LocalParkingEnv(
        config=replace(DEFAULT_ENV_CONFIG, scene_pool_size=1),
        scene_config=cfg,
        action_mask=synthetic_action_mask,
        seed=303,
    )
    obs, info = env.reset(seed=303)
    assert obs.shape == (LocalParkingEnv.OBS_DIM,)
    assert env.last_front_lidar_m.shape == (DEFAULT_VEHICLE_PARAMS.lidar_beams,)
    assert env.current_mask.shape == (2, 11)
    assert info["scene_type"] == "loading_truck_rectangle_space"
    assert info["truck_in_front"] is True
    assert info["reset_feasible_mask_available"] is True
    assert not info["initial_collision"]


def test_rule_scenes_reset_all_curriculum_stages(synthetic_action_mask):
    for scene_type in (
        "mixing_station_bay_corridor",
        "loading_truck_rectangle_space",
    ):
        scene_config = replace(DEFAULT_SCENE_CONFIG, scene_type=scene_type)
        for stage in (1, 2, 3, 4):
            env = LocalParkingEnv(
                config=replace(
                    DEFAULT_ENV_CONFIG,
                    curriculum_stage=stage,
                    scene_pool_size=1,
                ),
                scene_config=scene_config,
                action_mask=synthetic_action_mask,
                seed=700 + stage,
            )
            obs, info = env.reset(seed=700 + stage)
            assert obs.shape == (LocalParkingEnv.OBS_DIM,)
            assert info["scene_type"] == scene_type
            assert info["reset_feasible_mask_available"] is True
            _, _, terminated, truncated, step_info = env.step(
                np.array([0.0, 0.0], dtype=np.float32)
            )
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert "collision" in step_info


def test_target_slot_is_inside_bay_with_supported_orientation_modes():
    model = ArticulatedVehicleModel(DEFAULT_VEHICLE_PARAMS)
    modes = set()
    for index, task_family in enumerate(TASK_FAMILIES * 12):
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
        assert scene.metadata["scene_complexity_bucket"] in {
            "normal",
            "complex",
            "extreme",
        }
        assert scene.metadata["constructed_obstacle_feature_count"] > 0
        assert scene.metadata["constructed_wall_feature_count"] > 0
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
        assert abs(float(np.dot(goal_direction, corridor_direction))) < 1e-6
        assert float(np.dot(goal_direction, inward_direction)) > 0.999
    assert modes == {"head_in"}


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
            task_family="head_in",
        )
        for index, first in enumerate(scene.parking_bays):
            for second in scene.parking_bays[index + 1 :]:
                assert first.polygon.intersection(second.polygon).area <= 1e-8


def test_recovery_samples_are_diverse_near_obstacles_and_articulated(
    synthetic_action_mask,
):
    env = LocalParkingEnv(
        config=replace(DEFAULT_ENV_CONFIG, curriculum_stage=4, scene_pool_size=1),
        action_mask=synthetic_action_mask,
        seed=10,
    )
    samples = []
    state_signatures = set()
    collisions = []
    for _ in range(24):
        _, info = env.reset()
        samples.append(info)
        state_signatures.add(tuple(np.round(env.state.as_array()[:4], 3)))
        collisions.append(env._state_collides(env.state))
        assert _in_target_bay_approach_band(env.scene, env.state)
    assert all(item["scenario_type"] == "recovery" for item in samples)
    assert all(item["min_lidar_distance"] <= 2.2 for item in samples)
    clearances = [float(item["reset_initial_body_clearance_m"]) for item in samples]
    assert min(clearances) >= DEFAULT_ENV_CONFIG.stage4_reset_min_body_clearance
    assert max(clearances) <= DEFAULT_ENV_CONFIG.recovery_max_body_clearance
    assert float(np.median(clearances)) >= 0.20
    assert all(item["reset_candidate_bank_size"] > 0 for item in samples)
    assert all(item["reset_candidate_bank_valid_count"] > 0 for item in samples)
    assert not any(item["reset_candidate_bank_empty"] for item in samples)
    assert not any(item["reset_initial_mask_all_zero"] for item in samples)
    clearance_buckets = {
        item["reset_candidate_selected_clearance_bucket"] for item in samples
    }
    pose_buckets = {item["reset_candidate_selected_pose_bucket"] for item in samples}
    assert clearance_buckets <= {
        "tight_recover",
        "narrow_recover",
        "moderate_recover",
    }
    assert clearance_buckets & {"tight_recover", "narrow_recover"}
    assert len(pose_buckets) >= 2
    assert not any(collisions)
    assert len(state_signatures) >= 6


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
    assert max(item[0] for item in states) - min(item[0] for item in states) > 2.0
    assert max(item[1] for item in states) > 45.0
    assert max(item[2] for item in states) > 2.0
    assert max(item[3] for item in states) > 10.0


def test_deprecated_parallel_families_are_rejected():
    with pytest.raises(ValueError, match="unsupported task family"):
        CachedScenePool(
            stage=1,
            pool_size=1,
            base_seed=23,
            family_schedule=("parallel_rev",),
        )


def test_cached_scene_pool_retries_after_shapely_generation_failure(monkeypatch):
    original_generate_scene = CachedScenePool._generate_scene
    calls = {"count": 0, "indices": []}

    def fail_first_scene_then_generate(self, pool_index, task_family=None, scene_type=None):
        calls["count"] += 1
        calls["indices"].append(int(pool_index))
        if calls["count"] == 1:
            raise GEOSException("TopologyException: side location conflict")
        return original_generate_scene(
            self,
            pool_index,
            task_family=task_family,
            scene_type=scene_type,
        )

    monkeypatch.setattr(
        CachedScenePool,
        "_generate_scene",
        fail_first_scene_then_generate,
    )

    pool = CachedScenePool(
        stage=1,
        pool_size=1,
        base_seed=31,
        family_schedule=("head_in",),
    )
    scene = pool.get(0)
    failed_seed = derive_scene_seed(31, 0, "head_in", 1)
    success_seed = derive_scene_seed(31, 1, "head_in", 1)

    assert calls["indices"] == [0, 1]
    assert scene.metadata["scene_generation_attempt_count"] == 2
    assert int(scene.metadata["seed"]) == success_seed
    assert int(scene.metadata["seed"]) != failed_seed


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
            scene_family_schedule=("head_in",),
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
    assert info["task_family"] == "head_in"
    assert info["reset_scene_retry_count"] == 1
    assert info["reset_scene_last_failed_seed"] == failed_seed
    assert info["scene_seed"] != failed_seed
    assert info["reset_scene_success_seed"] == info["scene_seed"]
