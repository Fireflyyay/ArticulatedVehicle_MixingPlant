from dataclasses import dataclass, replace
from functools import lru_cache
import math
from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.errors import GEOSException

from config import (
    DEFAULT_ENV_CONFIG,
    DEFAULT_SCENE_CONFIG,
    DEFAULT_VEHICLE_PARAMS,
    MixingPlantSceneConfig,
)
from env.geometry import DirectedParkingSlot, oriented_box, overlap_ratio, wrap_to_pi
from env.vehicle import ArticulatedState, ArticulatedVehicleModel


TASK_FAMILIES = ("head_in",)
_TASK_FAMILY_TO_INDEX = dict((family, index) for index, family in enumerate(TASK_FAMILIES))
DEFAULT_SCENE_TYPES = ("default", "existing", "cached_rule_carved_mixing_plant")
RULE_SCENE_TYPES = (
    "mixing_station_bay_corridor",
    "loading_truck_rectangle_space",
)
SUPPORTED_SCENE_TYPES = DEFAULT_SCENE_TYPES + RULE_SCENE_TYPES
_RETRYABLE_SCENE_GENERATION_ERRORS = (RuntimeError, GEOSException)


@dataclass(frozen=True)
class ParkingBay:
    name: str
    polygon: Polygon
    mouth_center: Tuple[float, float]
    mouth_segment: Tuple[Tuple[float, float], Tuple[float, float]]
    inward_heading: float
    corridor_heading: float
    goal_orientation_mode: str
    goal_heading: float
    is_target: bool = False


@dataclass
class MixingPlantScene:
    occupancy_grid: np.ndarray
    obstacle_polygons: List[Polygon]
    obstacle_edges: np.ndarray
    parking_bays: List[ParkingBay]
    target_bay: ParkingBay
    slot: DirectedParkingSlot
    world_bounds: Tuple[float, float, float, float]
    resolution: float
    metadata: Dict[str, object]
    obstacle_union: object
    prepared_obstacles: object

    def is_occupied_world(self, x, y):
        xmin, ymin, xmax, ymax = self.world_bounds
        if x < xmin or x >= xmax or y < ymin or y >= ymax:
            return True
        col = int((float(x) - xmin) / self.resolution)
        row = int((float(y) - ymin) / self.resolution)
        return bool(self.occupancy_grid[row, col])


@dataclass(frozen=True)
class _SceneLayout:
    corridor_heading: float
    corridor_origin: Tuple[float, float]
    task_family: str
    target_side: float
    target_along: float
    first_branch_along: float
    second_branch_along: float
    branch_bay_along: float
    obstacle_variant: int
    jitter_token: int


def _quantize(value, resolution):
    return float(round(float(value) / float(resolution)) * float(resolution))


def normalize_task_family(task_family):
    task_family = str(task_family)
    if task_family not in _TASK_FAMILY_TO_INDEX:
        raise ValueError(
            "unsupported task family '{}'; expected one of {}".format(
                task_family,
                TASK_FAMILIES,
            )
        )
    return task_family


def normalize_family_schedule(family_schedule):
    if family_schedule is None:
        family_schedule = TASK_FAMILIES
    if isinstance(family_schedule, str):
        family_schedule = tuple(
            part.strip() for part in family_schedule.split(",") if part.strip()
        )
    else:
        family_schedule = tuple(family_schedule)
    if not family_schedule:
        raise ValueError("scene family schedule must not be empty")
    return tuple(normalize_task_family(family) for family in family_schedule)


def normalize_scene_type(scene_type):
    scene_type = "default" if scene_type is None else str(scene_type)
    if scene_type not in SUPPORTED_SCENE_TYPES:
        raise ValueError(
            "unsupported scene_type '{}'; expected one of {}".format(
                scene_type,
                SUPPORTED_SCENE_TYPES,
            )
        )
    return scene_type


def normalize_scene_type_schedule(scene_type_schedule):
    if isinstance(scene_type_schedule, str):
        scene_type_schedule = tuple(
            part.strip() for part in scene_type_schedule.split(",") if part.strip()
        )
    else:
        scene_type_schedule = tuple(scene_type_schedule)
    if not scene_type_schedule:
        raise ValueError("scene type schedule must not be empty")
    return tuple(normalize_scene_type(scene_type) for scene_type in scene_type_schedule)


def family_to_goal_mode(task_family):
    normalize_task_family(task_family)
    return "head_in"


def derive_scene_seed(base_seed, pool_index, task_family, stage):
    task_family = normalize_task_family(task_family)
    seed_items = [
        int(base_seed) & 0xFFFFFFFF,
        int(stage) & 0xFFFFFFFF,
        int(pool_index) & 0xFFFFFFFF,
        _TASK_FAMILY_TO_INDEX[task_family],
    ]
    seed_sequence = np.random.SeedSequence(seed_items)
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def _bucket_clearance(clearance_m):
    clearance_m = float(clearance_m)
    if clearance_m < 0.50:
        return "tight"
    if clearance_m < 1.00:
        return "narrow"
    if clearance_m < 2.00:
        return "normal"
    return "open"


def _sample_layout(seed, scene_config, task_family):
    """Map every seed to a reproducible constructive layout."""
    seed = int(seed)
    task_family = normalize_task_family(task_family)
    rng = np.random.default_rng(np.random.SeedSequence(seed & 0xFFFFFFFF))
    resolution = float(scene_config.resolution)
    corridor_heading = float(rng.choice((0.0, 0.5 * math.pi)))
    axis, normal = _cardinal_frame(corridor_heading)
    jitter = float(scene_config.main_origin_jitter)
    along_shift = _quantize(rng.uniform(-jitter, jitter), resolution)
    normal_shift = _quantize(rng.uniform(-jitter, jitter), resolution)
    origin = axis * along_shift + normal * normal_shift
    target_side = float(rng.choice((-1.0, 1.0)))

    def sample_range(bounds):
        return _quantize(rng.uniform(float(bounds[0]), float(bounds[1])), resolution)

    target_along = sample_range(scene_config.target_bay_along_range)
    branch_candidates = [
        _quantize(float(anchor) + rng.uniform(-1.0, 1.0), resolution)
        for anchor in scene_config.branch_anchor_positions
    ]
    rng.shuffle(branch_candidates)
    branch_positions = []
    for candidate in branch_candidates:
        if abs(candidate - target_along) < float(scene_config.target_branch_clearance):
            continue
        if any(
            abs(candidate - selected)
            < float(scene_config.branch_to_branch_clearance)
            for selected in branch_positions
        ):
            continue
        branch_positions.append(candidate)
        if len(branch_positions) == 2:
            break
    if len(branch_positions) < 2:
        fallback_positions = (-24.0, 24.0)
        branch_positions = sorted(
            fallback_positions,
            key=lambda value: abs(float(value) - target_along),
            reverse=True,
        )

    return _SceneLayout(
        corridor_heading=float(corridor_heading),
        corridor_origin=(float(origin[0]), float(origin[1])),
        task_family=task_family,
        target_side=target_side,
        target_along=target_along,
        first_branch_along=float(branch_positions[0]),
        second_branch_along=float(branch_positions[1]),
        branch_bay_along=sample_range(scene_config.branch_bay_along_range),
        obstacle_variant=int(rng.integers(0, 8)),
        jitter_token=int(rng.integers(0, 2**31 - 1)),
    )


def _carve_rect(occupancy, x0, y0, x1, y1):
    height, width = occupancy.shape
    xa = max(0, min(width, int(math.floor(x0))))
    xb = max(0, min(width, int(math.ceil(x1))))
    ya = max(0, min(height, int(math.floor(y0))))
    yb = max(0, min(height, int(math.ceil(y1))))
    occupancy[ya:yb, xa:xb] = 0


def _carve_world_polygon(occupancy, polygon, world_min, resolution):
    x0, y0, x1, y1 = polygon.bounds
    _carve_rect(
        occupancy,
        (x0 - world_min) / resolution,
        (y0 - world_min) / resolution,
        (x1 - world_min) / resolution,
        (y1 - world_min) / resolution,
    )


def _occupy_world_polygon(occupancy, polygon, world_min, resolution):
    x0, y0, x1, y1 = polygon.bounds
    height, width = occupancy.shape
    xa = max(0, min(width, int(math.floor((x0 - world_min) / resolution))))
    xb = max(0, min(width, int(math.ceil((x1 - world_min) / resolution))))
    ya = max(0, min(height, int(math.floor((y0 - world_min) / resolution))))
    yb = max(0, min(height, int(math.ceil((y1 - world_min) / resolution))))
    occupancy[ya:yb, xa:xb] = 1


def _merge_occupied_cells(occupancy, origin, resolution):
    """Greedy exact rectangle cover for a binary occupancy grid."""
    remaining = occupancy.astype(bool).copy()
    height, width = remaining.shape
    rectangles = []
    ox, oy = origin
    for row in range(height):
        col = 0
        while col < width:
            if not remaining[row, col]:
                col += 1
                continue
            end_col = col
            while end_col < width and remaining[row, end_col]:
                end_col += 1
            end_row = row + 1
            while end_row < height and np.all(remaining[end_row, col:end_col]):
                end_row += 1
            remaining[row:end_row, col:end_col] = False
            rectangles.append(
                box(
                    ox + col * resolution,
                    oy + row * resolution,
                    ox + end_col * resolution,
                    oy + end_row * resolution,
                )
            )
            col = end_col
    return rectangles


def _edges_from_polygons(polygons):
    edges = []
    for polygon in polygons:
        coords = np.asarray(polygon.exterior.coords, dtype=np.float64)
        edges.extend(np.stack([coords[:-1], coords[1:]], axis=1))
    if not edges:
        return np.zeros((0, 2, 2), dtype=np.float64)
    return np.asarray(edges, dtype=np.float64)


def _cardinal_frame(heading):
    axis = np.asarray([math.cos(float(heading)), math.sin(float(heading))])
    normal = np.asarray([-axis[1], axis[0]])
    return axis, normal


def _frame_polygon(origin, heading, u0, u1, n0, n1):
    axis, normal = _cardinal_frame(heading)
    origin_vec = np.asarray(origin, dtype=np.float64)
    corners = [
        origin_vec + axis * u0 + normal * n0,
        origin_vec + axis * u1 + normal * n0,
        origin_vec + axis * u1 + normal * n1,
        origin_vec + axis * u0 + normal * n1,
    ]
    return Polygon(corners)


def _build_bay(
    name,
    corridor_origin,
    corridor_heading,
    corridor_width,
    along_center,
    side,
    mode,
    scene_config,
    is_target=False,
):
    axis, normal = _cardinal_frame(corridor_heading)
    side = 1.0 if float(side) >= 0.0 else -1.0
    if mode == "head_in":
        along_length = float(scene_config.head_in_bay_width)
        depth = float(scene_config.head_in_bay_depth)
    else:
        raise ValueError("unknown parking-bay mode '{}'".format(mode))

    u0 = float(along_center) - 0.5 * along_length
    u1 = float(along_center) + 0.5 * along_length
    edge_n = side * 0.5 * float(corridor_width)
    far_n = edge_n + side * depth
    n0, n1 = sorted((edge_n, far_n))
    polygon = _frame_polygon(corridor_origin, corridor_heading, u0, u1, n0, n1)
    mouth_center = (
        np.asarray(corridor_origin)
        + axis * float(along_center)
        + normal * edge_n
    )
    mouth_a = np.asarray(corridor_origin) + axis * u0 + normal * edge_n
    mouth_b = np.asarray(corridor_origin) + axis * u1 + normal * edge_n
    inward_heading = math.atan2(side * normal[1], side * normal[0])

    goal_heading = inward_heading
    return ParkingBay(
        name=str(name),
        polygon=polygon,
        mouth_center=(float(mouth_center[0]), float(mouth_center[1])),
        mouth_segment=(
            (float(mouth_a[0]), float(mouth_a[1])),
            (float(mouth_b[0]), float(mouth_b[1])),
        ),
        inward_heading=float(wrap_to_pi(inward_heading)),
        corridor_heading=float(wrap_to_pi(corridor_heading)),
        goal_orientation_mode=str(mode),
        goal_heading=float(goal_heading),
        is_target=bool(is_target),
    )


def _goal_center_in_bay(bay, scene_config):
    params = DEFAULT_VEHICLE_PARAMS
    goal_axis = np.asarray(
        [math.cos(float(bay.goal_heading)), math.sin(float(bay.goal_heading))],
        dtype=np.float64,
    )
    center = np.asarray(bay.polygon.centroid.coords[0], dtype=np.float64)
    coords = np.asarray(bay.polygon.exterior.coords[:-1], dtype=np.float64)
    far_projection = float(np.max(coords.dot(goal_axis)))
    center_projection = float(center.dot(goal_axis))
    target_projection = (
        far_projection
        - float(scene_config.parking_head_wall_clearance)
        - params.front_body_length * 0.5
    )
    return center + goal_axis * (target_projection - center_projection)


def _target_feasibility_audit(target_bay, slot, obstacle_union, prepared_obstacles):
    params = DEFAULT_VEHICLE_PARAMS
    model = ArticulatedVehicleModel(params)
    target_state = ArticulatedState(
        x_front=float(slot.x_goal),
        y_front=float(slot.y_goal),
        theta_front=float(slot.theta_goal),
        theta_rear=float(slot.theta_goal),
    )
    target_front, target_rear = model.body_boxes(target_state)
    nominal_target_collision = bool(
        prepared_obstacles.intersects(target_front)
        or prepared_obstacles.intersects(target_rear)
    )
    target_front_in_bay = bool(target_bay.polygon.covers(target_front))
    target_rear_in_bay = bool(target_bay.polygon.covers(target_rear))
    nominal_clearance = float(
        min(target_front.distance(obstacle_union), target_rear.distance(obstacle_union))
    )

    axis = np.asarray(
        [math.cos(float(slot.theta_goal)), math.sin(float(slot.theta_goal))],
        dtype=np.float64,
    )
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    center = np.asarray(slot.center, dtype=np.float64)
    heading_threshold = math.radians(15.0)
    success_overlap_threshold = 0.80
    longitudinal_offsets = np.linspace(-0.40, 0.40, 5)
    lateral_offsets = np.linspace(-0.30, 0.30, 5)
    heading_offsets = np.linspace(-heading_threshold, heading_threshold, 5)
    sample_count = 0
    collision_free_count = 0
    feasible_count = 0
    for longitudinal in longitudinal_offsets:
        for lateral in lateral_offsets:
            for heading_error in heading_offsets:
                sample_count += 1
                sample_center = center + axis * longitudinal + normal * lateral
                theta = wrap_to_pi(float(slot.theta_goal) + float(heading_error))
                sample_state = ArticulatedState(
                    x_front=float(sample_center[0]),
                    y_front=float(sample_center[1]),
                    theta_front=float(theta),
                    theta_rear=float(theta),
                )
                front_box, rear_box = model.body_boxes(sample_state)
                collision = bool(
                    prepared_obstacles.intersects(front_box)
                    or prepared_obstacles.intersects(rear_box)
                )
                if not collision:
                    collision_free_count += 1
                if (
                    not collision
                    and overlap_ratio(front_box, slot.front_box())
                    >= success_overlap_threshold
                    and abs(float(heading_error)) <= heading_threshold
                ):
                    feasible_count += 1

    return {
        "nominal_target_collision": nominal_target_collision,
        "nominal_target_front_in_bay": target_front_in_bay,
        "nominal_target_rear_in_bay": target_rear_in_bay,
        "nominal_target_clearance_m": nominal_clearance,
        "success_neighborhood_sample_count": int(sample_count),
        "success_neighborhood_collision_free_count": int(collision_free_count),
        "success_neighborhood_feasible_count": int(feasible_count),
        "success_neighborhood_overlap_threshold": float(success_overlap_threshold),
        "success_neighborhood_heading_threshold_deg": float(
            math.degrees(heading_threshold)
        ),
    }


def _reset_geometry_audit(slot, obstacle_union, prepared_obstacles):
    """Low-cost reset geometry probe; action-mask viability stays in LocalParkingEnv."""
    params = DEFAULT_VEHICLE_PARAMS
    model = ArticulatedVehicleModel(params)
    axis = np.asarray(
        [math.cos(float(slot.theta_goal)), math.sin(float(slot.theta_goal))],
        dtype=np.float64,
    )
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    center = np.asarray(slot.center, dtype=np.float64)
    distances = (4.5, 6.0, 8.0, 10.0, 12.0, 14.0, 18.0)
    laterals = (-4.0, -2.2, 0.0, 2.2, 4.0)
    heading_offsets = tuple(math.radians(v) for v in (-40.0, -20.0, 0.0, 20.0, 40.0))
    phi_offsets = tuple(math.radians(v) for v in (-30.0, -18.0, 0.0, 18.0, 30.0))
    min_clearance = float(DEFAULT_ENV_CONFIG.stage4_reset_min_body_clearance)
    max_clearance = float(DEFAULT_ENV_CONFIG.recovery_max_body_clearance)
    sample_count = 0
    collision_free_count = 0
    recovery_band_count = 0
    for distance in distances:
        for lateral in laterals:
            sample_center = center - float(distance) * axis + float(lateral) * normal
            for heading_error in heading_offsets:
                theta_front = wrap_to_pi(float(slot.theta_goal) + float(heading_error))
                for phi in phi_offsets:
                    sample_count += 1
                    sample_state = ArticulatedState(
                        x_front=float(sample_center[0]),
                        y_front=float(sample_center[1]),
                        theta_front=float(theta_front),
                        theta_rear=float(wrap_to_pi(theta_front - float(phi))),
                    )
                    front_box, rear_box = model.body_boxes(sample_state)
                    collision = bool(
                        prepared_obstacles.intersects(front_box)
                        or prepared_obstacles.intersects(rear_box)
                    )
                    if collision:
                        continue
                    collision_free_count += 1
                    clearance = float(
                        min(
                            front_box.distance(obstacle_union),
                            rear_box.distance(obstacle_union),
                        )
                    )
                    if min_clearance <= clearance <= max_clearance:
                        recovery_band_count += 1
    return {
        "reset_geometry_candidate_count": int(sample_count),
        "reset_geometry_collision_free_count": int(collision_free_count),
        "reset_geometry_recovery_band_count": int(recovery_band_count),
    }


def _difficulty_metadata(layout, target_bay, audit):
    clearance_bucket = _bucket_clearance(audit["nominal_target_clearance_m"])
    approach_side_bucket = "left_bay" if float(layout.target_side) > 0.0 else "right_bay"
    return {
        "clearance_bucket": clearance_bucket,
        "approach_side_bucket": approach_side_bucket,
        "scene_complexity_bucket": _scene_complexity_bucket(
            int(audit.get("constructed_obstacle_feature_count", 0)),
        ),
        "difficulty_label": "{}|{}|{}".format(
            clearance_bucket,
            approach_side_bucket,
            "head_in",
        ),
        "bay_goal_alignment_deg": float(
            math.degrees(
                abs(wrap_to_pi(target_bay.goal_heading - target_bay.inward_heading))
            )
        ),
    }


def _scene_complexity_bucket(feature_count):
    feature_count = int(feature_count)
    if feature_count <= 2:
        return "normal"
    if feature_count <= 4:
        return "complex"
    return "extreme"


def _wall_stub(origin, heading, along_center, side, length, depth, corridor_width):
    side = 1.0 if float(side) >= 0.0 else -1.0
    edge = side * 0.5 * float(corridor_width)
    inner = edge - side * float(depth)
    n0, n1 = sorted((edge, inner))
    return _frame_polygon(
        origin,
        heading,
        float(along_center) - 0.5 * float(length),
        float(along_center) + 0.5 * float(length),
        n0,
        n1,
    )


def _equipment_block(origin, heading, along_center, lateral_center, length, width):
    return _frame_polygon(
        origin,
        heading,
        float(along_center) - 0.5 * float(length),
        float(along_center) + 0.5 * float(length),
        float(lateral_center) - 0.5 * float(width),
        float(lateral_center) + 0.5 * float(width),
    )


def _layout_unit_noise(layout, salt):
    token = (
        int(layout.jitter_token)
        + 0x9E3779B9 * (int(layout.obstacle_variant) + 1)
        + 0x85EBCA6B * (int(salt) + 17)
    ) & 0xFFFFFFFF
    token ^= (token >> 16)
    token = (token * 0x7FEB352D) & 0xFFFFFFFF
    token ^= (token >> 15)
    token = (token * 0x846CA68B) & 0xFFFFFFFF
    token ^= (token >> 16)
    return (float(token) / float(0xFFFFFFFF)) * 2.0 - 1.0


def _jittered_dimension(base_value, layout, salt, jitter, lower, upper):
    return float(
        np.clip(
            float(base_value) + _layout_unit_noise(layout, salt) * float(jitter),
            float(lower),
            float(upper),
        )
    )


def _bounded_lateral_center(base_lateral, width, corridor_width):
    limit = max(0.0, 0.5 * float(corridor_width) - 0.5 * float(width) - 0.05)
    if limit <= 0.0:
        return 0.0
    return float(np.clip(float(base_lateral), -limit, limit))


def _target_approach_keepout(layout, corridor_heading, corridor_width, scene_config):
    bay_half = 0.5 * float(scene_config.head_in_bay_width)
    keepout_along = bay_half + float(scene_config.target_approach_keepout_along)
    return _frame_polygon(
        layout.corridor_origin,
        corridor_heading,
        float(layout.target_along) - keepout_along,
        float(layout.target_along) + keepout_along,
        -0.5 * float(corridor_width),
        0.5 * float(corridor_width),
    )


def _constrained_obstacle_features(
    stage,
    layout,
    corridor_polygons,
    parking_bays,
    branches,
    corridor_heading,
    corridor_width,
    branch_width,
    scene_config,
):
    """Build deterministic non-critical obstacles without random rejection loops."""
    desired_by_stage = tuple(scene_config.noncritical_obstacle_count_by_stage)
    desired = int(desired_by_stage[max(0, min(3, int(stage) - 1))])
    free_space = unary_union(
        list(corridor_polygons) + [bay.polygon for bay in parking_bays]
    )
    target_bay = parking_bays[0]
    critical_keepout = unary_union(
        [
            target_bay.polygon.buffer(
                float(scene_config.target_obstacle_keepout),
                cap_style=2,
                join_style=2,
            ),
            _target_approach_keepout(
                layout,
                corridor_heading,
                corridor_width,
                scene_config,
            ),
        ]
    )
    candidates = []
    main_anchors = (-32.0, -24.0, -12.0, 12.0, 24.0, 32.0)
    shift = (int(layout.obstacle_variant) % 3 - 1) * 1.0
    for idx, along in enumerate(main_anchors):
        wall_length = _jittered_dimension(
            scene_config.wall_stub_length,
            layout,
            10 + idx,
            0.70,
            4.20,
            5.80,
        )
        wall_depth = _jittered_dimension(
            scene_config.wall_stub_depth,
            layout,
            30 + idx,
            0.30,
            1.60,
            min(2.35, 0.5 * float(corridor_width) - 0.10),
        )
        wall_along = (
            float(along)
            + shift
            + _layout_unit_noise(layout, 50 + idx) * 1.25
        )
        side = -float(layout.target_side) if idx % 2 == 0 else float(layout.target_side)
        wall = _wall_stub(
            layout.corridor_origin,
            corridor_heading,
            wall_along,
            side,
            wall_length,
            wall_depth,
            corridor_width,
        )
        candidates.append(("wall_stub", wall))
        block_length = _jittered_dimension(
            scene_config.equipment_obstacle_length,
            layout,
            70 + idx,
            0.55,
            2.40,
            3.70,
        )
        block_width = _jittered_dimension(
            scene_config.equipment_obstacle_width,
            layout,
            90 + idx,
            0.35,
            1.60,
            2.35,
        )
        lateral = (
            0.25 * float(corridor_width) * (-1.0 if idx % 2 == 0 else 1.0)
            + _layout_unit_noise(layout, 110 + idx) * 0.35
        )
        lateral = _bounded_lateral_center(lateral, block_width, corridor_width)
        block = _equipment_block(
            layout.corridor_origin,
            corridor_heading,
            float(along) - shift + _layout_unit_noise(layout, 130 + idx) * 0.80,
            lateral,
            block_length,
            block_width,
        )
        candidates.append(("equipment_island", block))

    for branch_index, (branch_origin, branch_heading) in enumerate(branches):
        branch_shift = shift if branch_index % 2 == 0 else -shift
        for local_index, (along, side) in enumerate(
            ((18.0 + branch_shift, -1.0), (-18.0 - branch_shift, 1.0))
        ):
            salt = 170 + branch_index * 20 + local_index
            wall_length = _jittered_dimension(
                scene_config.wall_stub_length,
                layout,
                salt,
                0.65,
                4.20,
                5.80,
            )
            wall_depth = _jittered_dimension(
                scene_config.wall_stub_depth,
                layout,
                salt + 7,
                0.35,
                1.60,
                min(2.50, 0.5 * float(branch_width) - 0.10),
            )
            candidates.append(
                (
                    "branch_wall_stub",
                    _wall_stub(
                        branch_origin,
                        branch_heading,
                        float(along) + _layout_unit_noise(layout, salt + 13) * 1.30,
                        side,
                        wall_length,
                        wall_depth,
                        branch_width,
                    ),
                )
            )
        branch_block_length = _jittered_dimension(
            scene_config.equipment_obstacle_length,
            layout,
            230 + branch_index,
            0.55,
            2.40,
            3.70,
        )
        branch_block_width = _jittered_dimension(
            scene_config.equipment_obstacle_width,
            layout,
            240 + branch_index,
            0.35,
            1.60,
            2.35,
        )
        branch_lateral = _layout_unit_noise(layout, 250 + branch_index) * 0.60
        branch_lateral = _bounded_lateral_center(
            branch_lateral,
            branch_block_width,
            branch_width,
        )
        candidates.append(
            (
                "branch_equipment_island",
                _equipment_block(
                    branch_origin,
                    branch_heading,
                    8.0
                    + branch_shift
                    + _layout_unit_noise(layout, 260 + branch_index) * 1.10,
                    branch_lateral,
                    branch_block_length,
                    branch_block_width,
                ),
            )
        )

    if int(stage) >= 4:
        yard_offsets = ((-4.0, -3.0), (4.0, 3.0), (0.0, 5.0))
        for idx, offset in enumerate(yard_offsets):
            length = _jittered_dimension(
                scene_config.equipment_obstacle_length,
                layout,
                300 + idx,
                0.65,
                2.40,
                3.90,
            )
            width = _jittered_dimension(
                scene_config.equipment_obstacle_width,
                layout,
                320 + idx,
                0.45,
                1.60,
                2.50,
            )
            along_center = (
                2.0
                + float(offset[0])
                + _layout_unit_noise(layout, 340 + idx) * 1.20
            )
            lateral_center = (
                -18.0
                + float(offset[1])
                + _layout_unit_noise(layout, 360 + idx) * 1.20
            )
            candidates.append(
                (
                    "yard_equipment_island",
                    _equipment_block(
                        layout.corridor_origin,
                        corridor_heading,
                        along_center,
                        lateral_center,
                        length,
                        width,
                    ),
                )
            )

    if candidates:
        rotation = int(layout.obstacle_variant) % len(candidates)
        candidates = candidates[rotation:] + candidates[:rotation]

    selected = []
    labels = []

    def candidate_is_valid(candidate):
        if not free_space.covers(candidate):
            return False
        if candidate.intersects(critical_keepout):
            return False
        if any(candidate.intersects(bay.polygon.buffer(0.25)) for bay in parking_bays):
            return False
        if any(
            candidate.distance(existing) < float(scene_config.noncritical_obstacle_spacing)
            for existing in selected
        ):
            return False
        return True

    def try_add(label, candidate):
        if len(selected) >= desired:
            return False
        if not candidate_is_valid(candidate):
            return False
        selected.append(candidate)
        labels.append(label)
        return True

    if int(stage) >= 4:
        category_predicates = (
            lambda label: label == "wall_stub",
            lambda label: label == "equipment_island",
            lambda label: label.startswith("branch_"),
            lambda label: label == "yard_equipment_island",
        )
        for predicate in category_predicates:
            for label, candidate in candidates:
                if predicate(label) and try_add(label, candidate):
                    break

    for label, candidate in candidates:
        try_add(label, candidate)
        if len(selected) >= desired:
            break
    return selected, labels


def _corridor_polygons(stage, layout, scene_config):
    corridor_width = scene_config.corridor_width_for_stage(stage)
    branch_width = max(
        11.0,
        corridor_width * float(scene_config.branch_width_ratio),
    )
    corridor_heading = float(layout.corridor_heading)
    origin = tuple(layout.corridor_origin)
    axis, normal = _cardinal_frame(corridor_heading)
    main_half_length = 0.5 * float(scene_config.main_corridor_length)
    main = _frame_polygon(
        origin,
        corridor_heading,
        -main_half_length,
        main_half_length,
        -0.5 * corridor_width,
        0.5 * corridor_width,
    )
    polygons = [main]
    branch_specs = []

    if stage >= 2:
        branch_origin = np.asarray(origin) + axis * float(layout.first_branch_along)
        branch_heading = wrap_to_pi(corridor_heading + 0.5 * math.pi)
        branch = _frame_polygon(
            branch_origin,
            branch_heading,
            -0.5 * scene_config.branch_corridor_length,
            0.5 * scene_config.branch_corridor_length,
            -0.5 * branch_width,
            0.5 * branch_width,
        )
        polygons.append(branch)
        branch_specs.append((branch_origin, branch_heading))
    if stage >= 3:
        branch_origin = np.asarray(origin) + axis * float(layout.second_branch_along)
        branch_heading = wrap_to_pi(corridor_heading + 0.5 * math.pi)
        branch = _frame_polygon(
            branch_origin,
            branch_heading,
            -0.5 * scene_config.branch_corridor_length,
            6.0,
            -0.5 * branch_width,
            0.5 * branch_width,
        )
        polygons.append(branch)
        branch_specs.append((branch_origin, branch_heading))
    if stage >= 4:
        turning_yard_center = np.asarray(origin) + axis * 2.0 - normal * 18.0
        yard = box(
            turning_yard_center[0] - 9.0,
            turning_yard_center[1] - 9.0,
            turning_yard_center[0] + 9.0,
            turning_yard_center[1] + 9.0,
        )
        polygons.append(yard)
    return polygons, corridor_heading, corridor_width, branch_width, branch_specs


def _scene_type_from_config(scene_config):
    config = DEFAULT_SCENE_CONFIG if scene_config is None else scene_config
    return normalize_scene_type(getattr(config, "scene_type", "default"))


def _rng_from_seed(seed, salt):
    seed_sequence = np.random.SeedSequence(
        [int(seed) & 0xFFFFFFFF, int(salt) & 0xFFFFFFFF]
    )
    return np.random.default_rng(seed_sequence)


def _clip_index(index, count):
    return int(np.clip(int(index), 0, max(0, int(count) - 1)))


def _noise_tuple(values, length):
    values = tuple(float(item) for item in values)
    if len(values) >= length:
        return values[:length]
    return values + (0.0,) * (length - len(values))


def _state_from_front_pose(x_front, y_front, theta_front, phi=0.0):
    theta_front = float(wrap_to_pi(theta_front))
    phi = float(np.clip(float(phi), -DEFAULT_VEHICLE_PARAMS.phi_max, DEFAULT_VEHICLE_PARAMS.phi_max))
    return ArticulatedState(
        x_front=float(x_front),
        y_front=float(y_front),
        theta_front=theta_front,
        theta_rear=float(wrap_to_pi(theta_front - phi)),
    )


def _vehicle_pose_collision_free(x_front, y_front, theta_front, phi, prepared_obstacles):
    model = ArticulatedVehicleModel(DEFAULT_VEHICLE_PARAMS)
    state = _state_from_front_pose(x_front, y_front, theta_front, phi)
    front_box, rear_box = model.body_boxes(state)
    collision = bool(
        prepared_obstacles.intersects(front_box)
        or prepared_obstacles.intersects(rear_box)
    )
    return not collision, state, front_box, rear_box


def _rasterize_obstacle_polygons(obstacle_polygons, world_bounds, resolution):
    xmin, ymin, xmax, ymax = tuple(float(value) for value in world_bounds)
    resolution = float(resolution)
    width = int(math.ceil((xmax - xmin) / resolution))
    height = int(math.ceil((ymax - ymin) / resolution))
    occupancy = np.zeros((height, width), dtype=np.uint8)
    for polygon in obstacle_polygons:
        bx0, by0, bx1, by1 = polygon.bounds
        col0 = max(0, int(math.floor((bx0 - xmin) / resolution)))
        col1 = min(width, int(math.ceil((bx1 - xmin) / resolution)))
        row0 = max(0, int(math.floor((by0 - ymin) / resolution)))
        row1 = min(height, int(math.ceil((by1 - ymin) / resolution)))
        for row in range(row0, row1):
            cell_y0 = ymin + row * resolution
            cell_y1 = cell_y0 + resolution
            for col in range(col0, col1):
                cell_x0 = xmin + col * resolution
                cell_x1 = cell_x0 + resolution
                if polygon.intersects(box(cell_x0, cell_y0, cell_x1, cell_y1)):
                    occupancy[row, col] = 1
    return occupancy


def _build_scene_from_rule_polygons(
    obstacle_polygons,
    parking_bays,
    target_bay,
    slot,
    world_bounds,
    resolution,
    metadata,
):
    obstacle_polygons = list(obstacle_polygons)
    obstacle_union = unary_union(obstacle_polygons)
    prepared_obstacles = prep(obstacle_union)
    occupancy = _rasterize_obstacle_polygons(
        obstacle_polygons,
        world_bounds,
        resolution,
    )
    metadata = dict(metadata)
    metadata.setdefault("obstacle_count", len(obstacle_polygons))
    metadata.setdefault("grid_width", int(occupancy.shape[1]))
    metadata.setdefault("grid_height", int(occupancy.shape[0]))
    metadata.setdefault("world_bounds", tuple(world_bounds))
    metadata.setdefault("resolution", float(resolution))
    return MixingPlantScene(
        occupancy_grid=occupancy,
        obstacle_polygons=obstacle_polygons,
        obstacle_edges=_edges_from_polygons(obstacle_polygons),
        parking_bays=list(parking_bays),
        target_bay=target_bay,
        slot=slot,
        world_bounds=tuple(float(value) for value in world_bounds),
        resolution=float(resolution),
        metadata=metadata,
        obstacle_union=obstacle_union,
        prepared_obstacles=prepared_obstacles,
    )


def _local_box(x0, x1, y0, y1, offset):
    ox, oy = offset
    return box(
        float(x0) + float(ox),
        float(y0) + float(oy),
        float(x1) + float(ox),
        float(y1) + float(oy),
    )


def _local_point(point, offset):
    ox, oy = offset
    return (float(point[0]) + float(ox), float(point[1]) + float(oy))


def _bay_inner_bounds(index, bay_width, bay_depth, wall_thickness):
    index = int(index)
    bay_width = float(bay_width)
    wall_thickness = float(wall_thickness)
    x0 = index * bay_width
    x1 = (index + 1) * bay_width
    if index > 0:
        x0 += 0.5 * wall_thickness
    x1 -= 0.5 * wall_thickness
    return x0, x1, 0.0, float(bay_depth)


def _build_rule_parking_bay(index, target_index, config, offset):
    bay_width = float(config.bay_width)
    bay_depth = float(config.bay_depth)
    wall_thickness = float(config.wall_thickness)
    x0, x1, y0, y1 = _bay_inner_bounds(index, bay_width, bay_depth, wall_thickness)
    polygon = _local_box(x0, x1, y0, y1, offset)
    mouth_a = _local_point((x0, bay_depth), offset)
    mouth_b = _local_point((x1, bay_depth), offset)
    mouth_center = _local_point((0.5 * (x0 + x1), bay_depth), offset)
    return ParkingBay(
        name="bay_{}".format(int(index)),
        polygon=polygon,
        mouth_center=mouth_center,
        mouth_segment=(mouth_a, mouth_b),
        inward_heading=float(-0.5 * math.pi),
        corridor_heading=0.0,
        goal_orientation_mode="head_in",
        goal_heading=float(-0.5 * math.pi),
        is_target=bool(int(index) == int(target_index)),
    )


def _target_bay_index(config, rng, bay_count, stage):
    mode = str(getattr(config, "target_bay_sampling_mode", "uniform"))
    if mode == "fixed":
        return _clip_index(getattr(config, "fixed_target_bay_index", 0), bay_count)
    if mode == "curriculum":
        if int(stage) <= 2:
            return _clip_index((int(bay_count) - 1) // 2, bay_count)
        return int(rng.integers(0, int(bay_count)))
    if mode == "uniform":
        return int(rng.integers(0, int(bay_count)))
    raise ValueError("unsupported target_bay_sampling_mode '{}'".format(mode))


def _valid_mixing_target_pose(x, y, theta, phi, target_bay, prepared_obstacles):
    valid, state, front_box, rear_box = _vehicle_pose_collision_free(
        x,
        y,
        theta,
        phi,
        prepared_obstacles,
    )
    if not valid:
        return False, state
    if not target_bay.polygon.covers(front_box):
        return False, state
    if not target_bay.polygon.covers(rear_box):
        return False, state
    heading_into_bay = math.cos(float(wrap_to_pi(theta - target_bay.inward_heading))) > 0.995
    return bool(heading_into_bay), state


def _sample_mixing_target_pose(config, rng, target_bay, prepared_obstacles):
    params = DEFAULT_VEHICLE_PARAMS
    x0, y0, x1, y1 = target_bay.polygon.bounds
    x_margin = 0.5 * params.front_body_width + 0.25
    y_min = y0 + 0.5 * params.front_body_length + float(config.parking_head_wall_clearance)
    y_max = y1 - (0.5 * params.front_body_length + params.rear_body_length) - 0.35
    if y_max < y_min:
        y_min = y0 + 0.5 * params.front_body_length + 0.10
        y_max = max(y_min, y1 - (0.5 * params.front_body_length + params.rear_body_length) - 0.10)
    base_x = 0.5 * (x0 + x1)
    base_y = float(np.clip(y_min, y0 + 0.5 * params.front_body_length + 0.10, y_max))
    base_theta = float(target_bay.goal_heading)
    noise = _noise_tuple(getattr(config, "target_pose_noise", (0.0, 0.0, 0.0, 0.0)), 4)
    max_attempts = max(1, int(getattr(config, "max_pose_sampling_attempts", 32)))
    for attempt in range(max_attempts):
        if attempt == 0:
            dx = dy = dtheta = dphi = 0.0
        else:
            dx = float(rng.uniform(-noise[0], noise[0]))
            dy = float(rng.uniform(-noise[1], noise[1]))
            dtheta = float(rng.uniform(-noise[2], noise[2]))
            dphi = float(rng.uniform(-noise[3], noise[3]))
        x = float(np.clip(base_x + dx, x0 + x_margin, x1 - x_margin))
        y = float(np.clip(base_y + dy, y_min, y_max))
        theta = float(wrap_to_pi(base_theta + dtheta))
        valid, state = _valid_mixing_target_pose(
            x,
            y,
            theta,
            dphi,
            target_bay,
            prepared_obstacles,
        )
        if valid:
            return state, attempt + 1
    raise RuntimeError("failed to sample collision-free mixing-station target pose")


def _mixing_initial_candidate_specs(config, target_state, target_bay_index, offset):
    bay_count = max(1, int(config.bay_count))
    bay_width = float(config.bay_width)
    bay_depth = float(config.bay_depth)
    corridor_width = float(config.corridor_width)
    wall_thickness = float(config.wall_thickness)
    requested_spawn_mode = str(getattr(config, "initial_spawn_mode", "mixed"))
    spawn_mode = "corridor" if requested_spawn_mode in ("bay", "mixed") else requested_spawn_mode
    heading_mode = str(getattr(config, "corridor_initial_heading_mode", "mixed"))
    fixed_initial = _clip_index(getattr(config, "fixed_initial_bay_index", 0), bay_count)

    specs = []
    bay_indices = [fixed_initial] if spawn_mode == "bay" else list(range(bay_count))
    if spawn_mode in ("bay", "mixed"):
        for bay_index in bay_indices:
            x0, x1, _, _ = _bay_inner_bounds(
                bay_index,
                bay_width,
                bay_depth,
                wall_thickness,
            )
            x = 0.5 * (x0 + x1)
            y = max(
                0.5 * DEFAULT_VEHICLE_PARAMS.front_body_length + 0.35,
                bay_depth - (0.5 * DEFAULT_VEHICLE_PARAMS.front_body_length + DEFAULT_VEHICLE_PARAMS.rear_body_length) - 0.50,
            )
            gx, gy = _local_point((x, y), offset)
            specs.append(("bay", int(bay_index), gx, gy, -0.5 * math.pi))
    if spawn_mode in ("corridor", "mixed"):
        headings = []
        if heading_mode in ("along_corridor", "mixed"):
            headings.extend((0.0, math.pi))
        if heading_mode in ("face_bay", "mixed"):
            headings.append(-0.5 * math.pi)
        if not headings:
            raise ValueError(
                "unsupported corridor_initial_heading_mode '{}'".format(heading_mode)
            )
        y = bay_depth + 0.5 * corridor_width
        x_values = [0.5 * bay_width + index * bay_width for index in range(bay_count)]
        x_values.extend((0.25 * bay_width, (bay_count - 0.25) * bay_width))
        for x in x_values:
            for heading in headings:
                gx, gy = _local_point((x, y), offset)
                specs.append(("corridor", -1, float(gx), float(gy), float(heading)))
    if not specs:
        raise ValueError("unsupported initial_spawn_mode '{}'".format(spawn_mode))

    min_sep = float(getattr(config, "min_initial_target_separation", 0.0))
    max_sep = float(getattr(config, "max_initial_target_separation", 1e6))
    filtered = []
    target_xy = np.asarray((target_state.x_front, target_state.y_front), dtype=np.float64)
    for spec in specs:
        xy = np.asarray((spec[2], spec[3]), dtype=np.float64)
        distance = float(np.linalg.norm(xy - target_xy))
        if min_sep <= distance <= max_sep:
            filtered.append(spec)
    return tuple(filtered if filtered else specs)


def generate_mixing_station_bay_corridor_scene(
    stage=1,
    seed=0,
    scene_config=DEFAULT_SCENE_CONFIG,
    task_family="head_in",
):
    stage = int(np.clip(stage, 1, 4))
    config = DEFAULT_SCENE_CONFIG if scene_config is None else scene_config
    task_family = normalize_task_family(task_family)
    rng = _rng_from_seed(seed, 0xBACC0)
    bay_count = max(2, int(config.bay_count))
    bay_width = float(config.bay_width)
    bay_depth = float(config.bay_depth)
    corridor_width = float(config.corridor_width)
    wall_thickness = float(config.wall_thickness)
    total_width = bay_count * bay_width
    total_depth = bay_depth + corridor_width
    offset = (-0.5 * total_width, -0.5 * total_depth)

    walls = []
    wall_labels = []
    walls.append(_local_box(0.0, total_width, -wall_thickness, 0.0, offset))
    wall_labels.append("bottom_wall")
    walls.append(_local_box(0.0, total_width, total_depth, total_depth + wall_thickness, offset))
    wall_labels.append("corridor_outer_wall")
    walls.append(_local_box(-wall_thickness, 0.0, -wall_thickness, bay_depth, offset))
    wall_labels.append("left_side_wall")
    walls.append(_local_box(total_width, total_width + wall_thickness, -wall_thickness, bay_depth, offset))
    wall_labels.append("right_side_wall")
    for wall_index in range(1, bay_count):
        center_x = wall_index * bay_width
        walls.append(
            _local_box(
                center_x - 0.5 * wall_thickness,
                center_x + 0.5 * wall_thickness,
                0.0,
                bay_depth,
                offset,
            )
        )
        wall_labels.append("partition_wall")

    world_margin_x = float(getattr(config, "world_margin_x", config.world_margin))
    world_margin_y = float(config.world_margin)
    world_bounds = (
        offset[0] - world_margin_x - wall_thickness,
        offset[1] - world_margin_y - wall_thickness,
        offset[0] + total_width + world_margin_x + wall_thickness,
        offset[1] + total_depth + world_margin_y + wall_thickness,
    )
    target_index = _target_bay_index(config, rng, bay_count, stage)
    parking_bays = [
        _build_rule_parking_bay(index, target_index, config, offset)
        for index in range(bay_count)
    ]
    target_bay = parking_bays[target_index]
    obstacle_union = unary_union(walls)
    prepared_obstacles = prep(obstacle_union)
    target_state, target_attempts = _sample_mixing_target_pose(
        config,
        rng,
        target_bay,
        prepared_obstacles,
    )
    slot = DirectedParkingSlot(
        x_goal=float(target_state.x_front),
        y_goal=float(target_state.y_front),
        theta_goal=float(target_state.theta_front),
        front_body_length=DEFAULT_VEHICLE_PARAMS.front_body_length,
        front_body_width=DEFAULT_VEHICLE_PARAMS.front_body_width,
    )
    initial_specs = _mixing_initial_candidate_specs(
        config,
        target_state,
        target_index,
        offset,
    )
    target_audit = _target_feasibility_audit(
        target_bay=target_bay,
        slot=slot,
        obstacle_union=obstacle_union,
        prepared_obstacles=prepared_obstacles,
    )
    reset_audit = _reset_geometry_audit(
        slot=slot,
        obstacle_union=obstacle_union,
        prepared_obstacles=prepared_obstacles,
    )
    corridor_region = _local_box(0.0, total_width, bay_depth, total_depth, offset)
    metadata = {
        "scene_type": "mixing_station_bay_corridor",
        "stage": int(stage),
        "seed": int(seed),
        "task_family": task_family,
        "generation_mode": "rule_bay_corridor_walls",
        "goal_type": "parking_bay",
        "goal_orientation_mode": "head_in",
        "bay_count": int(bay_count),
        "bay_width": float(bay_width),
        "bay_depth": float(bay_depth),
        "corridor_width": float(corridor_width),
        "wall_thickness": float(wall_thickness),
        "target_bay_index": int(target_index),
        "target_bay_name": target_bay.name,
        "bay_inward_heading": float(target_bay.inward_heading),
        "initial_pose_candidates": tuple(initial_specs),
        "initial_spawn_mode": "corridor",
        "requested_initial_spawn_mode": str(getattr(config, "initial_spawn_mode", "mixed")),
        "corridor_initial_heading_mode": str(
            getattr(config, "corridor_initial_heading_mode", "mixed")
        ),
        "min_initial_target_separation": float(
            getattr(config, "min_initial_target_separation", 0.0)
        ),
        "max_initial_target_separation": float(
            getattr(config, "max_initial_target_separation", 1e6)
        ),
        "target_pose_sampling_attempts": int(target_attempts),
        "target_pose_noise": tuple(
            float(v) for v in _noise_tuple(getattr(config, "target_pose_noise", (0.0, 0.0, 0.0, 0.0)), 4)
        ),
        "initial_pose_noise": tuple(
            float(v) for v in _noise_tuple(getattr(config, "initial_pose_noise", (0.0, 0.0, 0.0, 0.0)), 4)
        ),
        "max_pose_sampling_attempts": int(
            getattr(config, "max_pose_sampling_attempts", 32)
        ),
        "ensure_feasible_reset": bool(getattr(config, "ensure_feasible_reset", True)),
        "corridor_outer_wall_exists": True,
        "partition_wall_count": int(bay_count - 1),
        "side_wall_count": 2,
        "corridor_end_wall_count": 0,
        "corridor_ends_open": True,
        "bottom_wall_exists": True,
        "constructed_obstacle_labels": tuple(wall_labels),
        "constructed_obstacle_feature_count": 0,
        "constructed_wall_feature_count": int(len(walls)),
        "obstacle_count": int(len(walls)),
        "corridor_region_bounds": tuple(float(v) for v in corridor_region.bounds),
        "target_heading_into_bay": True,
        "free_ratio": 0.0,
        "scene_generation_attempts": 1,
        "scene_generation_attempt_count": 1,
    }
    metadata.update(target_audit)
    metadata.update(reset_audit)
    metadata.update(
        {
            "clearance_bucket": _bucket_clearance(
                target_audit["nominal_target_clearance_m"]
            ),
            "approach_side_bucket": "multi_bay",
            "scene_complexity_bucket": _scene_complexity_bucket(len(walls)),
            "difficulty_label": "mixing_station_bay_corridor|bay{}".format(
                target_index
            ),
            "bay_goal_alignment_deg": 0.0,
        }
    )
    scene = _build_scene_from_rule_polygons(
        obstacle_polygons=walls,
        parking_bays=parking_bays,
        target_bay=target_bay,
        slot=slot,
        world_bounds=world_bounds,
        resolution=float(config.resolution),
        metadata=metadata,
    )
    scene.metadata["free_ratio"] = float(np.mean(scene.occupancy_grid == 0))
    return scene


def _region_grid_points(region, resolution):
    x0, y0, x1, y1 = tuple(float(value) for value in region)
    resolution = max(0.25, float(resolution))
    xs = np.arange(x0, x1 + 1e-6, resolution, dtype=np.float64)
    ys = np.arange(y0, y1 + 1e-6, resolution, dtype=np.float64)
    return [(float(x), float(y)) for x in xs for y in ys]


def _sample_region_pose(region, rng, noise, theta):
    x0, y0, x1, y1 = tuple(float(value) for value in region)
    x = float(rng.uniform(x0, x1))
    y = float(rng.uniform(y0, y1))
    dx, dy, dtheta, dphi = _noise_tuple(noise, 4)
    if dx > 0.0:
        x = float(np.clip(x + rng.uniform(-dx, dx), x0, x1))
    if dy > 0.0:
        y = float(np.clip(y + rng.uniform(-dy, dy), y0, y1))
    if dtheta > 0.0:
        theta = float(wrap_to_pi(theta + rng.uniform(-dtheta, dtheta)))
    phi = float(rng.uniform(-dphi, dphi)) if dphi > 0.0 else 0.0
    return x, y, theta, phi


def _loading_truck_pose(config, rng, target_x, target_y, target_heading):
    forward = np.asarray(
        [math.cos(float(target_heading)), math.sin(float(target_heading))],
        dtype=np.float64,
    )
    lateral = np.asarray([-forward[1], forward[0]], dtype=np.float64)
    lateral_bounds = tuple(
        float(item) for item in getattr(config, "truck_lateral_offset_range", (0.0, 0.0))
    )
    if len(lateral_bounds) < 2:
        lateral_bounds = (0.0, 0.0)
    lateral_offset = float(rng.uniform(lateral_bounds[0], lateral_bounds[1]))
    center = (
        np.asarray((target_x, target_y), dtype=np.float64)
        + forward * float(config.truck_offset_ahead_of_target)
        + lateral * lateral_offset
    )
    mode = str(getattr(config, "truck_heading_mode", "perpendicular_to_target"))
    if mode == "perpendicular_to_target":
        heading = float(wrap_to_pi(float(target_heading) + 0.5 * math.pi))
    elif mode == "fixed":
        heading = 0.5 * math.pi
    elif mode == "random_small_noise":
        heading = float(wrap_to_pi(float(target_heading) + 0.5 * math.pi + rng.uniform(-0.10, 0.10)))
    else:
        raise ValueError("unsupported truck_heading_mode '{}'".format(mode))
    return (float(center[0]), float(center[1])), heading, lateral_offset


def _loading_initial_candidate_specs(config, rng, target_state, prepared_obstacles):
    initial_region = tuple(float(v) for v in config.initial_pose_sampling_region)
    resolution = min(
        2.0,
        max(0.75, float(getattr(config, "obstacle_candidate_grid_resolution", 3.0))),
    )
    points = _region_grid_points(initial_region, resolution)
    if points:
        order = list(rng.permutation(len(points)))
    else:
        order = []
    x0, y0, x1, y1 = initial_region
    center_point = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
    ordered_points = [center_point]
    ordered_points.extend(points[int(index)] for index in order)
    target_heading = float(target_state.theta_front)
    heading_offsets = (
        0.0,
        math.radians(12.0),
        -math.radians(12.0),
        math.radians(25.0),
        -math.radians(25.0),
    )
    phi_values = (0.0,)
    min_sep = float(getattr(config, "min_initial_target_separation", 0.0))
    max_sep = float(getattr(config, "max_initial_target_separation", 1e6))
    target_xy = np.asarray(
        (target_state.x_front, target_state.y_front),
        dtype=np.float64,
    )
    max_candidates = max(1, int(getattr(config, "max_pose_sampling_attempts", 32)))
    specs = []
    seen = set()
    for point in ordered_points:
        point_xy = np.asarray(point, dtype=np.float64)
        distance = float(np.linalg.norm(point_xy - target_xy))
        if not (min_sep <= distance <= max_sep):
            continue
        for heading_offset in heading_offsets:
            heading = float(wrap_to_pi(target_heading + heading_offset))
            for phi in phi_values:
                key = (
                    round(float(point[0]), 3),
                    round(float(point[1]), 3),
                    round(heading, 3),
                    round(float(phi), 3),
                )
                if key in seen:
                    continue
                valid, _, _, _ = _vehicle_pose_collision_free(
                    float(point[0]),
                    float(point[1]),
                    heading,
                    float(phi),
                    prepared_obstacles,
                )
                if not valid:
                    continue
                seen.add(key)
                specs.append(
                    (
                        "rectangle_space",
                        -1,
                        float(point[0]),
                        float(point[1]),
                        heading,
                        float(phi),
                    )
                )
                if len(specs) >= max_candidates:
                    return tuple(specs)
    return tuple(specs)


def _boundary_walls(world_length, world_width, wall_thickness):
    half_l = 0.5 * float(world_length)
    half_w = 0.5 * float(world_width)
    t = float(wall_thickness)
    return [
        box(-half_l - t, -half_w - t, half_l + t, -half_w),
        box(-half_l - t, half_w, half_l + t, half_w + t),
        box(-half_l - t, -half_w, -half_l, half_w),
        box(half_l, -half_w, half_l + t, half_w),
    ]


def _loading_pose_pair(config, rng, boundary_obstacles):
    max_attempts = max(1, int(getattr(config, "max_pose_sampling_attempts", 32)))
    target_region = tuple(float(v) for v in config.target_pose_sampling_region)
    target_noise = getattr(config, "target_pose_noise", (0.0, 0.0, 0.0, 0.0))

    last_reason = "no_candidate"
    for attempt in range(max_attempts):
        target_x, target_y, target_heading, target_phi = _sample_region_pose(
            target_region,
            rng,
            target_noise,
            0.0,
        )
        truck_center, truck_heading, truck_lateral = _loading_truck_pose(
            config,
            rng,
            target_x,
            target_y,
            target_heading,
        )
        truck_polygon = oriented_box(
            truck_center,
            truck_heading,
            float(config.truck_length),
            float(config.truck_width),
        )
        obstacles = list(boundary_obstacles) + [truck_polygon]
        prepared = prep(unary_union(obstacles))
        valid_target, target_state, _, _ = _vehicle_pose_collision_free(
            target_x,
            target_y,
            target_heading,
            target_phi,
            prepared,
        )
        if not valid_target:
            last_reason = "target_collision"
            continue
        initial_candidates = _loading_initial_candidate_specs(
            config,
            rng,
            target_state,
            prepared,
        )
        if not initial_candidates:
            last_reason = "initial_collision"
            continue
        first = initial_candidates[0]
        initial_state = ArticulatedState(
            x_front=float(first[2]),
            y_front=float(first[3]),
            theta_front=float(first[4]),
            theta_rear=float(wrap_to_pi(float(first[4]) - float(first[5]))),
        )
        return {
            "target_state": target_state,
            "initial_state": initial_state,
            "initial_candidates": initial_candidates,
            "truck_polygon": truck_polygon,
            "truck_center": truck_center,
            "truck_heading": truck_heading,
            "truck_lateral_offset": truck_lateral,
            "attempts": int(attempt + 1),
        }
    raise RuntimeError(
        "failed to sample loading-truck target/initial poses: {}".format(last_reason)
    )


def _discrete_obstacle_polygon(config, rng, center):
    size_min, size_max = tuple(float(v) for v in config.discrete_obstacle_size_range)
    size_min = max(0.10, size_min)
    size_max = max(size_min, size_max)
    shape_mode = str(getattr(config, "discrete_obstacle_shape", "mixed"))
    if shape_mode == "mixed":
        shape_mode = "rectangle" if bool(rng.integers(0, 2)) else "circle"
    if shape_mode == "rectangle":
        length = float(rng.uniform(size_min, size_max))
        width = float(rng.uniform(size_min, size_max))
        heading = float(rng.choice((0.0, 0.5 * math.pi)))
        return oriented_box(center, heading, length, width), "rectangle"
    if shape_mode == "circle":
        radius = 0.5 * float(rng.uniform(size_min, size_max))
        return Point(float(center[0]), float(center[1])).buffer(radius, resolution=8), "circle"
    raise ValueError("unsupported discrete_obstacle_shape '{}'".format(shape_mode))


def _generate_discrete_loading_obstacles(config, rng, initial_states, target_state, truck_polygon):
    half_l = 0.5 * float(config.world_length)
    half_w = 0.5 * float(config.world_width)
    boundary_margin = float(config.boundary_wall_thickness) + 0.75
    interior = box(
        -half_l + boundary_margin,
        -half_w + boundary_margin,
        half_l - boundary_margin,
        half_w - boundary_margin,
    )
    resolution = float(getattr(config, "obstacle_candidate_grid_resolution", 3.0))
    points = _region_grid_points(interior.bounds, resolution)
    order = rng.permutation(len(points)) if points else []
    desired = max(0, int(getattr(config, "discrete_obstacle_count", 0)))
    max_checks = max(1, int(getattr(config, "max_obstacle_sampling_attempts", 64)))
    selected = []
    labels = []
    if isinstance(initial_states, ArticulatedState):
        initial_states = (initial_states,)
    initial_xys = tuple(
        np.asarray((state.x_front, state.y_front), dtype=np.float64)
        for state in initial_states
    )
    target_xy = np.asarray((target_state.x_front, target_state.y_front), dtype=np.float64)
    target_exclusion = Point(float(target_xy[0]), float(target_xy[1])).buffer(
        float(getattr(config, "obstacle_exclusion_radius_around_target", 0.0)),
        resolution=16,
    )
    truck_exclusion = truck_polygon.buffer(
        float(getattr(config, "obstacle_exclusion_radius_around_truck", 0.0)),
        cap_style=2,
        join_style=2,
    )
    checks = 0
    for point_index in order:
        if len(selected) >= desired or checks >= max_checks:
            break
        checks += 1
        point = points[int(point_index)]
        point_xy = np.asarray(point, dtype=np.float64)
        if any(
            np.linalg.norm(point_xy - initial_xy)
            < float(config.obstacle_exclusion_radius_around_initial)
            for initial_xy in initial_xys
        ):
            continue
        if target_exclusion.contains(Point(point)):
            continue
        if truck_exclusion.contains(Point(point)):
            continue
        candidate, label = _discrete_obstacle_polygon(config, rng, point)
        if not interior.covers(candidate):
            continue
        if candidate.intersects(target_exclusion):
            continue
        if candidate.intersects(truck_exclusion):
            continue
        if any(
            candidate.distance(existing) < float(config.obstacle_min_pairwise_distance)
            for existing in selected
        ):
            continue
        selected.append(candidate)
        labels.append(label)
    return selected, labels, checks, len(points)


def _loading_target_bay(slot):
    params = DEFAULT_VEHICLE_PARAMS
    axis = np.asarray(
        [math.cos(float(slot.theta_goal)), math.sin(float(slot.theta_goal))],
        dtype=np.float64,
    )
    center = np.asarray(slot.center, dtype=np.float64) - 0.5 * params.rear_body_length * axis
    polygon = oriented_box(
        center,
        slot.theta_goal,
        params.overall_length + 1.0,
        params.overall_width + 1.0,
    )
    front = np.asarray(slot.center, dtype=np.float64) + axis * (0.5 * params.front_body_length)
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    half_width = 0.5 * (params.overall_width + 1.0)
    mouth_a = front - normal * half_width
    mouth_b = front + normal * half_width
    return ParkingBay(
        name="loading_target_zone",
        polygon=polygon,
        mouth_center=(float(front[0]), float(front[1])),
        mouth_segment=((float(mouth_a[0]), float(mouth_a[1])), (float(mouth_b[0]), float(mouth_b[1]))),
        inward_heading=float(slot.theta_goal),
        corridor_heading=float(slot.theta_goal),
        goal_orientation_mode="head_in",
        goal_heading=float(slot.theta_goal),
        is_target=True,
    )


def generate_loading_truck_rectangle_space_scene(
    stage=1,
    seed=0,
    scene_config=DEFAULT_SCENE_CONFIG,
    task_family="head_in",
):
    stage = int(np.clip(stage, 1, 4))
    config = DEFAULT_SCENE_CONFIG if scene_config is None else scene_config
    task_family = normalize_task_family(task_family)
    rng = _rng_from_seed(seed, 0x710AD)
    boundary_walls = []
    pair = _loading_pose_pair(config, rng, boundary_walls)
    target_state = pair["target_state"]
    initial_candidates = tuple(pair["initial_candidates"])
    truck_polygon = pair["truck_polygon"]
    initial_states = tuple(
        ArticulatedState(
            x_front=float(candidate[2]),
            y_front=float(candidate[3]),
            theta_front=float(candidate[4]),
            theta_rear=float(wrap_to_pi(float(candidate[4]) - float(candidate[5]))),
        )
        for candidate in initial_candidates
    )
    discrete_obstacles, obstacle_labels, obstacle_checks, candidate_count = (
        _generate_discrete_loading_obstacles(
            config,
            rng,
            initial_states,
            target_state,
            truck_polygon,
        )
    )
    obstacle_polygons = [truck_polygon] + list(discrete_obstacles)
    obstacle_union = unary_union(obstacle_polygons)
    prepared_obstacles = prep(obstacle_union)
    slot = DirectedParkingSlot(
        x_goal=float(target_state.x_front),
        y_goal=float(target_state.y_front),
        theta_goal=float(target_state.theta_front),
        front_body_length=DEFAULT_VEHICLE_PARAMS.front_body_length,
        front_body_width=DEFAULT_VEHICLE_PARAMS.front_body_width,
    )
    target_bay = _loading_target_bay(slot)
    target_audit = _target_feasibility_audit(
        target_bay=target_bay,
        slot=slot,
        obstacle_union=obstacle_union,
        prepared_obstacles=prepared_obstacles,
    )
    reset_audit = _reset_geometry_audit(
        slot=slot,
        obstacle_union=obstacle_union,
        prepared_obstacles=prepared_obstacles,
    )
    forward = np.asarray(
        [math.cos(float(target_state.theta_front)), math.sin(float(target_state.theta_front))],
        dtype=np.float64,
    )
    target_xy = np.asarray((target_state.x_front, target_state.y_front), dtype=np.float64)
    truck_delta = np.asarray(pair["truck_center"], dtype=np.float64) - target_xy
    truck_ahead = float(np.dot(truck_delta, forward)) > 0.0
    truck_perpendicular = abs(
        abs(math.cos(wrap_to_pi(pair["truck_heading"] - target_state.theta_front)))
    ) < 0.10
    half_l = 0.5 * float(config.world_length)
    half_w = 0.5 * float(config.world_width)
    margin = float(getattr(config, "world_margin", 4.0))
    t = float(config.boundary_wall_thickness)
    world_bounds = (-half_l - margin - t, -half_w - margin - t, half_l + margin + t, half_w + margin + t)
    metadata = {
        "scene_type": "loading_truck_rectangle_space",
        "stage": int(stage),
        "seed": int(seed),
        "task_family": task_family,
        "generation_mode": "rule_rectangle_truck_grid_obstacles",
        "goal_type": "loading_truck_target",
        "goal_orientation_mode": "head_in",
        "world_length": float(config.world_length),
        "world_width": float(config.world_width),
        "boundary_wall_thickness": float(config.boundary_wall_thickness),
        "boundary_wall_count": 0,
        "truck_length": float(config.truck_length),
        "truck_width": float(config.truck_width),
        "truck_center": tuple(float(v) for v in pair["truck_center"]),
        "truck_heading": float(pair["truck_heading"]),
        "truck_lateral_offset": float(pair["truck_lateral_offset"]),
        "truck_in_front": bool(truck_ahead),
        "truck_perpendicular": bool(truck_perpendicular),
        "initial_pose_candidates": initial_candidates,
        "initial_candidate_count": int(len(initial_candidates)),
        "initial_spawn_mode": "rectangle_space",
        "initial_spawn_region": "rectangle_space",
        "initial_bay_index": -1,
        "min_initial_target_separation": float(
            getattr(config, "min_initial_target_separation", 0.0)
        ),
        "max_initial_target_separation": float(
            getattr(config, "max_initial_target_separation", 1e6)
        ),
        "target_pose_sampling_attempts": int(pair["attempts"]),
        "target_pose_noise": tuple(
            float(v) for v in _noise_tuple(getattr(config, "target_pose_noise", (0.0, 0.0, 0.0, 0.0)), 4)
        ),
        "initial_pose_noise": tuple(
            float(v) for v in _noise_tuple(getattr(config, "initial_pose_noise", (0.0, 0.0, 0.0, 0.0)), 4)
        ),
        "max_pose_sampling_attempts": int(
            getattr(config, "max_pose_sampling_attempts", 32)
        ),
        "ensure_feasible_reset": bool(getattr(config, "ensure_feasible_reset", True)),
        "discrete_obstacle_count_requested": int(config.discrete_obstacle_count),
        "discrete_obstacle_count": int(len(discrete_obstacles)),
        "discrete_obstacle_shape": str(getattr(config, "discrete_obstacle_shape", "mixed")),
        "discrete_obstacle_labels": tuple(obstacle_labels),
        "obstacle_candidate_count": int(candidate_count),
        "obstacle_sampling_checks": int(obstacle_checks),
        "constructed_obstacle_feature_count": int(1 + len(discrete_obstacles)),
        "constructed_wall_feature_count": 0,
        "constructed_obstacle_labels": tuple(["truck_obstacle"] + obstacle_labels),
        "obstacle_exclusion_radius_around_initial": float(
            config.obstacle_exclusion_radius_around_initial
        ),
        "obstacle_exclusion_radius_around_target": float(
            config.obstacle_exclusion_radius_around_target
        ),
        "obstacle_exclusion_radius_around_truck": float(
            config.obstacle_exclusion_radius_around_truck
        ),
        "obstacle_min_pairwise_distance": float(config.obstacle_min_pairwise_distance),
        "obstacle_exclusion_valid": True,
        "scene_generation_attempts": 1,
        "scene_generation_attempt_count": 1,
    }
    metadata.update(target_audit)
    metadata.update(reset_audit)
    metadata.update(
        {
            "clearance_bucket": _bucket_clearance(
                target_audit["nominal_target_clearance_m"]
            ),
            "approach_side_bucket": "loading_truck",
            "scene_complexity_bucket": _scene_complexity_bucket(
                1 + len(discrete_obstacles)
            ),
            "difficulty_label": "loading_truck_rectangle_space|obstacles{}".format(
                len(discrete_obstacles)
            ),
            "bay_goal_alignment_deg": 0.0,
        }
    )
    scene = _build_scene_from_rule_polygons(
        obstacle_polygons=obstacle_polygons,
        parking_bays=[target_bay],
        target_bay=target_bay,
        slot=slot,
        world_bounds=world_bounds,
        resolution=float(config.resolution),
        metadata=metadata,
    )
    scene.metadata["free_ratio"] = float(np.mean(scene.occupancy_grid == 0))
    return scene


@lru_cache(maxsize=256)
def generate_cached_mixing_plant_scene(
    stage=1,
    seed=0,
    scene_config=DEFAULT_SCENE_CONFIG,
    task_family="head_in",
):
    """Build a cached 80 m mixing plant with explicit corridors and bays."""
    stage = int(np.clip(stage, 1, 4))
    config = DEFAULT_SCENE_CONFIG if scene_config is None else scene_config
    task_family = normalize_task_family(task_family)
    scene_type = _scene_type_from_config(config)
    if scene_type == "mixing_station_bay_corridor":
        return generate_mixing_station_bay_corridor_scene(
            stage=stage,
            seed=seed,
            scene_config=config,
            task_family=task_family,
        )
    if scene_type == "loading_truck_rectangle_space":
        return generate_loading_truck_rectangle_space_scene(
            stage=stage,
            seed=seed,
            scene_config=config,
            task_family=task_family,
        )
    layout = _sample_layout(seed, config, task_family)
    half_extent = float(config.world_half_extent)
    resolution = float(config.resolution)
    grid_size = int(round(2.0 * half_extent / resolution))
    world_min = -half_extent
    occupancy = np.ones((grid_size, grid_size), dtype=np.uint8)

    corridor_polygons, corridor_heading, corridor_width, branch_width, branches = (
        _corridor_polygons(stage, layout, config)
    )
    for corridor in corridor_polygons:
        _carve_world_polygon(occupancy, corridor, world_min, resolution)

    target_bay = _build_bay(
        name="target_bay",
        corridor_origin=layout.corridor_origin,
        corridor_heading=corridor_heading,
        corridor_width=corridor_width,
        along_center=layout.target_along,
        side=layout.target_side,
        mode=family_to_goal_mode(task_family),
        scene_config=config,
        is_target=True,
    )
    parking_bays = [target_bay]

    if stage >= 2:
        parking_bays.append(
            _build_bay(
                name="secondary_main_bay",
                corridor_origin=layout.corridor_origin,
                corridor_heading=corridor_heading,
                corridor_width=corridor_width,
                along_center=-22.0 if layout.target_along >= 0.0 else 22.0,
                side=-layout.target_side,
                mode="head_in",
                scene_config=config,
            )
        )
    if stage >= 3 and branches:
        branch_origin, branch_heading = branches[0]
        branch_bay_candidates = (
            layout.branch_bay_along,
            float(config.branch_bay_along_range[0]),
            float(config.branch_bay_along_range[1]),
        )
        for along_center in branch_bay_candidates:
            candidate = _build_bay(
                name="secondary_branch_bay",
                corridor_origin=branch_origin,
                corridor_heading=branch_heading,
                corridor_width=branch_width,
                along_center=along_center,
                side=-1.0,
                mode="head_in",
                scene_config=config,
            )
            if all(
                candidate.polygon.intersection(existing.polygon).area <= 1e-8
                for existing in parking_bays
            ):
                parking_bays.append(candidate)
                break

    for bay in parking_bays:
        _carve_world_polygon(occupancy, bay.polygon, world_min, resolution)

    constructed_obstacles, constructed_labels = _constrained_obstacle_features(
        stage=stage,
        layout=layout,
        corridor_polygons=corridor_polygons,
        parking_bays=parking_bays,
        branches=branches,
        corridor_heading=corridor_heading,
        corridor_width=corridor_width,
        branch_width=branch_width,
        scene_config=config,
    )
    for obstacle in constructed_obstacles:
        _occupy_world_polygon(occupancy, obstacle, world_min, resolution)

    obstacle_polygons = _merge_occupied_cells(
        occupancy,
        (world_min, world_min),
        resolution,
    )
    obstacle_union = unary_union(obstacle_polygons)
    prepared_obstacles = prep(obstacle_union)
    goal_center = _goal_center_in_bay(target_bay, config)
    slot = DirectedParkingSlot(
        x_goal=float(goal_center[0]),
        y_goal=float(goal_center[1]),
        theta_goal=float(target_bay.goal_heading),
        front_body_length=DEFAULT_VEHICLE_PARAMS.front_body_length,
        front_body_width=DEFAULT_VEHICLE_PARAMS.front_body_width,
    )
    audit = _target_feasibility_audit(
        target_bay=target_bay,
        slot=slot,
        obstacle_union=obstacle_union,
        prepared_obstacles=prepared_obstacles,
    )
    audit.update(
        _reset_geometry_audit(
            slot=slot,
            obstacle_union=obstacle_union,
            prepared_obstacles=prepared_obstacles,
        )
    )
    audit["constructed_obstacle_feature_count"] = int(len(constructed_obstacles))
    audit["constructed_wall_feature_count"] = int(
        sum(1 for label in constructed_labels if "wall" in label)
    )
    free_ratio = float(np.mean(occupancy == 0))
    metadata = {
        "scene_type": "cached_rule_carved_mixing_plant",
        "stage": stage,
        "seed": int(seed),
        "task_family": task_family,
        "generation_mode": "blocked_grid_then_constructive_corridor_and_bay_carve",
        "family_sampling_mode": "explicit_family_then_derived_scene_seed",
        "world_bounds": (-half_extent, -half_extent, half_extent, half_extent),
        "grid_width": grid_size,
        "grid_height": grid_size,
        "resolution": resolution,
        "corridor_heading": float(corridor_heading),
        "corridor_origin": tuple(layout.corridor_origin),
        "corridor_width": float(corridor_width),
        "branch_width": float(branch_width),
        "target_bay_along": float(layout.target_along),
        "target_bay_side": float(layout.target_side),
        "parking_bay_count": len(parking_bays),
        "target_bay_name": target_bay.name,
        "goal_type": "parking_bay",
        "goal_orientation_mode": target_bay.goal_orientation_mode,
        "obstacle_layout_variant": int(layout.obstacle_variant),
        "constructed_obstacle_labels": tuple(constructed_labels),
        "bay_inward_heading": float(target_bay.inward_heading),
        "free_ratio": free_ratio,
        "obstacle_count": len(obstacle_polygons),
    }
    metadata.update(audit)
    metadata.update(_difficulty_metadata(layout, target_bay, audit))
    return MixingPlantScene(
        occupancy_grid=occupancy,
        obstacle_polygons=obstacle_polygons,
        obstacle_edges=_edges_from_polygons(obstacle_polygons),
        parking_bays=parking_bays,
        target_bay=target_bay,
        slot=slot,
        world_bounds=(-half_extent, -half_extent, half_extent, half_extent),
        resolution=resolution,
        metadata=metadata,
        obstacle_union=obstacle_union,
        prepared_obstacles=prepared_obstacles,
    )


class CachedScenePool:
    def __init__(
        self,
        stage=1,
        pool_size=16,
        base_seed=0,
        scene_config=DEFAULT_SCENE_CONFIG,
        family_schedule=TASK_FAMILIES,
        scene_type_schedule=None,
        validate_scene_audit=True,
    ):
        self.stage = int(stage)
        requested_pool_size = max(1, int(pool_size))
        self.base_seed = int(base_seed)
        self.scene_config = DEFAULT_SCENE_CONFIG if scene_config is None else scene_config
        self.family_schedule = normalize_family_schedule(family_schedule)
        if scene_type_schedule is None:
            scene_type_schedule = (
                _scene_type_from_config(self.scene_config),
            )
        self.scene_type_schedule = normalize_scene_type_schedule(scene_type_schedule)
        schedule_size = len(self.family_schedule) * len(self.scene_type_schedule)
        if schedule_size > 1:
            remainder = requested_pool_size % schedule_size
            if remainder:
                requested_pool_size += schedule_size - remainder
        self.pool_size = requested_pool_size
        self.validate_scene_audit = bool(validate_scene_audit)
        self._scenes = []
        self._next_replacement_index = self.pool_size
        for index in range(self.pool_size):
            self._scenes.append(self._generate_scene_with_retries(index))

    def __len__(self):
        return len(self._scenes)

    def _scene_type_for_index(self, pool_index):
        family_count = max(1, len(self.family_schedule))
        scene_index = (int(pool_index) // family_count) % len(self.scene_type_schedule)
        return self.scene_type_schedule[scene_index]

    def _scene_config_for_type(self, scene_type):
        scene_type = normalize_scene_type(scene_type)
        if _scene_type_from_config(self.scene_config) == scene_type:
            return self.scene_config
        return replace(self.scene_config, scene_type=scene_type)

    def _generate_scene(self, pool_index, task_family=None, scene_type=None):
        pool_index = int(pool_index)
        if task_family is None:
            task_family = self.family_schedule[pool_index % len(self.family_schedule)]
        task_family = normalize_task_family(task_family)
        if scene_type is None:
            scene_type = self._scene_type_for_index(pool_index)
        scene_type = normalize_scene_type(scene_type)
        scene_seed = derive_scene_seed(
            base_seed=self.base_seed,
            pool_index=pool_index,
            task_family=task_family,
            stage=self.stage,
        )
        scene = generate_cached_mixing_plant_scene(
            stage=self.stage,
            seed=scene_seed,
            scene_config=self._scene_config_for_type(scene_type),
            task_family=task_family,
        )
        scene.metadata["requested_scene_type"] = scene_type
        scene.metadata["scene_type_schedule_size"] = int(len(self.scene_type_schedule))
        if self.validate_scene_audit:
            if bool(scene.metadata.get("nominal_target_collision", True)):
                raise RuntimeError(
                    "scene audit failed: target collision for seed {} family {}".format(
                        scene_seed,
                        task_family,
                    )
                )
            if int(scene.metadata.get("success_neighborhood_feasible_count", 0)) <= 0:
                raise RuntimeError(
                    "scene audit failed: empty success neighborhood for seed {} family {}".format(
                        scene_seed,
                        task_family,
                    )
                )
            if self.stage >= 4 and int(
                scene.metadata.get("reset_geometry_recovery_band_count", 0)
            ) <= 0:
                raise RuntimeError(
                    "scene audit failed: empty reset recovery band for seed {} family {}".format(
                        scene_seed,
                        task_family,
                    )
                )
        return scene

    def _generate_scene_with_retries(
        self,
        pool_index,
        task_family=None,
        scene_type=None,
        max_attempts=8,
    ):
        if task_family is None:
            task_family = self.family_schedule[
                int(pool_index) % len(self.family_schedule)
            ]
        task_family = normalize_task_family(task_family)
        if scene_type is None:
            scene_type = self._scene_type_for_index(pool_index)
        scene_type = normalize_scene_type(scene_type)
        last_error = None
        for attempt in range(max(1, int(max_attempts))):
            if attempt == 0:
                candidate_index = int(pool_index)
            else:
                candidate_index = self._next_replacement_index
                self._next_replacement_index += 1
            try:
                scene = self._generate_scene(
                    candidate_index,
                    task_family=task_family,
                    scene_type=scene_type,
                )
            except _RETRYABLE_SCENE_GENERATION_ERRORS as exc:
                last_error = exc
                continue
            scene.metadata["scene_generation_attempt_count"] = int(attempt + 1)
            scene.metadata["scene_generation_attempts"] = int(attempt + 1)
            return scene
        raise RuntimeError(
            "failed to generate valid scene family {} type {} after {} attempts: {}".format(
                task_family,
                scene_type,
                max(1, int(max_attempts)),
                last_error,
            )
        )

    def get(self, episode_index):
        return self._scenes[int(episode_index) % len(self._scenes)]

    def replace(self, episode_index, task_family=None, scene_type=None, max_attempts=16):
        scene_slot = int(episode_index) % len(self._scenes)
        if task_family is None:
            task_family = self._scenes[scene_slot].metadata.get("task_family")
        task_family = normalize_task_family(task_family)
        if scene_type is None:
            scene_type = self._scenes[scene_slot].metadata.get(
                "requested_scene_type",
                self._scenes[scene_slot].metadata.get("scene_type"),
            )
        scene_type = normalize_scene_type(scene_type)
        last_error = None
        for attempt_index in range(1, max(1, int(max_attempts)) + 1):
            replacement_index = self._next_replacement_index
            self._next_replacement_index += 1
            try:
                scene = self._generate_scene(
                    replacement_index,
                    task_family=task_family,
                    scene_type=scene_type,
                )
            except _RETRYABLE_SCENE_GENERATION_ERRORS as exc:
                last_error = exc
                continue
            scene.metadata["scene_generation_attempt_count"] = int(attempt_index)
            self._scenes[scene_slot] = scene
            return scene
        raise RuntimeError(
            "failed to replace scene slot {} family {} type {} after {} attempts: {}".format(
                scene_slot,
                task_family,
                scene_type,
                max(1, int(max_attempts)),
                last_error,
            )
        )
