from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep

from config import DEFAULT_SCENE_CONFIG, DEFAULT_VEHICLE_PARAMS, MixingPlantSceneConfig
from env.geometry import DirectedParkingSlot, wrap_to_pi


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
    target_mode: str
    target_side: float
    target_along: float
    parallel_reverse: bool
    first_branch_along: float
    second_branch_along: float
    branch_bay_along: float


def _quantize(value, resolution):
    return float(round(float(value) / float(resolution)) * float(resolution))


def _sample_layout(seed, scene_config):
    """Map every seed to a reproducible constructive layout."""
    seed = int(seed)
    rng = np.random.default_rng(np.random.SeedSequence(seed & 0xFFFFFFFF))
    resolution = float(scene_config.resolution)
    corridor_heading = 0.0 if (seed // 2) % 2 == 0 else 0.5 * math.pi
    axis, normal = _cardinal_frame(corridor_heading)
    jitter = float(scene_config.main_origin_jitter)
    along_shift = _quantize(rng.uniform(-jitter, jitter), resolution)
    normal_shift = _quantize(rng.uniform(-jitter, jitter), resolution)
    origin = axis * along_shift + normal * normal_shift
    target_mode = "head_in" if seed % 2 == 0 else "parallel"
    target_side = 1.0 if (seed // 4) % 2 == 0 else -1.0

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
        target_mode=target_mode,
        target_side=target_side,
        target_along=target_along,
        parallel_reverse=bool((seed // 8) % 2),
        first_branch_along=float(branch_positions[0]),
        second_branch_along=float(branch_positions[1]),
        branch_bay_along=sample_range(scene_config.branch_bay_along_range),
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
    parallel_reverse=False,
):
    axis, normal = _cardinal_frame(corridor_heading)
    side = 1.0 if float(side) >= 0.0 else -1.0
    if mode == "head_in":
        along_length = float(scene_config.head_in_bay_width)
        depth = float(scene_config.head_in_bay_depth)
    elif mode == "parallel":
        along_length = float(scene_config.parallel_bay_length)
        depth = float(scene_config.parallel_bay_depth)
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

    if mode == "head_in":
        goal_heading = inward_heading
    else:
        goal_heading = wrap_to_pi(
            float(corridor_heading) + (math.pi if parallel_reverse else 0.0)
        )
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


def _corridor_polygons(stage, layout, scene_config):
    corridor_width = scene_config.corridor_width(stage)
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


@lru_cache(maxsize=64)
def generate_cached_mixing_plant_scene(
    stage=1,
    seed=0,
    scene_config=DEFAULT_SCENE_CONFIG,
):
    """Build a cached 80 m mixing plant with explicit corridors and bays."""
    stage = int(np.clip(stage, 1, 4))
    config = scene_config
    layout = _sample_layout(seed, config)
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
        mode=layout.target_mode,
        scene_config=config,
        is_target=True,
        parallel_reverse=layout.parallel_reverse,
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
                mode="parallel",
                scene_config=config,
                parallel_reverse=not layout.parallel_reverse,
            )
            if all(
                candidate.polygon.intersection(existing.polygon).area <= 1e-8
                for existing in parking_bays
            ):
                parking_bays.append(candidate)
                break

    for bay in parking_bays:
        _carve_world_polygon(occupancy, bay.polygon, world_min, resolution)

    obstacle_polygons = _merge_occupied_cells(
        occupancy,
        (world_min, world_min),
        resolution,
    )
    obstacle_union = unary_union(obstacle_polygons)
    goal_center = _goal_center_in_bay(target_bay, config)
    slot = DirectedParkingSlot(
        x_goal=float(goal_center[0]),
        y_goal=float(goal_center[1]),
        theta_goal=float(target_bay.goal_heading),
        front_body_length=DEFAULT_VEHICLE_PARAMS.front_body_length,
        front_body_width=DEFAULT_VEHICLE_PARAMS.front_body_width,
    )
    free_ratio = float(np.mean(occupancy == 0))
    metadata = {
        "scene_type": "cached_rule_carved_mixing_plant",
        "stage": stage,
        "seed": int(seed),
        "generation_mode": "blocked_grid_then_constructive_corridor_and_bay_carve",
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
        "parallel_reverse": bool(layout.parallel_reverse),
        "parking_bay_count": len(parking_bays),
        "target_bay_name": target_bay.name,
        "goal_type": "parking_bay",
        "goal_orientation_mode": target_bay.goal_orientation_mode,
        "bay_inward_heading": float(target_bay.inward_heading),
        "free_ratio": free_ratio,
        "obstacle_count": len(obstacle_polygons),
    }
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
        prepared_obstacles=prep(obstacle_union),
    )


class CachedScenePool:
    def __init__(
        self,
        stage=1,
        pool_size=16,
        base_seed=0,
        scene_config=DEFAULT_SCENE_CONFIG,
    ):
        self.stage = int(stage)
        self.pool_size = max(1, int(pool_size))
        self.base_seed = int(base_seed)
        self.scene_config = scene_config
        self._scenes = [
            generate_cached_mixing_plant_scene(
                self.stage,
                self.base_seed + index,
                self.scene_config,
            )
            for index in range(self.pool_size)
        ]

    def get(self, episode_index):
        return self._scenes[int(episode_index) % len(self._scenes)]
