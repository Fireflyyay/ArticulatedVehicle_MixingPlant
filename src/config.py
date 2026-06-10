from dataclasses import asdict, dataclass
import math
from typing import Dict, Tuple


@dataclass(frozen=True)
class ZL50GNVehicleParams:
    """Single source of truth for the ZL50GN-style articulated loader."""

    overall_length: float = 8.165
    overall_width: float = 3.016
    overall_height: float = 3.485
    bucket_width: float = 3.016
    operating_mass_kg: float = 17_500.0
    rated_load_kg: float = 5_000.0
    max_traction_kn: float = 165.0
    max_traction_tolerance_kn: float = 5.0
    max_lift_kn: float = 170.0
    minimum_turning_radius: float = 6.400
    ground_clearance: float = 0.450
    forward_gear_1_max_kmh: float = 11.5
    forward_gear_2_max_kmh: float = 38.0
    reverse_max_kmh: float = 16.5

    # TODO(vehicle-calibration): the source table labels both 3.300 m and
    # 2.250 m as "wheelbase". Confirm whether these are total wheelbase,
    # axle-to-articulation distances, or front/rear body geometry dimensions.
    source_wheelbase_primary: float = 3.300
    source_wheelbase_secondary: float = 2.250

    # Geometry used by this local model. The two boxes meet at the hinge when
    # phi=0 and sum to the published overall length.
    front_body_length: float = 4.450
    rear_body_length: float = 3.715
    front_body_width: float = 3.016
    rear_body_width: float = 3.016

    # Parking control deliberately stays far below the published travel speeds.
    parking_v_forward_max: float = 1.5
    parking_v_reverse_max: float = 1.2
    phi_max: float = math.radians(35.0)
    phi_dot_max: float = 0.5
    dt: float = 0.2
    integration_substeps: int = 8

    lidar_beams: int = 54
    lidar_range: float = 20.0

    @property
    def front_center_to_hinge(self) -> float:
        return 0.5 * self.front_body_length

    @property
    def rear_center_to_hinge(self) -> float:
        return 0.5 * self.rear_body_length

    @property
    def parking_v_max(self) -> float:
        return max(self.parking_v_forward_max, self.parking_v_reverse_max)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ActionMaskConfig:
    n_phi_dot_bins: int = 11
    n_speed_bins: int = 8
    n_phi_state_bins: int = 13
    safety_margin: float = 0.10
    min_safe_ratio: float = 1e-3
    table_horizon_steps: int = 1


@dataclass(frozen=True)
class MixingPlantSceneConfig:
    """Constructive corridor-and-bay scene geometry in meters."""

    world_half_extent: float = 40.0
    resolution: float = 1.0
    boundary_margin: float = 4.0
    main_corridor_length: float = 72.0
    branch_corridor_length: float = 56.0
    corridor_width_by_stage: Tuple[float, float, float, float] = (
        14.0,
        12.0,
        10.0,
        8.0,
    )
    branch_width_ratio: float = 0.85
    head_in_bay_width: float = 6.0
    head_in_bay_depth: float = 9.0
    parallel_bay_length: float = 6.0
    parallel_bay_depth: float = 9.0
    parking_head_wall_clearance: float = 1.0
    main_origin_jitter: float = 2.0
    target_bay_along_range: Tuple[float, float] = (-18.0, 18.0)
    branch_anchor_positions: Tuple[float, ...] = (-24.0, -12.0, 12.0, 24.0)
    target_branch_clearance: float = 17.0
    branch_to_branch_clearance: float = 11.0
    branch_bay_along_range: Tuple[float, float] = (-16.0, 16.0)

    def corridor_width(self, stage: int) -> float:
        index = max(0, min(3, int(stage) - 1))
        return float(self.corridor_width_by_stage[index])


@dataclass(frozen=True)
class LocalParkingEnvConfig:
    max_steps: int = 400
    collision_tolerance: float = 1e-8
    articulation_tolerance: float = math.radians(1.0)
    success_overlap: float = 0.80
    success_heading_error: float = math.radians(10.0)
    curriculum_stage: int = 1
    scene_pool_size: int = 16
    use_hybrid_astar: bool = True
    initial_sampling_attempts: int = 128
    stage_distance_ranges: Tuple[Tuple[float, float], ...] = (
        (8.0, 15.0),
        (10.0, 18.0),
        (8.0, 20.0),
        (4.0, 20.0),
    )
    stage_lateral_ranges: Tuple[float, ...] = (2.5, 4.0, 4.5, 4.5)
    stage_heading_ranges_deg: Tuple[float, ...] = (45.0, 90.0, 120.0, 50.0)
    stage_phi_ranges_deg: Tuple[float, ...] = (12.0, 20.0, 30.0, 30.0)
    poor_pose_min_heading_deg: float = 35.0
    poor_pose_min_lateral: float = 1.5
    poor_pose_min_abs_phi_deg: float = 15.0
    recovery_min_abs_phi_deg: float = 18.0
    recovery_max_body_clearance: float = 0.80
    recovery_max_lidar_distance: float = 2.20

    success_reward: float = 5.0
    failure_reward: float = -5.0
    distance_d_min: float = 1.0
    w_iou: float = 1.0
    w_dist: float = 0.05
    w_heading: float = 0.3
    w_time: float = 0.1
    w_hybrid: float = 3.0

    # --- Planner potential oracle (Hybrid A* → PBRS) ---
    planner_position_tolerance: float = 0.8
    planner_heading_tolerance_deg: float = 15.0

    # PBRS core:  Φ(s) = -J(s) / planner_cost_scale
    # reward = planner_potential_coef * (gamma^tau * Φ' - Φ)
    planner_cost_scale: float = 25.0
    planner_potential_coef: float = 0.5
    planner_potential_clip: float = 1.0
    planner_max_cost: float = 80.0

    # J_state = remaining_path_length
    #   + planner_lateral_residual_weight * clip(lateral, planner_lateral_clip)
    #   + planner_goal_heading_weight * abs(wrap_to_pi(theta - theta_goal))
    planner_lateral_residual_weight: float = 0.5
    planner_goal_heading_weight: float = 1.0
    planner_lateral_clip: float = 5.0

    # Fallback cost when planner fails (pose-based J)
    planner_fallback_position_weight: float = 1.0
    planner_fallback_heading_weight: float = 2.0
    planner_failure_bias: float = 3.0

    # --- Safe-speed-ratio action parameterization ---
    gear_deadband: float = 0.10
    mask_cost_stop_weight: float = 0.5
    mask_cost_abs_weight: float = 0.15
    mask_cost_rel_weight: float = 0.10
    mask_cost_rel_delta: float = 0.05
    mask_cost_clip_weight: float = 0.05
    mask_cost_safe_threshold: float = 0.15
    mask_cost_max: float = 3.0
    mask_cost_coef_final: float = 0.8


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_steps: int = 2048
    ppo_epochs: int = 8
    batch_size: int = 128


DEFAULT_VEHICLE_PARAMS = ZL50GNVehicleParams()
DEFAULT_MASK_CONFIG = ActionMaskConfig()
DEFAULT_SCENE_CONFIG = MixingPlantSceneConfig()
DEFAULT_ENV_CONFIG = LocalParkingEnvConfig()
DEFAULT_PPO_CONFIG = PPOConfig()
