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
    dt: float = 0.5
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
    """Constructive corridor, wall, obstacle, and bay scene geometry in meters."""

    scene_type: str = "default"
    world_half_extent: float = 40.0
    resolution: float = 1.0
    boundary_margin: float = 4.0
    main_corridor_length: float = 72.0
    branch_corridor_length: float = 56.0
    corridor_width_by_stage: Tuple[float, float, float, float] = (
        10.0,
        8.0,
        5.0,
        5.0,
    )
    branch_width_ratio: float = 0.85
    head_in_bay_width: float = 8.0
    head_in_bay_depth: float = 10.0
    parking_head_wall_clearance: float = 1.0
    main_origin_jitter: float = 2.0
    target_bay_along_range: Tuple[float, float] = (-18.0, 18.0)
    branch_anchor_positions: Tuple[float, ...] = (-24.0, -12.0, 12.0, 24.0)
    target_branch_clearance: float = 17.0
    branch_to_branch_clearance: float = 11.0
    branch_bay_along_range: Tuple[float, float] = (-16.0, 16.0)
    target_obstacle_keepout: float = 6.0
    target_approach_keepout_along: float = 7.0
    noncritical_obstacle_count_by_stage: Tuple[int, ...] = (
        4,
        5,
        6,
        7,
    )
    noncritical_obstacle_spacing: float = 2.0
    wall_stub_length: float = 5.0
    wall_stub_depth: float = 2.0
    equipment_obstacle_length: float = 3.0
    equipment_obstacle_width: float = 2.0

    # --- Rule-based mixing-station bay corridor scene ---
    bay_count: int = 5
    bay_width: float = 8.0
    bay_depth: float = 10.0
    wall_thickness: float = 0.50
    corridor_width: float = 7.0
    world_margin: float = 4.0
    world_margin_x: float = 14.5
    initial_spawn_mode: str = "mixed"
    corridor_initial_heading_mode: str = "mixed"
    target_bay_sampling_mode: str = "uniform"
    fixed_target_bay_index: int = 0
    fixed_initial_bay_index: int = 0
    min_initial_target_separation: float = 8.0
    max_initial_target_separation: float = 30.0
    target_pose_noise: Tuple[float, float, float, float] = (
        0.25,
        0.25,
        math.radians(3.0),
        0.0,
    )
    initial_pose_noise: Tuple[float, float, float, float] = (
        0.50,
        0.40,
        math.radians(8.0),
        math.radians(5.0),
    )
    ensure_feasible_reset: bool = True
    max_pose_sampling_attempts: int = 32

    # --- Rule-based loading-truck rectangle scene ---
    world_length: float = 42.0
    world_width: float = 30.0
    boundary_wall_thickness: float = 0.50
    target_pose_sampling_region: Tuple[float, float, float, float] = (
        0.0,
        -4.0,
        8.0,
        4.0,
    )
    initial_pose_sampling_region: Tuple[float, float, float, float] = (
        -16.0,
        -6.0,
        -8.0,
        6.0,
    )
    truck_length: float = 9.0
    truck_width: float = 2.8
    truck_offset_ahead_of_target: float = 5.0
    truck_lateral_offset_range: Tuple[float, float] = (-1.0, 1.0)
    truck_heading_mode: str = "perpendicular_to_target"
    discrete_obstacle_count: int = 7
    discrete_obstacle_shape: str = "mixed"
    discrete_obstacle_size_range: Tuple[float, float] = (1.25, 3.0)
    obstacle_exclusion_radius_around_initial: float = 8.0
    obstacle_exclusion_radius_around_target: float = 8.0
    obstacle_exclusion_radius_around_truck: float = 4.0
    obstacle_min_pairwise_distance: float = 3.0
    obstacle_candidate_grid_resolution: float = 3.0
    max_obstacle_sampling_attempts: int = 64

    def corridor_width_for_stage(self, stage: int) -> float:
        index = max(0, min(3, int(stage) - 1))
        return float(self.corridor_width_by_stage[index])


@dataclass(frozen=True)
class LocalParkingEnvConfig:
    max_steps: int = 600
    collision_tolerance: float = 1e-8
    articulation_tolerance: float = math.radians(1.0)
    success_overlap: float = 0.80
    success_heading_error: float = math.radians(15.0)
    curriculum_stage: int = 1
    scene_pool_size: int = 18
    scene_family_schedule: Tuple[str, ...] = (
        "head_in",
    )
    use_hybrid_astar: bool = False
    initial_sampling_attempts: int = 128
    reset_scene_retry_count: int = 18
    reset_min_mask_safe_ratio: float = 1e-3
    stage4_reset_min_mask_safe_ratio: float = 0.125
    stage4_reset_min_body_clearance: float = 0.15
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
    w_dist: float = 0.01
    w_heading: float = 0.3
    w_time: float = 0.1
    w_hybrid: float = 4.0

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

    # --- Near-goal Reeds-Shepp potential ---
    rs_potential_enabled: bool = True
    rs_potential_d_rs: float = 15.0
    rs_potential_k: int = 2
    rs_potential_coef: float = 0.5
    rs_potential_clip: float = 1.0

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
    enable_mask_floor_fallback: bool = False
    mask_degenerate_eps: float = 1e-3
    mask_floor_value: float = 0.01
    apply_floor_only_when_all_zero: bool = False

    # --- Strict-mask local DWA deadlock recovery ---
    enable_dwa_recovery: bool = False
    dwa_override_policy_action: bool = True
    dwa_enable_deadlock_termination: bool = False
    dwa_all_zero_eps: float = 1e-3
    dwa_low_safe_ratio: float = 0.12
    dwa_unlock_safe_ratio: float = 0.08
    dwa_forced_stop_patience: int = 3
    dwa_no_progress_patience: int = 3
    dwa_deadlock_patience: int = 8
    dwa_horizon_steps: int = 4
    dwa_speed_ratios: Tuple[float, ...] = (0.25, 0.5, 1.0)

    # --- Training-only HOPE teacher guidance ---
    enable_hope_teacher: bool = False
    hope_code_dir: str = "../HOPE"
    hope_weight_path: str = "../HOPE/src/model/ckpt/HOPE_PPO.pt"
    hope_cache_dir: str = "runs/hope_teacher_cache"
    use_teacher_reward: bool = False
    guide_weight_initial: float = 0.5
    guide_weight_final: float = 0.0
    guide_anneal_start_episode: int = 0
    guide_anneal_end_episode: int = 10_000
    guide_dropout_initial: float = 0.0
    guide_dropout_final: float = 0.8
    teacher_corridor_width: float = 2.5
    teacher_anchor_weight: float = 0.4
    teacher_heading_weight: float = 0.2
    teacher_progress_weight: float = 1.0
    teacher_gear_weight: float = 0.15
    teacher_reward_clip: float = 1.0
    enable_offpath_reset: bool = False
    enable_failure_aggregation: bool = False
    no_guide_eval_interval: int = 0

    # --- Training-only hard-case replay from actual failed rollouts ---
    hard_case_replay_enabled: bool = False
    hard_case_replay_ratio: float = 0.20
    hard_case_replay_capacity: int = 4096
    hard_case_replay_tail_steps: int = 12
    hard_case_replay_attempts: int = 32
    hard_case_replay_xy_std: float = 0.45
    hard_case_replay_heading_std_deg: float = 6.0
    hard_case_replay_phi_std_deg: float = 5.0


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_steps: int = 4096
    ppo_epochs: int = 6
    batch_size: int = 256
    target_kl: float = 0.03
    kl_early_stop_multiplier: float = 1.5
    log_std_init: float = -0.7
    log_std_min: float = -2.5
    log_std_max: float = -0.3
    policy_loss_weight_head_in: float = 1.0
    checkpoint_score_weight_head_in: float = 1.0


DEFAULT_VEHICLE_PARAMS = ZL50GNVehicleParams()
DEFAULT_MASK_CONFIG = ActionMaskConfig()
DEFAULT_SCENE_CONFIG = MixingPlantSceneConfig()
DEFAULT_ENV_CONFIG = LocalParkingEnvConfig()
DEFAULT_PPO_CONFIG = PPOConfig()
