import math
import os
from typing import Optional

import numpy as np

from config import (
    DEFAULT_ENV_CONFIG,
    DEFAULT_SCENE_CONFIG,
    DEFAULT_VEHICLE_PARAMS,
    LocalParkingEnvConfig,
)
from env.articulated_action_mask import (
    FORWARD_GEAR,
    REVERSE_GEAR,
    STOP_GEAR,
    ArticulatedActionMask,
)
from env.dwa_recovery import DWARecoveryController, DWAResult
from env.geometry import overlap_ratio, wrap_to_pi
from env.hybrid_astar_reward import OptionalHybridAStarReward
from env.lidar import DualBodyLidar
from env.mixing_plant_scene import CachedScenePool, RULE_SCENE_TYPES, TASK_FAMILIES
from env.reward import LocalParkingReward
from env.rs_potential import RSPotentialOracle, RSPotentialPlanner
from env.vehicle import ArticulatedState, ArticulatedVehicleModel
from teachers.hope_adapter import HopeTeacherAdapter


class BoxSpace:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = low
        self.high = high
        self.shape = tuple(shape)
        self.dtype = dtype

    def sample(self, rng=None):
        generator = np.random.default_rng() if rng is None else rng
        return generator.uniform(self.low, self.high, size=self.shape).astype(self.dtype)


class ResetInitialStateError(RuntimeError):
    pass


class LocalParkingEnv:
    SLOT_FEATURE_DIM = 13
    VEHICLE_FEATURE_DIM = 6
    LIDAR_FEATURE_DIM = 108
    MASK_FEATURE_DIM = 22
    OBS_DIM = 149
    OBS_SLICES = {
        "slot": slice(0, 13),
        "vehicle": slice(13, 19),
        "lidar": slice(19, 127),
        "mask": slice(127, 149),
    }

    def __init__(
        self,
        config=DEFAULT_ENV_CONFIG,
        vehicle_params=DEFAULT_VEHICLE_PARAMS,
        action_mask=None,
        action_mask_path=None,
        hybrid_planner=None,
        rs_planner=None,
        hope_teacher=None,
        scene_config=DEFAULT_SCENE_CONFIG,
        seed=0,
        multi_stage_pool=None,
    ):
        self.config = config
        self.vehicle_params = vehicle_params
        self.rng = np.random.default_rng(int(seed))
        self.vehicle_model = ArticulatedVehicleModel(vehicle_params)
        self.lidar = DualBodyLidar(vehicle_params)
        self.action_mask = (
            action_mask
            if action_mask is not None
            else ArticulatedActionMask.load(action_mask_path, vehicle_params=vehicle_params)
        )
        self.dwa_recovery = (
            DWARecoveryController(config)
            if bool(getattr(config, "enable_dwa_recovery", False))
            else None
        )
        if self.action_mask.feature_dim != self.MASK_FEATURE_DIM:
            raise ValueError("LocalParkingEnv currently requires 11 phi_dot mask bins")
        self._multi_pool = multi_stage_pool
        self._active_stage = int(config.curriculum_stage)
        if multi_stage_pool is not None:
            self.scene_pool = multi_stage_pool.pool_for(self._active_stage)
        else:
            self.scene_pool = CachedScenePool(
                stage=config.curriculum_stage,
                pool_size=config.scene_pool_size,
                base_seed=int(seed),
                scene_config=scene_config,
                family_schedule=config.scene_family_schedule,
            )
        self.hybrid_reward = OptionalHybridAStarReward(
            planner=hybrid_planner if config.use_hybrid_astar else None,
            gamma=0.98,
            cost_scale=config.planner_cost_scale,
            potential_coef=config.planner_potential_coef,
            potential_clip=config.planner_potential_clip,
            max_cost=config.planner_max_cost,
            lateral_residual_weight=config.planner_lateral_residual_weight,
            goal_heading_weight=config.planner_goal_heading_weight,
            lateral_clip=config.planner_lateral_clip,
            fallback_position_weight=config.planner_fallback_position_weight,
            fallback_heading_weight=config.planner_fallback_heading_weight,
            failure_bias=config.planner_failure_bias,
        )
        if rs_planner is None and config.rs_potential_enabled:
            collision_checker = hybrid_planner
            if not hasattr(collision_checker, "_is_rectangle_occupied"):
                from planning.passenger_hybrid_astar import PassengerHybridAStar

                collision_checker = PassengerHybridAStar(
                    goal_pos_tol=config.planner_position_tolerance,
                    goal_heading_tol_deg=config.planner_heading_tolerance_deg,
                    front_half_length=0.5 * vehicle_params.front_body_length,
                    front_half_width=0.5 * vehicle_params.front_body_width,
                    rear_half_length=0.5 * vehicle_params.rear_body_length,
                    rear_half_width=0.5 * vehicle_params.rear_body_width,
                    front_center_to_hinge=vehicle_params.front_center_to_hinge,
                    rear_center_to_hinge=vehicle_params.rear_center_to_hinge,
                )
            rs_planner = RSPotentialPlanner(
                collision_checker=collision_checker,
                turning_radius=vehicle_params.minimum_turning_radius,
                candidate_limit=config.rs_potential_k,
                sample_step=float(getattr(collision_checker, "step_length", 1.0))
                / float(getattr(collision_checker, "intermediate_checks", 2) + 1),
                vehicle_params=vehicle_params,
            )
        self.rs_potential = RSPotentialOracle(
            planner=rs_planner,
            enabled=config.rs_potential_enabled,
            d_rs=config.rs_potential_d_rs,
            gamma=0.98,
            cost_scale=config.planner_cost_scale,
            potential_coef=config.rs_potential_coef,
            potential_clip=config.rs_potential_clip,
            max_cost=config.planner_max_cost,
            lateral_weight=config.planner_lateral_residual_weight,
            heading_weight=config.planner_goal_heading_weight,
            lateral_clip=config.planner_lateral_clip,
        )
        self.hope_teacher = hope_teacher
        if self.hope_teacher is None and bool(config.enable_hope_teacher):
            self.hope_teacher = HopeTeacherAdapter(
                config=config,
                vehicle_params=vehicle_params,
                rng=self.rng,
            )
        self.reward_model = LocalParkingReward(config)
        self.observation_space = BoxSpace(-np.inf, np.inf, (self.OBS_DIM,))
        self.action_space = BoxSpace(-1.0, 1.0, (2,))
        self.episode_index = 0
        self.step_count = 0
        self.state = None
        self.scene = None
        self.slot = None
        self.current_mask = None
        self.current_normal_mask = None
        self.current_recovery_mask = None
        self.current_mask_floor_info = self._empty_mask_floor_info()
        self.current_dwa_recovery_result = DWAResult(reason="not_triggered")
        self.last_front_lidar_m = None
        self.last_rear_lidar_m = None
        self.prev_motion_gear = None
        self.prev_gear_in_obs = 0.0
        self.dwa_forced_stop_streak = 0
        self.dwa_no_progress_streak = 0
        self.dwa_deadlock_streak = 0
        self.last_dwa_info = self._empty_dwa_info()
        self._last_progress_metrics = None
        self.scenario_type = ""
        self.initial_sampling_diagnostics = {}
        self.hope_teacher_trajectory = None
        self.hope_teacher_info = HopeTeacherAdapter.disabled_diagnostics()
        self.guide_step_info = self._zero_guide_step_info()
        self.guide_weight_current = 0.0
        self.guide_dropout_rate = 1.0
        self.guide_dropped = True
        self._reset_candidate_cache = {}
        self._reset_candidate_bucket_counts = {}

    def set_active_stage(self, stage):
        stage = int(np.clip(stage, 1, 4))
        if self._multi_pool is None:
            raise RuntimeError("set_active_stage requires a MultiStageScenePool")
        self._active_stage = stage
        self.scene_pool = self._multi_pool.pool_for(stage)

    def _state_collides(self, state):
        front_box, rear_box = self.vehicle_model.body_boxes(state)
        return bool(
            self.scene.prepared_obstacles.intersects(front_box)
            or self.scene.prepared_obstacles.intersects(rear_box)
        )

    def _body_clearance(self, state):
        front_box, rear_box = self.vehicle_model.body_boxes(state)
        return float(
            min(
                front_box.distance(self.scene.obstacle_union),
                rear_box.distance(self.scene.obstacle_union),
            )
        )

    def _reset_mask_threshold(self, stage):
        action_min = float(getattr(self.action_mask, "min_safe_ratio", 1e-3))
        threshold = max(
            action_min,
            float(getattr(self.config, "reset_min_mask_safe_ratio", action_min)),
        )
        if int(stage) == 4:
            threshold = max(
                threshold,
                float(
                    getattr(
                        self.config,
                        "stage4_reset_min_mask_safe_ratio",
                        threshold,
                    )
                ),
        )
        return float(threshold)

    @staticmethod
    def _empty_mask_floor_info():
        return {
            "mask_degenerate": False,
            "mask_floor_applied": False,
            "mask_all_zero_before_floor": False,
            "mask_max_before_floor": 0.0,
        }

    @staticmethod
    def _score_to_info(score):
        if score is None:
            return ()
        if isinstance(score, tuple):
            return tuple(
                bool(item) if isinstance(item, (bool, np.bool_)) else float(item)
                for item in score
            )
        return score

    def _empty_dwa_info(self):
        return {
            "dwa_enabled": bool(getattr(self.config, "enable_dwa_recovery", False)),
            "dwa_triggered": False,
            "dwa_used": False,
            "dwa_mode": "none",
            "dwa_reason": "",
            "dwa_candidate_count": 0,
            "dwa_valid_candidate_count": 0,
            "dwa_unlock_success": False,
            "dwa_unlock_step": -1,
            "dwa_deadlock": False,
            "dwa_final_max_safe_ratio": 0.0,
            "dwa_best_score": (),
            "dwa_override_policy_action": False,
            "dwa_teacher_action_valid": False,
            "dwa_policy_loss_weight": 1.0,
            "dwa_policy_invalid_trigger": False,
            "dwa_low_safe_trigger": False,
            "dwa_all_zero_trigger": False,
            "dwa_policy_raw_action": np.zeros(2, dtype=np.float32),
            "dwa_raw_action": np.zeros(2, dtype=np.float32),
            "dwa_executed_action_preview": np.zeros(2, dtype=np.float32),
            "normal_mask_max": 0.0,
            "recovery_mask_applied": False,
            "recovery_mask_nonzero_count": 0,
            "recovery_mask_max": 0.0,
            "effective_mask_max": 0.0,
            "deadlock": False,
        }

    def _dwa_info_from_result(
        self,
        result,
        triggered=False,
        override_applied=False,
        policy_invalid_trigger=False,
        low_safe_trigger=False,
        all_zero_trigger=False,
        policy_raw_action=None,
        normal_mask_max=0.0,
        recovery_mask=None,
        effective_mask=None,
        dwa_policy_loss_weight=1.0,
    ):
        info = self._empty_dwa_info()
        if result is None:
            result = DWAResult()
        policy_raw = (
            np.zeros(2, dtype=np.float32)
            if policy_raw_action is None
            else np.clip(np.asarray(policy_raw_action, dtype=np.float32), -1.0, 1.0)
        )
        raw_action = result.raw_action
        if raw_action is None:
            raw_action = np.zeros(2, dtype=np.float32)
        preview = result.executed_action_preview
        if preview is None:
            preview = np.zeros(2, dtype=np.float32)
        recovery = (
            np.zeros((2, self.action_mask.phi_dot_bins.size), dtype=np.float32)
            if recovery_mask is None
            else np.asarray(recovery_mask, dtype=np.float32)
        )
        effective = (
            np.zeros_like(recovery, dtype=np.float32)
            if effective_mask is None
            else np.asarray(effective_mask, dtype=np.float32)
        )
        recovery_nonzero = int(
            np.count_nonzero(recovery > float(getattr(self.action_mask, "min_safe_ratio", 1e-3)))
        )
        info.update(
            {
                "dwa_triggered": bool(triggered),
                "dwa_used": bool(result.used),
                "dwa_mode": str(result.mode),
                "dwa_reason": str(result.reason),
                "dwa_candidate_count": int(result.candidate_count),
                "dwa_valid_candidate_count": int(result.valid_candidate_count),
                "dwa_unlock_success": bool(result.unlock_success),
                "dwa_unlock_step": int(getattr(result, "unlock_step", -1)),
                "dwa_deadlock": bool(result.deadlock),
                "dwa_final_max_safe_ratio": float(result.final_max_safe_ratio),
                "dwa_best_score": self._score_to_info(result.best_score),
                "dwa_override_policy_action": bool(override_applied),
                "dwa_teacher_action_valid": bool(
                    getattr(result, "teacher_action_valid", False)
                    and result.raw_action is not None
                ),
                "dwa_policy_loss_weight": float(
                    np.clip(float(dwa_policy_loss_weight), 0.0, 1.0)
                ),
                "dwa_policy_invalid_trigger": bool(policy_invalid_trigger),
                "dwa_low_safe_trigger": bool(low_safe_trigger),
                "dwa_all_zero_trigger": bool(all_zero_trigger),
                "dwa_policy_raw_action": policy_raw,
                "dwa_raw_action": np.clip(
                    np.asarray(raw_action, dtype=np.float32),
                    -1.0,
                    1.0,
                ),
                "dwa_executed_action_preview": np.asarray(
                    preview,
                    dtype=np.float32,
                ),
                "normal_mask_max": float(normal_mask_max),
                "recovery_mask_applied": bool(recovery_nonzero > 0),
                "recovery_mask_nonzero_count": int(recovery_nonzero),
                "recovery_mask_max": float(np.max(recovery)) if recovery.size else 0.0,
                "effective_mask_max": float(np.max(effective)) if effective.size else 0.0,
            }
        )
        return info

    @staticmethod
    def _progress_metrics(metrics):
        heading_score = math.cos(abs(float(metrics["heading_error"])))
        return {
            "distance_to_goal": float(metrics["distance_to_goal"]),
            "front_overlap": float(metrics["front_overlap"]),
            "heading_score": float(heading_score),
        }

    def _update_dwa_stuck_counters(self, decoded, metrics):
        if bool(decoded.get("forced_stop", False)):
            self.dwa_forced_stop_streak += 1
        else:
            self.dwa_forced_stop_streak = 0

        current = self._progress_metrics(metrics)
        previous = self._last_progress_metrics
        if previous is None:
            self.dwa_no_progress_streak = 0
        else:
            distance_no_better = (
                current["distance_to_goal"]
                >= previous["distance_to_goal"] - 1e-3
            )
            overlap_no_better = (
                current["front_overlap"]
                <= previous["front_overlap"] + 1e-4
            )
            heading_no_better = (
                current["heading_score"]
                <= previous["heading_score"] + 1e-4
            )
            if distance_no_better and overlap_no_better and heading_no_better:
                self.dwa_no_progress_streak += 1
            else:
                self.dwa_no_progress_streak = 0
        self._last_progress_metrics = current

    def _mask_floor_state(self, mask, allow_floor=True):
        mask_array = np.asarray(mask, dtype=np.float32)
        if mask_array.size == 0:
            raise ValueError("action mask must not be empty")
        finite_mask = np.nan_to_num(
            mask_array,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        finite_mask = np.clip(finite_mask, 0.0, 1.0).astype(np.float32, copy=False)
        mask_max = float(np.max(finite_mask))
        all_zero = bool(np.all(finite_mask <= 0.0))
        eps = max(0.0, float(getattr(self.config, "mask_degenerate_eps", 0.0)))
        if bool(getattr(self.config, "apply_floor_only_when_all_zero", False)):
            degenerate = all_zero
        else:
            degenerate = bool(all_zero or mask_max < eps)

        floor_applied = bool(
            degenerate
            and bool(allow_floor)
            and bool(getattr(self.config, "enable_mask_floor_fallback", False))
        )
        if floor_applied:
            configured_floor = float(getattr(self.config, "mask_floor_value", 0.0))
            r_min = float(getattr(self.action_mask, "min_safe_ratio", 1e-3))
            floor_value = max(configured_floor, r_min + 1e-7)
            floor_value = float(np.clip(floor_value, 0.0, 1.0))
            effective_mask = np.full_like(finite_mask, floor_value, dtype=np.float32)
        else:
            effective_mask = finite_mask.copy()

        return effective_mask, {
            "mask_degenerate": bool(degenerate),
            "mask_floor_applied": bool(floor_applied),
            "mask_all_zero_before_floor": bool(all_zero),
            "mask_max_before_floor": float(mask_max),
        }

    def _dwa_mask_triggers(self, normal_mask, mask_floor_info):
        mask = np.asarray(normal_mask, dtype=np.float32)
        normal_max = float(np.max(mask)) if mask.size else 0.0
        all_zero = bool(
            mask_floor_info.get("mask_all_zero_before_floor", False)
            or normal_max <= float(getattr(self.config, "dwa_all_zero_eps", 1e-3))
        )
        low_safe = bool(
            normal_max < float(getattr(self.config, "dwa_low_safe_ratio", 0.05))
        )
        return bool(all_zero), bool(low_safe), float(normal_max)

    def _empty_recovery_mask(self, reference_mask):
        return np.zeros_like(
            np.asarray(reference_mask, dtype=np.float32),
            dtype=np.float32,
        )

    def _effective_mask_from_recovery(self, normal_mask, mask_floor_info):
        normal = np.asarray(normal_mask, dtype=np.float32)
        recovery_mask = self._empty_recovery_mask(normal)
        result = DWAResult(reason="not_triggered")
        if not bool(getattr(self.config, "enable_dwa_recovery", False)):
            return normal.copy(), recovery_mask, result
        all_zero, low_safe, _ = self._dwa_mask_triggers(normal, mask_floor_info)
        if not (all_zero or low_safe):
            return normal.copy(), recovery_mask, result
        if self.dwa_recovery is None:
            self.dwa_recovery = DWARecoveryController(self.config)
        result = self.dwa_recovery.run(
            "unlock",
            self.state,
            self.slot,
            self.scene,
            self.vehicle_model,
            self.lidar,
            self.action_mask,
            normal,
            self.last_front_lidar_m,
            self.last_rear_lidar_m,
            self.prev_motion_gear,
            self.config,
            reason="all_zero_mask" if all_zero else "low_safe_mask",
        )
        if result.recovery_mask is not None:
            recovery_mask = np.asarray(result.recovery_mask, dtype=np.float32)
            if recovery_mask.shape != normal.shape:
                recovery_mask = self._empty_recovery_mask(normal)
        effective_mask = np.maximum(normal, recovery_mask).astype(np.float32, copy=False)
        return effective_mask, recovery_mask, result

    def _reset_candidate_metrics(self, state, stage):
        front_lidar, rear_lidar = self.lidar.observe(
            state,
            self.vehicle_model,
            self.scene,
            normalize=False,
        )
        mask = self.action_mask.compute_mask(
            state.phi,
            front_lidar,
            rear_lidar,
        )
        _, floor_info = self._mask_floor_state(mask)
        return {
            "body_clearance": self._body_clearance(state),
            "min_lidar": float(min(np.min(front_lidar), np.min(rear_lidar))),
            "mask_max": float(np.max(mask)),
            "mask_degenerate": bool(floor_info["mask_degenerate"]),
            "mask_all_zero_before_floor": bool(
                floor_info["mask_all_zero_before_floor"]
            ),
            "mask_required": self._reset_mask_threshold(stage),
        }

    def _stage4_target_access_metrics(self, state):
        bay = self.scene.target_bay
        mouth = np.asarray(bay.mouth_center, dtype=np.float64)
        corridor_axis = np.asarray(
            [math.cos(float(bay.corridor_heading)), math.sin(float(bay.corridor_heading))],
            dtype=np.float64,
        )
        inward_axis = np.asarray(
            [math.cos(float(bay.inward_heading)), math.sin(float(bay.inward_heading))],
            dtype=np.float64,
        )
        center = np.asarray((state.x_front, state.y_front), dtype=np.float64)
        delta = center - mouth
        along = abs(float(np.dot(delta, corridor_axis)))
        inward = float(np.dot(delta, inward_axis))
        mouth_a = np.asarray(bay.mouth_segment[0], dtype=np.float64)
        mouth_b = np.asarray(bay.mouth_segment[1], dtype=np.float64)
        mouth_half_width = 0.5 * float(np.linalg.norm(mouth_b - mouth_a))
        bay_coords = np.asarray(bay.polygon.exterior.coords[:-1], dtype=np.float64)
        bay_depth = float(np.max((bay_coords - mouth).dot(inward_axis)))
        corridor_width = float(self.scene.metadata.get("corridor_width", 0.0))
        along_limit = mouth_half_width + float(self.config.stage_lateral_ranges[3]) + 0.75
        inward_min = -corridor_width - 0.75
        inward_max = bay_depth + 0.75
        valid = bool(along <= along_limit and inward_min <= inward <= inward_max)
        return valid, {
            "target_access_along_m": float(along),
            "target_access_inward_m": float(inward),
            "target_access_along_limit_m": float(along_limit),
            "target_access_inward_min_m": float(inward_min),
            "target_access_inward_max_m": float(inward_max),
        }

    def _reset_candidate_viability(self, state, stage):
        if self._state_collides(state):
            return False, "collision", {}

        rule_scene = self._is_rule_scene()
        target_access_metrics = {}
        if int(stage) == 4 and not rule_scene:
            target_access_valid, target_access_metrics = (
                self._stage4_target_access_metrics(state)
            )
            if not target_access_valid:
                return False, "target_access", target_access_metrics

        metrics = self._reset_candidate_metrics(state, stage)
        metrics.update(target_access_metrics)
        if float(metrics["mask_max"]) <= float(metrics["mask_required"]):
            return False, "mask", metrics

        if int(stage) == 4:
            min_clearance = float(
                getattr(self.config, "stage4_reset_min_body_clearance", 0.0)
            )
            metrics["stage4_min_body_clearance"] = min_clearance
            if min_clearance > 0.0 and float(metrics["body_clearance"]) < min_clearance:
                return False, "clearance", metrics
            if rule_scene:
                return True, "", metrics
            if float(metrics["body_clearance"]) > float(
                self.config.recovery_max_body_clearance
            ):
                return False, "recovery_clearance", metrics
            if float(metrics["min_lidar"]) > float(self.config.recovery_max_lidar_distance):
                return False, "recovery_lidar", metrics

        return True, "", metrics

    @staticmethod
    def _reset_reject_counts():
        return {
            "collision": 0,
            "mask": 0,
            "clearance": 0,
            "recovery_clearance": 0,
            "recovery_lidar": 0,
            "target_access": 0,
        }

    @staticmethod
    def _halton(index, base):
        result = 0.0
        fraction = 1.0 / float(base)
        value = int(index)
        while value > 0:
            result += fraction * float(value % int(base))
            value //= int(base)
            fraction /= float(base)
        return float(result)

    def _low_discrepancy_rows(self, count, dimensions, seed_items):
        seed_sequence = np.random.SeedSequence([int(item) & 0xFFFFFFFF for item in seed_items])
        local_rng = np.random.default_rng(seed_sequence)
        shifts = local_rng.random(int(dimensions))
        bases = (2, 3, 5, 7, 11, 13, 17)
        rows = np.zeros((int(count), int(dimensions)), dtype=np.float64)
        for row_index in range(int(count)):
            for dim_index in range(int(dimensions)):
                rows[row_index, dim_index] = (
                    self._halton(row_index + 1, bases[dim_index]) + shifts[dim_index]
                ) % 1.0
        return rows

    def _state_from_goal_offsets(self, goal, axis, normal, distance, lateral, heading_error, phi):
        center = (
            np.asarray(goal.center, dtype=np.float64)
            - float(distance) * np.asarray(axis, dtype=np.float64)
            + float(lateral) * np.asarray(normal, dtype=np.float64)
        )
        theta_front = wrap_to_pi(float(goal.theta_goal) + float(heading_error))
        return ArticulatedState(
            x_front=float(center[0]),
            y_front=float(center[1]),
            theta_front=float(theta_front),
            theta_rear=float(wrap_to_pi(theta_front - float(phi))),
        )

    def _reset_clearance_bucket(self, clearance_m, stage):
        clearance_m = float(clearance_m)
        if int(stage) == 4:
            if clearance_m < 0.30:
                return "tight_recover"
            if clearance_m < 0.50:
                return "narrow_recover"
            if clearance_m <= float(self.config.recovery_max_body_clearance):
                return "moderate_recover"
            return "open"
        if clearance_m < 0.50:
            return "tight"
        if clearance_m < 1.00:
            return "narrow"
        if clearance_m < 2.00:
            return "normal"
        return "open"

    @staticmethod
    def _reset_mask_bucket(mask_max, mask_required):
        mask_max = float(mask_max)
        mask_required = float(mask_required)
        span = max(1e-6, 1.0 - mask_required)
        quality = (mask_max - mask_required) / span
        if quality < 0.25:
            return "weak"
        if quality < 0.60:
            return "medium"
        return "strong"

    def _reset_pose_bucket(self, state, goal, stage):
        index = max(0, min(3, int(stage) - 1))
        slot_error = goal.position_error_in_slot_frame(state.x_front, state.y_front)
        heading_range = max(
            1e-6,
            math.radians(float(self.config.stage_heading_ranges_deg[index])),
        )
        lateral_range = max(1e-6, float(self.config.stage_lateral_ranges[index]))
        phi_range = max(
            1e-6,
            math.radians(float(self.config.stage_phi_ranges_deg[index])),
        )
        heading_score = abs(float(wrap_to_pi(state.theta_front - goal.theta_goal))) / heading_range
        lateral_score = abs(float(slot_error[1])) / lateral_range
        phi_score = abs(float(state.phi)) / phi_range
        scores = {
            "heading_dominant": heading_score,
            "lateral_dominant": lateral_score,
            "articulation_dominant": phi_score,
        }
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if len(ordered) >= 2 and ordered[1][1] >= 0.70 * max(ordered[0][1], 1e-6):
            return "mixed"
        return ordered[0][0]

    def _reset_distance_bucket(self, state, goal, stage):
        index = max(0, min(3, int(stage) - 1))
        distance_range = self.config.stage_distance_ranges[index]
        distance = math.hypot(
            float(state.x_front) - float(goal.x_goal),
            float(state.y_front) - float(goal.y_goal),
        )
        t = (distance - float(distance_range[0])) / max(
            1e-6,
            float(distance_range[1]) - float(distance_range[0]),
        )
        if t < 0.33:
            return "near"
        if t < 0.66:
            return "mid"
        return "far"

    def _reset_candidate_score(self, state, stage, metrics):
        mask_required = float(metrics.get("mask_required", self._reset_mask_threshold(stage)))
        mask_quality = (float(metrics.get("mask_max", 0.0)) - mask_required) / max(
            1e-6,
            1.0 - mask_required,
        )
        mask_quality = float(np.clip(mask_quality, 0.0, 1.0))
        if int(stage) == 4:
            min_clearance = float(self.config.stage4_reset_min_body_clearance)
            max_clearance = float(self.config.recovery_max_body_clearance)
            clearance = float(metrics.get("body_clearance", 0.0))
            target_clearance = 0.5 * (min_clearance + max_clearance)
            clearance_quality = 1.0 - abs(clearance - target_clearance) / max(
                1e-6,
                max_clearance - min_clearance,
            )
            clearance_quality = float(np.clip(clearance_quality, 0.0, 1.0))
            lidar_quality = 1.0 - float(metrics.get("min_lidar", 0.0)) / max(
                1e-6,
                float(self.config.recovery_max_lidar_distance),
            )
            lidar_quality = float(np.clip(lidar_quality, 0.0, 1.0))
            articulation_quality = abs(float(state.phi)) / max(
                1e-6,
                float(self.vehicle_params.phi_max),
            )
            articulation_quality = float(np.clip(articulation_quality, 0.0, 1.0))
            return float(
                0.40 * mask_quality
                + 0.30 * clearance_quality
                + 0.15 * lidar_quality
                + 0.15 * articulation_quality
            )
        clearance_quality = 1.0 / (1.0 + max(0.0, float(metrics.get("body_clearance", 0.0))))
        return float(0.70 * mask_quality + 0.30 * clearance_quality)

    def _annotate_reset_candidate_metrics(self, state, stage, metrics):
        annotated = dict(metrics)
        clearance_bucket = self._reset_clearance_bucket(
            annotated.get("body_clearance", 0.0),
            stage,
        )
        mask_bucket = self._reset_mask_bucket(
            annotated.get("mask_max", 0.0),
            annotated.get("mask_required", self._reset_mask_threshold(stage)),
        )
        pose_bucket = self._reset_pose_bucket(state, self.slot, stage)
        annotated["clearance_bucket"] = str(clearance_bucket)
        annotated["mask_bucket"] = str(mask_bucket)
        annotated["pose_bucket"] = str(pose_bucket)
        annotated["distance_bucket"] = str(
            self._reset_distance_bucket(state, self.slot, stage)
        )
        annotated["candidate_score"] = float(
            self._reset_candidate_score(state, stage, annotated)
        )
        return annotated

    def _record_reset_candidate(self, bank, state, scenario, stage):
        bank["size"] = int(bank.get("size", 0)) + 1
        valid, reject_reason, metrics = self._reset_candidate_viability(state, stage)
        if not valid:
            reject_counts = bank["reject_counts"]
            reject_counts[reject_reason] = reject_counts.get(reject_reason, 0) + 1
            return
        metrics = self._annotate_reset_candidate_metrics(state, stage, metrics)
        bucket_key = (
            str(metrics["clearance_bucket"]),
            str(metrics["mask_bucket"]),
            str(metrics["pose_bucket"]),
            str(metrics["distance_bucket"]),
        )
        bank["candidates"].append(
            {
                "state": state,
                "scenario": str(scenario),
                "metrics": metrics,
                "bucket_key": bucket_key,
                "score": float(metrics["candidate_score"]),
            }
        )

    def _build_reset_candidate_bank(self, stage, goal, axis, normal):
        scene_seed = int(self.scene.metadata["seed"])
        cache_key = (scene_seed, int(stage))
        cached = self._reset_candidate_cache.get(cache_key)
        if cached is not None:
            return cached

        bank = {
            "cache_key": cache_key,
            "size": 0,
            "candidates": [],
            "reject_counts": self._reset_reject_counts(),
        }
        index = max(0, min(3, int(stage) - 1))
        distance_range = self.config.stage_distance_ranges[index]
        lateral_range = float(self.config.stage_lateral_ranges[index])
        heading_range = math.radians(float(self.config.stage_heading_ranges_deg[index]))
        phi_range = math.radians(float(self.config.stage_phi_ranges_deg[index]))

        if int(stage) == 4:
            # Stage 4 needs near-obstacle recovery states. Pure uniform sampling spends
            # most attempts in collision or fully blocked-mask poses in the narrow 5 m corridor.
            for state, scenario in self._structured_recovery_state(goal, axis, normal):
                self._record_reset_candidate(bank, state, scenario, stage)
        else:
            count = 160 if int(stage) < 3 else 240
            rows = self._low_discrepancy_rows(
                count,
                4,
                (scene_seed, int(stage), 0xA71CE),
            )
            for candidate_index, row in enumerate(rows):
                distance = self._lerp(distance_range[0], distance_range[1], row[0])
                lateral = (2.0 * row[1] - 1.0) * lateral_range
                heading_error = (2.0 * row[2] - 1.0) * heading_range
                phi = (2.0 * row[3] - 1.0) * phi_range
                scenario = {
                    1: "near_goal",
                    2: "near_goal_obstacles",
                    3: "poor_terminal_pose",
                }[int(stage)]
                if int(stage) == 3:
                    pose_mode = int(candidate_index % 3)
                    if pose_mode == 0:
                        min_heading = math.radians(self.config.poor_pose_min_heading_deg)
                        magnitude = self._lerp(min_heading, heading_range, row[2])
                        heading_error = math.copysign(
                            magnitude,
                            -1.0 if row[3] < 0.5 else 1.0,
                        )
                        scenario = "poor_terminal_heading"
                    elif pose_mode == 1:
                        min_lateral = min(
                            float(self.config.poor_pose_min_lateral),
                            lateral_range,
                        )
                        magnitude = self._lerp(min_lateral, lateral_range, row[1])
                        lateral = math.copysign(
                            magnitude,
                            -1.0 if row[2] < 0.5 else 1.0,
                        )
                        scenario = "poor_terminal_lateral"
                    else:
                        min_phi = math.radians(self.config.poor_pose_min_abs_phi_deg)
                        magnitude = self._lerp(min_phi, phi_range, row[3])
                        phi = math.copysign(
                            magnitude,
                            -1.0 if row[1] < 0.5 else 1.0,
                        )
                        scenario = "poor_terminal_articulation"
                state = self._state_from_goal_offsets(
                    goal,
                    axis,
                    normal,
                    distance,
                    lateral,
                    heading_error,
                    phi,
                )
                self._record_reset_candidate(bank, state, scenario, stage)

        bank["valid_count"] = int(len(bank["candidates"]))
        bank["bucket_count"] = int(
            len({candidate["bucket_key"] for candidate in bank["candidates"]})
        )
        self._reset_candidate_cache[cache_key] = bank
        return bank

    def _select_reset_candidate_from_bank(self, bank):
        candidates = list(bank.get("candidates", ()))
        if not candidates:
            return None
        buckets = sorted({candidate["bucket_key"] for candidate in candidates})
        cache_key = bank.get("cache_key", ("", 0))
        counts = {
            bucket: int(
                self._reset_candidate_bucket_counts.get((cache_key, bucket), 0)
            )
            for bucket in buckets
        }
        min_count = min(counts.values())
        eligible_buckets = [bucket for bucket in buckets if counts[bucket] == min_count]
        bucket = eligible_buckets[int(self.rng.integers(0, len(eligible_buckets)))]
        bucket_candidates = [
            candidate for candidate in candidates if candidate["bucket_key"] == bucket
        ]
        weights = np.asarray(
            [
                0.50 + 0.50 * float(np.clip(candidate.get("score", 0.0), 0.0, 1.0))
                for candidate in bucket_candidates
            ],
            dtype=np.float64,
        )
        weights = weights / float(np.sum(weights))
        choice_index = int(self.rng.choice(len(bucket_candidates), p=weights))
        self._reset_candidate_bucket_counts[(cache_key, bucket)] = counts[bucket] + 1
        return bucket_candidates[choice_index]

    def _finalize_reset_candidate(
        self,
        state,
        scenario,
        fallback_used,
        stage,
        task_family,
        metrics,
        reject_counts,
    ):
        metrics = self._annotate_reset_candidate_metrics(state, stage, metrics)
        self.initial_sampling_diagnostics.update(
            {
                "reset_candidate_reject_collision_count": int(
                    reject_counts.get("collision", 0)
                ),
                "reset_candidate_reject_mask_count": int(
                    reject_counts.get("mask", 0)
                ),
                "reset_candidate_reject_clearance_count": int(
                    reject_counts.get("clearance", 0)
                ),
                "reset_candidate_reject_recovery_clearance_count": int(
                    reject_counts.get("recovery_clearance", 0)
                ),
                "reset_candidate_reject_recovery_lidar_count": int(
                    reject_counts.get("recovery_lidar", 0)
                ),
                "reset_candidate_reject_target_access_count": int(
                    reject_counts.get("target_access", 0)
                ),
                "reset_initial_mask_max": float(metrics.get("mask_max", 0.0)),
                "reset_initial_mask_degenerate": bool(
                    metrics.get("mask_degenerate", False)
                ),
                "reset_initial_mask_all_zero": bool(
                    metrics.get("mask_all_zero_before_floor", False)
                ),
                "reset_initial_mask_required": float(
                    metrics.get("mask_required", self._reset_mask_threshold(stage))
                ),
                "reset_initial_body_clearance_m": float(
                    metrics.get("body_clearance", 0.0)
                ),
                "reset_initial_min_lidar_m": float(metrics.get("min_lidar", 0.0)),
                "reset_candidate_selected_clearance_bucket": str(
                    metrics.get("clearance_bucket", "")
                ),
                "reset_candidate_selected_pose_bucket": str(
                    metrics.get("pose_bucket", "")
                ),
                "reset_candidate_selected_mask_max": float(
                    metrics.get("mask_max", 0.0)
                ),
            }
        )
        self.initial_sampling_diagnostics["initial_task_family"] = str(task_family)
        return state, scenario, fallback_used

    def _valid_recovery_state(self, state):
        valid, _, _ = self._reset_candidate_viability(state, stage=4)
        return bool(valid)

    def _sample_hard_case_replay_state(self, replay_case):
        diagnostics = {
            "hard_case_replay_attempted": True,
            "hard_case_replay_used": False,
            "hard_case_replay_reject_reason": "",
        }
        if replay_case is None:
            diagnostics["hard_case_replay_attempted"] = False
            return None, "", False, diagnostics
        scene = replay_case.get("scene")
        state = replay_case.get("state")
        if scene is None or state is None:
            diagnostics["hard_case_replay_reject_reason"] = "missing_scene_or_state"
            return None, "", False, diagnostics

        self.scene = scene
        self.slot = replay_case.get("slot", scene.slot)
        self.step_count = 0
        stage = int(np.clip(replay_case.get("stage", self._active_stage), 1, 4))
        attempts = max(1, int(self.config.hard_case_replay_attempts))
        xy_std = float(self.config.hard_case_replay_xy_std)
        heading_std = math.radians(float(self.config.hard_case_replay_heading_std_deg))
        phi_std = math.radians(float(self.config.hard_case_replay_phi_std_deg))
        reject_counts = {
            "collision": 0,
            "mask": 0,
            "clearance": 0,
            "recovery_clearance": 0,
            "recovery_lidar": 0,
            "target_access": 0,
        }
        last_reason = ""
        last_metrics = {}
        for attempt_index in range(attempts):
            if attempt_index == 0:
                dx = 0.0
                dy = 0.0
                dtheta = 0.0
                dphi = 0.0
            else:
                dx = float(self.rng.normal(0.0, xy_std))
                dy = float(self.rng.normal(0.0, xy_std))
                dtheta = float(self.rng.normal(0.0, heading_std))
                dphi = float(self.rng.normal(0.0, phi_std))
            phi = float(
                np.clip(
                    state.phi + dphi,
                    -self.vehicle_params.phi_max,
                    self.vehicle_params.phi_max,
                )
            )
            theta_front = float(wrap_to_pi(state.theta_front + dtheta))
            candidate = ArticulatedState(
                x_front=float(state.x_front + dx),
                y_front=float(state.y_front + dy),
                theta_front=theta_front,
                theta_rear=float(wrap_to_pi(theta_front - phi)),
            )
            valid, reject_reason, metrics = self._reset_candidate_viability(
                candidate,
                stage=stage,
            )
            if not valid:
                reject_counts[reject_reason] = reject_counts.get(reject_reason, 0) + 1
                last_reason = str(reject_reason)
                last_metrics = metrics
                continue
            scenario = "{}_hard_case_replay".format(
                str(replay_case.get("scenario_type", "hard_case"))
            )
            distance = math.hypot(
                candidate.x_front - self.slot.x_goal,
                candidate.y_front - self.slot.y_goal,
            )
            diagnostics.update(
                {
                    "hard_case_replay_used": True,
                    "hard_case_replay_source_episode": int(
                        replay_case.get("episode", -1)
                    ),
                    "hard_case_replay_source_stage": int(stage),
                    "hard_case_replay_source_scene_seed": int(
                        replay_case.get("scene_seed", self.scene.metadata["seed"])
                    ),
                    "hard_case_replay_source_failure": str(
                        replay_case.get("failure_type", "")
                    ),
                    "hard_case_replay_attempt_count": int(attempt_index + 1),
                    "initial_distance_min": float(distance),
                    "initial_distance_max": float(distance),
                    "initial_lateral_range": 0.0,
                    "initial_heading_range_deg": float(
                        math.degrees(abs(dtheta))
                    ),
                    "initial_phi_range_deg": float(math.degrees(abs(dphi))),
                    "reset_candidate_reject_collision_count": int(
                        reject_counts.get("collision", 0)
                    ),
                    "reset_candidate_reject_mask_count": int(
                        reject_counts.get("mask", 0)
                    ),
                    "reset_candidate_reject_clearance_count": int(
                        reject_counts.get("clearance", 0)
                    ),
                    "reset_candidate_reject_recovery_clearance_count": int(
                        reject_counts.get("recovery_clearance", 0)
                    ),
                    "reset_candidate_reject_recovery_lidar_count": int(
                        reject_counts.get("recovery_lidar", 0)
                    ),
                    "reset_candidate_reject_target_access_count": int(
                        reject_counts.get("target_access", 0)
                    ),
                    "reset_initial_mask_max": float(metrics.get("mask_max", 0.0)),
                    "reset_initial_mask_degenerate": bool(
                        metrics.get("mask_degenerate", False)
                    ),
                    "reset_initial_mask_all_zero": bool(
                        metrics.get("mask_all_zero_before_floor", False)
                    ),
                    "reset_initial_mask_required": float(
                        metrics.get("mask_required", self._reset_mask_threshold(stage))
                    ),
                    "reset_initial_body_clearance_m": float(
                        metrics.get("body_clearance", 0.0)
                    ),
                    "reset_initial_min_lidar_m": float(
                        metrics.get("min_lidar", 0.0)
                    ),
                    "initial_task_family": str(
                        replay_case.get("task_family", self._task_family_from_scene())
                    ),
                }
            )
            return candidate, scenario, False, diagnostics

        diagnostics.update(
            {
                "hard_case_replay_reject_reason": last_reason,
                "hard_case_replay_attempt_count": int(attempts),
                "hard_case_replay_last_mask_max": float(
                    last_metrics.get("mask_max", 0.0)
                ),
                "hard_case_replay_last_body_clearance_m": float(
                    last_metrics.get("body_clearance", 0.0)
                ),
                "hard_case_replay_last_min_lidar_m": float(
                    last_metrics.get("min_lidar", 0.0)
                ),
            }
        )
        return None, "", False, diagnostics

    def _task_family_from_scene(self):
        task_family = str(self.scene.metadata.get("task_family", ""))
        if task_family in TASK_FAMILIES:
            return task_family
        goal_orientation_mode = self.scene.metadata.get(
            "goal_orientation_mode",
            "head_in",
        )
        if goal_orientation_mode == "head_in":
            return "head_in"
        raise ValueError(
            "unsupported goal orientation mode: {}".format(goal_orientation_mode)
        )

    @staticmethod
    def _lerp(value_a, value_b, t):
        return float(value_a) + (float(value_b) - float(value_a)) * float(t)

    @staticmethod
    def _zero_guide_step_info():
        return {
            "guide_reward": 0.0,
            "guide_progress_reward": 0.0,
            "guide_lateral_error": 0.0,
            "guide_heading_error": 0.0,
            "guide_anchor_error": 0.0,
            "guide_gear_agreement": 0.0,
            "guide_corridor_penalty": 0.0,
            "guide_tangent_reward": 0.0,
            "guide_anchor_reward": 0.0,
            "guide_gear_reward": 0.0,
        }

    def _episode_schedule_value(self, initial, final, start_episode, end_episode):
        episode = max(0, int(self.episode_index) - 1)
        start_episode = int(start_episode)
        end_episode = int(end_episode)
        if end_episode <= start_episode:
            progress = 1.0 if episode >= end_episode else 0.0
        elif episode <= start_episode:
            progress = 0.0
        elif episode >= end_episode:
            progress = 1.0
        else:
            progress = (episode - start_episode) / float(end_episode - start_episode)
        return self._lerp(initial, final, progress), float(progress)

    def _reset_hope_teacher(self):
        self.hope_teacher_trajectory = None
        self.guide_step_info = self._zero_guide_step_info()
        self.guide_weight_current = 0.0
        self.guide_dropout_rate = 1.0
        self.guide_dropped = True
        self.hope_teacher_info = HopeTeacherAdapter.disabled_diagnostics()
        if self.hope_teacher is None or not bool(self.config.enable_hope_teacher):
            return

        base_weight, anneal_progress = self._episode_schedule_value(
            self.config.guide_weight_initial,
            self.config.guide_weight_final,
            self.config.guide_anneal_start_episode,
            self.config.guide_anneal_end_episode,
        )
        dropout_rate, _ = self._episode_schedule_value(
            self.config.guide_dropout_initial,
            self.config.guide_dropout_final,
            self.config.guide_anneal_start_episode,
            self.config.guide_anneal_end_episode,
        )
        dropout_rate = float(np.clip(dropout_rate, 0.0, 1.0))
        dropped = bool(self.rng.random() < dropout_rate)
        trajectory = self.hope_teacher.plan_episode(
            self.scene,
            self.state,
            self.slot,
            self.vehicle_model,
        )
        hard_case_replay_used = bool(
            self.initial_sampling_diagnostics.get("hard_case_replay_used", False)
        )
        if (
            bool(self.config.enable_offpath_reset)
            and trajectory.reward_available
            and not hard_case_replay_used
        ):
            offpath_state, offpath_reason = self.hope_teacher.sample_offpath_state(
                trajectory,
                self.rng,
                anneal_progress,
                self.scene,
                self.vehicle_model,
            )
            self.initial_sampling_diagnostics["hope_offpath_reset_attempted"] = True
            self.initial_sampling_diagnostics["hope_offpath_reset_reason"] = str(
                offpath_reason
            )
            if offpath_state is not None:
                valid, reject_reason, metrics = self._reset_candidate_viability(
                    offpath_state,
                    stage=self._active_stage,
                )
                self.initial_sampling_diagnostics["hope_offpath_reset_viable"] = bool(
                    valid
                )
                self.initial_sampling_diagnostics[
                    "hope_offpath_reset_reject_reason"
                ] = str(reject_reason)
            else:
                valid = False
                metrics = {}
                self.initial_sampling_diagnostics["hope_offpath_reset_viable"] = False
                self.initial_sampling_diagnostics[
                    "hope_offpath_reset_reject_reason"
                ] = ""
            if offpath_state is not None and valid:
                self.state = offpath_state
                self.scenario_type = "{}_hope_offpath".format(self.scenario_type)
                self.initial_sampling_diagnostics["hope_offpath_reset_used"] = True
                self.initial_sampling_diagnostics["reset_initial_mask_max"] = float(
                    metrics.get("mask_max", 0.0)
                )
                self.initial_sampling_diagnostics[
                    "reset_initial_mask_degenerate"
                ] = bool(metrics.get("mask_degenerate", False))
                self.initial_sampling_diagnostics["reset_initial_mask_all_zero"] = bool(
                    metrics.get("mask_all_zero_before_floor", False)
                )
                self.initial_sampling_diagnostics[
                    "reset_initial_body_clearance_m"
                ] = float(metrics.get("body_clearance", 0.0))
                self.initial_sampling_diagnostics["reset_initial_min_lidar_m"] = float(
                    metrics.get("min_lidar", 0.0)
                )
            else:
                self.initial_sampling_diagnostics["hope_offpath_reset_used"] = False
        else:
            self.initial_sampling_diagnostics["hope_offpath_reset_attempted"] = False
            self.initial_sampling_diagnostics["hope_offpath_reset_used"] = False
            self.initial_sampling_diagnostics["hope_offpath_reset_reason"] = ""

        effective_weight = 0.0
        if (
            bool(self.config.use_teacher_reward)
            and not dropped
            and trajectory.reward_available
        ):
            effective_weight = float(base_weight)
        self.hope_teacher_trajectory = trajectory
        self.guide_weight_current = float(effective_weight)
        self.guide_dropout_rate = float(dropout_rate)
        self.guide_dropped = bool(dropped)
        self.hope_teacher_info = self.hope_teacher.diagnostics(
            trajectory,
            guide_weight=effective_weight,
            dropout_rate=dropout_rate,
            dropped=dropped,
        )

    def _structured_recovery_state(self, goal, axis, normal):
        """Generate deterministic Stage 4 recovery candidates for the current scene."""
        candidates = []
        index = 3
        distance_range = self.config.stage_distance_ranges[index]
        lateral_range = float(self.config.stage_lateral_ranges[index])
        heading_limit_deg = float(self.config.stage_heading_ranges_deg[index])
        phi_limit_deg = float(self.config.stage_phi_ranges_deg[index])
        min_phi_deg = float(self.config.recovery_min_abs_phi_deg)

        def unique_sorted(values):
            result = []
            for value in values:
                value = float(np.round(float(value), 3))
                if value not in result:
                    result.append(value)
            return result

        def bounded_distance(value):
            return float(np.clip(float(value), float(distance_range[0]), float(distance_range[1])))

        def bounded_lateral(value):
            return float(np.clip(float(value), -lateral_range, lateral_range))

        def add_offset_candidate(distance, lateral, heading_deg, phi_deg):
            heading_deg = float(np.clip(float(heading_deg), -heading_limit_deg, heading_limit_deg))
            phi_deg = float(np.clip(float(phi_deg), -phi_limit_deg, phi_limit_deg))
            state = self._state_from_goal_offsets(
                goal,
                axis,
                normal,
                bounded_distance(distance),
                bounded_lateral(lateral),
                math.radians(heading_deg),
                math.radians(phi_deg),
            )
            candidates.append((state, "recovery"))

        toward_goal_modes = (
            (0.0, 0.0),
            (10.0, 0.0),
            (-10.0, 0.0),
            (0.0, 8.0),
            (0.0, -8.0),
        )
        slight_yaw_modes = (
            (18.0, 12.0),
            (-18.0, -12.0),
            (24.0, -min_phi_deg),
            (-24.0, min_phi_deg),
        )
        recovery_modes = (
            (35.0, min_phi_deg),
            (-35.0, -min_phi_deg),
            (45.0, 24.0),
            (-45.0, -24.0),
            (heading_limit_deg, phi_limit_deg),
            (-heading_limit_deg, -phi_limit_deg),
            (35.0, -24.0),
            (-35.0, 24.0),
        )

        core_distances = unique_sorted(
            bounded_distance(value)
            for value in (
                4.5,
                6.0,
                7.5,
                9.0,
                10.5,
                12.0,
                14.0,
                16.5,
                19.5,
            )
        )
        center_laterals = unique_sorted((0.0, -0.8, 0.8, -1.4, 1.4))
        near_wall_laterals = unique_sorted(
            (
                -0.55 * lateral_range,
                -0.78 * lateral_range,
                -0.96 * lateral_range,
                0.55 * lateral_range,
                0.78 * lateral_range,
                0.96 * lateral_range,
            )
        )

        for distance in core_distances:
            for lateral in center_laterals:
                for heading_deg, phi_deg in toward_goal_modes + slight_yaw_modes:
                    add_offset_candidate(distance, lateral, heading_deg, phi_deg)

        for distance in core_distances[1:-1]:
            for lateral in near_wall_laterals:
                for heading_deg, phi_deg in slight_yaw_modes + recovery_modes:
                    add_offset_candidate(distance, lateral, heading_deg, phi_deg)

        mouth_center = np.asarray(self.scene.target_bay.mouth_center, dtype=np.float64)
        goal_center = np.asarray(goal.center, dtype=np.float64)
        mouth_distance = float(np.dot(goal_center - mouth_center, axis))
        mouth_distances = unique_sorted(
            bounded_distance(mouth_distance + offset) for offset in (0.5, 2.0, 3.5, 5.0)
        )
        mouth_laterals = unique_sorted(
            (
                -0.90 * lateral_range,
                -0.45 * lateral_range,
                0.0,
                0.45 * lateral_range,
                0.90 * lateral_range,
            )
        )
        for distance in mouth_distances:
            for lateral in mouth_laterals:
                for heading_deg, phi_deg in toward_goal_modes + recovery_modes:
                    add_offset_candidate(distance, lateral, heading_deg, phi_deg)

        return candidates

    def _is_rule_scene(self):
        scene_type = str(self.scene.metadata.get("scene_type", ""))
        return scene_type in RULE_SCENE_TYPES

    def _rule_scene_candidate_state(self, candidate, attempt_index):
        region = str(candidate[0])
        bay_index = int(candidate[1]) if len(candidate) > 1 else -1
        x = float(candidate[2])
        y = float(candidate[3])
        theta = float(candidate[4])
        phi = float(candidate[5]) if len(candidate) > 5 else 0.0
        noise = self.scene.metadata.get(
            "initial_pose_noise",
            getattr(self.scene_pool.scene_config, "initial_pose_noise", (0.0, 0.0, 0.0, 0.0))
            if hasattr(self.scene_pool, "scene_config")
            else (0.0, 0.0, 0.0, 0.0),
        )
        noise = tuple(float(item) for item in noise)
        if len(noise) < 4:
            noise = noise + (0.0,) * (4 - len(noise))
        if int(attempt_index) > 0:
            x += float(self.rng.uniform(-noise[0], noise[0]))
            y += float(self.rng.uniform(-noise[1], noise[1]))
            theta = float(wrap_to_pi(theta + self.rng.uniform(-noise[2], noise[2])))
            phi = float(
                np.clip(
                    phi + self.rng.uniform(-noise[3], noise[3]),
                    -self.vehicle_params.phi_max,
                    self.vehicle_params.phi_max,
                )
            )
        if region == "bay":
            theta = float(-0.5 * math.pi)
        return (
            ArticulatedState(
                x_front=float(x),
                y_front=float(y),
                theta_front=float(wrap_to_pi(theta)),
                theta_rear=float(wrap_to_pi(theta - phi)),
            ),
            region,
            bay_index,
        )

    def _rule_scene_heading_valid(self, state, region):
        scene_type = str(self.scene.metadata.get("scene_type", ""))
        if scene_type == "mixing_station_bay_corridor":
            if region == "bay":
                return bool(
                    math.cos(wrap_to_pi(state.theta_front + 0.5 * math.pi)) > 0.995
                )
            if region == "corridor":
                mode = str(
                    self.scene.metadata.get(
                        "corridor_initial_heading_mode",
                        "mixed",
                    )
                )
                along = abs(math.sin(float(state.theta_front))) < 0.20
                face_bay = math.cos(wrap_to_pi(state.theta_front + 0.5 * math.pi)) > 0.95
                if mode == "along_corridor":
                    return bool(along)
                if mode == "face_bay":
                    return bool(face_bay)
                return bool(along or face_bay)
        return True

    def _rule_scene_separation_valid(self, state):
        min_sep = float(self.scene.metadata.get("min_initial_target_separation", 0.0))
        max_sep = float(self.scene.metadata.get("max_initial_target_separation", 1e9))
        distance = math.hypot(
            float(state.x_front) - float(self.slot.x_goal),
            float(state.y_front) - float(self.slot.y_goal),
        )
        return bool(min_sep <= distance <= max_sep), float(distance)

    def _sample_rule_scene_initial_state(self):
        stage = int(np.clip(self._active_stage, 1, 4))
        task_family = self._task_family_from_scene()
        candidates = tuple(self.scene.metadata.get("initial_pose_candidates", ()))
        if not candidates:
            raise ResetInitialStateError(
                "no rule-scene initial candidates for scene seed {}".format(
                    self.scene.metadata["seed"]
                )
            )
        max_attempts = max(
            1,
            int(self.scene.metadata.get("max_pose_sampling_attempts", 32)),
        )
        ensure_feasible = bool(
            self.scene.metadata.get("ensure_feasible_reset", True)
        )
        reject_counts = self._reset_reject_counts()
        reject_counts["separation"] = 0
        reject_counts["heading"] = 0
        last_reason = ""
        last_metrics = {}
        self.initial_sampling_diagnostics = {
            "initial_task_family": str(task_family),
            "initial_candidate_count": int(len(candidates)),
            "initial_distance_min": float(
                self.scene.metadata.get("min_initial_target_separation", 0.0)
            ),
            "initial_distance_max": float(
                self.scene.metadata.get("max_initial_target_separation", 0.0)
            ),
            "initial_lateral_range": 0.0,
            "initial_heading_range_deg": 0.0,
            "initial_phi_range_deg": 0.0,
            "reset_candidate_bank_size": int(len(candidates)),
            "reset_candidate_bank_valid_count": 0,
            "reset_candidate_bank_empty": False,
            "reset_candidate_selected_clearance_bucket": "",
            "reset_candidate_selected_pose_bucket": "",
            "reset_candidate_selected_mask_max": 0.0,
            "reset_feasible_mask_available": False,
        }
        order = list(self.rng.permutation(len(candidates)))
        attempt_count = 0
        selected_region = ""
        selected_bay_index = -1
        for attempt_index in range(max_attempts):
            candidate = candidates[order[attempt_index % len(order)]]
            state, region, bay_index = self._rule_scene_candidate_state(
                candidate,
                attempt_index,
            )
            attempt_count += 1
            if not self._rule_scene_heading_valid(state, region):
                reject_counts["heading"] += 1
                last_reason = "heading"
                continue
            separation_valid, distance = self._rule_scene_separation_valid(state)
            if not separation_valid:
                reject_counts["separation"] += 1
                last_reason = "separation"
                continue
            if ensure_feasible:
                valid, reject_reason, metrics = self._reset_candidate_viability(
                    state,
                    stage=stage,
                )
            else:
                if self._state_collides(state):
                    valid, reject_reason, metrics = False, "collision", {}
                else:
                    valid = True
                    reject_reason = ""
                    metrics = self._reset_candidate_metrics(state, stage)
            if not valid:
                reject_counts[reject_reason] = reject_counts.get(reject_reason, 0) + 1
                last_reason = str(reject_reason)
                last_metrics = metrics
                continue
            metrics = dict(metrics)
            metrics["initial_target_distance"] = float(distance)
            selected_region = str(region)
            selected_bay_index = int(bay_index)
            self.initial_sampling_diagnostics.update(
                {
                    "initial_spawn_region": selected_region,
                    "initial_bay_index": selected_bay_index,
                    "initial_target_distance": float(distance),
                    "reset_rule_pose_attempt_count": int(attempt_count),
                    "reset_candidate_bank_valid_count": 1,
                    "reset_feasible_mask_available": bool(
                        float(metrics.get("mask_max", 0.0))
                        > float(metrics.get("mask_required", self._reset_mask_threshold(stage)))
                    ),
                    "reset_candidate_reject_separation_count": int(
                        reject_counts.get("separation", 0)
                    ),
                    "reset_candidate_reject_heading_count": int(
                        reject_counts.get("heading", 0)
                    ),
                }
            )
            return self._finalize_reset_candidate(
                state,
                "{}_initial".format(selected_region),
                False,
                stage,
                task_family,
                metrics,
                reject_counts,
            )
        raise ResetInitialStateError(
            "no reset-viable rule-scene initial state for scene seed {} after {} attempts; "
            "last reason {} mask {:.3f}".format(
                self.scene.metadata["seed"],
                attempt_count,
                last_reason,
                float(last_metrics.get("mask_max", 0.0)),
            )
        )

    def _sample_initial_state(self):
        if self._is_rule_scene():
            return self._sample_rule_scene_initial_state()
        stage = int(np.clip(self._active_stage, 1, 4))
        goal = self.slot
        index = stage - 1
        distance_range = self.config.stage_distance_ranges[index]
        lateral_range = float(self.config.stage_lateral_ranges[index])
        heading_range = math.radians(self.config.stage_heading_ranges_deg[index])
        phi_range = math.radians(self.config.stage_phi_ranges_deg[index])
        task_family = self._task_family_from_scene()
        sampling_diagnostics = {"initial_task_family": str(task_family)}
        scenario = {
            1: "near_goal",
            2: "near_goal_obstacles",
            3: "poor_terminal_pose",
            4: "recovery",
        }[stage]

        goal_orientation_mode = self.scene.metadata.get("goal_orientation_mode", "head_in")
        if goal_orientation_mode != "head_in":
            raise ResetInitialStateError(
                "unsupported goal orientation mode {} for scene seed {}".format(
                    goal_orientation_mode,
                    self.scene.metadata["seed"],
                )
            )
        axis = np.asarray(
            [math.cos(goal.theta_goal), math.sin(goal.theta_goal)],
            dtype=np.float64,
        )
        normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
        ref_center = np.asarray(goal.center)
        effective_lateral_range = lateral_range

        sampling_diagnostics.update(
            {
                "initial_distance_min": float(distance_range[0]),
                "initial_distance_max": float(distance_range[1]),
                "initial_lateral_range": float(effective_lateral_range),
                "initial_heading_range_deg": float(math.degrees(heading_range)),
                "initial_phi_range_deg": float(math.degrees(phi_range)),
            }
        )
        self.initial_sampling_diagnostics = dict(sampling_diagnostics)

        fallback_used = False
        bank = self._build_reset_candidate_bank(stage, goal, axis, normal)
        self.initial_sampling_diagnostics.update(
            {
                "reset_candidate_bank_size": int(bank.get("size", 0)),
                "reset_candidate_bank_valid_count": int(
                    bank.get("valid_count", 0)
                ),
                "reset_candidate_selected_clearance_bucket": "",
                "reset_candidate_selected_pose_bucket": "",
                "reset_candidate_selected_mask_max": 0.0,
                "reset_candidate_bank_empty": bool(
                    int(bank.get("valid_count", 0)) <= 0
                ),
            }
        )
        reject_counts = dict(bank.get("reject_counts", self._reset_reject_counts()))
        selected_candidate = self._select_reset_candidate_from_bank(bank)
        if selected_candidate is not None:
            return self._finalize_reset_candidate(
                selected_candidate["state"],
                selected_candidate["scenario"],
                fallback_used,
                stage,
                task_family,
                selected_candidate["metrics"],
                reject_counts,
            )

        for _ in range(max(1, int(self.config.initial_sampling_attempts))):
            distance = self.rng.uniform(*distance_range)
            lateral = self.rng.uniform(-effective_lateral_range, effective_lateral_range)
            heading_error = self.rng.uniform(-heading_range, heading_range)
            phi = self.rng.uniform(-phi_range, phi_range)

            if stage == 3:
                pose_mode = int(self.rng.integers(0, 3))
                if pose_mode == 0:
                    min_heading = math.radians(self.config.poor_pose_min_heading_deg)
                    heading_error = math.copysign(
                        self.rng.uniform(min_heading, heading_range),
                        self.rng.choice((-1.0, 1.0)),
                    )
                    scenario = "poor_terminal_heading"
                elif pose_mode == 1:
                    min_lateral = min(
                        float(self.config.poor_pose_min_lateral),
                        effective_lateral_range,
                    )
                    lateral = math.copysign(
                        self.rng.uniform(min_lateral, effective_lateral_range),
                        self.rng.choice((-1.0, 1.0)),
                    )
                    scenario = "poor_terminal_lateral"
                else:
                    min_phi = math.radians(self.config.poor_pose_min_abs_phi_deg)
                    phi = math.copysign(
                        self.rng.uniform(min_phi, phi_range),
                        self.rng.choice((-1.0, 1.0)),
                    )
                    scenario = "poor_terminal_articulation"

            if stage == 4:
                min_phi = math.radians(self.config.recovery_min_abs_phi_deg)
                lateral = math.copysign(
                    self.rng.uniform(min(1.8, effective_lateral_range), effective_lateral_range),
                    self.rng.choice((-1.0, 1.0)),
                )
                phi = math.copysign(
                    self.rng.uniform(min_phi, phi_range),
                    self.rng.choice((-1.0, 1.0)),
                )

            center = ref_center - distance * axis + lateral * normal
            theta_front = wrap_to_pi(goal.theta_goal + heading_error)
            state = ArticulatedState(
                x_front=float(center[0]),
                y_front=float(center[1]),
                theta_front=float(theta_front),
                theta_rear=float(wrap_to_pi(theta_front - phi)),
            )
            valid, reject_reason, metrics = self._reset_candidate_viability(
                state,
                stage=stage,
            )
            if not valid:
                reject_counts[reject_reason] = reject_counts.get(reject_reason, 0) + 1
                continue
            return self._finalize_reset_candidate(
                state,
                scenario,
                fallback_used,
                stage,
                task_family,
                metrics,
                reject_counts,
            )

        if stage == 4:
            fallback_used = True
            recovery_candidates = self._structured_recovery_state(goal, axis, normal)
            state = None
            metrics = {}
            reject_reason = ""
            if recovery_candidates:
                for candidate_index in self.rng.permutation(len(recovery_candidates)):
                    candidate_state, _ = recovery_candidates[int(candidate_index)]
                    valid, reject_reason, candidate_metrics = (
                        self._reset_candidate_viability(
                            candidate_state,
                            stage=stage,
                        )
                    )
                    if not valid:
                        reject_counts[reject_reason] = (
                            reject_counts.get(reject_reason, 0) + 1
                        )
                        continue
                    state = candidate_state
                    metrics = candidate_metrics
                    break
            if state is None:
                raise ResetInitialStateError(
                    "no reset-viable near-obstacle recovery state for scene seed {}".format(
                        self.scene.metadata["seed"]
                    )
                )
            return self._finalize_reset_candidate(
                state,
                "recovery",
                fallback_used,
                stage,
                task_family,
                metrics,
                reject_counts,
            )

        fallback_used = True
        center = ref_center - 6.0 * axis
        state = ArticulatedState(
            x_front=float(center[0]),
            y_front=float(center[1]),
            theta_front=float(goal.theta_goal),
            theta_rear=float(goal.theta_goal),
        )
        valid, reject_reason, metrics = self._reset_candidate_viability(
            state,
            stage=stage,
        )
        if not valid:
            raise ResetInitialStateError(
                "fallback initial state failed reset viability for scene seed {} mode {} stage {} reason {}".format(
                    self.scene.metadata["seed"],
                    goal_orientation_mode,
                    stage,
                    reject_reason,
                )
            )
        return self._finalize_reset_candidate(
            state,
            scenario + "_fallback",
            fallback_used,
            stage,
            task_family,
            metrics,
            reject_counts,
        )

    def _boxes_and_metrics(self):
        front_box, rear_box = self.vehicle_model.body_boxes(self.state)
        target_front = self.slot.front_box()
        target_rear = self.vehicle_model.target_rear_box(
            self.slot.x_goal,
            self.slot.y_goal,
            self.slot.theta_goal,
        )
        front_overlap = overlap_ratio(front_box, target_front)
        rear_overlap = overlap_ratio(rear_box, target_rear)
        heading_error = float(wrap_to_pi(self.state.theta_front - self.slot.theta_goal))
        rear_heading_error = float(wrap_to_pi(self.state.theta_rear - self.slot.theta_goal))
        distance = math.hypot(
            self.state.x_front - self.slot.x_goal,
            self.state.y_front - self.slot.y_goal,
        )
        return {
            "front_box": front_box,
            "rear_box": rear_box,
            "front_overlap": front_overlap,
            "rear_overlap": rear_overlap,
            "heading_error": heading_error,
            "rear_heading_error": rear_heading_error,
            "distance_to_goal": distance,
        }

    def _update_sensors_and_mask(self):
        front_m, rear_m = self.lidar.observe(
            self.state,
            self.vehicle_model,
            self.scene,
            normalize=False,
        )
        self.last_front_lidar_m = front_m
        self.last_rear_lidar_m = rear_m
        self.current_mask = self.action_mask.compute_mask(
            self.state.phi,
            front_m,
            rear_m,
        )
        normal_mask = np.asarray(self.current_mask, dtype=np.float32)
        dwa_enabled = bool(getattr(self.config, "enable_dwa_recovery", False))
        normal_or_floor_mask, self.current_mask_floor_info = self._mask_floor_state(
            normal_mask,
            allow_floor=not dwa_enabled,
        )
        self.current_normal_mask = normal_mask
        (
            self.current_mask,
            self.current_recovery_mask,
            self.current_dwa_recovery_result,
        ) = self._effective_mask_from_recovery(
            normal_mask if dwa_enabled else normal_or_floor_mask,
            self.current_mask_floor_info,
        )

    def _observation(self, metrics=None):
        if metrics is None:
            metrics = self._boxes_and_metrics()
        slot_error = self.slot.position_error_in_slot_frame(
            self.state.x_front,
            self.state.y_front,
        )
        heading_error = metrics["heading_error"]
        corners = self.slot.target_corners_in_ego_frame(
            self.state.x_front,
            self.state.y_front,
            self.state.theta_front,
        ).reshape(-1)
        slot_features = np.concatenate(
            [
                slot_error,
                np.asarray(
                    [
                        math.cos(heading_error),
                        math.sin(heading_error),
                        metrics["front_overlap"],
                    ],
                    dtype=np.float32,
                ),
                corners,
            ]
        )
        p = self.vehicle_params
        phi = self.state.phi
        vehicle_features = np.asarray(
            [
                self.state.v / p.parking_v_max,
                phi / p.phi_max,
                self.state.phi_dot / p.phi_dot_max,
                math.cos(phi),
                math.sin(phi),
                self.prev_gear_in_obs,
            ],
            dtype=np.float32,
        )
        rear_lidar = self.last_rear_lidar_m
        if str(getattr(self.config, "rear_lidar_observation_mode", "normal")) == "zero":
            rear_lidar = np.zeros_like(self.last_rear_lidar_m)
        lidar_features = np.concatenate(
            [
                self.last_front_lidar_m / p.lidar_range,
                rear_lidar / p.lidar_range,
            ]
        ).astype(np.float32)
        mask_features = self.current_mask.reshape(-1).astype(np.float32)
        if bool(getattr(self.config, "disable_mask_observation", False)):
            mask_features = np.zeros_like(mask_features)
        observation = np.concatenate(
            [
                slot_features.astype(np.float32),
                vehicle_features,
                lidar_features,
                mask_features,
            ]
        )
        if observation.shape != (self.OBS_DIM,):
            raise RuntimeError("unexpected observation shape {}".format(observation.shape))
        return observation

    def _decode_action_without_mask_execution(
        self,
        raw_action,
        policy_decoded,
        prev_motion_gear,
        prev_gear_in_obs,
        mask_for_action,
    ):
        raw = np.clip(np.asarray(raw_action, dtype=np.float32), -1.0, 1.0)
        p = self.vehicle_params
        deadband = float(getattr(self.config, "gear_deadband", 0.10))
        phi_dot_exec = self.action_mask._decode_phi_dot(
            raw[1],
            self.state.phi,
            p.dt,
        )
        if abs(raw[0]) < deadband:
            gear = STOP_GEAR if prev_motion_gear is None else prev_motion_gear
            rho = 0.0
            v_exec = 0.0
        elif raw[0] >= 0.0:
            gear = FORWARD_GEAR
            rho = abs(float(raw[0]))
            v_exec = rho * float(p.parking_v_forward_max)
        else:
            gear = REVERSE_GEAR
            rho = abs(float(raw[0]))
            v_exec = -rho * float(p.parking_v_reverse_max)

        motion_eps = 1e-8
        if abs(v_exec) > motion_eps:
            new_motion_gear = FORWARD_GEAR if v_exec > 0.0 else REVERSE_GEAR
        else:
            new_motion_gear = prev_motion_gear
        if new_motion_gear is None:
            new_prev_gear_in_obs = 0.0
        elif new_motion_gear == FORWARD_GEAR:
            new_prev_gear_in_obs = 1.0
        else:
            new_prev_gear_in_obs = -1.0

        return {
            "v_exec": float(v_exec),
            "phi_dot_exec": float(phi_dot_exec),
            "gear": int(gear),
            "rho": float(rho),
            "r_raw": float(policy_decoded.get("r_raw", 0.0)),
            "r_exec": float(policy_decoded.get("r_raw", 0.0)),
            "r_max": float(np.max(mask_for_action)) if mask_for_action.size else 0.0,
            "forced_stop": False,
            "clip_ratio": 0.0,
            "mask_cost": float(policy_decoded.get("mask_cost", 0.0)),
            "prev_motion_gear": new_motion_gear,
            "prev_gear_in_obs": float(new_prev_gear_in_obs),
        }

    def reset(self, seed=None, replay_case=None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        replay_state, replay_scenario, replay_fallback, replay_diagnostics = (
            self._sample_hard_case_replay_state(replay_case)
        )
        if replay_state is not None:
            self.state = replay_state
            scenario_type = replay_scenario
            fallback_used = replay_fallback
            self.initial_sampling_diagnostics = dict(replay_diagnostics)
            self.episode_index += 1
            scene_failures = []
        else:
            max_scene_attempts = max(
                1,
                int(getattr(self.config, "reset_scene_retry_count", 1)),
            )
            start_episode_index = int(self.episode_index)
            scene_failures = []
            last_error = None
            for _ in range(max_scene_attempts):
                scene_episode_index = int(self.episode_index)
                self.scene = self.scene_pool.get(scene_episode_index)
                self.slot = self.scene.slot
                self.step_count = 0
                try:
                    self.state, scenario_type, fallback_used = (
                        self._sample_initial_state()
                    )
                except ResetInitialStateError as exc:
                    last_error = exc
                    failed_seed = int(self.scene.metadata["seed"])
                    failed_family = self._task_family_from_scene()
                    scene_failures.append(
                        {
                            "seed": failed_seed,
                            "task_family": failed_family,
                            "requested_scene_type": str(
                                self.scene.metadata.get("requested_scene_type", "")
                            ),
                            "reason": str(exc),
                        }
                    )
                    replace_scene = getattr(self.scene_pool, "replace", None)
                    if replace_scene is None:
                        self.episode_index = scene_episode_index + 1
                        continue
                    try:
                        replace_scene(
                            scene_episode_index,
                            task_family=failed_family,
                        )
                    except RuntimeError as replace_exc:
                        last_error = replace_exc
                        self.episode_index = scene_episode_index + 1
                    continue
                self.episode_index = scene_episode_index + 1
                break
            else:
                failed_seeds = ",".join(
                    str(item["seed"]) for item in scene_failures[-5:]
                )
                raise RuntimeError(
                    "reset viability failed after {} scene seeds from episode index {}; "
                    "recent failed seeds [{}]; last failure: {}".format(
                        len(scene_failures),
                        start_episode_index,
                        failed_seeds,
                        last_error,
                    )
                )
            if replay_diagnostics.get("hard_case_replay_attempted", False):
                self.initial_sampling_diagnostics.update(replay_diagnostics)
        if scene_failures:
            last_failure = scene_failures[-1]
            self.initial_sampling_diagnostics.update(
                {
                    "reset_scene_retry_count": int(len(scene_failures)),
                    "reset_scene_last_failed_seed": int(last_failure["seed"]),
                    "reset_scene_last_failed_family": str(
                        last_failure["task_family"]
                    ),
                    "reset_scene_last_failure": str(last_failure["reason"]),
                    "reset_scene_success_seed": int(self.scene.metadata["seed"]),
                }
            )
        else:
            self.initial_sampling_diagnostics.update(
                {
                    "reset_scene_retry_count": 0,
                    "reset_scene_last_failed_seed": -1,
                    "reset_scene_last_failed_family": "",
                    "reset_scene_last_failure": "",
                    "reset_scene_success_seed": int(self.scene.metadata["seed"]),
                }
            )
        self.scenario_type = str(scenario_type)
        self.prev_motion_gear = None
        self.prev_gear_in_obs = 0.0
        self.dwa_forced_stop_streak = 0
        self.dwa_no_progress_streak = 0
        self.dwa_deadlock_streak = 0
        self.last_dwa_info = self._empty_dwa_info()
        self._reset_hope_teacher()
        metrics = self._boxes_and_metrics()
        self._last_progress_metrics = self._progress_metrics(metrics)
        self.reward_model.reset(
            initial_distance=metrics["distance_to_goal"],
            initial_overlap=metrics["front_overlap"],
            initial_heading_error=metrics["heading_error"],
        )
        self.hybrid_reward.reset(self.scene, self.state, self.slot)
        self.rs_potential.reset()
        self._update_sensors_and_mask()
        obs = self._observation(metrics)
        info = self._base_info(metrics)
        info["scenario_type"] = str(self.scenario_type)
        info["scene_seed"] = int(self.scene.metadata["seed"])
        info["goal_orientation_mode"] = str(
            self.scene.metadata.get("goal_orientation_mode", "")
        )
        info["fallback_used"] = bool(fallback_used)
        info["initial_collision"] = self._state_collides(self.state)
        info["task_family"] = self._task_family_from_scene()
        info["initial_degenerate_mask"] = bool(
            self.current_mask_floor_info["mask_degenerate"]
        )
        info["initial_mask_floor_applied"] = bool(
            self.current_mask_floor_info["mask_floor_applied"]
        )
        info["initial_mask_max_before_floor"] = float(
            self.current_mask_floor_info["mask_max_before_floor"]
        )
        info["initial_mask_all_zero_before_floor"] = bool(
            self.current_mask_floor_info["mask_all_zero_before_floor"]
        )
        info.update(self.initial_sampling_diagnostics)
        for key in (
            "scene_type",
            "requested_scene_type",
            "clearance_bucket",
            "approach_side_bucket",
            "scene_complexity_bucket",
            "difficulty_label",
            "topology_variant",
            "branch_side_mode",
            "local_complexity_variant",
            "corridor_width",
            "bay_count",
            "bay_width",
            "bay_depth",
            "wall_thickness",
            "initial_spawn_mode",
            "requested_initial_spawn_mode",
            "target_bay_index",
            "target_heading_into_bay",
            "corridor_outer_wall_exists",
            "bottom_wall_exists",
            "partition_wall_count",
            "side_wall_count",
            "world_length",
            "world_width",
            "boundary_wall_thickness",
            "boundary_wall_count",
            "truck_length",
            "truck_width",
            "truck_center",
            "truck_heading",
            "truck_in_front",
            "truck_perpendicular",
            "discrete_obstacle_count_requested",
            "discrete_obstacle_count",
            "obstacle_candidate_count",
            "obstacle_sampling_checks",
            "obstacle_exclusion_valid",
            "obstacle_count",
            "scene_generation_attempts",
            "nominal_target_collision",
            "nominal_target_front_in_bay",
            "nominal_target_rear_in_bay",
            "nominal_target_clearance_m",
            "success_neighborhood_sample_count",
            "success_neighborhood_collision_free_count",
            "success_neighborhood_feasible_count",
            "reset_geometry_candidate_count",
            "reset_geometry_recovery_band_count",
            "constructed_obstacle_feature_count",
            "constructed_wall_feature_count",
            "constructed_obstacle_labels",
            "parked_vehicle_count",
            "parked_vehicle_labels",
            "parked_vehicle_headings",
            "scene_generation_attempt_count",
        ):
            if key in self.scene.metadata:
                info[key] = self.scene.metadata[key]
        return obs, info

    def _out_of_bounds(self, front_box, rear_box):
        xmin, ymin, xmax, ymax = self.scene.world_bounds
        for polygon in (front_box, rear_box):
            bx0, by0, bx1, by1 = polygon.bounds
            if bx0 < xmin or by0 < ymin or bx1 > xmax or by1 > ymax:
                return True
        return False

    def _base_info(self, metrics):
        min_lidar = float(
            min(np.min(self.last_front_lidar_m), np.min(self.last_rear_lidar_m))
        )
        info = {
            "front_overlap": float(metrics["front_overlap"]),
            "best_front_overlap": float(self.reward_model.best_front_overlap),
            "rear_body_overlap": float(metrics["rear_overlap"]),
            "heading_error_deg": math.degrees(abs(metrics["heading_error"])),
            "rear_heading_error_deg": math.degrees(abs(metrics["rear_heading_error"])),
            "distance_to_goal": float(metrics["distance_to_goal"]),
            "phi": float(self.state.phi),
            "min_lidar_distance": min_lidar,
            "hybrid_astar_valid_rate": 1.0 if self.hybrid_reward.valid else 0.0,
            "planner_valid": self.hybrid_reward.valid,
            "planner_fallback_used": self.hybrid_reward._fallback_used,
            "planner_fail_reason": str(self.hybrid_reward.fail_reason),
            "scene_seed": int(self.scene.metadata["seed"]),
            "task_family": self._task_family_from_scene(),
            "goal_orientation_mode": str(
                self.scene.metadata.get("goal_orientation_mode", "")
            ),
            "scenario_type": str(self.scenario_type),
            "max_safe_ratio": 0.0,
            "raw_safe_ratio": 0.0,
            "exec_safe_ratio": 0.0,
            "forced_stop": False,
            "degenerate_mask": False,
            "mask_floor_applied": False,
            "mask_all_zero_before_floor": False,
            "mask_max_before_floor": 0.0,
            "collision_after_mask_floor": False,
            "success_after_mask_floor": False,
            "clip_ratio": 0.0,
            "mask_cost": 0.0,
            "gear": 0,
        }
        info.update(self.last_dwa_info if self.last_dwa_info else self._empty_dwa_info())
        info.update(
            {
                key: value
                for key, value in self.rs_potential.diagnostics().items()
                if not key.startswith("planner_")
            }
        )
        info.update(self.hope_teacher_info)
        info.update(self.guide_step_info)
        info["hope_failure_aggregation_recorded"] = False
        return info

    def step(self, raw_action):
        if self.state is None:
            raise RuntimeError("reset() must be called before step()")
        previous_state = self.state
        policy_raw_action = np.clip(
            np.asarray(raw_action, dtype=np.float32),
            -1.0,
            1.0,
        )
        execution_raw_action = policy_raw_action.copy()
        mask_for_action = np.asarray(self.current_mask, dtype=np.float32).copy()
        normal_mask_for_action = (
            np.asarray(self.current_normal_mask, dtype=np.float32).copy()
            if self.current_normal_mask is not None
            else mask_for_action.copy()
        )
        recovery_mask_for_action = (
            np.asarray(self.current_recovery_mask, dtype=np.float32).copy()
            if self.current_recovery_mask is not None
            else self._empty_recovery_mask(mask_for_action)
        )
        mask_floor_info = dict(self.current_mask_floor_info)
        if (
            bool(mask_floor_info.get("mask_all_zero_before_floor", False))
            and mask_for_action.size
            and float(np.max(mask_for_action))
            <= float(getattr(self.config, "dwa_all_zero_eps", 1e-3))
        ):
            normal_mask_for_action = mask_for_action.copy()
        policy_decoded = self.action_mask.decode_safe_speed_and_cost(
            policy_raw_action,
            mask_for_action,
            self.state.phi,
            dt=self.vehicle_params.dt,
            prev_motion_gear=self.prev_motion_gear,
            config=self.config,
        )
        all_zero_trigger, low_safe_trigger, normal_rmax_now = self._dwa_mask_triggers(
            normal_mask_for_action,
            mask_floor_info,
        )
        effective_rmax_now = float(np.max(mask_for_action)) if mask_for_action.size else 0.0
        policy_invalid_trigger = bool(policy_decoded["forced_stop"])
        dwa_triggered = False
        dwa_override_applied = False
        dwa_result = (
            self.current_dwa_recovery_result
            if self.current_dwa_recovery_result is not None
            else DWAResult(reason="not_triggered")
        )
        dwa_enabled = bool(getattr(self.config, "enable_dwa_recovery", False))
        if dwa_enabled:
            if self.dwa_recovery is None:
                self.dwa_recovery = DWARecoveryController(self.config)
            dwa_triggered = bool(all_zero_trigger or low_safe_trigger)
            if (
                dwa_triggered
                and str(getattr(dwa_result, "reason", "")) in ("not_triggered", "disabled")
            ):
                dwa_result = self.dwa_recovery.run(
                    "unlock",
                    self.state,
                    self.slot,
                    self.scene,
                    self.vehicle_model,
                    self.lidar,
                    self.action_mask,
                    normal_mask_for_action,
                    self.last_front_lidar_m,
                    self.last_rear_lidar_m,
                    self.prev_motion_gear,
                    self.config,
                    reason="all_zero_mask" if all_zero_trigger else "low_safe_mask",
                )
                if dwa_result.recovery_mask is not None:
                    recovery_mask_for_action = np.asarray(
                        dwa_result.recovery_mask,
                        dtype=np.float32,
                    )
                    mask_for_action = np.maximum(
                        normal_mask_for_action,
                        recovery_mask_for_action,
                    ).astype(np.float32, copy=False)
                    policy_decoded = self.action_mask.decode_safe_speed_and_cost(
                        policy_raw_action,
                        mask_for_action,
                        self.state.phi,
                        dt=self.vehicle_params.dt,
                        prev_motion_gear=self.prev_motion_gear,
                        config=self.config,
                    )
                    effective_rmax_now = (
                        float(np.max(mask_for_action)) if mask_for_action.size else 0.0
                    )
                    policy_invalid_trigger = bool(policy_decoded["forced_stop"])
            if policy_invalid_trigger and not (all_zero_trigger or low_safe_trigger):
                dwa_triggered = True
                if bool(getattr(self.config, "dwa_override_policy_action", False)):
                    dwa_result = self.dwa_recovery.run(
                        "local",
                        self.state,
                        self.slot,
                        self.scene,
                        self.vehicle_model,
                        self.lidar,
                        self.action_mask,
                        mask_for_action,
                        self.last_front_lidar_m,
                        self.last_rear_lidar_m,
                        self.prev_motion_gear,
                        self.config,
                        reason="policy_forced_stop",
                    )
                else:
                    dwa_result = DWAResult(
                        used=False,
                        mode="none",
                        reason="policy_forced_stop_no_override",
                    )
            recovery_mode = str(
                getattr(self.config, "dwa_recovery_mode", "teacher_override")
            )
            override_requested = False
            if recovery_mode == "teacher_override":
                override_requested = bool(dwa_result.used)
            elif recovery_mode == "policy_with_recovery_mask":
                override_requested = bool(
                    policy_decoded["forced_stop"] and dwa_result.used
                )
            elif recovery_mode == "recovery_mask_only":
                override_requested = False
            if (
                bool(getattr(self.config, "dwa_override_policy_action", False))
                and bool(override_requested)
                and bool(getattr(dwa_result, "teacher_action_valid", False))
                and dwa_result.raw_action is not None
            ):
                execution_raw_action = np.clip(
                    np.asarray(dwa_result.raw_action, dtype=np.float32),
                    -1.0,
                    1.0,
                )
                dwa_override_applied = True
        else:
            all_zero_trigger = False
            low_safe_trigger = False
            policy_invalid_trigger = False
            dwa_result = DWAResult(reason="disabled")

        dwa_policy_loss_weight = 1.0
        if dwa_override_applied:
            dwa_policy_loss_weight = float(
                getattr(self.config, "dwa_override_policy_loss_weight", 0.0)
            )
            dwa_policy_loss_weight = float(np.clip(dwa_policy_loss_weight, 0.0, 1.0))

        decode_prev_motion_gear = self.prev_motion_gear
        decode_prev_gear_in_obs = self.prev_gear_in_obs
        if bool(getattr(self.config, "disable_action_mask_execution", False)):
            decoded = self._decode_action_without_mask_execution(
                execution_raw_action,
                policy_decoded,
                decode_prev_motion_gear,
                decode_prev_gear_in_obs,
                mask_for_action,
            )
        else:
            decoded = self.action_mask.decode_safe_speed_and_cost(
                execution_raw_action,
                mask_for_action,
                self.state.phi,
                dt=self.vehicle_params.dt,
                prev_motion_gear=decode_prev_motion_gear,
                config=self.config,
            )
        executed_action = np.asarray(
            [decoded["v_exec"], decoded["phi_dot_exec"]],
            dtype=np.float32,
        )
        self.state = self.vehicle_model.step(self.state, executed_action)
        self.step_count += 1
        self.prev_motion_gear = decoded["prev_motion_gear"]
        self.prev_gear_in_obs = decoded["prev_gear_in_obs"]
        metrics = self._boxes_and_metrics()

        collision = bool(
            self.scene.prepared_obstacles.intersects(metrics["front_box"])
            or self.scene.prepared_obstacles.intersects(metrics["rear_box"])
        )
        out_of_bounds = self._out_of_bounds(metrics["front_box"], metrics["rear_box"])
        articulation_violation = (
            abs(self.state.phi)
            > self.vehicle_params.phi_max + self.config.articulation_tolerance
        )
        success = (
            metrics["front_overlap"] >= self.config.success_overlap
            and abs(metrics["heading_error"]) <= self.config.success_heading_error
        )
        timeout = self.step_count >= self.config.max_steps
        self._update_dwa_stuck_counters(decoded, metrics)
        if bool(dwa_triggered and dwa_result.deadlock):
            self.dwa_deadlock_streak += 1
        else:
            self.dwa_deadlock_streak = 0
        deadlock = bool(
            bool(getattr(self.config, "enable_dwa_recovery", False))
            and bool(getattr(self.config, "dwa_enable_deadlock_termination", False))
            and self.dwa_deadlock_streak
            >= int(getattr(self.config, "dwa_deadlock_patience", 8))
        )
        failure = (
            collision
            or out_of_bounds
            or articulation_violation
            or deadlock
            or (timeout and not success)
        )
        if failure:
            success = False
        terminated = bool(
            success or collision or out_of_bounds or articulation_violation or deadlock
        )
        truncated = bool(timeout and not terminated)
        if success:
            failure_type = "success"
        elif collision:
            failure_type = "collision"
        elif deadlock:
            failure_type = "deadlock"
        elif timeout:
            failure_type = "timeout"
        elif out_of_bounds:
            failure_type = "out_of_bounds"
        elif articulation_violation:
            failure_type = "articulation"
        else:
            failure_type = ""
        mask_floor_applied = bool(mask_floor_info.get("mask_floor_applied", False))
        collision_after_mask_floor = bool(mask_floor_applied and collision)
        success_after_mask_floor = bool(mask_floor_applied and success)
        dwa_info = self._dwa_info_from_result(
            dwa_result,
            triggered=dwa_triggered,
            override_applied=dwa_override_applied,
            policy_invalid_trigger=policy_invalid_trigger,
            low_safe_trigger=low_safe_trigger,
            all_zero_trigger=all_zero_trigger,
            policy_raw_action=policy_raw_action,
            normal_mask_max=normal_rmax_now,
            recovery_mask=recovery_mask_for_action,
            effective_mask=mask_for_action,
            dwa_policy_loss_weight=dwa_policy_loss_weight,
        )
        dwa_info["deadlock"] = bool(deadlock)
        self.last_dwa_info = dwa_info

        rs_value, rs_info = self.rs_potential.step(
            self.scene,
            previous_state,
            self.state,
            self.slot,
        )
        if self.rs_potential.rs_latched:
            hybrid_value = 0.0
            hybrid_info = {
                "hybrid_astar_suppressed_by_rs": True,
            }
            planner_value = rs_value
            planner_source = "rs"
        elif (
            self.config.rs_potential_enabled
            and self.hybrid_reward.planner is None
        ):
            hybrid_value = 0.0
            hybrid_info = {
                "planner_valid": False,
                "planner_cost": 0.0,
                "planner_phi": 0.0,
                "planner_potential_reward": 0.0,
                "planner_fallback_used": False,
                "planner_fail_reason": "disabled",
            }
            planner_value = 0.0
            planner_source = "none"
        else:
            hybrid_value, hybrid_info = self.hybrid_reward.step(
                self.state.x_front,
                self.state.y_front,
                self.state.theta_front,
            )
            planner_value = hybrid_value
            planner_source = "hybrid_astar"
        reward, reward_components = self.reward_model.compute(
            front_overlap=metrics["front_overlap"],
            distance_to_goal=metrics["distance_to_goal"],
            heading_error=metrics["heading_error"],
            step_count=self.step_count,
            success=success,
            failure=failure,
            hybrid_reward=planner_value,
        )
        task_reward = float(reward)
        guide_value = 0.0
        self.guide_step_info = self._zero_guide_step_info()
        if self.hope_teacher is not None:
            guide_value, self.guide_step_info = self.hope_teacher.compute_guidance(
                self.hope_teacher_trajectory,
                previous_state,
                self.state,
                decoded["gear"],
                self.guide_weight_current,
            )
            reward = task_reward + float(guide_value)
        reward_components["hybrid_astar"] = float(hybrid_value)
        reward_components["rs_potential"] = (
            float(rs_value) if self.rs_potential.rs_latched else 0.0
        )
        reward_components["planner"] = float(planner_value)
        reward_components["planner_source"] = planner_source
        reward_components["task"] = float(task_reward)
        reward_components["hope_teacher"] = float(guide_value)
        reward_components["total"] = float(reward)
        self._update_sensors_and_mask()
        obs = self._observation(metrics)
        info = self._base_info(metrics)
        info.update(hybrid_info)
        info.update(
            rs_info
            if self.rs_potential.rs_latched
            else {
                key: value
                for key, value in rs_info.items()
                if not key.startswith("planner_")
            }
        )
        info.update(
            {
                "success": bool(success),
                "collision": bool(collision),
                "out_of_bounds": bool(out_of_bounds),
                "timeout": bool(timeout),
                "deadlock": bool(deadlock),
                "failure_type": str(failure_type),
                "articulation_limit_violation": bool(articulation_violation),
                "policy_raw_action": policy_raw_action,
                "raw_action": execution_raw_action,
                "executed_action": executed_action,
                "mask_safe_ratio": float(decoded["r_raw"]),
                "mask_safe_ratio_mean": float(np.mean(mask_for_action)),
                "mask_safe_ratio_min": float(np.min(mask_for_action)),
                "mask_zero_fraction": float(
                    np.mean(mask_for_action <= self.action_mask.min_safe_ratio)
                ),
                "mask_invalid_rate": 1.0 if decoded["forced_stop"] else 0.0,
                "selected_action_masked": bool(decoded["forced_stop"]),
                "speed_clip_rate": 1.0 if decoded["clip_ratio"] > 0.01 else 0.0,
                "reward_components": reward_components,
                "gear": int(decoded["gear"]),
                "policy_gear": int(policy_decoded["gear"]),
                "policy_forced_stop": bool(policy_decoded["forced_stop"]),
                "policy_raw_safe_ratio": float(policy_decoded["r_raw"]),
                "raw_safe_ratio": float(decoded["r_raw"]),
                "exec_safe_ratio": float(decoded["r_exec"]),
                "max_safe_ratio": float(decoded["r_max"]),
                "normal_mask_max": float(normal_rmax_now),
                "effective_mask_max": float(effective_rmax_now),
                "forced_stop": bool(decoded["forced_stop"]),
                "degenerate_mask": bool(mask_floor_info.get("mask_degenerate", False)),
                "mask_floor_applied": bool(mask_floor_applied),
                "mask_all_zero_before_floor": bool(
                    mask_floor_info.get("mask_all_zero_before_floor", False)
                ),
                "mask_max_before_floor": float(
                    mask_floor_info.get("mask_max_before_floor", 0.0)
                ),
                "collision_after_mask_floor": bool(collision_after_mask_floor),
                "success_after_mask_floor": bool(success_after_mask_floor),
                "clip_ratio": float(decoded["clip_ratio"]),
                "mask_cost": float(decoded["mask_cost"]),
                "planner_source": planner_source,
                "unsafe_no_action_mask_execution": bool(
                    getattr(self.config, "disable_action_mask_execution", False)
                ),
            }
        )
        if (
            bool(self.config.enable_failure_aggregation)
            and self.hope_teacher is not None
            and (terminated or truncated)
            and not success
        ):
            info["hope_failure_aggregation_recorded"] = bool(
                self.hope_teacher.record_failure(
                    self.scene,
                    self.state,
                    self.slot,
                    info,
                )
            )
        return obs, reward, terminated, truncated, info
