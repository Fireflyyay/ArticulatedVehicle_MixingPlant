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
from env.articulated_action_mask import ArticulatedActionMask
from env.geometry import overlap_ratio, wrap_to_pi
from env.hybrid_astar_reward import OptionalHybridAStarReward
from env.lidar import DualBodyLidar
from env.mixing_plant_scene import CachedScenePool, TASK_FAMILIES
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
        self.current_mask_floor_info = self._empty_mask_floor_info()
        self.last_front_lidar_m = None
        self.last_rear_lidar_m = None
        self.prev_motion_gear = None
        self.prev_gear_in_obs = 0.0
        self.scenario_type = ""
        self.initial_sampling_diagnostics = {}
        self.hope_teacher_trajectory = None
        self.hope_teacher_info = HopeTeacherAdapter.disabled_diagnostics()
        self.guide_step_info = self._zero_guide_step_info()
        self.guide_weight_current = 0.0
        self.guide_dropout_rate = 1.0
        self.guide_dropped = True

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

    def _mask_floor_state(self, mask):
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

    def _reset_candidate_viability(self, state, stage):
        if self._state_collides(state):
            return False, "collision", {}

        metrics = self._reset_candidate_metrics(state, stage)
        if float(metrics["mask_max"]) <= float(metrics["mask_required"]):
            return False, "mask", metrics

        if int(stage) == 4:
            min_clearance = float(
                getattr(self.config, "stage4_reset_min_body_clearance", 0.0)
            )
            metrics["stage4_min_body_clearance"] = min_clearance
            if min_clearance > 0.0 and float(metrics["body_clearance"]) < min_clearance:
                return False, "clearance", metrics
            if float(metrics["body_clearance"]) > float(
                self.config.recovery_max_body_clearance
            ):
                return False, "recovery_clearance", metrics
            if float(metrics["min_lidar"]) > float(self.config.recovery_max_lidar_distance):
                return False, "recovery_lidar", metrics

        return True, "", metrics

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
        """Search a small Bay-mouth boundary band after random sampling fails."""
        candidates = []
        for distance in np.arange(8.0, 12.01, 0.5):
            for lateral_abs in np.arange(3.5, 4.51, 0.25):
                for lateral_sign in (-1.0, 1.0):
                    for heading_abs in np.arange(30.0, 50.01, 5.0):
                        for heading_sign in (-1.0, 1.0):
                            for phi_deg in (-30.0, -24.0, -18.0, 18.0, 24.0, 30.0):
                                candidates.append(
                                    (
                                        float(distance),
                                        float(lateral_sign * lateral_abs),
                                        float(heading_sign * heading_abs),
                                        float(phi_deg),
                                    )
                                )
        for candidate_index in self.rng.permutation(len(candidates)):
            distance, lateral, heading_deg, phi_deg = candidates[candidate_index]
            center = np.asarray(goal.center) - distance * axis + lateral * normal
            theta_front = wrap_to_pi(goal.theta_goal + math.radians(heading_deg))
            state = ArticulatedState(
                x_front=float(center[0]),
                y_front=float(center[1]),
                theta_front=float(theta_front),
                theta_rear=float(wrap_to_pi(theta_front - math.radians(phi_deg))),
            )
            if self._valid_recovery_state(state):
                return state
        return None

    def _sample_initial_state(self):
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
        reject_counts = {
            "collision": 0,
            "mask": 0,
            "clearance": 0,
            "recovery_clearance": 0,
            "recovery_lidar": 0,
        }
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
            state = self._structured_recovery_state(goal, axis, normal)
            if state is None:
                raise ResetInitialStateError(
                    "no reset-viable near-obstacle recovery state for scene seed {}".format(
                        self.scene.metadata["seed"]
                    )
                )
            valid, reject_reason, metrics = self._reset_candidate_viability(
                state,
                stage=stage,
            )
            if not valid:
                raise ResetInitialStateError(
                    "structured recovery state failed reset viability for scene seed {} reason {}".format(
                        self.scene.metadata["seed"],
                        reject_reason,
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
        self.current_mask, self.current_mask_floor_info = self._mask_floor_state(
            self.current_mask
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
        lidar_features = np.concatenate(
            [
                self.last_front_lidar_m / p.lidar_range,
                self.last_rear_lidar_m / p.lidar_range,
            ]
        ).astype(np.float32)
        observation = np.concatenate(
            [
                slot_features.astype(np.float32),
                vehicle_features,
                lidar_features,
                self.current_mask.reshape(-1).astype(np.float32),
            ]
        )
        if observation.shape != (self.OBS_DIM,):
            raise RuntimeError("unexpected observation shape {}".format(observation.shape))
        return observation

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
        self._reset_hope_teacher()
        metrics = self._boxes_and_metrics()
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
            "clearance_bucket",
            "approach_side_bucket",
            "scene_complexity_bucket",
            "difficulty_label",
            "nominal_target_collision",
            "nominal_target_front_in_bay",
            "nominal_target_rear_in_bay",
            "nominal_target_clearance_m",
            "success_neighborhood_sample_count",
            "success_neighborhood_collision_free_count",
            "success_neighborhood_feasible_count",
            "constructed_obstacle_feature_count",
            "constructed_wall_feature_count",
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
        mask_for_action = np.asarray(self.current_mask, dtype=np.float32).copy()
        mask_floor_info = dict(self.current_mask_floor_info)
        decoded = self.action_mask.decode_safe_speed_and_cost(
            raw_action,
            mask_for_action,
            self.state.phi,
            dt=self.vehicle_params.dt,
            prev_motion_gear=self.prev_motion_gear,
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
        failure = (
            collision
            or out_of_bounds
            or articulation_violation
            or (timeout and not success)
        )
        if failure:
            success = False
        terminated = bool(success or collision or out_of_bounds or articulation_violation)
        truncated = bool(timeout and not terminated)
        mask_floor_applied = bool(mask_floor_info.get("mask_floor_applied", False))
        collision_after_mask_floor = bool(mask_floor_applied and collision)
        success_after_mask_floor = bool(mask_floor_applied and success)

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
                "articulation_limit_violation": bool(articulation_violation),
                "raw_action": np.clip(np.asarray(raw_action, dtype=np.float32), -1.0, 1.0),
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
                "raw_safe_ratio": float(decoded["r_raw"]),
                "exec_safe_ratio": float(decoded["r_exec"]),
                "max_safe_ratio": float(decoded["r_max"]),
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
