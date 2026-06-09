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
from env.mixing_plant_scene import CachedScenePool
from env.reward import LocalParkingReward
from env.vehicle import ArticulatedState, ArticulatedVehicleModel


class BoxSpace:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = low
        self.high = high
        self.shape = tuple(shape)
        self.dtype = dtype

    def sample(self, rng=None):
        generator = np.random.default_rng() if rng is None else rng
        return generator.uniform(self.low, self.high, size=self.shape).astype(self.dtype)


class LocalParkingEnv:
    SLOT_FEATURE_DIM = 13
    VEHICLE_FEATURE_DIM = 5
    LIDAR_FEATURE_DIM = 108
    MASK_FEATURE_DIM = 22
    OBS_DIM = 148
    OBS_SLICES = {
        "slot": slice(0, 13),
        "vehicle": slice(13, 18),
        "lidar": slice(18, 126),
        "mask": slice(126, 148),
    }

    def __init__(
        self,
        config=DEFAULT_ENV_CONFIG,
        vehicle_params=DEFAULT_VEHICLE_PARAMS,
        action_mask=None,
        action_mask_path=None,
        hybrid_planner=None,
        scene_config=DEFAULT_SCENE_CONFIG,
        seed=0,
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
        self.scene_pool = CachedScenePool(
            stage=config.curriculum_stage,
            pool_size=config.scene_pool_size,
            base_seed=int(seed),
            scene_config=scene_config,
        )
        self.hybrid_reward = OptionalHybridAStarReward(
            planner=hybrid_planner if config.use_hybrid_astar else None
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
        self.last_front_lidar_m = None
        self.last_rear_lidar_m = None

    def _state_collides(self, state):
        front_box, rear_box = self.vehicle_model.body_boxes(state)
        return bool(
            self.scene.prepared_obstacles.intersects(front_box)
            or self.scene.prepared_obstacles.intersects(rear_box)
        )

    def _valid_recovery_state(self, state):
        if self._state_collides(state):
            return False
        front_box, rear_box = self.vehicle_model.body_boxes(state)
        clearance = min(
            front_box.distance(self.scene.obstacle_union),
            rear_box.distance(self.scene.obstacle_union),
        )
        if clearance > float(self.config.recovery_max_body_clearance):
            return False
        front_lidar, rear_lidar = self.lidar.observe(
            state,
            self.vehicle_model,
            self.scene,
            normalize=False,
        )
        return min(float(np.min(front_lidar)), float(np.min(rear_lidar))) <= float(
            self.config.recovery_max_lidar_distance
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
        stage = int(np.clip(self.config.curriculum_stage, 1, 4))
        goal = self.slot
        axis = np.asarray(
            [math.cos(goal.theta_goal), math.sin(goal.theta_goal)],
            dtype=np.float64,
        )
        normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
        index = stage - 1
        distance_range = self.config.stage_distance_ranges[index]
        lateral_range = float(self.config.stage_lateral_ranges[index])
        heading_range = math.radians(self.config.stage_heading_ranges_deg[index])
        phi_range = math.radians(self.config.stage_phi_ranges_deg[index])
        scenario = {
            1: "near_goal",
            2: "near_goal_obstacles",
            3: "poor_terminal_pose",
            4: "recovery",
        }[stage]

        for _ in range(max(1, int(self.config.initial_sampling_attempts))):
            distance = self.rng.uniform(*distance_range)
            lateral = self.rng.uniform(-lateral_range, lateral_range)
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
                        lateral_range,
                    )
                    lateral = math.copysign(
                        self.rng.uniform(min_lateral, lateral_range),
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
                    self.rng.uniform(min(1.8, lateral_range), lateral_range),
                    self.rng.choice((-1.0, 1.0)),
                )
                phi = math.copysign(
                    self.rng.uniform(min_phi, phi_range),
                    self.rng.choice((-1.0, 1.0)),
                )

            center = np.asarray(goal.center) - distance * axis + lateral * normal
            theta_front = wrap_to_pi(goal.theta_goal + heading_error)
            state = ArticulatedState(
                x_front=float(center[0]),
                y_front=float(center[1]),
                theta_front=float(theta_front),
                theta_rear=float(wrap_to_pi(theta_front - phi)),
            )
            if stage == 4:
                if not self._valid_recovery_state(state):
                    continue
                return state, scenario
            if self._state_collides(state):
                continue
            return state, scenario

        if stage == 4:
            state = self._structured_recovery_state(goal, axis, normal)
            if state is None:
                raise RuntimeError(
                    "no collision-free near-obstacle recovery state for scene seed {}".format(
                        self.scene.metadata["seed"]
                    )
                )
            return state, "recovery"

        # Deterministic open-corridor fallback; scene construction guarantees it.
        center = np.asarray(goal.center) - 6.0 * axis
        return (
            ArticulatedState(
                x_front=float(center[0]),
                y_front=float(center[1]),
                theta_front=float(goal.theta_goal),
                theta_rear=float(goal.theta_goal),
            ),
            scenario + "_fallback",
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

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.scene = self.scene_pool.get(self.episode_index)
        self.episode_index += 1
        self.slot = self.scene.slot
        self.step_count = 0
        self.state, scenario_type = self._sample_initial_state()
        metrics = self._boxes_and_metrics()
        self.reward_model.reset(
            initial_distance=metrics["distance_to_goal"],
            initial_overlap=metrics["front_overlap"],
            initial_heading_error=metrics["heading_error"],
        )
        self.hybrid_reward.reset(self.scene, self.state, self.slot)
        self._update_sensors_and_mask()
        obs = self._observation(metrics)
        info = self._base_info(metrics)
        info["scenario_type"] = scenario_type
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
        return {
            "front_overlap": float(metrics["front_overlap"]),
            "best_front_overlap": float(self.reward_model.best_front_overlap),
            "rear_body_overlap": float(metrics["rear_overlap"]),
            "heading_error_deg": math.degrees(abs(metrics["heading_error"])),
            "rear_heading_error_deg": math.degrees(abs(metrics["rear_heading_error"])),
            "distance_to_goal": float(metrics["distance_to_goal"]),
            "phi": float(self.state.phi),
            "min_lidar_distance": min_lidar,
            "hybrid_astar_valid_rate": 1.0 if self.hybrid_reward.valid else 0.0,
        }

    def step(self, raw_action):
        if self.state is None:
            raise RuntimeError("reset() must be called before step()")
        execution = self.action_mask.filter_and_clip_action(
            raw_action,
            self.current_mask,
            self.state.phi,
            dt=self.vehicle_params.dt,
        )
        self.state = self.vehicle_model.step(self.state, execution.executed_action)
        self.step_count += 1
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

        hybrid_value, hybrid_info = self.hybrid_reward.step(
            self.state.x_front,
            self.state.y_front,
        )
        reward, reward_components = self.reward_model.compute(
            front_overlap=metrics["front_overlap"],
            distance_to_goal=metrics["distance_to_goal"],
            heading_error=metrics["heading_error"],
            step_count=self.step_count,
            success=success,
            failure=failure,
            hybrid_reward=hybrid_value,
        )
        self._update_sensors_and_mask()
        obs = self._observation(metrics)
        info = self._base_info(metrics)
        info.update(hybrid_info)
        info.update(
            {
                "success": bool(success),
                "collision": bool(collision),
                "out_of_bounds": bool(out_of_bounds),
                "timeout": bool(timeout),
                "articulation_limit_violation": bool(articulation_violation),
                "raw_action": execution.raw_action.copy(),
                "decoded_action": execution.decoded_action.copy(),
                "executed_action": execution.executed_action.copy(),
                "mask_safe_ratio": float(execution.safe_ratio),
                "mask_safe_ratio_mean": float(np.mean(self.current_mask)),
                "mask_safe_ratio_min": float(np.min(self.current_mask)),
                "mask_zero_fraction": float(
                    np.mean(self.current_mask <= self.action_mask.min_safe_ratio)
                ),
                "mask_invalid_rate": 1.0 if execution.invalid else 0.0,
                "selected_action_masked": bool(execution.invalid),
                "speed_clip_rate": 1.0 if execution.speed_clipped else 0.0,
                "reward_components": reward_components,
            }
        )
        return obs, reward, terminated, truncated, info
