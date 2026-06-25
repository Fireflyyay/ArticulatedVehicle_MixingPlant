import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from env.articulated_action_mask import FORWARD_GEAR, REVERSE_GEAR
from env.geometry import overlap_ratio, wrap_to_pi


@dataclass
class DWAResult:
    used: bool = False
    mode: str = "none"
    reason: str = ""
    raw_action: Optional[np.ndarray] = None
    executed_action_preview: Optional[np.ndarray] = None
    candidate_count: int = 0
    valid_candidate_count: int = 0
    final_max_safe_ratio: float = 0.0
    best_score: object = None
    unlock_success: bool = False
    unlock_step: int = -1
    deadlock: bool = False


class DWARecoveryController:
    """Strict-mask local recovery for low safe-speed regions."""

    def __init__(self, config):
        self.config = config

    @staticmethod
    def empty(reason=""):
        return DWAResult(reason=str(reason))

    @staticmethod
    def _max_safe_ratio(mask):
        array = np.asarray(mask, dtype=np.float32)
        if array.size == 0:
            return 0.0
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        return float(np.max(np.clip(array, 0.0, 1.0)))

    @staticmethod
    def _phi_dot_candidates(action_mask):
        bins = np.asarray(action_mask.phi_dot_bins, dtype=np.float64).reshape(-1)
        values = np.concatenate([bins, np.asarray([0.0], dtype=np.float64)])
        return np.unique(values).astype(np.float64)

    @staticmethod
    def _out_of_bounds(scene, front_box, rear_box):
        xmin, ymin, xmax, ymax = scene.world_bounds
        for polygon in (front_box, rear_box):
            bx0, by0, bx1, by1 = polygon.bounds
            if bx0 < xmin or by0 < ymin or bx1 > xmax or by1 > ymax:
                return True
        return False

    def _state_invalid(self, state, scene, vehicle_model):
        p = vehicle_model.params
        articulation_violation = bool(
            abs(float(state.phi))
            > float(p.phi_max) + float(getattr(self.config, "articulation_tolerance", 0.0))
        )
        if articulation_violation:
            return True, None, None
        front_box, rear_box = vehicle_model.body_boxes(state)
        prepared = getattr(scene, "prepared_obstacles", None)
        collision = bool(
            prepared is not None
            and (
                prepared.intersects(front_box)
                or prepared.intersects(rear_box)
            )
        )
        out_of_bounds = self._out_of_bounds(scene, front_box, rear_box)
        return bool(collision or out_of_bounds), front_box, rear_box

    def _metrics(
        self,
        state,
        slot,
        vehicle_model,
        front_box=None,
        target_front_box=None,
    ):
        if front_box is None:
            front_box, _ = vehicle_model.body_boxes(state)
        if target_front_box is None:
            target_front_box = slot.front_box()
        front_overlap = overlap_ratio(front_box, target_front_box)
        heading_error = float(wrap_to_pi(state.theta_front - slot.theta_goal))
        distance = math.hypot(
            float(state.x_front) - float(slot.x_goal),
            float(state.y_front) - float(slot.y_goal),
        )
        return {
            "front_overlap": float(front_overlap),
            "heading_error": float(heading_error),
            "heading_score": float(math.cos(abs(heading_error))),
            "distance_to_goal": float(distance),
            "success": bool(
                front_overlap >= float(getattr(self.config, "success_overlap", 0.80))
                and abs(heading_error)
                <= float(getattr(self.config, "success_heading_error", math.radians(15.0)))
            ),
        }

    def _future_mask_ratio(self, state, scene, vehicle_model, lidar, action_mask):
        front_lidar, rear_lidar = lidar.observe(
            state,
            vehicle_model,
            scene,
            normalize=False,
        )
        future_mask = action_mask.compute_mask(
            state.phi,
            front_lidar,
            rear_lidar,
        )
        return self._max_safe_ratio(future_mask)

    def _simulate(
        self,
        state,
        action,
        slot,
        scene,
        vehicle_model,
        lidar,
        action_mask,
        need_metrics=True,
        track_success=True,
        target_front_box=None,
        stop_safe_ratio=None,
    ):
        horizon = max(1, int(getattr(self.config, "dwa_horizon_steps", 1)))
        current = state
        max_future_ratio = 0.0
        final_ratio = 0.0
        success_reached = False
        if track_success and not need_metrics:
            need_metrics = True
        final_metrics = (
            self._metrics(
                current,
                slot,
                vehicle_model,
                target_front_box=target_front_box,
            )
            if need_metrics
            else None
        )
        stop_ratio = None if stop_safe_ratio is None else float(stop_safe_ratio)
        for step_index in range(1, horizon + 1):
            current = vehicle_model.step(current, action)
            invalid, front_box, _ = self._state_invalid(current, scene, vehicle_model)
            if invalid:
                return {
                    "valid": False,
                    "state": current,
                    "max_future_ratio": float(max_future_ratio),
                    "final_ratio": float(final_ratio),
                    "metrics": final_metrics,
                    "success_reached": bool(success_reached),
                    "step": int(step_index),
                    "unlock_step": -1,
                }
            final_ratio = self._future_mask_ratio(
                current,
                scene,
                vehicle_model,
                lidar,
                action_mask,
            )
            max_future_ratio = max(max_future_ratio, final_ratio)
            if stop_ratio is not None and final_ratio >= stop_ratio:
                return {
                    "valid": True,
                    "state": current,
                    "max_future_ratio": float(max_future_ratio),
                    "final_ratio": float(final_ratio),
                    "metrics": final_metrics,
                    "success_reached": bool(success_reached),
                    "step": int(step_index),
                    "unlock_step": int(step_index),
                }
            if need_metrics:
                final_metrics = self._metrics(
                    current,
                    slot,
                    vehicle_model,
                    front_box=front_box,
                    target_front_box=target_front_box,
                )
                success_reached = bool(success_reached or final_metrics["success"])
        return {
            "valid": True,
            "state": current,
            "max_future_ratio": float(max_future_ratio),
            "final_ratio": float(final_ratio),
            "metrics": final_metrics,
            "success_reached": bool(success_reached),
            "step": int(horizon),
            "unlock_step": -1,
        }

    def run_unlock(
        self,
        state,
        slot,
        scene,
        vehicle_model,
        lidar,
        action_mask,
        current_mask,
        front_lidar,
        rear_lidar,
        prev_motion_gear,
        config,
        reason="all_zero",
    ):
        del front_lidar, rear_lidar, prev_motion_gear, config
        best = None
        best_score = None
        best_raw_action = None
        best_preview = None
        best_collision_free_score = None
        best_future_ratio = 0.0
        candidate_count = 0
        valid_count = 0
        threshold = float(getattr(self.config, "dwa_unlock_safe_ratio", 0.08))
        current_ratio = self._max_safe_ratio(current_mask)
        for phi_dot in self._phi_dot_candidates(action_mask):
            candidate_count += 1
            action = np.asarray([0.0, float(phi_dot)], dtype=np.float32)
            if abs(float(phi_dot)) <= 1e-12:
                rollout = {
                    "valid": True,
                    "state": state,
                    "max_future_ratio": float(current_ratio),
                    "final_ratio": float(current_ratio),
                    "metrics": None,
                    "success_reached": False,
                    "step": 0,
                    "unlock_step": 0 if current_ratio >= threshold else -1,
                }
            else:
                rollout = self._simulate(
                    state,
                    action,
                    slot,
                    scene,
                    vehicle_model,
                    lidar,
                    action_mask,
                    need_metrics=False,
                    track_success=False,
                    stop_safe_ratio=threshold,
                )
            best_future_ratio = max(best_future_ratio, rollout["max_future_ratio"])
            if not bool(rollout["valid"]):
                continue
            unlock_reached = bool(rollout["max_future_ratio"] >= threshold)
            score = (
                bool(unlock_reached),
                float(rollout["max_future_ratio"]),
                float(rollout["final_ratio"]),
                -abs(float(phi_dot)),
            )
            if best_collision_free_score is None or score > best_collision_free_score:
                best_collision_free_score = score
            if not unlock_reached:
                continue
            valid_count += 1
            raw_action = np.asarray(
                [
                    0.0,
                    action_mask.encode_phi_dot_to_raw(
                        phi_dot,
                        state.phi,
                        vehicle_model.params.dt,
                    ),
                ],
                dtype=np.float32,
            )
            if best_score is None or score > best_score:
                best = rollout
                best_score = score
                best_raw_action = raw_action
                best_preview = action.copy()

        if best is None:
            return DWAResult(
                used=False,
                mode="unlock",
                reason=str(reason),
                candidate_count=int(candidate_count),
                valid_candidate_count=0,
                final_max_safe_ratio=float(best_future_ratio),
                best_score=best_collision_free_score,
                unlock_success=False,
                unlock_step=-1,
                deadlock=True,
            )
        return DWAResult(
            used=True,
            mode="unlock",
            reason=str(reason),
            raw_action=best_raw_action,
            executed_action_preview=best_preview,
            candidate_count=int(candidate_count),
            valid_candidate_count=int(valid_count),
            final_max_safe_ratio=float(best["final_ratio"]),
            best_score=best_score,
            unlock_success=True,
            unlock_step=int(best.get("unlock_step", -1)),
            deadlock=False,
        )

    def run_local(
        self,
        state,
        slot,
        scene,
        vehicle_model,
        lidar,
        action_mask,
        current_mask,
        front_lidar,
        rear_lidar,
        prev_motion_gear,
        config,
        reason="low_safe",
    ):
        del front_lidar, rear_lidar, config
        mask = np.asarray(current_mask, dtype=np.float32)
        target_front_box = slot.front_box()
        initial_metrics = self._metrics(
            state,
            slot,
            vehicle_model,
            target_front_box=target_front_box,
        )
        best = None
        best_score = None
        best_raw_action = None
        best_preview = None
        candidate_count = 0
        valid_count = 0
        speed_ratios = tuple(float(item) for item in getattr(self.config, "dwa_speed_ratios", (1.0,)))
        speed_ratios = tuple(ratio for ratio in speed_ratios if ratio > 0.0)
        min_safe_ratio = float(getattr(action_mask, "min_safe_ratio", 1e-3))

        for gear in (FORWARD_GEAR, REVERSE_GEAR):
            sign = 1.0 if gear == FORWARD_GEAR else -1.0
            gear_vmax = (
                vehicle_model.params.parking_v_forward_max
                if gear == FORWARD_GEAR
                else vehicle_model.params.parking_v_reverse_max
            )
            for phi_dot in self._phi_dot_candidates(action_mask):
                r_safe = float(
                    np.interp(
                        float(phi_dot),
                        np.asarray(action_mask.phi_dot_bins, dtype=np.float64),
                        mask[gear],
                    )
                )
                for speed_ratio in speed_ratios:
                    candidate_count += 1
                    if r_safe <= min_safe_ratio:
                        continue
                    rho = float(np.clip(speed_ratio, 0.0, 1.0))
                    speed = sign * rho * r_safe * float(gear_vmax)
                    action = np.asarray([speed, float(phi_dot)], dtype=np.float32)
                    rollout = self._simulate(
                        state,
                        action,
                        slot,
                        scene,
                        vehicle_model,
                        lidar,
                        action_mask,
                        need_metrics=True,
                        track_success=True,
                        target_front_box=target_front_box,
                    )
                    if not bool(rollout["valid"]):
                        continue
                    valid_count += 1
                    metrics = rollout["metrics"]
                    distance_reduction = (
                        float(initial_metrics["distance_to_goal"])
                        - float(metrics["distance_to_goal"])
                    ) / max(float(initial_metrics["distance_to_goal"]), 1.0)
                    distance_reduction = float(np.clip(distance_reduction, -1.0, 1.0))
                    overlap_improvement = (
                        float(metrics["front_overlap"])
                        - float(initial_metrics["front_overlap"])
                    )
                    heading_improvement = (
                        float(metrics["heading_score"])
                        - float(initial_metrics["heading_score"])
                    )
                    task_progress = (
                        distance_reduction
                        + overlap_improvement
                        + heading_improvement
                    )
                    gear_switch_penalty = 1.0 if (
                        prev_motion_gear in (FORWARD_GEAR, REVERSE_GEAR)
                        and int(prev_motion_gear) != int(gear)
                    ) else 0.0
                    score = (
                        bool(rollout["success_reached"]),
                        float(rollout["max_future_ratio"]),
                        float(task_progress),
                        float(metrics["front_overlap"]),
                        -float(metrics["distance_to_goal"]),
                        -abs(float(phi_dot)),
                        -float(gear_switch_penalty),
                    )
                    raw_action = np.asarray(
                        [
                            sign * rho,
                            action_mask.encode_phi_dot_to_raw(
                                phi_dot,
                                state.phi,
                                vehicle_model.params.dt,
                            ),
                        ],
                        dtype=np.float32,
                    )
                    if best_score is None or score > best_score:
                        best = rollout
                        best_score = score
                        best_raw_action = raw_action
                        best_preview = action.copy()

        if best is None:
            return DWAResult(
                used=False,
                mode="local",
                reason=str(reason),
                candidate_count=int(candidate_count),
                valid_candidate_count=0,
                final_max_safe_ratio=0.0,
                best_score=best_score,
                unlock_success=False,
                deadlock=True,
            )
        return DWAResult(
            used=True,
            mode="local",
            reason=str(reason),
            raw_action=best_raw_action,
            executed_action_preview=best_preview,
            candidate_count=int(candidate_count),
            valid_candidate_count=int(valid_count),
            final_max_safe_ratio=float(best["final_ratio"]),
            best_score=best_score,
            unlock_success=False,
            deadlock=False,
        )

    def run(
        self,
        mode,
        state,
        slot,
        scene,
        vehicle_model,
        lidar,
        action_mask,
        current_mask,
        front_lidar,
        rear_lidar,
        prev_motion_gear,
        config,
        reason="",
    ):
        if str(mode) == "unlock":
            return self.run_unlock(
                state,
                slot,
                scene,
                vehicle_model,
                lidar,
                action_mask,
                current_mask,
                front_lidar,
                rear_lidar,
                prev_motion_gear,
                config,
                reason=reason,
            )
        if str(mode) == "local":
            return self.run_local(
                state,
                slot,
                scene,
                vehicle_model,
                lidar,
                action_mask,
                current_mask,
                front_lidar,
                rear_lidar,
                prev_motion_gear,
                config,
                reason=reason,
            )
        return self.empty(reason=reason)
