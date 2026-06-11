import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from env.geometry import wrap_to_pi
from planning.reeds_shepp import generate_reeds_shepp_paths


@dataclass
class RSPlanResult:
    valid: bool
    path: Optional[np.ndarray]
    reason: str
    total_length: float = 0.0
    candidate_count: int = 0
    checked_candidates: int = 0
    collision_checks: int = 0
    sample_count: int = 0
    generation_time_ms: float = 0.0
    collision_time_ms: float = 0.0
    total_time_ms: float = 0.0


class RSPotentialPlanner:
    """Generate short RS candidates and validate them with Hybrid A* footprint checks."""

    def __init__(
        self,
        collision_checker,
        turning_radius=6.4,
        candidate_limit=2,
        sample_step=0.3,
        endpoint_heading_tolerance=1e-3,
    ):
        if not hasattr(collision_checker, "_is_rectangle_occupied"):
            raise TypeError(
                "collision_checker must provide PassengerHybridAStar footprint checks"
            )
        self.collision_checker = collision_checker
        self.turning_radius = float(turning_radius)
        self.candidate_limit = max(1, int(candidate_limit))
        self.sample_step = float(sample_step)
        self.endpoint_heading_tolerance = float(endpoint_heading_tolerance)

    def _collision_free(self, scene, path):
        checks = 0
        for x, y, theta in path:
            checks += 1
            if self.collision_checker._is_rectangle_occupied(
                scene, float(x), float(y), float(theta)
            ):
                return False, checks
        return True, checks

    def plan(self, scene, state, slot):
        total_start = time.perf_counter()
        generation_start = time.perf_counter()
        candidates = generate_reeds_shepp_paths(
            (
                float(state.x_front),
                float(state.y_front),
                float(state.theta_front),
            ),
            (
                float(slot.x_goal),
                float(slot.y_goal),
                float(slot.theta_goal),
            ),
            turning_radius=self.turning_radius,
            sample_step=self.sample_step,
        )
        generation_ms = (time.perf_counter() - generation_start) * 1000.0
        if not candidates:
            return RSPlanResult(
                valid=False,
                path=None,
                reason="no_rs_path",
                generation_time_ms=generation_ms,
                total_time_ms=(time.perf_counter() - total_start) * 1000.0,
            )

        checked = 0
        collision_checks = 0
        collision_ms = 0.0
        heading_valid_candidates = 0
        for candidate in candidates[: self.candidate_limit]:
            endpoint_error = abs(
                wrap_to_pi(float(candidate.poses[-1, 2]) - float(slot.theta_goal))
            )
            if endpoint_error > self.endpoint_heading_tolerance:
                continue
            heading_valid_candidates += 1
            checked += 1
            collision_start = time.perf_counter()
            collision_free, checks = self._collision_free(scene, candidate.poses)
            collision_ms += (time.perf_counter() - collision_start) * 1000.0
            collision_checks += checks
            if collision_free:
                return RSPlanResult(
                    valid=True,
                    path=candidate.poses.astype(np.float64, copy=True),
                    reason="success",
                    total_length=float(candidate.total_length),
                    candidate_count=len(candidates),
                    checked_candidates=checked,
                    collision_checks=collision_checks,
                    sample_count=int(candidate.poses.shape[0]),
                    generation_time_ms=generation_ms,
                    collision_time_ms=collision_ms,
                    total_time_ms=(time.perf_counter() - total_start) * 1000.0,
                )

        if heading_valid_candidates == 0:
            reason = "wrong_goal_heading"
        else:
            reason = "collision"
        return RSPlanResult(
            valid=False,
            path=None,
            reason=reason,
            candidate_count=len(candidates),
            checked_candidates=checked,
            collision_checks=collision_checks,
            generation_time_ms=generation_ms,
            collision_time_ms=collision_ms,
            total_time_ms=(time.perf_counter() - total_start) * 1000.0,
        )


class RSPotentialOracle:
    """Near-goal RS trigger, episode latch, and transition-level potential reward."""

    def __init__(
        self,
        planner=None,
        enabled=True,
        d_rs=10.0,
        gamma=0.98,
        cost_scale=25.0,
        potential_coef=1.0,
        potential_clip=1.0,
        max_cost=80.0,
        lateral_weight=0.5,
        heading_weight=1.0,
        lateral_clip=5.0,
    ):
        self.planner = planner
        self.enabled = bool(enabled)
        self.d_rs = float(d_rs)
        self.gamma = float(gamma)
        self.cost_scale = float(cost_scale)
        self.potential_coef = float(potential_coef)
        self.potential_clip = float(potential_clip)
        self.max_cost = float(max_cost)
        self.lateral_weight = float(lateral_weight)
        self.heading_weight = float(heading_weight)
        self.lateral_clip = float(lateral_clip)
        self.reset()

    def reset(self):
        self.rs_latched = False
        self.rs_path = None
        self.rs_planner_disabled_after_latch = False
        self.rs_last_cost = None
        self.rs_prev_potential = None
        self.rs_attempt_count = 0
        self.rs_success_count = 0
        self.rs_fail_reason = "disabled" if not self.enabled else "outside_d_rs"
        self._plan_times_ms = []
        self._cumulative_s = None
        self._total_length = 0.0
        self._last_remaining = 0.0
        self._last_projection_error = 0.0
        self._last_heading_error = 0.0
        self._last_plan_result = None

    def _project(self, x, y):
        point = np.asarray([float(x), float(y)], dtype=np.float64)
        starts = self.rs_path[:-1, :2]
        vectors = self.rs_path[1:, :2] - starts
        length_sq = np.sum(vectors * vectors, axis=1)
        safe_length_sq = np.maximum(length_sq, 1e-12)
        fractions = (
            np.sum((point[None, :] - starts) * vectors, axis=1) / safe_length_sq
        )
        fractions = np.clip(fractions, 0.0, 1.0)
        projections = starts + fractions[:, None] * vectors
        distances = np.linalg.norm(projections - point[None, :], axis=1)
        index = int(np.argmin(distances))
        segment_length = math.sqrt(float(safe_length_sq[index]))
        progress = self._cumulative_s[index] + fractions[index] * segment_length
        theta0 = float(self.rs_path[index, 2])
        theta1 = float(self.rs_path[index + 1, 2])
        path_heading = wrap_to_pi(
            theta0 + fractions[index] * wrap_to_pi(theta1 - theta0)
        )
        return float(progress), float(distances[index]), float(path_heading)

    def J_state(self, state):
        if not self.rs_latched or self.rs_path is None:
            raise RuntimeError("RS cost requested before a path was latched")
        progress, projection_error, path_heading = self._project(
            state.x_front, state.y_front
        )
        remaining = max(0.0, self._total_length - progress)
        heading_error = abs(wrap_to_pi(float(state.theta_front) - path_heading))
        cost = (
            remaining
            + self.lateral_weight * min(projection_error, self.lateral_clip)
            + self.heading_weight * heading_error
        )
        return (
            min(float(cost), self.max_cost),
            float(remaining),
            float(projection_error),
            float(heading_error),
        )

    def _latch(self, path):
        array = np.asarray(path, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 3:
            raise ValueError("RS planner returned an invalid path")
        self.rs_path = array[:, :3].copy()
        segment_lengths = np.linalg.norm(np.diff(self.rs_path[:, :2], axis=0), axis=1)
        self._cumulative_s = np.concatenate(
            [np.asarray([0.0], dtype=np.float64), np.cumsum(segment_lengths)]
        )
        self._total_length = float(self._cumulative_s[-1])
        self.rs_latched = True
        self.rs_planner_disabled_after_latch = True
        self.rs_success_count += 1
        self.rs_fail_reason = ""

    def _diagnostics(self, reward):
        attempt_count = max(self.rs_attempt_count, 1)
        mean_ms = (
            float(np.mean(self._plan_times_ms)) if self._plan_times_ms else 0.0
        )
        max_ms = max(self._plan_times_ms) if self._plan_times_ms else 0.0
        phi = (
            -float(self.rs_last_cost) / self.cost_scale
            if self.rs_last_cost is not None
            else 0.0
        )
        result = self._last_plan_result
        return {
            "rs_attempt_count": int(self.rs_attempt_count),
            "rs_success_count": int(self.rs_success_count),
            "rs_latched": bool(self.rs_latched),
            "rs_planner_disabled_after_latch": bool(
                self.rs_planner_disabled_after_latch
            ),
            "rs_valid_rate": float(self.rs_success_count) / float(attempt_count)
            if self.rs_attempt_count
            else 0.0,
            "rs_plan_time_ms_mean": mean_ms,
            "rs_plan_time_ms_max": float(max_ms),
            "rs_reward": float(reward),
            "rs_cost": float(self.rs_last_cost or 0.0),
            "rs_remaining_length": float(self._last_remaining),
            "rs_projection_error": float(self._last_projection_error),
            "rs_heading_error": float(self._last_heading_error),
            "rs_fail_reason": str(self.rs_fail_reason),
            "rs_candidate_count": int(getattr(result, "candidate_count", 0)),
            "rs_checked_candidates": int(
                getattr(result, "checked_candidates", 0)
            ),
            "rs_collision_checks": int(getattr(result, "collision_checks", 0)),
            "rs_sample_count": int(getattr(result, "sample_count", 0)),
            "rs_generation_time_ms": float(
                getattr(result, "generation_time_ms", 0.0)
            ),
            "rs_collision_time_ms": float(
                getattr(result, "collision_time_ms", 0.0)
            ),
            "planner_valid": bool(self.rs_latched),
            "planner_cost": float(self.rs_last_cost or 0.0),
            "planner_phi": float(phi),
            "planner_potential_reward": float(reward),
            "planner_fallback_used": False,
            "planner_fail_reason": str(self.rs_fail_reason),
        }

    def diagnostics(self):
        return self._diagnostics(0.0)

    def step(self, scene, previous_state, current_state, slot):
        if not self.enabled or self.planner is None:
            self.rs_fail_reason = "disabled"
            return 0.0, self._diagnostics(0.0)

        if not self.rs_latched:
            distance = math.hypot(
                float(current_state.x_front) - float(slot.x_goal),
                float(current_state.y_front) - float(slot.y_goal),
            )
            if distance >= self.d_rs:
                self.rs_fail_reason = "outside_d_rs"
                return 0.0, self._diagnostics(0.0)

            self.rs_attempt_count += 1
            started = time.perf_counter()
            try:
                result = self.planner.plan(scene, current_state, slot)
            except Exception:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self._plan_times_ms.append(elapsed_ms)
                self.rs_fail_reason = "planner_exception"
                self._last_plan_result = None
                return 0.0, self._diagnostics(0.0)
            elapsed_ms = float(
                getattr(
                    result,
                    "total_time_ms",
                    (time.perf_counter() - started) * 1000.0,
                )
            )
            self._plan_times_ms.append(elapsed_ms)
            self._last_plan_result = result
            if not result.valid:
                self.rs_fail_reason = str(result.reason)
                return 0.0, self._diagnostics(0.0)
            try:
                self._latch(result.path)
            except (TypeError, ValueError):
                self.rs_fail_reason = "no_rs_path"
                self.rs_latched = False
                self.rs_path = None
                return 0.0, self._diagnostics(0.0)
            previous_cost, _, _, _ = self.J_state(previous_state)
            self.rs_last_cost = previous_cost
            self.rs_prev_potential = -previous_cost / self.cost_scale

        current_cost, remaining, projection_error, heading_error = self.J_state(
            current_state
        )
        previous_phi = (
            self.rs_prev_potential
            if self.rs_prev_potential is not None
            else -current_cost / self.cost_scale
        )
        current_phi = -current_cost / self.cost_scale
        delta = self.gamma * current_phi - previous_phi
        clipped_delta = max(-self.potential_clip, min(self.potential_clip, delta))
        reward = self.potential_coef * clipped_delta

        self.rs_last_cost = current_cost
        self.rs_prev_potential = current_phi
        self._last_remaining = remaining
        self._last_projection_error = projection_error
        self._last_heading_error = heading_error
        return float(reward), self._diagnostics(reward)
