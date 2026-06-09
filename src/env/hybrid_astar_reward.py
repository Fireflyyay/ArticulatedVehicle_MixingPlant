import math

import numpy as np

from env.geometry import wrap_to_pi


class OptionalHybridAStarReward:
    """Planner cost-to-go potential reward (PBRS) replacing XY-path progress."""

    def __init__(
        self,
        planner=None,
        gamma=0.98,
        cost_scale=25.0,
        potential_coef=0.5,
        potential_clip=1.0,
        max_cost=80.0,
        lateral_residual_weight=0.5,
        goal_heading_weight=1.0,
        lateral_clip=5.0,
        fallback_position_weight=1.0,
        fallback_heading_weight=2.0,
        failure_bias=3.0,
        # Legacy XY-path progress reward — default OFF (weights = 0.0)
        legacy_progress_weight=0.0,
        legacy_lateral_weight=0.0,
        sigma_s=0.5,
        lateral_normalizer=5.0,
    ):
        self.planner = planner
        self._gamma = float(gamma)
        self._cost_scale = float(cost_scale)
        self._potential_coef = float(potential_coef)
        self._potential_clip = float(potential_clip)
        self._max_cost = float(max_cost)
        self._lateral_residual_weight = float(lateral_residual_weight)
        self._goal_heading_weight = float(goal_heading_weight)
        self._lateral_clip = float(lateral_clip)
        self._fallback_position_weight = float(fallback_position_weight)
        self._fallback_heading_weight = float(fallback_heading_weight)
        self._failure_bias = float(failure_bias)
        # Legacy internals
        self._legacy_progress_weight = float(legacy_progress_weight)
        self._legacy_lateral_weight = float(legacy_lateral_weight)
        self._sigma_s = float(sigma_s)
        self._lateral_normalizer = float(lateral_normalizer)

        # State carried across reset / step calls
        self.valid = False
        self.fail_reason = "disabled"
        self._previous_J = 0.0
        self._path = None
        self._total_path_length = 0.0
        self._cumulative_s = None
        self._goal_theta = 0.0
        self._fallback_used = False
        self._expanded_nodes = 0

        # Legacy state
        self._legacy_previous_progress = 0.0

    # ------------------------------------------------------------------
    #  Stage target (defaults to final slot; overridable for multi-stage)
    # ------------------------------------------------------------------
    class _SimpleTarget:
        __slots__ = ("x", "y", "theta")
        def __init__(self, x, y, theta):
            self.x = float(x)
            self.y = float(y)
            self.theta = float(theta)

    def _build_stage_target(self, slot):
        return self._SimpleTarget(slot.x_goal, slot.y_goal, slot.theta_goal)

    # ------------------------------------------------------------------
    #  J_fallback  —  pose-based rough cost when planner fails
    # ------------------------------------------------------------------
    def _J_fallback(self, x, y, theta):
        pos_err = math.hypot(
            float(x) - self._stage_target.x,
            float(y) - self._stage_target.y,
        )
        head_err = abs(wrap_to_pi(
            float(theta) - self._stage_target.theta
        ))
        J = (
            self._fallback_position_weight * pos_err
            + self._fallback_heading_weight * head_err
            + self._failure_bias
        )
        return min(J, self._max_cost)

    # ------------------------------------------------------------------
    #  Projection onto stored path  (reused from legacy, now internal)
    # ------------------------------------------------------------------
    def _project(self, point):
        point = np.asarray(point, dtype=np.float64)
        starts = self._path[:-1]
        vectors = self._path[1:] - starts
        length_sq = np.sum(vectors * vectors, axis=1)
        safe_len = np.maximum(length_sq, 1e-12)
        frac = np.sum((point[None, :] - starts) * vectors, axis=1) / safe_len
        frac = np.clip(frac, 0.0, 1.0)
        proj = starts + frac[:, None] * vectors
        dists = np.linalg.norm(proj - point[None, :], axis=1)
        idx = int(np.argmin(dists))
        s = self._cumulative_s[idx] + frac[idx] * math.sqrt(safe_len[idx])
        return float(s), float(dists[idx])

    # ------------------------------------------------------------------
    #  J_state  —  additive cost from (x, y, theta) to goal
    # ------------------------------------------------------------------
    def J_state(self, x, y, theta):
        if self.valid and self._path is not None and self._cumulative_s is not None:
            progress, lateral = self._project(np.array([x, y]))
            remaining = max(0.0, self._total_path_length - float(progress))
            lateral_clipped = min(float(lateral), self._lateral_clip)
            heading_err = abs(wrap_to_pi(
                float(theta) - self._goal_theta
            ))
            J = (
                remaining
                + self._lateral_residual_weight * lateral_clipped
                + self._goal_heading_weight * heading_err
            )
        else:
            J = self._J_fallback(x, y, theta)
        return min(J, self._max_cost)

    # ------------------------------------------------------------------
    #  reset  —  called once per episode
    # ------------------------------------------------------------------
    def reset(self, scene, state, slot):
        self.valid = False
        self.fail_reason = "disabled" if self.planner is None else "planner_failed"
        self._fallback_used = False
        self._expanded_nodes = 0
        self._path = None
        self._cumulative_s = None
        self._total_path_length = 0.0
        self._goal_theta = float(slot.theta_goal)
        self._stage_target = self._build_stage_target(slot)

        if self.planner is None:
            self._previous_J = self._J_fallback(
                state.x_front, state.y_front, state.theta_front
            )
            return False

        try:
            result = self.planner.plan_with_cost(scene, state, slot)
        except Exception as exc:
            self.fail_reason = "planner_exception:{}".format(type(exc).__name__)
            self._fallback_used = True
            self._previous_J = self._J_fallback(
                state.x_front, state.y_front, state.theta_front
            )
            return False

        self._expanded_nodes = getattr(result, "expanded_nodes", 0)

        if not result.valid:
            self.fail_reason = "planner_{}".format(result.reason)
            self._fallback_used = True
            self._previous_J = self._J_fallback(
                state.x_front, state.y_front, state.theta_front
            )
            return False

        path = np.asarray(result.path, dtype=np.float64)
        if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] < 2:
            self.fail_reason = "invalid_path"
            self._fallback_used = True
            self._previous_J = self._J_fallback(
                state.x_front, state.y_front, state.theta_front
            )
            return False

        self._path = path[:, :2]
        segment_lengths = np.linalg.norm(np.diff(self._path, axis=0), axis=1)
        self._cumulative_s = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        self._total_path_length = float(self._cumulative_s[-1])
        self.valid = True
        self.fail_reason = ""

        self._previous_J = self.J_state(
            state.x_front, state.y_front, state.theta_front
        )

        # Reset legacy state
        self._legacy_previous_progress, _ = self._project(
            np.array([state.x_front, state.y_front])
        )

        return True

    # ------------------------------------------------------------------
    #  step  —  PBRS potential reward (tau=1 for now)
    # ------------------------------------------------------------------
    def step(self, x, y, theta):
        J_curr = self.J_state(x, y, theta)

        Phi_prev = -self._previous_J / self._cost_scale
        Phi_curr = -J_curr / self._cost_scale
        # TODO: upgrade gamma^1 → gamma^tau when SMDP macro-action support lands
        reward = self._potential_coef * (self._gamma * Phi_curr - Phi_prev)
        reward = max(-self._potential_clip, min(self._potential_clip, reward))

        self._previous_J = J_curr

        heading_err = abs(wrap_to_pi(float(theta) - self._goal_theta))
        info = {
            "planner_valid": self.valid,
            "planner_cost": float(J_curr),
            "planner_phi": float(Phi_curr),
            "planner_potential_reward": float(reward),
            "planner_fallback_used": self._fallback_used,
            "planner_fail_reason": self.fail_reason,
            "planner_expanded_nodes": self._expanded_nodes,
            "goal_heading_error_single_direction": float(heading_err),
        }
        return float(reward), info

    # ------------------------------------------------------------------
    #  step_legacy  —  original XY-path progress / lateral reward (debug)
    # ------------------------------------------------------------------
    def step_legacy(self, x, y):
        if not self.valid:
            return 0.0, {
                "hybrid_astar_valid": False,
                "hybrid_astar_progress": 0.0,
                "hybrid_astar_lateral_error": 0.0,
                "hybrid_astar_fail_reason": self.fail_reason,
            }
        progress, lateral_error = self._project(
            np.asarray([float(x), float(y)], dtype=np.float64)
        )
        delta_s = progress - self._legacy_previous_progress
        self._legacy_previous_progress = progress
        reward = self._legacy_progress_weight * math.tanh(
            delta_s / max(self._sigma_s, 1e-6)
        )
        reward -= self._legacy_lateral_weight * min(
            lateral_error / max(self._lateral_normalizer, 1e-6),
            1.0,
        )
        return float(reward), {
            "hybrid_astar_valid": True,
            "hybrid_astar_progress": progress,
            "hybrid_astar_lateral_error": lateral_error,
            "hybrid_astar_fail_reason": "",
        }
