import math

import numpy as np


class OptionalHybridAStarReward:
    """Auxiliary path reward that is inert when planning fails or is disabled."""

    def __init__(
        self,
        planner=None,
        progress_weight=1.0,
        lateral_weight=0.25,
        sigma_s=0.5,
        lateral_normalizer=5.0,
    ):
        self.planner = planner
        self.progress_weight = float(progress_weight)
        self.lateral_weight = float(lateral_weight)
        self.sigma_s = float(sigma_s)
        self.lateral_normalizer = float(lateral_normalizer)
        self.path = None
        self.cumulative_s = None
        self.previous_progress = 0.0
        self.valid = False
        self.fail_reason = "disabled"

    def reset(self, scene, state, slot):
        self.path = None
        self.cumulative_s = None
        self.previous_progress = 0.0
        self.valid = False
        self.fail_reason = "disabled" if self.planner is None else "planner_failed"
        if self.planner is None:
            return False
        try:
            path = self.planner.plan(scene, state, slot)
        except Exception as exc:
            self.fail_reason = "planner_exception:{}".format(type(exc).__name__)
            return False
        if path is None:
            return False
        path = np.asarray(path, dtype=np.float64)
        if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] < 2:
            self.fail_reason = "invalid_path"
            return False
        self.path = path[:, :2]
        segment_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.cumulative_s = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        self.previous_progress, _ = self._project(
            np.asarray([state.x_front, state.y_front], dtype=np.float64)
        )
        self.valid = True
        self.fail_reason = ""
        return True

    def _project(self, point):
        starts = self.path[:-1]
        vectors = self.path[1:] - starts
        length_sq = np.sum(vectors * vectors, axis=1)
        safe_length_sq = np.maximum(length_sq, 1e-12)
        fractions = np.sum((point[None, :] - starts) * vectors, axis=1) / safe_length_sq
        fractions = np.clip(fractions, 0.0, 1.0)
        projections = starts + fractions[:, None] * vectors
        distances = np.linalg.norm(projections - point[None, :], axis=1)
        index = int(np.argmin(distances))
        progress = self.cumulative_s[index] + fractions[index] * math.sqrt(length_sq[index])
        return float(progress), float(distances[index])

    def step(self, x, y):
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
        delta_s = progress - self.previous_progress
        self.previous_progress = progress
        reward = self.progress_weight * math.tanh(delta_s / max(self.sigma_s, 1e-6))
        reward -= self.lateral_weight * min(
            lateral_error / max(self.lateral_normalizer, 1e-6),
            1.0,
        )
        return float(reward), {
            "hybrid_astar_valid": True,
            "hybrid_astar_progress": progress,
            "hybrid_astar_lateral_error": lateral_error,
            "hybrid_astar_fail_reason": "",
        }
