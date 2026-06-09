import math

import numpy as np


class LocalParkingReward:
    def __init__(self, config):
        if float(config.distance_d_min) <= 0.0:
            raise ValueError("distance_d_min must be positive")
        self.config = config
        self.initial_distance = 1.0
        self.best_front_overlap = 0.0
        self.best_heading_score = 0.0

    @staticmethod
    def heading_score(heading_error):
        return 0.5 * (1.0 + math.cos(float(heading_error)))

    def reset(self, initial_distance, initial_overlap, initial_heading_error):
        self.initial_distance = max(float(initial_distance), 0.0)
        self.best_front_overlap = float(initial_overlap)
        self.best_heading_score = self.heading_score(initial_heading_error)

    def compute(
        self,
        front_overlap,
        distance_to_goal,
        heading_error,
        step_count,
        success=False,
        failure=False,
        hybrid_reward=0.0,
    ):
        overlap = float(front_overlap)
        iou_improvement = max(overlap - self.best_front_overlap, 0.0)
        self.best_front_overlap = max(self.best_front_overlap, overlap)

        heading_score = self.heading_score(heading_error)
        heading_improvement = max(heading_score - self.best_heading_score, 0.0)
        self.best_heading_score = max(self.best_heading_score, heading_score)

        # HOPE Appendix B: this is relative to the episode's initial
        # distance, not the previous timestep's distance.
        distance_reward = -(
            float(distance_to_goal) - self.initial_distance
        ) / max(self.initial_distance, float(self.config.distance_d_min))
        time_reward = -math.tanh(
            float(step_count) / (10.0 * max(1, int(self.config.max_steps)))
        )
        terminal_reward = 0.0
        if success:
            terminal_reward = float(self.config.success_reward)
        elif failure:
            terminal_reward = float(self.config.failure_reward)

        components = {
            "terminal": terminal_reward,
            "iou_improvement": iou_improvement,
            "distance": distance_reward,
            "heading_improvement": heading_improvement,
            "time": time_reward,
            "hybrid_astar": float(hybrid_reward),
        }
        total = (
            terminal_reward
            + self.config.w_iou * iou_improvement
            + self.config.w_dist * distance_reward
            + self.config.w_heading * heading_improvement
            + self.config.w_time * time_reward
            + self.config.w_hybrid * float(hybrid_reward)
        )
        components["total"] = float(total)
        return float(total), components
