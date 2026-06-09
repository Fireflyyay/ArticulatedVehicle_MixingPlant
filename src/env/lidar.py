import math

import numpy as np

from config import DEFAULT_VEHICLE_PARAMS


class DualBodyLidar:
    """Vectorized front/rear body-center LiDAR over cached obstacle edges."""

    def __init__(self, params=DEFAULT_VEHICLE_PARAMS):
        self.params = params
        self.beam_angles = np.linspace(
            0.0,
            2.0 * math.pi,
            int(params.lidar_beams),
            endpoint=False,
            dtype=np.float64,
        )
        self.ray_directions = np.stack(
            [np.cos(self.beam_angles), np.sin(self.beam_angles)],
            axis=1,
        )

    @staticmethod
    def _cross(lhs, rhs):
        return lhs[..., 0] * rhs[..., 1] - lhs[..., 1] * rhs[..., 0]

    def scan(self, center, heading, obstacle_edges, normalize=True):
        edges = np.asarray(obstacle_edges, dtype=np.float64)
        if edges.size == 0:
            result = np.full(self.params.lidar_beams, self.params.lidar_range, dtype=np.float32)
            return result / self.params.lidar_range if normalize else result

        p = edges[:, 0] - np.asarray(center, dtype=np.float64)
        segment = edges[:, 1] - edges[:, 0]
        c = math.cos(float(heading))
        s = math.sin(float(heading))
        rotation = np.asarray([[c, s], [-s, c]], dtype=np.float64)
        p_local = p.dot(rotation.T)
        segment_local = segment.dot(rotation.T)

        rays = self.ray_directions[:, None, :]
        segs = segment_local[None, :, :]
        points = p_local[None, :, :]
        denom = self._cross(rays, segs)
        valid_denom = np.abs(denom) > 1e-10
        safe_denom = np.where(valid_denom, denom, 1.0)
        t = self._cross(points, segs) / safe_denom
        u = self._cross(points, rays) / safe_denom
        valid = valid_denom & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
        distances = np.where(valid, t, np.inf)
        nearest = np.min(distances, axis=1)
        nearest = np.clip(nearest, 0.0, self.params.lidar_range).astype(np.float32)
        return nearest / self.params.lidar_range if normalize else nearest

    def observe(self, state, vehicle_model, scene, normalize=True):
        rear_center = vehicle_model.rear_center(state)
        front = self.scan(
            (state.x_front, state.y_front),
            state.theta_front,
            scene.obstacle_edges,
            normalize=normalize,
        )
        rear = self.scan(
            rear_center,
            state.theta_rear,
            scene.obstacle_edges,
            normalize=normalize,
        )
        return front, rear
