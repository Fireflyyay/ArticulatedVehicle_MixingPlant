from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np

from config import DEFAULT_VEHICLE_PARAMS, ZL50GNVehicleParams
from env.geometry import oriented_box, wrap_to_pi


@dataclass
class ArticulatedState:
    x_front: float
    y_front: float
    theta_front: float
    theta_rear: float
    v: float = 0.0
    phi_dot: float = 0.0

    @property
    def phi(self) -> float:
        return float(wrap_to_pi(self.theta_front - self.theta_rear))

    def as_array(self):
        return np.asarray(
            [
                self.x_front,
                self.y_front,
                self.theta_front,
                self.theta_rear,
                self.v,
                self.phi_dot,
            ],
            dtype=np.float32,
        )


class ArticulatedVehicleModel:
    """Low-speed articulated kinematics with commanded articulation rate."""

    def __init__(self, params=DEFAULT_VEHICLE_PARAMS):
        self.params = params

    def rear_center(self, state):
        p = self.params
        hinge_x = state.x_front - p.front_center_to_hinge * math.cos(state.theta_front)
        hinge_y = state.y_front - p.front_center_to_hinge * math.sin(state.theta_front)
        return (
            hinge_x - p.rear_center_to_hinge * math.cos(state.theta_rear),
            hinge_y - p.rear_center_to_hinge * math.sin(state.theta_rear),
        )

    def body_boxes(self, state):
        rear_center = self.rear_center(state)
        front = oriented_box(
            (state.x_front, state.y_front),
            state.theta_front,
            self.params.front_body_length,
            self.params.front_body_width,
        )
        rear = oriented_box(
            rear_center,
            state.theta_rear,
            self.params.rear_body_length,
            self.params.rear_body_width,
        )
        return front, rear

    def target_rear_box(self, x_front, y_front, theta_front):
        state = ArticulatedState(
            x_front=float(x_front),
            y_front=float(y_front),
            theta_front=float(theta_front),
            theta_rear=float(theta_front),
        )
        _, rear = self.body_boxes(state)
        return rear

    def step(self, state, action, dt=None):
        p = self.params
        duration = p.dt if dt is None else float(dt)
        v_cmd = float(action[0])
        phi_dot_cmd = float(action[1])
        v = float(np.clip(v_cmd, -p.parking_v_reverse_max, p.parking_v_forward_max))
        phi_dot = float(np.clip(phi_dot_cmd, -p.phi_dot_max, p.phi_dot_max))
        substeps = max(1, int(p.integration_substeps))
        h = duration / float(substeps)

        x = float(state.x_front)
        y = float(state.y_front)
        theta_f = float(state.theta_front)
        theta_r = float(state.theta_rear)
        lf = p.front_center_to_hinge
        lr = p.rear_center_to_hinge

        for _ in range(substeps):
            phi = float(wrap_to_pi(theta_f - theta_r))
            denom = lf * math.cos(phi) + lr
            if abs(denom) < 1e-6:
                denom = math.copysign(1e-6, denom)
            theta_f_dot = (v * math.sin(phi) + lr * phi_dot) / denom
            theta_r_dot = theta_f_dot - phi_dot
            x += v * math.cos(theta_f) * h
            y += v * math.sin(theta_f) * h
            theta_f = float(wrap_to_pi(theta_f + theta_f_dot * h))
            theta_r = float(wrap_to_pi(theta_r + theta_r_dot * h))

        return ArticulatedState(
            x_front=x,
            y_front=y,
            theta_front=theta_f,
            theta_rear=theta_r,
            v=v,
            phi_dot=phi_dot,
        )


def clip_phi_dot_to_limit(phi, phi_dot, dt, phi_max):
    phi = float(phi)
    phi_dot = float(phi_dot)
    dt = max(float(dt), 1e-8)
    lower = (-float(phi_max) - phi) / dt
    upper = (float(phi_max) - phi) / dt
    return float(np.clip(phi_dot, lower, upper))
