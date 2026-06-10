import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from config import (
    DEFAULT_MASK_CONFIG,
    DEFAULT_VEHICLE_PARAMS,
    ActionMaskConfig,
    ZL50GNVehicleParams,
)
from env.vehicle import (
    ArticulatedState,
    ArticulatedVehicleModel,
    clip_phi_dot_to_limit,
)


FORWARD_GEAR = 0
REVERSE_GEAR = 1

STOP_GEAR = -1


def default_action_mask_path():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo_root, "assets", "action_mask", "zl50gn_articulated_mask.npz")


@dataclass
class ActionExecution:
    raw_action: np.ndarray
    decoded_action: np.ndarray
    executed_action: np.ndarray
    safe_ratio: float
    invalid: bool
    speed_clipped: bool


class ArticulatedActionMask:
    REQUIRED_KEYS = {
        "sweep_table_front",
        "sweep_table_rear",
        "phi_state_bins",
        "phi_dot_bins",
        "speed_bins_forward",
        "speed_bins_reverse",
        "beam_angles",
        "vehicle_params",
        "safety_margin",
        "metadata",
    }

    def __init__(
        self,
        sweep_table_front,
        sweep_table_rear,
        phi_state_bins,
        phi_dot_bins,
        speed_bins_forward,
        speed_bins_reverse,
        beam_angles,
        safety_margin,
        vehicle_params=DEFAULT_VEHICLE_PARAMS,
        min_safe_ratio=DEFAULT_MASK_CONFIG.min_safe_ratio,
        metadata=None,
    ):
        self.sweep_table_front = np.asarray(sweep_table_front, dtype=np.float32)
        self.sweep_table_rear = np.asarray(sweep_table_rear, dtype=np.float32)
        self.phi_state_bins = np.asarray(phi_state_bins, dtype=np.float32)
        self.phi_dot_bins = np.asarray(phi_dot_bins, dtype=np.float32)
        self.speed_bins_forward = np.asarray(speed_bins_forward, dtype=np.float32)
        self.speed_bins_reverse = np.asarray(speed_bins_reverse, dtype=np.float32)
        self.beam_angles = np.asarray(beam_angles, dtype=np.float32)
        self.safety_margin = float(safety_margin)
        self.vehicle_params = vehicle_params
        self.min_safe_ratio = float(min_safe_ratio)
        self.metadata = dict(metadata or {})
        self._validate_shapes()

    @property
    def feature_dim(self):
        return 2 * int(self.phi_dot_bins.size)

    def _validate_shapes(self):
        expected = (
            self.phi_state_bins.size,
            2,
            self.phi_dot_bins.size,
            self.speed_bins_forward.size,
            self.beam_angles.size,
        )
        if self.sweep_table_front.shape != expected:
            raise ValueError(
                "front sweep table shape {} does not match {}".format(
                    self.sweep_table_front.shape,
                    expected,
                )
            )
        if self.sweep_table_rear.shape != expected:
            raise ValueError(
                "rear sweep table shape {} does not match {}".format(
                    self.sweep_table_rear.shape,
                    expected,
                )
            )
        if self.speed_bins_forward.shape != self.speed_bins_reverse.shape:
            raise ValueError("forward and reverse speed-bin arrays must have equal shape")

    @classmethod
    def load(cls, path=None, vehicle_params=DEFAULT_VEHICLE_PARAMS):
        resolved = default_action_mask_path() if path is None else os.path.abspath(path)
        with np.load(resolved, allow_pickle=False) as data:
            missing = cls.REQUIRED_KEYS.difference(data.files)
            if missing:
                raise ValueError("action-mask table is missing keys: {}".format(sorted(missing)))
            stored_params = json.loads(str(data["vehicle_params"].item()))
            for key in (
                "front_body_length",
                "rear_body_length",
                "front_body_width",
                "rear_body_width",
                "phi_max",
                "phi_dot_max",
                "parking_v_forward_max",
                "parking_v_reverse_max",
                "lidar_beams",
            ):
                if not np.isclose(float(stored_params[key]), float(getattr(vehicle_params, key))):
                    raise ValueError("action-mask vehicle parameter mismatch for '{}'".format(key))
            metadata = json.loads(str(data["metadata"].item()))
            return cls(
                sweep_table_front=data["sweep_table_front"],
                sweep_table_rear=data["sweep_table_rear"],
                phi_state_bins=data["phi_state_bins"],
                phi_dot_bins=data["phi_dot_bins"],
                speed_bins_forward=data["speed_bins_forward"],
                speed_bins_reverse=data["speed_bins_reverse"],
                beam_angles=data["beam_angles"],
                safety_margin=float(data["safety_margin"].item()),
                vehicle_params=vehicle_params,
                metadata=metadata,
            )

    def _nearest_phi_index(self, phi):
        return int(np.argmin(np.abs(self.phi_state_bins - float(phi))))

    def compute_mask(self, phi, front_lidar_m, rear_lidar_m):
        """Online mask computation: table lookup plus vectorized comparisons."""
        front_lidar = np.asarray(front_lidar_m, dtype=np.float32).reshape(1, 1, -1)
        rear_lidar = np.asarray(rear_lidar_m, dtype=np.float32).reshape(1, 1, -1)
        if front_lidar.shape[-1] != self.beam_angles.size:
            raise ValueError("front LiDAR beam count does not match action-mask table")
        if rear_lidar.shape[-1] != self.beam_angles.size:
            raise ValueError("rear LiDAR beam count does not match action-mask table")

        phi_index = self._nearest_phi_index(phi)
        required_front = self.sweep_table_front[phi_index]
        required_rear = self.sweep_table_rear[phi_index]
        front_safe = front_lidar > required_front + self.safety_margin
        rear_safe = rear_lidar > required_rear + self.safety_margin
        speed_safe = np.all(front_safe & rear_safe, axis=-1)

        mask = np.zeros((2, self.phi_dot_bins.size), dtype=np.float32)
        for gear in (FORWARD_GEAR, REVERSE_GEAR):
            speed_bins = (
                self.speed_bins_forward
                if gear == FORWARD_GEAR
                else self.speed_bins_reverse
            )
            vmax = (
                self.vehicle_params.parking_v_forward_max
                if gear == FORWARD_GEAR
                else self.vehicle_params.parking_v_reverse_max
            )
            safe_speeds = np.where(speed_safe[gear], speed_bins[None, :], 0.0)
            mask[gear] = np.max(safe_speeds, axis=1) / float(vmax)
        return np.clip(mask, 0.0, 1.0)

    def filter_and_clip_action_fallback(self, v_exec, v_safe_max, phi_dot, phi, dt=None):
        p = self.vehicle_params
        duration = p.dt if dt is None else float(dt)
        phi_dot_executed = float(np.clip(phi_dot, -p.phi_dot_max, p.phi_dot_max))
        phi_dot_executed = clip_phi_dot_to_limit(
            phi,
            phi_dot_executed,
            duration,
            p.phi_max,
        )
        was_clipped = False
        if abs(v_exec) > v_safe_max + 1e-5:
            v_exec = math.copysign(v_safe_max, v_exec)
            was_clipped = True
        return v_exec, phi_dot_executed, was_clipped

    def _r_raw_at_gear_phi_dot(self, mask, gear, phi_dot_executed):
        return float(
            np.interp(
                phi_dot_executed,
                self.phi_dot_bins,
                np.asarray(mask, dtype=np.float32)[gear],
            )
        )

    def _r_max_overall(self, mask):
        return float(np.max(np.asarray(mask, dtype=np.float32)))

    def decode_safe_speed_and_cost(self, raw_action, mask, phi, dt, prev_motion_gear, config):
        raw = np.clip(np.asarray(raw_action, dtype=np.float32), -1.0, 1.0)
        if raw.shape != (2,):
            raise ValueError("raw_action must have shape (2,)")
        p = self.vehicle_params
        duration = p.dt if dt is None else float(dt)
        deadband = float(getattr(config, "gear_deadband", 0.10))

        phi_dot_raw = float(raw[1]) * p.phi_dot_max
        phi_dot_exec = clip_phi_dot_to_limit(phi, phi_dot_raw, duration, p.phi_max)

        if abs(raw[0]) < deadband:
            if prev_motion_gear is None:
                gear = STOP_GEAR
                rho = 0.0
            else:
                gear = prev_motion_gear
                rho = 0.0
            a0_sign = 0.0
        else:
            if raw[0] >= 0.0:
                gear = FORWARD_GEAR
            else:
                gear = REVERSE_GEAR
            rho = abs(float(raw[0]))
            a0_sign = math.copysign(1.0, raw[0])

        if gear == STOP_GEAR:
            v_exec = 0.0
            r_raw = 0.0
            forced_stop = False
            v_safe_max = 0.0
            clip_ratio = 0.0
        else:
            gear_vmax = (
                p.parking_v_forward_max if gear == FORWARD_GEAR
                else p.parking_v_reverse_max
            )
            r_raw = self._r_raw_at_gear_phi_dot(mask, gear, phi_dot_exec)
            r_min = float(self.min_safe_ratio)
            forced_stop = (r_raw <= r_min)
            if forced_stop:
                r_exec = 0.0
                v_safe_max = 0.0
                v_exec = 0.0
            else:
                r_exec = r_raw
                v_safe_max = r_exec * gear_vmax
                v_decoded = a0_sign * rho * v_safe_max
                v_exec = float(v_decoded)
            clip_ratio_check = (
                abs(rho * gear_vmax) + 1e-7 if not forced_stop else 0.0
            )

        v_exec, phi_dot_exec, was_clipped = self.filter_and_clip_action_fallback(
            v_exec, v_safe_max, phi_dot_exec, phi, dt
        )

        if gear != STOP_GEAR and not forced_stop:
            if v_safe_max > 1e-7:
                clip_ratio = abs(rho * v_safe_max - abs(v_exec)) / (abs(rho * v_safe_max) + 1e-6)
            else:
                clip_ratio = 0.0
        else:
            clip_ratio = 0.0
        if was_clipped:
            clip_ratio = max(clip_ratio, 0.01)

        r_max = self._r_max_overall(mask) if gear != STOP_GEAR else 0.0
        r_min = float(self.min_safe_ratio)
        tau_safe = float(getattr(config, "mask_cost_safe_threshold", 0.15))
        delta_rel = float(getattr(config, "mask_cost_rel_delta", 0.05))
        c_max = float(getattr(config, "mask_cost_max", 3.0))

        c_stop = float(getattr(config, "mask_cost_stop_weight", 0.5)) * (1.0 if forced_stop else 0.0)
        c_abs = float(getattr(config, "mask_cost_abs_weight", 0.15)) * max(0.0, tau_safe - r_raw)
        c_rel = (
            float(getattr(config, "mask_cost_rel_weight", 0.10))
            * max(0.0, r_max - r_raw - delta_rel)
        )
        c_clip = float(getattr(config, "mask_cost_clip_weight", 0.05)) * clip_ratio
        mask_cost = float(np.clip(c_stop + c_abs + c_rel + c_clip, 0.0, c_max))

        if abs(raw[0]) >= deadband:
            if raw[0] >= 0.0:
                new_motion_gear = FORWARD_GEAR
            else:
                new_motion_gear = REVERSE_GEAR
        else:
            new_motion_gear = prev_motion_gear

        if gear == STOP_GEAR:
            prev_gear_in_obs = 0.0
        elif gear == FORWARD_GEAR:
            prev_gear_in_obs = 1.0
        else:
            prev_gear_in_obs = -1.0

        return {
            "v_exec": float(v_exec),
            "phi_dot_exec": float(phi_dot_exec),
            "gear": int(gear),
            "rho": float(rho),
            "r_raw": float(r_raw),
            "r_exec": float(r_raw if not forced_stop else 0.0),
            "r_max": float(r_max),
            "forced_stop": bool(forced_stop),
            "clip_ratio": float(clip_ratio),
            "mask_cost": float(mask_cost),
            "prev_motion_gear": new_motion_gear,
            "prev_gear_in_obs": float(prev_gear_in_obs),
        }

    def filter_and_clip_action(self, raw_action, mask, phi, dt=None):
        raw = np.clip(np.asarray(raw_action, dtype=np.float32), -1.0, 1.0)
        if raw.shape != (2,):
            raise ValueError("raw_action must have shape (2,)")
        p = self.vehicle_params
        duration = p.dt if dt is None else float(dt)
        if raw[0] >= 0.0:
            gear = FORWARD_GEAR
            v_decoded = float(raw[0]) * p.parking_v_forward_max
            gear_vmax = p.parking_v_forward_max
        else:
            gear = REVERSE_GEAR
            v_decoded = float(raw[0]) * p.parking_v_reverse_max
            gear_vmax = p.parking_v_reverse_max
        phi_dot_decoded = float(raw[1]) * p.phi_dot_max
        safe_ratio = float(
            np.interp(
                phi_dot_decoded,
                self.phi_dot_bins,
                np.asarray(mask, dtype=np.float32)[gear],
            )
        )
        invalid = safe_ratio <= self.min_safe_ratio
        if invalid:
            v_executed = 0.0
        else:
            safe_speed = safe_ratio * gear_vmax
            v_executed = math.copysign(min(abs(v_decoded), safe_speed), v_decoded)

        phi_dot_executed = float(
            np.clip(phi_dot_decoded, -p.phi_dot_max, p.phi_dot_max)
        )
        phi_dot_executed = clip_phi_dot_to_limit(
            phi,
            phi_dot_executed,
            duration,
            p.phi_max,
        )
        speed_clipped = abs(v_executed) + 1e-7 < abs(v_decoded)
        return ActionExecution(
            raw_action=raw.copy(),
            decoded_action=np.asarray([v_decoded, phi_dot_decoded], dtype=np.float32),
            executed_action=np.asarray([v_executed, phi_dot_executed], dtype=np.float32),
            safe_ratio=safe_ratio,
            invalid=bool(invalid),
            speed_clipped=bool(speed_clipped),
        )


def _box_edges(polygon):
    coords = np.asarray(polygon.exterior.coords, dtype=np.float64)
    return np.stack([coords[:-1], coords[1:]], axis=1)


def _radial_extent_for_edges(edges, sensor_center, sensor_heading, beam_angles):
    translated = np.asarray(edges, dtype=np.float64) - np.asarray(sensor_center)[None, None, :]
    c = math.cos(float(sensor_heading))
    s = math.sin(float(sensor_heading))
    rotation = np.asarray([[c, s], [-s, c]], dtype=np.float64)
    local = translated.reshape(-1, 2).dot(rotation.T).reshape(translated.shape)
    p = local[:, 0]
    segment = local[:, 1] - local[:, 0]
    rays = np.stack([np.cos(beam_angles), np.sin(beam_angles)], axis=1)
    ray = rays[:, None, :]
    seg = segment[None, :, :]
    point = p[None, :, :]
    denom = ray[..., 0] * seg[..., 1] - ray[..., 1] * seg[..., 0]
    valid_denom = np.abs(denom) > 1e-10
    safe_denom = np.where(valid_denom, denom, 1.0)
    cross_point_seg = point[..., 0] * seg[..., 1] - point[..., 1] * seg[..., 0]
    cross_point_ray = point[..., 0] * ray[..., 1] - point[..., 1] * ray[..., 0]
    t = cross_point_seg / safe_denom
    u = cross_point_ray / safe_denom
    valid = valid_denom & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
    return np.max(np.where(valid, t, 0.0), axis=1)


def generate_sweep_tables(
    vehicle_params=DEFAULT_VEHICLE_PARAMS,
    mask_config=DEFAULT_MASK_CONFIG,
    trace_samples=8,
):
    p = vehicle_params
    cfg = mask_config
    model = ArticulatedVehicleModel(p)
    phi_state_bins = np.linspace(
        -p.phi_max,
        p.phi_max,
        cfg.n_phi_state_bins,
        dtype=np.float32,
    )
    phi_dot_bins = np.linspace(
        -p.phi_dot_max,
        p.phi_dot_max,
        cfg.n_phi_dot_bins,
        dtype=np.float32,
    )
    speed_bins_forward = np.linspace(
        p.parking_v_forward_max / cfg.n_speed_bins,
        p.parking_v_forward_max,
        cfg.n_speed_bins,
        dtype=np.float32,
    )
    speed_bins_reverse = np.linspace(
        p.parking_v_reverse_max / cfg.n_speed_bins,
        p.parking_v_reverse_max,
        cfg.n_speed_bins,
        dtype=np.float32,
    )
    beam_angles = np.linspace(
        0.0,
        2.0 * math.pi,
        p.lidar_beams,
        endpoint=False,
        dtype=np.float32,
    )
    shape = (
        cfg.n_phi_state_bins,
        2,
        cfg.n_phi_dot_bins,
        cfg.n_speed_bins,
        p.lidar_beams,
    )
    sweep_front = np.zeros(shape, dtype=np.float32)
    sweep_rear = np.zeros(shape, dtype=np.float32)
    sample_dt = p.dt * cfg.table_horizon_steps / float(max(1, trace_samples))

    for phi_index, phi in enumerate(phi_state_bins):
        initial = ArticulatedState(
            x_front=0.0,
            y_front=0.0,
            theta_front=0.0,
            theta_rear=-float(phi),
        )
        initial_rear_center = model.rear_center(initial)
        for gear in (FORWARD_GEAR, REVERSE_GEAR):
            speed_bins = speed_bins_forward if gear == FORWARD_GEAR else speed_bins_reverse
            sign = 1.0 if gear == FORWARD_GEAR else -1.0
            for phi_dot_index, phi_dot in enumerate(phi_dot_bins):
                for speed_index, speed in enumerate(speed_bins):
                    state = initial
                    front_extent = np.zeros(p.lidar_beams, dtype=np.float64)
                    rear_extent = np.zeros(p.lidar_beams, dtype=np.float64)
                    for _ in range(max(1, trace_samples) + 1):
                        front_box, rear_box = model.body_boxes(state)
                        front_extent = np.maximum(
                            front_extent,
                            _radial_extent_for_edges(
                                _box_edges(front_box),
                                (0.0, 0.0),
                                0.0,
                                beam_angles,
                            ),
                        )
                        rear_extent = np.maximum(
                            rear_extent,
                            _radial_extent_for_edges(
                                _box_edges(rear_box),
                                initial_rear_center,
                                -float(phi),
                                beam_angles,
                            ),
                        )
                        state = model.step(
                            state,
                            (sign * float(speed), float(phi_dot)),
                            dt=sample_dt,
                        )
                    sweep_front[phi_index, gear, phi_dot_index, speed_index] = front_extent
                    sweep_rear[phi_index, gear, phi_dot_index, speed_index] = rear_extent

    metadata = {
        "format_version": 1,
        "model": "zl50gn_articulated_dual_body_sweep",
        "online_algorithm": "lidar_vs_precomputed_sweep_matrix_compare",
        "trace_samples": int(trace_samples),
        "table_horizon_seconds": float(p.dt * cfg.table_horizon_steps),
        "mask_semantics": "max_safe_speed_ratio",
    }
    return {
        "sweep_table_front": sweep_front,
        "sweep_table_rear": sweep_rear,
        "phi_state_bins": phi_state_bins,
        "phi_dot_bins": phi_dot_bins,
        "speed_bins_forward": speed_bins_forward,
        "speed_bins_reverse": speed_bins_reverse,
        "beam_angles": beam_angles,
        "vehicle_params": np.asarray(json.dumps(p.to_dict(), sort_keys=True)),
        "safety_margin": np.asarray(cfg.safety_margin, dtype=np.float32),
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }


def save_sweep_tables(path, tables):
    resolved = os.path.abspath(path)
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    np.savez_compressed(resolved, **tables)
    return resolved
