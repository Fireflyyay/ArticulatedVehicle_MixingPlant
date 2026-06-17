import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from config import DEFAULT_VEHICLE_PARAMS
from env.vehicle import ArticulatedState, ArticulatedVehicleModel
from env.geometry import oriented_box, wrap_to_pi


@dataclass
class Primitive:
    gear: int
    speed_ratio: float
    phi_dot_norm: float
    duration: float
    sequence_idx: int
    v_cmd: float = 0.0
    phi_dot_cmd: float = 0.0


class PrimitiveSet:
    N_PHI_BINS = 15
    GRID_XY = 0.25
    N_THETA = 36

    def __init__(self, vehicle_params=None):
        self.p = vehicle_params or DEFAULT_VEHICLE_PARAMS
        self.dt = self.p.dt
        self.phi_max = self.p.phi_max
        self.phi_dot_max = self.p.phi_dot_max
        self._build_phi_bins()
        self._build_primitive_sets()
        self._build_precomputed_table()

    def _build_phi_bins(self):
        self.phi_bin_centers = np.linspace(
            -self.phi_max, self.phi_max, self.N_PHI_BINS, dtype=np.float64
        )

    def _nearest_phi_bin(self, phi: float) -> int:
        return int(np.argmin(np.abs(self.phi_bin_centers - float(phi))))

    def _build_primitive_sets(self):
        dt = self.dt
        durations_coarse = [2 * dt, 4 * dt, 6 * dt]
        durations_fine = [1 * dt, 2 * dt, 3 * dt]

        speed_coarse = [0.3, 0.5, 0.75, 1.0]
        speed_fine = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
        phi_dot_coarse = [-1.0, -0.5, 0.0, 0.5, 1.0]
        phi_dot_fine = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

        self.coarse_primitives: List[Primitive] = []
        self.fine_primitives: List[Primitive] = []

        idx = 0
        for gear in (0, 1):
            sign = 1.0 if gear == 0 else -1.0
            for rho in speed_coarse:
                for pd_norm in phi_dot_coarse:
                    for dur in durations_coarse:
                        if self._skip_coarse(rho, abs(pd_norm), dur):
                            continue
                        v_cmd = sign * rho * (
                            self.p.parking_v_forward_max if gear == 0
                            else self.p.parking_v_reverse_max
                        )
                        phi_dot_cmd = pd_norm * self.phi_dot_max
                        self.coarse_primitives.append(
                            Primitive(
                                gear=gear, speed_ratio=rho,
                                phi_dot_norm=pd_norm, duration=dur,
                                sequence_idx=idx, v_cmd=v_cmd,
                                phi_dot_cmd=phi_dot_cmd,
                            )
                        )
                        idx += 1

        fine_start = idx
        idx = 0
        for gear in (0, 1):
            sign = 1.0 if gear == 0 else -1.0
            for rho in speed_fine:
                for pd_norm in phi_dot_fine:
                    for dur in durations_fine:
                        if self._skip_fine(rho, abs(pd_norm), dur):
                            continue
                        if rho == 0.0 and pd_norm == 0.0:
                            continue
                        if rho == 0.0:
                            v_cmd = 0.0
                        else:
                            v_cmd = sign * rho * (
                                self.p.parking_v_forward_max if gear == 0
                                else self.p.parking_v_reverse_max
                            )
                        phi_dot_cmd = pd_norm * self.phi_dot_max
                        self.fine_primitives.append(
                            Primitive(
                                gear=gear, speed_ratio=rho,
                                phi_dot_norm=pd_norm, duration=dur,
                                sequence_idx=fine_start + idx,
                                v_cmd=v_cmd, phi_dot_cmd=phi_dot_cmd,
                            )
                        )
                        idx += 1

    def _skip_coarse(self, rho, abs_phi_dot_norm, dur):
        if rho >= 0.75 and abs_phi_dot_norm >= 1.0 and dur >= 4 * self.dt:
            return True
        if rho >= 1.0 and abs_phi_dot_norm >= 0.5 and dur >= 6 * self.dt:
            return True
        return False

    def _skip_fine(self, rho, abs_phi_dot_norm, dur):
        if rho >= 1.0 and abs_phi_dot_norm >= 1.0 and dur >= 3 * self.dt:
            return True
        if rho >= 1.0 and abs_phi_dot_norm >= 0.75 and dur >= 3 * self.dt:
            return True
        if rho >= 0.75 and abs_phi_dot_norm >= 0.5 and dur >= 3 * self.dt:
            return True
        return False

    def _build_precomputed_table(self):
        model = ArticulatedVehicleModel(self.p)
        n_coarse = len(self.coarse_primitives)
        n_fine = len(self.fine_primitives)

        self.coarse_table = [
            [None for _ in range(n_coarse)] for _ in range(self.N_PHI_BINS)
        ]
        self.fine_table = [
            [None for _ in range(n_fine)] for _ in range(self.N_PHI_BINS)
        ]

        for phi_idx, phi_center in enumerate(self.phi_bin_centers):
            for table, prim_list in [
                (self.coarse_table, self.coarse_primitives),
                (self.fine_table, self.fine_primitives),
            ]:
                for p_i, prim in enumerate(prim_list):
                    if table[phi_idx][p_i] is not None:
                        continue
                    entry = self._precompute_primitive(
                        prim, float(phi_center), model
                    )
                    table[phi_idx][p_i] = entry

    def _precompute_primitive(self, prim, initial_phi, model):
        dt = self.dt
        n_steps = max(1, int(round(prim.duration / dt)))
        state = ArticulatedState(
            x_front=0.0, y_front=0.0,
            theta_front=0.0, theta_rear=-initial_phi,
        )
        sweep_grid_points = []
        sweep_radius = 0.0
        all_front_corners = []
        all_rear_corners = []

        for step_idx in range(n_steps):
            state = model.step(state, (prim.v_cmd, prim.phi_dot_cmd), dt=dt)
            dist = math.hypot(state.x_front, state.y_front)
            if dist > sweep_radius:
                sweep_radius = dist
            sweep_grid_points.append((float(state.x_front), float(state.y_front)))
            front_box, rear_box = model.body_boxes(state)
            fc = np.asarray(front_box.exterior.coords[:-1], dtype=np.float64)
            rc = np.asarray(rear_box.exterior.coords[:-1], dtype=np.float64)
            all_front_corners.append(fc)
            all_rear_corners.append(rc)
            for pt in fc:
                d = math.hypot(pt[0], pt[1])
                if d > sweep_radius:
                    sweep_radius = d
            for pt in rc:
                d = math.hypot(pt[0], pt[1])
                if d > sweep_radius:
                    sweep_radius = d

        sweep_radius += 0.15

        all_fc = np.vstack(all_front_corners) if all_front_corners else np.zeros((0, 2))
        all_rc = np.vstack(all_rear_corners) if all_rear_corners else np.zeros((0, 2))
        front_corner_local = np.array([
            [-0.5*self.p.front_body_length, -0.5*self.p.front_body_width],
            [ 0.5*self.p.front_body_length, -0.5*self.p.front_body_width],
            [ 0.5*self.p.front_body_length,  0.5*self.p.front_body_width],
            [-0.5*self.p.front_body_length,  0.5*self.p.front_body_width],
        ])
        rear_corner_local = np.array([
            [-0.5*self.p.rear_body_length, -0.5*self.p.rear_body_width],
            [ 0.5*self.p.rear_body_length, -0.5*self.p.rear_body_width],
            [ 0.5*self.p.rear_body_length,  0.5*self.p.rear_body_width],
            [-0.5*self.p.rear_body_length,  0.5*self.p.rear_body_width],
        ])

        return {
            "final_dx": float(state.x_front),
            "final_dy": float(state.y_front),
            "final_dtheta_f": float(wrap_to_pi(state.theta_front)),
            "final_dtheta_r": float(wrap_to_pi(state.theta_rear)),
            "final_phi": float(state.phi),
            "final_gear": int(prim.gear),
            "n_dt_steps": n_steps,
            "sweep_radius": float(sweep_radius),
            "sweep_grid_points": sweep_grid_points,
            "prim_gear": int(prim.gear),
            "prim_speed_ratio": float(prim.speed_ratio),
            "prim_phi_dot_norm": float(prim.phi_dot_norm),
            "prim_duration": float(prim.duration),
            "prim_v_cmd": float(prim.v_cmd),
            "prim_phi_dot_cmd": float(prim.phi_dot_cmd),
        }

    def table_entry(self, phi: float, prim_idx: int, is_fine: bool):
        phi_bin = self._nearest_phi_bin(phi)
        table = self.fine_table if is_fine else self.coarse_table
        return table[phi_bin][prim_idx]

    def quick_occupancy_check(
        self, entry, theta_f, x_f, y_f, scene,
    ) -> bool:
        sweep_radius = entry["sweep_radius"]
        c = math.cos(theta_f)
        s = math.sin(theta_f)
        for (lx, ly) in entry["sweep_grid_points"]:
            wx = x_f + c * lx - s * ly
            wy = y_f + s * lx + c * ly
            if scene.is_occupied_world(wx, wy):
                return True
        return False

    def exact_rollout(
        self,
        state: ArticulatedState,
        prim_idx: int,
        is_fine: bool,
        vehicle_model: ArticulatedVehicleModel,
        scene,
    ):
        primitives = self.fine_primitives if is_fine else self.coarse_primitives
        prim = primitives[prim_idx]
        dt = self.dt
        n_steps = max(1, int(round(prim.duration / dt)))
        v_cmd = prim.v_cmd
        phi_dot_cmd = prim.phi_dot_cmd

        current = state
        intermediate_states = []

        for _ in range(n_steps):
            current = vehicle_model.step(current, (v_cmd, phi_dot_cmd), dt=dt)
            intermediate_states.append(current)
            front_box, rear_box = vehicle_model.body_boxes(current)
            if (
                scene.prepared_obstacles.intersects(front_box)
                or scene.prepared_obstacles.intersects(rear_box)
            ):
                return {
                    "final_state": current,
                    "intermediate_states": intermediate_states,
                    "collision": True,
                    "n_steps": n_steps,
                    "primitive": prim,
                }

        return {
            "final_state": current,
            "intermediate_states": intermediate_states,
            "collision": False,
            "n_steps": n_steps,
            "primitive": prim,
        }

    def select_primitive_set(self, d_goal, d_entry, clearance, phi_abs):
        if d_goal < 6.0 or d_entry < 3.0 or clearance < 1.0 or phi_abs > 0.4:
            return "fine"
        return "coarse"

    def get_primitives(self, set_name: str) -> List[Primitive]:
        if set_name == "fine":
            return self.fine_primitives
        return self.coarse_primitives

    def get_table(self, set_name: str):
        if set_name == "fine":
            return self.fine_table
        return self.coarse_table

    @property
    def n_coarse(self):
        return len(self.coarse_primitives)

    @property
    def n_fine(self):
        return len(self.fine_primitives)
