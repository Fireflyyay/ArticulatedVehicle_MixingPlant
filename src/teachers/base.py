import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from env.vehicle import ArticulatedState, ArticulatedVehicleModel
from env.geometry import DirectedParkingSlot, overlap_ratio, wrap_to_pi


@dataclass
class PlanResult:
    success: bool
    fail_reason: str = ""
    states: List[ArticulatedState] = field(default_factory=list)
    actions_normalized: List[np.ndarray] = field(default_factory=list)
    actions_physical: List[np.ndarray] = field(default_factory=list)
    masked_actions: Optional[List[np.ndarray]] = None
    total_cost: float = 0.0
    path_length: float = 0.0
    num_steps: int = 0
    num_gear_switches: int = 0
    num_zero_speed_steps: int = 0
    final_position_error: float = 0.0
    final_heading_error: float = 0.0
    final_phi: float = 0.0
    final_overlap: float = 0.0
    min_clearance: float = float("inf")
    collision: bool = False
    timeout: bool = False
    planning_time_ms: float = 0.0
    teacher_name: str = ""
    scenario_family: str = ""
    seed: int = 0
    selected_anchor_sequence: Optional[List[int]] = None
    refined: bool = False
    pre_refine_smoothness: Optional[float] = None
    mask_forced_stop_rate: Optional[float] = None
    mask_mean_r_raw: Optional[float] = None
    mask_low_r_raw_fraction: Optional[float] = None
    mask_speed_scaled_fraction: Optional[float] = None

    @property
    def env_success(self) -> bool:
        if self.final_overlap >= 0.80 and abs(self.final_heading_error) <= math.radians(15.0):
            return True
        if not self.states:
            return False
        final = self.states[-1]
        return False

    @property
    def clean_success(self) -> bool:
        clean_phi_threshold = math.radians(20.0)
        return self.env_success and abs(self.final_phi) <= clean_phi_threshold


class TeacherBase(ABC):
    def __init__(self, name="base"):
        self.name = name

    @abstractmethod
    def plan_from_state(
        self,
        state: ArticulatedState,
        scene,
        slot: DirectedParkingSlot,
        vehicle_model: ArticulatedVehicleModel,
    ) -> PlanResult:
        ...

    @abstractmethod
    def label_first_action(
        self,
        state: ArticulatedState,
        scene,
        slot: DirectedParkingSlot,
        vehicle_model: ArticulatedVehicleModel,
    ) -> Tuple[Optional[np.ndarray], Optional[PlanResult]]:
        ...

    @staticmethod
    def _back_compute_normalized_actions(
        states: List[ArticulatedState],
        v_cmds: List[float],
        phi_dot_cmds: List[float],
        vehicle_params,
    ) -> List[np.ndarray]:
        p = vehicle_params
        dt = p.dt
        phi_dot_max = p.phi_dot_max
        phi_max = p.phi_max
        parking_v_forward_max = p.parking_v_forward_max
        parking_v_reverse_max = p.parking_v_reverse_max
        actions = []
        for i, (v_cmd, phi_dot_cmd) in enumerate(zip(v_cmds, phi_dot_cmds)):
            phi = states[i].phi
            phi_dot_lower = max(-phi_dot_max, (-phi_max - phi) / dt)
            phi_dot_upper = min(phi_dot_max, (phi_max - phi) / dt)
            if phi_dot_upper <= phi_dot_lower:
                phi_dot_norm = 0.0
            else:
                alpha = (phi_dot_cmd - phi_dot_lower) / (phi_dot_upper - phi_dot_lower)
                phi_dot_norm = 2.0 * alpha - 1.0
            phi_dot_norm = float(np.clip(phi_dot_norm, -1.0, 1.0))
            if v_cmd >= 0:
                v_norm = v_cmd / parking_v_forward_max
            else:
                v_norm = v_cmd / parking_v_reverse_max
            v_norm = float(np.clip(v_norm, -1.0, 1.0))
            actions.append(np.array([v_norm, phi_dot_norm], dtype=np.float32))
        return actions

    @staticmethod
    def _scenario_family_from_scene(scene) -> str:
        task_family = str(scene.metadata.get("task_family", ""))
        if task_family in ("head_in", "parallel_fwd", "parallel_rev"):
            return task_family
        goal_mode = scene.metadata.get("goal_orientation_mode", "head_in")
        if goal_mode == "head_in":
            return "head_in"
        if bool(scene.metadata.get("parallel_reverse", False)):
            return "parallel_rev"
        return "parallel_fwd"

    @staticmethod
    def _compute_metrics(state, slot, vehicle_model):
        front_box, rear_box = vehicle_model.body_boxes(state)
        target_front = slot.front_box()
        front_overlap = overlap_ratio(front_box, target_front)
        heading_error = wrap_to_pi(state.theta_front - slot.theta_goal)
        distance = math.hypot(
            state.x_front - slot.x_goal,
            state.y_front - slot.y_goal,
        )
        return {
            "front_overlap": front_overlap,
            "heading_error": heading_error,
            "distance": distance,
            "phi": state.phi,
        }

    @staticmethod
    def _compute_clearance(state, scene, vehicle_model):
        front_box, rear_box = vehicle_model.body_boxes(state)
        try:
            front_clearance = front_box.distance(scene.obstacle_union)
            rear_clearance = rear_box.distance(scene.obstacle_union)
            return float(min(front_clearance, rear_clearance))
        except Exception:
            return 0.0

    @staticmethod
    def _check_collision(state, scene, vehicle_model):
        front_box, rear_box = vehicle_model.body_boxes(state)
        return bool(
            scene.prepared_obstacles.intersects(front_box)
            or scene.prepared_obstacles.intersects(rear_box)
        )

    @staticmethod
    def _check_success(state, slot, vehicle_model):
        metrics = TeacherBase._compute_metrics(state, slot, vehicle_model)
        return (
            metrics["front_overlap"] >= 0.80
            and abs(metrics["heading_error"]) <= math.radians(15.0)
        )
