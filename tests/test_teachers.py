import math
import sys
import os

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from teachers.base import PlanResult, TeacherBase
from teachers.primitive_set import PrimitiveSet
from teachers.heuristics import TeacherHeuristic, count_gear_switches
from teachers.lattice import ArticulatedLatticeTeacher
from teachers.multi_anchor import MultiAnchorTeacher


class TestPlanResult:
    def test_plan_result_defaults(self):
        pr = PlanResult(success=False, teacher_name="test")
        assert pr.success is False
        assert pr.fail_reason == ""
        assert pr.states == []
        assert pr.actions_normalized == []
        assert pr.num_steps == 0

    def test_env_success_flag(self):
        from env.vehicle import ArticulatedState
        from env.geometry import DirectedParkingSlot

        slot = DirectedParkingSlot(0, 0, 0, 4.45, 3.016)
        state_good = ArticulatedState(0, 0, 0, 0)
        state_bad = ArticulatedState(10, 0, 0, 0)

        from env.geometry import oriented_box, overlap_ratio
        from env.vehicle import ArticulatedVehicleModel
        vm = ArticulatedVehicleModel()

        pr = PlanResult(
            success=True,
            states=[state_good],
            teacher_name="test",
            final_overlap=overlap_ratio(
                vm.body_boxes(state_good)[0],
                slot.front_box(),
            ),
            final_heading_error=0.0,
            final_phi=0.0,
        )
        assert pr.env_success is True

        pr2 = PlanResult(
            success=True,
            states=[state_bad],
            teacher_name="test",
            final_overlap=overlap_ratio(
                vm.body_boxes(state_bad)[0],
                slot.front_box(),
            ),
            final_heading_error=math.pi,
            final_phi=0.0,
        )
        assert pr2.env_success is False

    def test_clean_success_requires_low_phi(self):
        pr = PlanResult(
            success=True,
            teacher_name="test",
            final_overlap=0.85,
            final_heading_error=0.1,
            final_phi=math.radians(10),
        )
        assert pr.clean_success is True

        pr2 = PlanResult(
            success=True,
            teacher_name="test",
            final_overlap=0.85,
            final_heading_error=0.1,
            final_phi=math.radians(25),
        )
        assert pr2.clean_success is False


class TestPrimitiveSet:
    def test_construction(self):
        ps = PrimitiveSet()
        assert ps.n_coarse > 0
        assert ps.n_fine > 0
        assert ps.N_PHI_BINS == 15
        assert len(ps.phi_bin_centers) == 15

    def test_phi_bin_centers_include_zero(self):
        ps = PrimitiveSet()
        assert 0.0 in ps.phi_bin_centers
        assert ps.phi_bin_centers[0] < 0
        assert ps.phi_bin_centers[-1] > 0

    def test_nearest_phi_bin(self):
        ps = PrimitiveSet()
        assert ps._nearest_phi_bin(0.0) >= 0
        assert ps._nearest_phi_bin(ps.phi_max) == ps.N_PHI_BINS - 1
        assert ps._nearest_phi_bin(-ps.phi_max) == 0

    def test_dt_aligned_durations(self):
        ps = PrimitiveSet()
        for prim in ps.coarse_primitives + ps.fine_primitives:
            if prim.speed_ratio == 0.0 and prim.phi_dot_norm == 0.0:
                continue
            remainder = prim.duration % ps.dt
            assert abs(remainder) < 1e-6 or abs(remainder - ps.dt) < 1e-6, (
                f"Primitive duration {prim.duration} not aligned with dt={ps.dt}"
            )

    def test_select_primitive_set(self):
        ps = PrimitiveSet()
        assert ps.select_primitive_set(10.0, 10.0, 2.0, 0.1) == "coarse"
        assert ps.select_primitive_set(3.0, 10.0, 2.0, 0.1) == "fine"
        assert ps.select_primitive_set(10.0, 1.0, 2.0, 0.1) == "fine"
        assert ps.select_primitive_set(10.0, 10.0, 0.5, 0.1) == "fine"
        assert ps.select_primitive_set(10.0, 10.0, 2.0, 0.5) == "fine"

    def test_no_zero_speed_zero_phidot_primitive(self):
        ps = PrimitiveSet()
        for prim in ps.fine_primitives:
            assert not (prim.speed_ratio == 0.0 and prim.phi_dot_norm == 0.0)

    def test_table_lookup(self):
        ps = PrimitiveSet()
        entry = ps.table_entry(0.0, 0, is_fine=False)
        assert entry is not None
        assert "final_dx" in entry
        assert "sweep_radius" in entry

    def test_exact_rollout(self):
        from env.vehicle import ArticulatedVehicleModel
        from env.vehicle import ArticulatedState

        ps = PrimitiveSet()
        vm = ArticulatedVehicleModel()
        state = ArticulatedState(0, 0, 0, 0)

        class DummyScene:
            prepared_obstacles = None
            class _:
                @staticmethod
                def intersects(p):
                    return False
            prepared_obstacles = _()

        result = ps.exact_rollout(state, 0, False, vm, DummyScene())
        assert not result["collision"]
        assert "final_state" in result

    def test_quick_occupancy_check(self):
        ps = PrimitiveSet()
        entry = ps.table_entry(0.0, 0, is_fine=False)

        class FreeScene:
            @staticmethod
            def is_occupied_world(x, y):
                return False

        result = ps.quick_occupancy_check(entry, 0.0, 0.0, 0.0, FreeScene())
        assert not result


class TestTeacherHeuristic:
    def test_default_weights(self):
        h = TeacherHeuristic()
        assert h.weights["w_pos"] == 1.0
        assert h.weights["w_grid"] == 2.0

    def test_configure_rejects_deprecated_family(self):
        h = TeacherHeuristic()
        with pytest.raises(ValueError, match="unsupported task family"):
            h.configure_for_family("parallel_rev")

    def test_configure_head_in(self):
        h = TeacherHeuristic()
        h.configure_for_family("head_in")
        assert h.weights["w_anchor"] == 0.0

    def test_compute_returns_positive_value(self):
        h = TeacherHeuristic()

        class MockSlot:
            x_goal = 10.0
            y_goal = 0.0
            theta_goal = 0.0

        class MockScene:
            world_bounds = (-40, -40, 40, 40)
            resolution = 1.0
            target_bay = None

            @staticmethod
            def is_occupied_world(x, y):
                return False

        result = h.compute(0, 0, 0, 0, MockSlot(), MockScene())
        assert result > 0

    def test_grid_distance_caching(self):
        h = TeacherHeuristic()

        class MockSlot:
            x_goal = 0.0
            y_goal = 0.0
            theta_goal = 0.0

        class MockScene:
            world_bounds = (-40, -40, 40, 40)
            resolution = 1.0
            target_bay = None

            @staticmethod
            def is_occupied_world(x, y):
                return False

        result1 = h.compute(5, 0, 0, 0, MockSlot(), MockScene())
        result2 = h.compute(5, 0, 0, 0, MockSlot(), MockScene())
        assert result1 == result2


class TestHelpers:
    def test_count_gear_switches(self):
        assert count_gear_switches([]) == 0
        assert count_gear_switches([0]) == 0
        assert count_gear_switches([0, 0, 0]) == 0
        assert count_gear_switches([0, 1]) == 1
        assert count_gear_switches([0, 1, 0]) == 2
        assert count_gear_switches([0, 1, 1, 0]) == 2


class TestLatticeTeacher:
    def test_construction(self):
        teacher = ArticulatedLatticeTeacher()
        assert teacher.name == "lattice"
        assert teacher.max_expansions == 8000

    def test_label_first_action_sets_result(self):
        teacher = ArticulatedLatticeTeacher(max_expansions=10, max_time_ms=1000)

        from dataclasses import replace
        from config import DEFAULT_ENV_CONFIG
        from env.local_parking_env import LocalParkingEnv

        env_config = replace(DEFAULT_ENV_CONFIG, curriculum_stage=1, scene_pool_size=1,
                             scene_family_schedule=("head_in",), use_hybrid_astar=False,
                             rs_potential_enabled=False)
        env = LocalParkingEnv(config=env_config, seed=42)
        env.reset()

        label, result = teacher.label_first_action(
            env.state, env.scene, env.slot, env.vehicle_model,
        )
        assert isinstance(result, PlanResult)

    def test_back_compute_normalized_actions(self):
        from env.vehicle import ArticulatedState
        from config import DEFAULT_VEHICLE_PARAMS

        state = ArticulatedState(0, 0, 0, 0)
        actions = TeacherBase._back_compute_normalized_actions(
            [state], [0.5], [0.0], DEFAULT_VEHICLE_PARAMS,
        )
        assert len(actions) == 1
        assert -1.0 <= actions[0][0] <= 1.0
        assert -1.0 <= actions[0][1] <= 1.0


class TestMultiAnchorTeacher:
    def test_construction(self):
        teacher = MultiAnchorTeacher()
        assert teacher.name == "multi_anchor"
        assert teacher.top_anchors == 5

    def test_anchor_generation_returns_filtered_list(self):
        from dataclasses import replace
        from config import DEFAULT_ENV_CONFIG
        from env.local_parking_env import LocalParkingEnv

        env_config = replace(DEFAULT_ENV_CONFIG, curriculum_stage=1, scene_pool_size=1,
                             scene_family_schedule=("head_in",), use_hybrid_astar=False,
                             rs_potential_enabled=False)
        env = LocalParkingEnv(config=env_config, seed=42)
        env.reset()

        teacher = MultiAnchorTeacher()
        anchors = teacher._generate_anchors(env.slot, env.scene, env.vehicle_model)
        assert len(anchors) > 0
        for a in anchors:
            assert "x" in a
            assert "y" in a
            assert "heading" in a

    def test_label_first_action(self):
        from dataclasses import replace
        from config import DEFAULT_ENV_CONFIG
        from env.local_parking_env import LocalParkingEnv

        env_config = replace(DEFAULT_ENV_CONFIG, curriculum_stage=1, scene_pool_size=1,
                             scene_family_schedule=("head_in",), use_hybrid_astar=False,
                             rs_potential_enabled=False)
        env = LocalParkingEnv(config=env_config, seed=42)
        env.reset()

        teacher = MultiAnchorTeacher(max_total_time_ms=5000)
        label, result = teacher.label_first_action(
            env.state, env.scene, env.slot, env.vehicle_model,
        )
        assert isinstance(result, PlanResult)
