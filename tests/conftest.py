import os
import sys

import numpy as np
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import DEFAULT_VEHICLE_PARAMS  # noqa: E402
from env.articulated_action_mask import ArticulatedActionMask  # noqa: E402


@pytest.fixture
def synthetic_action_mask():
    p = DEFAULT_VEHICLE_PARAMS
    phi_bins = np.linspace(-p.phi_max, p.phi_max, 3, dtype=np.float32)
    phi_dot_bins = np.linspace(-p.phi_dot_max, p.phi_dot_max, 11, dtype=np.float32)
    speed_forward = np.linspace(p.parking_v_forward_max / 2.0, p.parking_v_forward_max, 2)
    speed_reverse = np.linspace(p.parking_v_reverse_max / 2.0, p.parking_v_reverse_max, 2)
    beam_angles = np.linspace(0.0, 2.0 * np.pi, p.lidar_beams, endpoint=False)
    shape = (3, 2, 11, 2, p.lidar_beams)
    return ArticulatedActionMask(
        sweep_table_front=np.ones(shape, dtype=np.float32),
        sweep_table_rear=np.ones(shape, dtype=np.float32),
        phi_state_bins=phi_bins,
        phi_dot_bins=phi_dot_bins,
        speed_bins_forward=speed_forward,
        speed_bins_reverse=speed_reverse,
        beam_angles=beam_angles,
        safety_margin=0.0,
    )
