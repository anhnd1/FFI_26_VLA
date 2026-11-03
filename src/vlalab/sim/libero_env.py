"""LIBERO environment and observation preprocessing (Python 3.8, robosuite 1.4.1, mujoco 3.2.3).

Same conventions as openpi's LIBERO training data and examples/libero/main.py.
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

DUMMY_ACTION = [0.0] * 6 + [-1.0]  # no motion, gripper open: used while objects settle after reset
# Episode step budget per suite: a bit above the longest training demo of the suite.
MAX_STEPS = {
    "libero_spatial": 220,  # longest demo 193 steps
    "libero_object": 280,  # 254
    "libero_goal": 300,  # 270
    "libero_10": 520,  # 505
    "libero_90": 400,  # 373
}


def task_suite(name: str):
    return benchmark.get_benchmark_dict()[name]()


def make_env(task, resolution: int, seed: int):
    """Off-screen env for one task. Returns (env, task language)."""
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)  # affects object poses even with a fixed initial state
    return env, task.language


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """robosuite transform_utils.quat2axisangle (xyzw quaternion -> axis * angle)."""
    quat = quat.copy()
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def policy_image(raw: np.ndarray, size: int) -> np.ndarray:
    """Rotate 180 deg (LIBERO renders upside down; the training data was flipped) and resize with padding."""
    img = np.ascontiguousarray(raw[::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, size, size))


def policy_observation(obs: dict, prompt: str, size: int) -> dict:
    """LIBERO obs -> the dict sent to the policy server (see policy/pipeline.py)."""
    return {
        "observation/image": policy_image(obs["agentview_image"], size),
        "observation/wrist_image": policy_image(obs["robot0_eye_in_hand_image"], size),
        "observation/state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
        "prompt": str(prompt),
    }
