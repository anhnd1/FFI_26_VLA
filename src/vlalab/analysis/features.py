"""Per-episode trajectory features from the recorded policy calls.

Everything is computed from what the policy saw and said, at call resolution (every replan_steps env steps):
  state   (n, 8)    eef position (3), axis-angle (3), gripper finger positions (2)
  actions (n, H, 7) predicted chunk: 6 delta eef pose (normalized controller units) + gripper (-1 open, +1 close)
"""
from __future__ import annotations

import numpy as np


def executed_actions(ep: dict) -> np.ndarray:
    """(env_steps, 7): the actions actually sent to the env (first replan_steps of each chunk)."""
    r = ep["meta"]["replan_steps"]
    acts = np.concatenate([chunk[:r] for chunk in ep["actions"]])
    return acts[: ep["meta"]["env_steps"]]


MIN_HOLD = 10  # env steps (0.5 s at 20 Hz): a gripper state held shorter than this is flicker, not a grasp/release


def gripper_switches(ep: dict) -> np.ndarray:
    """Env steps at which the executed gripper command changes (open <-> close), flicker included."""
    closed = np.concatenate([[False], executed_actions(ep)[:, 6] > 0])  # the gripper starts open (settle steps)
    return np.flatnonzero(np.diff(closed.astype(int)))


def gripper_events(ep: dict, min_hold: int = MIN_HOLD) -> tuple[np.ndarray, np.ndarray]:
    """Env steps where a sustained grasp (closed for >= min_hold steps) and a sustained release start.

    Runs shorter than min_hold are flicker: they are merged into the surrounding state, so a burst of open/close
    toggles over a target counts once (or not at all), not as many grasp attempts.
    """
    closed = executed_actions(ep)[:, 6] > 0
    starts = np.flatnonzero(np.diff(np.concatenate([[-1], closed.astype(int)])))  # run starts
    lengths = np.diff(np.concatenate([starts, [len(closed)]]))
    state, grasps, releases = False, [], []  # the gripper starts open
    for start, n in zip(starts, lengths):
        if n >= min_hold and closed[start] != state:
            (grasps if closed[start] else releases).append(start)
            state = bool(closed[start])
    return np.array(grasps, int), np.array(releases, int)


def replan_disagreement(ep: dict) -> np.ndarray:
    """(n-1,) per re-plan: mean L2 distance (translation dims) between what the previous chunk still planned and
    what the new chunk plans for the same env steps. High = the policy changed its mind. NaN if chunks don't overlap.
    """
    r = ep["meta"]["replan_steps"]
    a = ep["actions"]
    h = a.shape[1]
    if len(a) < 2 or r >= h:
        return np.full(max(len(a) - 1, 0), np.nan)
    prev_tail, new_head = a[:-1, r:, :3], a[1:, : h - r, :3]
    return np.linalg.norm(prev_tail - new_head, axis=-1).mean(axis=-1)


def stalled(ep: dict, window_calls: int = 4, tol_m: float = 0.01) -> np.ndarray:
    """(n,) bool per call: eef moved less than tol_m over the last window_calls calls (20 env steps at replan 5)."""
    pos = ep["state"][:, :3]
    out = np.zeros(len(pos), bool)
    if len(pos) > window_calls:
        out[window_calls:] = np.linalg.norm(pos[window_calls:] - pos[:-window_calls], axis=-1) < tol_m
    return out


def episode_features(ep: dict) -> dict:
    m = ep["meta"]
    pos = ep["state"][:, :3]
    grasps, releases = gripper_events(ep)
    dis = replan_disagreement(ep)
    stall = stalled(ep)
    return {
        "task_id": m["task_id"],
        "episode_idx": m["episode_idx"],
        "success": m["success"],
        "env_steps": m["env_steps"],
        "n_calls": len(ep["call_index"]),
        "grasp_attempts": len(grasps),  # sustained closes (>= MIN_HOLD steps)
        "releases": len(releases),
        "gripper_switches": len(gripper_switches(ep)),  # raw command changes, flicker included
        "path_m": float(np.linalg.norm(np.diff(pos, axis=0), axis=-1).sum()),
        "stall_frac": float(stall.mean()),
        "disagree_mean": float(np.nanmean(dis)) if np.isfinite(dis).any() else np.nan,
        "disagree_p90": float(np.nanpercentile(dis, 90)) if np.isfinite(dis).any() else np.nan,
    }
