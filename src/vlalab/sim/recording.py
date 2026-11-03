"""Run output format, written by sim/evaluate.py and read by policy/replay.py and analysis/.

<run>/<arm>/
  config.json        resolved ArmConfig
  episodes.csv       one row per episode: suite, task_id, task, episode_idx, success, env_steps, n_calls
  summary.json       success rate, rollout wall time, ...
  latency.csv        client/server latency per benchmark call (sim/bench.py)
  episodes/<task_id>_ep<episode_idx>.npz   one per episode, one entry per policy call:
      call_index (n,)              server call counter (the noise-stream position in noise="stream")
      t          (n,)              policy env step at which the call was made (0 = first step after the settle wait)
      state      (n, 8)            observation/state sent to the policy
      actions    (n, H, 7)         full chunk returned (the first replan_steps were executed)
      image, wrist_image (n, 224, 224, 3) uint8   policy inputs (if record_images)
      meta       JSON string       suite, task_id, task, prompt, episode_idx, success, env_steps, replan_steps
  videos/<task_id>_ep<episode_idx>_<success|failure>.mp4   agent view, 10 fps, one frame per env step

Python 3.8 compatible, numpy only.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional

import numpy as np

EPISODE_FIELDS = ["suite", "task_id", "task", "episode_idx", "success", "env_steps", "n_calls"]


def episode_uid(task_id: int, episode_idx: int) -> int:
    """Stable integer id of (task, initial state), used to seed noise="episode"."""
    return int(task_id) * 1000 + int(episode_idx)


def episode_name(task_id: int, episode_idx: int) -> str:
    return f"{int(task_id):02d}_ep{int(episode_idx):02d}"


class EpisodeRecorder:
    def __init__(self, record_images: bool):
        self.record_images = record_images
        self.calls: Dict[str, List[np.ndarray]] = {k: [] for k in ("call_index", "t", "state", "actions")}
        if record_images:
            self.calls.update(image=[], wrist_image=[])

    def add_call(self, call_index: int, t: int, obs: dict, actions: np.ndarray) -> None:
        self.calls["call_index"].append(np.asarray(call_index))
        self.calls["t"].append(np.asarray(t))
        self.calls["state"].append(np.asarray(obs["observation/state"]))
        self.calls["actions"].append(np.asarray(actions))
        if self.record_images:
            self.calls["image"].append(obs["observation/image"])
            self.calls["wrist_image"].append(obs["observation/wrist_image"])

    @property
    def n_calls(self) -> int:
        return len(self.calls["call_index"])

    def save(self, path: str, meta: dict) -> None:
        arrays = {k: np.stack(v) if v else np.zeros((0,)) for k, v in self.calls.items()}
        np.savez_compressed(path, meta=np.asarray(json.dumps(meta)), **arrays)


def load_episode(path) -> dict:
    with np.load(path) as f:
        ep = {k: f[k] for k in f.files}
    ep["meta"] = json.loads(str(ep["meta"]))
    return ep


class EpisodeTable:
    """Append-only episodes.csv, flushed after every episode so a killed job keeps its results."""

    def __init__(self, path: str):
        self.path = path
        new = not os.path.exists(path)
        self._f = open(path, "a", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=EPISODE_FIELDS)
        if new:
            self._w.writeheader()

    def add(self, row: dict) -> None:
        self._w.writerow({k: row[k] for k in EPISODE_FIELDS})
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def write_json(path: str, obj: dict, indent: Optional[int] = 2) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)
