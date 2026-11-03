"""Load fetched runs (runs/<exp>/<arm>/, format in sim/recording.py) for analysis. Local, CPU only."""
from __future__ import annotations

import dataclasses
import functools
import json
import pathlib

import pandas as pd

from vlalab.sim import recording

ROOT = pathlib.Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"


@dataclasses.dataclass
class Arm:
    exp: str
    name: str
    dir: pathlib.Path
    config: dict
    episodes: pd.DataFrame
    summary: dict

    @property
    def key(self) -> str:
        return f"{self.exp}/{self.name}"

    @property
    def label(self) -> str:
        """Short config label, e.g. '10 steps, replan 5'."""
        n = self.config["policy"]["num_steps"]
        return f"{n} step{'s' if n > 1 else ''}, replan {self.config['eval']['replan_steps']}"

    @property
    def trials(self) -> str:
        s = self.config["eval"]["start_trial"]
        return f"{s}-{s + self.config['eval']['trials'] - 1}"

    @functools.cached_property
    def latency(self) -> pd.DataFrame | None:
        p = self.dir / "latency.csv"
        return pd.read_csv(p) if p.exists() else None

    def episode(self, task_id: int, episode_idx: int) -> dict:
        return recording.load_episode(self.dir / "episodes" / (recording.episode_name(task_id, episode_idx) + ".npz"))

    def video(self, task_id: int, episode_idx: int) -> pathlib.Path | None:
        hits = list((self.dir / "videos").glob(recording.episode_name(task_id, episode_idx) + "_*.mp4"))
        return hits[0] if hits else None


def load_arm(exp: str, name: str, runs: pathlib.Path = RUNS) -> Arm:
    d = runs / exp / name
    return Arm(
        exp=exp,
        name=name,
        dir=d,
        config=json.loads((d / "config.json").read_text()),
        episodes=pd.read_csv(d / "episodes.csv"),
        summary=json.loads((d / "summary.json").read_text()),
    )


def load_all(runs: pathlib.Path = RUNS) -> list[Arm]:
    arms = []
    for exp in sorted(p for p in runs.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for d in sorted(exp.iterdir()):
            if (d / "episodes.csv").exists():
                arms.append(load_arm(exp.name, d.name, runs))
    return arms


def vram_peak_gb(exp: str, runs: pathlib.Path = RUNS) -> dict[int, float]:
    """Peak memory.used per GPU index from the job's nvidia-smi log."""
    p = runs / exp / "vram.csv"
    if not p.exists():
        return {}
    v = pd.read_csv(p, skipinitialspace=True)
    v["mib"] = v["memory.used [MiB]"].str.replace(" MiB", "").astype(float)
    if "index" not in v:  # single-GPU log
        v["index"] = 0
    return {int(i): round(g["mib"].max() / 1024, 1) for i, g in v.groupby("index")}


def markdown(df: pd.DataFrame, floatfmt: str = ".3g") -> str:
    """DataFrame -> GitHub markdown table (no tabulate dependency)."""
    def cell(v):
        if isinstance(v, float):
            return "" if v != v else format(v, floatfmt)
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])
