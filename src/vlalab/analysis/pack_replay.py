"""Build the dataset for a replay job (experiments/verify.yaml) from local runs.

Copies the first episodes of the selected arms, truncated to --max-calls calls, into one flat directory as
<exp>__<arm>__<episode>.npz (Kaggle datasets are flat), plus the vlalab_pack.json marker kaggle/job.sh looks for.

Usage: python -m vlalab.analysis.pack_replay --out build/replay_pack j1/base_t0 j2/steps1 j2/steps3
       kaggle datasets create -p build/replay_pack     (first time; dataset-metadata.json is written here)
       kaggle datasets version -p build/replay_pack -m "update"
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from vlalab.analysis.runs import RUNS
from vlalab.sim import recording


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="exp/arm, e.g. j1/base_t0")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--episodes", type=int, default=1, help="first N episodes of each arm")
    ap.add_argument("--max-calls", type=int, default=40)
    ap.add_argument("--slug", default="vlalab-replay-pack")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    index = []
    for arm in args.arms:
        exp, name = arm.split("/")
        for path in sorted((RUNS / exp / name / "episodes").glob("*.npz"))[: args.episodes]:
            ep = recording.load_episode(path)
            n = min(args.max_calls, len(ep["call_index"]))
            arrays = {k: v[:n] for k, v in ep.items() if k != "meta"}
            dst = args.out / f"{exp}__{name}__{path.stem}.npz"
            np.savez_compressed(dst, meta=np.asarray(json.dumps(ep["meta"])), **arrays)
            index.append({"file": dst.name, "calls": n, "first_call_index": int(ep["call_index"][0])})
            print(dst.name, n, "calls")
    (args.out / "vlalab_pack.json").write_text(json.dumps(index, indent=1))
    user = json.loads((pathlib.Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]
    (args.out / "dataset-metadata.json").write_text(json.dumps(
        {"title": args.slug, "id": f"{user}/{args.slug}", "licenses": [{"name": "CC0-1.0"}]}, indent=1))


if __name__ == "__main__":
    main()
