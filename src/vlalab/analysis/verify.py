"""Summarize a replay job (experiments/verify.yaml): our policy vs openpi's Policy vs the recorded chunks.

Usage: python -m vlalab.analysis.verify runs/verify     -> prints a table, writes results/tables/verify.md
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

from vlalab.analysis.runs import RESULTS, markdown


def main() -> None:
    run = pathlib.Path(sys.argv[1])
    out = {d.name: np.load(d / "replay.npz") for d in sorted(run.iterdir()) if (d / "replay.npz").exists()}
    rows = []
    for name, r in out.items():
        if not name.startswith("ours_"):
            continue
        steps = name[len("ours_"):]
        ref = out.get("openpi_" + steps)
        d_rec = np.abs(r["replayed"] - r["recorded"]).reshape(len(r["call_index"]), -1).max(1)
        row = {
            "config": steps,
            "calls": len(r["call_index"]),
            "ours vs recorded: max abs diff": d_rec.max(),
            "ours vs recorded: bitwise equal calls": int((d_rec == 0).sum()),
        }
        if ref is not None:
            assert (ref["call_index"] == r["call_index"]).all()
            d_ref = np.abs(r["replayed"] - ref["replayed"]).reshape(len(r["call_index"]), -1).max(1)
            d_ref_rec = np.abs(ref["replayed"] - ref["recorded"]).reshape(len(r["call_index"]), -1).max(1)
            row.update({
                "ours vs openpi: max abs diff": d_ref.max(),
                "ours vs openpi: bitwise equal calls": int((d_ref == 0).sum()),
                "openpi vs recorded: max abs diff": d_ref_rec.max(),
            })
        rows.append(row)
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    (RESULTS / "tables" / "verify.md").write_text(markdown(table) + "\n")


if __name__ == "__main__":
    main()
