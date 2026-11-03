"""Latency benchmark: `warmup` + `calls` policy calls with random LIBERO-shaped observations, timed at the client.

Runs against the same server right before the rollouts, so it also consumes the first warmup + calls positions of
the noise stream (noise="stream"); the LIBERO calls start at call_index = warmup + calls.

client_infer_ms is the latency to report: it covers serialization, the websocket round trip and the full
policy call including the device -> host copy. server_infer_ms covers the policy call only.

Usage: python -m vlalab.sim.bench --config arm.json --port 8000 --out latency.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import time

import numpy as np
from openpi_client import websocket_client_policy

from vlalab.config import ArmConfig

COLUMNS = ["client_infer_ms", "server_infer_ms", "server_prev_total_ms"]


def random_observation() -> dict:
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    cfg = ArmConfig.load(args.config)

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    for _ in range(cfg.bench.warmup):
        client.infer(random_observation())
    rows = []
    for _ in range(cfg.bench.calls):
        start = time.time()
        out = client.infer(random_observation())
        timing = out.get("server_timing", {})
        rows.append({
            "client_infer_ms": 1000 * (time.time() - start),
            "server_infer_ms": timing.get("infer_ms", float("nan")),
            "server_prev_total_ms": timing.get("prev_total_ms", float("nan")),
        })
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    for col in COLUMNS:
        v = np.array([r[col] for r in rows])
        logging.info("%-22s p50 %7.1f  mean %7.1f  p90 %7.1f ms", col, np.nanmedian(v), np.nanmean(v),
                     np.nanpercentile(v, 90))


if __name__ == "__main__":
    main()
