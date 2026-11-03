"""Websocket policy server: the LIBERO client (Python 3.8) and the policy (Python 3.11 + JAX) can't share a process.

Protocol (msgpack-numpy, as openpi_client's WebsocketClientPolicy expects):
  on connect   server -> client: metadata dict
  per call     client -> server: observation dict (pipeline.py), plus optional "meta/episode_uid" and
                                 "meta/call_in_episode" (used by noise="episode")
               server -> client: {"actions": (H, 7), "call_index": int, "server_timing": {"infer_ms", "prev_total_ms"}}
  GET /healthz  -> 200 once the model is loaded and compiled.

Usage: python -m vlalab.policy.server --config arm.json --port 8000
"""
from __future__ import annotations

import argparse
import asyncio
import http
import logging
import time
import traceback

import numpy as np
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

from vlalab.config import ArmConfig
from vlalab.policy.policy import Pi05Policy
from vlalab.policy.rng import NoiseKeys

logger = logging.getLogger("vlalab.server")
META_PREFIX = "meta/"


class PolicyServer:
    def __init__(self, policy: Pi05Policy, cfg: ArmConfig):
        self.policy = policy
        self.keys = NoiseKeys(cfg.policy.noise, cfg.policy.seed)
        self.call_index = 0
        self.metadata = {"arm": cfg.to_dict()}

    def infer(self, obs: dict) -> dict:
        meta = {k[len(META_PREFIX) :]: obs.pop(k) for k in list(obs) if k.startswith(META_PREFIX)}
        key = self.keys.next(meta)
        out = self.policy.infer(obs, key)
        out["call_index"] = self.call_index
        self.call_index += 1
        return out

    async def handler(self, websocket: _server.ServerConnection):
        logger.info("connection from %s", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.metadata))
        prev_total = None
        while True:
            try:
                start = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                t0 = time.monotonic()
                out = self.infer(obs)
                infer_ms = (time.monotonic() - t0) * 1000
                out["server_timing"] = {"infer_ms": infer_ms}
                if prev_total is not None:  # includes the previous send, so it lags by one call
                    out["server_timing"]["prev_total_ms"] = prev_total * 1000
                await websocket.send(packer.pack(out))
                prev_total = time.monotonic() - start
                logger.info("call %d: %.0f ms", out["call_index"], infer_ms)
            except websockets.ConnectionClosed:
                logger.info("connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="policy error")
                raise

    async def serve(self, port: int):
        async with _server.serve(
            self.handler, "0.0.0.0", port, compression=None, max_size=None, process_request=_health_check
        ) as server:
            logger.info("serving on port %d", port)
            await server.serve_forever()


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="arm JSON written by kaggle/push.py")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    cfg = ArmConfig.load(args.config)
    policy = Pi05Policy(cfg.policy)
    # Compile before reporting healthy, with a throwaway key so the noise stream is untouched.
    import jax

    h, _ = policy.action_shape
    dummy = {
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
        "observation/state": np.zeros(8),
        "prompt": "warm up",
    }
    t0 = time.monotonic()
    policy.infer(dummy, jax.random.key(12345))
    logger.info("compiled num_steps=%d in %.1f s (chunk of %d actions)", cfg.policy.num_steps, time.monotonic() - t0, h)
    asyncio.run(PolicyServer(policy, cfg).serve(args.port))


if __name__ == "__main__":
    main()
