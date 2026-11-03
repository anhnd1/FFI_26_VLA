"""Offline replay: feed recorded policy inputs back through the policy, no simulator needed.

Uses:
  * equivalence check: replay a run's recorded calls with the run's config and compare to the recorded chunks
    (noise="stream" keys are reproduced from the recorded server call_index);
  * sampling spread: --samples K draws K extra chunks per call with independent noise, to measure how
    multimodal the policy is at each point of a trajectory (e.g. just before a failure).

--impl openpi runs openpi's own Policy (policies/policy_config.py) instead of ours, as a reference.

Usage: python -m vlalab.policy.replay --config arm.json --data-root DIR --out replay.npz
The episodes glob, impl, max_calls and samples come from the arm's `replay` section (config.py:ReplayConfig).
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import time

import jax
import numpy as np

from vlalab.config import ArmConfig
from vlalab.sim import recording

logger = logging.getLogger("vlalab.replay")


def load_calls(paths: list[pathlib.Path], max_calls: int | None) -> list[dict]:
    """Recorded calls of ONE server (one arm), sorted by call_index."""
    calls = []
    for path in sorted(paths):
        ep = recording.load_episode(path)
        if "image" not in ep:
            raise ValueError(f"{path} has no recorded images (record_images was off)")
        n = len(ep["call_index"]) if max_calls is None else min(max_calls, len(ep["call_index"]))
        for i in range(n):
            calls.append(
                {
                    "episode": path.stem,
                    "call_index": int(ep["call_index"][i]),
                    "call_in_episode": i,
                    "episode_uid": recording.episode_uid(ep["meta"]["task_id"], ep["meta"]["episode_idx"]),
                    "obs": {
                        "observation/image": ep["image"][i],
                        "observation/wrist_image": ep["wrist_image"][i],
                        "observation/state": ep["state"][i],
                        "prompt": ep["meta"]["prompt"],
                    },
                    "recorded": ep["actions"][i],
                }
            )
    return sorted(calls, key=lambda c: c["call_index"])


def make_ours(cfg: ArmConfig):
    from vlalab.policy.policy import Pi05Policy
    from vlalab.policy.rng import NoiseKeys

    policy = Pi05Policy(cfg.policy)
    keys = NoiseKeys(cfg.policy.noise, cfg.policy.seed)
    state = {"next": 0}

    def infer(call):
        while state["next"] < call["call_index"]:  # skip calls not in the replay set, keeping the stream aligned
            keys.next()
            state["next"] += 1
        state["next"] += 1
        meta = {"episode_uid": call["episode_uid"], "call_in_episode": call["call_in_episode"]}
        return policy.infer(dict(call["obs"]), keys.next(meta))["actions"]

    def sample(call, key):
        return policy.infer(dict(call["obs"]), key)["actions"]

    return infer, sample


def make_openpi(cfg: ArmConfig):
    """openpi's unmodified Policy, built the way scripts/serve_policy.py builds it for --env LIBERO."""
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    if cfg.policy.noise != "stream":
        raise ValueError("openpi's Policy only implements the stream noise mode")
    # serve_policy.py passes no sample_kwargs by default (num_steps=10 inside the model).
    sample_kwargs = None if cfg.policy.num_steps == 10 else {"num_steps": cfg.policy.num_steps}
    policy = policy_config.create_trained_policy(
        openpi_config.get_config("pi05_libero"), cfg.policy.checkpoint, sample_kwargs=sample_kwargs
    )
    policy._rng = jax.random.key(cfg.policy.seed)
    state = {"next": 0}

    def infer(call):
        while state["next"] < call["call_index"]:
            policy._rng, _ = jax.random.split(policy._rng)
            state["next"] += 1
        state["next"] += 1
        return policy.infer(dict(call["obs"]))["actions"]

    return infer, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True, type=pathlib.Path, help="where the replay.episodes glob is resolved")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)

    cfg = ArmConfig.load(args.config)
    rc = cfg.replay
    paths = sorted(args.data_root.glob(rc.episodes))
    if not paths:
        raise FileNotFoundError(f"no episodes match {rc.episodes!r} under {args.data_root}")
    calls = load_calls(paths, rc.max_calls)
    logger.info("%d calls from %d episodes (%s), impl=%s, num_steps=%d", len(calls), len(paths), rc.episodes, rc.impl,
                cfg.policy.num_steps)
    infer, sample = (make_ours if rc.impl == "vlalab" else make_openpi)(cfg)
    if rc.samples and sample is None:
        raise ValueError("replay.samples needs impl=vlalab")

    replayed, samples, times = [], [], []
    for i, call in enumerate(calls):
        t0 = time.monotonic()
        replayed.append(infer(call))
        times.append(time.monotonic() - t0)
        if rc.samples:
            base = jax.random.fold_in(jax.random.key(cfg.policy.seed + 1), call["call_index"])
            samples.append([sample(call, jax.random.fold_in(base, k)) for k in range(rc.samples)])
        diff = np.abs(replayed[-1] - call["recorded"]).max()
        logger.info("%d/%d call %d (%s): max|diff| %.3g, %.0f ms", i + 1, len(calls), call["call_index"],
                    call["episode"], diff, times[-1] * 1000)

    out = {
        "episode": np.array([c["episode"] for c in calls]),
        "call_index": np.array([c["call_index"] for c in calls]),
        "recorded": np.stack([c["recorded"] for c in calls]),
        "replayed": np.stack(replayed),
        "infer_s": np.array(times),
    }
    if rc.samples:
        out["samples"] = np.array(samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    d = np.abs(out["replayed"] - out["recorded"])
    logger.info("max|diff| %.3g, mean|diff| %.3g, bitwise equal calls %d/%d", d.max(), d.mean(),
                int((d.reshape(len(calls), -1).max(1) == 0).sum()), len(calls))


if __name__ == "__main__":
    main()
