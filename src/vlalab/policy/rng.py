"""Where each call's initial flow noise comes from (PolicyConfig.noise)."""
from __future__ import annotations

import jax


class NoiseKeys:
    """Hands out one PRNG key per policy call.

    "stream":  key(seed) split once per call, in arrival order. This is openpi's Policy.infer behaviour, so the
               noise of a LIBERO call depends on every call made before it on the same server, including the
               latency-benchmark calls. Changing the number of calls per episode (e.g. replan_steps) therefore
               changes the noise of every later episode.
    "episode": fold_in(fold_in(key(seed), episode_uid), call_in_episode). An episode's noise sequence no longer
               depends on what ran before it, so configs can be compared on identical noise.
    """

    def __init__(self, mode: str, seed: int):
        self.mode = mode
        self._root = jax.random.key(seed)
        self._stream = self._root

    def next(self, meta: dict | None = None) -> jax.Array:
        # Always advance the stream, so the call counter and the stream stay aligned in both modes.
        self._stream, key = jax.random.split(self._stream)
        if self.mode == "stream":
            return key
        if not meta or "episode_uid" not in meta:
            return key  # calls without episode info (benchmark) fall back to the stream
        key = jax.random.fold_in(self._root, int(meta["episode_uid"]))
        return jax.random.fold_in(key, int(meta["call_in_episode"]))
