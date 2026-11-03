"""pi0.5 policy: client observation -> action chunk. Stateless apart from the compiled sampler; the caller owns the RNG.

Equivalent openpi path: policies/policy.py:Policy.infer.
"""
from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model

from vlalab.config import PolicyConfig
from vlalab.policy import checkpoint as _checkpoint
from vlalab.policy.pipeline import LiberoPipeline
from vlalab.policy.sampler import Sampler

logger = logging.getLogger(__name__)


class Pi05Policy:
    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg
        self.ckpt = _checkpoint.load(cfg)
        self.pipeline = LiberoPipeline(self.ckpt)
        self.sampler = Sampler(self.ckpt.model)

    @property
    def action_shape(self) -> tuple[int, int]:
        """Shape of the noise / raw model output (H, 32)."""
        return self.ckpt.model_config.action_horizon, self.ckpt.model_config.action_dim

    def infer(self, obs: dict, rng: jax.Array, *, num_steps: int | None = None, noise: np.ndarray | None = None) -> dict:
        """obs: client observation (see pipeline.py). Returns {"actions": (H, 7) float64}.

        noise: optional (H, 32) initial x_1; if given, rng is unused and the call is deterministic.
        """
        inputs = self.pipeline.inputs(obs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)  # batch of 1
        observation = _model.Observation.from_dict(inputs)  # uint8 images -> float32 in [-1, 1]
        if noise is not None:
            noise = jnp.asarray(noise)[np.newaxis, ...]
        actions = self.sampler(rng, observation, num_steps=num_steps or self.cfg.num_steps, noise=noise)
        actions = np.asarray(actions[0])  # device -> host, so timing around infer() is real
        return {"actions": self.pipeline.outputs(actions)}
