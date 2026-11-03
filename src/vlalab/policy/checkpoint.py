"""Load the released pi0.5 weights, normalization stats and tokenizer.

openpi is used here only as a library: the network definition (Pi0 = SigLIP + Gemma 2B + Gemma 300M action expert),
the Orbax checkpoint reader, the norm-stats file format and the PaliGemma SentencePiece tokenizer. Everything that
happens around the network (I/O pipeline, sampling, RNG, serving) is ours.

Equivalent openpi path: policies/policy_config.py:create_trained_policy (JAX branch).
"""
from __future__ import annotations

import dataclasses
import logging
import pathlib

import jax.numpy as jnp
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.shared import download
from openpi.shared import normalize

from vlalab.config import PolicyConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Checkpoint:
    model: _model.BaseModel  # openpi Pi0 nnx module, bf16 weights
    model_config: pi0_config.Pi0Config
    norm_stats: dict[str, normalize.NormStats]  # keys "state", "actions"; quantiles q01/q99 used by pi05
    tokenizer: _tokenizer.PaligemmaTokenizer
    path: pathlib.Path


def model_config(cfg: PolicyConfig) -> pi0_config.Pi0Config:
    m = cfg.model
    return pi0_config.Pi0Config(
        pi05=m.pi05,
        action_horizon=m.action_horizon,
        action_dim=m.action_dim,
        max_token_len=m.max_token_len,
        discrete_state_input=m.discrete_state_input,
    )


def load(cfg: PolicyConfig) -> Checkpoint:
    path = download.maybe_download(cfg.checkpoint)  # gs:// -> ~/.cache/openpi/..., local paths pass through
    mcfg = model_config(cfg)
    logger.info("Loading %s (%s)", path, mcfg)
    model = mcfg.load(_model.restore_params(path / "params", dtype=jnp.bfloat16))
    # Stats from the checkpoint itself (not a config assets dir), so they match what the model was trained with.
    norm_stats = normalize.load(path / "assets" / cfg.asset_id)
    tokenizer = _tokenizer.PaligemmaTokenizer(mcfg.max_token_len)
    return Checkpoint(model=model, model_config=mcfg, norm_stats=norm_stats, tokenizer=tokenizer, path=path)
