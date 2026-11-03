"""Flow-matching action sampler for pi0.5: noise -> action chunk.

The network (openpi `Pi0`) gives two pieces:
  prefix  = SigLIP tokens of the 3 camera slots + Gemma-2B embedding of the prompt. Run once per call; its
            key/value cache is reused by every integration step (the prefix never attends to the actions).
  v(x_t, t | prefix) = Gemma-300M action expert on the noisy chunk x_t, with t injected through adaRMSNorm (pi05).
Sampling integrates dx/dt = v from t = 1 (Gaussian noise) to t = 0 (actions) with `num_steps` Euler steps.
Time convention: openpi trains x_t = t * noise + (1 - t) * actions, target v = noise - actions, so t = 1 is noise
(the reverse of the pi0 paper's notation).

`sample_actions` reproduces openpi's `Pi0.sample_actions` (models/pi0.py) operation for operation, so the released
results are reproduced exactly; the pieces are separate functions so the integrator, the noise and the step schedule
can be changed independently.
"""
from __future__ import annotations

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask


def encode_prefix(model, obs: _model.Observation):
    """Images + prompt -> Gemma-2B KV cache. Returns (kv_cache, prefix_mask)."""
    tokens, mask, ar_mask = model.embed_prefix(obs)
    attn_mask = make_attn_mask(mask, ar_mask)
    positions = jnp.cumsum(mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([tokens, None], mask=attn_mask, positions=positions)
    return kv_cache, mask


def velocity(model, obs: _model.Observation, kv_cache, prefix_mask, x_t, t):
    """Action-expert velocity v(x_t, t) for a batch of noisy chunks x_t (b, H, D) at scalar time t."""
    batch_size = x_t.shape[0]
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
        obs, x_t, jnp.broadcast_to(t, batch_size)
    )
    # Action tokens attend to the whole prefix and causally (blockwise) among themselves.
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [None, suffix_tokens],
        mask=full_attn_mask,
        positions=positions,
        kv_cache=kv_cache,
        adarms_cond=[None, adarms_cond],
    )
    assert prefix_out is None
    return model.action_out_proj(suffix_out[:, -model.action_horizon :])


def euler(v_fn, noise, num_steps: int):
    """Integrate dx/dt = v_fn(x, t) from t=1 to t=0 with fixed steps (same loop and float ops as openpi)."""
    dt = -1.0 / num_steps

    def step(carry):
        x_t, t = carry
        return x_t + dt * v_fn(x_t, t), t + dt

    def cond(carry):
        _, t = carry
        return t >= -dt / 2  # robust to floating-point error in the accumulated t

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0


def sample_actions(model, rng, obs: _model.Observation, *, num_steps: int, noise=None):
    """Draw one action chunk per batch element: (b, H, D), in the model's normalized action space."""
    obs = _model.preprocess_observation(None, obs, train=False)  # image masks, no augmentation
    if noise is None:
        noise = jax.random.normal(rng, (obs.state.shape[0], model.action_horizon, model.action_dim))
    kv_cache, prefix_mask = encode_prefix(model, obs)
    return euler(lambda x_t, t: velocity(model, obs, kv_cache, prefix_mask, x_t, t), noise, num_steps)


class Sampler:
    """jit-compiled `sample_actions` with the model weights frozen in (as openpi's nnx_utils.module_jit).

    num_steps is static: each new value compiles once (~1 min on a T4).
    """

    def __init__(self, model):
        graphdef, state = nnx.split(model)

        def fn(state, rng, obs, noise, num_steps):
            return sample_actions(nnx.merge(graphdef, state), rng, obs, num_steps=num_steps, noise=noise)

        self._state = state
        self._fn = jax.jit(fn, static_argnames=("num_steps",))

    def __call__(self, rng, obs: _model.Observation, *, num_steps: int, noise=None):
        return self._fn(self._state, rng, obs, noise, num_steps=num_steps)
