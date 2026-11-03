"""LIBERO <-> pi0.5 input/output pipeline, written out step by step.

Client observation (sim/evaluate.py):
    observation/image        (224, 224, 3) uint8   agent view, already rotated 180 deg and resized with padding
    observation/wrist_image  (224, 224, 3) uint8   eye-in-hand view, same preprocessing
    observation/state        (8,) float            eef pos (3) + axis-angle (3) + gripper qpos (2)
    prompt                   str                   LIBERO task language

Model input (unbatched), consumed by openpi's `Observation.from_dict`:
    image/{base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb}  uint8; right wrist = zeros, masked out
    image_mask/{...}                                          bool
    state                     (32,)  quantile-normalized, zero-padded (pi05_libero ignores it, see ModelConfig)
    tokenized_prompt(_mask)   (200,) PaliGemma tokens of the prompt (+ discretized state if discrete_state_input)

Output: model actions (H, 32) in normalized space -> unnormalize -> first 7 dims (6 delta eef pose + gripper).

Same steps and order as openpi's chain for `pi05_libero` (policy_config.py:create_trained_policy):
LiberoInputs -> Normalize(quantile) -> ResizeImages(224) -> TokenizePrompt -> PadStatesAndActions, and
Unnormalize(quantile) -> LiberoOutputs. Each function below names the openpi transform it replaces.
"""
from __future__ import annotations

import numpy as np
from openpi.shared import image_tools
from openpi.shared import normalize

from vlalab.policy.checkpoint import Checkpoint

LIBERO_ACTION_DIM = 7
IMAGE_SIZE = 224


def parse_image(image) -> np.ndarray:
    """float CHW/HWC -> uint8 HWC (libero_policy.py:_parse_image)."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


def normalize_quantile(x: np.ndarray, stats: normalize.NormStats) -> np.ndarray:
    """Map [q01, q99] -> [-1, 1] (transforms.py:Normalize._normalize_quantile)."""
    q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def unnormalize_quantile(x: np.ndarray, stats: normalize.NormStats) -> np.ndarray:
    """Inverse of normalize_quantile; dims beyond the stats (padding) pass through (Unnormalize._unnormalize_quantile)."""
    q01, q99 = stats.q01, stats.q99
    if (dim := q01.shape[-1]) < x.shape[-1]:
        return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def pad_to_dim(x: np.ndarray, dim: int) -> np.ndarray:
    """Zero-pad the last axis (transforms.py:pad_to_dim)."""
    if x.shape[-1] < dim:
        return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, dim - x.shape[-1])])
    return x


class LiberoPipeline:
    def __init__(self, ckpt: Checkpoint):
        self.norm_stats = ckpt.norm_stats
        self.tokenizer = ckpt.tokenizer
        self.action_dim = ckpt.model_config.action_dim
        self.discrete_state_input = ckpt.model_config.discrete_state_input

    def inputs(self, obs: dict) -> dict:
        # 1. LiberoInputs: rename keys; pi0.5 has 3 camera slots, LIBERO fills 2 and masks the third.
        base = parse_image(obs["observation/image"])
        wrist = parse_image(obs["observation/wrist_image"])
        images = {"base_0_rgb": base, "left_wrist_0_rgb": wrist, "right_wrist_0_rgb": np.zeros_like(base)}
        image_mask = {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.False_}
        # 2. Normalize: only the state exists at inference time.
        state = normalize_quantile(np.asarray(obs["observation/state"]), self.norm_stats["state"])
        # 3. ResizeImages: a no-op for the client's 224x224 images, kept for other input sizes.
        images = {k: image_tools.resize_with_pad(v, IMAGE_SIZE, IMAGE_SIZE) for k, v in images.items()}
        # 4. TokenizePrompt: "<bos>prompt\n", or "Task: ..., State: <256-bin ids>;\nAction: " with discrete state.
        prompt = obs["prompt"] if isinstance(obs["prompt"], str) else obs["prompt"].item()
        tokens, token_mask = self.tokenizer.tokenize(prompt, state if self.discrete_state_input else None)
        # 5. PadStatesAndActions
        state = pad_to_dim(state, self.action_dim)
        return {
            "state": state,
            "image": images,
            "image_mask": image_mask,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
        }

    def outputs(self, actions: np.ndarray) -> np.ndarray:
        """(H, 32) normalized model actions -> (H, 7) LIBERO actions."""
        actions = unnormalize_quantile(actions, self.norm_stats["actions"])  # Unnormalize
        return np.asarray(actions[..., :LIBERO_ACTION_DIM])  # LiberoOutputs
