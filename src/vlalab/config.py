"""Experiment configuration shared by the policy server, the LIBERO client and the Kaggle launcher.

One *arm* = one policy server + one LIBERO client with its own settings. An experiment file (experiments/*.yaml) lists
arms; kaggle/push.py resolves it into one JSON file per arm, which both processes read with `ArmConfig.load`.

Must stay importable on Python 3.8 (the LIBERO client env): stdlib only, no `X | Y` types at runtime.
"""
from __future__ import annotations

import copy
import dataclasses
import json
from typing import Any, Dict, List, Optional


@dataclasses.dataclass
class ModelConfig:
    """Architecture of the checkpoint. Mirrors openpi's `pi05_libero` TrainConfig (training/config.py)."""

    pi05: bool = True  # adaRMSNorm time conditioning, no continuous state token
    action_horizon: int = 10  # actions per predicted chunk
    action_dim: int = 32  # padded action/state width of the action expert
    max_token_len: int = 200  # prompt tokens (pi05 default)
    discrete_state_input: bool = False  # pi05_libero does NOT put the robot state in the prompt


@dataclasses.dataclass
class PolicyConfig:
    checkpoint: str = "gs://openpi-assets/checkpoints/pi05_libero"
    asset_id: str = "physical-intelligence/libero"  # norm stats live in <checkpoint>/assets/<asset_id>/
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    num_steps: int = 10  # Euler steps of the flow ODE, t: 1 (noise) -> 0 (actions)
    # How the initial noise of each call is drawn (see policy/rng.py):
    #   "stream":   one key(seed), split once per server call. openpi's behaviour; a call's noise depends on how
    #               many calls came before it (benchmark calls included).
    #   "episode":  key = fold_in(fold_in(key(seed), episode_uid), call_in_episode). The same episode and call
    #               always get the same noise, whatever the call count of other episodes or configs.
    noise: str = "stream"
    seed: int = 0


@dataclasses.dataclass
class EvalConfig:
    suite: str = "libero_10"  # libero_spatial | libero_object | libero_goal | libero_10 | libero_90
    tasks: Optional[List[int]] = None  # task ids to run (None = all tasks of the suite)
    trials: int = 2  # episodes per task
    start_trial: int = 0  # first LIBERO initial-state index; arms can split the 50 fixed states
    replan_steps: int = 5  # actions executed per policy call (<= action_horizon)
    max_steps: Optional[int] = None  # None = suite default (sim/libero_env.py:MAX_STEPS)
    num_steps_wait: int = 10  # dummy steps while objects settle after reset
    resize: int = 224  # policy image size (resize with padding)
    env_resolution: int = 256  # LIBERO render size, as in the training data
    seed: int = 7  # numpy seed and env.seed (affects object poses even with fixed initial states)
    save_video: bool = True
    record_images: bool = True  # store the policy's input images per call (needed for offline replay)


@dataclasses.dataclass
class BenchConfig:
    """Latency benchmark run against the server before the rollouts (random LIBERO-shaped observations)."""

    warmup: int = 2
    calls: int = 10


@dataclasses.dataclass
class ReplayConfig:
    """Offline replay of recorded calls (policy/replay.py), used by experiments with `mode: replay`."""

    episodes: str = ""  # glob of episode .npz files, relative to the attached Kaggle dataset
    impl: str = "vlalab"  # "vlalab" (ours) or "openpi" (openpi's own Policy, as a reference)
    max_calls: Optional[int] = None  # replay only the first N calls of each episode
    samples: int = 0  # extra independent-noise chunks per call (sampling spread)


@dataclasses.dataclass
class ArmConfig:
    name: str = "base"
    policy: PolicyConfig = dataclasses.field(default_factory=PolicyConfig)
    eval: EvalConfig = dataclasses.field(default_factory=EvalConfig)
    bench: BenchConfig = dataclasses.field(default_factory=BenchConfig)
    replay: ReplayConfig = dataclasses.field(default_factory=ReplayConfig)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArmConfig":
        d = copy.deepcopy(d)
        policy = d.pop("policy", {})
        model = ModelConfig(**policy.pop("model", {}))
        arm = cls(
            name=d.pop("name", "base"),
            policy=PolicyConfig(model=model, **policy),
            eval=EvalConfig(**d.pop("eval", {})),
            bench=BenchConfig(**d.pop("bench", {})),
            replay=ReplayConfig(**d.pop("replay", {})),
        )
        if d:
            raise ValueError(f"unknown config keys: {sorted(d)}")
        arm.validate()
        return arm

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def load(cls, path: str) -> "ArmConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def validate(self) -> None:
        if not 1 <= self.eval.replan_steps <= self.policy.model.action_horizon:
            raise ValueError(f"replan_steps must be in [1, {self.policy.model.action_horizon}]")
        if self.policy.noise not in ("stream", "episode"):
            raise ValueError(f"unknown noise mode {self.policy.noise!r}")
        if self.policy.num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if self.replay.impl not in ("vlalab", "openpi"):
            raise ValueError(f"unknown replay impl {self.replay.impl!r}")


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_override(text: str) -> Dict[str, Any]:
    """'policy.num_steps=3' -> {'policy': {'num_steps': 3}}. Values are parsed as JSON when possible, else str."""
    key, sep, raw = text.partition("=")
    if not sep:
        raise ValueError(f"override must look like key.path=value, got {text!r}")
    try:
        value = json.loads(raw)
    except ValueError:
        value = raw
    out: Dict[str, Any] = value
    for part in reversed(key.strip().split(".")):
        out = {part: out}
    return out


def resolve_experiment(spec: Dict[str, Any], overrides: List[str] = ()) -> List[ArmConfig]:
    """Experiment dict (from experiments/*.yaml) -> one ArmConfig per arm.

    spec = {"defaults": {...}, "arms": {name: {...}}}. An override "arm.key=value" applies to one arm,
    "key=value" (first part being policy/eval/bench/replay) applies to all arms.
    """
    defaults = spec.get("defaults", {}) or {}
    arms = spec.get("arms") or {"base": {}}
    for text in overrides:
        o = parse_override(text)
        (head, rest), = o.items()
        if head in ("policy", "eval", "bench", "replay"):
            defaults = deep_merge(defaults, o)
        elif head in arms:
            arms[head] = deep_merge(arms[head] or {}, rest)
        else:
            raise ValueError(f"override {text!r}: {head!r} is neither a section nor an arm ({sorted(arms)})")
    return [ArmConfig.from_dict(deep_merge(defaults, dict(arm or {}, name=name))) for name, arm in arms.items()]
