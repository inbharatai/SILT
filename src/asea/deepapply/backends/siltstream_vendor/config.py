"""Model and streaming configuration.

The config is hashable so downstream trust layers (e.g. SILT's AdapterPacket)
can record exactly which configuration a parity check covered.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    """A small GPT-style decoder-only causal LM configuration."""

    vocab_size: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 64
    d_ff: int = 256
    max_seq_len: int = 128
    # LoRA
    lora_rank: int = 4
    lora_alpha: float = 8.0
    # LoRA target projections inside each block
    lora_targets: tuple = ("q", "v", "fc2")

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")


@dataclass(frozen=True)
class StreamConfig:
    """How layers are stored and moved.

    storage_tier: "ram"  -- layer weights held on host RAM (CPU tensors)
                  "disk" -- layer weights memory-mapped from per-layer files
    compute_device: torch device string for the math ("cpu", "cuda", "mps")
    prefetch: overlap next-layer fetch with current-layer compute (CUDA only;
              honestly a no-op on CPU where fetch IS compute-local)
    seed: global seed recorded for determinism/audit
    """

    storage_tier: str = "ram"
    compute_device: str = "cpu"
    prefetch: bool = False
    seed: int = 20260815
    disk_dir: str = ""

    def __post_init__(self) -> None:
        if self.storage_tier not in ("ram", "disk"):
            raise ValueError(f"unknown storage_tier: {self.storage_tier!r}")
        if self.storage_tier == "disk" and not self.disk_dir:
            raise ValueError("disk storage tier requires disk_dir")


def config_fingerprint(model_cfg: ModelConfig, stream_cfg: StreamConfig) -> str:
    """Stable hash of the full configuration, for audit metadata."""
    blob = json.dumps(
        {"model": asdict(model_cfg), "stream": asdict(stream_cfg)},
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
