"""Symmetric per-channel integer quantization for spring states.

Deliberately simple and inspectable: weights are quantized per output
channel (row) to int8/int4/int2 with a float32 scale per channel.
Dequantization is exact arithmetic (q * scale), so a quantized state is
DETERMINISTIC: the same bytes always produce the same dequantized weights.

This is the storage format for SiltSpring's compressed states. Norm/bias
vectors (1-D tensors) are kept in float32 -- they are tiny and quantizing
them costs quality for no meaningful memory win.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

_LEVELS: Dict[str, int] = {"int8": 8, "int4": 4, "int2": 2}


def quantize_tensor(w: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric quantization. Returns (q as int8 storage, scale)."""
    if w.dim() != 2:
        raise ValueError("quantize_tensor expects a 2-D weight matrix")
    qmax = 2 ** (bits - 1) - 1  # 127 / 7 / 1
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / qmax
    # Symmetric clamp [-qmax, qmax]: rounding can only reach ±qmax anyway,
    # but the explicit symmetric bound removes any asymmetric edge case.
    q = torch.clamp(torch.round(w / scale), -qmax, qmax).to(torch.int8)
    return q, scale.to(torch.float32)


def dequantize_tensor(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale


def quantize_state(state: Dict[str, torch.Tensor], level: str) -> Dict[str, object]:
    """Quantize one layer's state dict. 1-D tensors stay float32."""
    if level not in _LEVELS:
        raise ValueError(f"unknown quantization level {level!r}; known: {sorted(_LEVELS)}")
    bits = _LEVELS[level]
    out: Dict[str, object] = {"__level__": level}
    for key, w in state.items():
        if w.dim() == 2:
            q, scale = quantize_tensor(w, bits)
            out[key] = ("q", q, scale)
        else:
            out[key] = ("f", w.detach().clone().to(torch.float32))
    return out


def dequantize_state(qstate: Dict[str, object]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, packed in qstate.items():
        if key == "__level__":
            continue
        if packed[0] == "q":
            _, q, scale = packed
            out[key] = dequantize_tensor(q, scale)
        else:
            out[key] = packed[1]
    return out


def state_bytes(qstate: Dict[str, object]) -> int:
    """Actual storage bytes of a quantized state (int8 container per value;
    int4/int2 would pack tighter on disk -- we report honest container bytes
    and note the packing headroom separately)."""
    total = 0
    for key, packed in qstate.items():
        if key == "__level__":
            continue
        if packed[0] == "q":
            _, q, scale = packed
            total += q.numel() * q.element_size() + scale.numel() * scale.element_size()
        else:
            t = packed[1]
            total += t.numel() * t.element_size()
    return total


def packed_bytes(qstate: Dict[str, object]) -> int:
    """Theoretical bytes if sub-byte levels were bit-packed on disk."""
    level = qstate.get("__level__", "int8")
    bits = _LEVELS[str(level)]
    total = 0
    for key, packed in qstate.items():
        if key == "__level__":
            continue
        if packed[0] == "q":
            _, q, scale = packed
            total += (q.numel() * bits + 7) // 8
            total += scale.numel() * scale.element_size()
        else:
            total += packed[1].numel() * packed[1].element_size()
    return total
