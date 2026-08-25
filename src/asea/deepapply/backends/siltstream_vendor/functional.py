"""Pure-functional decoder block.

The block is a FUNCTION of (activations, frozen weights, LoRA params) with no
module state. This is the core design decision that makes streaming exact:
the same function runs whether the weights arrived from a resident dict or
were just fetched from the bank -- there is no module surgery, no buffer
mutation, nothing that can drift between the two modes.

Weight-key contract per layer (all frozen):
  ln1.w ln1.b
  attn.q.w attn.q.b attn.k.w attn.k.b attn.v.w attn.v.b attn.o.w attn.o.b
  ln2.w ln2.b
  mlp.fc1.w mlp.fc1.b mlp.fc2.w mlp.fc2.b

LoRA params (trainable, resident, per layer): {target: (A, B)} where the
adapted projection computes  y = x @ W.T + b + (alpha/r) * ((x @ A.T) @ B.T).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

LoraPair = Tuple[torch.Tensor, torch.Tensor]


def lora_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    lora: Optional[LoraPair],
    scaling: float,
) -> torch.Tensor:
    y = F.linear(x, w, b)
    if lora is not None:
        a, bb = lora
        y = y + F.linear(F.linear(x, a), bb) * scaling
    return y


def block_forward(
    x: torch.Tensor,
    weights: Dict[str, torch.Tensor],
    adapters: Dict[str, LoraPair],
    n_heads: int,
    scaling: float,
) -> torch.Tensor:
    """One pre-norm decoder block with causal self-attention."""
    bsz, seq, d_model = x.shape
    head_dim = d_model // n_heads

    h = F.layer_norm(x, (d_model,), weights["ln1.w"], weights["ln1.b"])
    q = lora_linear(h, weights["attn.q.w"], weights["attn.q.b"], adapters.get("q"), scaling)
    k = lora_linear(h, weights["attn.k.w"], weights["attn.k.b"], adapters.get("k"), scaling)
    v = lora_linear(h, weights["attn.v.w"], weights["attn.v.b"], adapters.get("v"), scaling)

    def split(t: torch.Tensor) -> torch.Tensor:
        return t.view(bsz, seq, n_heads, head_dim).transpose(1, 2)

    q, k, v = split(q), split(k), split(v)
    scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
    causal = torch.triu(
        torch.full((seq, seq), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1
    )
    scores = scores + causal
    attn = torch.softmax(scores, dim=-1)
    ctx = (attn @ v).transpose(1, 2).contiguous().view(bsz, seq, d_model)
    ctx = lora_linear(ctx, weights["attn.o.w"], weights["attn.o.b"], adapters.get("o"), scaling)
    x = x + ctx

    h = F.layer_norm(x, (d_model,), weights["ln2.w"], weights["ln2.b"])
    h = lora_linear(h, weights["mlp.fc1.w"], weights["mlp.fc1.b"], adapters.get("fc1"), scaling)
    h = F.gelu(h)
    h = lora_linear(h, weights["mlp.fc2.w"], weights["mlp.fc2.b"], adapters.get("fc2"), scaling)
    return x + h


def init_block_state(cfg, generator: torch.Generator) -> Dict[str, torch.Tensor]:
    """Random-init one block's frozen weights (deterministic via generator)."""
    d, ff = cfg.d_model, cfg.d_ff

    def lin(out_f: int, in_f: int) -> Tuple[torch.Tensor, torch.Tensor]:
        w = torch.empty(out_f, in_f)
        torch.nn.init.normal_(w, std=0.02, generator=generator)
        return w, torch.zeros(out_f)

    state: Dict[str, torch.Tensor] = {}
    state["ln1.w"], state["ln1.b"] = torch.ones(d), torch.zeros(d)
    for name in ("q", "k", "v", "o"):
        w, b = lin(d, d)
        state[f"attn.{name}.w"], state[f"attn.{name}.b"] = w, b
    state["ln2.w"], state["ln2.b"] = torch.ones(d), torch.zeros(d)
    w, b = lin(ff, d)
    state["mlp.fc1.w"], state["mlp.fc1.b"] = w, b
    w, b = lin(d, ff)
    state["mlp.fc2.w"], state["mlp.fc2.b"] = w, b
    return state
