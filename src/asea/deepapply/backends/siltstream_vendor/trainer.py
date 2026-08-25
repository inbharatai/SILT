"""Minimal LoRA trainer over streamed execution.

Deliberately small and inspectable: AdamW over the resident LoRA parameters
only; the frozen base streams through the bank every step (forward and
backward). Returns a typed report with real measured numbers -- never a
fabricated success.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

import torch

from .model import StreamedCausalLM


@dataclass
class TrainReport:
    steps: int
    initial_loss: float
    final_loss: float
    loss_curve: List[float]
    seconds: float
    streamed: bool
    config_fingerprint: str
    audit: Dict[str, object] = field(default_factory=dict)

    @property
    def improved(self) -> bool:
        return self.final_loss < self.initial_loss


def train_lora(
    model: StreamedCausalLM,
    batches: List[torch.Tensor],
    steps: int = 20,
    lr: float = 1e-3,
    streamed: bool = True,
) -> TrainReport:
    params = model.trainable_parameters()
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr)

    curve: List[float] = []
    t0 = time.time()
    for step in range(steps):
        batch = batches[step % len(batches)]
        opt.zero_grad(set_to_none=True)
        loss = model.loss(batch, streamed=streamed)
        loss.backward()
        opt.step()
        curve.append(float(loss.item()))
    seconds = time.time() - t0

    return TrainReport(
        steps=steps,
        initial_loss=curve[0],
        final_loss=curve[-1],
        loss_curve=curve,
        seconds=seconds,
        streamed=streamed,
        config_fingerprint=model.fingerprint,
        audit=model.audit_metadata(),
    )
