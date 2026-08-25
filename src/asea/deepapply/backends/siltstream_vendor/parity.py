"""Parity harness -- the acceptance bar for streamed execution.

Forward and backward are verified as TWO SEPARATE claims:

  forward parity : streamed logits vs resident logits
  backward parity: streamed LoRA gradients vs resident LoRA gradients

On CPU float32 with deterministic ops both must match BITWISE (max_abs_diff
== 0.0). On other devices/dtypes the caller may pass a tolerance, but must
justify it; the report records device, dtype and the achieved differences so
a trust layer can decide what "verified" means for that configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

from .errors import ParityError
from .model import StreamedCausalLM


@dataclass
class ParityReport:
    config_fingerprint: str
    device: str
    dtype: str
    forward_max_abs_diff: float
    backward_max_abs_diff: float
    forward_bitwise: bool
    backward_bitwise: bool
    n_params_compared: int
    tolerance: float
    passed: bool
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


def verify_parity(
    model: StreamedCausalLM,
    input_ids: torch.Tensor,
    tolerance: float = 0.0,
    raise_on_fail: bool = True,
) -> ParityReport:
    """Run the same batch resident and streamed; compare outputs and grads."""
    notes: List[str] = []

    # ---- forward parity (no grad) ------------------------------------------
    with torch.no_grad():
        logits_resident = model.forward(input_ids, streamed=False)
        logits_streamed = model.forward(input_ids, streamed=True)
    fwd_diff = (logits_resident - logits_streamed).abs().max().item()

    # ---- backward parity ----------------------------------------------------
    params = model.trainable_parameters()
    for p in params:
        p.requires_grad_(True)

    def grads_for(streamed: bool) -> List[torch.Tensor]:
        model.zero_grad(set_to_none=True)
        loss = model.loss(input_ids, streamed=streamed)
        loss.backward()
        out = []
        for p in params:
            if p.grad is None:
                raise ParityError("a trainable LoRA parameter received no gradient")
            out.append(p.grad.detach().clone())
        return out

    g_resident = grads_for(streamed=False)
    g_streamed = grads_for(streamed=True)
    bwd_diff = 0.0
    for gr, gs in zip(g_resident, g_streamed):
        bwd_diff = max(bwd_diff, (gr - gs).abs().max().item())
    # Clean up side effects: the harness must not leave gradients behind
    # (a later ZeroForge or inference call should see a pristine model).
    model.zero_grad(set_to_none=True)

    passed = fwd_diff <= tolerance and bwd_diff <= tolerance
    if tolerance == 0.0:
        notes.append("bitwise comparison (tolerance 0.0)")
    report = ParityReport(
        config_fingerprint=model.fingerprint,
        device=str(model.device_),
        dtype=str(next(iter(model.lm_head.parameters())).dtype),
        forward_max_abs_diff=fwd_diff,
        backward_max_abs_diff=bwd_diff,
        forward_bitwise=fwd_diff == 0.0,
        backward_bitwise=bwd_diff == 0.0,
        n_params_compared=len(params),
        tolerance=tolerance,
        passed=passed,
        notes=notes,
    )
    if raise_on_fail and not passed:
        raise ParityError(
            f"parity failed: forward max|diff|={fwd_diff:.3e}, "
            f"backward max|diff|={bwd_diff:.3e}, tolerance={tolerance:.3e} "
            f"(config {model.fingerprint})"
        )
    return report
