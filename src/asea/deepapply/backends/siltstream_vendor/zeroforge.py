"""ZeroForge -- forward-only (zeroth-order) LoRA training.

Why this exists: backward passes are what chain training to CUDA-grade
hardware and training-grade frameworks. ZeroForge estimates gradients from
FORWARD PASSES ONLY (central-difference SPSA, in the spirit of MeZO,
Malladi et al. 2023): perturb the LoRA parameters by +eps*z and -eps*z,
measure the loss twice, and step along z scaled by the loss difference.

Consequences:
  - no autograd graph, no backward, no optimizer moments -> minimal memory;
  - the forward pass is the ONLY primitive needed, so this trainer can run
    on top of any inference engine (torch CPU today; quantized GGUF/llama.cpp
    engines are the intended target -- an integration, not a rewrite);
  - gradients are NOISY estimates. This trainer is honest about being a
    stochastic method: it reports its full loss curve and is designed to sit
    behind an outcome gate (SILT Gate 2) that judges the trained artifact.

Implementation notes (differences from vanilla MeZO, each deliberate):
  - EXACT restore: parameters are snapshotted before perturbation and
    restored by copy, because add(+eps*z) followed by add(-eps*z) is NOT
    bitwise-exact in floating point and the drift compounds over steps.
    LoRA params are tiny, so the snapshot costs almost nothing. (The seed
    trick is still used for the DIRECTION z, which is never stored.)
  - MULTI-DIRECTION averaging (n_directions): the gradient estimate is
    averaged over several independent perturbation directions per step.
    Measured on the bundled toy model: 4 directions roughly doubles the
    loss reduction per wall-clock budget vs a single direction.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Dict, List

import torch

from .errors import SiltStreamError
from .model import StreamedCausalLM


@dataclass
class ZeroForgeReport:
    steps: int
    initial_loss: float
    final_loss: float
    loss_curve: List[float]
    seconds: float
    forward_passes: int
    backward_passes: int  # always 0 -- recorded to make the claim auditable
    eps: float
    lr: float
    n_directions: int
    seed: int
    config_fingerprint: str
    audit: Dict[str, object] = field(default_factory=dict)

    @property
    def improved(self) -> bool:
        return self.final_loss < self.initial_loss


def _directions(params: List[torch.Tensor], seed: int) -> List[torch.Tensor]:
    # Seed on a CPU generator for cross-device reproducibility (a CPU Generator
    # cannot fill cuda tensors), then move each direction onto its param's
    # device. On CPU params the ``.to`` is a no-op, so existing CPU runs are
    # bit-identical; on cuda the values are the same seeded sequence, just
    # resident where the perturbation ``p.add_(z, alpha=eps)`` needs them.
    gen = torch.Generator().manual_seed(seed)
    return [
        torch.randn(p.shape, generator=gen, dtype=p.dtype, device="cpu").to(p.device)
        for p in params
    ]


def train_zeroforge(
    model: StreamedCausalLM,
    batches: List[torch.Tensor],
    steps: int = 300,
    lr: float = 5e-2,
    eps: float = 1e-3,
    n_directions: int = 4,
    seed: int = 20260815,
    streamed: bool = True,
) -> ZeroForgeReport:
    """Forward-only LoRA training; preserve caller trainability flags.

    A failed/cancelled step is restored exactly (earlier completed steps remain).
    Invalid settings or nonfinite losses/updates raise SiltStreamError, never a
    successful nonfinite report. Zero steps and a zero learning rate are valid.
    """
    # Validate before either a forward or a parameter mutation. In particular,
    # bool is not an optimization count/scalar despite being an int in Python.
    for name, value, minimum in (("steps", steps, 0), ("n_directions", n_directions, 1)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise SiltStreamError(f"ZeroForge {name} must be an integer >= {minimum}")
    for name, value in (("lr", lr), ("eps", eps)):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise SiltStreamError(f"ZeroForge {name} must be a finite real number")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or value < 0 or (name == "eps" and value == 0):
            raise SiltStreamError(f"ZeroForge {name} must be finite and "
                                  + ("> 0" if name == "eps" else ">= 0"))
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise SiltStreamError("ZeroForge seed must be an integer")
    last_seed = seed + max(0, steps - 1) * 65537 + (n_directions - 1 if steps else 0)
    if seed < -(2**63) or seed > 2**64 - 1 or last_seed > 2**64 - 1:
        raise SiltStreamError("ZeroForge seed sequence exceeds torch's supported integer range")
    if not batches:
        raise SiltStreamError("ZeroForge requires at least one batch")

    params = list(model.trainable_parameters())
    # no_grad is sufficient: changing requires_grad permanently made a second
    # StreamedCausalLM run select no adapters. HF targets may return an explicit
    # list with mixed flags; preserve those too, rather than blanket-enabling.
    def finite_tensors(tensors, label):
        if any(not torch.isfinite(t).all().item() for t in tensors):
            raise SiltStreamError(f"ZeroForge {label} must be finite")

    def finite_loss(batch):
        value = model.loss(batch, streamed=streamed).item()
        if not math.isfinite(value):
            raise SiltStreamError("ZeroForge loss must be finite")
        return value

    finite_tensors(params, "starting parameters")
    # torch's in-place alpha must be representable in the destination dtype.
    alpha_limit = min((torch.finfo(p.dtype).max for p in params), default=float("inf"))
    if lr > alpha_limit or eps > alpha_limit:
        raise SiltStreamError("ZeroForge lr/eps exceeds parameter dtype range")

    curve: List[float] = []
    forwards = 0
    t0 = time.time()

    with torch.no_grad():
        curve.append(finite_loss(batches[0]))
        forwards += 1

        for step in range(steps):
            batch = batches[step % len(batches)]
            snapshot = [p.detach().clone() for p in params]
            update = [torch.zeros_like(p) for p in params]
            try:
                for d in range(n_directions):
                    dir_seed = seed + step * 65537 + d
                    zs = _directions(params, dir_seed)

                    for p, z in zip(params, zs):
                        p.add_(z, alpha=eps)
                    finite_tensors(params, "positive perturbation")
                    loss_plus = finite_loss(batch)

                    for p, s, z in zip(params, snapshot, zs):
                        p.copy_(s)
                        p.add_(z, alpha=-eps)
                    finite_tensors(params, "negative perturbation")
                    loss_minus = finite_loss(batch)

                    for p, s in zip(params, snapshot):
                        p.copy_(s)  # exact restore, by construction
                    forwards += 2

                    grad_scalar = (loss_plus - loss_minus) / (2.0 * eps)
                    if not math.isfinite(grad_scalar) or abs(grad_scalar / n_directions) > alpha_limit:
                        raise SiltStreamError("ZeroForge gradient estimate exceeds finite dtype range")
                    for u, z in zip(update, zs):
                        u.add_(z, alpha=grad_scalar / n_directions)
                    finite_tensors(update, "update")

                for p, u in zip(params, update):
                    p.add_(u, alpha=-lr)
                finite_tensors(params, "updated parameters")
                curve.append(finite_loss(batch))
                forwards += 1
            except BaseException:
                # Include KeyboardInterrupt/SystemExit cancellation, failed +/-
                # forwards, arithmetic, and the post-update loss in this step's
                # transaction. Re-raise the original exception, never swallow it.
                for p, s in zip(params, snapshot):
                    p.copy_(s)
                raise

    seconds = time.time() - t0
    return ZeroForgeReport(
        steps=steps,
        initial_loss=curve[0],
        final_loss=curve[-1],
        loss_curve=curve,
        seconds=seconds,
        forward_passes=forwards,
        backward_passes=0,
        eps=eps,
        lr=lr,
        n_directions=n_directions,
        seed=seed,
        config_fingerprint=model.fingerprint,
        audit=model.audit_metadata(),
    )
