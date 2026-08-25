"""SiltSpring -- one model, multiple verified compression states.

The spring metaphor: a model rests compressed (int2/int4 -- small enough for
weak hardware) and EXPANDS to higher-precision states when memory allows.
What makes this a trust component rather than a serving trick:

  1. Every state carries a CAPABILITY CERTIFICATE: each skill suite is
     evaluated at that state and compared against the full-precision
     reference; a state that degrades a suite beyond tolerance has that
     skill REVOKED at that state.
  2. The runtime picks the best state that fits the memory budget AND is
     certified for the skills the caller requires -- and REFUSES honestly
     (typed error) when no state satisfies both, rather than silently
     serving degraded answers.
  3. Certificates are BOUND to the exact skill (LoRA) parameters they were
     issued for. If the model learns anything new, old certificates become
     STALE and selection refuses until re-certification. A certificate for
     yesterday's model is a lie about today's.
  4. Skills (LoRA adapters) stay full-precision and ride on top of every
     state -- train once, carried always.

Memory model (honest): quantized states are stored as int8 containers and
dequantized ONE LAYER AT A TIME during the forward pass, so a state's
resident cost is its container bytes plus one dequantized layer. int4/int2
share the int8 container size in RAM; their tighter `packed` size applies to
on-disk storage. `release_states()` drops all states except the chosen one
-- deployment memory equals the active state, not the sum of all states.

Prior work honestly noted: multi-precision / nested quantized storage exists
in research (e.g. Matryoshka Quantization, Any-Precision LLM, 2024-25) and
static per-file quant levels are standard in the GGUF ecosystem. What is
ours here: per-state gate-style capability certification, staleness-bound
certificates, certified honest refusal, and the elastic pick conditioned on
BOTH resources and certified skills. (Skill-aware per-layer bit allocation
is a planned v2.)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from .config import ModelConfig, StreamConfig
from .errors import SiltStreamError
from .functional import block_forward
from .model import StreamedCausalLM
from .quant import dequantize_state, packed_bytes, quantize_state, state_bytes


class BudgetError(SiltStreamError):
    """No spring state fits the given memory budget."""


class StateNotCertifiedError(SiltStreamError):
    """The selected state is not certified for a required skill."""


class StaleCertificateError(SiltStreamError):
    """The model's skills changed after certification; re-certify first."""


FULL = "full"
DEFAULT_LEVELS: Sequence[str] = (FULL, "int8", "int4", "int2")


@dataclass
class StateCertificate:
    state: str
    bytes_actual: int
    bytes_packed: int
    reference_loss: Dict[str, float]
    state_loss: Dict[str, float]
    relative_degradation: Dict[str, float]
    certified_skills: List[str]
    revoked_skills: List[str]
    tolerance: float
    lora_fingerprint: str

    def covers(self, skills: Sequence[str]) -> bool:
        return all(s in self.certified_skills for s in skills)


class SpringModel:
    """Wraps a StreamedCausalLM base with multiple quantized spring states."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        stream_cfg: Optional[StreamConfig] = None,
        levels: Sequence[str] = DEFAULT_LEVELS,
    ) -> None:
        if FULL not in levels:
            raise ValueError("levels must include 'full' (the reference state)")
        self.base = StreamedCausalLM(model_cfg, stream_cfg or StreamConfig())
        self.model_cfg = model_cfg
        self.levels = list(levels)
        device = self.base.device_

        full_states = [
            self.base.bank.fetch(i, device)  # fetch returns isolated clones
            for i in range(model_cfg.n_layers)
        ]
        # FULL is stored as float32 states; compressed levels are stored as
        # QUANTIZED containers and dequantized per layer at forward time.
        self._full_states: List[Dict[str, torch.Tensor]] = full_states
        self._qstates: Dict[str, List[Dict[str, object]]] = {}
        full_bytes = sum(
            sum(v.numel() * v.element_size() for v in s.values()) for s in full_states
        )
        self._bytes: Dict[str, int] = {FULL: full_bytes}
        self._bytes_packed: Dict[str, int] = {FULL: full_bytes}
        for level in self.levels:
            if level == FULL:
                continue
            qs = [quantize_state(s, level) for s in full_states]
            self._qstates[level] = qs
            self._bytes[level] = sum(state_bytes(q) for q in qs)
            self._bytes_packed[level] = sum(packed_bytes(q) for q in qs)

        self.certificates: Dict[str, StateCertificate] = {}
        self._certified_lora_fp: Optional[str] = None
        self.active_state: str = FULL

    # -- skill (LoRA) fingerprint -------------------------------------------------

    def lora_fingerprint(self) -> str:
        """Stable hash over the CURRENT skill parameters. Certificates are
        only valid for the exact parameters they were issued against."""
        h = hashlib.sha256()
        for name, param in sorted(self.base.lora.named_parameters()):
            h.update(name.encode())
            h.update(param.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()[:16]

    # -- forward at a given state ------------------------------------------------

    def _layer_weights(self, state: str, li: int) -> Dict[str, torch.Tensor]:
        if state == FULL:
            return self._full_states[li]
        return dequantize_state(self._qstates[state][li])  # one layer at a time

    def forward(self, input_ids: torch.Tensor, state: Optional[str] = None) -> torch.Tensor:
        state = state or self.active_state
        available = ([FULL] if self._full_states else []) + sorted(self._qstates)
        if (state == FULL and not self._full_states) or (
            state != FULL and state not in self._qstates
        ):
            raise SiltStreamError(
                f"unknown or released spring state {state!r}; available: {available}"
            )
        b = self.base
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = b.tok_emb(input_ids) + b.pos_emb(pos)[None, :, :]
        for li in range(self.model_cfg.n_layers):
            x = block_forward(
                x,
                self._layer_weights(state, li),
                b._adapters_for(li),  # skills ride on top of EVERY state
                self.model_cfg.n_heads,
                b.scaling,
            )
        x = b.ln_f(x)
        return torch.nn.functional.linear(x, b.lm_head.weight)

    def loss(self, input_ids: torch.Tensor, state: Optional[str] = None) -> torch.Tensor:
        logits = self.forward(input_ids, state=state)
        return torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, self.model_cfg.vocab_size),
            input_ids[:, 1:].reshape(-1),
        )

    # -- certification -------------------------------------------------------------

    def certify(
        self, suites: Dict[str, torch.Tensor], tolerance: float = 0.02
    ) -> Dict[str, StateCertificate]:
        """Evaluate every suite at every state vs the full-precision reference.

        A skill is certified at a state iff its loss degrades by at most
        `tolerance` (relative). Certificates are bound to the current LoRA
        fingerprint; any later skill change makes them stale.
        """
        fp = self.lora_fingerprint()
        with torch.no_grad():
            reference = {
                name: float(self.loss(batch, state=FULL).item())
                for name, batch in suites.items()
            }
            for level in self.levels:
                if level != FULL and level not in self._qstates:
                    continue  # released
                state_loss: Dict[str, float] = {}
                rel: Dict[str, float] = {}
                certified: List[str] = []
                revoked: List[str] = []
                for name, batch in suites.items():
                    ls = float(self.loss(batch, state=level).item())
                    state_loss[name] = ls
                    ref = reference[name]
                    degradation = (ls - ref) / max(abs(ref), 1e-12)
                    rel[name] = degradation
                    (certified if degradation <= tolerance else revoked).append(name)
                self.certificates[level] = StateCertificate(
                    state=level,
                    bytes_actual=self._bytes[level],
                    bytes_packed=self._bytes_packed[level],
                    reference_loss=dict(reference),
                    state_loss=state_loss,
                    relative_degradation=rel,
                    certified_skills=certified,
                    revoked_skills=revoked,
                    tolerance=tolerance,
                    lora_fingerprint=fp,
                )
        self._certified_lora_fp = fp
        return self.certificates

    # -- elastic selection ----------------------------------------------------------

    def choose_state(
        self, budget_bytes: int, required_skills: Sequence[str] = ()
    ) -> str:
        """Highest-quality state that fits the budget AND certifies the skills.

        Quality order = the order of self.levels (full first). Honest refusal:
        StaleCertificateError if skills changed since certification;
        BudgetError if nothing fits; StateNotCertifiedError if states fit but
        none is certified for the required skills.
        """
        if not self.certificates:
            raise SiltStreamError("call certify() before choose_state()")
        current_fp = self.lora_fingerprint()
        if current_fp != self._certified_lora_fp:
            raise StaleCertificateError(
                "skill parameters changed since certification "
                f"(certified {self._certified_lora_fp}, current {current_fp}); "
                "re-run certify() -- certificates for a previous model state "
                "must never authorize the current one"
            )
        known: List[str] = sorted(
            set().union(*[set(c.state_loss) for c in self.certificates.values()])
        )
        unknown = [s for s in required_skills if s not in known]
        if unknown:
            raise StateNotCertifiedError(
                f"skills never certified: {unknown}; known skills: {known}. "
                "Add the suite to certify() before requiring it."
            )
        available = [lv for lv in self.levels if lv == FULL or lv in self._qstates]
        fitting = [lv for lv in available if self._bytes_packed[lv] <= budget_bytes]
        if not fitting:
            smallest = min(self._bytes_packed[lv] for lv in available)
            raise BudgetError(
                f"no spring state fits budget {budget_bytes} bytes "
                f"(smallest available state needs {smallest})"
            )
        for lv in fitting:  # levels are ordered best-first
            if self.certificates[lv].covers(required_skills):
                self.active_state = lv
                return lv
        raise StateNotCertifiedError(
            f"states {fitting} fit the budget, but none is certified for "
            f"skills {list(required_skills)}; refusing to serve degraded output"
        )

    # -- deployment ------------------------------------------------------------------

    def release_states(self, keep: str) -> Dict[str, int]:
        """Drop every compressed state except `keep` (deployment memory =
        the active state only). FULL reference states are also dropped unless
        keep == FULL. Returns the resident bytes remaining per state."""
        if keep != FULL and keep not in self._qstates:
            raise SiltStreamError(f"cannot keep unknown state {keep!r}")
        for lv in list(self._qstates):
            if lv != keep:
                del self._qstates[lv]
        if keep != FULL:
            self._full_states = []
        self.active_state = keep
        remaining = {}
        if self._full_states:
            remaining[FULL] = self._bytes[FULL]
        for lv in self._qstates:
            remaining[lv] = self._bytes[lv]
        return remaining

    def audit_metadata(self) -> Dict[str, object]:
        meta = self.base.audit_metadata()
        meta.update(
            {
                "component": "siltspring",
                "levels": self.levels,
                "active_state": self.active_state,
                "state_bytes_container": dict(self._bytes),
                "state_bytes_packed": dict(self._bytes_packed),
                "certified_lora_fingerprint": self._certified_lora_fp,
                "current_lora_fingerprint": self.lora_fingerprint(),
                "certified": {
                    lv: c.certified_skills for lv, c in self.certificates.items()
                },
            }
        )
        return meta
