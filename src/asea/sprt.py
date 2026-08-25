"""SPRT early-stop (B2, audit 2026-08-17).

A Sequential Probability Ratio Test over the held-out case-regression stream,
with a DELIBERATELY ASYMMETRIC stop rule: the test may stop EARLY to REJECT a
clearly-failing packet (saving real-GPU evaluation time), but it may NEVER stop
early to PROMOTE one. Promotion always requires the full held-out evaluation
plus the gate; the SPRT can only ever short-circuit the negative.

Why asymmetric. A symmetric SPRT (stop for either hypothesis) would let a
packet be PROMOTED on a PARTIAL evaluation -- a statistical "looks good so far"
replacing the measured full-evaluation gate. That is exactly the silent
premature-promotion the gate exists to prevent. So the upper (promote) boundary
is computed and REPORTED (a reader can see the LLR was heading there) but is
NEVER a stop trigger. Only the lower (reject) boundary stops. This is the
patentable-novel shape: SPRT-as-a-cost-saver on the reject side only, with the
promote side deliberately left to the full gate.

The model. The per-case regression indicator ``Xi`` (1 if the candidate scored
below the baseline on case ``i``, else 0) is modelled as Bernoulli under::

    H0: the packet is unacceptable -- P(regress) = p0  (default 0.5, random-or-worse)
    H1: the packet is acceptable   -- P(regress) = p1  (default 0.1, few regressions)

After each case the log-likelihood ratio is updated::

    LLR = sum [ Xi * log(p1/p0) + (1 - Xi) * log((1-p1)/(1-p0)) ]

and compared to Wald's boundaries::

    accept H0 (REJECT packet)     when LLR <= log B,  B = beta / (1 - alpha)
    accept H1 (promote eligible)  when LLR >= log A,  A = (1 - beta) / alpha   <- NEVER a stop

with error rates ``alpha`` (false-promote) and ``beta`` (false-reject), default
0.05 each -- i.e. the early-reject carries >=95% confidence (the false-reject
rate, the chance of early-rejecting an actually-acceptable packet, is bounded
by ``beta``).

HONESTY CONTRACT (binding):

  * :meth:`SPRT.should_stop` returns True ONLY on a REJECT verdict. A
    PROMOTE_ELIGIBLE verdict is reported but NEVER stops -- the asymmetry is
    enforced in THIS API, not left to the caller. A caller that simply loops
    ``while not sprt.should_stop()`` cannot accidentally early-promote.
  * An early stop record (config, case index, regressions, LLR, verdict,
    boundaries) is produced for the audit so a reader can tell a statistical
    early-stop from a full sweep -- the gate surfaces it as a distinct hard
    check (``no_statistical_early_reject``), never as a silent reject.
  * The test is a PURE function of the regression stream + config: no model
    state leaks in, no randomness. Deterministic and reproducible.
  * With SPRT disabled (the evaluator default), evaluation is byte-identical to
    pre-B2 -- the SPRT is opt-in via ``Evaluator(sprt=SprtConfig(...))``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .core.errors import SprtConfigError


#: Verdicts the test can return after a case. Only REJECT is a stop trigger.
REJECT = "reject"
PROMOTE_ELIGIBLE = "promote_eligible"
CONTINUE = "continue"


@dataclass(frozen=True)
class SprtConfig:
    """Configuration for the asymmetric SPRT.

    ``p0`` is the regression probability under H0 (unacceptable packet) and
    MUST exceed ``p1`` (regression probability under H1, acceptable packet):
    if it did not, the test would be inverted and would early-reject GOOD
    packets -- the worst silent failure for a gate. Validated at construction.
    """

    p0: float = 0.5
    p1: float = 0.1
    alpha: float = 0.05
    beta: float = 0.05

    def __post_init__(self) -> None:
        if not (0.0 < self.p1 < self.p0 < 1.0):
            raise SprtConfigError(
                "require 0 < p1 < p0 < 1; got p0={}, p1={} (p0 must exceed p1 or "
                "the test is inverted and early-rejects GOOD packets)".format(
                    self.p0, self.p1
                )
            )
        if not (0.0 < self.alpha < 1.0) or not (0.0 < self.beta < 1.0):
            raise SprtConfigError(
                "require 0 < alpha < 1 and 0 < beta < 1; got alpha={}, beta={}".format(
                    self.alpha, self.beta
                )
            )

    @property
    def log_A(self) -> float:
        # Upper (promote) boundary -- REPORTED but NEVER a stop trigger.
        return math.log((1.0 - self.beta) / self.alpha)

    @property
    def log_B(self) -> float:
        # Lower (reject) boundary -- the ONLY stop trigger.
        return math.log(self.beta / (1.0 - self.alpha))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p0": self.p0,
            "p1": self.p1,
            "alpha": self.alpha,
            "beta": self.beta,
            "log_A": round(self.log_A, 6),
            "log_B": round(self.log_B, 6),
        }


@dataclass
class SPRT:
    """The asymmetric sequential test.

    Call :meth:`update` once per held-out case (after the candidate's score for
    that case is known and compared to the baseline), then :meth:`should_stop`.
    Stop the evaluation and record :meth:`stop_record` ONLY when
    ``should_stop()`` is True -- which is only ever on a REJECT.
    """

    config: SprtConfig
    _llr: float = 0.0
    _n: int = 0
    _regressions: int = 0
    _verdict: str = CONTINUE
    _stopped_at: Optional[int] = None
    _promote_eligible_seen: bool = False
    #: Per-case LLR trail for the audit (small; held-out splits are small).
    _trail: list = field(default_factory=list)

    def update(self, regressed: bool) -> str:
        """Feed one case's regression indicator; return the current verdict.

        Idempotent after a REJECT: once the test has stopped on a reject,
        further updates are no-ops returning REJECT (the consumer should have
        stopped, but this keeps the test safe if it did not).
        """
        if self._verdict == REJECT:
            return REJECT
        self._n += 1
        if regressed:
            self._regressions += 1
            self._llr += math.log(self.config.p1 / self.config.p0)
        else:
            self._llr += math.log((1.0 - self.config.p1) / (1.0 - self.config.p0))
        self._trail.append(round(self._llr, 6))

        if self._llr <= self.config.log_B:
            self._verdict = REJECT
            self._stopped_at = self._n
        elif self._llr >= self.config.log_A:
            # PROMOTE_ELIGIBLE is REPORTED but is NEVER a stop. The asymmetry:
            # should_stop() stays False here. We note that the LLR crossed the
            # promote boundary (for the audit) but keep evaluating.
            self._verdict = PROMOTE_ELIGIBLE
            self._promote_eligible_seen = True
        else:
            self._verdict = CONTINUE
        return self._verdict

    def should_stop(self) -> bool:
        """True ONLY on a REJECT. The asymmetry is enforced here: a
        PROMOTE_ELIGIBLE verdict returns False, so a caller looping on
        ``while not sprt.should_stop()`` can never early-promote."""
        return self._verdict == REJECT

    def verdict(self) -> str:
        return self._verdict

    def stop_record(self) -> Dict[str, Any]:
        """An audit record for the early stop. Produced only when the test
        stopped on a reject; a reader uses this to tell a statistical
        early-stop from a full held-out sweep."""
        return {
            "config": self.config.to_dict(),
            "cases_evaluated": self._n,
            "regressions": self._regressions,
            "llr": round(self._llr, 6),
            "verdict": self._verdict,
            "stopped_at": self._stopped_at,
            "promote_eligible_seen": self._promote_eligible_seen,
            "llr_trail": list(self._trail),
        }

    @property
    def cases_evaluated(self) -> int:
        return self._n