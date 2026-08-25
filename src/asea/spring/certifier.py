"""SiltSpring compression certifier -- gate discipline over spring states.

The spring metaphor: a model rests compressed (int2/int4 -- small enough for
weak hardware) and EXPANDS to higher precision when memory allows. What makes
this a TRUST component rather than a serving trick, and what this module adds
on top of the vendored :class:`siltstream_vendor.spring.SpringModel`:

  1. **Per-(state, skill) capability certificates.** Each skill's held-out
     suite is evaluated at every quantization state and compared to the
     full-precision reference; a state that degrades a skill beyond tolerance
     has that skill REVOKED at that state. ``serve(state, skill)`` refuses a
     revoked pair with the named :class:`StateNotCertifiedError`.
  2. **Honest elastic refusal.** ``choose_state(budget, required_skills)``
     returns the best state that fits the budget AND is certified for the
     required skills, or raises :class:`BudgetError` /
     :class:`StateNotCertifiedError` -- never silently serves degraded output.
  3. **Staleness-bound certificates.** Certificates are bound to the exact
     LoRA fingerprint they were issued against. Admitting a new skill changes
     the fingerprint, so every prior certificate becomes STALE and selection
     refuses with :class:`StaleCertificateError` until re-certification. A
     certificate for yesterday's model is a lie about today's -- this module
     tests that invariant.
  4. **Audit chain.** ``certify`` / ``choose_state`` / ``serve`` /
     ``admit_skill`` each append an event to the SAME hash-chained
     :class:`AuditLog` as packets and adapters.

Certification suites reuse SILT's held-out split machinery via
:func:`suites_from_benchmark`, which turns a list of :class:`BenchmarkSuite` s
into the ``{skill_name: input_ids}`` dict the underlying certifier consumes.

Honest scope: the toy :class:`SpringModel` path (random-init
:class:`StreamedCausalLM`, no weights downloaded, <1s on CPU) is what the
unit tests exercise -- it is the validated siltstream path. Real HF models use
:func:`siltstream_vendor.hf_real.certify_hf_states` (forward-only, streams one
quantized layer at a time); that path is provided but only exercised by an
opt-in real test (it downloads a model).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..audit.logger import AuditLog


def suites_from_benchmark(
    suites: Sequence[Any],
    tokenizer: Any,
    max_len: int = 96,
    text_field: str = "prompt",
) -> Dict[str, Any]:
    """Turn SILT held-out :class:`BenchmarkSuite` s into spring certification
    suites: ``{suite_id: input_ids}``. This is the reuse of SILT's held-out
    split machinery -- the same held-out cases Gate 2 uses become the suites
    spring states are certified against.

    Each case's ``text_field`` (default ``prompt``; falls back to ``expected``
    then ``input``) is tokenized with ``tokenizer`` and padded to ``max_len``.
    """
    import torch  # lazy

    out: Dict[str, Any] = {}
    for suite in suites:
        texts: List[str] = []
        for case in getattr(suite, "cases", []):
            prompt = getattr(case, text_field, None)
            if prompt is None:
                prompt = getattr(case, "expected", None) or getattr(case, "input", None)
            if prompt is None:
                continue
            texts.append(str(prompt))
        if not texts:
            continue
        enc = tokenizer(
            texts, return_tensors="pt", padding="max_length",
            truncation=True, max_length=max_len,
        )
        out[getattr(suite, "suite_id", str(id(suite)))] = enc["input_ids"]
    return out


class CompressionCertifier:
    """Gate-disciplined wrapper over a vendored :class:`SpringModel`.

    Construct with a :class:`SpringModel` (toy contract) and an optional
    :class:`AuditLog`. The certifier records the certified LoRA fingerprint and
    refuses to serve/choose when it has gone stale (a new skill was admitted).
    """

    def __init__(
        self,
        spring: Any,
        audit: Optional[AuditLog] = None,
        actor: str = "spring",
    ) -> None:
        self.spring = spring
        self.audit = audit
        self.actor = actor
        # The fingerprint certificates were issued against (None = not yet).
        self._certified_fp: Optional[str] = None

    # --------------------------------------------------------------- helpers

    def _current_fp(self) -> str:
        return self.spring.lora_fingerprint()

    def is_stale(self) -> bool:
        """True if the model's skills changed since the last ``certify``."""
        if self._certified_fp is None:
            return True
        return self._current_fp() != self._certified_fp

    def _audit(self, event: str, detail: Dict[str, Any], session_id: str = "") -> None:
        if self.audit is not None:
            self.audit.append(event, actor=self.actor, session_id=session_id, detail=detail)

    # --------------------------------------------------------------- certify

    def certify(
        self,
        suites: Dict[str, Any],
        tolerance: float = 0.02,
        session_id: str = "",
    ) -> Dict[str, Any]:
        """Certify every (state, skill) pair against the full-precision
        reference. ``suites`` maps skill name -> an ``input_ids`` batch (use
        :func:`suites_from_benchmark` to build it from SILT held-out suites).

        Writes a ``spring_certified`` audit event with the per-state certified
        / revoked skill sets and the LoRA fingerprint the certificates bind to.
        Returns the vendor's ``{state: StateCertificate}`` dict.
        """
        certs = self.spring.certify(suites, tolerance=tolerance)
        self._certified_fp = self._current_fp()
        summary = {
            lv: {
                "certified": list(c.certified_skills),
                "revoked": list(c.revoked_skills),
                "bytes_packed": int(c.bytes_packed),
            }
            for lv, c in certs.items()
        }
        self._audit(
            "spring_certified",
            detail={
                "tolerance": float(tolerance),
                "lora_fingerprint": self._certified_fp,
                "skills": sorted(suites),
                "certificates": summary,
            },
            session_id=session_id,
        )
        return certs

    # ----------------------------------------------------------- choose_state

    def choose_state(
        self,
        budget_bytes: int,
        required_skills: Sequence[str] = (),
        session_id: str = "",
    ) -> str:
        """Best state that fits the budget AND is certified for the required
        skills. Delegates to :meth:`SpringModel.choose_state`, which raises
        :class:`StaleCertificateError` / :class:`BudgetError` /
        :class:`StateNotCertifiedError` -- this wrapper audits the outcome
        (chosen state, or the named refusal) and re-raises."""
        from ..deepapply.backends.siltstream_vendor.spring import (
            BudgetError,
            StateNotCertifiedError,
            StaleCertificateError,
        )

        try:
            state = self.spring.choose_state(int(budget_bytes), tuple(required_skills))
        except (StaleCertificateError, BudgetError, StateNotCertifiedError) as exc:
            self._audit(
                "spring_state_chosen",
                detail={
                    "budget_bytes": int(budget_bytes),
                    "required_skills": list(required_skills),
                    "refused": type(exc).__name__,
                    "reason": str(exc),
                    "stale": isinstance(exc, StaleCertificateError),
                },
                session_id=session_id,
            )
            raise
        self._audit(
            "spring_state_chosen",
            detail={
                "budget_bytes": int(budget_bytes),
                "required_skills": list(required_skills),
                "chosen_state": state,
            },
            session_id=session_id,
        )
        return state

    # --------------------------------------------------------------- serve

    def serve(
        self,
        state: str,
        skill: str,
        session_id: str = "",
    ) -> str:
        """Serve ``skill`` from ``state``. Refuses with
        :class:`StateNotCertifiedError` if the pair is revoked / never
        certified, or :class:`StaleCertificateError` if certificates have gone
        stale (a new skill was admitted since certification). Returns the
        state on success and audits the decision."""
        from ..deepapply.backends.siltstream_vendor.spring import (  # noqa: E402
            StateNotCertifiedError,
            StaleCertificateError,
        )

        if self.is_stale():
            err = StaleCertificateError(
                "spring certificates are STALE (a skill was admitted since "
                "certification; certified fp={}, current fp={}); re-run "
                "certify() before serving -- a certificate for a previous "
                "model state must never authorize the current one".format(
                    self._certified_fp, self._current_fp()
                )
            )
            self._audit(
                "spring_serve",
                detail={"state": state, "skill": skill, "refused": "StaleCertificateError",
                        "reason": str(err)},
                session_id=session_id,
            )
            raise err
        cert = self.spring.certificates.get(state)
        if cert is None:
            err = StateNotCertifiedError(
                "spring state {!r} has no certificate; call certify() first".format(state)
            )
            self._audit("spring_serve", detail={"state": state, "skill": skill,
                        "refused": "StateNotCertifiedError", "reason": str(err)},
                        session_id=session_id)
            raise err
        if skill not in cert.certified_skills:
            err = StateNotCertifiedError(
                "spring state {!r} is NOT certified for skill {!r} (revoked or "
                "never certified); certified={}, revoked={}. Refusing to serve "
                "degraded output.".format(state, skill, cert.certified_skills, cert.revoked_skills)
            )
            self._audit("spring_serve", detail={"state": state, "skill": skill,
                        "refused": "StateNotCertifiedError", "reason": str(err)},
                        session_id=session_id)
            raise err
        self._audit(
            "spring_serve",
            detail={"state": state, "skill": skill, "served": True},
            session_id=session_id,
        )
        return state

    # ----------------------------------------------------------- admit_skill

    def admit_skill(
        self,
        name: str,
        perturbation: float = 1e-4,
        session_id: str = "",
    ) -> None:
        """Simulate admitting a new gated skill (a new adapter that changes the
        LoRA parameters). This perturbs one LoRA parameter so the
        ``lora_fingerprint`` changes, which makes every prior certificate STALE.
        The next ``choose_state`` / ``serve`` refuses with
        :class:`StaleCertificateError` until ``certify`` is re-run with the new
        skill's suite included.

        In a wired SILT deployment this is triggered by the adapter store
        admitting a new PROMOTED adapter (which changes the model's LoRA);
        here it is modelled by a small perturbation so the staleness invariant
        is unit-testable without a full deep-apply run.
        """
        import torch  # lazy

        params = list(self.spring.base.lora.parameters())
        if not params:
            # No LoRA yet: simulate by touching the base embeddings instead,
            # which still changes the model state (certificates bind to the
            # LoRA fingerprint, so this only goes stale if LoRA exists). If
            # there is genuinely no LoRA, staleness cannot occur -- record it.
            self._audit(
                "spring_skill_admitted",
                detail={"skill": name, "note": "no LoRA params; staleness N/A"},
                session_id=session_id,
            )
            return
        with torch.no_grad():
            params[0].data.add_(float(perturbation))
        self._audit(
            "spring_skill_admitted",
            detail={
                "skill": name,
                "lora_fingerprint_before": self._certified_fp,
                "lora_fingerprint_after": self._current_fp(),
                "stale": True,
            },
            session_id=session_id,
        )