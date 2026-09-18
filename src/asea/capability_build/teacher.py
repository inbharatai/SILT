"""Teacher connectors for capability-build.

Two evidence classes, structurally separated (brief §E, §I):

  * ``behavioural_remote`` -- an API/cloud teacher such as Ollama cloud's
    ``glm-5.3-flash:cloud``. It legitimately exposes ONLY prompts, responses,
    tool calls, latencies and token counts. It never reveals routers, experts
    or internal state, and this connector never pretends otherwise.
  * ``internal_open_weight`` -- locally controlled open weights, instrumented
    through the isolated GLM worker (Phase 3). Only that path may produce
    :class:`asea.capability_build.schema.InternalRecord` observations.

Consent discipline (binding): a remote connector runs ONLY when the operator
explicitly selected it for THIS run -- ``allow_remote=True`` here corresponds
to ``--allow-remote`` on the CLI or ``remote_connector_selected: true`` in
the spec. Without consent the constructor raises
:class:`RemoteConsentRequired` BEFORE any network activity. No silent remote
fallback exists in this package: if the operator asked for the internal path
and it is blocked, the run reports ``BLOCKED_RESOURCE`` instead of quietly
calling a cloud API.

Transport follows :class:`asea.modules.real.ollama.OllamaConnector`:
deterministic options (temperature 0, fixed seed), stdlib ``urllib`` only --
no ML dependencies, so this module is safe for the bare ``import asea``
sanity gate.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .errors import RemoteConsentRequired
from .schema import (
    TRACE_CLASS_BEHAVIOURAL,
    BehaviouralRecord,
    CapabilitySpec,
    TraceOutcome,
)

#: A behavioural trace is bounded: the record schema caps prompt/response
#: length; the connector also refuses pathological repetition counts.
_MAX_ATTEMPTS_PER_CASE = 8


class BehaviouralOllamaTeacher:
    """Cloud/API teacher over a (possibly local) Ollama daemon.

    ``remote`` must be True if and only if the daemon reaches a cloud model
    (``glm-5.3-flash:cloud`` resolves through ollama.com). The operator sets
    it per run via ``--allow-remote``; default False means the constructor
    refuses before touching the network.
    """

    is_mock = False
    trace_class = TRACE_CLASS_BEHAVIOURAL

    def __init__(
        self,
        spec: CapabilitySpec,
        model: str,
        *,
        allow_remote: bool = False,
        host: str = "http://localhost:11434",
        seed: int = 0,
        max_new_tokens: int = 1024,
        timeout: int = 300,
        think: Optional[bool] = False,
    ) -> None:
        if not allow_remote and not spec.remote_connector_selected:
            raise RemoteConsentRequired(
                "remote teacher connector requested without per-run consent "
                "(model %r). Pass --allow-remote or set "
                "remote_connector_selected: true in the spec for THIS run; "
                "consent never carries over between runs." % model
            )
        if spec.teacher.access != TRACE_CLASS_BEHAVIOURAL:
            # The spec pinned the internal evidence class; a behavioural
            # connector must never silently stand in for it (rule: no silent
            # fallback between evidence classes).
            raise RemoteConsentRequired(
                "spec pins teacher access %r but a behavioural connector was "
                "constructed; use the isolated worker path instead"
                % spec.teacher.access
            )
        self.spec = spec
        self.model = model
        self.host = host.rstrip("/")
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.think = think

    # -- transport -----------------------------------------------------------

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(
            "{}{}".format(self.host, path),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def health(self) -> Dict[str, Any]:
        """Check the daemon is up and the exact model tag is present.

        Exact-match only (a ``startswith`` match made the historical
        connector report a false-positive model_present).
        """
        request = urllib.request.Request("{}/api/tags".format(self.host))
        with urllib.request.urlopen(request, timeout=10) as response:
            tags = json.loads(response.read().decode("utf-8"))
        available = [m.get("name") for m in tags.get("models", [])]
        present = self.model in available
        return {
            "host": self.host,
            "model": self.model,
            "model_present": present,
            "hint": None if present else "run: ollama pull %s" % self.model,
        }

    # -- inference -----------------------------------------------------------

    def chat(self, prompt: str, system: Optional[str] = None) -> Dict[str, Any]:
        """One deterministic chat turn. Returns the raw Ollama response dict
        (message content, thinking, token counts, timings) -- no cleaning, so
        the trace records what the teacher actually said."""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0,
                "top_p": 1,
                "seed": self.seed,
                "num_predict": self.max_new_tokens,
            },
        }
        if self.think is not None:
            payload["think"] = self.think
        return self._post("/api/chat", payload)

    # -- trace collection ---------------------------------------------------

    def behavioural_record(
        self,
        prompt: str,
        response: Dict[str, Any],
        *,
        attempts: int = 1,
        corrections: int = 0,
        tool_calls: Optional[List[str]] = None,
    ) -> BehaviouralRecord:
        """Collapse one chat turn into a schema-valid behavioural record.

        Latency and token counts come from the daemon's own accounting
        (``eval_count``, ``total_duration``); they are observations, never
        estimates. Missing fields stay ``None`` and are materialised as
        ``NOT_MEASURED`` at receipt time.
        """
        if attempts < 1 or attempts > _MAX_ATTEMPTS_PER_CASE:
            raise ValueError(
                "attempts must be in 1..%d" % _MAX_ATTEMPTS_PER_CASE
            )
        message = response.get("message") or {}
        content = message.get("content", "") or ""
        thinking = message.get("thinking", "") or ""
        # A reasoning model that put its answer in ``thinking`` must not be
        # recorded as an empty response (phantom-zero failure mode, fixed in
        # the real Ollama connector -- same discipline here).
        if not content and thinking:
            content = thinking
        prompt_tokens = response.get("prompt_eval_count")
        response_tokens = response.get("eval_count")
        total_ns = response.get("total_duration")
        latency_ms = (
            round(total_ns / 1_000_000.0, 3) if isinstance(total_ns, (int, float)) else None
        )
        return BehaviouralRecord(
            prompt=prompt[:65536],
            response=content[:131072],
            tool_calls=list(tool_calls or [])[:256],
            attempts=attempts,
            corrections=corrections,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
        )


def ask_teacher(
    teacher: BehaviouralOllamaTeacher,
    case: Dict[str, Any],
    *,
    build_prompt=None,
) -> Dict[str, Any]:
    """Run one evaluation case against the behavioural teacher.

    ``case`` carries ``sample_id``, ``group`` (target/control), ``prompt``
    and optional ``system``. ``build_prompt(case) -> str`` lets a capability
    module shape the prompt; default passes the case prompt through. Returns
    a dict with the behavioural record and an UNJUDGED outcome -- judging is
    the evaluator's job (host-owned verdicts, never the teacher grading
    itself).
    """
    if teacher.trace_class != TRACE_CLASS_BEHAVIOURAL:
        raise RemoteConsentRequired(
            "ask_teacher accepts a behavioural teacher only; internal "
            "instrumentation goes through the isolated worker"
        )
    prompt = (
        build_prompt(case) if build_prompt is not None else case["prompt"]
    )
    started = time.monotonic()
    response = teacher.chat(prompt, system=case.get("system"))
    wall_ms = round((time.monotonic() - started) * 1000.0, 3)
    record = teacher.behavioural_record(
        prompt,
        response,
        attempts=int(case.get("attempts", 1)),
        corrections=int(case.get("corrections", 0)),
        tool_calls=case.get("tool_calls"),
    )
    outcome = TraceOutcome(
        success=None,
        metrics={
            "wall_latency_ms": wall_ms,
            "judge": "UNJUDGED_teacher_cannot_grade_itself",
        },
    )
    return {
        "sample_id": case["sample_id"],
        "group": case["group"],
        "behavioural": record,
        "outcome": outcome,
    }