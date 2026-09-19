"""Capability-build MECHANISM tests.

Every test here exercises bookkeeping mechanics offline: schema validation,
consent gating, evidence-class separation, store/receipt integrity, the
intervention protocol on a fake adapter, and worker IPC. A mock trace or
fake adapter is a MECHANISM TEST, not a GLM capability result -- no test in
this file says anything about any teacher's behaviour.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from asea.capability_build import (
    BLOCKED_RESOURCE,
    NOT_MEASURED,
    TRACE_CLASS_BEHAVIOURAL,
    TRACE_CLASS_INTERNAL,
)
from asea.capability_build.errors import (
    BlockedResource,
    CapabilityBuildError,
    EvidenceClassError,
    InterventionInvalid,
    RemoteConsentRequired,
)
from asea.capability_build.distillation import LeakageError
from asea.capability_build.dataset import DatasetInvalid
from asea.capability_build.schema import (
    BehaviouralRecord,
    CapabilityBuildReceipt,
    CapabilitySpec,
    InternalRecord,
    InterventionEntry,
    TraceOutcome,
)
from asea.capability_build.spec import load_spec
from asea.capability_build.footprint import build_footprint, compute_enrichment
from asea.capability_build.intervention import (
    InterventionTarget,
    attempt_intervention,
    causal_evidence_only,
    run_intervention,
)
from asea.capability_build.receipt import sign_receipt, verify_receipt
from asea.capability_build.store import CapabilityStore
from asea.capability_build import worker_protocol as proto
from asea.capability_build.trace import (
    make_behavioural_trace,
    make_internal_trace,
    traces_same_class,
)
from asea.capability_build import worker_protocol as protocol_module


SPEC = {
    "schema": "silt.capability_spec.v1",
    "capability_id": "python_repo_debugging_v1",
    "modality": "code",
    "description": "Repair a failing python repository.",
    "teacher": {
        "provider": "ollama",
        "model": "glm-5.3-flash:cloud",
        "revision": "glm-5.3-flash-20260901",
        "access": "behavioural_remote",
    },
    "target_metrics": ["heldout_pass_rate"],
    "controls": ["math_reasoning"],
    "minimum_retention_ratio": 0.8,
    "maximum_control_regression": 0.05,
    "evaluation": {
        "training": "data/capability_v1/training.jsonl",
        "development": "data/capability_v1/development.jsonl",
        "heldout": "data/capability_v1/heldout.jsonl",
        "final": "data/capability_v1/final.jsonl",
    },
}


@pytest.fixture
def spec():
    return CapabilitySpec.model_validate(SPEC)


@pytest.fixture
def workspace(tmp_path):
    return CapabilityStore(tmp_path / "ws").root


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

def test_spec_validates_and_fingerprints(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(SPEC), encoding="utf-8")
    loaded = load_spec(path)
    assert loaded["spec"].capability_id == "python_repo_debugging_v1"
    assert len(loaded["spec_sha256"]) == 64
    assert loaded["spec_fingerprint"] == loaded["spec_sha256"]


def test_spec_rejects_wrong_schema(tmp_path):
    bad = dict(SPEC)
    bad["schema"] = "something.else.v9"
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(CapabilityBuildError):
        load_spec(path)


def test_spec_rejects_bad_capability_id():
    bad = dict(SPEC)
    bad["capability_id"] = "Not-A-Valid-Id"
    with pytest.raises(Exception):
        CapabilitySpec.model_validate(bad)


def test_spec_remote_consent_defaults_off(spec):
    assert spec.remote_connector_selected is False


# ---------------------------------------------------------------------------
# Evidence-class separation (never mixed)
# ---------------------------------------------------------------------------

def _behavioural_trace(sample_id="s1", group="target"):
    return make_behavioural_trace(
        capability_id="cap",
        sample_id=sample_id,
        model_revision="rev",
        prompt="prompt text",
        group=group,
        outcome=TraceOutcome(success=True),
        behavioural=BehaviouralRecord(prompt="prompt text", response="ok"),
    )


def _internal_trace(sample_id="s2", group="control"):
    return make_internal_trace(
        capability_id="cap",
        sample_id=sample_id,
        model_revision="rev",
        prompt="prompt text",
        group=group,
        outcome=TraceOutcome(success=True),
        internal=InternalRecord(layers={"0": {"usage_mass": 1.0}}),
    )


def test_behavioural_trace_refuses_internal_record():
    from asea.capability_build.trace import validate_trace_payload

    with pytest.raises(EvidenceClassError):
        validate_trace_payload({
            "schema": "silt.capability_trace.v1", "schema_version": 1,
            "capability_id": "cap", "sample_id": "s", "model_revision": "r",
            "prompt_hash": "0" * 64, "trace_class": TRACE_CLASS_BEHAVIOURAL,
            "group": "target", "outcome": {"success": True},
            "behavioural": {"prompt": "p", "response": "r"},
            "internal": {"layers": {}},
        })


def test_internal_trace_never_carries_behavioural():
    from asea.capability_build.trace import validate_trace_payload

    with pytest.raises(EvidenceClassError):
        validate_trace_payload({
            "schema": "silt.capability_trace.v1", "schema_version": 1,
            "capability_id": "cap", "sample_id": "s", "model_revision": "r",
            "prompt_hash": "0" * 64, "trace_class": TRACE_CLASS_INTERNAL,
            "group": "target", "outcome": {"success": True},
            "behavioural": {"prompt": "p", "response": "r"},
        })


def test_traces_same_class_mixed_is_none():
    assert traces_same_class([_behavioural_trace(), _internal_trace()]) is None
    assert traces_same_class([]) is None
    assert (
        traces_same_class([_behavioural_trace(), _behavioural_trace("s2")])
        == TRACE_CLASS_BEHAVIOURAL
    )


# ---------------------------------------------------------------------------
# Teacher consent (no silent remote)
# ---------------------------------------------------------------------------

def test_remote_teacher_requires_per_run_consent(spec):
    from asea.capability_build.teacher import BehaviouralOllamaTeacher

    with pytest.raises(RemoteConsentRequired):
        BehaviouralOllamaTeacher(spec, "glm-5.3-flash:cloud")
    # Consent is a constructor argument for THIS run only.
    with pytest.raises(RemoteConsentRequired):
        BehaviouralOllamaTeacher(spec, "glm-5.3-flash:cloud", allow_remote=False)


def test_remote_teacher_refuses_when_spec_pins_internal(spec):
    from asea.capability_build.teacher import BehaviouralOllamaTeacher

    internal_spec = SPEC.copy()
    internal_spec["teacher"] = dict(SPEC["teacher"])
    internal_spec["teacher"]["access"] = TRACE_CLASS_INTERNAL
    parsed = CapabilitySpec.model_validate(internal_spec)
    with pytest.raises(RemoteConsentRequired):
        BehaviouralOllamaTeacher(parsed, "glm-5.3-flash:cloud", allow_remote=True)


def test_ask_teacher_judge_is_unjudged(spec):
    from asea.capability_build.teacher import BehaviouralOllamaTeacher, ask_teacher

    teacher = BehaviouralOllamaTeacher(spec, "glm-5.3-flash:cloud", allow_remote=True)
    record = teacher.behavioural_record(
        "prompt",
        {"message": {"content": "answer"},
         "prompt_eval_count": 3, "eval_count": 2, "total_duration": 1_500_000},
    )
    assert record.response == "answer"
    assert record.prompt_tokens == 3
    assert record.response_tokens == 2
    assert record.latency_ms == 1.5


def test_reasoning_model_empty_content_uses_thinking(spec):
    from asea.capability_build.teacher import BehaviouralOllamaTeacher

    teacher = BehaviouralOllamaTeacher(spec, "glm-5.3-flash:cloud", allow_remote=True)
    record = teacher.behavioural_record(
        "prompt", {"message": {"content": "", "thinking": "the answer is 4"}}
    )
    assert record.response == "the answer is 4"


def test_teacher_transport_live_loopback_health_and_chat(spec):
    """REAL defect fix (GLM-5.3-Flash pilot, 2026-09-18): the teacher's
    transport methods lazily imported ``asea.modules.real.ollama`` with a
    THREE-dot relative import, which escapes the top-level package and
    raises ImportError at CALL time -- every prior teacher test exercised
    only behavioural_record, so nothing executed the import until the first
    live remote run crashed. This MECHANISM test drives the ACTUAL
    health()/chat() transport against a loopback socket server: no network,
    no model, but the lazy imports really execute."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from asea.capability_build.teacher import BehaviouralOllamaTeacher

    class Handler(BaseHTTPRequestHandler):
        def _json(self, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._json({"models": [{"name": "glm-5.3-flash:cloud"}]})

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self._json({
                "message": {"content": "loopback"},
                "prompt_eval_count": 1,
                "eval_count": 1,
                "total_duration": 1_000_000,
            })

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        teacher = BehaviouralOllamaTeacher(
            spec, "glm-5.3-flash:cloud", allow_remote=True,
            host="http://127.0.0.1:%d" % server.server_address[1],
        )
        health = teacher.health()  # ImportError here before the fix
        assert health["model_present"] is True
        assert health["model"] == "glm-5.3-flash:cloud"
        response = teacher.chat("ping")  # and here
        assert response["message"]["content"] == "loopback"
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Footprint: correlation-only enrichment
# ---------------------------------------------------------------------------

def _internal_with_usage(layer_mass, expert_mass, group, sample_id):
    return make_internal_trace(
        capability_id="cap",
        sample_id=sample_id,
        model_revision="rev",
        prompt="p-" + sample_id,
        group=group,
        outcome=TraceOutcome(success=True),
        internal=InternalRecord(
            layers={
                "3": {
                    "usage_mass": layer_mass,
                    "experts": {"7": {"usage_mass": expert_mass}},
                }
            }
        ),
    )


def test_enrichment_scores_target_vs_control_usage():
    traces = [
        _internal_with_usage(10.0, 8.0, "target", "t1"),
        _internal_with_usage(6.0, 4.0, "target", "t2"),
        _internal_with_usage(1.0, 1.0, "control", "c1"),
    ]
    result = compute_enrichment(traces)
    expert = result["components"]["expert:3/7"]
    assert expert["target_usage_mass"] == pytest.approx(12.0)
    assert expert["control_usage_mass"] == pytest.approx(1.0)
    assert expert["enrichment_log_ratio"] > 0
    assert expert["correlation_only"] is True


def test_enrichment_refuses_behavioural_and_empty():
    with pytest.raises(CapabilityBuildError):
        compute_enrichment([])
    with pytest.raises(CapabilityBuildError):
        compute_enrichment([_behavioural_trace()])


def test_footprint_requires_limitations_and_carries_correlation_sentence():
    footprint = build_footprint(
        capability="cap", teacher="t", teacher_revision="r",
        spec_fingerprint="0" * 64, evidence_class=TRACE_CLASS_BEHAVIOURAL,
        target_cases=3, control_cases=2,
    )
    assert any("not causal expert importance" in line for line in footprint.limitations)
    assert footprint.experts == {} and footprint.layers == {}
    with pytest.raises(Exception):
        # schema-level: a footprint without limitations is invalid on its face
        CapabilityBuildReceipt.model_validate({})


# ---------------------------------------------------------------------------
# Intervention protocol (fake adapter: MECHANISM, not a teacher result)
# ---------------------------------------------------------------------------

class FakeAdapter:
    """Deterministic stand-in honouring the intervention contract."""

    def __init__(self, restore_verifies=True, scores=None):
        self.restore_verifies = restore_verifies
        self.scores = scores or {}
        self.masked = None
        self.verified_calls = 0

    def verify_unchanged(self):
        self.verified_calls += 1
        if self.masked is not None:
            return {"unchanged": False, "detail": "masked"}
        if not self.restore_verifies:
            return {"unchanged": False, "detail": "hash mismatch after restore"}
        return {"unchanged": True, "detail": "hash-identical"}

    def temporary_mask(self, target):
        self.masked = target
        return {"key": target.key}

    def restore_mask(self, target):
        self.masked = None
        return {"restored": True}


def _cases(n, verdict=True, group="target"):
    return [
        {"sample_id": "c%d" % i, "group": group,
         "expected_verdict": verdict}
        for i in range(n)
    ]


def _judge(case, _note):
    return case["expected_verdict"]


def test_intervention_mask_measure_restore_verify():
    adapter = FakeAdapter()
    entry = run_intervention(
        adapter,
        InterventionTarget("expert", 3, 7),
        target_cases=_cases(4, verdict=True),
        control_cases=_cases(4, verdict=True),
        judge=_judge,
        seed=42,
    )
    assert entry.restored_and_verified is True
    assert entry.target_drop == pytest.approx(0.0)
    assert entry.control_drop == pytest.approx(0.0)
    assert entry.seed == 42
    assert adapter.masked is None


def test_intervention_detects_functional_drop():
    adapter = FakeAdapter()
    scores = {"base": {"target": 1.0, "control": 1.0},
              "masked": {"target": 0.0, "control": 1.0}}

    def judge(case, note):
        phase = "masked" if adapter.masked is not None else "base"
        return scores[phase][case["group"]] > 0.5

    entry = run_intervention(
        adapter,
        InterventionTarget("expert", 0, 1),
        target_cases=_cases(4),
        control_cases=_cases(4, group="control"),
        judge=judge,
        seed=1,
    )
    assert entry.target_drop == pytest.approx(1.0)
    assert entry.control_drop == pytest.approx(0.0)


def test_intervention_unrestorable_teacher_is_never_causal_evidence():
    adapter = FakeAdapter(restore_verifies=False)
    with pytest.raises(InterventionInvalid):
        run_intervention(
            adapter,
            InterventionTarget("expert", 1, 1),
            target_cases=_cases(2),
            control_cases=_cases(2),
            judge=_judge,
            seed=0,
        )
    outcome = attempt_intervention(
        adapter,
        InterventionTarget("expert", 1, 1),
        target_cases=_cases(2),
        control_cases=_cases(2),
        judge=_judge,
        seed=0,
    )
    assert outcome["ok"] is False
    assert outcome["entry"] is None
    assert outcome["error"]


def test_causal_evidence_only_filters_failed_interventions():
    good = InterventionEntry(
        component="expert:0/1", target_drop=0.5, control_drop=0.0,
        target_cases=4, control_cases=4, restored_and_verified=True, seed=1,
    )
    bad = InterventionEntry(
        component="expert:0/2", target_drop=0.5, control_drop=0.0,
        target_cases=4, control_cases=4, restored_and_verified=False, seed=1,
    )
    assert causal_evidence_only({"a": good, "b": bad}) == {"a": good}


def test_intervention_target_validation():
    with pytest.raises(InterventionInvalid):
        InterventionTarget("neuron", 0)
    with pytest.raises(InterventionInvalid):
        InterventionTarget("expert", -1)
    assert InterventionTarget("layer", 4).key == "layer:4"


# ---------------------------------------------------------------------------
# Store + receipts
# ---------------------------------------------------------------------------

def test_store_round_trip_and_no_overwrite(tmp_path):
    store = CapabilityStore(tmp_path / "ws")
    store.put("specs", "one", {"a": 1})
    with pytest.raises(CapabilityBuildError):
        store.put("specs", "one", {"a": 2})
    with pytest.raises(CapabilityBuildError):
        store.put("specs", "../evil", {"a": 1})
    assert store.get("specs", "one") == {"a": 1}
    tampered = tmp_path / "ws" / "specs" / "one.json"
    payload = json.loads(tampered.read_text(encoding="utf-8"))
    payload["a"] = 99
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CapabilityBuildError):
        store.get("specs", "one")


def _receipt_object(status="completed", error=None):
    return CapabilityBuildReceipt(
        command="trace",
        status=status,
        capability_id="cap",
        # A real-looking digest: the all-zero form is the CLI's no-spec-
        # loaded placeholder and may only ride on BLOCKED_RESOURCE receipts
        # (enforced by the schema since audit 2026-09-18).
        spec_sha256="a" * 64,
        evidence_class=TRACE_CLASS_BEHAVIOURAL,
        error=error,
        limitations=["Implementation success is not model-quality success."],
    )


def test_receipt_sign_verify_round_trip(tmp_path):
    workspace = tmp_path / "ws"
    signed = sign_receipt(workspace, _receipt_object())
    assert signed["signature_alg"] == "hmac-sha256-local"
    assert signed["honesty_note"]
    # NOT_MEASURED is materialised, never estimated
    assert signed["capability_retention"] == NOT_MEASURED
    assert signed["resources"]["ram_peak_bytes"] == NOT_MEASURED
    verified = verify_receipt(workspace, signed)
    assert verified["valid"] is True


def test_receipt_tampering_is_detected(tmp_path):
    workspace = tmp_path / "ws"
    signed = sign_receipt(workspace, _receipt_object())
    signed["capability_id"] = "someone_else"
    with pytest.raises(Exception):
        verify_receipt(workspace, signed)


def test_blocked_receipt_requires_error():
    with pytest.raises(Exception):
        CapabilityBuildReceipt(
            command="trace", status=BLOCKED_RESOURCE, capability_id="cap",
            spec_sha256="0" * 64, evidence_class=TRACE_CLASS_INTERNAL,
            limitations=["l"],
        )


def test_placeholder_spec_hash_only_on_blocked_receipts():
    """Defect fix (audit 2026-09-18): the CLI fills spec_sha256 with 64
    zeros when a run is blocked before any spec could be loaded. That
    placeholder may ride ONLY on a BLOCKED_RESOURCE receipt -- a completed
    receipt carrying it would claim to describe a spec it never read."""
    with pytest.raises(Exception):
        CapabilityBuildReceipt(
            command="trace", status="completed", capability_id="cap",
            spec_sha256="0" * 64, evidence_class=TRACE_CLASS_INTERNAL,
            limitations=["l"],
        )
    with pytest.raises(Exception):
        CapabilityBuildReceipt(
            command="trace", status="rejected", capability_id="cap",
            spec_sha256="0" * 64, evidence_class=TRACE_CLASS_INTERNAL,
            limitations=["l"],
        )
    # the placeholder is legal exactly where it means something: BLOCKED
    CapabilityBuildReceipt(
        command="trace", status=BLOCKED_RESOURCE, capability_id="cap",
        spec_sha256="0" * 64, evidence_class=TRACE_CLASS_INTERNAL,
        error="blocked before the spec could be loaded",
        limitations=["l"],
    )


# ---------------------------------------------------------------------------
# Worker IPC protocol
# ---------------------------------------------------------------------------

def test_protocol_round_trip():
    request = proto.make_request("hello", {"x": 1})
    response = proto.make_response(request, ok=True, result={"a": 2})
    assert proto.decode(proto.encode(request))["op"] == "hello"
    assert proto.decode(proto.encode(response))["result"] == {"a": 2}


def test_protocol_refuses_wrong_protocol_and_oversize():
    with pytest.raises(ValueError):
        proto.decode(b'{"protocol": "other", "protocol_version": 1, "id": "x", "op": "hello"}\n')
    with pytest.raises(ValueError):
        proto.make_request("not_an_op")
    big = {"protocol": proto.PROTOCOL, "protocol_version": 1, "id": "x",
           "op": "hello", "payload": {"k": "v" * (proto.MAX_FRAME_BYTES)}}
    with pytest.raises(ValueError):
        proto.encode(big)


def test_blocked_frame_carries_requirement_and_remedy():
    request = proto.make_request("hello")
    response = proto.blocked_response(request, "need 640 GiB", "use a GPU box")
    assert response["error"]["kind"] == "blocked"
    assert response["error"]["requirement"] == "need 640 GiB"


def test_worker_client_crash_preserves_partial_frames(tmp_path):
    """The client must surface WorkerCrashed (with honest framing) when the
    worker dies mid-conversation; the lost portion is never fabricated.
    MECHANISM test against a script fake, not the GLM worker."""
    fake = tmp_path / "fake_worker.py"
    fake.write_text(
        "import json, sys\n"
        "line = sys.stdin.readline()\n"
        "request = json.loads(line)\n"
        "sys.stdout.write(json.dumps({\n"
        "    'protocol': %r, 'protocol_version': 1,\n"
        "    'op': request['op'], 'id': request['id'], 'ok': True,\n"
        "    'result': {'transformers': 'fake'},\n"
        "}) + '\\n')\n"
        "sys.stdout.flush()\n"
        "sys.stdin.readline()\n"  # second request: die without answering
        "sys.exit(1)\n" % proto.PROTOCOL,
        encoding="utf-8",
    )
    from asea.capability_build.worker import GlmWorkerClient

    client = GlmWorkerClient(
        repo_root=tmp_path, checkpoint=str(tmp_path), use_docker=False,
        python=sys.executable,
    )
    # Fake worker is not at workers/glm53/worker.py; patch the command.
    client._command = lambda: [sys.executable, str(fake)]
    assert client.hello()["transformers"] == "fake"
    with pytest.raises(CapabilityBuildError):
        client.request("inspect")


def test_worker_client_enforces_deadline_kills_and_drains_stderr(tmp_path):
    """Defect fix: ``timeout`` is ENFORCED (a watchdog kills the worker
    tree at the deadline) instead of being accepted and ignored by an
    unbounded blocking read; stderr is drained so the failure carries the
    worker's own diagnostic instead of a thrown-away pipe."""
    from asea.capability_build.errors import WorkerTimeout
    from asea.capability_build.worker import GlmWorkerClient

    fake = tmp_path / "slow_worker.py"
    fake.write_text(
        "import sys, time\n"
        "sys.stderr.write('hanging on purpose\\n')\n"
        "sys.stderr.flush()\n"
        "time.sleep(600)\n",
        encoding="utf-8",
    )
    client = GlmWorkerClient(
        repo_root=tmp_path, checkpoint=str(tmp_path), use_docker=False,
        python=sys.executable, timeout=3600.0,
    )
    client._command = lambda: [sys.executable, str(fake)]
    with pytest.raises(WorkerTimeout) as excinfo:
        client.request("hello", timeout=0.5)
    assert "deadline" in str(excinfo.value)
    # the drained stderr rides the error, not a discarded pipe
    assert "hanging on purpose" in str(excinfo.value)
    # the worker process is dead and reaped, not left hanging
    assert client._process is not None
    assert client._process.wait(timeout=30) != 0
    client.stop()


def test_worker_deadline_covers_request_transmission(tmp_path):
    """Defect fix: the watchdog is armed BEFORE the request is written, not
    after it. A worker that NEVER reads stdin cannot stall the write past
    the deadline: it is killed at the deadline and the failure is reported
    as a timeout (the request was never accepted), not an unbounded hang.
    MECHANISM test against a script fake, not the GLM worker."""
    from asea.capability_build.errors import WorkerTimeout
    from asea.capability_build.worker import GlmWorkerClient

    fake = tmp_path / "never_reads_stdin.py"
    fake.write_text(
        "import time\n"
        "time.sleep(600)\n",
        encoding="utf-8",
    )
    client = GlmWorkerClient(
        repo_root=tmp_path, checkpoint=str(tmp_path), use_docker=False,
        python=sys.executable, timeout=3600.0,
    )
    client._command = lambda: [sys.executable, str(fake)]
    # ~4 MiB: far past the OS pipe buffer, so the write itself blocks
    # while no reader consumes it -- transmission must be inside the
    # protected window, not before it.
    payload = {"blobs": ["x" * (512 * 1024)] * 8}
    with pytest.raises(WorkerTimeout) as excinfo:
        client.request("routing", payload, timeout=0.5)
    message = str(excinfo.value)
    assert "deadline" in message
    assert "stopped reading its stdin" in message
    # the request was never accepted: no complete frames were recorded
    assert client._frames == []
    # the worker process is dead and reaped
    assert client._process is not None
    assert client._process.wait(timeout=30) != 0
    client.stop()


# ---------------------------------------------------------------------------
# Router-output masking (MECHANISM TEST: random fake routers, not GLM or
# Switch capability results)
# ---------------------------------------------------------------------------

def _fake_glm_router_stack(torch):
    """A minimal EXACT-SHAPE GLM-5.3-Flash stand-in for detect_architecture:
    45 decoder blocks (first 3 dense, 42 sparse with a router each), a vision
    tower module, and the config fields the adapter pins. The router is a
    FAITHFUL REPLICA of transformers 5.16.1 ``Glm5NextTextTopkRouter``
    (Apache-2.0): sigmoid scores, ``e_score_correction_bias`` added to the
    selection scores only, group top-k (top-2 within each of 4 groups, top-2
    groups kept), weights gathered from the PRE-bias scores, renorm,
    routed scaling -- returning the full ``(router_logits, topk_weights,
    topk_indices)`` contract the adapter's telemetry and mask hooks read.
    One router instance is SHARED by all 42 sparse blocks (the fake is a
    mechanics fixture; per-layer weights are not under test) so the stack
    stays small and hashing stays fast. Random weights -- mechanics only."""
    from asea.capability_build.adapters.glm53_flash import Glm53FlashAdapter

    class FakeGlm5NextTextTopkRouter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(5)
            self.weight = torch.nn.Parameter(
                torch.randn(288, 4096) * 0.02
            )
            self.e_score_correction_bias = torch.zeros(288)
            self.top_k = 8
            self.num_experts = 288
            self.num_group = 4
            self.topk_group = 2
            self.norm_topk_prob = True
            self.routed_scaling_factor = 2.5

        def forward(self, hidden):
            functional = torch.nn.functional
            router_logits = functional.linear(hidden.float(), self.weight.float())
            scores = torch.sigmoid(router_logits)
            scores_for_choice = scores + self.e_score_correction_bias.to(scores.dtype)
            group_scores = (
                scores_for_choice.view(-1, self.num_group, self.num_experts // self.num_group)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(-1, self.num_group, self.num_experts // self.num_group)
                .reshape(-1, self.num_experts)
            )
            scores_for_choice = scores_for_choice.masked_fill(
                ~score_mask.bool(), float("-inf")
            )
            topk_indices = torch.topk(
                scores_for_choice, k=self.top_k, dim=-1, sorted=False
            )[1]
            topk_weights = scores.gather(1, topk_indices)
            if self.norm_topk_prob:
                topk_weights = topk_weights / (
                    topk_weights.sum(dim=-1, keepdim=True) + 1e-20
                )
            topk_weights = topk_weights * self.routed_scaling_factor
            return router_logits, topk_weights, topk_indices

    class Glm5NextForCausalLM(torch.nn.Module):  # accepted class name
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList()
            router = FakeGlm5NextTextTopkRouter()
            for index in range(45):
                block = torch.nn.Module()
                if index < 3:
                    block.mlp = torch.nn.Linear(4096, 4)  # dense: no "experts"
                else:
                    block.mlp = torch.nn.Module()
                    block.mlp.gate = router  # the REAL module name
                    block.mlp.experts = torch.nn.Module()
                self.model.layers.append(block)
            self.vision_tower = torch.nn.Linear(4096, 4)

    class Cfg:
        pass

    cfg = Cfg()
    cfg.model_type = "glm5_next"
    cfg.num_hidden_layers = 45
    cfg.n_routed_experts = 288
    cfg.n_shared_experts = 1
    cfg.num_experts_per_tok = 8
    cfg.first_k_dense_replace = 3
    cfg.hidden_size = 4096
    cfg.max_position_embeddings = 1_048_576
    cfg.torch_dtype = "bfloat16"
    return Glm53FlashAdapter(Glm5NextForCausalLM(), torch, config=cfg)


def test_glm_mask_suppresses_expert_and_restore_succeeds():
    """Defect fix (external audit 2026-09-19): the mask must rewrite the
    router's FULL output tuple. The v5.16.1 router returns
    ``(router_logits, topk_weights, topk_indices)`` -- the dispatch lives in
    the tuple, so the old hook that rewrote only ``output[0]`` (the logits)
    changed NOTHING the MoE consumes and masked interventions silently
    measured an UNMASKED teacher. The hook now also recomputes the
    selection under the exact routing equation with the masked expert
    forced to ``-inf`` BEFORE the group stage (a nonzero
    ``e_score_correction_bias`` added after a ``-1e9`` logit could
    otherwise push the masked expert back into the top-8). No weights are
    touched, ever: the teacher verifies hash-unchanged even while
    masked."""
    torch = pytest.importorskip("torch")
    adapter = _fake_glm_router_stack(torch)
    target = InterventionTarget("expert", 0, 17)
    _, block = adapter.layers[0]
    router = adapter._router_of(block)[1]
    hidden = -torch.ones(3, 4096)  # NEGATIVE-sum hidden states
    adapter.freeze_baseline()
    with torch.no_grad():
        base_logits, base_weights, base_indices = router(hidden)
    # a nonzero correction bias must push expert 17 into the UNMASKED
    # dispatch (sigmoid(logits) may be low, but the +5 bias added to the
    # selection scores wins the top-8) -- this is the exact condition that
    # silently defeats a naive "-1e9 logit" mask
    router.e_score_correction_bias[17] = 5.0
    with torch.no_grad():
        _, _, biased_indices = router(hidden)
    assert 17 in biased_indices.reshape(-1).tolist()
    result = adapter.temporary_mask(target)
    assert result["weights_modified"] is False
    assert result["mechanism"] == "router_output_forward_hook_full_tuple"
    with torch.no_grad():
        # the SAME bias stays set: the mask must suppress 17 anyway,
        # because the suppression hits the selection scores BEFORE the
        # bias is added -- not just the logits
        masked_logits, masked_weights, masked_indices = router(hidden)
        # the masked expert's logit is -1e9 for every token, and the OTHER
        # experts' logits are untouched
        assert torch.all(masked_logits[:, 17] == -1.0e9)
        others = [e for e in range(288) if e != 17]
        assert torch.equal(masked_logits[:, others], base_logits[:, others])
        # the masked expert can never enter the dispatched top-8, bias or no
        assert 17 not in masked_indices.reshape(-1).tolist()
        # every token still dispatches exactly 8 experts whose weights are
        # the recomputed pre-bias gather (finite, rescaled)
        assert masked_indices.shape == base_indices.shape
        assert torch.isfinite(masked_weights).all()
    router.e_score_correction_bias[17] = 0.0
    # weights were never modified: unchanged holds DURING the mask window
    assert adapter.verify_unchanged()["unchanged"] is True
    assert adapter.active_masks() == ["3:expert:0/17"]
    restored = adapter.restore_mask(target)
    assert restored["restored"] is True
    assert restored["weights_modified"] is False
    with torch.no_grad():
        again = router(hidden)
        assert torch.equal(again[0], base_logits)
        assert torch.equal(again[1], base_weights)
        assert torch.equal(again[2], base_indices)
    assert adapter.verify_unchanged()["unchanged"] is True


def test_glm_mask_refuses_partial_scale_layer_kind_and_double_ops():
    torch = pytest.importorskip("torch")
    adapter = _fake_glm_router_stack(torch)
    with pytest.raises(InterventionInvalid):
        adapter.temporary_mask(InterventionTarget("expert", 0, 5), scale=0.5)
    with pytest.raises(InterventionInvalid):
        # masking every expert would leave top-k firing over all -1e9
        # logits: silent under-suppression, refused
        adapter.temporary_mask(InterventionTarget("layer", 0))
    with pytest.raises(InterventionInvalid):
        adapter.temporary_mask(InterventionTarget("expert", 0, 288))
    adapter.temporary_mask(InterventionTarget("expert", 0, 5))
    with pytest.raises(InterventionInvalid):
        adapter.temporary_mask(InterventionTarget("expert", 0, 5))
    with pytest.raises(InterventionInvalid):
        # cannot restore what was never masked
        adapter.restore_mask(InterventionTarget("expert", 1, 2))


def test_switch_mask_suppresses_expert_and_restore_succeeds():
    """Same defect class in the Switch adapter: the old ``-1e9`` weight-row
    mask made the masked expert's logit ``-1e9 * sum(hidden)`` -- POSITIVE
    for negative-sum hidden states. The classifier-OUTPUT hook is
    input-independent; restoration removes the hook."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.51.3":
        pytest.skip("Supported compiler runtime is transformers 4.51.3")
    from transformers import (
        SwitchTransformersConfig,
        SwitchTransformersForConditionalGeneration,
    )

    from asea.capability_build.adapters.switch import SwitchAdapter

    torch.manual_seed(7)
    config = SwitchTransformersConfig(
        vocab_size=12, d_model=8, d_ff=16, d_kv=4, num_heads=2, num_layers=2,
        num_decoder_layers=2, num_sparse_encoder_layers=1,
        num_sparse_decoder_layers=1, num_experts=4, expert_capacity=8,
        dropout_rate=0, router_jitter_noise=0, decoder_start_token_id=0,
        pad_token_id=0, eos_token_id=1,
    )
    model = SwitchTransformersForConditionalGeneration(config)
    adapter = SwitchAdapter(model, torch)
    target = InterventionTarget("expert", 0, 2)
    _, layer = adapter.layers[0]
    classifier = layer.router.classifier
    hidden = -torch.ones(2, 8)  # negative-sum hidden states
    with torch.no_grad():
        baseline = classifier(hidden).clone()
    adapter.freeze_baseline()
    result = adapter.temporary_mask(target)
    assert result["weights_modified"] is False
    with torch.no_grad():
        masked = classifier(hidden)
        assert torch.all(masked[:, 2] == -1.0e9)
        assert torch.equal(masked[:, [0, 1, 3]], baseline[:, [0, 1, 3]])
        assert 2 not in masked.argmax(dim=-1).tolist()
    assert adapter.verify_unchanged()["unchanged"] is True
    adapter.restore_mask(target)
    with torch.no_grad():
        assert torch.equal(classifier(hidden), baseline)
    assert adapter.verify_unchanged()["unchanged"] is True


# ---------------------------------------------------------------------------
# Evaluation platform honesty
# ---------------------------------------------------------------------------

def test_require_sandbox_matches_platform():
    import platform

    from asea.capability_build.evaluation import require_sandbox, sandbox_supported

    assert sandbox_supported() == (platform.system() == "Linux")
    if platform.system() != "Linux":
        with pytest.raises(BlockedResource) as excinfo:
            require_sandbox()
        assert "Remedy" in str(excinfo.value)


def test_evaluate_code_cases_validates_cases():
    import platform

    from asea.capability_build.evaluation import evaluate_code_cases

    if platform.system() != "Linux":
        # Fails closed with the exact remedy on non-Linux hosts.
        with pytest.raises(BlockedResource):
            evaluate_code_cases("def f(): pass", [{"group": "target"}])
        return
    with pytest.raises(ValueError):
        evaluate_code_cases("def f(): pass", [])
    with pytest.raises(ValueError):
        evaluate_code_cases("   ", [{"group": "target"}])
    with pytest.raises(ValueError):
        evaluate_code_cases("def f(): pass", [{"sample_id": "x"}])


# ---------------------------------------------------------------------------
# Audit 2026-09-19 fix: authored-denominator accounting. The previous
# release counted only boolean-matched verdicts, so every check the oracle
# could not judge (candidate import/call error, timeout, resource kill)
# silently vanished from the denominator and INFLATED every rate -- the
# defect that produced the wrong published A/B numbers (held-out was
# reported as improved when it had regressed). These tests pin the
# corrected contract; each was verified to FAIL on the pre-fix code.
# ---------------------------------------------------------------------------

def _fake_oracle(monkeypatch, result):
    import asea.capability_build.evaluation as evaluation_module

    def fake_evaluate_functions(source, cases, limits=None, trace_policy=None):
        return result

    monkeypatch.setattr(evaluation_module, "require_sandbox", lambda: None)
    import asea.certification.function_oracle as oracle_module

    monkeypatch.setattr(oracle_module, "evaluate_functions", fake_evaluate_functions)


def _ab_cases(n=3):
    return [
        {"id": "c%d" % i, "group": "target" if i < n - 1 else "control",
         "function": "f", "args": [i], "kwargs": {}, "expected": i}
        for i in range(n)
    ]


def test_evaluate_counts_candidate_errors_as_failed_checks(monkeypatch):
    """The headline defect: a candidate that crashed on call made its whole
    case disappear from the denominator (judged_cases=0, rate inflated for
    every other case). Authored-denominator contract: those checks count as
    FAILED, pass_rate is over the authored total and is 0.0 (not None) when
    nothing could be judged."""
    _fake_oracle(monkeypatch, {
        "status": "FAILED",
        "cases": [],  # the oracle returns nothing for a crashed candidate
        "diagnostic": {"event": "candidate_error",
                       "candidate_error": {"type_code": "ValueError"}},
    })
    from asea.capability_build.evaluation import evaluate_code_cases

    result = evaluate_code_cases("def f(i):\n    raise ValueError('boom')\n", _ab_cases())
    assert result["authored_cases"] == 3
    assert result["judged_cases"] == 0
    assert result["unjudged_cases"] == 3
    assert result["passed_cases"] == 0
    assert result["failed_cases"] == 3
    assert result["pass_rate"] == 0.0          # authored denominator, not None
    assert result["judged_pass_rate"] is None  # diagnostic only
    assert result["status"] == "FAILED"
    assert result["groups"]["target"] == {
        "passed": 0, "total": 2, "unjudged": 2, "pass_rate": 0.0}
    assert result["groups"]["control"] == {
        "passed": 0, "total": 1, "unjudged": 1, "pass_rate": 0.0}


def test_evaluate_partial_oracle_response_counts_missing_checks_failed(monkeypatch):
    """A partial oracle response (some cases judged, others lost to an
    aborted run) must not shrink the denominator to the surviving subset."""
    _fake_oracle(monkeypatch, {
        "status": "FAILED",
        "cases": [{"id": "c0", "matched": True}],
        "diagnostic": {"event": "candidate_error", "candidate_error": None},
    })
    from asea.capability_build.evaluation import evaluate_code_cases

    result = evaluate_code_cases("def f(i):\n    return i\n", _ab_cases())
    assert result["authored_cases"] == 3
    assert result["judged_cases"] == 1
    assert result["unjudged_cases"] == 2
    assert result["passed_cases"] == 1
    assert result["pass_rate"] == 1 / 3
    assert result["judged_pass_rate"] == 1.0


def test_evaluate_unsupported_return_value_is_a_judged_failure(monkeypatch):
    """An executed check whose output does not match is judged False: it
    lands in the judged set and deflates the rate -- never dropped."""
    _fake_oracle(monkeypatch, {
        "status": "FAILED",
        "cases": [{"id": "c0", "matched": False},
                  {"id": "c1", "matched": False},
                  {"id": "c2", "matched": False}],
    })
    from asea.capability_build.evaluation import evaluate_code_cases

    result = evaluate_code_cases("def f(i):\n    return None\n", _ab_cases())
    assert result["judged_cases"] == 3
    assert result["unjudged_cases"] == 0
    assert result["pass_rate"] == 0.0
    assert result["judged_pass_rate"] == 0.0


def test_evaluate_refuses_unknown_duplicate_and_verdictless_cases(monkeypatch):
    """The oracle response is host-owned evidence: an id that was never
    authored, a duplicated id, or a verdictless entry is a contract
    violation -- refuse the whole report rather than grade on it."""
    from asea.capability_build.evaluation import evaluate_code_cases

    _fake_oracle(monkeypatch, {"status": "FAILED",
                               "cases": [{"id": "cX", "matched": True}]})
    with pytest.raises(ValueError, match="not authored"):
        evaluate_code_cases("def f(i): return i", _ab_cases())

    _fake_oracle(monkeypatch, {"status": "FAILED",
                               "cases": [{"id": "c0", "matched": True},
                                         {"id": "c0", "matched": True}]})
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_code_cases("def f(i): return i", _ab_cases())

    _fake_oracle(monkeypatch, {"status": "FAILED",
                               "cases": [{"id": "c0", "matched": None}]})
    with pytest.raises(ValueError, match="no boolean verdict"):
        evaluate_code_cases("def f(i): return i", _ab_cases())


def test_evaluate_refuses_missing_status_never_defaults_to_measured(monkeypatch):
    """A missing oracle status must refuse the report; the previous release
    silently defaulted it to 'measured'."""
    _fake_oracle(monkeypatch, {"cases": [{"id": "c0", "matched": True}]})
    from asea.capability_build.evaluation import evaluate_code_cases

    with pytest.raises(ValueError, match="no status"):
        evaluate_code_cases("def f(i): return i", _ab_cases())


def test_evaluate_duplicate_authored_case_id_is_refused(monkeypatch):
    _fake_oracle(monkeypatch, {"status": "PASSED", "cases": []})
    from asea.capability_build.evaluation import evaluate_code_cases

    with pytest.raises(ValueError, match="duplicate oracle case id"):
        evaluate_code_cases("def f(i): return i", [
            {"id": "c0", "group": "target", "function": "f",
             "args": [0], "kwargs": {}, "expected": 0},
            {"id": "c0", "group": "target", "function": "f",
             "args": [1], "kwargs": {}, "expected": 1},
        ])


@pytest.mark.skipif(sys.platform != "linux", reason="real host oracle needs the Linux sandbox")
def test_evaluate_code_cases_real_oracle_counts_crashed_candidate_as_failed():
    """REAL integration companion to the mechanism tests: a candidate that
    raises on call, through the actual Linux sandbox. Every authored check
    must count as failed (0.0 pass rate over the authored denominator), the
    candidate-error diagnostic must be preserved, and the status must be
    the oracle's real one. Skips honestly where containment is unavailable."""
    from asea.certification import sandbox as _sandbox

    probe = _sandbox.probe_code_sandbox()
    if not probe.supported:
        pytest.skip("actual Linux containment unavailable: " + probe.reason + probe.stderr)
    assert probe.passed
    from asea.capability_build.evaluation import evaluate_code_cases

    source = "def add(a, b):\n    raise ValueError('candidate crashed')\n"
    cases = [
        {"id": "t1", "group": "target", "function": "add",
         "args": [1, 2], "kwargs": {}, "expected": 3},
        {"id": "t2", "group": "target", "function": "add",
         "args": [2, 2], "kwargs": {}, "expected": 4},
        {"id": "c1", "group": "control", "function": "add",
         "args": [0, 0], "kwargs": {}, "expected": 0},
    ]
    result = evaluate_code_cases(source, cases)
    assert result["blocked"] is False
    assert result["authored_cases"] == 3
    assert result["judged_cases"] == 0
    assert result["unjudged_cases"] == 3
    assert result["passed_cases"] == 0
    assert result["pass_rate"] == 0.0
    assert result["status"] in ("FAILED", "TIMEOUT", "RESOURCE_LIMIT")
    diagnostic = result["oracle"].get("diagnostic") or {}
    assert diagnostic.get("event") == "candidate_error"


# ---------------------------------------------------------------------------
# Audit 2026-09-19 item 8: the intervention generation+judging loop. The
# worker-side ``generate`` op plus make_worker_judge replace the old
# "generation+judging stage not wired" refusal; these tests pin the judge
# contract (regenerate under the CURRENT state, grade through the host
# oracle, refuse contaminated state attribution).
# ---------------------------------------------------------------------------

class _FakeWorkerClient:
    """Minimal worker-protocol stand-in: records requests, replays scripted
    ``generate`` responses. MECHANISM test: no GLM, no Docker."""

    def __init__(self, completions=None, active_masks=None):
        self.requests = []
        self._completions = list(completions or [])
        self._active_masks = list(active_masks or [])

    def request(self, op, payload):
        self.requests.append((op, payload))
        if op == "generate":
            item = self._completions.pop(0)
            return {"per_prompt": [dict(item)],
                    "active_masks": list(self._active_masks)}
        raise AssertionError("unexpected worker op %r" % op)


def _judge_case(sample_id="s1", group="target", frames=None, prompt="write f"):
    return {
        "sample_id": sample_id, "group": group, "prompt": prompt,
        "oracle": frames if frames is not None else [
            {"id": "o1", "function": "f", "args": [2],
             "kwargs": {}, "expected": 4},
        ],
    }


def test_extract_candidate_code_matches_pilot_rules():
    from asea.capability_build.evaluation import extract_candidate_code

    # the LAST fenced python block wins
    text = "intro\n```python\ndef f():\n    return 1\n```\nmore\n```python\ndef g():\n    return 2\n```\n"
    assert "def g" in extract_candidate_code(text)
    # no fence but a def: the whole response is the candidate
    assert "def f" in extract_candidate_code("Sure:\ndef f(x):\n    return x")
    # nothing extractable: candidate error (None, never a guess)
    assert extract_candidate_code("I cannot help with that.") is None
    assert extract_candidate_code("") is None


def test_worker_judge_refuses_cases_without_oracle_frames():
    from asea.capability_build.evaluation import make_worker_judge

    judge = make_worker_judge(_FakeWorkerClient())
    with pytest.raises(InterventionInvalid, match="oracle frames"):
        judge({"sample_id": "s1", "group": "target", "prompt": "p"},
              {"phase": "base", "masked_component": None})


def test_worker_judge_validates_frame_shape():
    from asea.capability_build.evaluation import make_worker_judge

    judge = make_worker_judge(_FakeWorkerClient())
    case = _judge_case(frames=[{"id": "o1", "function": "f", "args": [1]}])
    with pytest.raises(InterventionInvalid, match="expected"):
        judge(case, {"phase": "base", "masked_component": None})
    case = _judge_case(frames=[
        {"id": "o1", "function": "f", "args": [1], "kwargs": {}, "expected": 1},
        {"id": "o1", "function": "f", "args": [2], "kwargs": {}, "expected": 2},
    ])
    with pytest.raises(InterventionInvalid, match="reuses oracle frame id"):
        judge(case, {"phase": "base", "masked_component": None})


def test_worker_judge_refuses_contaminated_state_attribution():
    """The state-contamination guard: a verdict labeled "base" generated
    under a live mask (or labeled "masked" with no mask live) is refused
    by name -- the base and masked arms of an intervention must never be
    cross-contaminated silently."""
    from asea.capability_build.evaluation import make_worker_judge

    case = _judge_case()
    # base phase, but the worker reports a live mask
    client = _FakeWorkerClient(
        completions=[{"sample_id": "s1", "group": "target",
                      "completion": "```python\ndef f(i):\n    return i * i\n```",
                      "new_tokens": 20}],
        active_masks=["3:expert:0/17"])
    with pytest.raises(InterventionInvalid, match="not a baseline"):
        make_worker_judge(client)(case, {"phase": "base", "masked_component": None})
    # masked phase, but the worker reports NO live mask
    client = _FakeWorkerClient(
        completions=[{"sample_id": "s1", "group": "target", "completion": "x"}],
        active_masks=[])
    with pytest.raises(InterventionInvalid, match="NO active masks"):
        make_worker_judge(client)(case, {"phase": "masked",
                                         "masked_component": "expert:0/17"})


def test_worker_judge_regenerates_and_grades_through_oracle(monkeypatch):
    """The full verdict path: the judge regenerates the completion under
    the CURRENT state, extracts the candidate with the pilot's rule, and
    grades it through the (faked) host oracle over the case's authored
    frames -- all frames must pass. A completion with no extractable code
    is a candidate error: FAILED, never silently absent."""
    seen = {}

    def fake_evaluate_functions(source, cases, limits=None, trace_policy=None):
        seen["source"] = source
        seen["cases"] = cases
        return {"status": "OK",
                "cases": [{"id": c["id"], "matched": True} for c in cases]}

    import asea.capability_build.evaluation as evaluation_module
    import asea.certification.function_oracle as oracle_module

    monkeypatch.setattr(evaluation_module, "require_sandbox", lambda: None)
    monkeypatch.setattr(oracle_module, "evaluate_functions", fake_evaluate_functions)
    from asea.capability_build.evaluation import make_worker_judge

    completion = "Reasoning...\n```python\ndef f(i):\n    return i * i\n```"
    client = _FakeWorkerClient(completions=[
        {"sample_id": "s1", "group": "target", "completion": completion,
         "new_tokens": 24}])
    case = _judge_case()
    verdict = make_worker_judge(client, max_new_tokens=64)(
        case, {"phase": "base", "masked_component": None})
    assert verdict is True
    # the candidate the oracle graded is the EXTRACTED block verbatim
    # (including the block's trailing newline -- the extraction rule is
    # exact, never repaired), not the whole completion, and the oracle saw
    # the case's authored frames
    assert seen["source"] == "def f(i):\n    return i * i\n"
    assert [c["id"] for c in seen["cases"]] == ["o1"]
    # the wrapper strips the bookkeeping before the oracle sees the suite
    # (the oracle rejects any extra key, by contract)
    assert "group" not in seen["cases"][0]
    # the generation request carried the case's prompt and the knob
    op, payload = client.requests[0]
    assert op == "generate"
    assert payload["prompts"][0]["prompt"] == "write f"
    assert payload["max_new_tokens"] == 64

    # a completion with no extractable code: candidate error -> False, and
    # the oracle is never even asked
    client = _FakeWorkerClient(completions=[
        {"sample_id": "s1", "group": "target", "completion": "no code here",
         "new_tokens": 4}])
    assert make_worker_judge(client)(case, {"phase": "base",
                                           "masked_component": None}) is False


def test_worker_intervention_adapter_proxies_protocol_ops():
    """run_intervention's adapter contract proxied to the worker: verify/
    mask/restore are worker ops carrying the component as a dict."""
    from asea.capability_build.__main__ import _WorkerInterventionAdapter
    from asea.capability_build.intervention import InterventionTarget

    class _RecordingClient:
        def __init__(self):
            self.requests = []

        def request(self, op, payload):
            self.requests.append((op, payload))
            return {"unchanged": True, "active_masks": []}

    client = _RecordingClient()
    adapter = _WorkerInterventionAdapter(client)
    component = InterventionTarget("expert", 0, 17)
    assert adapter.verify_unchanged() == {"unchanged": True, "active_masks": []}
    assert adapter.temporary_mask(component) is not None
    assert adapter.restore_mask(component) is not None
    assert [(op, payload) for op, payload in client.requests] == [
        ("verify", {}),
        ("mask", {"target": {"kind": "expert", "layer": 0, "expert_id": 17}}),
        ("restore", {"target": {"kind": "expert", "layer": 0, "expert_id": 17}}),
    ]


def test_worker_judge_end_to_end_intervention_through_fake_worker(monkeypatch):
    """The WHOLE item-8 loop end to end at mechanism scale: run_intervention
    drives the worker facade -- verify-clean, base arm generates+judges
    every case, mask ON, masked arm repeats the IDENTICAL cases, mask OFF,
    verify-clean -- and the drops reflect the two arms' different verdicts.
    A replay judge could never produce a nonzero drop here; this test
    fails if the judge ignores the teacher state."""
    def fake_evaluate_functions(source, cases, limits=None, trace_policy=None):
        # base-arm candidates always pass; the masked arm never reaches the
        # oracle at all (its completion has no extractable code)
        return {"status": "OK",
                "cases": [{"id": c["id"], "matched": True} for c in cases]}

    import asea.capability_build.evaluation as evaluation_module
    import asea.certification.function_oracle as oracle_module

    monkeypatch.setattr(evaluation_module, "require_sandbox", lambda: None)
    monkeypatch.setattr(oracle_module, "evaluate_functions", fake_evaluate_functions)
    from asea.capability_build.__main__ import _WorkerInterventionAdapter
    from asea.capability_build.evaluation import make_worker_judge
    from asea.capability_build.intervention import InterventionTarget, run_intervention

    class _StateAwareClient:
        """Simulates the worker: masks toggle what generate returns, and
        the live-mask registry is reported with every generation exactly as
        the real worker does."""

        def __init__(self):
            self.requests = []
            self.masked = None

        def request(self, op, payload):
            self.requests.append((op, payload))
            if op == "verify":
                return {"unchanged": True, "detail": "hash-identical",
                        "active_masks": [] if self.masked is None
                        else [self.masked]}
            if op == "mask":
                self.masked = "3:expert:0/17"
                return {"key": self.masked, "weights_modified": False}
            if op == "restore":
                self.masked = None
                return {"restored": True, "weights_modified": False}
            if op == "generate":
                sample = payload["prompts"][0]
                completion = (
                    "```python\ndef f(i):\n    return i * i\n```"
                    if self.masked is None else "no code"
                )
                return {"per_prompt": [{
                            "sample_id": sample["sample_id"],
                            "group": sample["group"],
                            "completion": completion, "new_tokens": 12}],
                        "active_masks": [] if self.masked is None
                        else [self.masked]}
            raise AssertionError("unexpected op %r" % op)

    client = _StateAwareClient()
    frames = [{"id": "o1", "function": "f", "args": [2],
               "kwargs": {}, "expected": 4}]
    cases = [
        {"sample_id": "t%d" % i, "group": "target", "prompt": "p",
         "oracle": frames} for i in range(2)
    ] + [
        {"sample_id": "c%d" % i, "group": "control", "prompt": "p",
         "oracle": frames} for i in range(2)
    ]
    entry = run_intervention(
        _WorkerInterventionAdapter(client),
        InterventionTarget("expert", 0, 17),
        target_cases=[c for c in cases if c["group"] == "target"],
        control_cases=[c for c in cases if c["group"] == "control"],
        judge=make_worker_judge(client, max_new_tokens=32),
        seed=0,
    )
    # base arm: code extracted, oracle verdicts all True (rate 1.0);
    # masked arm: "no code" -> candidate error -> False (rate 0.0)
    assert entry.target_drop == 1.0
    assert entry.control_drop == 1.0
    assert entry.restored_and_verified is True
    # the protocol ran in order: verify, base generations, mask, masked
    # generations, restore -- and the mask was OFF again afterwards
    ops = [op for op, _ in client.requests]
    assert ops[0] == "verify"
    assert "mask" in ops and "restore" in ops
    assert ops.index("mask") < ops.index("restore")
    assert client.masked is None
    # the masked arm generated under the SAME case set in the same seeded
    # order (identical prompts appear once per case per arm: 4 cases x 2
    # arms = 8 generation requests)
    generate_payloads = [p for op, p in client.requests if op == "generate"]
    assert len(generate_payloads) == 8
    base_prompts = [p["prompts"][0]["sample_id"] for p in generate_payloads[:4]]
    masked_prompts = [p["prompts"][0]["sample_id"] for p in generate_payloads[4:]]
    assert sorted(base_prompts) == sorted(masked_prompts)


# ---------------------------------------------------------------------------
# CLI (in-process: exit codes and JSON contract)
# ---------------------------------------------------------------------------

def _write_spec(tmp_path, access="behavioural_remote"):
    spec = json.loads(json.dumps(SPEC))
    spec["teacher"]["access"] = access
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return str(path)


def test_cli_spec_validate(tmp_path, capsys):
    from asea.capability_build.__main__ import main

    code = main(["spec", "validate", "--spec", _write_spec(tmp_path)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["autoactivated"] is False
    assert payload["thresholds_are_operator_configuration"] is True


def test_cli_teacher_baseline_refuses_without_consent(tmp_path, capsys):
    from asea.capability_build.__main__ import main

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {"sample_id": "a", "group": "target", "prompt": "p"},
    ]), encoding="utf-8")
    code = main([
        "teacher-baseline", "--spec", _write_spec(tmp_path),
        "--cases", str(cases), "--workspace", str(tmp_path / "ws"),
    ])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "consent" in payload["error"]


def test_cli_pending_commands_refuse_honestly(tmp_path, capsys):
    from asea.capability_build.__main__ import main

    code = main(["reduce", "--spec", _write_spec(tmp_path),
                 "--workspace", str(tmp_path / "ws")])
    assert code == 0  # completed command surface; the payload states refusal
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "rejected"
    assert "not implemented" in payload["reason"]


def test_cli_intervene_needs_internal_spec(tmp_path, capsys):
    from asea.capability_build.__main__ import main

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {"sample_id": "t1", "group": "target", "prompt": "p",
         "oracle": [{"id": "o1", "function": "f", "args": [1],
                     "kwargs": {}, "expected": 1}]},
        {"sample_id": "c1", "group": "control", "prompt": "p",
         "oracle": [{"id": "o1", "function": "f", "args": [2],
                     "kwargs": {}, "expected": 2}]},
    ]), encoding="utf-8")
    code = main(["intervene", "--spec", _write_spec(tmp_path),
                 "--workspace", str(tmp_path / "ws"),
                 "--cases", str(cases),
                 "--component", "expert:3/7"])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    # behavioural spec: refused because internals do not exist remotely
    assert payload["ok"] is False


# ---------------------------------------------------------------------------
# Student baselines (MECHANISM TEST: a fake local student is not a
# Qwen capability result)
# ---------------------------------------------------------------------------

def test_local_student_refuses_remote_host():
    from asea.capability_build.student import local_ollama_student

    with pytest.raises(BlockedResource):
        local_ollama_student("m", host="http://gpu.box.example:11434")


def test_local_student_host_validation_parses_real_hostnames():
    """Defect fix: the old string-prefix check accepted
    ``http://localhost.example.invalid:11434`` as "local". A REAL URL
    parse must refuse it, along with the userinfo/path/query/scheme
    variants -- a resolvable name is never provably local."""
    from asea.capability_build.student import validate_local_student_host

    # literal loopback forms are accepted (parsed hostname returned)
    assert validate_local_student_host("http://localhost:11434") == "localhost"
    assert validate_local_student_host("http://127.0.0.1:11434") == "127.0.0.1"
    assert validate_local_student_host("http://[::1]:11434/") == "::1"
    assert validate_local_student_host("http://0.0.0.0:11434") == "0.0.0.0"
    # the reviewer's exact bypass and its cousins are all REFUSED
    for host in (
        "http://localhost.example.invalid:11434",
        "http://127.0.0.1.evil.example:11434",
        "http://localhost@evil.example:11434",        # userinfo trick
        "http://localhost:11434/redirect",             # path
        "https://localhost:11434",                    # non-http scheme
        "http://localhost:11434?next=http://evil.example",
        "http://localhost:11434#frag",
        "file:///etc/passwd",
        "not a url",
    ):
        with pytest.raises(BlockedResource):
            validate_local_student_host(host)


def test_measure_student_baseline_groups_and_note():
    from asea.capability_build.student import measure_student_baseline

    student = {"kind": "ollama_local", "model": "toy"}
    cases = [
        {"sample_id": "a", "group": "target", "prompt": "p1"},
        {"sample_id": "b", "group": "target", "prompt": "p2"},
        {"sample_id": "c", "group": "control", "prompt": "p3"},
    ]

    def infer(_student, case):
        return "answer-for-%s" % case["sample_id"] if case["sample_id"] != "b" else "wrong"

    def judge(case, output):
        return output == "answer-for-%s" % case["sample_id"]

    summary = measure_student_baseline(student, cases, infer=infer, judge=judge)
    assert summary["groups"]["target"] == {"passed": 1, "total": 2, "pass_rate": 0.5}
    assert summary["groups"]["control"] == {"passed": 1, "total": 1, "pass_rate": 1.0}
    # numbers only -- never a capability claim
    assert "not a capability claim" in summary["note"]


def test_measure_student_baseline_needs_cases():
    from asea.capability_build.student import measure_student_baseline

    with pytest.raises(BlockedResource):
        measure_student_baseline({}, [], infer=lambda *a: "", judge=lambda *a: True)


def test_measured_gap_actionability_threshold():
    from asea.capability_build.student import measured_gap

    big = measured_gap(0.9, 0.2)
    assert big["gap"] == 0.7 and big["actionable"] is True
    small = measured_gap(0.9, 0.88)
    assert small["gap"] == pytest.approx(0.02) and small["actionable"] is False


# ---------------------------------------------------------------------------
# Sequence-level KD pairs (leakage discipline poisons the whole set)
# ---------------------------------------------------------------------------

def _judged_behavioural_trace(sample_id, prompt, response, success):
    return make_behavioural_trace(
        capability_id="cap",
        sample_id=sample_id,
        model_revision="rev",
        prompt=prompt,
        group="target",
        outcome=TraceOutcome(success=success),
        behavioural=BehaviouralRecord(prompt=prompt, response=response),
    )


def test_sequence_pairs_split_success_failure_unjudged():
    from asea.capability_build.distillation import build_sequence_pairs

    traces = [
        _judged_behavioural_trace("a", "fix bug one", "patch one", True),
        _judged_behavioural_trace("b", "fix bug two", "bad attempt", False),
        _judged_behavioural_trace(
            "c", "fix bug three", "cannot say",
            None,  # UNJUDGED teacher output must never teach
        ),
    ]
    pairs = build_sequence_pairs(traces)
    assert [p["sample_id"] for p in pairs["positives"]] == ["a"]
    assert [p["sample_id"] for p in pairs["negatives"]] == ["b"]
    assert pairs["unjudged_excluded"] == 1
    assert pairs["sequence_level_only"] is True
    assert "no vocabulary" in pairs["compatibility_note"]


def test_sequence_pairs_leakage_poisons_whole_set():
    from asea.capability_build.distillation import build_sequence_pairs

    traces = [
        _judged_behavioural_trace("train-1", "fix bug one", "patch one", True),
        _judged_behavioural_trace("final-9", "fix bug two", "patch two", True),
    ]
    with pytest.raises(LeakageError):
        build_sequence_pairs(traces, protected_sample_ids={"final-9"})


def test_sequence_pairs_enforce_protected_families():
    """Defect fix: ``protected_families`` is ENFORCED, not accepted-and-
    ignored. A trace whose family is protected, or whose family cannot be
    resolved through the approved dataset, poisons the whole build."""
    from asea.capability_build.distillation import build_sequence_pairs

    traces = [_judged_behavioural_trace("t-1", "train prompt body", "r1", True)]
    # (1) the trace's own family is protected: a family never crosses splits
    with pytest.raises(LeakageError):
        build_sequence_pairs(
            traces, protected_families={"fam-train"},
            sample_families={"t-1": "fam-train"},
        )
    # (2) family not resolvable through the dataset: unverifiable is not safe
    with pytest.raises(LeakageError):
        build_sequence_pairs(
            traces, protected_families={"fam-heldout"}, sample_families={},
        )
    # (3) protected families supplied without a lookup: refused outright
    with pytest.raises(CapabilityBuildError):
        build_sequence_pairs(traces, protected_families={"fam-heldout"})
    # (4) a resolvable, non-protected family passes
    pairs = build_sequence_pairs(
        traces, protected_families={"fam-heldout"},
        sample_families={"t-1": "fam-train"},
    )
    assert pairs["positives"][0]["sample_id"] == "t-1"


def test_sequence_pairs_enforce_protected_prompts():
    """The exact PROMPT of a protected-split case poisons the build even
    when the response differs (content-hash checks alone would miss it)."""
    import hashlib

    from asea.capability_build.distillation import build_sequence_pairs

    traces = [_judged_behavioural_trace("t-1", "train prompt body", "different r", True)]
    with pytest.raises(LeakageError):
        build_sequence_pairs(
            traces,
            protected_prompts={hashlib.sha256(b"train prompt body").hexdigest()},
        )


def test_sequence_pairs_enforce_protected_prompt_shapes():
    """Defect fix: a protected prompt rewritten only in capitalisation or
    punctuation is STILL protected. The exact content-hash guard misses
    that; the token-shape comparison catches it."""
    from asea.capability_build.distillation import _token_shape, build_sequence_pairs

    traces = [_judged_behavioural_trace("t-1", "Fix, BUG: body!", "r1", True)]
    # "fix bug body" is the protected prompt with case/punctuation rewritten
    with pytest.raises(LeakageError) as excinfo:
        build_sequence_pairs(
            traces, protected_prompt_shapes={_token_shape("fix bug body")},
        )
    assert "token-shape" in str(excinfo.value)
    # a genuinely different prompt is untouched by the guard
    pairs = build_sequence_pairs(
        traces, protected_prompt_shapes={_token_shape("an unrelated prompt")},
    )
    assert pairs["positives"][0]["sample_id"] == "t-1"


def test_sequence_pairs_reject_duplicate_and_near_duplicate():
    from asea.capability_build.distillation import build_sequence_pairs

    exact = [
        _judged_behavioural_trace("a", "same prompt", "r1", True),
        _judged_behavioural_trace("b", "same prompt", "r2", True),
    ]
    with pytest.raises(LeakageError):
        build_sequence_pairs(exact)
    near = [
        _judged_behavioural_trace("a", "Fix, Bug: one!", "r1", True),
        _judged_behavioural_trace("b", "fix bug one", "r2", True),
    ]
    with pytest.raises(LeakageError):
        build_sequence_pairs(near)


def test_hand_to_deepapply_needs_positives_and_never_preapproves():
    from asea.capability_build.distillation import (
        build_sequence_pairs,
        hand_to_deepapply,
    )

    traces = [_judged_behavioural_trace("a", "p", "r", True)]
    pairs = build_sequence_pairs(traces)
    handoff = hand_to_deepapply(pairs)
    assert handoff["sink"].startswith("asea.deepapply.")
    assert handoff["intake"].startswith("Gate 2")
    assert handoff["pre_approval"] is False
    assert handoff["autoactivated"] is False
    with pytest.raises(CapabilityBuildError):
        hand_to_deepapply({"schema": "other", "positives": []})


# ---------------------------------------------------------------------------
# Minimum-capability search (MECHANISM TEST: synthetic candidates, not a
# reduced-model quality result)
# ---------------------------------------------------------------------------
def test_search_selects_smallest_passing_candidate(spec):
    """Defect fix: the search evaluates EVERY candidate and admits the
    SMALLEST passing one by measured size -- it must not stop at the
    first pass when a smaller passing candidate follows."""
    from asea.capability_build.search import search_with_reference

    plans = [{"keep": 10}, {"keep": 4}, {"keep": 2}]

    def evaluate_by_plan(candidate):
        keep = candidate["keep"]
        if keep == 10:
            return {"target_score": 1.0,
                    "control_scores": {"math_reasoning": 1.0},
                    "size_bytes": 100}
        if keep == 4:
            return {"target_score": 0.9,
                    "control_scores": {"math_reasoning": 0.95},
                    "size_bytes": 60}
        return {"target_score": 0.5,
                "control_scores": {"math_reasoning": 0.94},
                "size_bytes": 30}

    result = search_with_reference(
        spec, teacher_target_score=1.0,
        teacher_control_baselines={"math_reasoning": 0.95},
        candidate_plan=plans, build=lambda p: p, evaluate=evaluate_by_plan,
    )
    # threshold = 0.8 * 1.0. keep=10 passes too (size 100) but the search
    # must admit the smaller passing keep=4 (size 60), and the whole plan
    # was evaluated: the keep=2 rejection is visible in the iterations.
    assert result["accepted"]["plan"]["keep"] == 4
    assert result["accepted"]["size_bytes"] == 60
    assert result["accepted"]["admission"] == "CANDIDATE_UNADMITTED"
    assert result["autoactivated"] is False
    assert len(result["iterations"]) == 3
    assert result["iterations"][2]["status"] == "rejected"
    assert result["failed_experiments_visible"] is True


def test_search_rejects_candidate_below_threshold(spec):
    from asea.capability_build.search import search_with_reference

    result = search_with_reference(
        spec, teacher_target_score=1.0,
        teacher_control_baselines={"math_reasoning": 1.0},
        candidate_plan=[{"keep": 2}],
        build=lambda p: p,
        evaluate=lambda c: {"target_score": 0.5,
                            "control_scores": {"math_reasoning": 1.0},
                            "size_bytes": 10},
    )
    assert result["accepted"] is None
    assert result["iterations"][0]["status"] == "rejected"
    assert result["failed_experiments_visible"] is True


def test_search_rejects_unmeasured_controls_sizes_and_budget(spec):
    """Defect fix: candidates with NO control results, a zero/missing
    size, or a size over the spec's storage budget are all REJECTED with
    the reason recorded -- never accepted by default."""
    from asea.capability_build.schema import HardwareBudget
    from asea.capability_build.search import search_with_reference

    budgeted = spec.model_copy(update={
        "hardware_budget": HardwareBudget(max_model_storage_bytes=50),
    })
    plans = [{"name": "no-controls"}, {"name": "zero-size"},
             {"name": "missing-size"}, {"name": "over-budget"}]

    def evaluate(candidate):
        name = candidate["name"]
        if name == "no-controls":
            return {"target_score": 1.0, "control_scores": {}, "size_bytes": 10}
        if name == "zero-size":
            return {"target_score": 1.0,
                    "control_scores": {"math_reasoning": 1.0}, "size_bytes": 0}
        if name == "missing-size":
            return {"target_score": 1.0,
                    "control_scores": {"math_reasoning": 1.0}}
        return {"target_score": 1.0,
                "control_scores": {"math_reasoning": 1.0}, "size_bytes": 100}

    result = search_with_reference(
        budgeted, teacher_target_score=1.0,
        teacher_control_baselines={"math_reasoning": 1.0},
        candidate_plan=plans, build=lambda p: p, evaluate=evaluate,
    )
    assert result["accepted"] is None
    by_name = {i["plan"]["name"]: i for i in result["iterations"]}
    assert all(i["status"] == "rejected" for i in result["iterations"])
    assert "not measured" in by_name["no-controls"]["reason"]
    assert "size" in by_name["zero-size"]["reason"]
    assert "size" in by_name["missing-size"]["reason"]
    assert "exceeds" in by_name["over-budget"]["reason"]


def test_search_uses_measured_baseline_not_assumed_unity(spec):
    """Defect fix: control regression is computed against the MEASURED
    teacher control baseline, never an assumed 1.0. A candidate at the
    teacher's real 0.9 control baseline must NOT be counted as
    regressing (against an assumed 1.0 it would have been wrongly
    rejected at the 0.05 tolerance)."""
    from asea.capability_build.search import search_with_reference

    result = search_with_reference(
        spec, teacher_target_score=1.0,
        teacher_control_baselines={"math_reasoning": 0.9},
        candidate_plan=[{"keep": 4}],
        build=lambda p: p,
        evaluate=lambda c: {"target_score": 0.9,
                            "control_scores": {"math_reasoning": 0.9},
                            "size_bytes": 60},
    )
    assert result["accepted"] is not None
    assert result["accepted"]["control_regression"] == pytest.approx(0.0)


def test_search_records_failed_builders_and_never_fabricates(spec):
    from asea.capability_build.search import search_with_reference

    def boom(_plan):
        raise RuntimeError("OOM")

    result = search_with_reference(
        spec, teacher_target_score=1.0,
        teacher_control_baselines={"math_reasoning": 1.0},
        candidate_plan=[{"keep": 1}],
        build=boom, evaluate=lambda c: {},
    )
    assert result["iterations"][0]["status"] == "failed"
    assert "OOM" in result["iterations"][0]["error"]
    assert result["accepted"] is None


def test_search_refuses_unmeasured_teacher_and_empty_plan(spec):
    """The search refuses to RUN against an unmeasured teacher: an empty
    control-baseline dict (controls would be scored against an assumed
    1.0) and a spec control group with no measured baseline are both
    refused up front, not silently papered over."""
    from asea.capability_build.search import search_with_reference

    def build(p):
        return p

    def evaluate(c):
        return {}

    with pytest.raises(CapabilityBuildError):
        search_with_reference(spec, teacher_target_score=0.0,
                              teacher_control_baselines={"math_reasoning": 1.0},
                              candidate_plan=[{"keep": 1}],
                              build=build, evaluate=evaluate)
    with pytest.raises(CapabilityBuildError):
        search_with_reference(spec, teacher_target_score=1.0,
                              teacher_control_baselines={"math_reasoning": 1.0},
                              candidate_plan=[], build=build, evaluate=evaluate)
    with pytest.raises(CapabilityBuildError):
        search_with_reference(spec, teacher_target_score=1.0,
                              teacher_control_baselines={},
                              candidate_plan=[{"keep": 1}],
                              build=build, evaluate=evaluate)
    with pytest.raises(CapabilityBuildError):
        search_with_reference(spec, teacher_target_score=1.0,
                              teacher_control_baselines={"unrelated_group": 1.0},
                              candidate_plan=[{"keep": 1}],
                              build=build, evaluate=evaluate)


def test_search_rejects_non_finite_scores(spec):
    """Defect fix: NaN and infinity are not measurements. A NaN control
    score passes every regression comparison (``nan <= x`` is always
    False, so it showed ZERO regression) and an infinite target score
    satisfies any threshold; both are refused BEFORE arithmetic, and a
    non-finite teacher score or control baseline refuses to run at all."""
    from asea.capability_build.search import search_with_reference

    nan = float("nan")
    inf = float("inf")
    build = lambda plan: plan  # noqa: E731

    # non-finite teacher score / control baseline: the search refuses to run
    with pytest.raises(CapabilityBuildError):
        search_with_reference(
            spec, teacher_target_score=nan,
            teacher_control_baselines={"math_reasoning": 1.0},
            candidate_plan=[{"keep": 1}], build=build,
            evaluate=lambda c: {},
        )
    with pytest.raises(CapabilityBuildError):
        search_with_reference(
            spec, teacher_target_score=1.0,
            teacher_control_baselines={"math_reasoning": inf},
            candidate_plan=[{"keep": 1}], build=build,
            evaluate=lambda c: {},
        )
    # NaN control score: "zero regression" must NOT sneak the candidate in
    result = search_with_reference(
        spec, teacher_target_score=1.0,
        teacher_control_baselines={"math_reasoning": 1.0},
        candidate_plan=[{"name": "nan-control"}], build=build,
        evaluate=lambda c: {"target_score": 0.9,
                            "control_scores": {"math_reasoning": nan},
                            "size_bytes": 10},
    )
    assert result["accepted"] is None
    assert result["iterations"][0]["status"] == "rejected"
    assert "finite" in result["iterations"][0]["reason"]
    # infinite target score satisfies any retention threshold: still rejected
    result = search_with_reference(
        spec, teacher_target_score=1.0,
        teacher_control_baselines={"math_reasoning": 1.0},
        candidate_plan=[{"name": "inf-target"}], build=build,
        evaluate=lambda c: {"target_score": inf,
                            "control_scores": {"math_reasoning": 1.0},
                            "size_bytes": 10},
    )
    assert result["accepted"] is None
    assert result["iterations"][0]["status"] == "rejected"
    assert "finite" in result["iterations"][0]["reason"]


def test_ollama_transport_refuses_redirects():
    """Defect fix: 'local-only, no redirects' is enforced by the TRANSPORT,
    not just the URL parse. urllib's default handler silently follows a
    3xx to ANY location (including off-host); the shared transport now
    refuses redirects outright, so a redirecting "local" host can never
    smuggle the request elsewhere. MECHANISM test through the ACTUAL
    connector against throwaway loopback servers -- no real model, no
    real external host."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from asea.modules.real.ollama import OllamaConnector, OllamaConnectionError

    hits = {"evil": 0}

    class Evil(BaseHTTPRequestHandler):
        def do_POST(self):
            hits["evil"] += 1
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        do_GET = do_POST

        def log_message(self, *args):
            pass

    class Redirect(BaseHTTPRequestHandler):
        def _redirect(self):
            self.send_response(302)
            self.send_header("Location", self.server.redirect_target)
            self.end_headers()

        do_POST = _redirect
        do_GET = _redirect

        def log_message(self, *args):
            pass

    evil = HTTPServer(("127.0.0.1", 0), Evil)
    redirect = HTTPServer(("127.0.0.1", 0), Redirect)
    redirect.redirect_target = "http://127.0.0.1:%d/" % evil.server_address[1]
    servers = [evil, redirect]
    for server in servers:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        connector = OllamaConnector(
            "student-model", [],
            host="http://127.0.0.1:%d" % redirect.server_address[1],
        )
        # a chat POST answered with 302 must surface the refusal, never
        # follow it
        with pytest.raises(OllamaConnectionError) as excinfo:
            connector._chat([{"role": "user", "content": "hi"}])
        assert "redirect" in str(excinfo.value)
        # the availability probe (GET) is refused the same way
        with pytest.raises(OllamaConnectionError) as excinfo:
            connector.health()
        assert "redirect" in str(excinfo.value)
        # the redirect target was never contacted
        assert hits["evil"] == 0
    finally:
        for server in servers:
            server.shutdown()


# ---------------------------------------------------------------------------
# CLI wiring for student-baseline and distill
# ---------------------------------------------------------------------------

def test_cli_student_baseline_refuses_remote_host(tmp_path, capsys):
    from asea.capability_build.__main__ import main

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {"sample_id": "a", "group": "target", "prompt": "p"},
    ]), encoding="utf-8")
    code = main([
        "student-baseline", "--spec", _write_spec(tmp_path),
        "--cases", str(cases), "--workspace", str(tmp_path / "ws"),
        "--host", "http://gpu.box.example:11434",
    ])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == BLOCKED_RESOURCE
    assert "not a literal loopback" in payload["requirement"]


def test_cli_student_baseline_blocked_without_local_daemon(tmp_path, capsys):
    from asea.capability_build.__main__ import main

    # Hermetic: a localhost port with nothing listening (the localhost
    # check passes, the daemon probe fails) -- the run must honestly
    # BLOCK, never fabricate a baseline.
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {"sample_id": "a", "group": "target", "prompt": "p"},
    ]), encoding="utf-8")
    code = main([
        "student-baseline", "--spec", _write_spec(tmp_path),
        "--cases", str(cases), "--workspace", str(tmp_path / "ws"),
        "--host", "http://localhost:9",
    ])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == BLOCKED_RESOURCE


def _seed_traces(workspace, traces):
    from asea.capability_build.store import CapabilityStore

    store = CapabilityStore(workspace)
    for n, trace in enumerate(traces):
        store.put("traces", "t%d" % n, trace.model_dump(mode="json", by_alias=True))
    return store


def _training_trace(sample_id="training-0", prompt=None, response="ok",
                    capability_id="python_repo_debugging_v1",
                    revision="glm-5.3-flash-20260901"):
    prompt = prompt or ("distinct prompt body for %s" % sample_id)
    return make_behavioural_trace(
        capability_id=capability_id,
        sample_id=sample_id,
        model_revision=revision,
        prompt=prompt,
        group="target",
        outcome=TraceOutcome(success=True),
        behavioural=BehaviouralRecord(prompt=prompt, response=response),
    )


def _built_dataset(tmp_path):
    from asea.capability_build.__main__ import main

    cases = _write_cases(tmp_path, _full_case_list())
    out = tmp_path / "capability_v1"
    assert main(["dataset", "build", "--spec", _write_spec(tmp_path),
                 "--cases", str(cases), "--out", str(out)]) == 0
    return out


def test_cli_distill_needs_traces(tmp_path, capsys):
    from asea.capability_build.__main__ import main

    out = _built_dataset(tmp_path)
    capsys.readouterr()
    code = main(["distill", "--spec", _write_spec(tmp_path),
                 "--dataset", str(out),
                 "--workspace", str(tmp_path / "ws")])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == BLOCKED_RESOURCE
    assert "no traces" in payload["requirement"]


def test_cli_distill_refuses_missing_dataset(tmp_path, capsys):
    from asea.capability_build.__main__ import main

    code = main(["distill", "--spec", _write_spec(tmp_path),
                 "--dataset", str(tmp_path / "no-such-dataset"),
                 "--workspace", str(tmp_path / "ws")])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == BLOCKED_RESOURCE
    assert "dataset" in payload["requirement"]


def test_cli_distill_refuses_trace_outside_training_split(tmp_path, capsys):
    """Defect fix: distill is bound to the approved dataset. A trace whose
    sample is NOT part of the dataset's training split is refused -- it
    can never become teaching material."""
    from asea.capability_build.__main__ import main

    out = _built_dataset(tmp_path)
    capsys.readouterr()
    _seed_traces(tmp_path / "ws", [
        _training_trace(sample_id="heldout-0"),
    ])
    code = main(["distill", "--spec", _write_spec(tmp_path),
                 "--dataset", str(out),
                 "--workspace", str(tmp_path / "ws")])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "rejected"
    assert "training split" in payload["error"]


def test_cli_distill_refuses_identity_mismatch(tmp_path, capsys):
    """The dataset manifest must carry the SAME capability id, teacher pin
    and spec fingerprint as --spec: identity is verified, never assumed
    from a path."""
    from asea.capability_build.__main__ import main

    out = _built_dataset(tmp_path)
    capsys.readouterr()
    other = json.loads(json.dumps(SPEC))
    other["teacher"]["revision"] = "glm-5.3-flash-OTHER-20260902"
    other_path = tmp_path / "spec-other.json"
    other_path.write_text(json.dumps(other), encoding="utf-8")
    code = main(["distill", "--spec", str(other_path),
                 "--dataset", str(out),
                 "--workspace", str(tmp_path / "ws")])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "rejected"
    assert "identity mismatch" in payload["error"]


def test_cli_distill_refuses_prompt_mismatch_with_training_row(tmp_path, capsys):
    """Defect fix: a training sample ID is an identity claim, not a free
    pass. A trace filed under a training ID whose prompt does not equal
    the frozen training row byte for byte is refused: the rewritten
    prompt was never approved teaching material."""
    from asea.capability_build.__main__ import main

    out = _built_dataset(tmp_path)
    capsys.readouterr()
    _seed_traces(tmp_path / "ws", [
        _training_trace(
            sample_id="training-0",
            prompt="TAMPERED prompt body for training-0",
        ),
    ])
    code = main(["distill", "--spec", _write_spec(tmp_path),
                 "--dataset", str(out),
                 "--workspace", str(tmp_path / "ws")])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "rejected"
    assert "does not match the approved training row" in payload["error"]


def test_cli_distill_happy_path_builds_pairs_and_handoff(tmp_path, capsys):
    from asea.capability_build.__main__ import main

    out = _built_dataset(tmp_path)
    capsys.readouterr()
    _seed_traces(tmp_path / "ws", [
        _training_trace(sample_id="training-0", response="ok-training-0"),
    ])
    code = main(["distill", "--spec", _write_spec(tmp_path),
                 "--dataset", str(out),
                 "--workspace", str(tmp_path / "ws")])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["positives"] == 1
    assert payload["dataset"]["training_samples"] == 3
    assert payload["dataset"]["protected_splits_enforced"] == [
        "development", "heldout", "final", "controls",
    ]
    assert payload["deepapply_handoff"]["pre_approval"] is False
    assert payload["deepapply_handoff"]["autoactivated"] is False

# ---------------------------------------------------------------------------
# Dataset builder (MECHANISM TEST: synthetic cases, never real teacher or
# student material)
# ---------------------------------------------------------------------------

def _case(sample_id, split, group="target", family=None, prompt=None):
    return {
        "sample_id": sample_id,
        "split": split,
        "group": group,
        "family_id": family or ("fam-%s" % sample_id),
        "prompt": prompt or ("distinct prompt body for %s" % sample_id),
        "expected": "ok-%s" % sample_id,
        "provenance": "synthetic-mechanism-test",
        "license": "CC0-1.0",
    }


def _full_case_list():
    cases = []
    for split in ("training", "development", "heldout", "final", "controls"):
        for n in range(3):
            group = "control" if split == "controls" else "target"
            cases.append(_case("%s-%d" % (split, n), split, group=group))
    return cases


def _write_cases(tmp_path, cases, name="cases.jsonl"):
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case) + "\n")
    return path


def test_dataset_build_and_validate_round_trip(tmp_path, spec):
    from asea.capability_build.dataset import build_dataset, validate_dataset

    out = tmp_path / "capability_v1"
    manifest = build_dataset(_full_case_list(), output_dir=out, spec=spec)
    assert manifest["frozen_before_model_generation"] is True
    assert manifest["final_opened"] is False
    assert manifest["counts"]["training"] == 3
    assert manifest["capability_id"] == "python_repo_debugging_v1"
    assert manifest["teacher"]["model"] == "glm-5.3-flash:cloud"
    assert manifest["teacher"]["revision"] == spec.teacher.revision
    assert len(manifest["spec_fingerprint"]) == 64
    result = validate_dataset(out)
    assert result["ok"] is True
    assert result["capability_id"] == spec.capability_id
    assert result["spec_fingerprint"] == manifest["spec_fingerprint"]
    assert (out / "selection-lock.json").is_file()


def test_dataset_requires_spec_and_binds_identity(tmp_path, spec):
    """Defect fix: a dataset built without a spec cannot bind capability /
    teacher / spec-fingerprint identity and may never back training
    pairs; validation refuses a manifest with a missing identity field."""
    from asea.capability_build.dataset import build_dataset, validate_dataset

    with pytest.raises(DatasetInvalid):
        build_dataset(_full_case_list(), output_dir=tmp_path / "nospec")
    out = tmp_path / "capability_v1"
    build_dataset(_full_case_list(), output_dir=out, spec=spec)
    manifest_path = out / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw.pop("spec_fingerprint")
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DatasetInvalid):
        validate_dataset(out)


def test_dataset_refuses_id_family_and_nearduplicate_leakage(tmp_path, spec):
    from asea.capability_build.dataset import build_dataset

    cases = _full_case_list()
    # same sample_id in two splits
    clash = _case("training-0", "development", family="other-fam")
    with pytest.raises(DatasetInvalid):
        build_dataset(cases + [clash], output_dir=tmp_path / "a", spec=spec)
    # family crossing splits
    cross = _case("new-id", "heldout", family="fam-training-0")
    with pytest.raises(DatasetInvalid):
        build_dataset(cases + [cross], output_dir=tmp_path / "b", spec=spec)
    # near-duplicate prompt shape across splits (new guard)
    near = _case("near-id", "heldout", family="near-fam",
                 prompt="DISTINCT prompt BODY for training-0")
    with pytest.raises(DatasetInvalid):
        build_dataset(cases + [near], output_dir=tmp_path / "c", spec=spec)
    # empty split
    only = [_case("t-0", "training")]
    with pytest.raises(DatasetInvalid):
        build_dataset(only, output_dir=tmp_path / "d", spec=spec)


def test_dataset_quarantines_frozen_september_inputs(tmp_path):
    from asea.capability_build.dataset import read_cases

    frozen = tmp_path / "specialist-v1" / "final.jsonl"
    frozen.parent.mkdir()
    _write_cases(frozen.parent, [_case("x-0", "training")], name="final.jsonl")
    with pytest.raises(DatasetInvalid):
        read_cases(frozen)


def test_dataset_output_must_be_new(tmp_path, spec):
    from asea.capability_build.dataset import build_dataset

    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(DatasetInvalid):
        build_dataset(_full_case_list(), output_dir=out, spec=spec)


def test_dataset_validate_detects_post_build_tampering(tmp_path, spec):
    from asea.capability_build.dataset import build_dataset, validate_dataset

    out = tmp_path / "capability_v1"
    build_dataset(_full_case_list(), output_dir=out, spec=spec)
    victim = out / "training.jsonl"
    victim.write_text(victim.read_text(encoding="utf-8").replace(
        "ok-training-0", "edited"), encoding="utf-8")
    with pytest.raises(DatasetInvalid):
        validate_dataset(out)


def test_dataset_validate_requires_selection_lock(tmp_path, spec):
    """Defect fix: a dataset directory without its selection lock was
    never frozen by a build (or the lock was removed) and may never back
    training pairs, no matter how internally consistent the remaining
    files look."""
    from asea.capability_build.dataset import build_dataset, validate_dataset

    out = tmp_path / "capability_v1"
    build_dataset(_full_case_list(), output_dir=out, spec=spec)
    (out / "selection-lock.json").unlink()
    with pytest.raises(DatasetInvalid) as excinfo:
        validate_dataset(out)
    assert "selection-lock" in str(excinfo.value)


def test_dataset_validate_binds_manifest_to_selection_lock(tmp_path, spec):
    """Defect fix: the manifest is verified against the selection lock
    byte for byte. A manifest edited AFTER the dataset was frozen is
    refused even when the edit leaves the per-file content hashes
    internally consistent."""
    from asea.capability_build.dataset import build_dataset, validate_dataset

    out = tmp_path / "capability_v1"
    build_dataset(_full_case_list(), output_dir=out, spec=spec)
    manifest_path = out / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["notes"] = "edited after the dataset was frozen"
    manifest_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    with pytest.raises(DatasetInvalid) as excinfo:
        validate_dataset(out)
    assert "manifest_sha256" in str(excinfo.value)


def test_cli_dataset_build_and_validate(tmp_path, capsys):
    from asea.capability_build.__main__ import main

    cases = _write_cases(tmp_path, _full_case_list())
    out = tmp_path / "capability_v1"
    code = main(["dataset", "build", "--spec", _write_spec(tmp_path),
                 "--cases", str(cases), "--out", str(out)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["counts"]["final"] == 3
    assert payload["capability_id"] == "python_repo_debugging_v1"
    assert payload["teacher"]["model"] == "glm-5.3-flash:cloud"
    code = main(["dataset", "validate", "--dir", str(out)])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_cli_dataset_build_refuses_leaky_cases(tmp_path, capsys):
    from asea.capability_build.__main__ import main

    cases = _write_cases(tmp_path, [_case("t-0", "training")])
    code = main(["dataset", "build", "--spec", _write_spec(tmp_path),
                 "--cases", str(cases), "--out", str(tmp_path / "v1")])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "rejected"


@pytest.mark.skipif(sys.platform != "linux", reason="real host oracle needs the Linux sandbox")
def test_evaluate_code_cases_real_oracle_groups():
    """REAL integration: one candidate through the actual Linux code
    sandbox + function oracle (unshare/seccomp/chroot). Not a mock.
    Skips honestly when the host cannot provide containment (the sandbox
    failing closed there is correct behaviour, covered by mechanism tests)."""
    from asea.certification import sandbox as _sandbox

    probe = _sandbox.probe_code_sandbox()
    if not probe.supported:
        pytest.skip("actual Linux containment unavailable: " + probe.reason + probe.stderr)
    assert probe.passed
    from asea.capability_build.evaluation import evaluate_code_cases

    source = "def add(a, b):\n    return a + b\n"
    cases = [
        {"id": "t1", "group": "target", "function": "add",
         "args": [1, 2], "kwargs": {}, "expected": 3},
        {"id": "t2", "group": "target", "function": "add",
         "args": [1, 1], "kwargs": {}, "expected": 5},
        {"id": "c1", "group": "control", "function": "add",
         "args": [2, 2], "kwargs": {}, "expected": 4},
    ]
    result = evaluate_code_cases(source, cases)
    assert result["blocked"] is False
    assert result["authored_cases"] == 3
    assert result["judged_cases"] == 3
    assert result["unjudged_cases"] == 0
    assert result["passed_cases"] == 2
    assert result["pass_rate"] == 2 / 3
    assert result["groups"]["target"] == {
        "passed": 1, "total": 2, "unjudged": 0, "pass_rate": 0.5}
    assert result["groups"]["control"] == {
        "passed": 1, "total": 1, "unjudged": 0, "pass_rate": 1.0}


@pytest.mark.skipif(sys.platform != "linux", reason="real host oracle needs the Linux sandbox")
def test_evaluate_code_cases_real_oracle_rejects_extra_group_key_in_oracle_frame():
    """The wrapper must strip the bookkeeping 'group' key; the oracle itself
    refuses any extra key (exact-frame contract), so this proves the strip.
    Skips honestly when the host cannot provide containment."""
    from asea.certification import sandbox as _sandbox

    probe = _sandbox.probe_code_sandbox()
    if not probe.supported:
        pytest.skip("actual Linux containment unavailable: " + probe.reason + probe.stderr)
    assert probe.passed
    from asea.capability_build.evaluation import evaluate_code_cases

    source = "def add(a, b):\n    return a + b\n"
    # Direct oracle-frame call WITHOUT the wrapper would fail on 'group';
    # through the wrapper the strip makes the suite valid:
    result = evaluate_code_cases(source, [
        {"id": "t1", "group": "target", "function": "add",
         "args": [1, 2], "kwargs": {}, "expected": 3},
    ])
    assert result["blocked"] is False
    assert result["judged_cases"] == 1
    assert result["groups"]["target"]["pass_rate"] == 1.0

# ---------------------------------------------------------------------------
# Audit 2026-09-18 fixes: mixed-evidence-class and foreign-capability
# footprint refusal, replay-judge refusal, active-mask honesty, proxy-proof
# transport, blocked-evaluate exit discipline, portable ids, receipt
# dematerialisation, shapeless-prompt refusal, worker hardening.
# Every test below was verified to FAIL on the pre-fix code (stash
# comparison) unless marked otherwise in its docstring.
# ---------------------------------------------------------------------------


def _store_trace(store, trace):
    return store.put(
        "traces",
        "%s-%s" % (trace.capability_id, trace.sample_id),
        trace.model_dump(mode="json", by_alias=True),
    )


def test_cli_footprint_refuses_mixed_evidence_classes(tmp_path, capsys):
    """Defect fix (C1): a workspace holding BOTH a behavioural and an
    internal trace used to silently produce a behavioural footprint
    (``trace_class or TRACE_CLASS_BEHAVIOURAL``), erasing the internal
    traces' class and violating the never-mix-evidence-classes constraint.
    The footprint command must REFUSE with the exact class names."""
    from asea.capability_build.__main__ import main

    ws = CapabilityStore(tmp_path / "ws")
    # both traces belong to the SPEC's capability, so the foreign-capability
    # refusal does not preempt the mixed-class one
    _store_trace(ws, make_behavioural_trace(
        capability_id="python_repo_debugging_v1", sample_id="b1",
        model_revision="rev", prompt="p", group="target",
        outcome=TraceOutcome(success=True),
        behavioural=BehaviouralRecord(prompt="p", response="r"),
    ))
    _store_trace(ws, make_internal_trace(
        capability_id="python_repo_debugging_v1", sample_id="i1",
        model_revision="rev", prompt="p", group="control",
        outcome=TraceOutcome(success=True),
        internal=InternalRecord(layers={"0": {"usage_mass": 1.0}}),
    ))
    code = main(["footprint", "--spec", _write_spec(tmp_path),
                 "--workspace", str(tmp_path / "ws")])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "MIXED" in payload["error"]
    assert "behavioural_remote" in payload["error"]
    assert "internal_open_weight" in payload["error"]
    # no footprint artifact was written from contaminated evidence
    assert ws.list("footprints") == []


def test_cli_footprint_refuses_foreign_capability_traces(tmp_path, capsys):
    """Defect fix (C1): traces recorded under a DIFFERENT capability id
    used to be silently aggregated into this spec's footprint. A footprint
    binds to one spec; foreign evidence is refused by name."""
    from asea.capability_build.__main__ import main

    ws = CapabilityStore(tmp_path / "ws")
    _store_trace(ws, make_behavioural_trace(
        capability_id="python_repo_debugging_v1", sample_id="b1",
        model_revision="rev", prompt="p", group="target",
        outcome=TraceOutcome(success=True),
        behavioural=BehaviouralRecord(prompt="p", response="r"),
    ))
    _store_trace(ws, make_behavioural_trace(
        capability_id="another_capability_entirely", sample_id="x1",
        model_revision="rev", prompt="p", group="target",
        outcome=TraceOutcome(success=True),
        behavioural=BehaviouralRecord(prompt="p", response="r"),
    ))
    code = main(["footprint", "--spec", _write_spec(tmp_path),
                 "--workspace", str(tmp_path / "ws")])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "another_capability_entirely" in payload["error"]
    assert ws.list("footprints") == []


def test_functional_judge_stub_refuses_replay():
    """Defect fix (F5): the old ``functional_judge`` replayed pre-judged
    verdicts and was documented as THE intervention judge. A replay judge
    scores base and masked arms identically, so every intervention would
    report target_drop == 0 -- fabricated non-causality. The library
    boundary must refuse it with the reason named."""
    from asea.capability_build.evaluation import functional_judge

    with pytest.raises(InterventionInvalid) as excinfo:
        functional_judge({"expected_verdict": True}, {"phase": "base"})
    assert "regenerate" in str(excinfo.value)


def test_intervention_judge_receives_state_note_per_case():
    """Defect fix (F5/state contract): the judge MUST be told WHICH teacher
    state it is scoring ({"phase": "base"|"masked", "masked_component":
    key-or-None}), a fresh dict per case -- a regenerating judge needs this
    to score the masked arm differently; mutating one shared dict would
    let a sloppy judge contaminate the base arm."""
    adapter = FakeAdapter()
    seen = []

    def judge(case, note):
        seen.append(dict(note))
        return True

    run_intervention(
        adapter,
        InterventionTarget("expert", 2, 3),
        target_cases=_cases(3),
        control_cases=_cases(2, group="control"),
        judge=judge,
        seed=5,
    )
    # every case in both groups was scored in BOTH phases, with the note
    # naming the phase and (when masked) the exact component
    assert len(seen) == 10
    assert all(n["phase"] in ("base", "masked") for n in seen)
    assert all(
        n["masked_component"] == ("expert:2/3" if n["phase"] == "masked" else None)
        for n in seen
    )
    # fresh dicts: no shared mutable state between judge calls
    assert len({id(n) for n in seen}) == 10


class MaskedLeakAdapter(FakeAdapter):
    """Stands in for a teacher whose previous intervention's mask was never
    restored (a leaked hook or a crashed run): parameter hashes verify
    clean, but the mask registry is not empty."""

    def verify_unchanged(self):
        result = super().verify_unchanged()
        result["active_masks"] = ["expert:0/1"]
        return result


def test_intervention_refuses_masked_teacher_baseline():
    """Defect fix (F6): parameter hashes cannot see suppression hooks. A
    still-masked teacher used to verify as clean and serve as the BASELINE
    for a new intervention -- every measurement under it contaminated. The
    adapter's active-mask registry must abort the run by name."""
    adapter = MaskedLeakAdapter()
    with pytest.raises(InterventionInvalid) as excinfo:
        run_intervention(
            adapter,
            InterventionTarget("expert", 3, 7),
            target_cases=_cases(2),
            control_cases=_cases(2, group="control"),
            judge=_judge,
            seed=0,
        )
    assert "active mask" in str(excinfo.value)
    assert "expert:0/1" in str(excinfo.value)
    assert adapter.masked is None  # nothing was touched


def test_glm_verify_reports_active_masks_and_hooks_are_idempotent():
    """Defect fix (F6, adapter): (a) ``verify_unchanged`` reports
    ``active_masks`` so hook contamination is visible to parameter hashes;
    (b) re-registering router hooks no longer leaks the previous
    registration's live handles (the old code reassigned the handle list,
    leaving hooks whose entry-pointers wrote into a dict nobody read)."""
    torch = pytest.importorskip("torch")
    adapter = _fake_glm_router_stack(torch)
    # (a) clean teacher: no active masks reported
    assert adapter.verify_unchanged()["active_masks"] == []
    adapter.temporary_mask(InterventionTarget("expert", 0, 4))
    # registry keys are layer-prefixed ("<true layer index>:<target.key>");
    # layers[0] is the first SPARSE layer, layer 3 of the real stack
    assert adapter.verify_unchanged()["active_masks"] == ["3:expert:0/4"]
    adapter.restore_mask(InterventionTarget("expert", 0, 4))
    assert adapter.verify_unchanged()["active_masks"] == []

    # (b) hook idempotence: register twice, ONE forward pass, the collected
    # token count must reflect the model ONCE, not double-counted by a
    # leaked first registration.
    _, block = adapter.layers[0]
    router = adapter._router_of(block)[1]
    hidden = torch.ones(3, 4096)  # the faithful replica's hidden width
    adapter.register_router_hooks()
    with torch.no_grad():
        router(hidden)
    first = adapter.collect_routing(reset=True)
    adapter.register_router_hooks()  # re-register: old handles must be gone
    with torch.no_grad():
        router(hidden)
    second = adapter.collect_routing(reset=True)
    # keys are TRUE layer indices: layers[0] is sparse layer 3 of 45
    assert first["3"]["tokens"] == 3
    assert second["3"]["tokens"] == 3  # not 6: no stale duplicate hook
    adapter.remove_router_hooks()
    assert adapter.remove_router_hooks() is False  # idempotent, nothing left


def test_glm_restore_failure_keeps_mask_active():
    """Defect fix (F6, adapter): a restore whose handle cannot be removed
    must keep the mask record ACTIVE -- the mask still suppresses the
    expert, so the teacher must never verify as clean. The old code popped
    the registry entry before removing the hook: a failed remove left the
    hook live with no record of it."""
    torch = pytest.importorskip("torch")
    adapter = _fake_glm_router_stack(torch)
    target = InterventionTarget("expert", 0, 9)
    adapter.temporary_mask(target)

    class _BrokenHandle:
        def remove(self):
            raise RuntimeError("hook removal failed")

    # sabotage exactly the failure the fix guards: the handle cannot be
    # removed (the hook stays live on the router). The registry key is
    # layer-prefixed ("<true layer index>:<target.key>").
    adapter._masked["3:expert:0/9"]["handle"] = _BrokenHandle()
    with pytest.raises(Exception):
        adapter.restore_mask(target)
    # the mask is still ACTIVE and reported as such -- never "restored"
    assert adapter.verify_unchanged()["active_masks"] == ["3:expert:0/9"]
    # and a second mask of the same component is refused (already masked)
    with pytest.raises(InterventionInvalid):
        adapter.temporary_mask(target)


def test_ollama_transport_ignores_environment_proxies(monkeypatch):
    """Defect fix (F7): ``urllib.request.build_opener`` installs a
    ProxyHandler from the environment by DEFAULT -- an HTTP_PROXY pointing
    at an attacker-controlled (or merely broken) hop silently routed these
    "direct to this URL" requests through it. The transport must carry NO
    proxy handler even when the environment sets one. REAL socket test
    against a loopback HTTP server, not a mock."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    reached = {"handler": None}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            reached["handler"] = self.path
            body = b"direct-ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        # every proxy env var points at a dead port: honoring ANY of them
        # fails the request; the fix must bypass them all
        for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY",
                    "https_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.setenv(var, "http://127.0.0.1:9/")
        import urllib.request

        from asea.modules.real.ollama import urlopen_no_redirect

        request = urllib.request.Request("http://127.0.0.1:%d/api/tags" % port)
        with urlopen_no_redirect(request, timeout=10) as response:
            assert response.read() == b"direct-ok"
        # the request reached the loopback server DIRECTLY, not a proxy hop
        assert reached["handler"] == "/api/tags"
    finally:
        server.shutdown()
        server.server_close()


def test_cli_evaluate_blocked_reports_exit_2(tmp_path, capsys):
    """Defect fix (F8): a BLOCKED oracle run used to return ok:true with
    ``status: "BLOCKED_RESOURCE"`` -- a blocked evaluation reported as a
    successful command the operator had to notice by eye. The blocked
    outcome must go through the exit-2 discipline: ok:false, status
    BLOCKED_RESOURCE, requirement + remedy present."""
    import platform as _platform

    if _platform.system() == "Linux":
        pytest.skip("on Linux the real sandbox runs; the blocked path is "
                    "exercised by the non-Linux hosts and CI matrix")
    from asea.capability_build.__main__ import main

    source = tmp_path / "candidate.py"
    source.write_text("def f():\n    return 42\n", encoding="utf-8")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {"id": "t1", "group": "target", "function": "f",
         "args": [], "kwargs": {}, "expected": 42},
    ]), encoding="utf-8")
    code = main(["evaluate", "--source", str(source),
                "--cases", str(cases), "--workspace", str(tmp_path / "ws")])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["status"] == BLOCKED_RESOURCE
    assert payload["requirement"]
    assert payload["remedy"]
    assert "sandbox" in payload["remedy"] or "WSL2" in payload["remedy"]


def test_trace_schema_forbids_platform_illegal_sample_ids(tmp_path):
    """Defect fix (F2): the sample-id pattern allowed ':' (and the id is
    embedded verbatim in trace artifact filenames, ``<cap>-<sample_id>``),
    so a ':' would write fine on Linux and make the workspace
    un-checkoutable on Windows. The schema, the case loader and the store
    all refuse it."""
    import pydantic

    with pytest.raises(pydantic.ValidationError) as excinfo:
        make_behavioural_trace(
            capability_id="cap",
            sample_id="s:1",
            model_revision="rev",
            prompt="p",
            group="target",
            outcome=TraceOutcome(success=True),
            behavioural=BehaviouralRecord(prompt="p", response="r"),
        )
    assert "sample_id" in str(excinfo.value)

    # the CLI's case loader refuses the same before any artifact name is
    # built from it
    from asea.capability_build.__main__ import _load_cases

    case_file = tmp_path / "cases.json"
    case_file.write_text(json.dumps([
        {"sample_id": "s:1", "group": "target", "prompt": "p"},
    ]), encoding="utf-8")
    with pytest.raises(CapabilityBuildError) as excinfo:
        _load_cases(str(case_file))
    assert "portable" in str(excinfo.value)

    # the store refuses the full Windows-illegal set for artifact names
    store = CapabilityStore(tmp_path / "ws")
    for name in ("a:b", 'q"uote', "star*", "less<more", "pipe|",
                 "trailing.", " leading", "CON", "com1", "aux.txt"):
        with pytest.raises(CapabilityBuildError):
            store.put("specs", name, {"a": 1})
    # legal names still pass (a regression guard against over-refusal)
    store.put("specs", "good-name.2", {"a": 1})


def test_store_get_fails_closed_without_integrity_marker(tmp_path):
    """Defect fix (store honesty): ``get`` used to silently ACCEPT an
    artifact whose artifact_sha256 marker was absent -- a foreign or
    tampered file trusting itself. Every put-written artifact carries the
    marker; its absence is an integrity failure."""
    store = CapabilityStore(tmp_path / "ws")
    store.put("specs", "one", {"a": 1})
    raw = tmp_path / "ws" / "specs" / "one.json"
    payload = json.loads(raw.read_text(encoding="utf-8"))
    del payload["artifact_sha256"]
    raw.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CapabilityBuildError) as excinfo:
        store.get("specs", "one")
    assert "integrity marker" in str(excinfo.value)


def test_cli_receipt_dematerialises_not_measured_tokens(tmp_path, capsys):
    """Defect fix (F3): the receipt input file is the MATERIALISED (signed,
    on-disk) form -- unmeasured fields carry the literal NOT_MEASURED token.
    The CLI used to validate that form against the strict schema directly,
    so every honest receipt with unmeasured resources failed validation.
    It must dematerialise first (the receipt.py verify contract) and
    re-sign fresh for THIS workspace without trusting the input signature."""
    from asea.capability_build.__main__ import main

    sign_ws = tmp_path / "sign-ws"
    signed = sign_receipt(sign_ws, _receipt_object())
    assert signed["capability_retention"] == NOT_MEASURED
    receipt_file = tmp_path / "receipt.json"
    receipt_file.write_text(json.dumps(signed), encoding="utf-8")

    store_ws = tmp_path / "store-ws"
    code = main(["receipt", "--receipt", str(receipt_file),
                "--workspace", str(store_ws)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    # the stored artifact verifies against the NEW workspace's key
    stored = json.loads(Path(payload["artifact"]["path"]).read_text(encoding="utf-8"))
    verified = verify_receipt(store_ws, stored)
    assert verified["valid"] is True


def test_dataset_refuses_shapeless_prompts(spec):
    """Defect fix (F9): a prompt whose token shape is EMPTY (punctuation-
    only) silently bypassed the near-duplicate leakage guard -- the shape
    is the guard's key, and an all-punctuation prompt collides with
    nothing. Such a case cannot be guarded, so it is refused at build time
    (dataset) and at teaching time (distillation)."""
    from asea.capability_build.dataset import validate_case

    with pytest.raises(DatasetInvalid) as excinfo:
        validate_case({
            "sample_id": "s1", "split": "training", "group": "target",
            "prompt": "???!", "family_id": "f", "provenance": "p",
            "license": "MIT",
        })
    assert "token shape" in str(excinfo.value)

    from asea.capability_build.distillation import build_sequence_pairs

    with pytest.raises(CapabilityBuildError) as excinfo:
        build_sequence_pairs([
            _judged_behavioural_trace("t-1", "!!!???", "r1", True),
        ])
    assert "token shape" in str(excinfo.value)


def test_dataset_build_is_atomic_no_staging_leftover(tmp_path, spec):
    """Defect fix: the build used to write directly into the output dir; a
    crash mid-build left a corpse that looked like a dataset. The build
    now stages in a sibling and renames once -- the output either does not
    exist or IS the complete frozen build, and no staging leftover
    remains."""
    from asea.capability_build.dataset import build_dataset, validate_dataset

    def _case(sid, split, family):
        return {
            "sample_id": sid, "split": split, "group": "target",
            "prompt": "fix the bug numbered %s" % sid,
            "expected": "patch for %s" % sid,
            "family_id": family, "provenance": "synthetic", "license": "MIT",
        }

    cases = (
        [_case("t%d" % i, "training", "fam-t%d" % i) for i in range(4)]
        + [_case("d%d" % i, "development", "fam-d%d" % i) for i in range(2)]
        + [_case("h%d" % i, "heldout", "fam-h%d" % i) for i in range(2)]
        + [_case("f%d" % i, "final", "fam-f%d" % i) for i in range(2)]
        + [_case("c%d" % i, "controls", "fam-c%d" % i) for i in range(2)]
    )
    out = tmp_path / "dataset"
    build_dataset(cases, output_dir=out, spec=spec)
    # the complete frozen build appeared; no staging sibling remains
    validate_dataset(out)
    siblings = [p.name for p in tmp_path.iterdir() if "building-" in p.name]
    assert siblings == []

    # (b) the crash property the atomic build exists for: a build that
    # dies mid-write must leave NO output directory -- the old code wrote
    # directly into output_dir, so its corpse looked exactly like a
    # dataset minus its selection lock. A case carrying a value that
    # validates but cannot be JSON-serialised dies inside the first split
    # write, after the directory was created.
    crash = copy.deepcopy(cases)
    crash[0]["unserialisable"] = {1, 2, 3}  # a set: json.dumps raises
    out2 = tmp_path / "corpse"
    with pytest.raises(TypeError):
        build_dataset(crash, output_dir=out2, spec=spec)
    assert not out2.exists()  # the old code left a half-written dataset here
    # and the leftover staging dir is named as a crash leftover, never as
    # the dataset
    leftovers = [p.name for p in tmp_path.iterdir()
                 if "building-" in p.name and p.is_dir()]
    assert all(name.startswith("corpse.building-") for name in leftovers)


def test_worker_request_ids_are_monotonic_not_id_of_payload():
    """Defect fix: request ids were built from ``id(payload)`` -- a memory
    address. Two requests sharing one payload object silently reused the
    same id, defeating the echo check between interleaved consumers. The
    ids are now a process-lifetime monotonic counter."""
    first = proto.make_request("hello", {"x": 1})
    # the EXACT object the old id()-based scheme would have collided on
    payload = {"x": 1}
    second = proto.make_request("hello", payload)
    third = proto.make_request("hello", payload)
    ids = [first["id"], second["id"], third["id"]]
    assert len(set(ids)) == 3  # id()-based scheme: second == third here
    # monotonic: each op's counter suffix strictly increases
    def _suffix(frame_id):
        return int(frame_id.rsplit("-", 1)[1])
    assert _suffix(second["id"]) > _suffix(first["id"])
    assert _suffix(third["id"]) > _suffix(second["id"])


def test_student_baseline_judge_abstention_is_not_a_failure():
    """Defect fix: an abstaining judge (None) used to be coerced to False,
    folding ungradable cases into the failures and manufacturing a lower
    pass rate no verdict ever issued. Abstentions are excluded from the
    rate and counted separately."""
    from asea.capability_build.student import measure_student_baseline

    student = {"kind": "ollama_local", "model": "toy"}
    cases = [
        {"sample_id": "a", "group": "target", "prompt": "p1"},
        {"sample_id": "b", "group": "target", "prompt": "p2"},
        {"sample_id": "c", "group": "control", "prompt": "p3"},
    ]

    def infer(_student, case):
        return "answer-for-%s" % case["sample_id"]

    def judge(case, output):
        if case["sample_id"] == "b":
            return None  # ABSTAIN: no expected output for this case
        return output == "answer-for-%s" % case["sample_id"]

    summary = measure_student_baseline(student, cases, infer=infer, judge=judge)
    assert summary["groups"]["target"] == {"passed": 1, "total": 1, "pass_rate": 1.0}
    assert summary["unjudged_excluded"] == 1
    unjudged_rows = [row for row in summary["per_case"] if row.get("unjudged")]
    assert len(unjudged_rows) == 1 and unjudged_rows[0]["sample_id"] == "b"


def test_worker_client_respawns_after_deadline_kill(tmp_path):
    """Defect fix: after the watchdog killed the worker, the dead process
    handle stayed registered, so the next request wrote to a corpse and
    surfaced a generic WorkerCrashed. The client must reap the dead
    handle and start a FRESH worker for the next request."""
    from asea.capability_build.worker import GlmWorkerClient

    hanging = tmp_path / "hang.py"
    hanging.write_text("import time\ntime.sleep(600)\n", encoding="utf-8")
    answering = tmp_path / "answer.py"
    answering.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    if not line.strip():\n"
        "        continue\n"
        "    request = json.loads(line)\n"
        "    sys.stdout.write(json.dumps({\n"
        "        'protocol': %r, 'protocol_version': 1,\n"
        "        'op': request['op'], 'id': request['id'], 'ok': True,\n"
        "        'result': {'fresh': True},\n"
        "    }) + '\\n')\n"
        "    sys.stdout.flush()\n" % proto.PROTOCOL,
        encoding="utf-8",
    )
    client = GlmWorkerClient(
        repo_root=tmp_path, checkpoint=str(tmp_path), use_docker=False,
        python=sys.executable, timeout=3600.0,
    )
    client._command = lambda: [sys.executable, str(hanging)]
    with pytest.raises(Exception):
        client.request("hello", timeout=0.5)
    # the deadline killed the worker; the NEXT request must respawn a
    # working one instead of writing to the corpse
    client._command = lambda: [sys.executable, str(answering)]
    # request() returns the RESULT dict directly (unwrapped), so the
    # fresh-worker marker lives at the top level
    assert client.request("hello")["fresh"] is True
    client.stop()


def test_worker_unknown_op_reports_invalid_op_kind(tmp_path):
    """Defect fix: an unknown op used to surface as ``arch_mismatch``
    (it raised InterventionInvalid, and the worker mapped the whole class
    to that kind). The protocol defines ``invalid_op`` for this; the error
    kind must not lie. Runs the REAL worker script with a dummy checkpoint
    env: the unknown-op path builds its response before any handler or
    model load, so no GLM runtime is touched."""
    import os
    import subprocess as _sp

    worker = Path(__file__).resolve().parents[1] / "workers" / "glm53" / "worker.py"
    if not worker.is_file():
        pytest.skip("worker script not present in this checkout")
    env = dict(os.environ)
    env["GLM_CHECKPOINT"] = str(tmp_path / "dummy-checkpoint")
    frame = proto.encode({
        "protocol": proto.PROTOCOL, "protocol_version": proto.PROTOCOL_VERSION,
        "op": "definitely_not_an_op", "id": "x1", "payload": {},
    })
    proc = _sp.run(
        [sys.executable, str(worker)], input=frame, capture_output=True,
        env=env, timeout=60,
    )
    response = json.loads(proc.stdout.decode("utf-8").strip())
    assert response["ok"] is False
    assert response["error"]["kind"] == "invalid_op"
    assert "definitely_not_an_op" in response["error"]["message"]


def test_glm_telemetry_captures_real_bias_corrected_dispatch():
    """Defect fix (external audit 2026-09-19, replaces the old
    bias-REFUSAL test): the real transformers 5.16.1
    ``Glm5NextTextTopkRouter`` ALWAYS carries an ``e_score_correction_bias``
    buffer, and the GLM config declares no ``scoring_func`` field at all.
    The old adapter refused a config boolean named
    ``e_score_correction_bias`` -- a field the real config never has -- so
    it would refuse the REAL model while claiming a sigmoid-topk equation
    that was never in effect. The corrected adapter accepts the biased
    router and records the ACTUAL dispatched experts from the router's own
    ``(logits, topk_weights, topk_indices)`` output: usage evidence under
    the equation truly in effect, never a re-derived plain-sigmoid
    approximation. MECHANISM test on the exact-shape faithful-replica
    stack."""
    torch = pytest.importorskip("torch")
    adapter = _fake_glm_router_stack(torch)
    _, block = adapter.layers[0]
    router = adapter._router_of(block)[1]
    hidden = torch.ones(3, 4096)
    # bias expert 17 into every token's dispatch: plain sigmoid-mass
    # telemetry would not see this as usage, the real dispatch does
    router.e_score_correction_bias[17] = 5.0
    adapter.register_router_hooks()
    with torch.no_grad():
        _, weights, indices = router(hidden)  # the hook fires on this pass
    collected = adapter.collect_routing(reset=True)["3"]
    assert collected["correction_bias_nonzero"] is True
    assert collected["tokens"] == 3
    # dispatched_count is the ACTUAL dispatch: total == tokens x top-8 ...
    assert sum(collected["dispatched_count"]) == 3 * 8
    # ... expert 17 really dispatched under the bias ...
    assert collected["dispatched_count"][17] > 0
    # ... and the recorded dispatch equals a manual recount of the router's
    # own topk ids (the hook captured the real tuple, not a re-derivation)
    manual = torch.bincount(indices.reshape(-1).long(), minlength=288).tolist()
    assert collected["dispatched_count"] == manual
    # the recorded weight sums match the router's own topk_weights
    manual_weights = torch.zeros(288).scatter_add(
        0, indices.reshape(-1).long(), weights.reshape(-1).float()
    ).tolist()
    assert collected["dispatched_weight_sum"] == pytest.approx(manual_weights, abs=1e-6)
    # the selection parameters actually in effect ride along, so a reader
    # never has to guess which equation produced the numbers
    assert collected["router_params"]["num_group"] == 4
    assert collected["router_params"]["topk_group"] == 2
    assert collected["router_params"]["norm_topk_prob"] is True
    assert collected["router_params"]["routed_scaling_factor"] == 2.5
    assert collected["usage_evidence_not_causal_importance"] is True
    router.e_score_correction_bias[17] = 0.0
    adapter.remove_router_hooks()


def test_glm_telemetry_and_mask_refuse_to_be_live_together():
    """The telemetry hook and the mask hook both sit on the same router
    module: if telemetry is registered FIRST, it records the router's
    UNMASKED dispatch during a masked measurement -- contaminated usage
    evidence that looks exactly like clean data. Both directions are
    refused (external audit 2026-09-19)."""
    torch = pytest.importorskip("torch")
    adapter = _fake_glm_router_stack(torch)
    adapter.register_router_hooks()
    with pytest.raises(InterventionInvalid) as excinfo:
        adapter.temporary_mask(InterventionTarget("expert", 0, 5))
    assert "telemetry" in str(excinfo.value)
    adapter.remove_router_hooks()
    adapter.temporary_mask(InterventionTarget("expert", 0, 5))
    with pytest.raises(InterventionInvalid) as excinfo:
        adapter.register_router_hooks()
    assert "masks are active" in str(excinfo.value)
    adapter.restore_mask(InterventionTarget("expert", 0, 5))
    # with both sides clean, registration works again
    adapter.register_router_hooks()
    adapter.remove_router_hooks()


def test_glm_hook_names_non_tensor_router_output():
    """A transformers revision changing the router's return contract must
    be refused by NAME (layer, router, actual type), never surfaced as an
    opaque AttributeError deep inside the forward pass. The v5.16.1
    contract is the 3-tuple ``(router_logits, topk_weights,
    topk_indices)``; both a non-tuple return and a non-tensor INSIDE the
    tuple are typed detection failures."""
    torch = pytest.importorskip("torch")
    adapter = _fake_glm_router_stack(torch)
    adapter.register_router_hooks()
    _, block = adapter.layers[0]
    router = adapter._router_of(block)[1]
    original_forward = router.forward

    def broken_forward(x):
        return {"logits": original_forward(x)}  # a dict, not the tuple

    router.forward = broken_forward
    with pytest.raises(InterventionInvalid) as excinfo:
        with torch.no_grad():
            router(torch.ones(2, 4096))
    message = str(excinfo.value)
    # the layer named is the TRUE layer index (layers[0] is sparse layer 3)
    assert "layer 3" in message
    assert "dict" in message
    assert "tuple" in message
    adapter.remove_router_hooks()

    # a tuple with a non-tensor inside is the SAME class of contract break
    adapter.register_router_hooks()
    router = adapter._router_of(adapter.layers[0][1])[1]

    def half_broken(x):
        logits, weights, indices = original_forward(x)
        return logits, weights.tolist(), indices  # weights as a plain list

    router.forward = half_broken
    with pytest.raises(InterventionInvalid) as excinfo:
        with torch.no_grad():
            router(torch.ones(2, 4096))
    message = str(excinfo.value)
    assert "layer 3" in message
    assert "topk_weights" in message
    assert "list" in message
    adapter.remove_router_hooks()
