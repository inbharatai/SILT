"""Capability-build MECHANISM tests.

Every test here exercises bookkeeping mechanics offline: schema validation,
consent gating, evidence-class separation, store/receipt integrity, the
intervention protocol on a fake adapter, and worker IPC. A mock trace or
fake adapter is a MECHANISM TEST, not a GLM capability result -- no test in
this file says anything about any teacher's behaviour.
"""

from __future__ import annotations

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
        spec_sha256="0" * 64,
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
    45 decoder blocks (first 3 dense, 42 sparse with a 288-expert router
    each), a vision tower module, and the config fields the adapter pins.
    Random weights -- mechanics only."""
    from asea.capability_build.adapters.glm53_flash import Glm53FlashAdapter

    class Glm5NextForCausalLM(torch.nn.Module):  # accepted class name
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList()
            for index in range(45):
                block = torch.nn.Module()
                if index < 3:
                    block.mlp = torch.nn.Linear(8, 8)  # dense: no "experts"
                else:
                    block.experts = torch.nn.Module()
                    block.experts.router = torch.nn.Linear(8, 288)
                self.model.layers.append(block)
            self.vision_tower = torch.nn.Linear(8, 8)

    class Cfg:
        pass

    cfg = Cfg()
    cfg.model_type = "glm5_next"
    cfg.num_hidden_layers = 45
    cfg.n_routed_experts = 288
    cfg.n_shared_experts = 1
    cfg.num_experts_per_tok = 8
    cfg.scoring_func = "sigmoid"
    cfg.first_k_dense_replace = 3
    cfg.hidden_size = 4096
    cfg.max_position_embeddings = 1_048_576
    cfg.torch_dtype = "bfloat16"
    return Glm53FlashAdapter(Glm5NextForCausalLM(), torch, config=cfg)


def test_glm_mask_suppresses_expert_and_restore_succeeds():
    """Defect fix: the mask is a router-OUTPUT forward hook, not a
    weight-row rewrite. The old ``-30`` row made the masked expert's
    logit ``-30 * sum(hidden)`` -- STRONGLY POSITIVE for negative-sum
    hidden states, i.e. it INCREASED the selection probability. The hook
    gives ``-1e9`` for every token, input-independent, and touches no
    weights: the teacher verifies hash-unchanged even while masked."""
    torch = pytest.importorskip("torch")
    adapter = _fake_glm_router_stack(torch)
    target = InterventionTarget("expert", 0, 17)
    _, block = adapter.layers[0]
    router = adapter._router_of(block)[1]
    hidden = -torch.ones(3, 8)  # NEGATIVE-sum hidden states
    # the OLD mechanism would have produced a POSITIVE logit on this input
    assert -30.0 * float(hidden.sum(dim=-1)[0]) > 0
    adapter.freeze_baseline()
    with torch.no_grad():
        baseline = router(hidden).clone()
    result = adapter.temporary_mask(target)
    assert result["weights_modified"] is False
    assert result["mechanism"] == "router_output_forward_hook"
    with torch.no_grad():
        masked = router(hidden)
        assert torch.all(masked[:, 17] == -1.0e9)
        others = [e for e in range(288) if e != 17]
        assert torch.equal(masked[:, others], baseline[:, others])
        # the masked expert can never enter the top-8 selection
        top = torch.sigmoid(masked).topk(8, dim=-1).indices
        assert 17 not in top.reshape(-1).tolist()
    # weights were never modified: unchanged holds DURING the mask window
    assert adapter.verify_unchanged()["unchanged"] is True
    restored = adapter.restore_mask(target)
    assert restored["restored"] is True
    assert restored["weights_modified"] is False
    with torch.no_grad():
        assert torch.equal(router(hidden), baseline)
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

    code = main(["intervene", "--spec", _write_spec(tmp_path),
                 "--workspace", str(tmp_path / "ws"),
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
    assert result["judged_cases"] == 3
    assert result["passed_cases"] == 2
    assert result["groups"]["target"] == {"passed": 1, "total": 2, "pass_rate": 0.5}
    assert result["groups"]["control"] == {"passed": 1, "total": 1, "pass_rate": 1.0}


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
