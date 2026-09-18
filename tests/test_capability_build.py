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
def test_search_accepts_first_candidate_meeting_threshold(spec):
    from asea.capability_build.search import search_with_reference

    plans = [{"keep": 10}, {"keep": 4}, {"keep": 2}]

    def build(plan):
        return plan

    def evaluate_by_plan(candidate):
        keep = candidate["keep"]
        if keep == 10:
            return {"target_score": 1.0, "control_scores": {"math": 1.0},
                    "size_bytes": 100}
        if keep == 4:
            return {"target_score": 0.9, "control_scores": {"math": 1.0},
                    "size_bytes": 60}
        return {"target_score": 0.5, "control_scores": {"math": 0.99},
                "size_bytes": 30}

    result = search_with_reference(
        spec, teacher_target_score=1.0, candidate_plan=plans,
        build=build, evaluate=evaluate_by_plan,
    )
    # threshold = 0.8 * 1.0; keep=4 already meets it (0.9 >= 0.8), so the
    # LARGER keep=10 is accepted first (order matters, not optimality).
    assert result["accepted"]["plan"]["keep"] == 10
    assert result["accepted"]["admission"] == "CANDIDATE_UNADMITTED"
    assert result["autoactivated"] is False


def test_search_rejects_candidate_below_threshold(spec):
    from asea.capability_build.search import search_with_reference

    result = search_with_reference(
        spec, teacher_target_score=1.0, candidate_plan=[{"keep": 2}],
        build=lambda p: p,
        evaluate=lambda c: {"target_score": 0.5, "control_scores": {},
                           "size_bytes": 10},
    )
    assert result["accepted"] is None
    assert result["iterations"][0]["status"] == "rejected"
    assert result["failed_experiments_visible"] is True


def test_search_records_failed_builders_and_never_fabricates(spec):
    from asea.capability_build.search import search_with_reference

    def boom(_plan):
        raise RuntimeError("OOM")

    result = search_with_reference(
        spec, teacher_target_score=1.0, candidate_plan=[{"keep": 1}],
        build=boom, evaluate=lambda c: {},
    )
    assert result["iterations"][0]["status"] == "failed"
    assert "OOM" in result["iterations"][0]["error"]
    assert result["accepted"] is None


def test_search_refuses_unmeasured_teacher_and_empty_plan(spec):
    from asea.capability_build.search import search_with_reference

    with pytest.raises(CapabilityBuildError):
        search_with_reference(spec, teacher_target_score=0.0,
                              candidate_plan=[{"keep": 1}],
                              build=lambda p: p, evaluate=lambda c: {})
    with pytest.raises(CapabilityBuildError):
        search_with_reference(spec, teacher_target_score=1.0,
                              candidate_plan=[], build=lambda p: p,
                              evaluate=lambda c: {})


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
    assert "not local" in payload["requirement"]


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


def test_cli_distill_needs_traces(tmp_path, capsys):
    from asea.capability_build.__main__ import main

    code = main(["distill", "--spec", _write_spec(tmp_path),
                 "--workspace", str(tmp_path / "ws")])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == BLOCKED_RESOURCE
    assert "no traces" in payload["requirement"]

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


def test_dataset_build_and_validate_round_trip(tmp_path):
    from asea.capability_build.dataset import build_dataset, validate_dataset

    out = tmp_path / "capability_v1"
    manifest = build_dataset(_full_case_list(), output_dir=out)
    assert manifest["frozen_before_model_generation"] is True
    assert manifest["final_opened"] is False
    assert manifest["counts"]["training"] == 3
    result = validate_dataset(out)
    assert result["ok"] is True
    assert (out / "selection-lock.json").is_file()


def test_dataset_refuses_id_family_and_nearduplicate_leakage(tmp_path):
    from asea.capability_build.dataset import build_dataset

    cases = _full_case_list()
    # same sample_id in two splits
    clash = _case("training-0", "development", family="other-fam")
    with pytest.raises(DatasetInvalid):
        build_dataset(cases + [clash], output_dir=tmp_path / "a")
    # family crossing splits
    cross = _case("new-id", "heldout", family="fam-training-0")
    with pytest.raises(DatasetInvalid):
        build_dataset(cases + [cross], output_dir=tmp_path / "b")
    # near-duplicate prompt shape across splits (new guard)
    near = _case("near-id", "heldout", family="near-fam",
                 prompt="DISTINCT prompt BODY for training-0")
    with pytest.raises(DatasetInvalid):
        build_dataset(cases + [near], output_dir=tmp_path / "c")
    # empty split
    only = [_case("t-0", "training")]
    with pytest.raises(DatasetInvalid):
        build_dataset(only, output_dir=tmp_path / "d")


def test_dataset_quarantines_frozen_september_inputs(tmp_path):
    from asea.capability_build.dataset import read_cases

    frozen = tmp_path / "specialist-v1" / "final.jsonl"
    frozen.parent.mkdir()
    _write_cases(frozen.parent, [_case("x-0", "training")], name="final.jsonl")
    with pytest.raises(DatasetInvalid):
        read_cases(frozen)


def test_dataset_output_must_be_new(tmp_path):
    from asea.capability_build.dataset import build_dataset

    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(DatasetInvalid):
        build_dataset(_full_case_list(), output_dir=out)


def test_dataset_validate_detects_post_build_tampering(tmp_path):
    from asea.capability_build.dataset import build_dataset, validate_dataset

    out = tmp_path / "capability_v1"
    build_dataset(_full_case_list(), output_dir=out)
    victim = out / "training.jsonl"
    victim.write_text(victim.read_text(encoding="utf-8").replace(
        "ok-training-0", "edited"), encoding="utf-8")
    with pytest.raises(DatasetInvalid):
        validate_dataset(out)


def test_cli_dataset_build_and_validate(tmp_path, capsys):
    from asea.capability_build.__main__ import main

    cases = _write_cases(tmp_path, _full_case_list())
    out = tmp_path / "capability_v1"
    code = main(["dataset", "build", "--cases", str(cases), "--out", str(out)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["counts"]["final"] == 3
    code = main(["dataset", "validate", "--dir", str(out)])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_cli_dataset_build_refuses_leaky_cases(tmp_path, capsys):
    from asea.capability_build.__main__ import main

    cases = _write_cases(tmp_path, [_case("t-0", "training")])
    code = main(["dataset", "build", "--cases", str(cases),
                 "--out", str(tmp_path / "v1")])
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
