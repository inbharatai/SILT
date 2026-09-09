"""Certification regression harnesses, NOT trained-model quality evidence.

Positive admission tests mock adapters solely to exercise control flow. The tiny
random GPT2 negative test uses real inference and must never be called quality.
"""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from asea.artifacts import Blocked, Workspace, atomic_json, digest
import asea.certification as cert
from asea.compose.schema import CompositionSpec, EvaluationCase, EvaluationSuite, Limits
from asea.compose.__main__ import main
from asea.certification.sandbox import SandboxResult


TRUSTED_TESTS = "import unittest\nfrom candidate import f\nclass Checks(unittest.TestCase):\n    def test_f(self): self.assertEqual(f(), 1)\n"


def suite(metric="text_exact", threshold=1.0, **extra):
    return EvaluationSuite(name="Independent test literals", reference_source="unit harness only", cases=[
        EvaluationCase(id=g, group=g, input=g, reference=TRUSTED_TESTS if metric == "functional_code" else "correct", metric=metric, threshold=threshold)
        for g in ("target", "control")], **extra)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{}')
    (model / "model.safetensors").write_bytes(b"INVALID WEIGHTS: mock control-flow tests only")
    spec = CompositionSpec.model_validate({"name": "MOCK NOT INFERENCE", "input_type": "text", "nodes": [
        {"id": "text", "input": "$input", "component": {"kind": "hf_text", "task": "causal", "model_path": str(model),
         "risk": "high", "provenance": [{"source": "unit harness", "license": "test-only", "risk": "high"}]}}], "output_node": "text"})
    monkeypatch.setattr("asea.compose.runtime._adapter", lambda *a, **kw: {"type": "text", "text": "correct"})
    return Workspace(tmp_path / "workspace"), spec


@pytest.mark.parametrize("metric,threshold", [("text_exact", 0.0), ("text_exact", .99), ("functional_code", 0.0),
    ("functional_code", .9), ("text_similarity_proxy", .79), ("word_error_rate", .51),
    ("text_exact", float("nan")), ("word_error_rate", float("inf"))])
def test_hard_metric_floors(metric, threshold):
    with pytest.raises(ValidationError):
        suite(metric, threshold)
    assert not cert._passes(metric, 0.0, threshold)


@pytest.mark.parametrize("metric,threshold", [("text_exact", 1.0), ("functional_code", 1.0),
    ("text_similarity_proxy", .8), ("word_error_rate", .5), ("word_error_rate", 0.0)])
def test_valid_floors(metric, threshold):
    assert suite(metric, threshold)


@pytest.mark.parametrize("field", ["max_seconds", "max_peak_rss_mb"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_finite_limits(field, value):
    with pytest.raises(ValidationError):
        Limits(**{field: value})


@pytest.mark.parametrize("field", ["input", "reference"])
def test_whitespace_case_invalid(field):
    raw = suite().model_dump()
    raw["cases"][0][field] = "  "
    with pytest.raises(ValidationError):
        EvaluationSuite.model_validate(raw)


def test_wrong_output_evaluate_then_activate(harness, monkeypatch):
    ws, spec = harness
    monkeypatch.setattr("asea.compose.runtime._adapter", lambda *a, **kw: {"type": "text", "text": "wrong"})
    result = cert.evaluate(ws, spec, suite())
    assert result["status"] == "rejected"
    assert [c["score"] for c in result["case_evidence"]] == [0, 0]
    with pytest.raises(Blocked):
        cert.activate(ws, result["id"], approver="Reviewer")
    assert not (ws.root / "active.json").exists()


def test_construct_bypass_cannot_admit_zero(harness):
    ws, spec = harness
    valid = suite()
    invalid = valid.model_copy(update={"cases": [c.model_copy(update={"threshold": 0.0}) for c in valid.cases]})
    result = cert.evaluate(ws, spec, invalid)
    assert result["status"] == "blocked"
    with pytest.raises(Blocked):
        cert.activate(ws, result["id"], approver="Reviewer")


@pytest.mark.parametrize("claim", ["coding", "safety"])
def test_proxy_cannot_admit_capability_claim(harness, claim):
    ws, spec = harness
    result = cert.evaluate(ws, spec, suite("text_similarity_proxy", .8, claims=[claim]))
    assert result["status"] == "blocked"
    assert result["case_evidence"] == []


def test_named_highrisk_approver(harness):
    ws, spec = harness
    result = cert.evaluate(ws, spec, suite())
    assert result["status"] == "admitted"  # mocked adapter control flow only
    with pytest.raises(Blocked, match="named"):
        cert.activate(ws, result["id"], approve_high_risk=True)
    with pytest.raises(Blocked, match="nonempty"):
        cert.activate(ws, result["id"], approver="  ")
    deployment = cert.activate(ws, result["id"], approver="Test Reviewer")
    assert deployment["approver"] == "Test Reviewer"
    assert deployment["policy_hash"] == digest(cert.ADMISSION_POLICY)


def test_activation_recomputes_and_binds_policy(harness, monkeypatch):
    ws, spec = harness
    result = cert.evaluate(ws, spec, suite())
    original = ws.read_record
    def changed(collection, identifier):
        record = original(collection, identifier)
        if collection == "evaluations":
            record.pop("policy_hash")
        return record
    monkeypatch.setattr(ws, "read_record", changed)
    with pytest.raises(Blocked, match="policy"):
        cert.activate(ws, result["id"], approver="Reviewer")
    monkeypatch.setattr(ws, "read_record", original)
    monkeypatch.setattr(cert, "metric_score", lambda *a: 0.0)
    with pytest.raises(Blocked, match="evidence mismatch"):
        cert.activate(ws, result["id"], approver="Reviewer")
    assert not (ws.root / "active.json").exists()


@pytest.mark.parametrize("output", ["```python\npass\n```\n```python\npass\n```", "```js\npass\n```", "~~~python\npass\n~~~"])
def test_reject_ambiguous_code_markup(output):
    with pytest.raises(Blocked, match="preprocessing"):
        cert.preprocess_code(output)


def test_single_code_block_preprocessing():
    assert cert.preprocess_code("Explanation\n```python\ndef add(a,b): return a+b\n```\nDone") == "def add(a,b): return a+b"
    assert cert.preprocess_code("def f(): return 1") == "def f(): return 1"


def test_functional_per_case_and_activation_rerun(harness, monkeypatch):
    ws, spec = harness
    calls = []
    monkeypatch.setattr("asea.certification.sandbox.probe_code_sandbox", lambda: SandboxResult("PASSED", True, True))
    def code(source, tests):
        calls.append((source, tests))
        return SandboxResult("PASSED", True, True, tests_run=2)
    monkeypatch.setattr("asea.certification.sandbox.evaluate_code", code)
    monkeypatch.setattr("asea.compose.runtime._adapter", lambda *a, **kw: {"type": "text", "text": "```python\ndef f(): return 1\n```"})
    result = cert.evaluate(ws, spec, suite("functional_code", claims=["coding"]))
    assert result["status"] == "admitted"
    assert calls == [("def f(): return 1", TRUSTED_TESTS)] * 2
    cert.activate(ws, result["id"], approver="Reviewer")
    assert len(calls) == 4
    assert "not adversarial correctness proof" in result["scope"]
    monkeypatch.setattr("asea.certification.sandbox.evaluate_code", lambda *a: SandboxResult("BLOCKED", False, reason="disabled"))
    with pytest.raises(Blocked, match="sandbox unavailable"):
        cert.activate(ws, result["id"], approver="Reviewer")


def test_functional_sandbox_unavailable_before_inference(harness, monkeypatch):
    ws, spec = harness
    monkeypatch.setattr("asea.certification.sandbox.probe_code_sandbox", lambda: SandboxResult("BLOCKED", False, reason="no namespaces"))
    result = cert.evaluate(ws, spec, suite("functional_code"))
    assert result["status"] == "blocked"
    assert not list((ws.root / "runs").iterdir())


@pytest.mark.parametrize("event", ["activate_intent", "activate"])
@pytest.mark.parametrize("existing", [False, True])
def test_audit_failure_preserves_previous_pointer(harness, monkeypatch, event, existing):
    ws, spec = harness
    ev = cert.evaluate(ws, spec, suite())
    pointer = ws.root / "active.json"
    if existing:
        cert.activate(ws, ev["id"], approver="Previous")
    before = pointer.read_bytes() if existing else None
    audit = ws.audit
    def fail(name, **fields):
        if name == event:
            raise OSError("audit disk failure")
        return audit(name, **fields)
    monkeypatch.setattr(ws, "audit", fail)
    with pytest.raises(OSError, match="audit disk"):
        cert.activate(ws, ev["id"], approver="Next")
    assert (pointer.read_bytes() if pointer.exists() else None) == before
    assert not (ws.root / "active-transaction.json").exists()


def test_failed_restore_keeps_recovery_journal(harness, monkeypatch):
    ws, spec = harness
    ev = cert.evaluate(ws, spec, suite())
    original_audit, original_restore = ws.audit, cert._restore_pointer
    def fail_audit(event, **fields):
        if event == "activate": raise OSError("audit failure")
        return original_audit(event, **fields)
    def fail_restore(*a): raise OSError("restore failure")
    monkeypatch.setattr(ws, "audit", fail_audit)
    monkeypatch.setattr(cert, "_restore_pointer", fail_restore)
    with pytest.raises(OSError, match="restore failure"):
        cert.activate(ws, ev["id"], approver="Reviewer")
    assert (ws.root / "active-transaction.json").exists()
    monkeypatch.setattr(cert, "_restore_pointer", original_restore)
    monkeypatch.setattr(ws, "audit", original_audit)
    with ws.writer(): cert.recover_active(ws)
    assert not (ws.root / "active.json").exists()
    assert not (ws.root / "active-transaction.json").exists()


def test_selection_suite_exact_graph_disk_and_no_promotion(harness):
    ws, spec = harness
    ev = cert.evaluate(ws, spec, suite())
    args = (ws, [ev["id"]], "text", "text", 1000.0, 10000.0)
    assert cert.select_measured(*args)["status"] == "no_feasible_bundle"
    assert cert.select_measured(*args, suite_hash="0"*64)["status"] == "no_feasible_bundle"
    assert cert.select_measured(*args, suite_hash=ev["suite_hash"], graph_hash="0"*64)["status"] == "no_feasible_bundle"
    assert cert.select_measured(*args, suite_hash=ev["suite_hash"], max_disk_bytes=1)["status"] == "no_feasible_bundle"
    selected = cert.select_measured(*args, suite_hash=ev["suite_hash"], graph_hash=ev["graph_hash"], max_disk_bytes=10000)
    assert selected["status"] == "selected"
    assert selected["dependency_disk_bytes"] > 1
    assert selected["activated"] is False
    assert not (ws.root / "active.json").exists()
    with pytest.raises(Blocked): cert.select_measured(ws, [], "text", "text", float("nan"), 10.0)


def test_plan_and_select_cli(harness, tmp_path, capsys):
    ws, spec = harness
    path = tmp_path / "spec.json"
    path.write_text(spec.model_dump_json())
    assert main(["--workspace", str(ws.root), "plan", "--spec", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["result"]["status"] == "draft"
    ev = cert.evaluate(ws, spec, suite())
    assert main(["--workspace", str(ws.root), "select", "--evaluations", ev["id"], "--input-type", "text",
                 "--output-type", "text", "--max-wall-seconds", "1000", "--max-peak-rss-mb", "10000", "--suite-hash", ev["suite_hash"]]) == 0
    assert json.loads(capsys.readouterr().out)["result"]["status"] == "selected"
    assert not (ws.root / "active.json").exists()


def test_real_sandbox_metric_integration(harness, monkeypatch):
    # Adapter output is a fixture, but the OS sandbox and trusted tests are real.
    from asea.certification.sandbox import probe_code_sandbox
    probe = probe_code_sandbox()
    if not probe.supported or not probe.passed:
        pytest.skip("OS isolation unavailable: " + probe.reason)
    ws, spec = harness
    monkeypatch.setattr("asea.compose.runtime._adapter", lambda *a, **kw: {"type": "text", "text": "```python\ndef f(): return 1\n```"})
    ev = cert.evaluate(ws, spec, suite("functional_code", claims=["coding"]))
    assert ev["status"] == "admitted", ev
    cert.activate(ws, ev["id"], approver="Unit sandbox reviewer")
    monkeypatch.setattr("asea.compose.runtime._adapter", lambda *a, **kw: {"type": "text", "text": "def f(): return 2"})
    failed = cert.evaluate(ws, spec, suite("functional_code", claims=["coding"]))
    assert failed["status"] == "rejected", failed
    assert all(row["score"] == 0 for row in failed["case_evidence"])


def test_publication_postreplace_failure_restores_pointer(harness, monkeypatch):
    ws, spec = harness
    ev = cert.evaluate(ws, spec, suite())
    cert.activate(ws, ev["id"], approver="Previous")
    before = (ws.root / "active.json").read_bytes()
    original = cert.atomic_json
    failed = []
    def fail(path, value):
        original(path, value)
        if Path(path).name == "active.json" and not failed:
            failed.append(True)
            raise OSError("postreplace fsync failure")
    monkeypatch.setattr(cert, "atomic_json", fail)
    with pytest.raises(OSError, match="postreplace"):
        cert.activate(ws, ev["id"], approver="Next")
    assert (ws.root / "active.json").read_bytes() == before


def test_real_random_gpt2_wrong_output_never_admitted(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    torch.manual_seed(12)
    model_path = tmp_path / "random-model"
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "hello": 3, "bye": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]", model_max_length=32)
    fast.save_pretrained(model_path)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=5, n_positions=32, n_embd=8, n_layer=1, n_head=1, eos_token_id=2, pad_token_id=1, bos_token_id=2))
    model.save_pretrained(model_path, safe_serialization=True)
    spec = CompositionSpec.model_validate({"name": "RANDOM diagnostic only", "input_type": "text", "nodes": [{"id": "text", "input": "$input", "max_new_tokens": 2,
        "component": {"kind": "hf_text", "task": "causal", "model_path": str(model_path), "risk": "low", "provenance": [{"source": "random negative test", "risk": "low", "license": "test-only"}]}}], "output_node": "text"})
    raw = suite().model_dump()
    for case, text in zip(raw["cases"], ["hello", "bye"]):
        case.update(input=text, reference="IMPOSSIBLE REFERENCE", threshold=0.0)
    with pytest.raises(ValidationError): EvaluationSuite.model_validate(raw)
    for case in raw["cases"]: case["threshold"] = 1.0
    ws = Workspace(tmp_path / "workspace")
    ev = cert.evaluate(ws, spec, EvaluationSuite.model_validate(raw))
    assert ev["status"] == "rejected", ev
    assert [c["score"] for c in ev["case_evidence"]] == [0.0, 0.0]
    with pytest.raises(Blocked): cert.activate(ws, ev["id"])
    assert not (ws.root / "active.json").exists()
