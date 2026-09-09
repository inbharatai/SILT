"""Fresh offline control-flow fixtures, NOT pretrained-model quality evidence.

Adapter/oracle doubles are explicitly synthetic except the named contained-oracle
integration test. Detailed real transport/codec privacy coverage also lives in
test_oracle_observability and test_function_oracle.
"""
import copy
import json
import zipfile

import pytest

import asea.certification as cert
import asea.certification.function_oracle as oracle
import asea.compose.__main__ as cli
from asea.artifacts import Blocked, Workspace, digest
from asea.certification.sandbox import SandboxLimits, SandboxResult, probe_code_sandbox
from asea.compose import runtime
from asea.compose.generation_contract import new_trace, render_prompt
from asea.compose.schema import CompositionSpec, EvaluationCase, EvaluationSuite


REAL_EVALUATE_FUNCTIONS = oracle.evaluate_functions
POLICY = lambda retention: {"schema_version": 1, "return_retention": retention}
FUNCTIONS = [{"id": "fresh-call", "function": "f", "args": [], "kwargs": {}, "expected": "fresh-value"}]


def fresh_suite(metric="function_io", formats=(None, None)):
    return EvaluationSuite(name="fresh observability unit suite", reference_source="new synthetic literals only",
        claims=["coding"] if metric == "function_io" else ["reference_text"], cases=[
            EvaluationCase(id=group, group=group, input="fresh task " + group,
                reference="fresh source or reference", metric=metric, threshold=1.0,
                function_cases=copy.deepcopy(FUNCTIONS) if metric == "function_io" else None,
                output_format=fmt) for group, fmt in zip(("target", "control"), formats)])


@pytest.fixture
def harness(tmp_path, monkeypatch):
    model = tmp_path / "new-synthetic-model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"not weights; never loaded")
    spec = CompositionSpec.model_validate({"name": "NEW SYNTHETIC CONTROL FLOW", "input_type": "text",
        "nodes": [{"id": "out", "input": "$input", "component": {
            "kind": "hf_text", "task": "causal", "model_path": str(model), "risk": "low",
            "provenance": [{"source": "synthetic unit fixture", "license": "test-only", "risk": "low"}]}}],
        "output_node": "out"})
    calls = {"prompts": [], "oracle_policies": [], "returned": "fresh-value"}

    def adapter(node, value, output_path, limits, allow_fixtures, trace):
        calls["prompts"].append(render_prompt(node, value["text"]))
        return {"type": "text", "text": "fresh source or reference"}

    def host(source, cases, limits=None, trace_policy=None):
        policy = oracle._trace_policy(trace_policy)
        calls["oracle_policies"].append(policy)
        frozen, inputs = oracle._suite(source, cases, SandboxLimits())
        result = oracle._base_result(source, frozen, inputs, SandboxLimits())
        completed = [{"id": c["id"], "matched": oracle._equal(c["expected"], calls["returned"])} for c in cases]
        passed = all(c["matched"] for c in completed)
        result.update(status="PASSED" if passed else "FAILED", supported=True, passed=passed,
            returncode=0, tests_run=len(cases), tests_passed=sum(c["matched"] for c in completed), cases=completed,
            return_trace={"schema_version": 1, "policy": policy,
                          "returns": [oracle._return_trace(c["id"], calls["returned"], policy) for c in cases]},
            diagnostic={"schema_version": 1, "host_phase": "response", "event": "completed", "candidate_error": None},
            termination={"schema_version": 1, "observed_exit": 0, "observed_exit_state": "known",
                         "final_exit": 0, "final_exit_state": "known", "grace_seconds": 0.001},
            evidence_flags=["sensitive_actual_value_trace"] if policy["return_retention"] == "value" else [])
        result["resources"]["limits_established"] = True
        return result

    monkeypatch.setattr(runtime, "_adapter_impl", adapter)
    monkeypatch.setattr(oracle, "evaluate_functions", host)
    monkeypatch.setattr("asea.certification.sandbox.probe_code_sandbox", lambda: SandboxResult("PASSED", True, True))
    return Workspace(tmp_path / "new-workspace"), spec, calls


def strict(spec, fmt="raw_python"):
    raw = spec.model_dump(mode="json")
    raw["nodes"][-1]["output_contract"] = {"schema_version": 1, "format": fmt}
    return CompositionSpec.model_validate(raw)


def test_legacy_case_bytes_and_optional_field():
    case = fresh_suite().cases[0]
    dumped = case.model_dump(mode="json")
    assert "output_format" not in dumped
    assert dumped["input_file"] is None
    assert EvaluationCase.model_validate({**dumped, "output_format": None}).model_dump(mode="json") == dumped
    changed = EvaluationCase.model_validate({**dumped, "output_format": "raw_python"})
    assert changed.model_dump(mode="json")["output_format"] == "raw_python"
    assert digest(changed.model_dump(mode="json")) != digest(dumped)
    with pytest.raises(ValueError):
        EvaluationCase.model_validate({**dumped, "output_format": "guess_from_prompt"})


@pytest.mark.parametrize("fmt", ["raw_python", "fenced_python"])
def test_declaration_requires_strict_node_before_first_inference(harness, fmt):
    ws, spec, calls = harness
    result = cert.evaluate(ws, spec, fresh_suite(formats=(None, fmt)))
    assert result["status"] == "blocked" and not calls["prompts"]
    assert "strict output_contract" in result["reason"]
    assert not list((ws.root / "runs").iterdir())


@pytest.mark.parametrize("formats", [("raw_python", "fenced_python"), (None, "fenced_python")])
def test_all_cases_preflight_before_inference(harness, formats):
    ws, spec, calls = harness
    result = cert.evaluate(ws, strict(spec), fresh_suite(formats=formats))
    assert result["status"] == "blocked" and calls["prompts"] == []


def test_intermediate_strict_contract_conflict_preflight(harness):
    ws, spec, calls = harness
    raw = strict(spec).model_dump(mode="json")
    first = copy.deepcopy(raw["nodes"][0])
    first.update(id="first", input="$input")
    first["output_contract"]["format"] = "fenced_python"
    raw["nodes"][0]["input"] = "first"
    raw["nodes"].insert(0, first)
    result = cert.evaluate(ws, CompositionSpec.model_validate(raw), fresh_suite(formats=("raw_python", None)))
    assert result["status"] == "blocked" and not calls["prompts"]


def test_strict_format_only_adds_instruction_task_bytes_unchanged(harness):
    ws, spec, calls = harness
    suite = fresh_suite(formats=("raw_python", "raw_python"))
    original = suite.model_dump(mode="json")
    result = cert.evaluate(ws, strict(spec), suite)
    assert result["status"] == "admitted"  # synthetic control flow only
    for case, prompt in zip(suite.cases, calls["prompts"]):
        assert prompt.endswith("\n\n" + case.input)
    assert suite.model_dump(mode="json") == original
    assert result["suite"] == original


@pytest.mark.parametrize("retention", ["none", "digest", "value"])
def test_evidence_policy_hash_bindings_and_activation(harness, retention):
    ws, spec, calls = harness
    result = cert.evaluate(ws, spec, fresh_suite(), trace_policy=retention)
    assert result["status"] == "admitted"
    assert result["evidence_schema"] == "obs_v1"
    assert result["trace_policy"] == POLICY(retention)
    assert result["trace_policy_hash"] == digest(POLICY(retention))
    assert len(calls["oracle_policies"]) == 2
    for row in result["case_evidence"]:
        assert row["oracle_evidence"]["cases"] == [{"id": "fresh-call", "matched": True}]
        assert row["obs_v1_hash"] == digest(row["obs_v1"])
        assert row["obs_v1"]["generation"]["status"] == "captured"
        trace = row["oracle_evidence"]["return_trace"]
        assert row["obs_v1"]["oracle"]["sha256"] == digest(trace)
        assert ("value" in trace["returns"][0]) == (retention == "value")
        assert ("sha256" in trace["returns"][0]) == (retention != "none")
    cert.activate(ws, result["id"])
    assert calls["oracle_policies"] == [POLICY(retention)] * 4


@pytest.mark.parametrize("retention", ["none", "digest", "value"])
def test_wrong_answer_never_changes_pass_rules(harness, retention):
    ws, spec, calls = harness
    calls["returned"] = "fresh wrong value"
    result = cert.evaluate(ws, spec, fresh_suite(), trace_policy=POLICY(retention))
    assert result["status"] == "rejected"
    assert all(row["oracle_evidence"]["cases"] == [{"id": "fresh-call", "matched": False}] for row in result["case_evidence"])
    with pytest.raises(Blocked):
        cert.activate(ws, result["id"])


@pytest.mark.parametrize("field", ["trace_policy_hash", "evidence_flags", "obs_v1_hash", "return_trace", "generation_binding"])
def test_activation_rejects_captured_metadata_tampering(harness, monkeypatch, field):
    ws, spec, calls = harness
    result = cert.evaluate(ws, spec, fresh_suite(), trace_policy="value")
    original = ws.read_record

    def tampered(collection, identifier):
        value = original(collection, identifier)
        if collection == "evaluations":
            row = value["case_evidence"][0]
            if field == "trace_policy_hash":
                value[field] = "0" * 64
            elif field == "evidence_flags":
                value[field] = []
            elif field == "obs_v1_hash":
                row[field] = "0" * 64
            elif field == "return_trace":
                row["oracle_evidence"]["return_trace"]["returns"][0]["value"] = "changed"
                row["oracle_evidence_hash"] = digest(row["oracle_evidence"])
            else:
                row["obs_v1"]["generation"]["sha256"] = "0" * 64
                row["obs_v1_hash"] = digest(row["obs_v1"])
        return value

    monkeypatch.setattr(ws, "read_record", tampered)
    with pytest.raises(ValueError):
        cert.activate(ws, result["id"])
    assert not (ws.root / "active.json").exists()


def test_timing_and_diagnostic_text_not_compared_for_verdict(harness, monkeypatch):
    ws, spec, _ = harness
    result = cert.evaluate(ws, spec, fresh_suite())
    original = oracle.evaluate_functions

    def changed(*args, **kwargs):
        value = original(*args, **kwargs)
        value["termination"]["grace_seconds"] = 0.0027
        value["diagnostic"]["message"] = "fresh untrusted diagnostic variation"
        return value

    monkeypatch.setattr(oracle, "evaluate_functions", changed)
    cert.activate(ws, result["id"])
    assert ws.read_record("evaluations", result["id"]) == result


def test_missing_original_trace_marked_unknown_never_backfilled(harness, monkeypatch):
    ws, spec, _ = harness
    original = oracle.evaluate_functions

    def legacy(*args, **kwargs):
        value = original(*args, **kwargs)
        for key in ("return_trace", "diagnostic", "termination", "evidence_flags"):
            value.pop(key)
        return value

    monkeypatch.setattr(oracle, "evaluate_functions", legacy)
    monkeypatch.setattr(runtime, "_adapter", lambda *args: {"type": "text", "text": "fresh source or reference"})
    result = cert.evaluate(ws, spec, fresh_suite())
    row = result["case_evidence"][0]
    assert row["obs_v1"]["oracle"]["status"] == "original_unknown"
    assert row["obs_v1"]["generation"]["status"] == "original_unknown"
    before = (ws.root / "evaluations" / (result["id"] + ".json")).read_bytes()
    monkeypatch.setattr(oracle, "evaluate_functions", original)
    cert.activate(ws, result["id"])
    assert (ws.root / "evaluations" / (result["id"] + ".json")).read_bytes() == before


def test_fingerprint_includes_renderer(harness, monkeypatch):
    ws, spec, _ = harness
    result = cert.evaluate(ws, spec, fresh_suite())
    assert "compose/generation_contract.py" in result["implementation_fingerprint"]["implementation_hashes"]
    original = cert.implementation_fingerprint

    def changed():
        value = original()
        value["implementation_hashes"]["compose/generation_contract.py"]["sha256"] = "0" * 64
        return value

    monkeypatch.setattr(cert, "implementation_fingerprint", changed)
    with pytest.raises(Blocked, match="stale_implementation"):
        cert.activate(ws, result["id"])


@pytest.mark.parametrize("payload", [
    {"trace_policy": {"return_retention": "VaLuE"}},
    {"trace_policy": "VaLuE"},
    {"return_trace": {"returns": [{"retention": "VaLuE"}]}},
    {"return_trace": {"returns": [{"retention": "none", "value": None}]}},
    {"evidence_flags": ["SENSITIVE_ACTUAL_VALUE_TRACE"]},
    {"generation_trace_v1": [{"generated_token_ids": [0]}]},
    {"stdout": "potential PII"}, {"stderr": "potential PII"},
    {"diagnostic": {"message": "potential PII"}},
])
def test_privacy_scan_covers_payload_even_wrong_case_or_missing_flags(payload):
    assert cert._sensitive_observations(payload)


@pytest.mark.parametrize("retention", ["none", "digest", "value"])
def test_sensitive_export_refuses_without_mutation_and_opt_in_keeps_bundle(harness, tmp_path, retention):
    ws, spec, _ = harness
    result = cert.evaluate(ws, spec, fresh_suite(), trace_policy=retention)
    deployment = cert.activate(ws, result["id"])
    before = {str(p): p.read_bytes() for collection in ("evaluations", "runs", "deployments")
              for p in (ws.root / collection).glob("*.json")}
    destination = tmp_path / "new-evidence.zip"
    with pytest.raises(cert.SENSITIVE_TRACE_EXPORT_REQUIRES_OPT_IN):
        cert.export_deployment(ws, deployment["id"], destination)
    assert not destination.exists()
    exported = cert.export_deployment(ws, deployment["id"], destination, include_sensitive_traces=True)
    assert exported["format_version"] == 1
    with zipfile.ZipFile(destination) as archive:
        manifest = json.loads(archive.read("composition-export-v1.json"))
    assert manifest["evaluation"] == result
    assert before == {path: __import__("pathlib").Path(path).read_bytes() for path in before}


def test_nonsensitive_text_export_legacy_shape(harness, tmp_path):
    ws, spec, _ = harness
    result = cert.evaluate(ws, spec, fresh_suite("text_exact"))
    deployment = cert.activate(ws, result["id"])
    assert result["evidence_flags"] == []
    cert.export_deployment(ws, deployment["id"], tmp_path / "nonsensitive.zip")


def test_sensitive_opt_in_never_bypasses_admission(harness, tmp_path):
    ws, spec, calls = harness
    calls["returned"] = "wrong"
    result = cert.evaluate(ws, spec, fresh_suite(), trace_policy="value")
    identifier = ws.new_id("deployment")
    ws.write_record("deployments", identifier, {"evaluation_id": result["id"]})
    with pytest.raises(Blocked, match="real admitted"):
        cert.export_deployment(ws, identifier, tmp_path / "bad.zip", include_sensitive_traces=True)
    assert not (tmp_path / "bad.zip").exists()


def test_blocked_oracle_evidence_retained_without_admission(harness, monkeypatch):
    ws, spec, _ = harness
    original = oracle.evaluate_functions

    def blocked(*args, **kwargs):
        value = original(*args, **kwargs)
        value.update(status="BLOCKED", supported=False, passed=False, reason="synthetic setup failure")
        return value

    monkeypatch.setattr(oracle, "evaluate_functions", blocked)
    result = cert.evaluate(ws, spec, fresh_suite(), trace_policy="value")
    assert result["status"] == "blocked" and result["case_evidence"] == []
    assert result["blocked_oracle_evidence"]["oracle_evidence"]["status"] == "BLOCKED"
    assert result["evidence_flags"] == ["sensitive_observations"]
    with pytest.raises(Blocked):
        cert.activate(ws, result["id"])


@pytest.mark.parametrize("with_workspace", [False, True])
def test_preview_no_workspace_or_model_side_effects(tmp_path, monkeypatch, capsys, with_workspace):
    raw = {"name": "read only unavailable models", "input_type": "audio", "nodes": [
        {"id": "asr", "input": "$input", "component": {"kind": "hf_asr", "task": "whisper"}},
        {"id": "text", "input": "asr", "prompt_prefix": "human must review this prefix",
         "component": {"kind": "hf_text", "task": "causal"}},
        {"id": "tts", "input": "text", "component": {"kind": "hf_tts", "task": "vits"}}], "output_node": "tts"}
    for node in raw["nodes"]:
        node["component"].update(model_path=str(tmp_path / "DOES-NOT-EXIST"), risk="low",
            provenance=[{"source": "fresh fixture", "license": "test-only", "risk": "low"}])
    path = tmp_path / "new-preview.json"
    path.write_text(json.dumps(raw))
    original = path.read_bytes()
    monkeypatch.setattr(cli, "Workspace", lambda *a: pytest.fail("preview must not create workspace"))
    monkeypatch.setattr("asea.artifacts.model_inventory", lambda *a: pytest.fail("preview must not inventory models"))
    args = ["preview", "--spec", str(path), "--output-format", "raw_python"]
    if with_workspace:
        args = ["--workspace", str(tmp_path / "no-workspace")] + args
    assert cli.main(args) == 0
    output = json.loads(capsys.readouterr().out)["result"]
    assert output["applied"] is False and output["read_only"] is True
    assert [n["support"] for n in output["nodes"]] == ["unsupported_adapter", "supported", "unsupported_adapter"]
    assert output["nodes"][1]["requires_prefix_review"] is True
    assert all(n["applied"] is False for n in output["nodes"])
    assert path.read_bytes() == original
    assert not (tmp_path / "no-workspace").exists()


def test_cli_defaults_and_explicit_policy(harness, tmp_path, monkeypatch, capsys):
    ws, spec, _ = harness
    args = cli.parser().parse_args(["--workspace", str(ws.root), "evaluate", "--spec", "s", "--suite", "t"])
    assert (args.trace_policy, args.resource_profile, args.memory_mib, args.case_timeout) == ("none", "observe_only", 512, 60.0)
    export = cli.parser().parse_args(["--workspace", str(ws.root), "export", "--deployment", "d", "--output", "z"])
    assert export.include_sensitive_traces is False and export.include_dependencies is False
    spec_path, suite_path = tmp_path / "spec.json", tmp_path / "suite.json"
    spec_path.write_text(spec.model_dump_json())
    suite_path.write_text(fresh_suite().model_dump_json())
    seen = []
    monkeypatch.setattr(cli, "evaluate", lambda *a, **kw: (seen.append(kw) or {"status": "admitted"}))
    assert cli.main(["--workspace", str(ws.root), "evaluate", "--spec", str(spec_path), "--suite", str(suite_path),
                     "--trace-policy", "digest"]) == 0
    assert seen[0]["trace_policy"] == "digest"
    assert json.loads(capsys.readouterr().out)["ok"] is True


@pytest.mark.parametrize("bad", ["VALUE", False, {"schema_version": True, "return_retention": "none"}])
def test_bad_policy_never_generates(harness, bad):
    ws, spec, calls = harness
    with pytest.raises(ValueError):
        cert.evaluate(ws, spec, fresh_suite(), trace_policy=bad)
    assert calls["prompts"] == []


def test_genuine_near_cap_codec_validation_wrong_case_return():
    # Full actual-value validation is reused, even for a wrong answer. Not repr,
    # truncation, expected echo, or a shortcut based on a passing boolean.
    value = ["é" * 8192, "é" * 8192]
    cases = copy.deepcopy(FUNCTIONS)
    policy = POLICY("value")
    evidence = {"return_trace": {"schema_version": 1, "policy": policy,
                                "returns": [oracle._return_trace("fresh-call", value, policy)]},
                "cases": [{"id": "fresh-call", "matched": False}],
                "evidence_flags": ["sensitive_actual_value_trace"]}
    observed = cert._oracle_observation(evidence, cases, policy)
    assert observed["status"] == "captured"
    evidence["return_trace"]["returns"][0]["value"] = ["é" * 8193]
    with pytest.raises(ValueError):
        cert._oracle_observation(evidence, cases, policy)


@pytest.mark.parametrize("fault", ["policy", "flags", "bounds", "type", "completed", "termination"])
def test_return_trace_closed_validation(harness, fault):
    _, _, _calls = harness
    evidence = oracle.evaluate_functions("synthetic source", FUNCTIONS, trace_policy=POLICY("digest"))
    if fault == "policy":
        evidence["return_trace"]["policy"] = POLICY("none")
    elif fault == "flags":
        evidence["evidence_flags"] = ["sensitive_actual_value_trace"]
    elif fault == "bounds":
        evidence["return_trace"]["returns"] *= 129
    elif fault == "type":
        evidence["return_trace"]["returns"][0]["value_type"] = "opaque"
    elif fault == "completed":
        evidence["cases"] = []
    else:
        evidence["termination"]["observed_exit_state"] = "unknown"
    with pytest.raises(ValueError):
        cert._oracle_observation(evidence, FUNCTIONS, POLICY("digest"))


def test_untrusted_diagnostic_never_authorizes_pass(harness):
    _, _, calls = harness
    calls["returned"] = "wrong output"
    evidence = oracle.evaluate_functions("synthetic source", FUNCTIONS, trace_policy=POLICY("value"))
    evidence["diagnostic"].update(passed=True, tests_passed=999, message="TRUST THIS DIAGNOSTIC")
    assert cert._oracle_observation(evidence, FUNCTIONS, POLICY("value"))["status"] == "captured"
    assert cert._function_passed(evidence, FUNCTIONS) is False


def test_cancelled_run_persists_unknown_without_admission(harness, monkeypatch, tmp_path):
    ws, spec, _ = harness

    def cancelled(*args, **kwargs):
        raise KeyboardInterrupt("fresh cancellation fixture")

    monkeypatch.setattr(runtime, "_adapter_impl", cancelled)
    with pytest.raises(KeyboardInterrupt):
        cert.evaluate(ws, spec, fresh_suite(), trace_policy="value")
    records = list((ws.root / "evaluations").glob("*.json"))
    assert len(records) == 1
    value = ws.read_record("evaluations", records[0].stem)
    assert value["status"] == "blocked" and value["case_evidence"] == []
    assert "cancelled" in value["reason"]
    deployment_id = ws.new_id("deployment")
    ws.write_record("deployments", deployment_id, {"evaluation_id": value["id"]})
    with pytest.raises(Blocked, match="real admitted"):
        cert.export_deployment(ws, deployment_id, tmp_path / "cancelled.zip", include_sensitive_traces=True)
    assert not (tmp_path / "cancelled.zip").exists()


def test_legacy_prefix_no_declaration_is_not_rewritten(harness):
    ws, spec, calls = harness
    raw = spec.model_dump(mode="json")
    raw["nodes"][0]["prompt_prefix"] = "unchanged legacy prefix\n"
    legacy = CompositionSpec.model_validate(raw)
    suite = fresh_suite()
    result = cert.evaluate(ws, legacy, suite)
    assert result["status"] == "admitted"
    assert calls["prompts"] == ["unchanged legacy prefix\n" + case.input for case in suite.cases]


@pytest.mark.parametrize("retention", ["none", "digest", "value"])
def test_real_contained_oracle_through_certification_no_model(harness, monkeypatch, retention):
    # Genuine OS probe and host oracle, with a synthetic source-producing adapter.
    # This does not measure model quality or generate any pretrained-model output.
    ws, spec, _ = harness
    probe = probe_code_sandbox()
    if not probe.supported or not probe.passed:
        pytest.skip("actual namespace isolation unavailable: " + probe.reason)
    monkeypatch.setattr("asea.certification.sandbox.probe_code_sandbox", probe_code_sandbox)
    monkeypatch.setattr(oracle, "evaluate_functions", REAL_EVALUATE_FUNCTIONS)
    monkeypatch.setattr(runtime, "_adapter_impl", lambda *a: {"type": "text", "text": "def f(): return 'fresh-value'"})
    result = cert.evaluate(ws, spec, fresh_suite(), trace_policy=retention)
    assert result["status"] == "admitted", result.get("reason")
    for row in result["case_evidence"]:
        original = row["oracle_evidence"]
        assert original["resources"]["limits_established"] is True
        assert original["return_trace"]["policy"] == POLICY(retention)
        assert row["obs_v1"]["oracle"]["termination_status"] == "captured"
    cert.activate(ws, result["id"])


def test_null_optional_diagnostic_and_termination_are_original_unknown(harness):
    evidence = oracle.evaluate_functions("synthetic source", FUNCTIONS, trace_policy=POLICY("none"))
    evidence.update(diagnostic=None, termination=None)
    observed = cert._oracle_observation(evidence, FUNCTIONS, POLICY("none"))
    assert observed["diagnostic_status"] == observed["termination_status"] == "original_unknown"
    assert observed["diagnostic_sha256"] is observed["termination_sha256"] is None


def test_generation_trace_bound_validation(harness):
    _, spec, _ = harness
    trace = new_trace(spec.nodes[0], {"type": "text", "text": "fresh task"})
    trace.update(status="succeeded", generated_token_count=513, generated_token_ids=list(range(513)))
    run = {"id": "synthetic", "graph_hash": "synthetic", "output": {"type": "text", "text": "x"},
           "generation_trace_schema": "generation_trace_v1", "generation_trace_v1": [trace]}
    with pytest.raises(Blocked, match="bounds"):
        cert._generation_observation(run, spec)
