"""Offline unit fixtures only. No real inference/evidence/certification is claimed."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from asea.artifacts import Blocked, Workspace, atomic_json, file_hash, model_inventory
from asea.certification import activate, evaluate, export_deployment, metric_score, rollback, select_measured
from asea.compose import CompositionSpec, EvaluationSuite, main, run_spec
from asea.compose.schema import load_json


@pytest.fixture
def fixture_spec(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "fixture.json").write_text('{"prefix":"echo:"}')
    return CompositionSpec.model_validate({"schema_version": 1, "name": "UNIT FIXTURE ONLY", "input_type": "text",
        "nodes": [{"id": "echo", "input": "$input", "component": {
            "kind": "fixture_text", "task": "fixture", "model_path": str(model), "risk": "low",
            "provenance": [{"source": "unit fixture, not an ML model", "risk": "medium", "license": "test-only"}]}}],
        "output_node": "echo"})


@pytest.fixture
def suite():
    return EvaluationSuite.model_validate({"name": "independent fixed UNIT references", "reference_source": "unit test literals",
        "cases": [{"id": "target", "group": "target", "input": "hello", "reference": "echo:hello", "metric": "text_exact", "threshold": 1.0},
                  {"id": "control", "group": "control", "input": "bye", "reference": "echo:bye", "metric": "text_similarity_proxy", "threshold": 0.99}]})


def test_schema_is_closed_and_typed(fixture_spec):
    raw = fixture_spec.model_dump()
    raw["command"] = "rm -rf /"
    with pytest.raises(ValidationError):
        CompositionSpec.model_validate(raw)
    raw.pop("command")
    raw["nodes"][0]["max_new_tokens"] = "20"
    with pytest.raises(ValidationError):
        CompositionSpec.model_validate(raw)


def test_cycles_and_type_mismatch(fixture_spec):
    raw = fixture_spec.model_dump()
    raw["nodes"][0]["input"] = "echo"
    with pytest.raises(ValidationError, match="cycle"):
        CompositionSpec.model_validate(raw)
    raw["nodes"][0]["input"] = "$input"
    raw["input_type"] = "audio"
    with pytest.raises(ValidationError, match="type mismatch"):
        CompositionSpec.model_validate(raw)


def test_max_risk_and_full_graph_fingerprint(tmp_path, fixture_spec):
    workspace = Workspace(tmp_path / "store")
    assert fixture_spec.effective_risk == "medium"
    first = run_spec(workspace, fixture_spec, "a", allow_fixtures=True)
    raw = fixture_spec.model_dump()
    raw["nodes"][0]["prompt_prefix"] = "new config"
    changed = run_spec(workspace, CompositionSpec.model_validate(raw), "a", allow_fixtures=True)
    assert first["candidate_id"] != changed["candidate_id"]
    assert first["graph_hash"] != changed["graph_hash"]


def test_fixture_run_is_not_admission_or_activation(tmp_path, fixture_spec, suite):
    workspace = Workspace(tmp_path / "store")
    with pytest.raises(Blocked, match="unit-test-only"):
        run_spec(workspace, fixture_spec, "hello")
    run = run_spec(workspace, fixture_spec, "hello", allow_fixtures=True)
    assert run["status"] == "succeeded"
    assert run["output"]["text"] == "echo:hello"
    assert run["resources"]["process_peak_rss_mb"] > 0
    assert not (workspace.root / "active.json").exists()
    evaluation = evaluate(workspace, fixture_spec, suite, allow_fixtures=True)
    assert all(case["passed"] for case in evaluation["case_evidence"])
    assert evaluation["status"] == "rejected"
    with pytest.raises(Blocked, match="real admitted"):
        activate(workspace, evaluation["id"], approve_high_risk=True)
    assert not (workspace.root / "active.json").exists()


def test_nonempty_targets_controls_and_thresholds(suite):
    raw = suite.model_dump()
    raw["cases"] = raw["cases"][:1]
    with pytest.raises(ValidationError):
        EvaluationSuite.model_validate(raw)
    raw = suite.model_dump()
    del raw["cases"][0]["threshold"]
    with pytest.raises(ValidationError):
        EvaluationSuite.model_validate(raw)
    raw = suite.model_dump()
    raw["status"] = "admitted"
    with pytest.raises(ValidationError):
        EvaluationSuite.model_validate(raw)


def test_functional_code_is_blocked_before_running(tmp_path, fixture_spec, suite):
    workspace = Workspace(tmp_path / "store")
    raw = suite.model_dump()
    raw["cases"][0]["metric"] = "functional_code"
    result = evaluate(workspace, fixture_spec, EvaluationSuite.model_validate(raw), allow_fixtures=True)
    assert result["status"] == "blocked"
    assert "sandbox" in result["reason"]
    assert list((workspace.root / "runs").iterdir()) == []


def test_metric_semantics():
    assert metric_score("word_error_rate", "a c", "a b") == 0.5
    assert metric_score("text_exact", "ab", "a b") == 0
    assert metric_score("text_similarity_proxy", "abc", "abc") == 1
    with pytest.raises(Blocked):
        metric_score("functional_code", "pass", "pass")


def test_symlinks_traversal_pickle_and_import(tmp_path):
    source = tmp_path / "model"
    source.mkdir()
    (source / "config.json").write_text("{}")
    workspace = Workspace(tmp_path / "store")
    with workspace.writer():
        imported = workspace.import_directory(source)
    assert model_inventory(imported) == model_inventory(source)
    (source / "alias.json").symlink_to(source / "config.json")
    with pytest.raises(Blocked, match="symlink"):
        model_inventory(source)
    (source / "alias.json").unlink()
    with pytest.raises(Blocked, match="traversal"):
        model_inventory(source / ".." / "model")
    (source / "pytorch_model.bin").write_bytes(b"bad")
    with pytest.raises(Blocked, match="pickle"):
        model_inventory(source)


def test_file_mutation_invalidates_registration(tmp_path, fixture_spec):
    workspace = Workspace(tmp_path / "store")
    with workspace.writer():
        identifier = workspace.register(fixture_spec.nodes[0].component)
        (Path(fixture_spec.nodes[0].component.model_path) / "fixture.json").write_text('{"prefix":"changed"}')
        with pytest.raises(Blocked, match="changed"):
            workspace.verify_artifact(identifier)


def test_record_tampering_does_not_promote(tmp_path, fixture_spec, suite):
    workspace = Workspace(tmp_path / "store")
    result = evaluate(workspace, fixture_spec, suite, allow_fixtures=True)
    path = workspace.root / "evaluations" / (result["id"] + ".json")
    raw = json.loads(path.read_text())
    raw["payload"]["status"] = "admitted"
    raw["payload"]["fixture_only"] = False
    path.write_text(json.dumps(raw))
    with pytest.raises(Blocked, match="integrity"):
        activate(workspace, result["id"])


def test_atomic_pointer_and_known_rollback(tmp_path):
    workspace = Workspace(tmp_path / "store")
    pointer = workspace.root / "active.json"
    atomic_json(pointer, {"deployment_id": "before"})
    atomic_json(pointer, {"deployment_id": "after"})
    assert json.loads(pointer.read_text()) == {"deployment_id": "after"}
    with pytest.raises(Blocked):
        rollback(workspace, "../../unknown")
    with pytest.raises(Blocked):
        export_deployment(workspace, "unknown", tmp_path / "out.zip")
    assert not (tmp_path / "out.zip").exists()


def test_single_writer_and_isolated_workspace(tmp_path):
    workspace = Workspace(tmp_path / "store")
    with workspace.writer():
        with pytest.raises(Blocked, match="locked"):
            with workspace.writer():
                pass
    unsafe = tmp_path / "legacy"
    unsafe.mkdir()
    (unsafe / "existing.json").write_text("{}")
    with pytest.raises(Blocked, match="empty"):
        Workspace(unsafe)


def test_selection_no_feasible(tmp_path, fixture_spec, suite):
    workspace = Workspace(tmp_path / "store")
    result = evaluate(workspace, fixture_spec, suite, allow_fixtures=True)
    selected = select_measured(workspace, [result["id"]], "text", "text", 20.0, 4000.0)
    assert selected["status"] == "no_feasible_bundle"


def test_output_bound_failed_run_audited(tmp_path, fixture_spec):
    workspace = Workspace(tmp_path / "store")
    raw = fixture_spec.model_dump()
    raw["limits"]["max_output_chars"] = 3
    with pytest.raises(Blocked, match="output exceeds"):
        run_spec(workspace, CompositionSpec.model_validate(raw), "hello", allow_fixtures=True)
    records = list((workspace.root / "runs").glob("*.json"))
    assert workspace.read_record("runs", records[0].stem)["status"] == "blocked"
    assert len((workspace.root / "audit.jsonl").read_text().splitlines()) == 1


def test_cli_json_stdout_and_return_int(tmp_path, fixture_spec, capsys):
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(fixture_spec.model_dump_json())
    code = main(["--workspace", str(tmp_path / "store"), "run", "--spec", str(spec_file), "--input", "hello"])
    assert code == 2
    assert json.loads(capsys.readouterr().out)["ok"] is False
    assert main(["--workspace", str(tmp_path / "store"), "list"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert main(["--bad-option"]) != 0
    assert "error" in json.loads(capsys.readouterr().out)
    assert main(["--help"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]


def test_duplicate_json_and_unsupported_vision(tmp_path, fixture_spec):
    path = tmp_path / "duplicate.json"
    path.write_text('{"name":"one","name":"two"}')
    with pytest.raises(ValueError, match="duplicate"):
        load_json(path, CompositionSpec)
    raw = fixture_spec.model_dump()
    raw["nodes"][0]["component"]["kind"] = "hf_vision"
    with pytest.raises(ValidationError):
        CompositionSpec.model_validate(raw)


def test_modules_import_without_optional_ml():
    program = "import sys; import asea.compose, asea.certification, asea.artifacts; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_shard_and_tokenizer_indirection_rejected(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"UNIT TEST INVALID WEIGHTS")
    index = model / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"layer.weight": "../outside.safetensors"}}))
    with pytest.raises(Blocked, match="outside registered"):
        model_inventory(model)
    index.unlink()
    (model / "tokenizer_config.json").write_text(json.dumps({"vocab_file": "/etc/passwd"}))
    with pytest.raises(Blocked, match="outside registered"):
        model_inventory(model)


def test_audio_contract_without_ml(tmp_path):
    import wave
    from asea.compose.runtime import _audio_info
    from asea.compose.schema import Limits
    path = tmp_path / "unit.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * 16000)
    assert _audio_info(path, Limits())["duration_seconds"] == 1.0
    with pytest.raises(Blocked, match="byte limit"):
        _audio_info(path, Limits(max_input_bytes=100))


def test_checkpoint_mismatch_blocks_before_optional_import(tmp_path, fixture_spec):
    workspace = Workspace(tmp_path / "store")
    model = Path(fixture_spec.nodes[0].component.model_path)
    (model / "model.safetensors").write_bytes(b"UNIT TEST INVALID WEIGHTS")
    (model / "config.json").write_text('{"model_type":"vits"}')
    raw = fixture_spec.model_dump()
    raw["input_type"] = "audio"
    raw["nodes"][0]["component"].update(kind="hf_asr", task="whisper")
    # Exercise only pre-load adapter validation, never a fixture as production evidence.
    from asea.compose.runtime import _adapter
    spec = CompositionSpec.model_validate(raw)
    with pytest.raises(Blocked, match="model_type"):
        _adapter(spec.nodes[0], {"type": "audio", "path": "unused"}, tmp_path / "unused.wav", spec.limits, False)
    assert list((workspace.root / "evaluations").iterdir()) == []


def test_example_is_exact_schema():
    path = Path(__file__).resolve().parents[1] / "configs" / "compose_text_example.json"
    spec = load_json(path, CompositionSpec)
    assert spec.nodes[0].component.task == "causal"
