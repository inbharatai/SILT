"""Admission mechanics on synthetic scores/headers, not real model quality evidence."""
import copy
import json
import math
import struct
import time
from types import SimpleNamespace

import pytest

from asea.compiler import core
from asea.compiler.__main__ import main


def rows(n=600):
    return [{"id": str(i), "prompt": "independent heldout %d" % i,
             "references": ["correct"], "group": "target" if i < n // 2 else "control"}
            for i in range(n)]


def scores(cases, good=True):
    return core.score_generations(cases, ["correct" if good else "wrong"] * len(cases))


def test_zero_zero_and_identical_wrong_generations_never_quality():
    cases = rows()
    result = scores(cases, False)
    groups, reasons = core.quality_decision(cases, result, result, core.admission_policy())
    assert {r["code"] for r in reasons} >= {"REFERENCE_SCORE_FLOOR", "CANDIDATE_SCORE_FLOOR"}
    assert groups["target"]["candidate_minus_reference"] == 0


def test_twenty_identical_correct_cases_do_not_automatically_pass():
    cases = rows(20)
    groups, reasons = core.quality_decision(cases, scores(cases), scores(cases), core.admission_policy())
    assert groups["control"]["noninferiority_lower"] < -0.05
    assert {r["group"] for r in reasons if r["code"] == "NONINFERIORITY_UNPROVEN"} == {"target", "control"}


def test_sufficient_identical_correct_cases_can_establish_narrow_noninferiority():
    cases = rows()
    groups, reasons = core.quality_decision(cases, scores(cases), scores(cases), core.admission_policy())
    assert not reasons
    assert -0.02 < groups["target"]["noninferiority_lower"] < 0
    assert core.quality_decision(cases, scores(cases), scores(cases), core.admission_policy())[0] == groups


def test_control_regression_is_not_hidden_by_target_success():
    cases = rows()
    candidate = core.score_generations(cases, ["correct" if r["group"] == "target" else "wrong" for r in cases])
    _, reasons = core.quality_decision(cases, candidate, scores(cases), core.admission_policy())
    assert any(r["code"] == "NONINFERIORITY_UNPROVEN" and r["group"] == "control" for r in reasons)


def test_absolute_quality_uses_lower_bound_not_just_observed_half():
    cases = rows()
    result = core.score_generations(cases, ["correct" if i % 2 else "wrong" for i in range(len(cases))])
    groups, reasons = core.quality_decision(cases, result, result, core.admission_policy())
    assert groups["target"]["reference_score"] == 0.5
    assert groups["target"]["reference_score_lower"] < 0.5
    assert {r["code"] for r in reasons} >= {"REFERENCE_SCORE_FLOOR", "CANDIDATE_SCORE_FLOOR"}


def test_paired_losses_not_only_aggregate_difference():
    cases = rows()
    c = core.score_generations(cases, ["correct" if i % 10 else "wrong" for i in range(len(cases))])
    r = core.score_generations(cases, ["correct" if i % 10 != 1 else "wrong" for i in range(len(cases))])
    groups, reasons = core.quality_decision(cases, c, r, core.admission_policy())
    assert groups["target"]["candidate_minus_reference"] == 0
    assert any(x["code"] == "NONINFERIORITY_UNPROVEN" for x in reasons)


@pytest.mark.parametrize("kwargs", [{"minimum_cases": 19}, {"minimum_cases": True},
    {"max_loss": 0.051}, {"max_loss": -0.1}, {"max_loss": float("nan")},
    {"minimum_reference_score": 0.49}, {"minimum_candidate_score": 0.49},
    {"minimum_size_reduction": 0.099}, {"minimum_size_reduction": float("inf")},
    {"minimum_candidate_score": 1.1}])
def test_policy_hard_floors(kwargs):
    with pytest.raises(core.CompilerError) as error:
        core.admission_policy(**kwargs)
    assert error.value.code == "INVALID_POLICY"


@pytest.mark.parametrize("mutation,code", [
    (lambda c: c[:19], "INSUFFICIENT_CASES"),
    (lambda c: [{k: v for k, v in r.items() if k != "group"} for r in c], "MISSING_GROUPS"),
    (lambda c: [{**r, "group": "target"} for r in c], "MISSING_CONTROLS"),
    (lambda c: [{**r, "prompt": "same prompt"} for r in c], "DUPLICATE_CASES"),
    (lambda c: [{**r, "references": [""]} for r in c], "INVALID_DATASET")])
def test_missing_controls_duplicate_and_insufficient_cases(mutation, code):
    with pytest.raises(core.CompilerError) as error:
        core.admission_suite(mutation(rows(20)), core.admission_policy())
    assert error.value.code == code


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, 2, True])
def test_invalid_synthetic_aggregate_scores(bad):
    cases = rows(20)
    candidate = scores(cases)
    candidate["exact_match"] = bad
    with pytest.raises(core.CompilerError):
        core.quality_decision(cases, candidate, scores(cases), core.admission_policy())


def test_edited_boolean_and_unpaired_case_rejected():
    cases = rows(20)
    for key, value in (("exact_match", False), ("id", "not-the-case"), ("group", "other")):
        candidate = scores(cases)
        candidate["cases"][0][key] = value
        with pytest.raises(core.CompilerError):
            core.quality_decision(cases, candidate, scores(cases), core.admission_policy())


def test_optional_groups_old_evaluation_schema(tmp_path):
    case = {"id": "old", "prompt": "old suite", "references": ["text"]}
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({"split": "heldout", "cases": [case]}))
    _, cases = core.load_samples(path, "suite")
    assert "group" not in core.score_generations(cases, ["text"])["cases"][0]
    case["group"] = "unknown"
    path.write_text(json.dumps({"split": "heldout", "cases": [case]}))
    with pytest.raises(core.CompilerError):
        core.load_samples(path, "suite")


def test_exact_binomial_boundary_not_zero_variance():
    assert core.binomial_upper(0, 20) == pytest.approx(1 - core.ADMISSION_ALPHA ** (1 / 20))
    assert core.binomial_lower(0, 20) == 0
    assert core.binomial_upper(20, 20) == 1
    assert core.binomial_lower(20, 20) < 1
    assert core.binomial_upper(1, 20) > core.binomial_upper(0, 20)


@pytest.fixture
def models(tmp_path, monkeypatch):
    """Synthetic safetensors headers and tiny fake loader: no real quality claim."""
    source, candidate = tmp_path / "source", tmp_path / "candidate"
    for root, experts, count, dtype, width in ((source, 4, 100, "F32", 4), (candidate, 2, 60, "F16", 2)):
        root.mkdir()
        (root / "config.json").write_text(json.dumps({"architectures": [core.ARCH], "model_type": "switch_transformers", "num_experts": experts}))
        header = json.dumps({"synthetic": {"dtype": dtype, "shape": [count], "data_offsets": [0, count * width]}}).encode()
        (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(count * width))
    src, dst = core.inspect_model(source), core.inspect_model(candidate)
    manifest = {"schema": core.SCHEMA, "architecture": core.ARCH, "evidence_scope": core.SCOPE,
                "source": {"artifact_sha256": src["artifact_sha256"], "files": src["files"], "parameters": 100},
                "candidate": {"artifact_sha256": dst["artifact_sha256"], "files": dst["files"], "parameters": 60},
                "source_unchanged": True, "source_after_sha256": src["artifact_sha256"],
                "num_experts": 2, "original_num_experts": 4, "layers": {"synthetic_layer": {"retained_source_indices": [0, 1]}},
                "preflight": {"dtype": "float16"}, "repair_training": {"supported": False},
                "calibration": {"sample_prompt_sha256": [], "sha256": "synthetic-calibration"}}
    (candidate / "compiler_manifest.json").write_text(json.dumps(manifest))
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"split": "heldout", "cases": rows()}))
    calls = []

    class Parameter:
        dtype = "torch.float16"

        def __init__(self, n):
            self.n = n

        def numel(self):
            return self.n

        def element_size(self):
            return 2

    class Model:
        def __init__(self, n):
            self.n = n

        def named_parameters(self):
            return [("synthetic", Parameter(self.n))]

    def fake_load(root, info, dtype, memory_budget_mib):
        calls.append((str(root), dtype))
        return Model(info["weights"]["stored_parameters"]), None, None, {"dtype": dtype}

    monkeypatch.setattr(core, "load_model", fake_load)
    monkeypatch.setattr(core, "generate", lambda *a: "correct")
    monkeypatch.setattr(core, "runtime_versions", lambda: {"python": "synthetic-test", "torch": "synthetic-test",
        "transformers": "synthetic-test", "accelerate": "synthetic-test", "safetensors": "synthetic-test", "tokenizers": "synthetic-test"})
    return source, candidate, suite, calls


def cert_args(models, tmp_path):
    source, candidate, suite, _ = models
    return {"model": str(candidate), "reference_model": str(source), "suite": str(suite),
            "dtype": "float16", "certificate_output": str(tmp_path / "certificate.json")}


def test_cli_runs_own_evaluation_and_never_mutates_model(models, tmp_path, capsys):
    source, candidate, suite, calls = models
    before = core.inventory(candidate), core.inventory(source)
    args = cert_args(models, tmp_path)
    cli = ["certify"] + [item for k, v in args.items() for item in ("--" + k.replace("_", "-"), v)]
    assert main(cli) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "CERTIFIED"
    assert result["admission"] == "ADMITTED_SEQ2SEQ_TEXT_ONLY"
    assert not result["certifies_coding_skills"] and not result["autoactivated"]
    assert calls == [(str(candidate), "float16"), (str(source), "float16")]
    assert before == (core.inventory(candidate), core.inventory(source))
    certificate = json.loads((tmp_path / "certificate.json").read_text())
    h = certificate.pop("certificate_sha256")
    assert core.canonical_hash(certificate) == h
    sizes = result["sizes"]
    assert sizes["reductions"]["normalized_weight_bytes"] == pytest.approx(0.4)
    assert sizes["reference"]["on_disk_weight_bytes"] > sizes["reference"]["normalized_weight_bytes"]
    for measured in (result["evaluation"]["resources"], result["evaluation"]["reference"]["resources"]):
        assert measured["observed"] and "not_isolated" in measured["rss_scope"]
        assert all(type(measured[k]) is float and math.isfinite(measured[k]) for k in ("wall_seconds", "generation_wall_seconds", "process_peak_rss_bytes"))
    assert main(cli) == 2
    assert json.loads(capsys.readouterr().out)["reasons"][0]["code"] == "OUTPUT_EXISTS"
    assert len(calls) == 2


def test_rejection_certificate_and_cli_exit_two(models, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(core, "generate", lambda *a: "wrong")
    args = cert_args(models, tmp_path)
    cli = ["certify"] + [item for k, v in args.items() for item in ("--" + k.replace("_", "-"), v)]
    assert main(cli) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "REJECTED" and not result["ok"]
    assert result["admission"] == "UNADMITTED"
    assert (tmp_path / "certificate.json").exists()


@pytest.mark.parametrize("which", [0, 1])
def test_unsafe_output_rejected_before_load(models, tmp_path, which):
    args = cert_args(models, tmp_path)
    args["certificate_output"] = str(models[which] / "certificate.json")
    with pytest.raises(core.CompilerError) as error:
        core.certify_model(**args)
    assert error.value.code == "UNSAFE_OUTPUT"
    assert not models[3]


def test_output_appearing_during_evaluation_is_not_overwritten(models, tmp_path, monkeypatch):
    args = cert_args(models, tmp_path)
    output = tmp_path / "certificate.json"
    def collide(*a):
        output.write_text("another writer owns this")
        return "correct"
    monkeypatch.setattr(core, "generate", collide)
    with pytest.raises(core.CompilerError) as error:
        core.certify_model(**args)
    assert error.value.code == "OUTPUT_EXISTS"
    assert output.read_text() == "another writer owns this"


def test_output_symlink_is_rejected(models, tmp_path):
    output = tmp_path / "certificate.json"
    output.symlink_to(models[0] / "do-not-create")
    args = cert_args(models, tmp_path)
    with pytest.raises(core.CompilerError) as error:
        core.certify_model(**args)
    assert error.value.code == "UNSAFE_PATH"
    assert not models[3]


def test_source_relation_and_prune_dtype_required(models, tmp_path):
    args = cert_args(models, tmp_path)
    args["dtype"] = "float32"
    with pytest.raises(core.CompilerError) as error:
        core.certify_model(**args)
    assert error.value.code == "DTYPE_MISMATCH"
    args["dtype"] = "float16"
    (models[0] / "extra").write_text("source mutation")
    with pytest.raises(core.CompilerError) as error:
        core.certify_model(**args)
    assert error.value.code == "REFERENCE_MISMATCH"
    assert not models[3]


@pytest.mark.parametrize("changed", ["source", "candidate", "suite"])
def test_rehash_after_inference(models, tmp_path, monkeypatch, changed):
    source, candidate, suite, _ = models
    def mutate(*args):
        if changed == "suite":
            suite.write_text(suite.read_text() + " ")
        else:
            ((source if changed == "source" else candidate) / "mutated").write_text("changed")
        return "correct"
    monkeypatch.setattr(core, "generate", mutate)
    with pytest.raises(core.CompilerError) as error:
        core.certify_model(**cert_args(models, tmp_path))
    assert error.value.code in ("MODEL_CHANGED", "SOURCE_CHANGED", "SUITE_CHANGED")
    assert not (tmp_path / "certificate.json").exists()


def test_dtype_only_shrink_fails_size_gate(models, tmp_path):
    args = cert_args(models, tmp_path)
    result = core.certify_model(**args)
    source, candidate = core.inspect_model(models[0]), core.inspect_model(models[1])
    manifest = core.read_json(models[1] / "compiler_manifest.json")
    evidence = copy.deepcopy(result["evaluation"])
    # Claim a small disk artifact but same actual parameter/normalized weight count.
    evidence["loaded_structure"] = copy.deepcopy(evidence["reference"]["loaded_structure"])
    manifest["candidate"]["parameters"] = manifest["source"]["parameters"]
    candidate["weights"]["stored_parameters"] = source["weights"]["stored_parameters"]
    _, reasons = core.size_decision(candidate, source, manifest, evidence, core.admission_policy(), "float16")
    assert len(reasons) == 3


def test_missing_or_nonfinite_resource_measurement_rejected(models, tmp_path, monkeypatch):
    monkeypatch.setattr(core, "observed_resources", lambda *a: {"observed": True, "wall_seconds": float("nan")})
    # evaluate hashes with allow_nan=False, so failure cannot become a certificate.
    cli = ["certify"] + [item for k, v in cert_args(models, tmp_path).items() for item in ("--" + k.replace("_", "-"), v)]
    assert main(cli) == 2
    assert not (tmp_path / "certificate.json").exists()


def test_infer_has_observed_resources_and_rehash(models):
    result = core.infer_model(str(models[1]), "new inference", dtype="float16")
    assert result["model_unchanged"]
    assert result["resources"]["wall_seconds"] > 0
    assert result["admission"] == "UNADMITTED"


def test_certify_parse_errors_and_no_evidence_import(capsys):
    for extra in ([], ["--certificate-output", "/tmp/cert.json", "--evidence-input", "edited.json"]):
        assert main(["certify", "--model", "/missing", "--reference-model", "/missing2", "--suite", "/missing3"] + extra) == 2
        result = json.loads(capsys.readouterr().out)
        assert result["status"] == "REJECTED"
        assert result["reasons"][0]["code"] == "INVALID_ARGUMENT"


def test_runtime_metrics_are_actual_finite_floats():
    start = time.perf_counter()
    result = core.observed_resources(start, 0.001)
    assert result["wall_seconds"] >= 0
    assert type(result["process_peak_rss_bytes"]) is float
    assert result["process_peak_rss_bytes"] > 0

