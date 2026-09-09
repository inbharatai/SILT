"""Frozen specialist data checks: no model weights, reference execution, or network."""
import ast
from collections import Counter
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("specialist_builder", ROOT / "scripts/prepare_specialist_data.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
DATA = ROOT / "data/specialist-v1"


def read(name):
    return json.loads((DATA / name).read_bytes())


@pytest.fixture(scope="module")
def splits():
    return {name: read(name + ".json")["samples"] for name in ("train", "calibration", "validation", "final")}


@pytest.fixture(scope="module")
def all_samples(splits):
    return splits["train"] + splits["validation"] + splits["final"]


def test_frozen_counts_and_deliberate_train_calibration_overlap(splits):
    assert {k: len(v) for k, v in splits.items()} == {"train": 42, "calibration": 32, "validation": 8, "final": 16}
    train = {s["id"]: s for s in splits["train"]}
    assert len({s["id"] for s in splits["calibration"]}) == 32
    assert all(train[s["id"]] == s for s in splits["calibration"])
    assert Counter(s["group"] for s in splits["validation"]) == {"target": 6, "control": 2}
    assert Counter(s["group"] for s in splits["final"]) == {"target": 12, "control": 4}
    assert Counter(s["group"] for s in splits["train"]) == {"target": 40, "control": 2}


def test_fully_disjoint_ids_families_and_all_fingerprints(all_samples):
    for key in ("id", "family", "canonical_prompt_sha256", "canonical_source_sha256", "alpha_source_sha256", "source_record_sha256", "response_sha256"):
        assert len({s[key] for s in all_samples}) == len(all_samples), key
    tags = [set(s["operation_tags"]) for s in all_samples]
    for i, a in enumerate(tags):
        assert all(not (a & b) for b in tags[:i])
    clusters, _ = builder.families([dict(s) for s in all_samples])
    assert len(clusters) == len(all_samples)


def test_excludes_every_consumed_id_and_expanded_old_families(all_samples):
    lock = read("selection-lock.json")
    old = lock["prior_consumed_selection"]
    assert len(old) == 32
    assert Counter(s["split"] for s in old) == {"development": 8, "final": 24}
    selected = {s["id"] for s in all_samples}
    assert selected.isdisjoint(s["id"] for s in old)
    audits = {a["id"]: a for a in read("eligibility-audit.json")["tasks"]}
    consumed_families = {audits[s["id"]]["family"] for s in old}
    assert consumed_families.isdisjoint(s["family"] for s in all_samples)
    assert all(builder.initial_family(s) is None for s in all_samples)
    # Source-only semantic regression cases: toggle/flip, split words, regex
    # repetition variants, set difference, power mapping and content filters.
    near_consumed = {"mbpp_557", "mbpp_285", "mbpp_118", "he_7", "he_29", "mbpp_161", "mbpp_777", "mbpp_447", "mbpp_623", "mbpp_604"}
    assert selected.isdisjoint(near_consumed)
    assert all(audits[i]["reason"] == "consumed_or_initial_six_near_family" for i in near_consumed)


def test_complete_original_response_and_original_prompt_fidelity(all_samples):
    rows = {builder.metadata(source, row)["id"]: (source, row) for source, row in builder.pilot.get_tasks(builder.pilot.read_sources(DATA))}
    for s in all_samples:
        source, row = rows[s["id"]]
        assert s["response"] == builder.response_for(source, row)
        assert s["original_prompt"] == row["prompt"]
        assert s["prompt"] == builder.make_prompt(s)
        assert s["source_task_id"] == str(row["task_id"])
        assert s["source_record_sha256"] == builder.sha(builder.canonical(row))
        assert s["response_sha256"] == builder.sha(s["response"].encode())
        functions = [n for n in ast.parse(s["response"]).body if isinstance(n, ast.FunctionDef)]
        assert len(functions) == 1 and functions[0].name == s["entrypoint"]
        assert s["license"] == ("MIT" if source == "HumanEval" else "CC-BY-4.0")


def test_cases_are_literal_original_assertions_not_synthetic(splits):
    for split, filename in (("train", "train-cases.json"), ("validation", "validation-provenance.json"), ("final", "final-provenance.json")):
        cases = {t["id"]: t["function_cases"] for t in read(filename)["tasks"]}
        assert set(cases) == {s["id"] for s in splits[split]}
        for s in splits[split]:
            retained = cases[s["id"]]
            assert 2 <= len(retained) <= 8
            assert len({builder.canonical(c["expected"]) for c in retained}) >= 2
            assert s["case_statistics"]["nonempty_expected"] >= 1
            for c in retained:
                decoded = builder.pilot.parse_assertion(c["upstream_assertion"], s["entrypoint"], "candidate" if s["source"] == "HumanEval" else None)
                assert decoded == {k: v for k, v in c.items() if k not in ("id", "upstream_assertion")}


def test_external_evaluation_suite_contract_and_no_execution(splits):
    from asea.compose.schema import EvaluationSuite
    for split in ("validation", "final"):
        data = read(split + "-suite.json")
        model = EvaluationSuite.model_validate(data)
        assert model.claims == ["coding"]
        assert {c.id for c in model.cases} == {s["id"] for s in splits[split]}
        for c in model.cases:
            assert c.metric == "function_io" and c.threshold == 1.0
            assert c.output_format == "raw_python"
            assert all(set(io) == {"id", "function", "args", "kwargs", "expected"} for io in c.function_cases)


def test_api_schema_and_no_final_contamination(splits):
    for split, samples in splits.items():
        data = read(split + ".json")
        assert data["schema"] == "silt.specialist.supervised.v1"
        assert data["split_policy"] == "standalone_LOCAL_not_official_benchmark_split"
        assert all({"id", "prompt", "response"} <= set(s) for s in samples)
    training_ids = {s["id"] for s in splits["train"] + splits["calibration"]}
    heldout_ids = {s["id"] for s in splits["validation"] + splits["final"]}
    assert training_ids.isdisjoint(heldout_ids)


def test_token_bounds_recorded_without_truncation(all_samples):
    for s in all_samples:
        n = s["token_lengths"]
        assert 0 < n["response_tokens"] <= 192
        assert 0 < n["flat_tokens_with_eos"] <= 256
        assert 0 < n["chat_tokens"] <= 256
        assert n["generation_prompt_tokens"] < n["chat_tokens"]
    contract = read("selection-lock.json")["token_contract"]
    assert contract["truncation"] is False
    assert contract["both_encodings_must_fit"] is True


def test_pinned_licenses_and_source_bytes():
    manifest = read("manifest.json")
    for source in manifest["sources"]:
        assert builder.sha((DATA / source["file"]).read_bytes()) == source["sha256"]
        assert (DATA / source["file"]).read_bytes() == (ROOT / "data/pilot-v2" / source["file"]).read_bytes()
    assert "CC-BY-4.0" in (DATA / "sources/mbpp-LICENSE-evidence.md").read_text()
    assert "Copyright (c) OpenAI" in (DATA / "sources/HumanEval-LICENSE.txt").read_text()
    attribution = (DATA / "ATTRIBUTION.md").read_text()
    assert "Additional modifications: specialist-v1" in attribution
    assert "Austin, Jacob" in attribution and "No endorsement" in attribution


def test_sha_inventory_covers_all_frozen_data_and_builder():
    manifest = read("manifest.json")
    for name, expected in manifest["artifact_sha256"].items():
        assert builder.sha((DATA / name).read_bytes()) == expected, name
    names = set()
    for line in (DATA / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        names.add(name)
        assert builder.sha((DATA / name).read_bytes()) == expected, name
    assert names == {str(p.relative_to(DATA)) for p in DATA.rglob("*") if p.is_file()} - {"SHA256SUMS"}
    assert builder.sha((ROOT / "scripts/prepare_specialist_data.py").read_bytes()) == manifest["builder_sha256"]
    assert builder.sha((ROOT / "scripts/prepare_coding_pilot.py").read_bytes()) == manifest["audited_parser_sha256"]
    assert builder.sha((ROOT / "data/pilot-v2/selection-lock.json").read_bytes()) == builder.PRIOR_LOCK_SHA256
    assert manifest["official_split"] is False
    assert manifest["environment"]["weights_loaded"] is False
    assert manifest["environment"]["network_used"] is False


def test_audit_is_exhaustive_and_shortfall_is_not_padded():
    manifest = read("manifest.json")
    audits = read("eligibility-audit.json")["tasks"]
    assert len(audits) == len({a["id"] for a in audits}) == 591
    assert Counter(a["reason"] for a in audits if a["status"] == "excluded") == manifest["counts"]["exclusions_by_reason"]
    assert sum(a["status"] == "selected" for a in audits) == 66
    assert sum(a["status"] == "eligible_not_selected" for a in audits) == 1
    assert manifest["counts"]["train_shortfall"] == 22
    assert manifest["counts"]["eligible_independent_families"] == 67


@pytest.mark.parametrize("source", [
    'assert fn(__import__("os").system("false")) == 0',
    'assert fn(1) == eval("2")',
    'assert fn(x=1) == 2',
    'assert fn((1, 2)) == [1, 2]',
    'assert fn(2) == 1 + 1',
])
def test_audited_literal_parser_rejects_executable_or_unsupported_data(source):
    with pytest.raises(builder.pilot.Unsupported):
        builder.pilot.parse_assertion(source, "fn")


def test_reference_ast_is_inert_and_alpha_fingerprint_catches_renaming():
    # The body would throw if executed. Fingerprinting only traverses inert AST.
    a = 'def first(items):\n    """Original doc."""\n    raise RuntimeError("MUST NEVER RUN")\n'
    b = 'def second(values):\n    """Different doc."""\n    raise RuntimeError("MUST NEVER RUN")\n'
    fa, fb = builder.fingerprints("one", a), builder.fingerprints("two", b)
    assert fa["alpha_source_sha256"] == fb["alpha_source_sha256"]
    assert fa["canonical_source_sha256"] != fb["canonical_source_sha256"]


def test_offline_tokenizer_reproduction_and_full_byte_rebuild(all_samples):
    if not builder.DEFAULT_TOKENIZER.exists():
        pytest.skip("Exact offline tokenizer snapshot required for byte rebuild; no download allowed")
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(builder.DEFAULT_TOKENIZER), local_files_only=True, trust_remote_code=False)
    for s in all_samples:
        assert builder.token_lengths(tokenizer, s["prompt"], s["response"]) == s["token_lengths"]
    data, _ = builder.render(ROOT / "data/pilot-v2", builder.DEFAULT_TOKENIZER)
    assert all((DATA / name).read_bytes() == content for name, content in data.items())
