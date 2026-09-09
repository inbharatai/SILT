"""Pass-2 MECHANICS ONLY: tiny random native Switch (<1 MB), never source runs.

No downloads, real-task accuracy, training, certification, or privileged setup.
"""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from asea.compiler.__main__ import main, parser
from asea.compiler.core import (CompilerError, RoutingTelemetry, inspect_model,
    inventory, load_model, preflight, prune_model, selection_scores, sparse_layers,
    validate_lineage)
from asea.compiler.diagnostics import (SCHEMA, DEFAULT_SUITE, diagnostic_bounds,
    finite_json, normalize, parse_spans, token_trace, validate_suite)


class TokenizerMechanics:
    pad_token_id, eos_token_id = 0, 1
    def get_vocab(self):
        return {"<pad>": 0, "</s>": 1, "<extra_id_0>": 2, "<extra_id_1>": 3,
                "<extra_id_2>": 4, "Paris": 5, ".": 6, "blue": 7}
    def decode(self, ids, skip_special_tokens=False):
        inverse = {v: k for k, v in self.get_vocab().items()}
        return " ".join(inverse[i] for i in ids if not skip_special_tokens or i >= 5)


def test_span_metric_preserves_tail_and_does_not_rewrite_literal():
    trace = token_trace([0, 2, 5, 3, 6, 1], TokenizerMechanics(), ["Paris"], 5)
    assert trace["parsed"]["valid"]
    assert trace["all_spans_correct"]
    assert not trace["parsed"]["full_output_well_formed"]
    assert trace["parsed"]["trailing_text"] == "."
    assert trace["original_skip_special_decode"] == "Paris ."
    assert trace["full_special_decode"] == "<pad> <extra_id_0> Paris <extra_id_1> . </s>"
    assert trace["eos_emitted"] and trace["termination"] == "eos"
    assert trace["raw_token_ids"] == [0, 2, 5, 3, 6, 1]
    assert normalize("Paris.") != normalize("Paris")


@pytest.mark.parametrize("ids", [[0, 5, 1], [0, 2, 5, 1], [0, 3, 5, 2, 1], [0, 2, 5, 2, 3, 1]])
def test_missing_misordered_duplicate_sentinels_never_valid(ids):
    assert not parse_spans(ids, TokenizerMechanics(), 1)["valid"]


def test_multispan_and_repetition():
    trace = token_trace([0, 2, 5, 3, 7, 4, 1], TokenizerMechanics(), ["Paris", "blue"], 6)
    assert trace["span_exact_match"] == [True, True]
    assert trace["parsed"]["full_output_well_formed"]
    trace = token_trace([0, 6, 6, 6, 6], TokenizerMechanics(), ["Paris"], 4)
    assert trace["termination"] == "max_new_tokens"
    assert trace["longest_repeated_token_run"] == 4
    assert trace["repeated_ngrams"]["2"] == 2
    assert finite_json({"nan": float("nan"), "inf": [float("inf")]}) == {"nan": None, "inf": [None]}


def test_suite_is_versioned_and_diagnostic_only():
    assert validate_suite(DEFAULT_SUITE)
    for bad in ({"split": "heldout", "cases": []}, {**DEFAULT_SUITE, "split": "heldout"}):
        with pytest.raises(CompilerError) as error:
            validate_suite(bad)
        assert error.value.code == "INVALID_DIAGNOSTIC_SUITE"
    with pytest.raises(CompilerError):
        diagnostic_bounds("float32", 64, 16, 16, .001, .001)
    with pytest.raises(CompilerError):
        diagnostic_bounds("bfloat16", 65, 16, 16, .001, .001)
    with pytest.raises(CompilerError):
        diagnostic_bounds("bfloat16", 64, 16, 16, float("nan"), .001)


def test_bfloat16_keeps_memory_ceiling():
    small = {"weights": {"stored_parameters": 1024}}
    assert preflight(small, "bfloat16")["estimated_peak_bytes"] == preflight(small, "float16")["estimated_peak_bytes"]
    with pytest.raises(CompilerError) as error:
        preflight({"weights": {"stored_parameters": 2_000_000_000}}, "bfloat16", 999999)
    assert error.value.code == "MEMORY_PREFLIGHT"


def test_public_commands_and_typed_coverage():
    args = parser().parse_args(["diagnose", "--model", "local", "--suite", "suite.json", "--output", "new.json", "--dtype", "bfloat16", "--preserve-router-fp32"])
    assert args.dtype == "bfloat16" and args.preserve_router_fp32
    args = parser().parse_args(["roundtrip", "--model", "local", "--output", "new", "--dtype", "float16"])
    assert args.suite is None
    row = {"expert_output_count": [2, 0], "dispatched_count": [2, 0], "reap_dispatched": [1.5, None]}
    with pytest.raises(CompilerError) as error:
        selection_scores({"decoder.layer": row}, "reap_dispatched")
    assert error.value.code == "INSUFFICIENT_EXPERT_COVERAGE"
    assert "decoder.layer" in str(error.value) and "[1]" in str(error.value)
    row = {"expert_output_count": [2, 1], "dispatched_count": [2, 1], "reap_dispatched": [1.5, 1.0]}
    assert selection_scores({"x": row}, "reap_dispatched")["x"] == [1.5, 1.0]
    with pytest.raises(CompilerError) as error:
        selection_scores({"x": row}, "reap_dispatched", 2)
    assert error.value.code == "INSUFFICIENT_EXPERT_COVERAGE"
    with pytest.raises(CompilerError) as error:
        validate_lineage({}, {}, {"experiment_config": {"audit_only": True}}, "bfloat16")
    assert error.value.code == "AUDIT_ONLY_ARTIFACT"


def test_reap_real_post_capacity_conditional_mean():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.51.3":
        pytest.skip("Pinned native runtime")
    from transformers import SwitchTransformersConfig
    from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersSparseMLP
    config = SwitchTransformersConfig(d_model=2, d_ff=2, num_experts=2, expert_capacity=1,
                                      router_jitter_noise=0, dropout_rate=0)
    model = torch.nn.Module()
    model.config = config
    model.layer = SwitchTransformersSparseMLP(config)
    class Scale(torch.nn.Module):
        def __init__(self, factor):
            super().__init__()
            self.factor = factor
        def forward(self, value):
            return value * self.factor
    model.layer.experts = torch.nn.ModuleDict({"expert_0": Scale(2), "expert_1": Scale(3)})
    with torch.no_grad():
        model.layer.router.classifier.weight.copy_(torch.tensor([[10., 0.], [0., 10.]]))
    model.eval()
    with RoutingTelemetry(model, torch, reap=True) as telemetry:
        model.layer(torch.tensor([[[1., 0.], [1., 0.], [0., 1.]]]))
    row = telemetry.result()["layer"]
    assert row["top1_count"] == [2, 1]
    assert row["dispatched_count"] == row["expert_output_count"] == [1, 1]
    assert row["dropped_positions"] == 1
    gate = torch.softmax(torch.tensor([10., 0.]), dim=-1)[0].item()
    assert row["reap_dispatched"] == pytest.approx([gate * 2, gate * 3])
    assert row["ean_dispatched"] == pytest.approx([2, 3])
    assert not model.layer.router._forward_hooks
    # Native BF16 probability cast ties the two gates; analytic FP32 favors 1.
    with torch.no_grad():
        model.layer.router.classifier.weight.copy_(torch.tensor([[0., 0.], [.0001, 0.]]))
    with RoutingTelemetry(model, torch, reap=True) as telemetry:
        model.layer(torch.tensor([[[1., 0.]]], dtype=torch.bfloat16))
    row = telemetry.result()["layer"]
    assert row["top1_count"] == [1, 0]
    assert row["analytic_fp32_top1_count"] == [0, 1]
    assert row["postcast_vs_analytic_disagreements"] == 1
    assert row["reap_dispatched"][1] is None


@pytest.fixture
def tiny_source(tmp_path, request):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    pytest.importorskip("safetensors")
    tokenizers = pytest.importorskip("tokenizers")
    if transformers.__version__ != "4.51.3":
        pytest.skip("Pinned native runtime")
    from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration, PreTrainedTokenizerFast
    torch.manual_seed(8)
    config = SwitchTransformersConfig(vocab_size=12, d_model=8, d_ff=16, d_kv=4, num_heads=2,
        num_layers=2, num_decoder_layers=2, num_sparse_encoder_layers=1, num_sparse_decoder_layers=1,
        num_experts=getattr(request, "param", 4), expert_capacity=8, dropout_rate=0, router_jitter_noise=0,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    model = SwitchTransformersForConditionalGeneration(config)
    for _, layer in sparse_layers(model):
        with torch.no_grad():
            layer.router.classifier.weight[0, 0] = 0.123456789
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"<pad>": 0, "</s>": 1, "<unk>": 2, "hello": 3, "world": 4, "good": 5, "day": 6,
         "<extra_id_0>": 7, "<extra_id_1>": 8, "answer": 9, "other": 10, "text": 11}, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="</s>",
                                       additional_special_tokens=["<extra_id_0>", "<extra_id_1>"])
    source = tmp_path / "tiny-random-mechanics"
    model.save_pretrained(source, safe_serialization=True)
    tokenizer.save_pretrained(source)
    assert (source / "model.safetensors").stat().st_size < 1_000_000
    del model, tokenizer
    gc.collect()
    return source


def test_restores_original_f32_not_upcast_rounded(tiny_source):
    from safetensors import safe_open
    model, tokenizer, torch, plan = load_model(tiny_source, inspect_model(tiny_source), "bfloat16", None, True)
    restored = plan["router_precision"]["tensors"]
    assert restored
    with safe_open(str(tiny_source / "model.safetensors"), framework="pt") as handle:
        for name in restored:
            original = handle.get_tensor(name)
            actual = dict(model.named_parameters())[name]
            assert actual.dtype == torch.float32
            assert torch.equal(actual, original)
            assert not torch.equal(original, original.bfloat16().float())


def public_cli(args):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    proc = subprocess.run([sys.executable, "-m", "asea.compiler"] + args, env=env,
                          text=True, capture_output=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    receipt = json.loads(proc.stdout)
    assert len(proc.stdout.encode()) < 8192
    assert receipt["stdout_contract"] == "bounded_receipt_v1" and "runs" not in receipt
    from asea.compiler.core import digest
    assert receipt["report_sha256"] == digest(receipt["report"])
    assert receipt["report_bytes"] == Path(receipt["report"]).stat().st_size
    return {**json.loads(Path(receipt["report"]).read_text()), "output": receipt["output"]}


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_public_diagnose_roundtrip_fresh_native_mechanics(tiny_source, tmp_path, dtype):
    before = inventory(tiny_source)
    suite = tmp_path / "diagnostics.json"
    suite.write_text(json.dumps({"schema": SCHEMA, "split": "diagnostic", "cases": [
        {"id": "mechanics", "prompt": "hello <extra_id_0>", "target": "<extra_id_0> world <extra_id_1>",
         "expected_spans": ["world"], "references": ["<extra_id_0> world <extra_id_1>"]}]}))
    common = ["--model", str(tiny_source), "--suite", str(suite), "--dtype", dtype,
              "--max-length", "8", "--max-new-tokens", "2", "--teacher-tokens", "8", "--preserve-router-fp32"]
    audit = public_cli(["diagnose", "--output", str(tmp_path / (dtype + ".json"))] + common)
    assert audit["status"] == "AUDIT_ONLY" and audit["admission"] == "UNADMITTED" and not audit["pruned"]
    comparison = audit["comparisons"]["native_vs_wrapper"]
    assert comparison["all_generation_ids_equal"] and comparison["all_short_logits_within_tolerance"]
    assert audit["runs"]["native"]["resources"]["pid"] != audit["runs"]["wrapper"]["resources"]["pid"]
    case = audit["runs"]["wrapper"]["cases"][0]
    assert case["teacher_forced"]["target_token_ids"][-1] == 1
    assert case["teacher_forced"]["numeric"]["all_finite"]
    assert "_logits_file" not in case["teacher_forced"]
    output = tmp_path / (dtype + "-identity")
    audit = public_cli(["roundtrip", "--output", str(output)] + common)
    assert audit["removed_parameters"] == 0 and audit["source_parameters"] == audit["roundtrip_parameters"]
    assert audit["num_experts"] == 4 and not (output / "compiler_manifest.json").exists()
    assert all(c["all_generation_ids_equal"] and c["all_short_logits_within_tolerance"] for c in audit["comparisons"].values())
    assert len({r["resources"]["pid"] for r in audit["runs"].values()}) == 3
    assert all(v["retained_source_indices"] == [0, 1, 2, 3] for v in audit["identity_mapping"].values())
    assert inventory(tiny_source) == before
    assert not list(tmp_path.glob(".pass2-*"))
    with pytest.raises(CompilerError) as error:
        prune_model(tiny_source, tmp_path / "illegal-keep-total", 4, suite, dtype=dtype)
    assert error.value.code == "INVALID_KEEP"


def test_diagnostic_paths_reject_before_loading(tiny_source, tmp_path, capsys):
    assert main(["diagnose", "--model", str(tiny_source), "--suite", "missing", "--output", str(tiny_source / "inside.json"), "--dtype", "bfloat16"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["type"] == "UNSAFE_OUTPUT"


@pytest.mark.parametrize("ids", [
    [0, 1, 2, 5, 3, 1],  # EOS before content
    [0, 0, 2, 5, 3, 1],  # repeated decoder start
    [0, 2, 5, 3, 1, 1],  # repeated EOS
    [0, 2, 5, 3, 1, 6],  # content after EOS
    [0, 2, 5, 3, 0, 1],  # padding before EOS
    [0, 2, 5, 3],        # complete spans, incomplete generation
    [2, 5, 3, 1],        # missing decoder start
    [0, 2, 0, 5, 3, 1], # padding inside span
])
def test_strict_complete_generation_boundaries(ids):
    trace = token_trace(ids, TokenizerMechanics(), ["Paris"], 16)
    assert not trace["parsed"]["complete_generation_well_formed"]
    assert not trace["complete_output_correct"]
    if ids == [0, 1, 2, 5, 3, 1]:
        assert not trace["expected_spans_correct"]
        assert trace["termination"] == "invalid_eos_order"


@pytest.mark.parametrize("ids", [[0, 2, 999, 3, 1], [0, 2, True, 3, 1], [0, 2, {}, 3, 1], "not IDs"])
def test_malformed_token_ids_fail_closed(ids):
    assert not parse_spans(ids, TokenizerMechanics(), 1)["valid"]
    with pytest.raises(CompilerError):
        token_trace(ids, TokenizerMechanics(), ["Paris"], 16)


def test_terminal_padding_and_explicit_generation_policy():
    trace = token_trace([0, 2, 5, 3, 1, 0, 0], TokenizerMechanics(), ["Paris"], 16)
    assert trace["complete_output_correct"]
    trace = token_trace([0, 2, 5, 3, 6], TokenizerMechanics(), ["Paris"], 16, [6])
    assert trace["complete_output_correct"] and trace["termination"] == "eos"
    assert not token_trace([0, 2, 5, 3, 1], TokenizerMechanics(), ["Paris"], 16, [])["complete_output_correct"]
    disabled = token_trace([0, 2, 5, 3, 1], TokenizerMechanics(), ["Paris"], 16, None)
    assert disabled["generation_eos_token_ids"] == [] and not disabled["complete_output_correct"]


@pytest.mark.parametrize("change", [
    {"target": "<extra_id_0> London <extra_id_1>"},
    {"target": "man"}, {"target": "<extra_id_1> man <extra_id_0>"},
    {"target": "<extra_id_0> man <extra_id_1> extra"},
    {"prompt": "Missing required sentinel"},
    {"expected_spans": ["man", "other"]},
])
def test_contradictory_teacher_objectives_rejected_before_model(change):
    row = {**DEFAULT_SUITE["cases"][0], **change}
    with pytest.raises(CompilerError) as exc:
        validate_suite({**DEFAULT_SUITE, "cases": [row]})
    assert exc.value.code == "INVALID_DIAGNOSTIC_SUITE"


def test_tokenized_teacher_and_prompt_contract(tiny_source):
    from transformers import AutoTokenizer
    from asea.compiler.diagnostics import validate_tokenized_case
    tokenizer = AutoTokenizer.from_pretrained(tiny_source, local_files_only=True)
    row = {"expected_spans": ["world"]}
    validate_tokenized_case(row, tokenizer, [3, 7], [3, 7], [7, 4, 8, 1])
    for full, truncated, target in [([3, 7], [3], [7, 4, 8, 1]),
            ([3, 7], [3, 7], [7, 10, 8, 1]), ([3, 7], [3, 7], [1, 7, 4, 8, 1]),
            ([3, 7], [3, 7], [7, 4, 1])]:
        with pytest.raises(CompilerError):
            validate_tokenized_case(row, tokenizer, full, truncated, target)


def test_bf16_api_and_cli_never_reach_v1_evaluator(monkeypatch, capsys):
    import asea.compiler.core as core
    monkeypatch.setattr(core, "evaluate_model", lambda **kw: pytest.fail("evaluator reached"))
    with pytest.raises(CompilerError) as exc:
        core.certify_model("missing", "missing", "missing", "missing", dtype="bfloat16")
    assert exc.value.code == "INVALID_DTYPE"
    rc = main(["certify", "--model", "missing", "--reference-model", "missing", "--suite", "missing",
               "--certificate-output", "missing", "--dtype", "bfloat16"])
    assert rc == 2
    assert json.loads(capsys.readouterr().out)["status"] == "REJECTED"


@pytest.mark.parametrize("manifest", [
    {"preflight": {"dtype": "bfloat16"}, "experiment_config": {"audit_only": False}},
    {"experiment_config": {"preserve_router_fp32": True, "audit_only": False}},
    {"experiment_config": {"scorer": "reap_dispatched", "audit_only": False}},
    {"schema": "switch-identity-roundtrip-v2", "experiment_config": {"audit_only": False}},
])
def test_research_flags_cannot_silently_become_v1(manifest):
    with pytest.raises(CompilerError) as exc:
        validate_lineage({}, {}, manifest, "float16")
    assert exc.value.code == "AUDIT_ONLY_ARTIFACT"


def test_research_bf16_prune_is_audit_only_and_binds_reference(tiny_source, tmp_path, monkeypatch):
    from asea.compiler.core import canonical_hash
    from asea.compiler.diagnostics import precision_binding, diagnose_model
    import asea.compiler.diagnostics as diagnostics
    before = inspect_model(tiny_source)
    calibration = tmp_path / "cal.json"
    calibration.write_text(json.dumps({"samples": [{"prompt": "hello", "target": "world"}]}))
    candidate = tmp_path / "bf16-pruned"
    result = prune_model(tiny_source, candidate, 2, calibration, dtype="bfloat16", max_length=8)
    assert result["audit_only"] and result["status"] == "AUDIT_ONLY"
    manifest = json.loads((candidate / "compiler_manifest.json").read_text())
    assert manifest["experiment_config"]["audit_only"]
    assert inventory(tiny_source) == before["files"]
    precision_binding(candidate, "bfloat16", False, before)
    descendant = tmp_path / "half-reprune-of-research"
    inherited = prune_model(candidate, descendant, 1, calibration, dtype="float16", max_length=8)
    assert inherited["status"] == "AUDIT_ONLY" and inherited["audit_only"]
    assert json.loads((descendant / "compiler_manifest.json").read_text())["experiment_config"]["source_research_only"]
    with pytest.raises(CompilerError) as exc:
        precision_binding(candidate, "float16", False, before)
    assert exc.value.code == "DTYPE_MISMATCH"
    unrelated = {"files": {}, "artifact_sha256": canonical_hash({})}
    with pytest.raises(CompilerError) as exc:
        precision_binding(candidate, "bfloat16", False, unrelated)
    assert exc.value.code == "REFERENCE_MISMATCH"
    suite = tmp_path / "diag.json"
    suite.write_text(json.dumps(DEFAULT_SUITE))
    monkeypatch.setattr(diagnostics, "invoke_worker", lambda *a, **k: pytest.fail("worker reached"))
    with pytest.raises(CompilerError) as exc:
        diagnose_model(candidate, suite, tmp_path / "never.json", "float16", reference_model=tiny_source)
    assert exc.value.code == "DTYPE_MISMATCH"
    manifest["candidate"]["artifact_sha256"] = "0" * 64
    (candidate / "compiler_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(CompilerError) as exc:
        precision_binding(candidate, "bfloat16", False, before)
    assert exc.value.code == "LINEAGE_MISMATCH"


@pytest.mark.parametrize("code,expected", [
    ("import time; time.sleep(5)", "DIAGNOSTIC_WORKER_TIMEOUT"),
    ("import os; os.write(1, b'x'*100000)", "DIAGNOSTIC_WORKER_OUTPUT_LIMIT"),
    ("import os; os.write(2, b'x'*100000)", "DIAGNOSTIC_WORKER_OUTPUT_LIMIT"),
    ("import os; os.write(1, b'x'*600); os.write(2, b'x'*600)", "DIAGNOSTIC_WORKER_OUTPUT_LIMIT"),
])
def test_worker_wall_and_both_output_streams_are_bounded(code, expected):
    from asea.compiler.diagnostics import bounded_process
    with pytest.raises(CompilerError) as exc:
        bounded_process([sys.executable, "-c", code], dict(os.environ), .3, 1024)
    assert exc.value.code == expected


def test_worker_small_output_is_hashed_not_reemitted(capsys):
    from asea.compiler.core import canonical_hash
    from asea.compiler.diagnostics import bounded_process
    rc, tail, logs = bounded_process([sys.executable, "-c", "import sys; print('ok'); print('warning', file=sys.stderr)"], dict(os.environ), 5, 1024)
    assert rc == 0 and tail == "warning\n"
    assert logs["stdout"]["bytes"] == 3 and logs["stderr"]["bytes"] == 8
    assert len(logs["stdout"]["sha256"]) == 64
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("limits", [(float("inf"), 1024, 8192), (float("nan"), 1024, 8192),
    (0, 1024, 8192), (1, 0, 8192), (1, 1024, None), (1, 1024, 99999)])
def test_worker_limits_cannot_be_disabled(limits):
    from asea.compiler.diagnostics import worker_limits
    with pytest.raises(CompilerError):
        worker_limits(*limits)


def test_public_worker_limits_and_report_file_ceiling(tmp_path, monkeypatch):
    import asea.compiler.diagnostics as diagnostics
    args = parser().parse_args(["roundtrip", "--model", "local", "--output", "new", "--dtype", "float16",
        "--worker-timeout-seconds", "10", "--worker-output-limit-bytes", "4096", "--worker-address-space-mib", "2048"])
    assert (args.worker_timeout_seconds, args.worker_output_limit_bytes, args.worker_address_space_mib) == (10, 4096, 2048)
    monkeypatch.setattr(diagnostics, "MAX_REPORT_BYTES", 128)
    path = tmp_path / "report.json"
    with pytest.raises(CompilerError) as exc:
        diagnostics.bounded_write_json(path, {"too_large": "x" * 200})
    assert exc.value.code == "DIAGNOSTIC_REPORT_LIMIT" and not path.exists()
    diagnostics.bounded_write_json(path, {"ok": True})
    with pytest.raises(FileExistsError):
        diagnostics.bounded_write_json(path, {"no_overwrite": True})
    assert json.loads(path.read_text()) == {"ok": True}


def test_worker_rehash_before_load(tiny_source, tmp_path, monkeypatch):
    import asea.compiler.diagnostics as diagnostics
    source = inspect_model(tiny_source)
    config = {"model": str(tiny_source), "expected_model_files": {}, "dtype": "float16", "preserve_router_fp32": False}
    monkeypatch.setattr(diagnostics, "runtime", lambda: pytest.fail("runtime reached"))
    with pytest.raises(CompilerError) as exc:
        diagnostics.worker(config)
    assert exc.value.code == "INPUT_CHANGED"
    assert inventory(tiny_source) == source["files"]


def test_worker_rehash_after_cases_detects_mutation(tiny_source, tmp_path, monkeypatch):
    import asea.compiler.diagnostics as diagnostics
    original = diagnostics.run_cases
    def mutate_after_inference(*args):
        result = original(*args)
        (tiny_source / "changed-during-worker.txt").write_text("mutation fixture")
        return result
    monkeypatch.setattr(diagnostics, "run_cases", mutate_after_inference)
    config = {"model": str(tiny_source), "expected_model_files": inventory(tiny_source),
              "dtype": "float16", "preserve_router_fp32": False, "memory_budget_mib": None,
              "loader": "native", "workdir": str(tmp_path), "max_length": 8, "max_new_tokens": 2,
              "teacher_tokens": 8, "suite": {"schema": SCHEMA, "split": "diagnostic", "cases": [
                  {"id": "mutation-mechanics", "prompt": "hello <extra_id_0>",
                   "target": "<extra_id_0> world <extra_id_1>", "expected_spans": ["world"]}]}}
    with pytest.raises(CompilerError) as exc:
        diagnostics.worker(config)
    assert exc.value.code == "SOURCE_CHANGED"


def test_large_report_has_compact_hash_receipt(tmp_path):
    from asea.compiler.core import canonical_hash, digest
    from asea.compiler.diagnostics import bounded_write_json, diagnostic_receipt
    report = {"schema": SCHEMA, "command": "diagnose", "status": "AUDIT_ONLY", "admission": "UNADMITTED",
              "pruned": False, "quality_certification": False, "certifies_coding_skills": False,
              "source_artifact_sha256": "0" * 64, "source_unchanged": True, "dtype": "bfloat16",
              "preserve_router_fp32": False, "runs": {"native": {"cases": ["x" * 65536] * 32}},
              "comparisons": {"native_vs_wrapper": {"all_generation_ids_equal": True,
                  "all_short_logits_within_tolerance": True, "quality_certification": False}}}
    report["evidence_sha256"] = canonical_hash(report)
    output = tmp_path / "large-report.json"
    bounded_write_json(output, report)
    receipt = diagnostic_receipt({**report, "output": str(output)})
    assert output.stat().st_size > 1024 * 1024
    assert len(json.dumps(receipt).encode()) < 4096 and "runs" not in receipt
    assert receipt["report_sha256"] == digest(output)
    assert not receipt["logit_arrays_in_stdout"]


@pytest.mark.parametrize("tiny_source", [8], indirect=True)
def test_keep_all_eight_is_only_identity_audit(tiny_source, tmp_path):
    from asea.compiler.core import canonical_hash
    before = inventory(tiny_source)
    suite = tmp_path / "keep8-suite.json"
    suite.write_text(json.dumps({"schema": SCHEMA, "split": "diagnostic", "cases": [
        {"id": "keep8-mechanics", "prompt": "hello <extra_id_0>",
         "target": "<extra_id_0> world <extra_id_1>", "expected_spans": ["world"]}]}))
    output = tmp_path / "keep8-identity"
    report = public_cli(["roundtrip", "--model", str(tiny_source), "--output", str(output),
                        "--suite", str(suite), "--dtype", "bfloat16", "--max-length", "8",
                        "--max-new-tokens", "2", "--teacher-tokens", "8"])
    assert report["status"] == "AUDIT_ONLY" and report["admission"] == "UNADMITTED"
    assert not report["quality_certification"] and not report["certifies_coding_skills"] and not report["pruned"]
    assert report["num_experts"] == 8 and report["removed_parameters"] == 0
    assert report["source_parameters"] == report["roundtrip_parameters"]
    assert report["source_artifact_sha256"] == canonical_hash(before)
    assert inventory(tiny_source) == before
    manifest = json.loads((output / "roundtrip_manifest.json").read_text())
    with pytest.raises(CompilerError) as exc:
        validate_lineage({}, {}, manifest, "float16")
    assert exc.value.code == "AUDIT_ONLY_ARTIFACT"

