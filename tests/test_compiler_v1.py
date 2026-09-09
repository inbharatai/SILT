"""Compiler safety tests; optional random Switch tests are MECHANICS ONLY.

No downloaded model, real-task accuracy claim, or package-level model mock.
"""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from asea.compiler.core import (
    ARCH, CompilerError, apply_selection, bounds, canonical_hash, inspect_model,
    inventory, load_samples, preflight, prune_model, safe_path, score_generations,
    validate_config,
)
from asea.compiler.__main__ import main


def write_json(path, data):
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def header_fixture(tmp_path):
    """Header-validation fixture, explicitly NOT a usable model checkpoint."""
    root = tmp_path / "header-only"
    root.mkdir()
    write_json(root / "config.json", {"architectures": [ARCH], "model_type": "switch_transformers", "num_experts": 4})
    header = json.dumps({"fixture_tensor": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    return root


def test_header_inspection_is_lightweight(header_fixture):
    result = inspect_model(header_fixture)
    assert result["weights"]["stored_parameters"] == 1
    assert result["num_experts"] == 4
    assert result["admission"] == "UNADMITTED"
    assert result["repair_training"]["supported"] is False
    assert len(result["artifact_sha256"]) == 64


@pytest.mark.parametrize("config", [{}, {"model_type": "llama", "architectures": [ARCH]},
    {"model_type": "switch_transformers", "architectures": [ARCH], "num_experts": 4, "auto_map": {"AutoModel": "evil.code"}}])
def test_unsupported_config(config):
    with pytest.raises(CompilerError):
        validate_config(config)


def test_symlinks_rejected(header_fixture, tmp_path):
    link = tmp_path / "linked"
    link.symlink_to(header_fixture, target_is_directory=True)
    with pytest.raises(CompilerError, match="Symlinks"):
        safe_path(link / "config.json")
    (header_fixture / "linked-weight").symlink_to(header_fixture / "model.safetensors")
    with pytest.raises(CompilerError, match="symlink"):
        inventory(header_fixture)


@pytest.mark.parametrize("keep", [0, -1, 4, 5, True])
def test_invalid_keep_before_runtime(header_fixture, tmp_path, keep):
    with pytest.raises(CompilerError) as exc:
        prune_model(header_fixture, tmp_path / "candidate", keep, tmp_path / "absent")
    assert exc.value.code == "INVALID_KEEP"


def test_output_rejected_before_runtime(header_fixture, tmp_path):
    with pytest.raises(CompilerError) as exc:
        prune_model(header_fixture, header_fixture / "child", 2, "missing")
    assert exc.value.code == "UNSAFE_OUTPUT"
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(CompilerError) as exc:
        prune_model(header_fixture, existing, 2, "missing")
    assert exc.value.code == "OUTPUT_EXISTS"


def test_memory_bound_and_lengths():
    with pytest.raises(CompilerError) as exc:
        preflight({"weights": {"stored_parameters": 1_000_000_000}}, "float32", 512)
    assert exc.value.code == "MEMORY_PREFLIGHT"
    for length, new in [(0, 1), (257, 1), (1, 129)]:
        with pytest.raises(CompilerError):
            bounds(length, new)


def test_calibration_requires_decoder_target(tmp_path):
    path = write_json(tmp_path / "cal.json", {"samples": [{"prompt": "text"}]})
    with pytest.raises(CompilerError) as exc:
        load_samples(path, "calibration")
    assert exc.value.code == "INVALID_DATASET"


def test_suite_requires_heldout_and_references(tmp_path):
    path = write_json(tmp_path / "suite.json", {"cases": [{"id": "a", "prompt": "text", "references": ["answer"]}]})
    with pytest.raises(CompilerError):
        load_samples(path, "suite")
    write_json(path, {"split": "heldout", "cases": [{"id": "a", "prompt": "text", "references": ["answer"]}]})
    assert len(load_samples(path, "suite")[1]) == 1


def test_exact_match_not_functional_execution():
    result = score_generations([{"id": "x", "prompt": "x", "references": ["hello"]}], ["hello "])
    assert result["exact_match"] == 0
    assert result["metric"] == "literal_exact_match_text_proxy"


def test_manifest_tampering(header_fixture):
    files = inventory(header_fixture)
    write_json(header_fixture / "compiler_manifest.json", {"candidate": {"files": files}})
    inspect_model(header_fixture)
    (header_fixture / "new-file").write_text("changed")
    with pytest.raises(CompilerError) as exc:
        inspect_model(header_fixture)
    assert exc.value.code == "LINEAGE_MISMATCH"


def test_typed_cli_errors(capsys):
    assert main(["infer", "--model", "/missing", "--prompt", "test", "--max-new-tokens", "0"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["type"] == "INVALID_ARGUMENT"
    assert main(["prune", "--model", "/missing", "--output", "/none", "--calibration", "/none", "--keep-experts", "1", "--repair-training"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["error"]["type"] == "UNSUPPORTED_REPAIR"
    assert result["supported"] is False
    assert main(["unknown"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["type"] == "INVALID_ARGUMENT"


@pytest.fixture
def random_switch(tmp_path):
    """Tiny RANDOM initialization with local word tokenizer: mechanics, not skill evidence."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    pytest.importorskip("safetensors")
    tokenizers = pytest.importorskip("tokenizers")
    if transformers.__version__ != "4.51.3":
        pytest.skip("Supported compiler runtime is transformers 4.51.3")
    from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration, PreTrainedTokenizerFast
    torch.manual_seed(7)
    config = SwitchTransformersConfig(vocab_size=12, d_model=8, d_ff=16, d_kv=4, num_heads=2,
        num_layers=2, num_decoder_layers=2, num_sparse_encoder_layers=1, num_sparse_decoder_layers=1,
        num_experts=4, expert_capacity=8, dropout_rate=0, router_jitter_noise=0,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    model = SwitchTransformersForConditionalGeneration(config)
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"<pad>": 0, "</s>": 1, "<unk>": 2, "hello": 3, "world": 4, "good": 5, "day": 6,
         "test": 7, "heldout": 8, "answer": 9, "other": 10, "text": 11}, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="</s>")
    source = tmp_path / "random-mechanics-source"
    model.save_pretrained(source, safe_serialization=True)
    tokenizer.save_pretrained(source)
    return source, model, torch


def test_random_mechanics_retains_dense_and_router_rows(random_switch):
    source, model, torch = random_switch
    from asea.compiler.core import sparse_layers
    dense = model.encoder.block[0].layer[-1].mlp.wi.weight
    dense_copy = dense.detach().clone()
    saved = {}
    stats = {}
    for name, layer in sparse_layers(model):
        saved[name] = (layer.experts["expert_1"], layer.experts["expert_3"], layer.router.classifier.weight.detach().clone())
        stats[name] = {"probability_sum": [0, 3, 1, 4]}
    selection = apply_selection(model, torch, stats, 2)
    assert model.config.num_experts == 2
    assert torch.equal(dense, dense_copy)
    for name, layer in sparse_layers(model):
        assert layer.experts["expert_0"] is saved[name][0]
        assert layer.experts["expert_1"] is saved[name][1]
        assert torch.equal(layer.router.classifier.weight, saved[name][2][[1, 3]])
        assert selection[name]["retained_source_indices"] == [1, 3]


def test_random_mechanics_cli_roundtrip_fresh_process(random_switch, tmp_path):
    source, model, torch = random_switch
    del model
    before = inventory(source)
    calibration = write_json(tmp_path / "cal.json", {"samples": [{"prompt": "hello world", "target": "good day"}]})
    output = tmp_path / "candidate"
    result = prune_model(source, output, 2, calibration, max_length=8)
    assert result["candidate_parameters"] < result["source_parameters"]
    assert inventory(source) == before
    manifest = json.loads((output / "compiler_manifest.json").read_text())
    assert manifest["source"]["files"] == before
    assert manifest["source_after_sha256"] == canonical_hash(before)
    assert all(layer["routing"]["tokens"] > 0 for layer in manifest["layers"].values())
    # Fresh interpreter with source path removed: standalone candidate is sufficient.
    source.rename(tmp_path / "source-unavailable")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"), HF_HUB_OFFLINE="1")
    proc = subprocess.run([sys.executable, "-m", "asea.compiler", "infer", "--model", str(output),
                           "--prompt", "heldout text", "--max-new-tokens", "2", "--max-length", "8"],
                          env=env, text=True, capture_output=True, timeout=90)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert isinstance(json.loads(proc.stdout)["generation"], str)
    from asea.compiler.core import evaluate_model
    suite = write_json(tmp_path / "suite.json", {"split": "heldout", "cases": [{"id": "h1", "prompt": "heldout text", "references": ["answer"]}]})
    evidence = evaluate_model(output, suite, max_length=8, max_new_tokens=2, evidence_output=tmp_path / "evidence.json")
    assert evidence["certifies_coding_skills"] is False
    assert evidence["admission"] == "UNADMITTED"
    assert "generation" in evidence["result"]["cases"][0]
    write_json(suite, {"split": "heldout", "cases": [{"id": "h1", "prompt": "hello world", "references": ["answer"]}]})
    with pytest.raises(CompilerError) as exc:
        evaluate_model(output, suite, max_length=8, max_new_tokens=2)
    assert exc.value.code == "CALIBRATION_LEAKAGE"
