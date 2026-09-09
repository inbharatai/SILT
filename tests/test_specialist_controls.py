"""Tiny native mechanical tests only; no downloads, teacher, training or quality study.

CONTROL_PROPOSAL_MODULE_PATH imports the isolated draft by exact absolute path.
After adoption, omit it to import the installed standalone module normally.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from asea.artifacts import Blocked, model_inventory


@pytest.fixture(scope="module")
def controls():
    path = os.environ.get("CONTROL_PROPOSAL_MODULE_PATH")
    if path:
        assert Path(path).is_absolute() and Path(path).is_file()
        spec = importlib.util.spec_from_file_location("silt_random_mlp_control_proposal", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert Path(module.__file__).resolve() == Path(path).resolve()
        return module
    from asea.specialist import controls as module
    return module


@pytest.fixture(scope="module")
def ml():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    pytest.importorskip("tokenizers")
    torch.set_num_threads(1)
    return torch, transformers


def _source(tmp_path, ml, dtype="float32", tied=True, sharded=False):
    torch, tf = ml
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        config = tf.Qwen2Config(vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=32, tie_word_embeddings=tied,
            initializer_range=0.037, eos_token_id=1, pad_token_id=0)
        model = tf.Qwen2ForCausalLM(config).to(getattr(torch, dtype))
    source = tmp_path / "source"
    model.save_pretrained(source, safe_serialization=True, max_shard_size="3KB" if sharded else "1GB")
    tok = Tokenizer(WordLevel({"<pad>": 0, "</s>": 1, "<unk>": 2, "hello": 3}, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    tf.PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="</s>", unk_token="<unk>").save_pretrained(source)
    (source / "reconstruction_manifest.json").write_text(json.dumps({"status": "RECONSTRUCTED_UNVALIDATED"}))
    return source, model


def _stored(path):
    from safetensors import safe_open
    state = {}
    for file in sorted(path.glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as handle:
            # Detach fixture tensors from mmap before any test rewrites a source file.
            state.update({key: handle.get_tensor(key).clone() for key in handle.keys()})
    return state


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("tied,sharded", [(True, False), (False, True)])
def test_exact_native_shape_backbone_sizes_and_offline_reload(tmp_path, controls, ml, dtype, tied, sharded):
    torch, tf = ml
    source, original = _source(tmp_path, ml, dtype, tied, sharded)
    before_inventory = model_inventory(source)
    output = tmp_path / "control"
    rng_before = torch.get_rng_state().clone()
    manifest = controls.randomize_mlp_control(source, output)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert model_inventory(source) == before_inventory
    assert manifest == json.loads((output / "random_mlp_control_manifest.json").read_text())
    assert manifest["status"] == "BASELINE_RANDOM_MLP_UNVALIDATED"
    assert manifest["counts"]["source"] == manifest["counts"]["output"]
    assert manifest["counts"]["source"]["parameters"] == sum(p.numel() for p in original.parameters())
    assert manifest["counts"]["randomized_mlp_tensors"] == 6
    assert manifest["method"]["initializer_range"] == 0.037
    assert not manifest["method"]["fallback_for_requested_model"]
    assert not manifest["method"]["skill_extraction_claimed"]
    assert manifest["omitted_source_ancillary_files"] == ["reconstruction_manifest.json"]
    assert not (output / "reconstruction_manifest.json").exists()
    before, after = _stored(source), _stored(output)
    assert set(before) == set(after) == set(manifest["tensor_provenance"])
    for key, actual in after.items():
        expected = before[key]
        randomized = ".mlp." in key
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert torch.equal(actual, expected) != randomized, key
        row = manifest["tensor_provenance"][key]
        assert row["changed"] == randomized
        assert (row["before_sha256"] != row["after_sha256"]) == randomized
        assert row["owner"] == ("random_mlp_control" if randomized else "inherited_shared_backbone")
        if randomized:
            # Independent exact reconstruction of the documented RNG recipe.
            import hashlib
            value = ("silt-random-mlp-v1\0" + "17\0" + key).encode()
            seed = int.from_bytes(hashlib.sha256(value).digest()[:8], "little") & ((1 << 63) - 1)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            draw = torch.empty(actual.numel(), dtype=torch.float32).normal_(0, 0.037, generator=generator)
            assert torch.equal(actual, draw.to(actual.dtype).reshape(actual.shape))
    for name, value in manifest["output"]["files"].items():
        assert value["size"] == before_inventory[name]["size"]
        if not name.endswith(".safetensors"):
            assert value == before_inventory[name]
        assert (source / name).stat().st_ino != (output / name).stat().st_ino
    source.rename(tmp_path / "source-offline")
    model, loading = tf.AutoModelForCausalLM.from_pretrained(output, local_files_only=True,
        trust_remote_code=False, use_safetensors=True, torch_dtype=getattr(torch, dtype), output_loading_info=True)
    assert type(model) is tf.Qwen2ForCausalLM
    assert not any(loading[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in original.parameters())
    assert (model.model.embed_tokens.weight is model.lm_head.weight) == tied
    tokenizer = tf.AutoTokenizer.from_pretrained(output, local_files_only=True, trust_remote_code=False)
    batch = tokenizer("hello", return_tensors="pt")
    batch.pop("token_type_ids", None)
    with torch.inference_mode():
        assert torch.isfinite(model(**batch).logits).all()  # Tiny serialization smoke check, not quality.
    assert not manifest["verification"]["checkpoint_load_performed"]
    assert not manifest["verification"]["teacher_loaded"]
    assert not manifest["verification"]["quality_pass"]


def test_seed_reproducibility_and_shard_independence(tmp_path, controls, ml):
    torch, _ = ml
    roots = [tmp_path / name for name in ("single", "sharded")]
    for root in roots:
        root.mkdir()
    single, _ = _source(roots[0], ml)
    sharded, _ = _source(roots[1], ml, sharded=True)
    for source, name, seed in [(single, "a", 17), (single, "b", 17), (single, "c", 18), (sharded, "d", 17)]:
        controls.randomize_mlp_control(source, tmp_path / name, seed=seed)
    a, b, c, d = [_stored(tmp_path / name) for name in "abcd"]
    for key in a:
        assert torch.equal(a[key], b[key]) and torch.equal(a[key], d[key])
        assert torch.equal(a[key], c[key]) == (".mlp." not in key)


def test_no_actual_checkpoint_or_teacher_loader(tmp_path, controls, ml, monkeypatch):
    _, tf = ml
    source, _ = _source(tmp_path, ml)
    def forbidden(*args, **kwargs):
        raise AssertionError("real checkpoint/tokenizer loader forbidden in control construction")
    monkeypatch.setattr(tf.PreTrainedModel, "from_pretrained", forbidden)
    monkeypatch.setattr(tf.AutoTokenizer, "from_pretrained", forbidden)
    manifest = controls.randomize_mlp_control(source, tmp_path / "out")
    assert manifest["verification"]["strict_native_meta_keys_shapes_and_aliases"]


@pytest.mark.parametrize("seed", [True, -1, 2**63, 1.1, "17"])
def test_invalid_seed_fails_without_output(tmp_path, controls, seed):
    with pytest.raises(controls.ControlBlocked, match="seed"):
        controls.randomize_mlp_control(tmp_path / "missing", tmp_path / "out", seed)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("std", [None, 0, -0.02, True, "0.02", 1e309])
def test_invalid_config_std_no_fallback(tmp_path, controls, ml, std):
    source, _ = _source(tmp_path, ml)
    config_path = source / "config.json"
    value = json.loads(config_path.read_text())
    if std is None:
        del value["initializer_range"]
    else:
        value["initializer_range"] = std
    config_path.write_text(json.dumps(value))
    with pytest.raises(Blocked):
        controls.randomize_mlp_control(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mutation", ["missing", "shape", "unexpected", "dtype"])
def test_invalid_tensor_state_never_fills_missing_with_random(tmp_path, controls, ml, mutation):
    torch, _ = ml
    source, _ = _source(tmp_path, ml, tied=False)
    state = _stored(source)
    key = "model.layers.0.mlp.up_proj.weight"
    if mutation == "missing":
        del state[key]
    elif mutation == "shape":
        state[key] = torch.ones(1)
    elif mutation == "unexpected":
        state["bogus.weight"] = torch.ones(1)
    else:
        state[key] = state[key].to(torch.float16)
    from safetensors.torch import save_file
    save_file(state, source / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(Blocked):
        controls.randomize_mlp_control(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_non_qwen_rejected_not_substituted(tmp_path, controls, ml):
    source, _ = _source(tmp_path, ml)
    path = source / "config.json"
    value = json.loads(path.read_text())
    value.update(model_type="t5", architectures=["T5ForConditionalGeneration"])
    path.write_text(json.dumps(value))
    with pytest.raises(controls.ControlBlocked, match="Qwen2 only; no fallback"):
        controls.randomize_mlp_control(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_paths_symlinks_existing_output_and_cleanup(tmp_path, controls, ml, monkeypatch):
    source, _ = _source(tmp_path, ml)
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel"
    sentinel.write_text("keep")
    for output in (existing, source, source / "nested", tmp_path / "absent" / "out"):
        with pytest.raises(Blocked):
            controls.randomize_mlp_control(source, output)
    linked = tmp_path / "linked"
    linked.symlink_to(source, target_is_directory=True)
    with pytest.raises(Blocked):
        controls.randomize_mlp_control(linked, tmp_path / "out")
    def failed(*args, **kwargs):
        raise OSError("injected write failure")
    monkeypatch.setattr(controls, "_replace_tensor", failed)
    with pytest.raises(OSError, match="injected"):
        controls.randomize_mlp_control(source, tmp_path / "out")
    assert sentinel.read_text() == "keep"
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".*.control-stage-*"))


def test_unchanged_random_tensor_and_publication_race_fail_closed(tmp_path, controls, ml, monkeypatch):
    source, _ = _source(tmp_path, ml)
    before = model_inventory(source)
    original = controls._replace_tensor
    monkeypatch.setattr(controls, "_replace_tensor", lambda *args: None)
    with pytest.raises(controls.ControlBlocked, match="MLP must change"):
        controls.randomize_mlp_control(source, tmp_path / "out")
    monkeypatch.setattr(controls, "_replace_tensor", original)
    publish = controls._publish
    def race(stage, output):
        output.mkdir()
        (output / "sentinel").write_text("winner")
        return publish(stage, output)
    monkeypatch.setattr(controls, "_publish", race)
    with pytest.raises(Blocked, match="refusing replacement"):
        controls.randomize_mlp_control(source, tmp_path / "out")
    assert (tmp_path / "out" / "sentinel").read_text() == "winner"
    assert model_inventory(source) == before
    assert not list(tmp_path.glob(".*.control-stage-*"))


@pytest.mark.parametrize("dtype,bytes_per_element", [("F32", 4), ("BF16", 2)])
def test_bounded_writer_crosses_chunk_boundary_without_touching_neighbors(tmp_path, controls, ml, dtype, bytes_per_element):
    torch, _ = ml
    count = controls._CHUNK_ELEMENTS + 7
    path = tmp_path / "payload"
    path.write_bytes(b"prefix" + b"\0" * (count * bytes_per_element) + b"suffix")
    entry = {"numel": count, "dtype": dtype}
    controls._replace_tensor(path, 6, entry, 0.037, 17, "some.mlp.weight", torch)
    raw = path.read_bytes()
    assert raw[:6] == b"prefix" and raw[-6:] == b"suffix"
    assert len(raw) == 12 + count * bytes_per_element
    generator = torch.Generator(device="cpu").manual_seed(controls._tensor_seed(17, "some.mlp.weight"))
    target_dtype = torch.float32 if dtype == "F32" else torch.bfloat16
    chunks = [torch.empty(n).normal_(0, 0.037, generator=generator).to(target_dtype)
              for n in (controls._CHUNK_ELEMENTS, 7)]
    expected = torch.cat(chunks).view(torch.uint8).numpy().tobytes()
    assert raw[6:-6] == expected


def test_source_mutation_during_write_refuses_publication(tmp_path, controls, ml, monkeypatch):
    source, _ = _source(tmp_path, ml)
    original = controls._replace_tensor
    def mutation(*args):
        result = original(*args)
        path = source / "config.json"
        path.write_bytes(path.read_bytes() + b"\n")
        return result
    monkeypatch.setattr(controls, "_replace_tensor", mutation)
    with pytest.raises(controls.ControlBlocked, match="source changed"):
        controls.randomize_mlp_control(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".*.control-stage-*"))


def test_mixed_original_storage_dtypes_preserved(tmp_path, controls, ml):
    torch, _ = ml
    source, _ = _source(tmp_path, ml, dtype="bfloat16", tied=False)
    state = _stored(source)
    state["model.norm.weight"] = state["model.norm.weight"].float()
    state["model.layers.0.mlp.up_proj.weight"] = state["model.layers.0.mlp.up_proj.weight"].float()
    from safetensors.torch import save_file
    save_file(state, source / "model.safetensors", metadata={"format": "pt"})
    manifest = controls.randomize_mlp_control(source, tmp_path / "out")
    actual = _stored(tmp_path / "out")
    for key in state:
        assert actual[key].dtype == state[key].dtype
        assert torch.equal(actual[key], state[key]) == (".mlp." not in key)
    assert manifest["counts"]["source"] == manifest["counts"]["output"]


def test_memory_admission_precedes_meta(tmp_path, controls, ml, monkeypatch):
    source, _ = _source(tmp_path, ml)
    monkeypatch.setattr(controls, "_memory_budget", lambda: {"limit_bytes": 1})
    monkeypatch.setattr(controls, "_native_meta", lambda *args: pytest.fail("meta called before admission"))
    with pytest.raises(controls.ControlBlocked, match="memory admission"):
        controls.randomize_mlp_control(source, tmp_path / "out")


def test_import_without_ml_and_standalone_cli(tmp_path, controls, ml):
    path = str(Path(controls.__file__).resolve())
    code = ("import importlib.util,sys; s=importlib.util.spec_from_file_location('draft', %r); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules") % path
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True)
    source, _ = _source(tmp_path, ml)
    result = subprocess.run([sys.executable, path, "--input-student", str(source),
        "--output", str(tmp_path / "out"), "--seed", "17"], env=env, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["status"] == "BASELINE_RANDOM_MLP_UNVALIDATED"
    result = subprocess.run([sys.executable, path, "--input-student", str(source),
        "--output", str(tmp_path / "out")], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert json.loads(result.stderr)["status"] == "BLOCKED"
