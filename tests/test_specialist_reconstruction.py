"""Mechanical checks only; new memory tests use metadata/constant tensors, no real teachers."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from asea.specialist import ReconstructionBlocked, reconstruct
from asea.specialist.reconstruction import _admit, _publish
from asea.artifacts import model_inventory


def _ml():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    torch.set_num_threads(2)
    return torch, transformers


def _tokenizer(path, transformers):
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    tok = tokenizers.Tokenizer(WordLevel({"<pad>": 0, "</s>": 1, "<unk>": 2, "write": 3, "code": 4, "hello": 5, "return": 6, "one": 7}, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    from tokenizers.processors import TemplateProcessing
    tok.post_processor = TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
    transformers.PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="</s>", unk_token="<unk>").save_pretrained(path)


def _source(tmp_path, family="qwen2", tied=True):
    torch, tf = _ml()
    torch.manual_seed(12)
    path = tmp_path / "source"
    if family == "qwen2":
        config = tf.Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=192,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128, tie_word_embeddings=tied, eos_token_id=1, pad_token_id=0)
        model = tf.Qwen2ForCausalLM(config)
    else:
        config = tf.SwitchTransformersConfig(vocab_size=32, d_model=16, d_kv=4, d_ff=64,
            num_layers=2, num_decoder_layers=2, num_heads=4, num_experts=3,
            num_sparse_encoder_layers=1, num_sparse_decoder_layers=1,
            decoder_start_token_id=0, eos_token_id=1, pad_token_id=0,
            dropout_rate=0, router_jitter_noise=0, tie_word_embeddings=tied)
        model = tf.SwitchTransformersForConditionalGeneration(config)
    model.save_pretrained(path, safe_serialization=True, max_shard_size="8KB")
    _tokenizer(path, tf)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"samples": [{"prompt": "write code", "response": "return one"}, {"prompt": "hello code"}]}))
    return path, calibration, model


@pytest.mark.parametrize("method", ["activation", "magnitude", "uniform"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_qwen_structural_native_weights_and_standalone(tmp_path, method, dtype):
    torch, tf = _ml()
    source, calibration, teacher = _source(tmp_path)
    before = model_inventory(source)
    output = tmp_path / "specialist"
    manifest = reconstruct(source, output, calibration, method=method, dtype=dtype, max_length=16)
    assert model_inventory(source) == before
    assert manifest == json.loads((output / "reconstruction_manifest.json").read_text())
    assert manifest["status"] == "RECONSTRUCTED_UNVALIDATED"
    assert manifest["output"]["standalone"] and not manifest["output"]["teacher_required_at_serve"]
    assert manifest["counts"]["output"]["parameters"] < manifest["counts"]["source"]["parameters"]
    assert manifest["counts"]["output"]["safetensors_bytes"] < manifest["counts"]["source"]["safetensors_bytes"]
    assert manifest["verification"]["source_loading_strict"]
    admission = manifest["memory_admission"]
    assert admission["materialized_source_count"]["parameters"] == admission["header_numel"]
    assert admission["materialized_source_count"]["tensor_bytes"] == admission["header_loaded_bytes"]
    assert [s["stage"] for s in admission["stage_measurements"]] == [
        "pre_load", "post_load_pre_observation", "post_transform_pre_forward", "post_save"]
    if method == "activation":
        assert manifest["verification"]["hook_parity"] == "exact_logits_first_sample"
        assert set(manifest["verification"]["observed_tokens_by_layer"].values()) == {7}
    teacher_state = teacher.to(getattr(torch, dtype)).state_dict()
    # Prove output reload has no source dependency, rather than trusting a flag.
    source.rename(tmp_path / "offline-teacher")
    reloaded, loading = tf.AutoModelForCausalLM.from_pretrained(output, local_files_only=True, trust_remote_code=False,
        torch_dtype=getattr(torch, dtype), output_loading_info=True)
    assert not loading["missing_keys"] and not loading["unexpected_keys"]
    assert type(reloaded) is tf.Qwen2ForCausalLM
    assert reloaded.config.intermediate_size == 128
    for key, actual in reloaded.state_dict().items():
        provenance = manifest["weights_provenance"][key]
        expected = teacher_state[provenance["source_key"]]
        if provenance["operation"] == "index_select":
            indices = manifest["selected_indices"][provenance["selected_indices_ref"]]
            assert len(indices) == len(set(indices)) == 128
            assert indices == sorted(indices)
            expected = expected.index_select(provenance["axis"], torch.tensor(indices))
        assert torch.equal(actual, expected), key
    tok = tf.AutoTokenizer.from_pretrained(output, local_files_only=True, trust_remote_code=False)
    batch = tok("write code", return_tensors="pt")
    batch.pop("token_type_ids", None)
    with torch.inference_mode():
        assert torch.isfinite(reloaded(**batch).logits).all()
        assert reloaded.generate(**batch, max_new_tokens=2).shape[-1] > batch["input_ids"].shape[-1]
    assert not list(output.glob("*.bin"))


@pytest.mark.parametrize("method", ["activation", "magnitude", "uniform"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_switch_top1_to_real_dense_t5_exact_weight_mapping(tmp_path, method, dtype):
    torch, tf = _ml()
    source, calibration, teacher = _source(tmp_path, "switch_transformers")
    before = model_inventory(source)
    output = tmp_path / "dense"
    manifest = reconstruct(source, output, calibration, method=method, dtype=dtype, max_length=16)
    assert model_inventory(source) == before
    assert manifest["output"]["architecture"] == "T5ForConditionalGeneration"
    assert manifest["method"]["initialization"] == "structural_initialized_requires_repair_and_evaluation"
    assert not manifest["method"]["teacher_output_parity_claimed"]
    assert manifest["method"]["experts_per_sparse_layer"] == 1
    assert len(manifest["selected_indices"]) == 2
    source.rename(tmp_path / "offline-teacher")
    target, loading = tf.AutoModelForSeq2SeqLM.from_pretrained(output, local_files_only=True, trust_remote_code=False,
        torch_dtype=getattr(torch, dtype), output_loading_info=True)
    assert type(target) is tf.T5ForConditionalGeneration
    assert not loading["missing_keys"] and not loading["unexpected_keys"]
    assert target.config.d_ff == 64
    state = teacher.to(getattr(torch, dtype)).state_dict()
    for key, tensor in target.state_dict().items():
        assert "router" not in key and "experts" not in key
        assert torch.equal(tensor, state[manifest["weights_provenance"][key]["source_key"]]), key
    retained = {entry["source_key"] for entry in manifest["weights_provenance"].values()}
    removed = {entry["source_key"] for entry in manifest["removed_source_tensors"]}
    assert retained.isdisjoint(removed)
    assert retained | removed == set(state)
    for block in list(target.encoder.block) + list(target.decoder.block):
        assert block.layer[-1].DenseReluDense.wi.out_features == 64
    assert manifest["reduction"]["parameters_removed"] > 0
    with torch.inference_mode():
        assert torch.isfinite(target(input_ids=torch.tensor([[3, 4]]), decoder_input_ids=torch.tensor([[0, 6]])).logits).all()
        assert target.generate(input_ids=torch.tensor([[3, 4]]), max_new_tokens=2).shape[-1] >= 2


def test_activation_is_native_swiglu_contribution_and_instrumentation_inert(tmp_path):
    torch, tf = _ml()
    from asea.specialist.reconstruction import _observe
    source, calibration, model = _source(tmp_path)
    model.eval()
    tok = tf.AutoTokenizer.from_pretrained(source, local_files_only=True)
    rows = json.loads(calibration.read_text())["samples"]
    scores = {"model.layers.%d.mlp" % i: torch.zeros(192) for i in range(2)}
    _observe(model, tok, rows, "qwen2", 16, scores, torch)
    # Independently recompute native SwiGLU once per calibration sample.
    explicit = {key: torch.zeros(192) for key in scores}
    counts = {key: 0 for key in scores}
    handles = []
    for i, layer in enumerate(model.model.layers):
        def hook(module, args, key="model.layers.%d.mlp" % i):
            x = args[0]
            product = torch.nn.functional.silu(module.gate_proj(x)) * module.up_proj(x)
            explicit[key] += product.detach().float().abs().sum((0, 1))
            counts[key] += x.shape[0] * x.shape[1]
        handles.append(layer.mlp.register_forward_pre_hook(hook))
    from asea.specialist.reconstruction import _batches
    with torch.inference_mode():
        for batch in _batches(tok, rows, "qwen2", model.config, 16, torch):
            model(**batch, use_cache=False)
    for handle in handles:
        handle.remove()
    for key in scores:
        torch.testing.assert_close(scores[key], explicit[key] / counts[key], rtol=0, atol=0)


def test_memory_guard_refuses_before_ml_allocation():
    with pytest.raises(ReconstructionBlocked, match="memory admission"):
        _admit({"huge": {"numel": 2_000_000_000}}, {"vocab_size": 32, "d_ff": 64}, "bfloat16", 128, memory_budget_bytes=4 * 1024**3)


def test_import_has_no_ml_dependency():
    root = str(Path(__file__).resolve().parents[1] / "src")
    code = "import sys; sys.path.insert(0, %r); import asea.specialist; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules" % root
    subprocess.run([sys.executable, "-c", code], check=True)


def test_paths_and_no_replace(tmp_path):
    _ml()
    source, calibration, _ = _source(tmp_path)
    with pytest.raises(ReconstructionBlocked, match="new and disjoint"):
        reconstruct(source, source / "child", calibration)
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "untouched"
    marker.write_text("sentinel")
    with pytest.raises(ReconstructionBlocked):
        reconstruct(source, existing, calibration)
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(ReconstructionBlocked, match="refusing replacement"):
        _publish(stage, existing)
    assert marker.read_text() == "sentinel" and stage.is_dir()


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "shape"])
def test_invalid_weight_state_fails_not_randomly_initialized(tmp_path, mutation):
    torch, tf = _ml()
    source, calibration, teacher = _source(tmp_path, tied=False)
    # Replace sharded fixture with one deliberately invalid safe state file.
    for path in source.glob("*.safetensors*"):
        path.unlink()
    from safetensors.torch import save_file
    state = {key: tensor.clone() for key, tensor in teacher.state_dict().items()}
    if mutation == "missing":
        del state["model.layers.0.mlp.up_proj.weight"]
    elif mutation == "unexpected":
        state["made_up.weight"] = torch.ones(1)
    else:
        state["model.layers.0.mlp.up_proj.weight"] = torch.ones(64, 16)
    save_file(state, source / "model.safetensors", metadata={"format": "pt"})
    output = tmp_path / "must-not-exist"
    with pytest.raises(ReconstructionBlocked, match="missing source|unmatched source|source shape"):
        reconstruct(source, output, calibration, dtype="float32")
    assert not output.exists()


def test_failed_save_removes_stage_without_publishing(tmp_path, monkeypatch):
    torch, tf = _ml()
    source, calibration, _ = _source(tmp_path)
    def broken(*args, **kwargs):
        raise RuntimeError("test disk failure")
    monkeypatch.setattr(tf.Qwen2ForCausalLM, "save_pretrained", broken)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="test disk failure"):
        reconstruct(source, output, calibration, method="uniform", dtype="float32")
    assert not output.exists()
    assert not list(tmp_path.glob(".output.stage-*"))


def test_unsupported_and_invalid_retention(tmp_path):
    _ml()
    source, calibration, _ = _source(tmp_path)
    with pytest.raises(ReconstructionBlocked, match="unsupported family"):
        reconstruct(source, tmp_path / "x", calibration, family="llama")
    with pytest.raises(ReconstructionBlocked, match="multiple of 64"):
        reconstruct(source, tmp_path / "x", calibration, retention=0.01, dtype="float32")
    assert not (tmp_path / "x").exists()


def test_shared_calibration_exact_native_chat_response_and_prompt_only(tmp_path):
    from asea.specialist.reconstruction import _batches, _calibration_encoding
    from asea.specialist.recovery import _encode
    torch, tf = _ml()
    source, _, model = _source(tmp_path)
    tokenizer = tf.AutoTokenizer.from_pretrained(source, local_files_only=True)
    tokenizer.chat_template = "{% for m in messages %}{% if m['role']=='user' %}{{ 'user ' + m['content'] + '\nassistant ' }}{% else %}{{ m['content'] + eos_token + '\n' }}{% endif %}{% endfor %}"
    row = {"id": "native", "prompt": "write code", "response": "return one"}
    expected = _encode(row, tokenizer, "causal", 256)
    batch = next(_batches(tokenizer, [row], "qwen2", model.config, 256, torch))
    assert batch["input_ids"].tolist() == [expected["input_ids"]]
    assert batch["attention_mask"].tolist() == [expected["attention_mask"]]
    prompt_only = {"id": "unlabeled", "prompt": row["prompt"]}
    example, evidence = _calibration_encoding(tokenizer, prompt_only, "qwen2", 256)
    prefix = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True)
    assert example["input_ids"] == tokenizer.encode(prefix, add_special_tokens=False)
    assert evidence["prompt_only"]
    with pytest.raises(ReconstructionBlocked, match="no truncation"):
        _calibration_encoding(tokenizer, row, "qwen2", 4)


def test_calibration_rejects_overlong_unselected_row_before_weights(tmp_path, monkeypatch):
    torch, tf = _ml()
    source, calibration, _ = _source(tmp_path)
    calibration.write_text(json.dumps({"samples": [
        {"id": "first", "prompt": "write code", "response": "return one"},
        {"id": "unselected-too-long", "prompt": "hello " * 300}]}))
    def forbidden(*args, **kwargs):
        pytest.fail("weights loaded before complete calibration validation")
    monkeypatch.setattr(tf.Qwen2ForCausalLM, "from_pretrained", forbidden)
    with pytest.raises(ReconstructionBlocked, match="unselected-too-long"):
        reconstruct(source, tmp_path / "bad", calibration, max_length=256, max_samples=1)


def test_original_f32_router_values_restored_not_rounded_upcast(tmp_path):
    from asea.specialist.reconstruction import _artifact_preflight, _restore_f32_routers
    torch, tf = _ml()
    source, _, original = _source(tmp_path, "switch_transformers")
    _, _, headers = _artifact_preflight(source)
    loaded = tf.SwitchTransformersForConditionalGeneration.from_pretrained(
        source, local_files_only=True, use_safetensors=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True)
    routers = [name for name in headers if name.endswith("router.classifier.weight")]
    assert routers
    assert any(not torch.equal(loaded.get_parameter(n).float(), original.get_parameter(n)) for n in routers)
    receipt = _restore_f32_routers(loaded, source, headers, torch)
    assert len(receipt["restored"]) == len(routers)
    for name in routers:
        assert loaded.get_parameter(name).dtype == torch.float32
        assert torch.equal(loaded.get_parameter(name), original.get_parameter(name))
    assert all(entry["tensor_sha256"] for entry in receipt["restored"])


def test_memory_budget_automatic_large_host_and_no_override(monkeypatch):
    from asea.specialist.reconstruction import _memory_budget
    headers = {"large": {"numel": 2_000_000_000, "dtype": "BF16"}}
    config = {"vocab_size": 32, "d_ff": 64}
    monkeypatch.setattr("asea.specialist.reconstruction._available_ram", lambda: 32 * 1024**3)
    accepted = _admit(headers, config, "bfloat16", 256)
    assert accepted["estimated_peak_bytes"] > 4 * 1024**3
    assert accepted["requested_memory_budget_bytes"] is None
    with pytest.raises(ReconstructionBlocked, match="memory admission"):
        _admit(headers, config, "bfloat16", 256, memory_budget_bytes=4 * 1024**3)
    monkeypatch.setattr("asea.specialist.reconstruction._available_ram", lambda: 3 * 1024**3)
    assert _memory_budget(64 * 1024**3)["limit_bytes"] < 3 * 1024**3
    with pytest.raises(ReconstructionBlocked, match="memory admission"):
        _admit(headers, config, "bfloat16", 256, memory_budget_bytes=64 * 1024**3)
    for bad in (0, -1, True, 1.5, "4096"):
        with pytest.raises(ReconstructionBlocked, match="positive integer"):
            _memory_budget(bad)


def test_available_memory_accounts_for_process_cgroup_ancestors(monkeypatch):
    from asea.specialist.reconstruction import _available_ram
    files = {"/proc/meminfo": "MemAvailable: 33554432 kB\n",
             "/proc/self/cgroup": "0::/jobs/worker\n",
             "/proc/self/mountinfo": "1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
             "/sys/fs/cgroup/memory.max": "max", "/sys/fs/cgroup/jobs/worker/memory.max": str(8 * 1024**3),
             "/sys/fs/cgroup/jobs/worker/memory.current": str(1024**3),
             "/sys/fs/cgroup/jobs/memory.max": str(4 * 1024**3),
             "/sys/fs/cgroup/jobs/memory.current": str(2 * 1024**3)}
    def read(path, *args, **kwargs):
        if str(path) not in files:
            raise FileNotFoundError(str(path))
        return files[str(path)]
    monkeypatch.setattr(Path, "read_text", read)
    assert _available_ram() == 2 * 1024**3


def test_duplicate_safetensors_header_key_is_rejected(tmp_path):
    import struct
    from asea.specialist.reconstruction import _headers
    payload = b'{"x":{"shape":[1],"dtype":"F32","data_offsets":[0,4]},"x":{"shape":[1],"dtype":"F32","data_offsets":[0,4]}}'
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(payload)) + payload + b"0000")
    with pytest.raises(ReconstructionBlocked, match="invalid safetensors header"):
        _headers(tmp_path, {"model.safetensors": {}})


def _metadata_plan(headers, dtype="bfloat16"):
    return {"strategy": "explicit_safetensors_tensor_meta_assign",
            "target_dtypes": {key: dtype for key in headers}, "native_keep_in_fp32_modules": []}


def test_same_dtype_metadata_budget_mmap_once_without_runtime_overlap(monkeypatch):
    import asea.specialist.reconstruction as r
    monkeypatch.setattr(r, "_available_ram", lambda: 3_113_062_400)
    counts = [151936 * 896]
    remainder = 494_032_768 - counts[0]
    counts.extend([remainder // 24] * 23 + [remainder - 23 * (remainder // 24)])
    headers = {str(i): {"numel": count, "dtype": "BF16", "file": "model.safetensors"}
               for i, count in enumerate(counts)}
    config = {"model_type": "qwen2", "vocab_size": 151936, "hidden_size": 896,
              "intermediate_size": 4864, "num_hidden_layers": 24, "num_attention_heads": 14}
    admitted = _admit(headers, config, "bfloat16", 220, load_plan=_metadata_plan(headers),
                      runtime_rss_bytes=232_808_448)
    assert admitted["header_loaded_bytes"] == admitted["source_payload_bytes"] == 988_065_536
    assert admitted["load_cast_transient_bytes"] == 0
    assert admitted["replacement_mlp_bytes"] == 470_679_552  # ALL replacements, mmap originals can survive
    assert admitted["estimated_peak_bytes"] == 2_081_606_912
    assert admitted["runtime_remaining_bytes"] == 512 * 1024**2 - 232_808_448
    assert admitted["limit_bytes"] == 2_801_756_160
    assert admitted["estimated_peak_bytes"] < admitted["limit_bytes"]
    for ceiling in (1, 1024**3):
        with pytest.raises(ReconstructionBlocked, match="memory admission"):
            _admit(headers, config, "bfloat16", 220, ceiling, load_plan=_metadata_plan(headers))
    with pytest.raises(ReconstructionBlocked, match="memory admission"):
        _admit(headers, config, "bfloat16", 220)  # unknown loader gets no mmap credit
    with pytest.raises(ReconstructionBlocked, match="memory admission"):
        _admit(headers, config, "float32", 220, load_plan=_metadata_plan(headers, "float32"))


def test_cast_plan_conservative_pages_and_truly_big_model(monkeypatch):
    import asea.specialist.reconstruction as r
    monkeypatch.setattr(r, "_available_ram", lambda: 32 * 1024**3)
    headers = {"a": {"numel": 100_000_000, "dtype": "F32", "file": "one.safetensors"},
               "b": {"numel": 50_000_000, "dtype": "F32", "file": "two.safetensors"},
               "c": {"numel": 25_000_000, "dtype": "BF16", "file": "one.safetensors"}}
    plan = _metadata_plan(headers)
    value = _admit(headers, {}, "bfloat16", 1, load_plan=plan, runtime_rss_bytes=1024**3)
    assert value["header_loaded_bytes"] == 350_000_000
    assert value["load_cast_transient_bytes"] == 600_000_000  # not unproven largest-shard-only
    assert value["largest_cast_tensor_bytes"] == 400_000_000
    assert value["largest_shard_bytes"] == 450_000_000
    assert value["runtime_remaining_bytes"] == 0
    assert [(row["key"], row["source_bytes"], row["target_bytes"]) for row in value["cast_tensor_plan"]] == [
        ("a", 400_000_000, 200_000_000), ("b", 200_000_000, 100_000_000)]
    headers["a"]["numel"] = 20_000_000_000
    with pytest.raises(ReconstructionBlocked, match="memory admission"):
        _admit(headers, {}, "bfloat16", 1, load_plan=plan)


def test_future_workspace_does_not_recharge_loaded_weights(monkeypatch):
    import asea.specialist.reconstruction as r
    monkeypatch.setattr(r, "_available_ram", lambda: 700 * 1024**2)
    monkeypatch.setattr(r, "_rss_bytes", lambda: 1700 * 1024**2)
    admission = {"header_loaded_bytes": 1000 * 1024**2, "future_workspace_bytes": 300 * 1024**2,
                 "runtime_remaining_bytes": 0, "replacement_mlp_bytes": 200 * 1024**2}
    r._future_admit(admission, "post_load", None)
    r._future_admit(admission, "post_transform", None, replacements_allocated=True)
    assert [s["future_incremental_bytes"] for s in admission["stage_measurements"]] == [300 * 1024**2, 100 * 1024**2]
    assert admission["stage_measurements"][0]["rss_bytes"] == 1700 * 1024**2
    with pytest.raises(ReconstructionBlocked, match="future workspace"):
        r._future_admit(admission, "post_load", 100 * 1024**2)


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_explicit_load_constant_storage_alias_and_dtype(tmp_path, monkeypatch, dtype):
    torch, tf = _ml()
    from accelerate import init_empty_weights
    import safetensors
    from safetensors.torch import save_file
    from asea.specialist.reconstruction import _headers, _load_plan, _load_source, _count
    config = tf.Qwen2Config(vocab_size=8, hidden_size=8, intermediate_size=64,
                           num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                           tie_word_embeddings=True)
    with init_empty_weights(include_buffers=False):
        meta = tf.Qwen2ForCausalLM(config)
    meta.tie_weights()
    state = {name: torch.full(tuple(p.shape), 0.125, dtype=torch.bfloat16) for name, p in meta.named_parameters()}
    save_file(state, tmp_path / "model.safetensors")
    headers = _headers(tmp_path, {"model.safetensors": {}})
    plan = _load_plan(meta, headers, dtype)
    assert plan["native_keep_in_fp32_modules"] == []
    assert plan["aliases"]["lm_head.weight"] == "model.embed_tokens.weight"
    original_open, pointers = safetensors.safe_open, {}
    class TrackedOpen:
        def __init__(self, *args, **kwargs):
            self.handle = original_open(*args, **kwargs)
        def __enter__(self):
            self.handle.__enter__()
            return self
        def __exit__(self, *args):
            return self.handle.__exit__(*args)
        def get_tensor(self, key):
            value = self.handle.get_tensor(key)
            pointers[key] = value.data_ptr()
            return value
    monkeypatch.setattr(safetensors, "safe_open", TrackedOpen)
    loaded = _load_source(meta, tmp_path, headers, plan, torch)
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    for key, expected in state.items():
        actual = loaded.get_parameter(key)
        assert torch.equal(actual, expected.to(getattr(torch, dtype)))
        assert (actual.data_ptr() == pointers[key]) == (dtype == "bfloat16")
    assert all(not v.is_meta for v in list(loaded.parameters()) + list(loaded.buffers()))
    assert _count(loaded)["tensor_bytes"] == sum(h["numel"] for h in headers.values()) * (2 if dtype == "bfloat16" else 4)


def test_switch_wo_not_blanket_f32_but_native_t5_policy_is():
    torch, tf = _ml()
    from asea.specialist.reconstruction import _load_plan
    for family in ("switch", "t5"):
        config_cls, model_cls = ((tf.SwitchTransformersConfig, tf.SwitchTransformersForConditionalGeneration)
                                if family == "switch" else (tf.T5Config, tf.T5ForConditionalGeneration))
        config = config_cls(vocab_size=8, d_model=8, d_ff=16, d_kv=4, num_heads=2,
                            num_layers=2, num_decoder_layers=2, num_experts=2,
                            num_sparse_encoder_layers=1, num_sparse_decoder_layers=1)
        with torch.device("meta"):
            meta = model_cls(config)
        headers = {k: {"shape": list(p.shape), "numel": p.numel(), "dtype": "BF16", "file": "x"}
                   for k, p in meta.named_parameters()}
        router = next((k for k in headers if k.endswith("router.classifier.weight")), None)
        if router:
            headers[router]["dtype"] = "F32"
        plan = _load_plan(meta, headers, "bfloat16")
        wo = [d for k, d in plan["target_dtypes"].items() if k.endswith(".wo.weight")]
        assert wo and set(wo) == ({"bfloat16"} if family == "switch" else {"float32"})
        if router:
            assert plan["target_dtypes"][router] == "float32"


def test_observe_sequential_full_logits_cache_restored_on_success_and_error(monkeypatch):
    import weakref
    from types import SimpleNamespace
    torch, _ = _ml()
    import asea.specialist.reconstruction as r
    monkeypatch.setattr(r, "_batches", lambda *args: iter([{}, {}, {}]))
    class NativeProbe:
        config = SimpleNamespace(use_cache=True)
        prior = None
        calls = 0
        def named_modules(self):
            return []
        def __call__(self, **kwargs):
            assert kwargs["use_cache"] is False and self.config.use_cache is False
            assert self.prior is None or self.prior() is None
            logits = torch.full((1, 4, 8), 0.125, dtype=torch.bfloat16)
            self.prior = weakref.ref(logits)
            self.calls += 1
            return SimpleNamespace(logits=logits)
    model = NativeProbe()
    result = r._observe(model, None, [{}, {}, {}], "switch_transformers", 4, {}, torch)
    assert model.config.use_cache is True and model.calls == 4
    assert "sequential" in result["parity_comparison"]
    class BrokenProbe(NativeProbe):
        def __call__(self, **kwargs):
            assert self.config.use_cache is False
            raise RuntimeError("probe failed")
    broken = BrokenProbe()
    with pytest.raises(RuntimeError, match="probe failed"):
        r._observe(broken, None, [{}], "switch_transformers", 4, {}, torch)
    assert broken.config.use_cache is True
