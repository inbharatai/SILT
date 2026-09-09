"""Real tiny RANDOM native-model mechanics; no pretrained quality claim/downloads."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import (PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM,
                          SwitchTransformersConfig, SwitchTransformersForConditionalGeneration,
                          T5Config, T5ForConditionalGeneration, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, AutoTokenizer)
from asea.specialist.recovery import (recover, _encode, _family, _full_kl,
                                      _response_logits, _store_hash, _publish_no_replace,
                                      _encode_details, _encoding_contract, _read_samples,
                                      _bounded_json, RecoveryRejected, MAX_DATA_BYTES, MAX_DATA_ROWS)


@pytest.fixture(autouse=True)
def single_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tokenizer_at(directory):
    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, "<bos>": 3,
             "hello": 4, "world": 5, "task": 6, "answer": 7,
             "other": 8, "test": 9, "train": 10, "dev": 11}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                  eos_token="</s>", unk_token="<unk>", bos_token="<bos>")
    tokenizer.save_pretrained(directory)
    return tokenizer


def tiny_stores(tmp_path, family):
    torch.manual_seed(5)
    teacher_dir, student_dir = tmp_path / "teacher", tmp_path / "student"
    teacher_dir.mkdir()
    student_dir.mkdir()
    tokenizer_at(teacher_dir)
    tokenizer_at(student_dir)
    if family == "causal":
        common = dict(vocab_size=12, hidden_size=16, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1,
                      max_position_embeddings=64, bos_token_id=3,
                      eos_token_id=1, pad_token_id=0, attention_dropout=0.0)
        teacher = Qwen2ForCausalLM(Qwen2Config(intermediate_size=32, **common))
        student = Qwen2ForCausalLM(Qwen2Config(intermediate_size=24, **common))
    else:
        common = dict(vocab_size=12, d_model=16, d_kv=8, d_ff=32,
                      num_layers=1, num_decoder_layers=1, num_heads=2,
                      dropout_rate=0.0, decoder_start_token_id=0,
                      eos_token_id=1, pad_token_id=0)
        teacher = SwitchTransformersForConditionalGeneration(SwitchTransformersConfig(
            num_experts=2, num_sparse_encoder_layers=1, num_sparse_decoder_layers=1,
            encoder_sparse_step=1, decoder_sparse_step=1, expert_capacity=8, **common))
        student = T5ForConditionalGeneration(T5Config(**common))
        # Tiny explicit expert->dense mapping establishes a real reconstructed candidate.
        source = teacher.state_dict()
        with torch.no_grad():
            for name, parameter in student.named_parameters():
                switch_name = name.replace(".DenseReluDense.", ".mlp.")
                if switch_name not in source:
                    switch_name = switch_name.replace(".mlp.", ".mlp.experts.expert_0.")
                if switch_name in source and source[switch_name].shape == parameter.shape:
                    parameter.copy_(source[switch_name])
    teacher.save_pretrained(teacher_dir, safe_serialization=True)
    student.save_pretrained(student_dir, safe_serialization=True)
    train, dev = tmp_path / "train.json", tmp_path / "dev.json"
    train.write_text(json.dumps({"samples": [
        {"id": "train-1", "prompt": "hello train", "response": "answer world"},
        {"id": "train-2", "prompt": "task train", "response": "world"}]}))
    dev.write_text(json.dumps({"samples": [
        {"id": "dev-1", "prompt": "hello dev", "response": "other answer"}]}))
    return teacher_dir, student_dir, train, dev


@pytest.mark.parametrize("family", ["causal", "seq2seq"])
def test_real_backward_merge_standalone_reload_and_immutable_stores(tmp_path, family):
    teacher, student, train, dev = tiny_stores(tmp_path, family)
    before_teacher, before_student = _store_hash(teacher), _store_hash(student)
    output = tmp_path / "recovered"
    result = recover(teacher, student, output, train, dev, steps=2,
                     learning_rate=0.01, rank=2, max_length=12, dtype="float32")
    assert result["status"] == "completed", result
    assert result["actual_steps"] == 2
    assert result["teacher_forward_calls"] == 2
    assert result["validation_teacher_forward_calls"] == 1
    assert result["adapter_delta_l2"] > 0
    assert result["adapter_hash_before"] != result["adapter_hash_after"]
    assert result["frozen_parameter_hash_before"] == result["frozen_parameter_hash_after"]
    assert all(item["gradient_norm_pre_clip"] > 0 for item in result["training_history"])
    assert all(item["sample_id"].startswith("train-") for item in result["training_history"])
    assert result["validation_pre"]["response_tokens"] == 3
    assert result["validation_post"]["forward_kl"] is not None
    assert result["trainable_parameters"] < result["total_parameters_with_adapter"]
    assert before_teacher == _store_hash(teacher)
    assert before_student == _store_hash(student)
    assert not (output / "adapter_config.json").exists()
    assert (output / "recovery_report.json").exists()
    # Prove loading/inference cannot use teacher or source student directory.
    teacher.rename(tmp_path / "teacher-hidden")
    student.rename(tmp_path / "student-hidden")
    cls = AutoModelForCausalLM if family == "causal" else AutoModelForSeq2SeqLM
    reloaded = cls.from_pretrained(output, local_files_only=True, use_safetensors=True)
    assert reloaded.config.use_cache is True
    assert not any("lora_" in name for name, _ in reloaded.named_parameters())
    tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    inputs = tokenizer("hello world", return_tensors="pt", return_token_type_ids=False)
    reloaded.eval()
    with torch.no_grad():
        generated = reloaded.generate(**inputs, max_new_tokens=2)
    assert generated.shape[-1] > 0


def test_supervised_baseline_never_forwards_teacher(tmp_path, monkeypatch):
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")
    original = AutoModelForCausalLM.from_pretrained
    loaded = []

    def audited(path, *args, **kwargs):
        loaded.append(Path(path))
        return original(path, *args, **kwargs)

    from asea.specialist import recovery as recovery_module
    original_explicit = recovery_module._load_recovery_model
    def audited_explicit(meta, path, headers, plan, torch):
        loaded.append(Path(path))
        return original_explicit(meta, path, headers, plan, torch)
    monkeypatch.setattr(recovery_module, "_load_recovery_model", audited_explicit)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", audited)
    result = recover(teacher, student, tmp_path / "baseline", train, dev,
                     steps=1, rank=2, dtype="float32", max_length=12,
                     method="supervised_lora", kd_weight=0)
    assert result["status"] == "completed", result
    assert len(loaded) == 2 and loaded[0] == student
    assert loaded[1].name.startswith(".baseline.recovery-")  # independent native export reload
    assert teacher not in loaded
    assert result["teacher_forward_calls"] == result["validation_teacher_forward_calls"] == 0
    assert result["validation_pre"]["forward_kl"] is None


def test_response_mask_shift_padding_and_eos(tmp_path):
    tokenizer = tokenizer_at(tmp_path)
    tokenizer.pad_token = tokenizer.eos_token
    item = _encode({"prompt": "hello world", "response": "answer"}, tokenizer, "causal", 8)
    assert item["labels"] == [-100, -100, 7, 1]
    # A padded EOS is ignored by label masking; the real terminal EOS is retained.
    labels = torch.tensor([item["labels"] + [-100, -100]])
    logits = torch.arange(1 * 6 * 12).reshape(1, 6, 12).float()
    selected, target = _response_logits(logits, labels, "causal")
    assert target.tolist() == [7, 1]
    assert torch.equal(selected, logits[0, 1:3])
    with pytest.raises(RecoveryRejected, match="Complete token length"):
        _encode({"prompt": "hello " * 20, "response": "answer " * 20}, tokenizer, "causal", 5)
    seq = _encode({"prompt": "hello", "response": "answer"}, tokenizer, "seq2seq", 5)
    assert seq["labels"] == [7, 1]


def test_full_vocabulary_kl_matches_dense_and_gradient():
    torch.manual_seed(9)
    student = torch.randn(7, 19, requires_grad=True)
    teacher = torch.randn(7, 19)
    actual = _full_kl(student, teacher, chunk_size=4)
    expected = torch.nn.functional.kl_div(student.log_softmax(-1), teacher.softmax(-1), reduction="batchmean")
    assert torch.allclose(actual, expected, atol=2e-7)
    grad = torch.autograd.grad(actual, student, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, student)[0]
    assert torch.allclose(grad, expected_grad, atol=2e-7)
    assert teacher.grad is None


@pytest.mark.parametrize("leak", ["id", "content"])
def test_split_leakage_rejected_before_load(tmp_path, leak):
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")
    row = json.loads(train.read_text())["samples"][0]
    if leak == "content":
        row["id"] = "different-id"
    else:
        row["response"] = "different answer"
    dev.write_text(json.dumps({"samples": [row]}))
    output = tmp_path / "rejected"
    result = recover(teacher, student, output, train, dev, steps=1)
    assert result["status"] == "rejected"
    assert "leakage" in result["error"]["message"]
    assert not result["artifact_admitted"] and not output.exists()


def test_vocab_mismatch_rejected(tmp_path):
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")
    path = student / "tokenizer.json"
    payload = json.loads(path.read_text())
    vocab = payload["model"]["vocab"]
    vocab["hello"], vocab["world"] = vocab["world"], vocab["hello"]
    path.write_text(json.dumps(payload))
    result = recover(teacher, student, tmp_path / "bad", train, dev, steps=1)
    assert "vocabulary mismatch" in result["error"]["message"]
    assert result["actual_steps"] == 0


def test_bfloat16_distillation_only_backward_and_merge(tmp_path):
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")
    result = recover(teacher, student, tmp_path / "kd-only", train, dev,
                     steps=1, learning_rate=0.01, rank=2, max_length=12,
                     dtype="bfloat16", kd_weight=1.0)
    assert result["status"] == "completed", result
    assert result["merged_native_matrices_changed"] > 0
    assert result["teacher_frozen_no_grad"]
    assert result["training_history"][0]["objective"] == result["training_history"][0]["forward_kl"]
    resources = result["resources"]
    assert resources["lora_optimizer_training_bytes"] == result["trainable_parameters"] * 16
    assert resources["merge_workspace_bytes"] == resources["student"]["largest_lora_target_parameters"] * 8


def test_incomplete_native_checkpoint_rejected(tmp_path):
    from safetensors.torch import load_file, save_file
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")
    weights = student / "model.safetensors"
    state = load_file(weights)
    del state["model.layers.0.mlp.down_proj.weight"]
    save_file(state, weights, metadata={"format": "pt"})
    result = recover(teacher, student, tmp_path / "incomplete", train, dev,
                     steps=1, dtype="float32", max_length=12)
    assert result["status"] == "rejected"
    assert "missing source keys" in result["error"]["message"]
    assert result["actual_steps"] == 0


def test_output_rejects_existing_and_source_overlap(tmp_path):
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")
    for output in (teacher, student, teacher / "nested", student / "nested"):
        result = recover(teacher, student, output, train, dev, steps=1)
        assert result["status"] == "rejected" and result["actual_steps"] == 0


def test_atomic_publish_never_overwrites_existing_empty_directory(tmp_path):
    source, target = tmp_path / "staging", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    with pytest.raises(OSError):
        _publish_no_replace(source, target)
    assert source.exists() and target.exists()


def test_failed_backward_is_rejected_without_artifact(tmp_path, monkeypatch):
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")

    def fail(*args, **kwargs):
        raise RuntimeError("injected optimizer failure")

    monkeypatch.setattr(torch.optim.AdamW, "step", fail)
    output = tmp_path / "failed"
    result = recover(teacher, student, output, train, dev, steps=1, rank=2, dtype="float32", max_length=12)
    assert result["status"] == "rejected", result
    assert "injected optimizer failure" in result["error"]["message"]
    assert not output.exists() and not list(tmp_path.glob(".failed.recovery-*"))
    assert result["source_unchanged"] and result["student_store_unchanged"]


def test_memory_preflight_rejects_before_model_load(tmp_path, monkeypatch):
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")
    monkeypatch.setattr("asea.specialist.recovery._available_ram", lambda: 1)
    result = recover(teacher, student, tmp_path / "no-memory", train, dev, steps=1)
    assert result["status"] == "rejected"
    assert "memory headroom" in result["error"]["message"]
    assert result["actual_steps"] == 0


@pytest.mark.parametrize("pair", [("qwen2", "t5"), ("switch_transformers", "switch_transformers"),
                                   ("llama", "qwen2"), ("t5", "qwen2")])
def test_restrict_native_family_pairs(pair):
    with pytest.raises(ValueError, match="Unsupported"):
        _family(*pair)


# Qwen's default user/assistant framing, tested with an offline character-level
# tokenizer so whitespace and suffix newline masking remain observable.
QWEN_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}{% endif %}"
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def character_tokenizer(chat=True):
    from tokenizers.models import BPE
    vocab = {"<unk>": 0, "<|im_start|>": 1, "<|im_end|>": 2}
    vocab.update({chr(i): len(vocab) + j for j, i in enumerate(range(32, 127))})
    vocab["\n"] = len(vocab)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(BPE(vocab, [], unk_token="<unk>")),
                                        eos_token="<|im_end|>", unk_token="<unk>",
                                        additional_special_tokens=["<|im_start|>"])
    from tokenizers.decoders import Fuse
    tokenizer.backend_tokenizer.decoder = Fuse()
    tokenizer.chat_template = QWEN_CHAT_TEMPLATE if chat else None
    return tokenizer


@pytest.mark.parametrize("response", ["answer", "  answer\n\n", "\nanswer ", "def f():\n    return 1\n"])
def test_complete_qwen_chat_prefix_mask_eos_and_response_bytes(response):
    import hashlib
    tokenizer = character_tokenizer()
    row = {"id": "chat", "prompt": "task", "response": response}
    encoded, evidence = _encode_details(row, tokenizer, "causal", 256)
    user = [{"role": "user", "content": row["prompt"]}]
    prefix = tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(user + [{"role": "assistant", "content": response}],
                                        tokenize=False, add_generation_prompt=False)
    joint = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    assert encoded["input_ids"] == joint["input_ids"]
    assert encoded["input_ids"][:len(prefix_ids)] == prefix_ids
    assert encoded["labels"][:len(prefix_ids)] == [-100] * len(prefix_ids)
    targets = [i for i, label in enumerate(encoded["labels"]) if label != -100]
    assert tokenizer.decode([encoded["labels"][i] for i in targets], clean_up_tokenization_spaces=False) == response + tokenizer.eos_token
    assert encoded["labels"].count(tokenizer.eos_token_id) == 1
    assert encoded["labels"][-1] == -100  # template newline, not response newline
    assert evidence["response_sha256"] == hashlib.sha256(response.encode()).hexdigest()
    assert evidence["effective_prefill_tokens"] == len(prefix_ids)
    assert evidence["response_characters"] == len(response)
    assert evidence["masked_boundary_tokens"] == 0
    assert _encoding_contract(tokenizer, "causal")["template_sha256"] == hashlib.sha256(QWEN_CHAT_TEMPLATE.encode()).hexdigest()


def test_flat_newline_is_joint_encoded_not_separate_concat():
    tokenizer = character_tokenizer(chat=False)
    row = {"prompt": "task", "response": " answer\n"}
    encoded = _encode(row, tokenizer, "causal", 40)
    assert encoded["input_ids"] == tokenizer.encode("task\n answer\n" + tokenizer.eos_token, add_special_tokens=False)
    assert encoded["input_ids"] != (tokenizer.encode("task", add_special_tokens=False)
                                    + tokenizer.encode(row["response"], add_special_tokens=False)
                                    + [tokenizer.eos_token_id])
    assert encoded["labels"][:5] == [-100] * 5
    assert encoded["labels"][-1] == tokenizer.eos_token_id


@pytest.mark.parametrize("chat", [False, True])
def test_boundary_spanning_whitespace_refused_without_losing_first_response_character(chat):
    from tokenizers.models import BPE
    tokenizer = character_tokenizer(chat=chat)
    # A real joint token merges the prefix newline with the response's first
    # space; independently encoded prefix IDs cannot be the training prefix.
    vocab = tokenizer.get_vocab()
    vocab["\n "] = len(vocab)
    tokenizer.backend_tokenizer.model = BPE(vocab, [("\n", " ")], unk_token="<unk>")
    with pytest.raises(RecoveryRejected, match="Incompatible template.*prefix IDs"):
        _encode({"prompt": "task", "response": " answer"}, tokenizer, "causal", 256)


def test_offset_boundary_rejected_even_if_prefix_ids_happen_to_match():
    tokenizer = character_tokenizer(chat=False)
    class BadOffsets:
        def __getattr__(self, name):
            return getattr(tokenizer, name)
        def __call__(self, text, **kwargs):
            result = tokenizer(text, **kwargs)
            offsets = result["offset_mapping"]
            offsets[5] = (4, 6)  # response token spans the prompt delimiter
            return result
    with pytest.raises(RecoveryRejected, match="boundary-spanning token"):
        _encode({"prompt": "task", "response": "answer"}, BadOffsets(), "causal", 40)


@pytest.mark.parametrize("bad", ["trim", "suffix", "prefix"])
def test_incompatible_chat_templates_fail_named(bad):
    tokenizer = character_tokenizer()
    if bad == "trim":
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE.replace("message['content']", "message['content'] | trim")
    elif bad == "suffix":
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE + "ROLE_CAP"
    else:
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE.replace("{% if add_generation_prompt %}", "{% if false %}")
    with pytest.raises(RecoveryRejected, match="Incompatible chat template"):
        _encode({"prompt": "task", "response": " answer "}, tokenizer, "causal", 256)


def test_encode_only_dummy_flat_strict_stability():
    tokenizer = character_tokenizer(chat=False)
    class EncodeOnly:
        def __getattr__(self, name):
            return getattr(tokenizer, name)
    row = {"prompt": "task", "response": "answer"}
    assert _encode(row, EncodeOnly(), "causal", 32) == _encode(row, tokenizer, "causal", 32)


@pytest.mark.parametrize("family", ["causal", "seq2seq"])
def test_unselected_overlength_row_rejected_before_any_model_load(tmp_path, monkeypatch, family):
    teacher, student, train, dev = tiny_stores(tmp_path, family)
    payload = json.loads(train.read_text())
    payload["samples"].append({"id": "too-long", "prompt": "hello " * 100, "response": "answer"})
    train.write_text(json.dumps(payload))
    def never_load(*args, **kwargs):
        pytest.fail("Weights must not be loaded before complete all-row admission")
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", never_load)
    monkeypatch.setattr(AutoModelForSeq2SeqLM, "from_pretrained", never_load)
    result = recover(teacher, student, tmp_path / "long-reject", train, dev,
                     max_samples=1, max_length=12, steps=1, dtype="float32")
    assert result["status"] == "rejected" and result["actual_steps"] == 0
    assert "Complete token length" in result["error"]["message"]
    assert not (tmp_path / "long-reject").exists()


@pytest.mark.parametrize("family", ["causal", "seq2seq"])
def test_response_overlength_and_embedded_eos_never_rewritten(tmp_path, family):
    tokenizer = tokenizer_at(tmp_path)
    row = {"prompt": "task", "response": "answer " * 30}
    before = dict(row)
    with pytest.raises(RecoveryRejected, match="Complete token length"):
        _encode(row, tokenizer, family, 12)
    assert row == before
    with pytest.raises(RecoveryRejected, match="terminal EOS"):
        _encode({"prompt": "task", "response": "answer </s> world"}, tokenizer, family, 12)


@pytest.mark.parametrize("key", ["family", "family_id", "task_family"])
def test_family_leakage_before_subsampling(tmp_path, key):
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")
    for path in (train, dev):
        data = json.loads(path.read_text())
        data["samples"][0][key] = "same-family"
        path.write_text(json.dumps(data))
    result = recover(teacher, student, tmp_path / "families", train, dev, max_samples=1)
    assert result["status"] == "rejected" and result["actual_steps"] == 0
    assert "family leakage" in result["error"]["message"]


@pytest.mark.parametrize("text,match", [
    ('{"samples":[],"samples":[]}', "Duplicate JSON key"),
    ('{"samples":[{"id":"a","id":"b","prompt":"task","response":"answer"}]}', "Duplicate JSON key"),
    ('{"samples":[],"provenance":{"score":NaN}}', "Nonfinite JSON"),
    ('{"samples":[],"provenance":{"score":Infinity}}', "Nonfinite JSON"),
    ('{"samples":[],"provenance":{"score":1e999}}', "Nonfinite JSON"),
])
def test_duplicate_json_keys_and_nonfinite_rejected(tmp_path, text, match):
    path = tmp_path / "invalid.json"
    path.write_text(text)
    with pytest.raises(RecoveryRejected, match=match):
        _read_samples(path)


@pytest.mark.parametrize("identifier", ["", " ", " padded", "a\nb", "a\x00b", "a" * 257])
def test_invalid_sample_ids(tmp_path, identifier):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"samples": [{"id": identifier, "prompt": "task", "response": "answer"}]}))
    with pytest.raises(RecoveryRejected, match="Invalid sample id"):
        _read_samples(path)


def test_files_and_rows_bounded_before_json_object_parse(tmp_path, monkeypatch):
    path = tmp_path / "huge.json"
    with path.open("wb") as handle:
        handle.truncate(MAX_DATA_BYTES + 1)  # sparse, no multi-GB allocation
    def never_parse(*args, **kwargs):
        pytest.fail("json.loads must not run before byte/row bounds")
    monkeypatch.setattr("asea.specialist.recovery.json.loads", never_parse)
    with pytest.raises(RecoveryRejected, match="byte limit"):
        _bounded_json(path)
    path.write_text('{"samples":[' + ','.join('{}' for _ in range(MAX_DATA_ROWS + 1)) + ']}')
    with pytest.raises(RecoveryRejected, match="10000"):
        _bounded_json(path)


def test_provenance_and_escaped_json_delimiters_allowed(tmp_path):
    path = tmp_path / "valid.json"
    row = {"id": "a", "prompt": 'string with ] [ } { \\" delimiters', "response": "answer\n",
           "family": "family-a", "provenance": {"nested": ["a", "b"]}}
    path.write_text(json.dumps({"samples": [row], "schema": "parent-schema"}))
    assert _read_samples(path)[0] == [row]


def test_full_vocabulary_workspace_uses_actual_complete_lengths_not_padding(tmp_path):
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")
    result = recover(teacher, student, tmp_path / "full-budget", train, dev,
                     steps=1, rank=2, max_length=256, dtype="float32")
    assert result["status"] == "completed", result
    resources = result["resources"]
    assert resources["actual_max_complete_length"] == 5
    assert resources["actual_max_response_tokens"] == 3
    assert resources["logits_workspace_bytes"] == 5 * 12 * 4 + 3 * 12 * (4 + 24)
    assert result["merge_parity_probe"]["passed"]
    assert result["standalone_reload_probe"]["passed"]
    assert result["standalone_reload_probe"]["cache_restored"] is True
    assert result["validation_post"]["model_state"].startswith("trained_adapter_premerge")
    assert result["data"]["all_rows_length_checked_before_weight_load"]
    assert len(result["data"]["encoding_audit"]["training"]) == 2
    assert result["encoding"]["format"] == "flat_newline_response_only_v1"


def test_export_restores_original_false_cache_setting(tmp_path):
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")
    path = student / "config.json"
    config = json.loads(path.read_text())
    config["use_cache"] = False
    path.write_text(json.dumps(config))
    result = recover(teacher, student, tmp_path / "false-cache", train, dev, steps=1,
                     rank=2, max_length=12, dtype="float32", method="supervised_lora", kd_weight=0)
    assert result["status"] == "completed", result
    assert result["standalone_reload_probe"]["passed"]
    assert result["cache_restored"] is False
    assert json.loads((tmp_path / "false-cache" / "config.json").read_text())["use_cache"] is False


def test_native_chat_actual_backward_and_reload_with_positional_eos(tmp_path):
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")
    tokenizer = character_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    for path, intermediate in ((teacher, 32), (student, 24)):
        tokenizer.save_pretrained(path)
        config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=intermediate,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                             max_position_embeddings=256, bos_token_id=None,
                             eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        Qwen2ForCausalLM(config).save_pretrained(path, safe_serialization=True)
    result = recover(teacher, student, tmp_path / "chat-native", train, dev,
                     steps=1, rank=2, max_length=128, dtype="float32")
    assert result["status"] == "completed", result
    assert result["encoding"]["format"] == "native_chat_response_only_v1"
    assert result["validation_post"]["response_tokens"] == len("other answer") + 1
    assert result["merge_parity_probe"]["passed"] and result["standalone_reload_probe"]["passed"]
    audit = result["data"]["encoding_audit"]["validation"][0]
    assert audit["response_content_tokens"] == len("other answer")
    assert audit["terminal_eos_positions"] == 1 and audit["boundary_engine"] == "offset_mapping"


@pytest.mark.parametrize("family", ["causal", "seq2seq"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_cached_matches_resident_real_losses_gradients_and_merged_output(tmp_path, family, dtype):
    teacher, student, train, dev = tiny_stores(tmp_path, family)
    results = {}
    for mode in ("cached", "resident"):
        results[mode] = recover(teacher, student, tmp_path / mode, train, dev, steps=3,
                                learning_rate=0.01, rank=2, max_length=256, dtype=dtype,
                                teacher_mode=mode, memory_budget_bytes=4 * 1024**3)
        assert results[mode]["status"] == "completed", results[mode]
    cached, resident = results["cached"], results["resident"]
    assert cached["teacher_unloaded_before_student"]
    assert cached["teacher_forward_calls"] == 2 and cached["validation_teacher_forward_calls"] == 1
    assert resident["teacher_forward_calls"] == 3 and resident["validation_teacher_forward_calls"] == 2
    assert cached["validation_pre"] == resident["validation_pre"]
    assert cached["validation_post"] == resident["validation_post"]
    assert cached["training_history"] == resident["training_history"]
    assert cached["adapter_hash_after"] == resident["adapter_hash_after"]
    assert cached["output_weight_sha256"] == resident["output_weight_sha256"]
    bank = cached["teacher_bank"]
    assert bank["full_vocabulary"] and bank["response_positions_only"]
    assert bank["storage_dtype"] == "float32" and bank["removed_before_export"]
    assert bank["actual_tensor_bytes"] == (3 + 2 + 3) * 12 * 4
    assert len(bank["rows"]) == 3
    assert all(row["shape"][1] == 12 for row in bank["rows"])
    assert not Path(bank["path"]).exists()
    assert not list((tmp_path / "cached").glob("**/*bank*"))
    assert cached["standalone_tokenizer_verified"] and cached["standalone_inventory_and_native_keys_verified"]


def test_cached_teacher_is_collected_before_loading_student(tmp_path, monkeypatch):
    import weakref
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")
    original = AutoModelForCausalLM.from_pretrained
    refs = []
    def inspect(path, *args, **kwargs):
        if Path(path) == student:
            assert refs and refs[0]() is None, "resident teacher leaked into student phase"
        result = original(path, *args, **kwargs)
        if Path(path) == teacher:
            refs.append(weakref.ref(result[0]))
        return result
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", inspect)
    result = recover(teacher, student, tmp_path / "serial", train, dev,
                     steps=1, rank=2, max_length=12, dtype="float32")
    assert result["status"] == "completed", result


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_bank_lossless_full_logits_integrity_and_no_external_adoption(tmp_path, dtype):
    from asea.specialist.recovery import _TeacherBank
    sample = {"input_ids": [2, 3, 4, 1], "attention_mask": [1] * 4, "labels": [-100, -100, 4, 1]}
    logits = torch.randn(1, 4, 19).to(dtype)
    selected, _ = _response_logits(logits, torch.tensor([sample["labels"]]), "causal")
    bank = _TeacherBank(tmp_path / "bank", {"model_files_sha256": "test-owned"}, 19, 2 * 19 * 4, 20000)
    bank.write(sample, selected, "training", "owned-row")
    loaded = bank.read(sample, torch.device("cpu"))
    assert torch.equal(loaded, selected.float())
    student = torch.randn(2, 19, requires_grad=True)
    loss = _full_kl(student, loaded, chunk_size=5)
    resident = _full_kl(student, selected, chunk_size=5)
    assert torch.equal(loss, resident)
    assert torch.equal(torch.autograd.grad(loss, student)[0], torch.autograd.grad(resident, student)[0])
    with pytest.raises(FileExistsError):
        _TeacherBank(bank.directory, {}, 19, 1000, 20000)
    path = bank.directory / bank.receipt()["rows"][0]["file"]
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(RecoveryRejected, match="integrity"):
        bank.read(sample, torch.device("cpu"))


@pytest.mark.parametrize("mutation", ["teacher_peft", "student_peft", "auto_map", "tokenizer_auto_map", "version", "symlink", "traversal", "extra", "shape", "unsafe_dimension"])
def test_shared_admission_before_any_auto_loader(tmp_path, monkeypatch, mutation):
    from transformers import AutoConfig
    from safetensors.torch import load_file, save_file
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")
    if mutation.endswith("peft"):
        ((teacher if mutation == "teacher_peft" else student) / "adapter_config.json").write_text('{}')
    elif mutation == "symlink":
        alias = tmp_path / "teacher-link"
        alias.symlink_to(teacher, target_is_directory=True)
        teacher = alias
    elif mutation == "traversal":
        teacher = teacher / ".." / "teacher"
    elif mutation in ("extra", "shape"):
        path = student / "model.safetensors"
        state = load_file(path)
        state["unmatched.native.weight" if mutation == "extra" else "model.layers.0.mlp.down_proj.weight"] = torch.ones(1)
        save_file(state, path, metadata={"format": "pt"})
    else:
        path = teacher / ("tokenizer_config.json" if mutation == "tokenizer_auto_map" else "config.json")
        payload = json.loads(path.read_text())
        key, value = ("transformers_version", "999.0.0") if mutation == "version" else (("hidden_size", 2**30) if mutation == "unsafe_dimension" else ("auto_map", {}))
        payload[key] = value
        path.write_text(json.dumps(payload))
    def forbidden(*args, **kwargs):
        pytest.fail("unsafe artifact reached Auto loader")
    for cls in (AutoConfig, AutoTokenizer, AutoModelForCausalLM):
        monkeypatch.setattr(cls, "from_pretrained", forbidden)
    result = recover(teacher, student, tmp_path / "unsafe", train, dev, steps=1)
    assert result["status"] == "rejected" and result["actual_steps"] == 0
    assert not (tmp_path / "unsafe").exists()


def test_cached_failure_removes_bank_and_keeps_executed_history(tmp_path, monkeypatch):
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")
    original = torch.optim.AdamW.step
    calls = []
    def stop(self, *args, **kwargs):
        if calls:
            raise RuntimeError("second-step-failure")
        calls.append(1)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, "step", stop)
    result = recover(teacher, student, tmp_path / "partial", train, dev,
                     steps=2, rank=2, max_length=12, dtype="float32")
    assert result["status"] == "rejected" and result["actual_steps"] == 1
    assert len(result["training_history"]) == 1
    assert not Path(result["teacher_bank"]["path"]).exists()
    assert not list(tmp_path.glob(".partial.recovery-*"))


def test_bank_disk_and_operator_memory_preflight_are_bounded(tmp_path, monkeypatch):
    import shutil
    from asea.specialist.recovery import _preflight
    teacher, student, _, _ = tiny_stores(tmp_path, "causal")
    config = Qwen2Config.from_pretrained(student, local_files_only=True)
    monkeypatch.setattr("asea.specialist.recovery._available_ram", lambda: 32 * 1024**3)
    with pytest.raises(RecoveryRejected, match="memory headroom"):
        _preflight(teacher, student, tmp_path, config, 2, ["q_proj"], 4, 5, True, torch,
                   memory_budget_bytes=1024, total_response_tokens=8, bank_rows=3)
    monkeypatch.setattr("asea.specialist.recovery.shutil.disk_usage", lambda path: shutil._ntuple_diskusage(1, 0, 1))
    with pytest.raises(RecoveryRejected, match="disk headroom"):
        _preflight(teacher, student, tmp_path, config, 2, ["q_proj"], 4, 5, True, torch,
                   total_response_tokens=8, bank_rows=3)


@pytest.mark.parametrize("family", ["causal", "seq2seq"])
@pytest.mark.parametrize("teacher_mode", ["cached", "resident"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_factor_bundle_real_two_steps_exact_reload_without_sources(tmp_path, monkeypatch, family, teacher_mode, dtype):
    from peft.tuners.lora import LoraModel
    from asea.specialist.standalone import load_standalone, inspect_bundle, COMPONENT
    teacher, student, train, dev = tiny_stores(tmp_path, family)
    teacher_hash, student_hash = _store_hash(teacher), _store_hash(student)
    def forbidden_merge(*args, **kwargs):
        pytest.fail("factor export must never merge")
    monkeypatch.setattr(LoraModel, "merge_and_unload", forbidden_merge)
    def forbidden_save(*args, **kwargs):
        pytest.fail("factor export must not call opaque model/tokenizer serialization")
    monkeypatch.setattr(Qwen2ForCausalLM, "save_pretrained", forbidden_save)
    monkeypatch.setattr(T5ForConditionalGeneration, "save_pretrained", forbidden_save)
    monkeypatch.setattr(PreTrainedTokenizerFast, "save_pretrained", forbidden_save)
    output = tmp_path / "factor"
    result = recover(teacher, student, output, train, dev, steps=2, learning_rate=0.01,
                     rank=2, max_length=12, dtype=dtype, teacher_mode=teacher_mode,
                     export_mode="factor_preserving")
    assert result["status"] == "completed", result
    assert result["actual_steps"] == 2 and result["adapter_delta_l2"] > 0
    assert result["factor_preserving"] and not result["merged"] and not result["standalone_native"]
    assert result["standalone_teacher_independent"]
    assert result["trained_artifact_capture"]["before_merge"]
    assert result["resources"]["merge_workspace_bytes"] == 0
    assert result["resources"]["factor_export_disk_bytes_bound"] == result["trainable_parameters"] * 8
    assert result["bounded_export"]["tensor_staging_bytes"] == 0
    assert result["bounded_export"]["whole_model_clone"] is False
    releases = result["resources"]["release_checks"]
    assert releases[-1]["phase"] == "factor-export-reload" and releases[-1]["tracked_objects"] > 2
    assert all(r["all_collected"] and r["reclaimed_memory_credit_bytes"] == 0 for r in releases)
    probe = result["standalone_reload_probe"]
    assert probe["forward_bitwise_equal"] and probe["generation_bitwise_equal"]
    assert probe["max_abs_error"] == 0 and probe["atol"] == probe["rtol"] == 0
    assert teacher_hash == _store_hash(teacher) and student_hash == _store_hash(student)
    assert not (output / "config.json").exists()
    assert not (output / ".teacher-bank").exists()
    assert json.loads((output / "base/config.json").read_text())["silt_specialist_component"] == COMPONENT
    teacher.rename(tmp_path / "teacher-hidden")
    student.rename(tmp_path / "student-hidden")
    header = inspect_bundle(output)
    assert header["counts"]["total_parameters"] == result["total_parameters_with_adapter"]
    assert header["counts"]["factor_parameters"] == result["trainable_parameters"]
    assert header["counts"]["original_teacher_parameters"] == result["resources"]["teacher"]["parameters"]
    for cfg in (output / "base").glob("*.json"):
        assert str(teacher) not in cfg.read_text() and str(student) not in cfg.read_text()
    loaded = load_standalone(output, dtype=getattr(torch, dtype))
    assert loaded.base_path == output / "base"
    assert any("lora_" in name for name, _ in loaded.model.named_parameters())
    from asea.specialist.reconstruction import _artifact_preflight
    from asea.specialist.recovery import _parameter_hash
    _, _, headers = _artifact_preflight(output / "base")
    assert all(entry["dtype"] == ("F32" if dtype == "float32" or
               (family == "seq2seq" and ".wo." in key) else "BF16")
               for key, entry in headers.items())
    assert _parameter_hash((n, p) for n, p in loaded.model.named_parameters() if "lora_" not in n) == result["frozen_parameter_hash_after"]
    assert not any(p.is_meta for p in list(loaded.model.parameters()) + list(loaded.model.buffers()))
    with torch.no_grad():
        actual = loaded.model.generate(input_ids=torch.tensor(probe["generation_input_ids"]),
                  attention_mask=torch.ones_like(torch.tensor(probe["generation_input_ids"])),
                  do_sample=False, max_new_tokens=2)
    assert actual.tolist() == probe["generation_before_ids"]
    assert str(loaded.model.peft_config["default"].base_model_name_or_path) == str(output / "base")


@pytest.fixture
def factor_store(tmp_path):
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")
    output = tmp_path / "factor"
    result = recover(teacher, student, output, train, dev, steps=2, rank=2,
                     max_length=12, dtype="bfloat16", export_mode="factor_preserving")
    assert result["status"] == "completed", result
    return output


def _rehash_bundle_file(output, relative):
    from asea.specialist.recovery import _file_hash
    p = output / "specialist_bundle.json"
    value = json.loads(p.read_text())
    value["files"][relative] = {"sha256": _file_hash(output / relative), "bytes": (output / relative).stat().st_size}
    p.write_text(json.dumps(value))


@pytest.mark.parametrize("mutation", ["external_base", "rank", "target", "unknown_option", "shape", "missing_factor",
                                      "extra_factor", "corrupt", "symlink", "traversal", "model_class", "count", "version",
                                      "auto_map", "root_native", "runner"])
def test_factor_bundle_rejects_hostile_metadata_before_weights(factor_store, monkeypatch, mutation):
    from asea.specialist.standalone import load_standalone
    from safetensors.torch import load_file, save_file
    output = factor_store
    cfgpath = output / "adapter/adapter_config.json"
    if mutation in ("external_base", "rank", "target", "unknown_option"):
        value = json.loads(cfgpath.read_text())
        if mutation == "external_base": value["base_model_name_or_path"] = "/outside/teacher"
        elif mutation == "rank": value["r"] = 128
        elif mutation == "target": value["target_modules"] = ["lm_head"]
        else: value["use_dora"] = True
        cfgpath.write_text(json.dumps(value))
        _rehash_bundle_file(output, "adapter/adapter_config.json")
    elif mutation in ("shape", "missing_factor", "extra_factor", "corrupt"):
        path = output / "adapter/adapter_model.safetensors"
        if mutation == "corrupt":
            with path.open("ab") as f: f.write(b"bad")
        else:
            state = load_file(path)
            key = next(iter(state))
            if mutation == "shape": state[key] = torch.ones(1, 1)
            elif mutation == "missing_factor": del state[key]
            else: state["unknown.lora_A.weight"] = torch.ones(1, 1)
            save_file(state, path)
        _rehash_bundle_file(output, "adapter/adapter_model.safetensors")
    elif mutation == "symlink":
        cfgpath.rename(output.parent / "external.json")
        cfgpath.symlink_to(output.parent / "external.json")
    elif mutation == "auto_map":
        path = output / "base/tokenizer_config.json"
        value = json.loads(path.read_text()); value["auto_map"] = {}
        path.write_text(json.dumps(value)); _rehash_bundle_file(output, "base/tokenizer_config.json")
    elif mutation == "root_native":
        (output / "config.json").write_text('{}')
    elif mutation == "runner":
        (output / "run_specialist.py").write_text('raise RuntimeError("untrusted code")')
        _rehash_bundle_file(output, "run_specialist.py")
    else:
        path = output / "specialist_bundle.json"
        value = json.loads(path.read_text())
        if mutation == "traversal": value["paths"]["base"] = "../teacher"
        elif mutation == "model_class": value["model_class"] = "UntrustedModel"
        elif mutation == "count": value["counts"]["total_parameters"] += 1
        else: value["runtime_versions"]["peft"] = "999.0.0"
        path.write_text(json.dumps(value))
    def forbidden(*args, **kwargs):
        pytest.fail("unsafe bundle reached weight loader")
    monkeypatch.setattr(Qwen2ForCausalLM, "from_pretrained", forbidden)
    monkeypatch.setattr("asea.specialist.recovery._load_recovery_model", forbidden)
    with pytest.raises(ValueError):
        load_standalone(output)


def test_inspect_bundle_no_weight_load_and_dtype_no_silent_recast(factor_store, monkeypatch):
    from asea.specialist.standalone import inspect_bundle, load_standalone
    def forbidden(*args, **kwargs):
        pytest.fail("inspection must not load weights")
    monkeypatch.setattr(Qwen2ForCausalLM, "from_pretrained", forbidden)
    monkeypatch.setattr("safetensors.torch.load_file", forbidden)
    monkeypatch.setattr("asea.specialist.recovery._load_recovery_model", forbidden)
    assert inspect_bundle(factor_store)["export_mode"] == "factor_preserving"
    with pytest.raises(RecoveryRejected, match="local-only"):
        load_standalone(factor_store, local_only=False)
    with pytest.raises(RecoveryRejected, match="dtype must match"):
        load_standalone(factor_store, dtype="float32")


def test_explicit_export_mode_and_native_failure_closed(tmp_path, monkeypatch):
    from peft.tuners.lora import LoraModel
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")
    rejected = recover(teacher, student, tmp_path / "unknown", train, dev, export_mode="auto")
    assert rejected["status"] == "rejected" and rejected["actual_steps"] == 0
    original = LoraModel.merge_and_unload
    def damaged_merge(self, *args, **kwargs):
        merged = original(self, *args, **kwargs)
        with torch.no_grad():
            merged.lm_head.weight.add_(1000)
        return merged
    monkeypatch.setattr(LoraModel, "merge_and_unload", damaged_merge)
    receipt = tmp_path / "old-v3-receipt.json"
    receipt.write_text('{"status":"rejected","actual_steps":64}')
    result = recover(teacher, student, tmp_path / "bad-native", train, dev, steps=2, rank=2,
                     max_length=12, dtype="float32")
    assert result["export_mode"] == "native_merged" and result["status"] == "rejected"
    assert not result["merge_parity_probe"]["passed"]
    assert not (tmp_path / "bad-native").exists()
    assert receipt.read_text() == '{"status":"rejected","actual_steps":64}'
    assert not list(tmp_path.glob(".bad-native.recovery-*"))


def test_factor_separate_offline_process_and_copied_public_runner(factor_store):
    import os
    import subprocess
    import sys
    output = factor_store
    (output.parent / "teacher").rename(output.parent / "teacher-unavailable")
    (output.parent / "student").rename(output.parent / "student-unavailable")
    report = json.loads((output / "recovery_report.json").read_text())
    probe = report["standalone_reload_probe"]
    script = '''import json,sys,torch
from asea.specialist.standalone import load_standalone
torch.set_num_threads(1)
b = load_standalone(sys.argv[1])
x = torch.tensor(json.loads(sys.argv[2]))
with torch.no_grad():
    ids = b.model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False, max_new_tokens=2)
print(json.dumps(ids.tolist()))
'''
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", script, str(output), json.dumps(probe["generation_input_ids"])],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == probe["generation_before_ids"]
    runner = subprocess.run([sys.executable, str(output / "run_specialist.py"), "--prompt", "hello dev",
                             "--max-new-tokens", "2"], env=env, capture_output=True, text=True, timeout=60)
    assert runner.returncode == 0, runner.stderr


def test_factor_reload_changed_factors_rejects_and_cleans_stage(tmp_path, monkeypatch):
    import asea.specialist.standalone as standalone
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")
    original = standalone.load_standalone
    def corrupt_reload(*args, **kwargs):
        bundle = original(*args, **kwargs)
        with torch.no_grad():
            next(p for n, p in bundle.model.named_parameters() if "lora_" in n).add_(1)
        return bundle
    monkeypatch.setattr(standalone, "load_standalone", corrupt_reload)
    output = tmp_path / "failed-factor"
    result = recover(teacher, student, output, train, dev, steps=2, rank=2,
                     max_length=12, dtype="bfloat16", export_mode="factor_preserving")
    assert result["status"] == "rejected" and result["actual_steps"] == 2
    assert "Reload changed trained adapter" in result["error"]["message"]
    assert result["trained_artifact_capture"]["before_merge"]
    assert not output.exists() and not list(tmp_path.glob(".failed-factor.recovery-*"))


def _mixed_recovery_store(directory, model_type, sharded=False):
    """Rewrite only tiny fixtures: native source BF16 plus original F32 routers."""
    from safetensors.torch import load_file, save_file
    path = directory / "model.safetensors"
    state = load_file(path)
    state = {key: (value.float() if key.endswith("router.classifier.weight") else value.bfloat16())
             for key, value in state.items()}
    if not sharded:
        save_file(state, path, metadata={"format": "pt"})
        return
    path.unlink()
    groups = [dict(list(state.items())[::2]), dict(list(state.items())[1::2])]
    weight_map = {}
    for index, group in enumerate(groups):
        name = "model-%05d-of-00002.safetensors" % (index + 1)
        save_file(group, directory / name, metadata={"format": "pt"})
        weight_map.update({key: name for key in group})
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


@pytest.mark.parametrize("model_type", ["qwen2", "switch_transformers", "t5"])
@pytest.mark.parametrize("requested", ["bfloat16", "float32"])
@pytest.mark.parametrize("sharded", [False, True])
def test_explicit_recovery_typed_plan_values_aliases_and_mmap(tmp_path, monkeypatch, model_type, requested, sharded):
    import safetensors
    from asea.specialist.reconstruction import _artifact_preflight
    from asea.specialist.recovery import _recovery_inputs, _header_stats, _load_recovery_model
    teacher, student, _, _ = tiny_stores(tmp_path, "causal" if model_type == "qwen2" else "seq2seq")
    directory = student if model_type == "t5" else teacher
    _mixed_recovery_store(directory, model_type, sharded)
    inventory, raw, headers = _artifact_preflight(directory)
    meta, plan = _recovery_inputs(raw, headers, requested, torch)
    stats = _header_stats(directory, 2, ["q", "v", "wo", "q_proj"], requested, (inventory, headers, plan))
    assert plan["native_keep_in_fp32_modules"] == (["wo"] if model_type == "t5" else [])
    expected_casts, expected_shared = [], 0
    for key, entry in headers.items():
        target = ("float32" if (model_type == "t5" and ".wo." in key) or
                  key.endswith("router.classifier.weight") else requested)
        assert plan["target_dtypes"][key] == target
        if entry["dtype"] != {"float32": "F32", "bfloat16": "BF16"}[target]:
            expected_casts.append(key)
        else:
            expected_shared += entry["numel"] * (4 if target == "float32" else 2)
    assert {item["key"] for item in stats["cast_tensor_plan"]} == set(expected_casts)
    assert stats["shared_mmap_payload_bytes"] == expected_shared
    assert stats["loaded_bytes"] == expected_shared + stats["cast_target_allocation_bytes"]
    assert stats["cast_source_pages_bytes"] == sum(item["source_bytes"] for item in stats["cast_tensor_plan"])
    assert stats["load_peak_bytes"] == (stats["loaded_bytes"] + stats["cast_source_pages_bytes"] +
                                         stats["largest_cast_tensor_scratch_bytes"] + stats["mapping_overhead_bytes"])
    if model_type in ("switch_transformers", "qwen2") and requested == "bfloat16":
        assert not expected_casts and stats["largest_cast_tensor_scratch_bytes"] == 0
    # Observe storage within the SAME safe_open handle; independent mmaps can have
    # different virtual addresses even when they share physical file pages.
    real_open, originals = safetensors.safe_open, {}
    class AuditedOpen:
        def __init__(self, *args, **kwargs):
            self.inner = real_open(*args, **kwargs)
        def __enter__(self):
            self.inner.__enter__()
            return self
        def __exit__(self, *args):
            return self.inner.__exit__(*args)
        def get_tensor(self, key):
            value = self.inner.get_tensor(key)
            originals[key] = value
            return value
    monkeypatch.setattr(safetensors, "safe_open", AuditedOpen)
    model = _load_recovery_model(meta, directory, headers, plan, torch)
    for alias, stored in plan["aliases"].items():
        value = model.get_parameter(alias)
        original = originals[stored]
        assert torch.equal(value, original.to(dtype=getattr(torch, plan["target_dtypes"][stored])))
        if stored not in expected_casts:
            assert value.data_ptr() == original.data_ptr()
        else:
            assert value.data_ptr() != original.data_ptr()
        assert value is model.get_parameter(stored)
    assert not any(p.is_meta for p in list(model.parameters()) + list(model.buffers()))
    assert sum(p.numel() * p.element_size() for p in model.parameters()) == stats["loaded_bytes"]


@pytest.mark.parametrize("teacher_kind", ["qwen2", "switch_transformers", "t5"])
@pytest.mark.parametrize("mode", ["cached", "resident"])
def test_recovery_phase_policy_uses_both_exact_plans(tmp_path, monkeypatch, teacher_kind, mode):
    from asea.specialist.recovery import _preflight
    teacher, student, _, _ = tiny_stores(tmp_path, "causal" if teacher_kind == "qwen2" else "seq2seq")
    if teacher_kind == "t5":
        model = T5ForConditionalGeneration.from_pretrained(student, local_files_only=True)
        model.save_pretrained(teacher, safe_serialization=True)
    for root in (teacher, student):
        _mixed_recovery_store(root, teacher_kind)
    config = transformers.AutoConfig.from_pretrained(student, local_files_only=True)
    tc = transformers.AutoConfig.from_pretrained(teacher, local_files_only=True)
    monkeypatch.setattr("asea.specialist.recovery._available_ram", lambda: 16 * 1024**3)
    result = _preflight(teacher, student, tmp_path, config, 2,
                        ["q_proj", "v_proj", "down_proj"] if teacher_kind == "qwen2" else ["q", "v", "wo"],
                        2, 5, True, torch, teacher_mode=mode, teacher_config=tc,
                        total_response_tokens=8, bank_rows=3, export_mode="factor_preserving")
    assert result["teacher"]["load_plan"]["native_keep_in_fp32_modules"] == (["wo"] if teacher_kind == "t5" else [])
    assert result["student"]["load_plan"]["native_keep_in_fp32_modules"] == ([] if teacher_kind == "qwen2" else ["wo"])
    assert result["student_phase_estimated_bytes"] == result["student_phase_incremental_bytes"] + (
        result["teacher_phase_estimated_bytes"] if mode == "resident" else 0)
    assert result["weight_bytes"] == result["student"]["loaded_bytes"] + (
        result["teacher"]["loaded_bytes"] if mode == "resident" else 0)
    assert result["estimated_peak_bytes"] == max(result["teacher_phase_estimated_bytes"],
        result["student_phase_estimated_bytes"], result["export_write_peak_estimated_bytes"],
        result["export_reload_estimated_bytes"])


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("mode", ["cached", "resident"])
def test_qwen_one_real_step_exact_old_loader_values_outputs(tmp_path, monkeypatch, dtype, mode):
    from asea.specialist import recovery as r
    from asea.specialist.standalone import load_standalone
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")
    _mixed_recovery_store(teacher, "qwen2")
    _mixed_recovery_store(student, "qwen2")
    kwargs = dict(steps=1, rank=2, max_length=12, dtype=dtype, teacher_mode=mode,
                  export_mode="factor_preserving", learning_rate=0.01)
    actual = recover(teacher, student, tmp_path / "explicit", train, dev, **kwargs)
    def prior(meta, path, headers, plan, torch):
        return AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
            trust_remote_code=False, use_safetensors=True, torch_dtype=getattr(torch, dtype),
            low_cpu_mem_usage=True, device_map={"": "cpu"}, attn_implementation="eager").to("cpu")
    monkeypatch.setattr(r, "_load_recovery_model", prior)
    old = recover(teacher, student, tmp_path / "prior", train, dev, **kwargs)
    assert actual["status"] == old["status"] == "completed", (actual.get("error"), old.get("error"))
    for key in ("training_history", "validation_pre", "validation_post", "adapter_hash_before", "adapter_hash_after",
                "frozen_parameter_hash_before", "frozen_parameter_hash_after"):
        assert actual[key] == old[key], key
    models = [load_standalone(tmp_path / name).model for name in ("explicit", "prior")]
    batch = {"input_ids": torch.tensor([[4, 5, 1]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    with torch.no_grad():
        assert torch.equal(models[0](**batch, use_cache=False).logits, models[1](**batch, use_cache=False).logits)
        assert torch.equal(models[0].generate(**batch, do_sample=False, max_new_tokens=2),
                           models[1].generate(**batch, do_sample=False, max_new_tokens=2))


def test_phase_guard_reobserves_pressure_and_never_adds_reclamation_credit(monkeypatch):
    from asea.specialist.recovery import _phase_guard
    resources = {"runtime_phase_checks": []}
    values = iter([2 * 1024**3, 300 * 1024**2])
    monkeypatch.setattr("asea.specialist.recovery._available_ram", lambda: next(values))
    first = _phase_guard(resources, "teacher-load", 128 * 1024**2, 256 * 1024**2)
    assert first["limit_bytes"] == 256 * 1024**2
    with pytest.raises(RecoveryRejected, match="student-phase memory headroom"):
        _phase_guard(resources, "student-phase", 128 * 1024**2, 256 * 1024**2)
    assert len(resources["runtime_phase_checks"]) == 2
    assert resources["runtime_phase_checks"][-1]["limit_bytes"] == 44 * 1024**2


@pytest.mark.parametrize("mutation", ["manifest", "base_header"])
def test_factor_base_dtype_policy_rejects_before_tensor_read(factor_store, monkeypatch, mutation):
    from asea.specialist.standalone import inspect_bundle, load_standalone
    from safetensors.torch import load_file, save_file
    if mutation == "manifest":
        path = factor_store / "specialist_bundle.json"
        manifest = json.loads(path.read_text())
        manifest["dtype"] = "float32"
        path.write_text(json.dumps(manifest))
    else:
        path = factor_store / "base/model.safetensors"
        state = load_file(path)
        key = next(iter(state))
        state[key] = state[key].float()
        save_file(state, path, metadata={"format": "pt"})
        _rehash_bundle_file(factor_store, "base/model.safetensors")
    def forbidden(*args, **kwargs):
        pytest.fail("dtype-inconsistent bundle reached a tensor read")
    monkeypatch.setattr("safetensors.safe_open", forbidden)
    monkeypatch.setattr("safetensors.torch.load_file", forbidden)
    monkeypatch.setattr("asea.specialist.recovery._load_recovery_model", forbidden)
    for operation in (inspect_bundle, load_standalone):
        with pytest.raises(RecoveryRejected, match="base header dtype"):
            operation(factor_store)


@pytest.mark.parametrize("sharded", [False, True])
def test_mixed_switch_recovery_has_router_only_f32_no_full_shard_doublecount(tmp_path, monkeypatch, sharded):
    from asea.specialist.recovery import _header_stats
    from asea.specialist.reconstruction import _artifact_preflight
    teacher, _, _, _ = tiny_stores(tmp_path, "seq2seq")
    _mixed_recovery_store(teacher, "switch_transformers", sharded)
    inventory, _, headers = _artifact_preflight(teacher)
    def forbidden(*args, **kwargs):
        pytest.fail("header memory admission must not read tensors")
    monkeypatch.setattr("safetensors.safe_open", forbidden)
    stats = _header_stats(teacher, 2, ["q", "v", "wo"], "bfloat16")
    f32 = {k for k, dtype in stats["load_plan"]["target_dtypes"].items() if dtype == "float32"}
    assert f32 == {k for k in headers if k.endswith("router.classifier.weight")}
    assert f32 and not any(".wo." in k for k in f32)
    payload = sum(h["numel"] * (4 if h["dtype"] == "F32" else 2) for h in headers.values())
    files = [v["size"] for k, v in inventory.items() if k.endswith(".safetensors")]
    assert stats["loaded_bytes"] == stats["shared_mmap_payload_bytes"] == payload
    assert stats["cast_source_pages_bytes"] == stats["cast_target_allocation_bytes"] == 0
    assert stats["largest_cast_tensor_scratch_bytes"] == 0
    assert stats["load_peak_bytes"] == sum(files) + len(files) * 8192


@pytest.mark.parametrize("teacher_mode", ["cached", "resident"])
def test_t5_factor_reload_preserves_nondefault_dropout_and_generation_config(tmp_path, teacher_mode):
    from asea.specialist.standalone import load_standalone
    teacher, student, train, dev = tiny_stores(tmp_path, "seq2seq")
    cfg = student / "config.json"
    value = json.loads(cfg.read_text())
    value["dropout_rate"] = 0.17
    cfg.write_text(json.dumps(value))
    generation = student / "generation_config.json"
    value = json.loads(generation.read_text())
    value["repetition_penalty"] = 1.07
    generation.write_text(json.dumps(value))
    result = recover(teacher, student, tmp_path / "factor", train, dev, steps=2,
                     learning_rate=0.01, rank=2, max_length=12, dtype="bfloat16",
                     teacher_mode=teacher_mode, export_mode="factor_preserving")
    assert result["status"] == "completed", result.get("error")
    assert result["standalone_reload_probe"]["forward_bitwise_equal"]
    assert result["standalone_reload_probe"]["generation_bitwise_equal"]
    bundle = load_standalone(tmp_path / "factor")
    assert bundle.model.config.dropout_rate == 0.17
    assert bundle.model.generation_config.repetition_penalty == 1.07
    assert any(isinstance(m, torch.nn.Dropout) and m.p == 0.17 for m in bundle.model.modules())


def _mixed_encoder_f32_decoder_bf16_switch(directory, router_bias=False, router_dtype="float32"):
    """Tiny fixture matching the real mixed-router header pattern; never a real store."""
    from safetensors.torch import load_file, save_file
    config = SwitchTransformersConfig(vocab_size=12, d_model=16, d_kv=8, d_ff=32,
        num_layers=4, num_decoder_layers=4, num_heads=2, dropout_rate=0,
        decoder_start_token_id=0, eos_token_id=1, pad_token_id=0, num_experts=2,
        num_sparse_encoder_layers=2, num_sparse_decoder_layers=2, expert_capacity=8,
        router_bias=router_bias, router_dtype=router_dtype, router_jitter_noise=0)
    model = SwitchTransformersForConditionalGeneration(config)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".router.classifier." in name:
                p.view(-1)[0] = 0.123456789  # visibly not BF16-representable
    model.save_pretrained(directory, safe_serialization=True)
    path = directory / "model.safetensors"
    # Detach tiny expected values from the old mmap before rewriting the fixture.
    state = {k: (v.float().clone() if k.startswith("encoder.") and ".router.classifier." in k
                 and router_dtype == "float32" else v.bfloat16())
             for k, v in load_file(path).items()}
    save_file(state, path, metadata={"format": "pt"})
    return state


@pytest.mark.parametrize("requested", ["bfloat16", "float32"])
@pytest.mark.parametrize("router_bias", [False, True])
def test_all_switch_routers_canonical_before_first_native_forward(tmp_path, monkeypatch, requested, router_bias):
    from asea.specialist.reconstruction import _artifact_preflight
    from asea.specialist.recovery import _recovery_inputs, _load_recovery_model, _header_stats, _parameter_hash
    from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersTop1Router
    teacher, _, _, _ = tiny_stores(tmp_path, "seq2seq")
    originals = _mixed_encoder_f32_decoder_bf16_switch(teacher, router_bias)
    files_before = _store_hash(teacher)
    inventory, raw, headers = _artifact_preflight(teacher)
    meta, plan = _recovery_inputs(raw, headers, requested, torch)
    router_keys = {k for k in headers if ".router.classifier." in k}
    assert len(router_keys) == (8 if router_bias else 4)
    assert set(plan["native_router_tensors"]) == router_keys
    assert plan["native_router_dtype"] == "float32"
    stats = _header_stats(teacher, 2, ["q", "v", "wo"], requested, (inventory, headers, plan))
    router_casts = [item for item in stats["cast_tensor_plan"] if item["key"] in router_keys]
    assert {item["key"] for item in router_casts} == {k for k in router_keys if k.startswith("decoder.")}
    for item in router_casts:
        assert item["source_dtype"] == "BF16" and item["loaded_dtype"] == "float32"
        assert item["target_bytes"] == 2 * item["source_bytes"]
    if requested == "bfloat16":
        assert stats["cast_source_pages_bytes"] == sum(item["source_bytes"] for item in router_casts)
        assert stats["cast_target_allocation_bytes"] == sum(item["target_bytes"] for item in router_casts)
        assert stats["largest_cast_tensor_scratch_bytes"] == max(item["source_bytes"] + item["target_bytes"] for item in router_casts)
        assert stats["cast_target_allocation_bytes"] < 4096
    model = _load_recovery_model(meta, teacher, headers, plan, torch)
    assert sum(p.numel() * p.element_size() for p in model.parameters()) == stats["loaded_bytes"]
    for key in router_keys:
        p = model.get_parameter(key)
        assert p.dtype == torch.float32
        assert torch.equal(p.detach().view(torch.uint8), originals[key].float().view(torch.uint8))
        if key.startswith("encoder."):
            assert not torch.equal(p, originals[key].bfloat16().float())
    before = {name: (p.dtype, p.data_ptr(), p.detach().clone()) for name, p in model.named_parameters()}
    digest = _parameter_hash(model.named_parameters())
    native_cast, calls = SwitchTransformersTop1Router._cast_classifier, []
    def observed_cast(router):
        pointers = [p.data_ptr() for p in router.classifier.parameters()]
        native_cast(router)  # execute the actual HF implementation, not a substitute
        assert pointers == [p.data_ptr() for p in router.classifier.parameters()]
        calls.append(router)
    monkeypatch.setattr(SwitchTransformersTop1Router, "_cast_classifier", observed_cast)
    with torch.no_grad():
        for _ in range(2):
            output = model(input_ids=torch.tensor([[4, 5, 1]]), decoder_input_ids=torch.tensor([[0, 7, 1]]), use_cache=False)
            assert torch.isfinite(output.logits).all()
            assert _parameter_hash(model.named_parameters()) == digest
    assert len(calls) == 8 and len({id(m) for m in calls}) == 4
    for name, p in model.named_parameters():
        dtype, pointer, value = before[name]
        assert p.dtype == dtype and p.data_ptr() == pointer and torch.equal(p, value)
        assert not p.requires_grad and p.grad is None
    assert _store_hash(teacher) == files_before


@pytest.mark.parametrize("teacher_mode", ["cached", "resident"])
def test_real_mixed_router_switch_t5_recovery_completes_with_frozen_teacher(tmp_path, teacher_mode):
    teacher, student, train, dev = tiny_stores(tmp_path, "seq2seq")
    _mixed_encoder_f32_decoder_bf16_switch(teacher)
    before = _store_hash(teacher), _store_hash(student)
    result = recover(teacher, student, tmp_path / "factor", train, dev, steps=1,
                     learning_rate=0.01, rank=2, max_length=12, dtype="bfloat16",
                     teacher_mode=teacher_mode, export_mode="factor_preserving")
    assert result["status"] == "completed", result.get("error")
    assert result["actual_steps"] == 1 and result["teacher_frozen_no_grad"]
    assert result["frozen_parameter_hash_before"] == result["frozen_parameter_hash_after"]
    assert result["standalone_reload_probe"]["forward_bitwise_equal"]
    assert result["standalone_reload_probe"]["generation_bitwise_equal"]
    assert before == (_store_hash(teacher), _store_hash(student))


@pytest.mark.parametrize("router_dtype", [None, [], "float16", "float64", "int64", "__dict__"])
def test_unexpected_native_router_dtype_blocked_before_tensor_read(tmp_path, monkeypatch, router_dtype):
    from asea.specialist.reconstruction import _artifact_preflight, ReconstructionBlocked
    teacher, _, _, _ = tiny_stores(tmp_path, "seq2seq")
    _, raw, _ = _artifact_preflight(teacher)
    raw["router_dtype"] = router_dtype
    (teacher / "config.json").write_text(json.dumps(raw))
    monkeypatch.setattr("safetensors.safe_open", lambda *a, **k: pytest.fail("unexpected tensor read"))
    with pytest.raises(ReconstructionBlocked, match="unsupported native Switch router_dtype"):
        _artifact_preflight(teacher)


def test_known_bf16_router_policy_and_lossy_f32_source_conflict(tmp_path):
    from asea.specialist.reconstruction import _artifact_preflight, ReconstructionBlocked
    from asea.specialist.recovery import _recovery_inputs, _load_recovery_model, _parameter_hash
    teacher, _, _, _ = tiny_stores(tmp_path, "seq2seq")
    _mixed_encoder_f32_decoder_bf16_switch(teacher, router_bias=True, router_dtype="bfloat16")
    _, raw, headers = _artifact_preflight(teacher)
    meta, plan = _recovery_inputs(raw, headers, "float32", torch)
    assert plan["native_router_dtype"] == "bfloat16"
    assert all(plan["target_dtypes"][k] == "bfloat16" for k in plan["native_router_tensors"])
    model = _load_recovery_model(meta, teacher, headers, plan, torch)
    digest = _parameter_hash(model.named_parameters())
    with torch.no_grad():
        model(input_ids=torch.tensor([[4, 1]]), decoder_input_ids=torch.tensor([[0, 7]]), use_cache=False)
    assert _parameter_hash(model.named_parameters()) == digest
    key = next(iter(plan["native_router_tensors"]))
    headers[key]["dtype"] = "F32"
    with pytest.raises(ReconstructionBlocked, match="would round original F32"):
        _recovery_inputs(raw, headers, "float32", torch)
    raw["router_dtype"] = "float32"
    headers[key]["dtype"] = "F16"
    with pytest.raises(ReconstructionBlocked, match="unsupported native Switch router source dtype"):
        _recovery_inputs(raw, headers, "bfloat16", torch)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_stream_writer_zero_copy_standard_offsets_and_dtype(tmp_path, monkeypatch, dtype):
    import hashlib
    import struct
    import asea.specialist.standalone as s
    from safetensors import safe_open
    values = {"tensor.weight": torch.arange(111).reshape(3, 37).to(dtype),
              "bias": torch.tensor([1., -0., 3.], dtype=dtype)}
    real_bytes, observations = s._tensor_bytes, []
    def observed(value):
        view = real_bytes(value)
        assert isinstance(view, memoryview)
        assert view.obj.__array_interface__["data"][0] == value.data_ptr()
        observations.append(view.nbytes)
        return view
    def forbidden(*args, **kwargs):
        pytest.fail("tensor staging/conversion/clone in bounded writer")
    monkeypatch.setattr(s, "_tensor_bytes", observed)
    monkeypatch.setattr(s, "EXPORT_CHUNK_BYTES", 37)
    path = tmp_path / "stream.safetensors"
    with monkeypatch.context() as patch:
        for name in ("clone", "contiguous", "cpu", "float", "to"):
            patch.setattr(torch.Tensor, name, forbidden)
        digests = s._stream_safetensors(path, values)
    assert observations
    raw = path.read_bytes()  # tiny test payload only
    length = struct.unpack("<Q", raw[:8])[0]
    assert length % 8 == 0
    header = json.loads(raw[8:8 + length])
    cursor = 0
    with safe_open(path, framework="pt", device="cpu") as reader:
        assert reader.metadata() == {"format": "pt"}
        for key in sorted(values):
            entry = header[key]
            assert entry["data_offsets"][0] == cursor
            cursor = entry["data_offsets"][1]
            value = reader.get_tensor(key)
            assert value.dtype == dtype and torch.equal(value.view(torch.uint8), values[key].view(torch.uint8))
            assert digests[key] == hashlib.sha256(real_bytes(value)).hexdigest()
    assert cursor == len(raw) - 8 - length


def test_stream_dirty_page_checkpoint_rechecks_even_when_cache_advice_fails(tmp_path, monkeypatch):
    from asea.specialist.standalone import _stream_safetensors
    events = []
    def no_advice(path, sync=False):
        events.append("advice-failed")
        return False
    def guard():
        events.append("guard")
        if events.count("guard") > 1:
            raise RecoveryRejected("new observed pressure")
    monkeypatch.setattr("asea.specialist.reconstruction._release_file_cache", no_advice)
    path = tmp_path / "partial.safetensors"
    # Synthetic 9 MiB tensor, not a checkpoint/model; crosses one 8 MiB window.
    value = torch.zeros(9 * 1024**2 // 4, dtype=torch.float32)
    with pytest.raises(RecoveryRejected, match="observed pressure"):
        _stream_safetensors(path, {"synthetic": value}, guard)
    assert events == ["guard", "advice-failed", "guard"]
    assert 8 * 1024**2 <= path.stat().st_size < value.numel() * 4


def test_stream_writer_rejects_noncontiguous_without_staging(tmp_path):
    from asea.specialist.standalone import _stream_safetensors
    path = tmp_path / "bad.safetensors"
    with pytest.raises(RecoveryRejected, match="contiguous"):
        _stream_safetensors(path, {"bad": torch.ones(3, 4).t()})
    assert not path.exists()


@pytest.mark.parametrize("family", ["causal", "seq2seq"])
def test_streamed_shards_tied_aliases_native_casts_and_asset_allowlist(tmp_path, monkeypatch, family):
    import asea.specialist.standalone as s
    from asea.specialist.reconstruction import _artifact_preflight
    from safetensors import safe_open
    teacher, student, train, dev = tiny_stores(tmp_path, family)
    _mixed_recovery_store(student, "qwen2" if family == "causal" else "t5", sharded=True)
    (student / "reconstruction_manifest.json").write_text('{"audit_only":"never deploy"}')
    (student / "unexpected.txt").write_text("unused original asset")
    cfg = student / "config.json"
    config = json.loads(cfg.read_text())
    config["use_cache"] = False
    cfg.write_text(json.dumps(config))
    before = _store_hash(student)
    monkeypatch.setattr(s, "EXPORT_SHARD_BYTES", 256)  # tiny fixture, many oversized single tensors
    output = tmp_path / "streamed"
    result = recover(teacher, student, output, train, dev, steps=1, rank=2, max_length=12,
                     dtype="bfloat16", export_mode="factor_preserving")
    assert result["status"] == "completed", result.get("error")
    assert result["standalone_reload_probe"]["forward_bitwise_equal"]
    assert result["standalone_reload_probe"]["generation_bitwise_equal"]
    assert result["cache_restored"] is False
    assert _store_hash(student) == before
    assert not (output / "base/reconstruction_manifest.json").exists()
    assert not (output / "base/unexpected.txt").exists()
    assert (output / "base/model.safetensors.index.json").is_file()
    _, _, original = _artifact_preflight(student)
    _, _, written = _artifact_preflight(output / "base")
    assert set(written) == set(original)  # exactly one source-admitted tied alias
    for name in s.TOKENIZER_ASSETS.intersection(before) - {"tokenizer_config.json"}:
        assert (output / "base" / name).read_bytes() == (student / name).read_bytes()
        assert (output / "base" / name).stat().st_ino != (student / name).stat().st_ino
    for key, entry in original.items():
        with safe_open(student / entry["file"], framework="pt") as source:
            with safe_open(output / "base" / written[key]["file"], framework="pt") as deployed:
                value = deployed.get_tensor(key)
                expected = source.get_tensor(key).to(value.dtype)
                assert torch.equal(value.view(torch.uint8), expected.view(torch.uint8))
                if family == "seq2seq" and ".wo." in key:
                    assert entry["dtype"] == "BF16" and written[key]["dtype"] == "F32"
    assert any(p.stat().st_size > s.EXPORT_SHARD_BYTES for p in (output / "base").glob("*.safetensors"))


@pytest.mark.parametrize("family", ["causal", "seq2seq"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_stream_factor_cached_resident_bitwise_training_metrics(tmp_path, family, dtype):
    teacher, student, train, dev = tiny_stores(tmp_path, family)
    results = [recover(teacher, student, tmp_path / mode, train, dev, steps=2, rank=2,
        max_length=12, dtype=dtype, teacher_mode=mode, export_mode="factor_preserving")
        for mode in ("cached", "resident")]
    assert all(r["status"] == "completed" for r in results), [r.get("error") for r in results]
    for key in ("training_history", "validation_pre", "validation_post", "adapter_hash_after",
                "frozen_parameter_hash_after", "output_weight_sha256"):
        assert results[0][key] == results[1][key], key
    assert all(r["standalone_reload_probe"]["forward_bitwise_equal"] for r in results)


def test_factor_reload_blocks_leaked_training_parameter_before_loader(tmp_path, monkeypatch):
    import asea.specialist.standalone as s
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")
    original, retained = s._write_bundle, []
    def leaking(root, model, *args):
        retained.append(next(p for n, p in model.named_parameters() if "lora_" not in n))
        return original(root, model, *args)
    def forbidden(*args, **kwargs):
        pytest.fail("one-model reload admitted with live training reference")
    monkeypatch.setattr(s, "_write_bundle", leaking)
    monkeypatch.setattr(s, "load_standalone", forbidden)
    result = recover(teacher, student, tmp_path / "leaked", train, dev, steps=1, rank=2,
                     max_length=12, dtype="bfloat16", export_mode="factor_preserving")
    assert result["status"] == "rejected" and result["actual_steps"] == 1
    assert "Training references retained" in result["error"]["message"]
    check = result["resources"]["release_checks"][-1]
    assert not check["all_collected"] and check["surviving_objects"]
    assert not (tmp_path / "leaked").exists() and not list(tmp_path.glob(".leaked.recovery-*"))


def test_factor_reload_reobserves_low_available_after_verified_release(tmp_path, monkeypatch):
    import asea.specialist.recovery as r
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")
    original, pressure = r._phase_guard, []
    monkeypatch.setattr(r, "_available_ram", lambda: 300 * 1024**2 if pressure else 16 * 1024**3)
    def pressure_after_release(resources, phase, required, budget):
        if phase == "export-reload":
            assert resources["release_checks"][-1]["all_collected"]
            pressure.append(True)
        return original(resources, phase, required, budget)
    monkeypatch.setattr(r, "_phase_guard", pressure_after_release)
    result = recover(teacher, student, tmp_path / "pressured", train, dev, steps=1, rank=2,
                     max_length=12, dtype="bfloat16", export_mode="factor_preserving")
    assert result["status"] == "rejected" and result["actual_steps"] == 1
    assert "export-reload memory headroom" in result["error"]["message"]
    assert result["resources"]["release_checks"][-1]["reclaimed_memory_credit_bytes"] == 0
    assert not (tmp_path / "pressured").exists()


def test_large_constant_header_factor_budget_has_no_second_base_or_embedding_cast(tmp_path, monkeypatch):
    import asea.specialist.recovery as r
    # Arithmetic-only fake metadata: never construct/load a large tensor/model.
    stats = dict(parameters=425000000, adapter_parameters=1250000,
        loaded_bytes=850000000, load_peak_bytes=850016384, tensor_count=200,
        tokenizer_asset_bytes=8 * 1024**2, largest_lora_target_parameters=2000000,
        lora_target_loaded_bytes=100000000, largest_tensor_parameters=134217728)
    monkeypatch.setattr(r, "_header_stats", lambda *a, **k: dict(stats))
    monkeypatch.setattr(r, "_available_ram", lambda: 8 * 1024**3)
    config = Qwen2Config(vocab_size=151936, hidden_size=896, num_hidden_layers=24,
                         num_attention_heads=14, num_key_value_heads=2)
    kwargs = dict(memory_budget_bytes=2669510247, export_mode="factor_preserving")
    result = r._preflight(None, None, tmp_path, config, 8, ["q_proj"], 2, 32, False, torch, **kwargs)
    assert result["estimated_peak_bytes"] < 2669510247
    assert result["export_strategy"] == "native_cpu_contiguous_storage_stream_v1"
    weights = result["export_reload_weights_and_factors_bytes"]
    assert stats["loaded_bytes"] < weights < stats["loaded_bytes"] * 1.1
    assert result["export_reload_estimated_bytes"] == weights + result["export_io_and_metadata_bytes"] + result["export_full_probe_workspace_bytes"]
    with pytest.raises(RecoveryRejected, match="memory headroom"):
        r._preflight(None, None, tmp_path, config, 8, ["q_proj"], 2, 32, False, torch,
                     memory_budget_bytes=2669510247, export_mode="native_merged")
