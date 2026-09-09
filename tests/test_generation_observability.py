"""Offline UNIT model spies/native CPU tensors only; no pretrained performance claims."""
import hashlib
import json
import sys
import wave
from types import SimpleNamespace
from pathlib import Path

import pytest
from pydantic import ValidationError

from asea.artifacts import Blocked, Workspace, digest
from asea.compose import runtime
from asea.compose.schema import CompositionSpec, Node, Limits, dump_spec, fingerprint
from asea.compose import generation_contract as gc


def node_data(kind="hf_text", task="causal", path="UNIT-no-model", **updates):
    return {"id": "unit", "input": "$input", "component": {"kind": kind, "task": task,
            "model_path": str(path), "risk": "low", "provenance": [
                {"source": "UNIT spy, not pretrained", "risk": "low", "license": "UNIT-only"}]},
            **updates}


def contract(format="raw_python"):
    return {"schema_version": 1, "format": format}


def test_legacy_canonical_serialization_including_nested_and_json():
    node = Node.model_validate(node_data())
    expected = node_data(prompt_prefix="", max_new_tokens=128, dtype="float32", use_cache=False)
    expected["component"]["schema_version"] = 1
    assert dump_spec(node) == expected
    assert node.model_dump() == expected
    assert json.loads(node.model_dump_json()) == expected
    assert fingerprint(node) == digest(expected)
    spec = CompositionSpec(name="UNIT", input_type="text", nodes=[node], output_node="unit")
    assert dump_spec(spec)["nodes"] == [expected]
    assert json.loads(spec.model_dump_json())["nodes"] == [expected]
    explicit_null = Node.model_validate({**node_data(), "output_contract": None, "cache_policy": None})
    assert dump_spec(explicit_null) == expected


@pytest.mark.parametrize("field,value", [("cache_policy", "enabled"), ("cache_policy", "model_default"),
                                         ("cache_policy", "disabled"), ("output_contract", contract())])
def test_new_explicit_settings_are_bound(field, value):
    original = Node.model_validate(node_data())
    new = Node.model_validate(node_data(**{field: value}))
    assert dump_spec(new)[field] == value
    assert fingerprint(new) != fingerprint(original)
    assert "use_cache" in dump_spec(new)


@pytest.mark.parametrize("format", ["raw_python", "fenced_python"])
def test_single_renderer_does_not_interpret_or_rewrite_prose(format):
    node = Node.model_validate(node_data(output_contract=contract(format)))
    task = "The task itself mentions fenced_python and raw_python.\n```broken\n  α\n"
    rendered = gc.render_prompt(node, task)
    assert rendered == gc.migration_preview(node, format)["instruction"] + "\n\n" + task
    assert rendered.endswith(task)
    gc.preflight_node(node, [format, format])
    with pytest.raises(ValueError, match="conflicting"):
        gc.preflight_node(node, ["fenced_python" if format == "raw_python" else "raw_python"])


def test_migration_is_read_only_and_prefix_conflict_blocks():
    node = Node.model_validate(node_data(prompt_prefix="Write a function. Return fences. "))
    before = dump_spec(node)
    preview = gc.migration_preview(node, "raw_python")
    assert preview["read_only"] and preview["requires_prefix_review"] and not preview["applied"]
    assert preview["existing_prompt_prefix"] == node.prompt_prefix
    assert dump_spec(node) == before
    assert gc.render_prompt(node, "task") == node.prompt_prefix + "task"
    with pytest.raises(ValidationError, match="prompt_prefix"):
        Node.model_validate({**before, "output_contract": contract()})


@pytest.mark.parametrize("bad", [{"schema_version": True, "format": "raw_python"},
                                 {"schema_version": 2, "format": "raw_python"},
                                 {"schema_version": 1, "format": "json"},
                                 {"schema_version": 1, "format": "raw_python", "repair": True}])
def test_closed_contract(bad):
    with pytest.raises(ValidationError):
        Node.model_validate(node_data(output_contract=bad))


@pytest.mark.parametrize("kind,task,policy,expected", [
    ("hf_text", "causal", None, {"use_cache": False}),
    ("hf_asr", "whisper", None, {}), ("hf_vision", "smolvlm", None, {"use_cache": False}),
    ("hf_tts", "vits", None, {}),
    ("hf_text", "causal", "enabled", {"use_cache": True}),
    ("hf_text", "causal", "disabled", {"use_cache": False}),
    ("hf_text", "seq2seq", "model_default", {}),
    ("hf_asr", "whisper", "enabled", {"use_cache": True}),
    ("hf_asr", "whisper", "disabled", {"use_cache": False}),
    ("hf_asr", "whisper", "model_default", {}),
    ("hf_vision", "smolvlm", "disabled", {"use_cache": False}),
])
def test_adapter_cache_matrix(kind, task, policy, expected):
    node = Node.model_validate(node_data(kind, task, cache_policy=policy))
    assert gc.cache_kwargs(node) == expected
    # Legacy flag is not a competing setting for new policy or speech/vision.
    changed = node.model_copy(update={"use_cache": True})
    assert gc.cache_kwargs(changed) == ({"use_cache": True} if kind == "hf_text" and policy is None else expected)


@pytest.mark.parametrize("kind,task,update", [
    ("hf_tts", "vits", {"output_contract": contract()}),
    ("hf_asr", "whisper", {"output_contract": contract()}),
    ("hf_vision", "smolvlm", {"output_contract": contract()}),
    ("hf_tts", "vits", {"cache_policy": "model_default"}),
    ("hf_tts", "vits", {"cache_policy": "disabled"}),
    ("hf_vision", "smolvlm", {"cache_policy": "enabled"}),
    ("hf_vision", "smolvlm", {"cache_policy": "model_default"}),
])
def test_unsupported_settings_block_before_any_file_or_model_load(tmp_path, monkeypatch, kind, task, update):
    raw = node_data(kind, task, path=tmp_path / "DOES-NOT-EXIST", **update)
    with pytest.raises(ValidationError):
        Node.model_validate(raw)
    node = Node.model_validate(node_data(kind, task)).model_copy(update=update)
    monkeypatch.setattr(runtime, "safe_file", lambda *args: pytest.fail("must block before loading"))
    traces = []
    with gc.capture_traces(traces), pytest.raises(Blocked):
        runtime._adapter(node, {"text": "task"}, tmp_path / "unused.wav", Limits(), False)
    assert traces[0]["stage"] == "preflight"
    assert traces[0]["generated_token_count"] is None


@pytest.fixture
def spy(monkeypatch):
    """No native checkpoint, downloading, HF caches or pretrained inference."""
    torch = pytest.importorskip("torch")
    calls = {"decoded_text": "```broken fence left unchanged", "generation_count": 0}

    class Tokenizer:
        chat_template = "UNIT"
        pad_token_id = None
        eos_token_id = 9
        model_max_length = 2048

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["tokenizer_load"] = kwargs
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            calls["messages"] = messages
            calls["template_kwargs"] = kwargs
            return "UNIT-CHAT:" + messages[0]["content"]

        def __call__(self, *args, **kwargs):
            calls["tokenizer_call"] = (args, kwargs)
            return {"input_ids": torch.tensor([[11, 12]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}

        def decode(self, tokens, **kwargs):
            calls["decoded_ids"] = tokens.tolist()
            return calls["decoded_text"]

        def batch_decode(self, tokens, **kwargs):
            calls["decoded_ids"] = tokens.tolist()
            return [calls["decoded_text"]]

    class Model:
        config = SimpleNamespace(max_position_embeddings=2048, use_cache=True, eos_token_id=9,
                                 sampling_rate=16000, decoder_start_token_id=1)
        generation_config = SimpleNamespace(use_cache=True, eos_token_id=[9, 10], decoder_start_token_id=1,
                                            do_sample=False, num_beams=1, output_scores=False)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["model_load"] = kwargs
            return cls(), {}

        def parameters(self):
            return iter(())

        def cpu(self):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            calls["generation_count"] += 1
            calls["generate"] = kwargs
            if calls.get("fail"):
                raise RuntimeError("UNIT generation failed")
            return torch.tensor([calls.get("returned_ids", [11, 12, 77, 9])])

        def __call__(self, **kwargs):
            calls["model_call"] = kwargs
            return SimpleNamespace(waveform=torch.tensor([[0.0, 0.25, -0.25, 0.0]]))

    fake = SimpleNamespace(__version__="4.51.3", AutoTokenizer=Tokenizer, WhisperProcessor=Tokenizer,
        AutoModelForCausalLM=Model, AutoModelForSeq2SeqLM=Model, WhisperForConditionalGeneration=Model,
        VitsModel=Model, utils=SimpleNamespace(logging=SimpleNamespace(set_verbosity_error=lambda: None)))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return calls, fake


def make_node(tmp_path, kind="hf_text", task="causal", **fields):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": {"hf_asr": "whisper", "hf_tts": "vits"}.get(kind, "llama"),
                                                   "is_encoder_decoder": task == "seq2seq"}))
    return Node.model_validate(node_data(kind, task, path=tmp_path, max_new_tokens=2, **fields))


@pytest.mark.parametrize("policy,forwarded", [(None, False), ("model_default", "omitted"), ("enabled", True), ("disabled", False)])
def test_text_spy_actual_kwargs_tokens_and_unchanged_output(tmp_path, spy, policy, forwarded):
    calls, _ = spy
    node = make_node(tmp_path, cache_policy=policy)
    traces = []
    with gc.capture_traces(traces):
        result = runtime._adapter(node, {"type": "text", "text": "original task"}, tmp_path / "out.wav", Limits(), False)
    assert result == {"type": "text", "text": calls["decoded_text"]}
    assert calls["generation_count"] == 1
    assert calls["messages"] == [{"role": "user", "content": "original task"}]
    assert calls["tokenizer_call"][1] == {"return_tensors": "pt", "truncation": False, "add_special_tokens": False}
    assert str(calls["model_load"]["torch_dtype"]) == "torch.float32"
    assert calls["generate"].get("use_cache", "omitted") == forwarded
    assert calls["generate"]["max_new_tokens"] == 2
    assert calls["generate"]["do_sample"] is False and calls["generate"]["num_beams"] == 1
    assert calls["generate"]["pad_token_id"] == 9
    assert "return_dict_in_generate" not in calls["generate"] and "output_scores" not in calls["generate"]
    trace = traces[0]
    assert trace["schema"] == "generation_trace_v1" and trace["status"] == "succeeded"
    assert trace["input_token_count"] == 2 and trace["output_token_count"] == 2
    assert trace["generated_token_ids"] == [77, 9] and trace["generated_token_count"] == 2
    assert trace["eos_positions"] == [1] and trace["cap_reached"] is True
    assert trace["stop_reason"] == "unknown"  # EOS and cap coincide; don't pretend certainty.
    assert trace["requested"]["cache_policy"] == policy
    assert trace["resolved"]["config"]["use_cache"] is True
    assert trace["resolved"]["cache_internal_state"] == "not_observed"
    assert trace["input_binding"]["text_sha256"] == hashlib.sha256(b"original task").hexdigest()
    assert trace["forwarded"]["input_tensors"]["input_ids"] == {"shape": [1, 2], "dtype": "torch.int64"}
    assert len(trace["config_metadata_sha256"]) == 64
    assert len(json.dumps(trace).encode()) < gc.MAX_TRACE_BYTES


def test_explicit_instruction_once_before_unchanged_chat_template(tmp_path, spy):
    calls, _ = spy
    node = make_node(tmp_path, output_contract=contract("fenced_python"))
    task = "Return a function. raw_python mentioned in prose."
    result = runtime._adapter(node, {"text": task}, tmp_path / "out.wav", Limits(), False)
    assert calls["messages"][0]["content"] == gc.render_prompt(node, task)
    assert calls["tokenizer_call"][0][0] == "UNIT-CHAT:" + gc.render_prompt(node, task)
    assert result["text"] == "```broken fence left unchanged"


def test_seq2seq_ids_are_not_retokenized_or_counted_as_prompt(tmp_path, spy):
    calls, _ = spy
    calls["returned_ids"] = [1, 77, 9]
    node = make_node(tmp_path, task="seq2seq")
    traces = []
    with gc.capture_traces(traces):
        runtime._adapter(node, {"text": "task"}, tmp_path / "unused.wav", Limits(), False)
    assert calls["decoded_ids"] == [1, 77, 9]  # Historic decoder input remains unchanged.
    assert traces[0]["output_token_count"] == 3
    assert traces[0]["generated_token_ids"] == [77, 9]
    assert traces[0]["generated_token_count"] == 2


def hostile_exception(hooks):
    def trap(*args, **kwargs):
        hooks.append("hook")
        raise SystemExit("diagnostic hook must not execute")

    class Meta(type):
        __getattribute__ = trap
        __repr__ = trap
        __str__ = trap
        __name__ = property(trap)
        __mro__ = property(trap)

    class HostileError(Exception, metaclass=Meta):
        __str__ = trap
        __repr__ = trap
        __class__ = property(trap)
        args = property(trap)

    return HostileError("untrusted")


@pytest.mark.parametrize("case", ["surrogate", "throwing_str", "hostile", "argument", "cancelled"])
@pytest.mark.parametrize("capture", [False, True])
def test_failure_diagnostics_never_invoke_hooks_or_replace_original(monkeypatch, case, capture):
    hooks = []
    class BrokenStringError(Exception):
        def __str__(self):
            hooks.append("str")
            raise ValueError("secondary failure")
        def __repr__(self):
            hooks.append("repr")
            raise ValueError("secondary failure")
    original = {"surrogate": RuntimeError("\ud800"), "throwing_str": BrokenStringError(),
                "hostile": hostile_exception(hooks), "argument": RuntimeError(BrokenStringError()),
                "cancelled": KeyboardInterrupt("cancel")}[case]
    def fail(*args):
        if args[-1] is not None:
            args[-1]["stage"] = "generation"
        raise original
    monkeypatch.setattr(runtime, "_adapter_impl", fail)
    traces = []
    with gc.capture_traces(traces if capture else None):
        try:
            runtime._adapter(Node.model_validate(node_data()), {"text": "task"}, "unused", Limits(), False)
        except BaseException as observed:
            assert observed is original
        else:
            pytest.fail("original failure did not propagate")
    assert hooks == []
    if capture:
        assert traces[0]["status"] == ("cancelled" if case == "cancelled" else "failed")
        assert traces[0]["stage"] == "generation"
        assert traces[0]["failure"]["message_availability"] == "not_collected_no_exception_hooks"
        assert "message_sha256" not in traces[0]["failure"]
        assert len(gc._bytes(traces[0])) <= gc.MAX_TRACE_BYTES


@pytest.mark.parametrize("hook", ["new_trace", "failure_metadata", "append_trace"])
def test_diagnostic_baseexceptions_do_not_replace_primary(monkeypatch, hook):
    original = RuntimeError("original")
    def fail(*args):
        raise original
    def broken(*args):
        raise SystemExit("diagnostic failure")
    monkeypatch.setattr(runtime, "_adapter_impl", fail)
    monkeypatch.setattr(runtime, hook, broken)
    traces = []
    with gc.capture_traces(traces), pytest.raises(RuntimeError) as caught:
        runtime._adapter(Node.model_validate(node_data()), {"text": "task"}, "unused", Limits(), False)
    assert caught.value is original
    if hook == "failure_metadata":
        assert traces[0]["status"] == "failed"


def test_observation_failure_with_hostile_metadata_is_fail_open():
    hooks = []
    error = hostile_exception(hooks)
    def fail(trace):
        raise error
    trace = {}
    gc.observe(trace, fail)
    assert trace == {"observation_error": "opaque"}
    assert hooks == []


def test_failure_trace_has_observed_inputs_but_no_fake_output_counts(tmp_path, spy):
    calls, _ = spy
    calls["fail"] = True
    node = make_node(tmp_path)
    traces = []
    with gc.capture_traces(traces), pytest.raises(RuntimeError, match="UNIT generation failed"):
        runtime._adapter(node, {"text": "task"}, tmp_path / "unused.wav", Limits(), False)
    trace = traces[0]
    assert trace["status"] == "failed" and trace["stage"] == "generation"
    assert trace["failure"]["exception_type"] == "RuntimeError"
    assert trace["input_token_count"] == 2
    for key in ("output_token_count", "generated_token_count", "generated_token_ids", "eos_positions", "cap_reached"):
        assert trace[key] is None


def test_observation_error_cannot_change_generation(tmp_path, spy, monkeypatch):
    calls, _ = spy
    node = make_node(tmp_path)
    def broken(*args, **kwargs):
        raise RuntimeError("UNIT observation failure")
    monkeypatch.setattr(runtime, "observe_tokens", broken)
    traces = []
    with gc.capture_traces(traces):
        result = runtime._adapter(node, {"text": "task"}, tmp_path / "unused.wav", Limits(), False)
    assert result["text"] == calls["decoded_text"]
    assert calls["generation_count"] == 1 and traces[0]["status"] == "succeeded"
    assert traces[0]["observation_error"] == "RuntimeError"


@pytest.mark.parametrize("policy,forwarded", [(None, "omitted"), ("model_default", "omitted"), ("enabled", True), ("disabled", False)])
def test_asr_actual_audio_and_processor_facts(tmp_path, spy, policy, forwarded):
    calls, _ = spy
    node = make_node(tmp_path, "hf_asr", "whisper", cache_policy=policy)
    wav = tmp_path / "unit.wav"
    pcm = b"\x00\x00" * 160
    with wave.open(str(wav), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(pcm)
    value = {"type": "audio", **runtime._audio_info(wav, Limits())}
    traces = []
    with gc.capture_traces(traces):
        runtime._adapter(node, value, tmp_path / "unused.wav", Limits(), False)
    trace = traces[0]
    assert calls["generate"].get("use_cache", "omitted") == forwarded
    assert calls["tokenizer_call"][1] == {"sampling_rate": 16000, "return_tensors": "pt", "return_attention_mask": True}
    assert trace["media"]["input_audio_sample_count"] == 160
    assert trace["media"]["input_audio_sample_rate"] == 8000
    assert trace["media"]["processed_audio_sample_count"] == 320
    assert trace["media"]["processed_audio_sample_rate"] == 16000
    assert trace["media"]["input_audio_duration_seconds"] == trace["media"]["processed_audio_duration_seconds"] == 0.02
    assert trace["media"]["decoded_pcm_sha256"] == hashlib.sha256(pcm).hexdigest()
    assert trace["input_binding"]["sha256"] == value["sha256"]
    assert trace["returned_sequence_token_ids"] == [11, 12, 77, 9]
    assert trace["generated_token_count"] is None and trace["stop_reason"] == "unknown"


def test_tts_waveform_facts_do_not_introduce_generate_kwargs(tmp_path, spy):
    calls, _ = spy
    node = make_node(tmp_path, "hf_tts", "vits")
    traces = []
    with gc.capture_traces(traces):
        result = runtime._adapter(node, {"text": "task"}, tmp_path / "out.wav", Limits(), False)
    assert result["type"] == "audio" and result["duration_seconds"] == 4 / 16000
    assert calls["generation_count"] == 0
    assert traces[0]["media"] == {"output_audio_sample_count": 4, "output_audio_sample_rate": 16000,
                                 "output_audio_duration_seconds": 4 / 16000}
    assert "generate_kwargs" not in traces[0]["forwarded"]
    assert traces[0]["generated_token_count"] is None


@pytest.mark.parametrize("source", ["generation_config", "config"])
@pytest.mark.parametrize("size", [1, 32, 33, 40, 2000])
def test_eos_full_config_binding_and_explicit_truncation(spy, source, size):
    torch = pytest.importorskip("torch")
    _, fake = spy
    model = fake.AutoModelForCausalLM()
    eos = list(range(100, 100 + size))
    model.generation_config = SimpleNamespace(eos_token_id=eos if source == "generation_config" else None)
    model.config = SimpleNamespace(eos_token_id=eos if source == "config" else [9999])
    node = Node.model_validate(node_data())
    trace = gc.new_trace(node, {"text": "task"})
    kwargs = {"do_sample": False, "num_beams": 1}
    gc.observe_call(trace, model, {"input_ids": torch.tensor([[1]])}, kwargs, node)
    gc.observe_tokens(trace, torch.tensor([[1, eos[-1], 7]]), model, node, 1)
    expected_hash = hashlib.sha256(json.dumps(eos, separators=(",", ":")).encode()).hexdigest()
    assert trace["eos_token_ids"] == eos[:32]
    assert trace["eos_token_ids_total"] == size
    assert trace["eos_token_ids_truncated"] is (size > 32)
    assert trace["eos_configuration"] == {
        "source": "model." + source + ".eos_token_id", "sha256": expected_hash,
        "hash_basis": "ordered_integer_list_json", "observation": "post_generation_model_configuration",
        "effective_stopping_configuration": "not_verified"}
    assert trace["eos_positions"] == [0] and trace["matched_eos_token_ids"] == [eos[-1]]
    assert trace["eos_positions_basis"] == "generated_token_ids"
    assert trace["token_ids_truncated"] is False and trace["stop_reason"] == "unknown"
    declaration = trace["resolved"][source]["eos_token_id"]
    if size > 32:
        assert declaration["sha256"] == expected_hash
        assert declaration["total"] == size and declaration["truncated"] is True
        assert declaration["values"] == eos[:32]
    else:
        assert declaration == eos
    assert getattr(model, source).eos_token_id is eos
    assert kwargs == {"do_sample": False, "num_beams": 1}
    assert gc.bounded_trace(trace)["eos_configuration"]["sha256"] == expected_hash


def test_eos_hash_binds_tail_and_reports_post_generation_provenance(spy):
    torch = pytest.importorskip("torch")
    _, fake = spy
    model = fake.AutoModelForCausalLM()
    model.generation_config = SimpleNamespace(eos_token_id=list(range(40)))
    node = Node.model_validate(node_data())
    trace = gc.new_trace(node, {"text": "task"})
    gc.observe_call(trace, model, {}, {}, node)
    before = trace["resolved"]["generation_config"]["eos_token_id"]["sha256"]
    model.generation_config.eos_token_id[-1] = 999
    gc.observe_tokens(trace, torch.tensor([[1, 999]]), model, node, 1)
    assert trace["eos_configuration"]["sha256"] != before
    assert trace["eos_configuration"]["sha256"] == gc._hash(model.generation_config.eos_token_id)
    assert trace["eos_positions"] == [0] and trace["matched_eos_token_ids"] == [999]


def test_bounded_ids_and_metadata_no_large_tensor_copies(spy):
    torch = pytest.importorskip("torch")
    _, fake = spy
    node = Node.model_validate(node_data(max_new_tokens=512))
    trace = gc.new_trace(node, {"text": "task"})
    gc.observe_tokens(trace, torch.arange(2000).reshape(1, -1), fake.AutoModelForCausalLM(), node, 2)
    assert len(trace["generated_token_ids"]) == 512
    assert trace["token_ids_truncated"] is True and trace["generated_token_count"] == 1998
    trace["resolved"] = {"too_large": "x" * (gc.MAX_METADATA_BYTES + 1)}
    bounded = gc.bounded_trace(trace)
    assert "resolved" not in bounded and bounded["metadata_omitted"]["reason"] == "byte_limit"
    assert len(gc._bytes(bounded)) <= gc.MAX_TRACE_BYTES
    sink = []
    with gc.capture_traces(sink):
        for i in range(16):
            payload = {**trace, "node": "node" + str(i), "resolved": {"big": "x" * 6000}, "extra": "x" * 6000}
            gc.append_trace(payload)
    assert len(sink) == 16 and len(gc._bytes(sink)) <= gc.MAX_RUN_TRACE_BYTES
    assert any(t.get("evidence_omitted") == "run_byte_limit" for t in sink)


def test_vision_observed_tiles_shapes_and_legacy_false_cache(tmp_path, spy, monkeypatch):
    torch = pytest.importorskip("torch")
    Image = pytest.importorskip("PIL.Image")
    calls, fake = spy
    class VisionProcessor(fake.AutoTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            calls["vision_messages"] = messages
            return "UNIT visual task"

        def __call__(self, *args, **kwargs):
            calls["vision_processor_kwargs"] = kwargs
            return {"input_ids": torch.tensor([[11, 12]]),
                    "pixel_values": torch.zeros(1, 3, 3, 2, 2),
                    "pixel_attention_mask": torch.tensor([[[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[0, 0], [0, 0]]]])}
    fake.AutoProcessor = fake.Idefics3Processor = VisionProcessor
    fake.Idefics3ForConditionalGeneration = fake.AutoModelForCausalLM
    fake.AutoModelForCausalLM.config.text_config = SimpleNamespace(max_position_embeddings=128, use_cache=True)
    monkeypatch.setattr(runtime, "_rss_mb", lambda: 100.0)
    (tmp_path / "config.json").write_text('{"model_type":"idefics3"}')
    node = Node.model_validate(node_data("hf_vision", "smolvlm", path=tmp_path, max_new_tokens=2, use_cache=True))
    image_path = tmp_path / "unit.png"
    Image.new("RGB", (16, 12), "red").save(image_path)
    value = {"type": "image", "text": "UNIT task", **runtime._image_info(image_path, Limits())}
    traces = []
    with gc.capture_traces(traces):
        result = runtime._adapter(node, value, tmp_path / "unused.wav", Limits(), False)
    assert result["text"] == calls["decoded_text"]
    assert calls["generate"]["use_cache"] is False
    assert calls["vision_processor_kwargs"]["add_special_tokens"] is False
    trace = traces[0]
    assert trace["media"]["observed_nonempty_image_tile_count"] == 2
    assert trace["forwarded"]["input_tensors"]["pixel_values"]["shape"] == [1, 3, 3, 2, 2]
    assert trace["input_binding"]["sha256"] == value["sha256"]
    assert trace["generated_token_ids"] == [77, 9]
    assert trace["media"]["image_width"] == 16 and trace["media"]["image_height"] == 12


def test_hooks_leave_spy_input_output_and_generate_kwargs_identical(tmp_path, spy, monkeypatch):
    calls, _ = spy
    node = make_node(tmp_path, prompt_prefix="legacy:")
    first = runtime._adapter(node, {"text": "task"}, tmp_path / "unused.wav", Limits(), False)
    before = {k: v.clone() if hasattr(v, "clone") else v for k, v in calls["generate"].items()}
    prompt = calls["tokenizer_call"]
    monkeypatch.setattr(runtime, "observe", lambda *args, **kwargs: None)
    second = runtime._adapter(node, {"text": "task"}, tmp_path / "unused.wav", Limits(), False)
    assert first == second and prompt == calls["tokenizer_call"]
    assert before.keys() == calls["generate"].keys()
    for key, value in before.items():
        actual = calls["generate"][key]
        assert value.equal(actual) if hasattr(value, "equal") else value == actual


def test_old_fake_node_without_new_fields_keeps_legacy_contract():
    node = SimpleNamespace(component=SimpleNamespace(kind="hf_text"), prompt_prefix="prefix:", use_cache=True)
    assert gc.render_prompt(node, "task") == "prefix:task"
    assert gc.cache_kwargs(node) == {"use_cache": True}


@pytest.mark.parametrize("case", ["normal", "empty", "surrogate", "hostile", "blocked"])
def test_run_primary_error_safe_fallback_and_original_identity(tmp_path, monkeypatch, case):
    hooks = []
    original = {"normal": RuntimeError("legacy error"), "empty": RuntimeError(),
                "surrogate": RuntimeError("\ud800"), "hostile": hostile_exception(hooks),
                "blocked": Blocked("legacy block")}[case]
    model = tmp_path / "fixture"
    model.mkdir()
    (model / "fixture.json").write_text('{"prefix":"echo:"}')
    node = Node.model_validate(node_data("fixture_text", "fixture", model))
    spec = CompositionSpec(name="UNIT", input_type="text", nodes=[node], output_node="unit")
    ws = Workspace(tmp_path / "workspace")
    monkeypatch.setattr(runtime, "_rss_mb", lambda: 100.0)
    def fail(*args):
        raise original
    monkeypatch.setattr(runtime, "_adapter_impl", fail)
    try:
        runtime.run_spec(ws, spec, "task", allow_fixtures=True)
    except BaseException as observed:
        assert observed is original
    else:
        pytest.fail("original error did not propagate")
    assert hooks == []
    records = [ws.read_record("runs", p.stem) for p in (ws.root / "runs").glob("*.json")]
    assert len(records) == 1
    record = records[0]
    assert record["status"] == ("blocked" if case == "blocked" else "failed")
    assert record["generation_trace_v1"][0]["status"] == record["status"]
    if case == "hostile":
        assert record["error"] == "exception message unavailable (unsafe formatting omitted)"
        assert record["error_metadata"] == {"exception_type": "opaque",
                                             "message_availability": "unavailable_unsafe_formatting"}
    else:
        assert record["error"] == {"normal": "legacy error", "empty": "RuntimeError",
                                   "surrogate": "\ud800", "blocked": "legacy block"}[case]
        assert "error_metadata" not in record
    assert "output" not in record


def test_run_appends_per_node_evidence_and_persists_failures(tmp_path, monkeypatch):
    model = tmp_path / "fixture"
    model.mkdir()
    (model / "fixture.json").write_text('{"prefix":"echo:"}')
    node = Node.model_validate(node_data("fixture_text", "fixture", model))
    spec = CompositionSpec(name="UNIT only", input_type="text", nodes=[node], output_node="unit")
    ws = Workspace(tmp_path / "workspace")
    monkeypatch.setattr(runtime, "_rss_mb", lambda: 100.0)
    run = runtime.run_spec(ws, spec, "task", allow_fixtures=True)
    assert run["output"] == {"type": "text", "text": "echo:task"}
    assert run["generation_trace_schema"] == gc.TRACE_SCHEMA
    assert run["generation_trace_v1"][0]["node"] == "unit"
    assert "output_contract" not in run.get("spec", {})
    def fail(*args, **kwargs):
        raise RuntimeError("UNIT fail after wrapper")
    monkeypatch.setattr(runtime, "_adapter_impl", fail)
    with pytest.raises(RuntimeError, match="UNIT fail"):
        runtime.run_spec(ws, spec, "task", allow_fixtures=True)
    records = [ws.read_record("runs", p.stem) for p in (ws.root / "runs").glob("*.json")]
    failed = next(r for r in records if r["status"] == "failed")
    assert failed["generation_trace_v1"][0]["status"] == "failed"
    assert failed["generation_trace_v1"][0]["generated_token_ids"] is None
    assert "output" not in failed
