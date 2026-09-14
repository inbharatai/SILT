"""Real random tiny-model decoding mechanics, never pretrained quality evidence."""
import copy
import hashlib
from types import SimpleNamespace

import pytest

from asea.artifacts import model_inventory
from asea.specialist import evaluation as e
from asea.specialist.reconstruction import ReconstructionBlocked


@pytest.fixture
def native(tmp_path):
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    if tf.__version__ != "4.51.3":
        pytest.skip("real generation regression is scoped to validated transformers 4.51.3")
    from test_native_evaluation_loading import tiny_source
    old_threads = torch.get_num_threads()
    with torch.random.fork_rng():
        _, _, root = tiny_source(tmp_path, "qwen2", "float32")
        # Save a real HF config carrying the same five dangerous defaults as
        # retained Qwen. This is a tiny random local fixture, not that artifact.
        cfg = tf.GenerationConfig.from_pretrained(root, local_files_only=True)
        cfg.do_sample, cfg.temperature, cfg.top_k, cfg.top_p = True, 0.7, 20, 0.8
        cfg.repetition_penalty = 1.05
        cfg.save_pretrained(root)
        before = model_inventory(root)
        gen = e.NativeGenerator(root, "float32", 8, max_input_tokens=16)
        try:
            yield gen, root, before
        finally:
            gen.close()
            assert model_inventory(root) == before
            torch.set_num_threads(old_threads)


def instrument(monkeypatch):
    import torch
    from transformers.generation.utils import GenerationMixin
    sample, multinomial = GenerationMixin._sample, torch.multinomial
    observed = {"modes": [], "configs": [], "multinomial": 0}

    def sampling_loop(self, input_ids, logits_processor, stopping_criteria,
                      generation_config, synced_gpus, streamer, **model_kwargs):
        # _sample implements both branches: the *actual* mode/config at this
        # native execution boundary, not the requested cfg seen by a fake model.
        observed["modes"].append(generation_config.get_generation_mode().value)
        observed["configs"].append(generation_config.to_dict())
        return sample(self, input_ids, logits_processor, stopping_criteria,
                      generation_config, synced_gpus, streamer, **model_kwargs)

    def counted(*args, **kwargs):
        observed["multinomial"] += 1
        return multinomial(*args, **kwargs)

    monkeypatch.setattr(GenerationMixin, "_sample", sampling_loop)
    monkeypatch.setattr(torch, "multinomial", counted)
    return observed


def state_hash(model):
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().view(-1).view(__import__("torch").uint8).numpy().tobytes())
    return result.hexdigest()


@pytest.mark.parametrize("factor", [False, True], ids=["native", "peft-causal"])
def test_real_qwen_before_sample_after_greedy(native, monkeypatch, factor):
    import torch
    from transformers import GenerationConfig
    gen, root, before = native
    if factor:
        peft = pytest.importorskip("peft")
        gen.model = peft.get_peft_model(gen.model, peft.LoraConfig(
            task_type="CAUSAL_LM", r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"], lora_dropout=0))
        gen.model.eval().requires_grad_(False)
    observed = instrument(monkeypatch)
    weights_before = state_hash(gen.model)
    saved_before = copy.deepcopy(gen.model.generation_config.to_dict())
    # Exact old NativeGenerator config construction. False is deliberately not
    # enough without the generate use_model_defaults argument in 4.51.3.
    old_cfg = GenerationConfig.from_model_config(gen.model.config)
    old_cfg.do_sample, old_cfg.num_beams, old_cfg.num_return_sequences = False, 1, 1
    old_cfg.max_new_tokens, old_cfg.use_cache = 8, True
    old_cfg.min_length, old_cfg.min_new_tokens = 0, 0
    old_cfg.forced_bos_token_id, old_cfg.forced_eos_token_id = None, None
    old_cfg.repetition_penalty, old_cfg.no_repeat_ngram_size = 1.0, 0
    old_cfg.return_dict_in_generate, old_cfg.output_scores = False, False
    old_cfg.output_logits, old_cfg.output_attentions, old_cfg.output_hidden_states = False, False, False
    old_cfg.pad_token_id = gen.tokenizer.pad_token_id
    inputs = dict(input_ids=torch.tensor([[3, 4, 1]]), attention_mask=torch.ones(1, 3, dtype=torch.long))
    with torch.random.fork_rng(), torch.inference_mode():
        torch.manual_seed(0)
        gen.model.generate(**inputs, generation_config=old_cfg)
    assert old_cfg.do_sample is False  # Old receipt concealed the effective mode.
    assert observed["modes"] == ["sample"]
    assert observed["multinomial"] == 8
    assert observed["configs"][0]["repetition_penalty"] == 1.05
    observed["modes"].clear()
    observed["configs"].clear()
    observed["multinomial"] = 0
    # Instrument actual resolution separately. A pre-resolution cfg copy cannot
    # satisfy this assertion: the receipt must match the resolver's output.
    base = gen.model.get_base_model() if factor else gen.model
    resolver = base._prepare_generation_config
    resolutions = []
    def resolved(*args, **kwargs):
        result = resolver(*args, **kwargs)
        resolutions.append((kwargs.get("use_model_defaults"), result[0].to_dict()))
        return result
    monkeypatch.setattr(base, "_prepare_generation_config", resolved)
    forward_cache = []
    def cache_hook(module, args, kwargs):
        forward_cache.append((kwargs.get("use_cache"), type(kwargs.get("past_key_values")).__name__))
    handle = base.register_forward_pre_hook(cache_hook, with_kwargs=True)
    outputs = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        rng_before = torch.random.get_rng_state().clone()
        output = gen.generate("hello world")
        assert torch.equal(rng_before, torch.random.get_rng_state())
        trace, policy = output["generation"], output["generation"]["generation_policy"]
        assert policy["use_model_defaults"] is False
        assert policy["checkpoint_policy_defaults_disabled"]
        assert not policy["live_generation_mode_observed"]
        assert policy["generation_mode"] == "greedy_search" and trace["greedy"]
        assert resolutions[-1][1] == policy["resolved_generate_config"] == trace["generate_config"]
        cfg = observed["configs"][-1]
        for key, value in dict(do_sample=False, num_beams=1, num_return_sequences=1,
                max_new_tokens=8, use_cache=True, temperature=1.0, top_k=50, top_p=1.0,
                repetition_penalty=1.0, suppress_tokens=None, no_repeat_ngram_size=0).items():
            assert cfg[key] == trace["generate_config"][key] == value
        assert trace["dtype"] == "float32"
        assert all(p.dtype == torch.float32 for p in gen.model.parameters())
        outputs.append(trace["generated_token_ids"])
    handle.remove()
    assert forward_cache and all(value is True and kind == "DynamicCache" for value, kind in forward_cache)
    assert observed["modes"] == ["greedy_search", "greedy_search"]
    assert observed["multinomial"] == 0
    assert outputs[0] == outputs[1]
    assert state_hash(gen.model) == weights_before
    assert gen.model.generation_config.to_dict() == saved_before
    assert model_inventory(root) == before


@pytest.mark.parametrize("factor", [False, True], ids=["native-seq2seq", "peft-seq2seq"])
def test_real_seq2seq_saved_special_tokens_and_wrapper(tmp_path, monkeypatch, factor):
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    from test_native_evaluation_loading import tiny_source
    _, _, root = tiny_source(tmp_path, "t5", "float32")
    gen = e.NativeGenerator(root, "float32", 3, max_input_tokens=16)
    try:
        if factor:
            peft = pytest.importorskip("peft")
            gen.model = peft.get_peft_model(gen.model, peft.LoraConfig(
                task_type="SEQ_2_SEQ_LM", r=2, target_modules=["q", "v"], lora_dropout=0))
            gen.model.eval().requires_grad_(False)
        # Only saved generation metadata supplies decoder start + EOS. The
        # tokenizer supplies prompt EOS/padding; it must not overwrite saved EOS.
        gen.model.config.decoder_start_token_id = None
        gen.model.config.eos_token_id = None
        gen.model.generation_config.decoder_start_token_id = 0
        gen.model.generation_config.eos_token_id = [1, 2]
        gen.model.generation_config.do_sample = True
        observed = instrument(monkeypatch)
        output = gen.generate("hello world")
        policy = output["generation"]["generation_policy"]
        assert policy["requested_generate_config"]["decoder_start_token_id"] is None
        assert policy["resolved_generate_config"]["decoder_start_token_id"] == 0
        assert policy["resolved_generate_config"]["eos_token_id"] == [1, 2]
        assert observed["modes"] == ["greedy_search"] and observed["multinomial"] == 0
        assert observed["configs"][0]["decoder_start_token_id"] == 0
    finally:
        gen.close()


def test_model_config_policy_drift_is_not_inherited(native):
    gen, _, _ = native
    for cfg in (gen.model.config, gen.model.generation_config):
        cfg.do_sample, cfg.temperature, cfg.top_k, cfg.top_p = True, 0.7, 20, 0.8
        cfg.repetition_penalty, cfg.num_beams, cfg.use_cache = 1.05, 4, False
        cfg.suppress_tokens, cfg.bad_words_ids = [7], [[8]]
        cfg.forced_bos_token_id, cfg.forced_eos_token_id = 3, 1
        cfg.max_new_tokens = 42
    cfg, policy = e._generation_policy(gen.model, gen.tokenizer, "causal", 8)
    assert cfg.do_sample is False and cfg.num_beams == 1 and cfg.use_cache is True
    assert cfg.temperature == cfg.top_p == cfg.repetition_penalty == 1.0
    assert cfg.top_k == 50 and cfg.max_new_tokens == 8
    assert cfg.suppress_tokens is cfg.bad_words_ids is cfg.forced_bos_token_id is cfg.forced_eos_token_id is None
    assert policy["resolved_generate_config"] == cfg.to_dict()


@pytest.mark.parametrize("version", ["4.49.0", "4.51.2", "4.52.0", "5.0.0", "unknown"])
def test_unsupported_transformers_fails_closed(native, monkeypatch, version):
    import transformers
    gen, _, _ = native
    monkeypatch.setattr(transformers, "__version__", version)
    monkeypatch.setattr(gen.model, "generate", lambda **kw: pytest.fail("unsupported native generate called"))
    with pytest.raises(ReconstructionBlocked, match="validated transformers"):
        gen.generate("hello")
    assert gen.last_trace["generate_config"] is None
    assert "greedy" not in gen.last_trace


def test_missing_supported_api_fails_closed(native, monkeypatch):
    from transformers.generation.utils import GenerationMixin
    gen, _, _ = native
    monkeypatch.setattr(GenerationMixin, "generate", lambda self, **kw: pytest.fail("called unsupported API"))
    with pytest.raises(ReconstructionBlocked, match="use_model_defaults API"):
        gen.generate("hello")


@pytest.mark.parametrize("missing", ["saved_config", "version", "bad_version", "eos", "bad_eos", "decoder_start"])
def test_missing_or_invalid_metadata_fails_closed(native, missing):
    gen, _, _ = native
    family = "causal"
    if missing == "saved_config":
        gen.model.generation_config = None
    elif missing in ("version", "bad_version"):
        gen.model.generation_config.transformers_version = None if missing == "version" else "nonsense"
    elif missing in ("eos", "bad_eos"):
        gen.model.config.eos_token_id = None
        gen.model.generation_config.eos_token_id = None if missing == "eos" else []
    else:
        family = "seq2seq"
        gen.model.config.decoder_start_token_id = gen.model.generation_config.decoder_start_token_id = None
    with pytest.raises(ReconstructionBlocked, match="metadata"):
        e._generation_policy(gen.model, gen.tokenizer, family, 8)


@pytest.mark.parametrize("field,value", [("do_sample", True), ("max_new_tokens", 9),
    ("use_cache", False), ("temperature", 0.7), ("eos_token_id", 2)])
def test_resolver_policy_drift_fails_closed(native, monkeypatch, field, value):
    gen, _, _ = native
    resolver = gen.model._prepare_generation_config
    def drift(*a, **kw):
        cfg, rest = resolver(*a, **kw)
        setattr(cfg, field, value)
        return cfg, rest
    monkeypatch.setattr(gen.model, "_prepare_generation_config", drift)
    with pytest.raises(ReconstructionBlocked, match="policy drift"):
        gen.generate("hello")


def test_tokenizer_pad_fallback_and_native_special_precedence(native):
    gen, _, _ = native
    gen.model.config.bos_token_id = 3
    gen.model.generation_config.bos_token_id = 4
    cfg, _ = e._generation_policy(gen.model, SimpleNamespace(pad_token_id=None, eos_token_id=1), "causal", 8)
    assert cfg.pad_token_id == 1 and cfg.bos_token_id == 3


def test_public_infer_and_evaluate_keep_resolved_policy(native, tmp_path, monkeypatch):
    from test_specialist_workflow import suite_at
    gen, root, before = native
    gen.close()  # One actual model at a time; source has the sampling config.
    observed = instrument(monkeypatch)
    output = e.infer(root, "hello world", dtype="float32", max_new_tokens=3)
    assert output["generation"]["generation_policy"]["use_model_defaults"] is False
    monkeypatch.setattr(e, "oracle_preflight", lambda: {"available": True})
    # Local original fixture only. No retained data, final tasks or rescoring.
    suite = suite_at(tmp_path / "policy-mechanics-suite.json")
    report = e.evaluate(root, suite, dtype="float32", max_new_tokens=3)
    assert report["inputs_unchanged"] and report["tasks_total"] == 2
    for case in report["cases"]:
        result = case["generation"]
        assert result["status"] == "completed"
        assert result["generation"]["generation_policy"]["generation_mode"] == "greedy_search"
    assert observed["modes"] == ["greedy_search"] * 3
    assert observed["multinomial"] == 0
    assert model_inventory(root) == before


def test_unknown_peft_wrapper_version_fails_closed(native, monkeypatch):
    peft = pytest.importorskip("peft")
    gen, _, _ = native
    gen.model = peft.get_peft_model(gen.model, peft.LoraConfig(task_type="CAUSAL_LM", r=2, target_modules=["q_proj"]))
    monkeypatch.setattr(peft, "__version__", "unknown")
    with pytest.raises(ReconstructionBlocked, match="PEFT 0.15.2"):
        gen.generate("hello")
