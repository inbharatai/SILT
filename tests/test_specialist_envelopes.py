"""Lifetime arithmetic + actual tiny CPU storage evidence. NOT GPU validation."""
import copy
import json
import weakref
from pathlib import Path

import pytest

from asea.specialist import devices as d
from asea.specialist import evaluation as e

MiB = 1024**2


def config(depth=2, vocab=257):
    return dict(model_type='qwen2', architectures=['Qwen2ForCausalLM'], hidden_size=32,
        intermediate_size=96, num_hidden_layers=depth, vocab_size=vocab,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048)


def test_serial_host_loader_forward_peak_not_sum_or_twice_base():
    budget = d.inference_host_phases(100, 200, 300, runtime_reserve_bytes=7, tokenizer_bytes=3)
    assert budget['loader_bytes'] == 310
    assert budget['forward_bytes'] == budget['estimated_additional_peak_bytes'] == 410
    assert not budget['current_rss_included']
    assert budget['runtime_reserve_is_incremental']


def test_training_depth_linear_never_lm_head_times_depth_and_quadratic_attention():
    def estimate(depth=2, vocab=257, length=16):
        return d.recovery_workspace(config(depth, vocab), 'bfloat16', length, length//2,
            training=True, factor_rank=2, factor_parameters=200, kd=True)
    shallow, deep = estimate(), estimate(depth=12)
    assert shallow['full_output_logits_bytes'] == deep['full_output_logits_bytes']
    assert shallow['response_ce_kl_bytes'] == deep['response_ce_kl_bytes']
    assert shallow['recompute_backward_bytes'] == deep['recompute_backward_bytes']
    assert deep['estimated_additional_peak_bytes'] - shallow['estimated_additional_peak_bytes'] == 10*16*32*4
    wide, wide_deep = estimate(vocab=8192), estimate(depth=12, vocab=8192)
    assert wide_deep['estimated_additional_peak_bytes']-wide['estimated_additional_peak_bytes'] == 10*16*32*4
    x, y, z = (estimate(length=n)['one_block_forward_bytes'] for n in (16,32,64))
    # The linear hidden/MLP term cancels; doubling L doubles excess by four.
    assert z-2*y == 4*(y-2*x) > 0
    assert shallow['factor_parameter_bytes'] == 800
    assert shallow['optimizer_state_bytes'] == 1600
    assert shallow['gradient_bytes'] == 800
    assert shallow['saved_weight_reference_extra_bytes'] == 0
    assert shallow['library_reserve_bytes'] == 128*MiB
    initial = d.recovery_incremental_workspace(shallow, resident_factors=800)
    later = d.recovery_incremental_workspace(shallow, resident_factors=800, resident_optimizer=1600, resident_gradients=800)
    assert initial == shallow['estimated_additional_peak_bytes']-800
    assert later == shallow['forward_backward_transient_bytes']
    assert later == d.recovery_incremental_workspace(shallow, resident_factors=10**9,
        resident_optimizer=10**9, resident_gradients=10**9)  # never credit beyond known modeled buffers


@pytest.mark.parametrize('bound', [0, -1, True, 1.5, 2049])
def test_invalid_declared_input_envelope(bound):
    with pytest.raises((ValueError, d.DeviceRejected)):
        d.inference_envelope(config(), 'float32', 256, 123, max_input_tokens=bound)


def test_input_profile_native_chat_rendering_and_all_inputs():
    class Tokenizer:
        chat_template = 'fixture'
        eos_token_id = 1
        model_max_length = 2048
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert not tokenize and add_generation_prompt
            return 'user role ' + messages[0]['content'] + ' assistant role'
        def encode(self, text, add_special_tokens, truncation):
            assert not add_special_tokens and not truncation
            return list(range(len(text.split())))
    p = e.profile_prompts(Tokenizer(), 'causal', config(), ['one', 'two three four'], 256)
    assert p['input_token_lengths'] == [5,7]
    assert p['max_input_tokens'] == 7 and p['all_inputs_profiled']
    with pytest.raises(ValueError, match='never truncate'):
        e.profile_prompts(Tokenizer(), 'causal', config(), ['one '*1800], 256)


@pytest.mark.parametrize('family', ['causal', 'seq2seq'])
def test_actual_cpu_public_prompt_profile_and_runtime_guard(tmp_path, family, monkeypatch):
    torch = pytest.importorskip('torch')
    from test_specialist_recovery import tiny_stores
    _, student, _, _ = tiny_stores(tmp_path, family)
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = e.infer(student, 'hello world', dtype='float32', max_new_tokens=2, execution_device='auto')
        count = result['generation']['input_token_count']
        assert result['resources']['inference_envelope']['prefill_tokens'] == count
        assert result['resources']['inference_envelope']['max_new_tokens'] == 2
        generator = e.NativeGenerator(student, 'float32', 2, max_input_tokens=count)
        try:
            monkeypatch.setattr(generator.model, 'generate', lambda **kw: pytest.fail('oversize must reject before forward'))
            with pytest.raises(ValueError, match='max_input_tokens.*never truncate'):
                generator.generate('hello world hello world')
        finally:
            generator.close()
    finally:
        torch.set_num_threads(old)


@pytest.mark.parametrize('family', ['causal', 'seq2seq'])
@pytest.mark.parametrize('depth', [2,4])
@pytest.mark.parametrize('length', [16,32])
def test_actual_cpu_saved_tensor_and_live_storage_bounds(family, depth, length, record_property):
    """CPU TorchDispatch live tensor storages + saved_tensors_hooks, not RSS/GPU.

    Unique storage IDs de-duplicate views and aliases. Frozen parameters saved
    by autograd remain references to already-counted model storage. Weakrefs do
    not retain outputs artificially; checkpoint recomputation is observed too.
    """
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    peft = pytest.importorskip('peft')
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    from asea.specialist.recovery import _full_kl, _response_logits
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    response = length//2
    if family == 'causal':
        raw = transformers.Qwen2Config(**{k:v for k,v in config(depth).items() if k!='model_type'})
        cls, task, targets = transformers.Qwen2ForCausalLM, peft.TaskType.CAUSAL_LM, ['q_proj','v_proj','down_proj']
    else:
        raw = transformers.T5Config(vocab_size=257, d_model=32, d_ff=96, d_kv=8, num_heads=4,
            num_layers=depth, num_decoder_layers=depth, decoder_start_token_id=0, pad_token_id=0,
            dropout_rate=0.0)
        cls, task, targets = transformers.T5ForConditionalGeneration, peft.TaskType.SEQ_2_SEQ_LM, ['q','v','wo']
    raw._attn_implementation = 'eager'
    model = peft.get_peft_model(cls(raw), peft.LoraConfig(r=2,lora_alpha=4,lora_dropout=0,
        bias='none',target_modules=targets,task_type=task))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    model.train()
    def storage(t):
        st = t.untyped_storage()
        return (st.data_ptr(), st.nbytes())
    parameters = {storage(p) for p in model.parameters()}
    tensors, stores, saved = {}, {}, {}
    peak = [0]
    parameter_references = [0]
    def track(t):
        if not isinstance(t, torch.Tensor) or not t.numel() or id(t) in tensors:
            return
        key, ident = storage(t), id(t)
        if key in parameters:
            return
        stores[key] = stores.get(key, 0)+1
        def gone(ref):
            tensors.pop(ident, None)
            stores[key] -= 1
            if not stores[key]:
                del stores[key]
        tensors[ident] = weakref.ref(t, gone)
        peak[0] = max(peak[0], sum(k[1] for k in stores))
    class ObserveStorage(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for t in tree_flatten(out)[0]:
                track(t)
            return out
    def pack(t):
        key = storage(t)
        if key in parameters:
            parameter_references[0] += 1
        elif t.numel():
            saved[key] = key[1]
        track(t)
        return t
    factors = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bound = d.recovery_workspace(raw.to_dict(), 'float32', length, response, training=True,
        factor_rank=2, factor_parameters=factors, kd=True)
    try:
        with ObserveStorage(), torch.autograd.graph.saved_tensors_hooks(pack, lambda x:x):
            ids = torch.arange(length).unsqueeze(0) % 257
            labels = ids.clone()
            labels[:, :length-response] = -100
            batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            if family != 'causal':
                batch['decoder_input_ids'] = ids
            logits = model(**batch).logits
            selected, target = _response_logits(logits, labels, family)
            teacher = torch.zeros_like(selected)
            loss = torch.nn.functional.cross_entropy(selected.float(), target) + _full_kl(selected, teacher)
            forward_saved = sum(saved.values())
            forward_live = sum(k[1] for k in stores)
            loss.backward()
            backward_peak = peak[0]
        # Exclude the large fixed reserve: actual tensors fit the tensor terms.
        tensor_bound = bound['estimated_additional_peak_bytes']-bound['library_reserve_bytes']
        assert parameter_references[0] > 0  # LM-head/native saved weights are aliases
        assert forward_saved <= tensor_bound
        assert backward_peak <= tensor_bound
        assert bound['resident_base_weights_charged_bytes'] == 0
        evidence = {'scope':'tiny_random_CPU_storage_only_GPU_UNVERIFIED', 'family':family, 'depth':depth, 'length':length,
            'saved_unique_nonparameter_bytes':forward_saved, 'forward_live_nonparameter_bytes':forward_live,
            'forward_backward_peak_nonparameter_bytes':backward_peak, 'tensor_bound_no_reserve_bytes':tensor_bound,
            'saved_parameter_reference_count':parameter_references[0]}
        record_property('storage_evidence', json.dumps(evidence, sort_keys=True))
    finally:
        torch.set_num_threads(old)


def test_planner_profiles_all_governed_dev_inputs_only_and_rejects_changed_suite(tmp_path, monkeypatch):
    from test_hardware_planning import recipe, profile
    from asea.hardware import plan_specialist, data
    from asea.artifacts import file_hash
    cfg = recipe(tmp_path)
    cfg['recovery']['max_samples'] = 1
    suite_path, manifest_path = Path(cfg['validation_suite']), Path(cfg['data_manifest'])
    suite = json.loads(suite_path.read_text())
    suite['cases'][-1]['input'] = 'gamma '*40  # actual suite, NOT training prompt length
    suite_path.write_text(json.dumps(suite))
    manifest = json.loads(manifest_path.read_text())
    manifest['artifact_sha256']['validation-suite.json'] = file_hash(suite_path)['sha256']
    manifest_path.write_text(json.dumps(manifest))
    native_profile = data.native_profile
    def inspect_transport(cfg, raw, splits):
        assert len(splits)==3 and all(type(p) is str for p in splits[2])
        assert all('not tokenized' not in p for p in splits[2])
        return native_profile(cfg, raw, splits)
    monkeypatch.setattr(data, 'native_profile', inspect_transport)
    result = plan_specialist(cfg, profile(tmp_path), workspace_parent=tmp_path)
    assert result['status']=='READY', result['reasons']
    inp = result['data_profile']['evaluation_input_profile']
    assert inp['input_count']==2 and inp['max_input_tokens']==40
    for env in result['estimates']['inference_envelopes'].values():
        assert env['prefill_tokens']==40 and env['max_new_tokens']==cfg['max_new_tokens']
    assert result['estimates']['final_evaluation']['status']=='UNPROFILED_NOT_ADMITTED'
    assert not result['data_profile']['_metadata']['final_opened']
    final = plan_specialist(cfg, profile(tmp_path), workspace_parent=tmp_path, inference_scope='final')
    assert final['status']=='READY' and not final['final_input_fit_proven']
    assert {p['name'] for p in final['phases']} == {'source_evaluation','inference_evaluation'}
    assert all(v['prefill_tokens']==2048 for v in final['estimates']['inference_envelopes'].values())
    assert final['estimates']['final_evaluation']['status']=='CONSERVATIVE_UNPROFILED_ESTIMATE'
    assert final['estimates']['workspace_disk_required_bytes']==128*MiB
    assert not final['data_profile']['_metadata']['final_opened']
    suite_path.write_text(json.dumps(suite)+' ')
    blocked = plan_specialist(cfg, profile(tmp_path), workspace_parent=tmp_path)
    assert blocked['status']=='BLOCKED' and any('suite hash mismatch' in r for r in blocked['reasons'])
