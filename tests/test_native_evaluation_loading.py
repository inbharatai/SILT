"""Actual tiny local CPU loading; no pretrained/download or CUDA validation claims."""
import hashlib

import pytest

from asea.specialist import evaluation as e
from asea.specialist import reconstruction as r
from asea.specialist import devices as d


def tiny_source(tmp_path, kind, dtype, tied=True):
    torch = pytest.importorskip('torch')
    tf = pytest.importorskip('transformers')
    from test_specialist_reconstruction import _tokenizer
    torch.set_num_threads(2)
    torch.manual_seed(138)
    if kind in ('qwen2', 'llama'):
        cls = tf.Qwen2ForCausalLM if kind == 'qwen2' else tf.LlamaForCausalLM
        config_cls = tf.Qwen2Config if kind == 'qwen2' else tf.LlamaConfig
        config = config_cls(vocab_size=32, hidden_size=16, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128, tie_word_embeddings=tied, eos_token_id=1, pad_token_id=0)
    else:
        cls = tf.T5ForConditionalGeneration if kind == 't5' else tf.SwitchTransformersForConditionalGeneration
        config_cls = tf.T5Config if kind == 't5' else tf.SwitchTransformersConfig
        opts = dict(num_experts=3, num_sparse_encoder_layers=1, num_sparse_decoder_layers=1,
                    router_jitter_noise=0, router_dtype='float32') if kind != 't5' else {}
        config = config_cls(vocab_size=32, d_model=16, d_kv=4, d_ff=64, num_layers=2,
            num_decoder_layers=2, num_heads=4, decoder_start_token_id=0, eos_token_id=1,
            pad_token_id=0, tie_word_embeddings=tied, dropout_rate=0, **opts)
    config._attn_implementation = 'eager'
    model = cls(config).to(getattr(torch, dtype)).eval()
    # Original F32 router bits are an exception in both old and new loaders.
    for name, parameter in model.named_parameters():
        if '.router.classifier.' in name:
            parameter.data = torch.randn(parameter.shape, dtype=torch.float32)
    path = tmp_path / 'source'
    model.generation_config.suppress_tokens = [7]
    model.generation_config.repetition_penalty = 1.17
    model.generation_config.max_length = 29
    model.save_pretrained(path, safe_serialization=True, max_shard_size='8KB')
    _tokenizer(path, tf)
    del model
    return torch, cls, path


@pytest.mark.parametrize('kind', ['qwen2', 't5', 'switch_transformers', 'llama'])
@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
@pytest.mark.parametrize('tied', [False, True])
def test_native_cpu_bitwise_hf_eager_parity(tmp_path, monkeypatch, kind, dtype, tied):
    torch, cls, path = tiny_source(tmp_path, kind, dtype, tied)
    before, raw, headers = e._evaluation_preflight(path)
    old = cls.from_pretrained(path, local_files_only=True, trust_remote_code=False,
        torch_dtype=getattr(torch, dtype), use_safetensors=True, low_cpu_mem_usage=True,
        attn_implementation='eager').eval().requires_grad_(False)
    r._restore_f32_routers(old, path, headers, torch)
    expected_generation = old.generation_config.to_dict()
    # The actual NativeGenerator must not use an opaque HF model loader.
    def opaque_forbidden(*args, **kwargs):
        raise AssertionError('opaque model loader used')
    monkeypatch.setattr(cls, 'from_pretrained', opaque_forbidden)
    if kind == 'switch_transformers' and not tied:
        # HF leaves missing encoder/decoder embeddings on this checkpoint. The
        # old strict loading-info guard refused it; the explicit plan also must.
        with pytest.raises(r.ReconstructionBlocked, match='missing source keys'):
            e.NativeGenerator(path, dtype=dtype, max_new_tokens=3, max_input_tokens=3)
        assert e._evaluation_preflight(path)[0] == before
        return
    native = e.NativeGenerator(path, dtype=dtype, max_new_tokens=3, max_input_tokens=3)
    assert native.load_plan['strategy'] == d.EXPLICIT_NATIVE_LOADER
    assert native.model.generation_config.to_dict() == expected_generation
    assert all(not t.is_meta for t in list(native.model.parameters()) + list(native.model.buffers()))
    for alias, key in native.load_plan['aliases'].items():
        actual, previous = native.model.get_parameter(alias), old.get_parameter(alias)
        if kind == 't5' and dtype == 'bfloat16' and '.wo.weight' in alias:
            assert actual.dtype == torch.float32 and previous.dtype == torch.bfloat16
        else:
            assert actual.dtype == previous.dtype, alias
        assert torch.equal(actual, previous), alias
        assert actual is native.model.get_parameter(next(a for a, k in native.load_plan['aliases'].items() if k == key))
    inputs = dict(input_ids=torch.tensor([[3, 4, 1]]), attention_mask=torch.ones(1, 3, dtype=torch.long))
    forward = dict(inputs)
    if kind not in ('qwen2', 'llama'):
        forward['decoder_input_ids'] = torch.tensor([[0, 3]])
    with torch.inference_mode():
        if kind == 't5' and dtype == 'bfloat16':
            # HF 4.51.3 rounds/keeps wo in BF16 despite native _keep_in_fp32_modules.
            # This intentional native precision correction is NOT old-HF bitwise
            # parity. Prove the difference, then exact parity to canonical wo F32.
            assert not torch.equal(native.model(**forward).logits, old(**forward).logits)
            for name, parameter in old.named_parameters():
                if '.wo.weight' in name:
                    parameter.data = parameter.data.float()
        assert torch.equal(native.model(**forward).logits, old(**forward).logits)
        assert torch.equal(native.model.generate(**inputs, max_new_tokens=3, do_sample=False),
                           old.generate(**inputs, max_new_tokens=3, do_sample=False))
    assert e._evaluation_preflight(path)[0] == before
    native.close()


@pytest.mark.parametrize('kind', ['qwen2', 't5', 'switch_transformers', 'llama'])
def test_actual_cast_plan_preserves_source_exception_bits(tmp_path, kind):
    torch, _, path = tiny_source(tmp_path, kind, 'float32')
    before, raw, headers = e._evaluation_preflight(path)
    meta, plan = e._evaluation_inputs(raw, headers, 'bfloat16', torch)
    stats = d.explicit_loader_storage(headers, plan, before)
    assert stats['retained_cast_source_bytes'] > 0
    assert stats['largest_cast_scratch_bytes'] > 0
    from asea.specialist.recovery import _load_recovery_model
    from safetensors import safe_open
    model = _load_recovery_model(meta, path, headers, plan, torch)
    for key, entry in headers.items():
        with safe_open(str(path / entry['file']), framework='pt') as handle:
            original = handle.get_tensor(key)
        alias = next(a for a, stored in plan['aliases'].items() if stored == key)
        actual = model.get_parameter(alias)
        target = getattr(torch, plan['target_dtypes'][key])
        assert actual.dtype == target
        assert torch.equal(actual, original.to(target))
        if kind == 't5' and '.wo.weight' in key or '.router.classifier.' in key:
            assert actual.dtype == torch.float32
            assert hashlib.sha256(actual.detach().numpy().tobytes()).digest() == hashlib.sha256(original.numpy().tobytes()).digest()
    assert sum(p.numel()*p.element_size() for p in model.parameters()) == stats['loaded_bytes']
    assert e._evaluation_preflight(path)[0] == before


def test_header_plan_never_reads_weight_payload_and_cpu_buffers_are_real(tmp_path, monkeypatch):
    torch, _, path = tiny_source(tmp_path, 'qwen2', 'bfloat16')
    before, raw, headers = e._evaluation_preflight(path)
    import safetensors
    monkeypatch.setattr(safetensors, 'safe_open', lambda *a, **kw: pytest.fail('weight read during meta planning'))
    meta, plan = e._evaluation_inputs(raw, headers, 'bfloat16', torch)
    assert all(p.is_meta for p in meta.parameters())
    assert list(meta.buffers()) and all(not b.is_meta and b.device.type == 'cpu' for b in meta.buffers())
    stats = d.explicit_loader_storage(headers, plan, before)
    assert stats['retained_cast_source_bytes'] == stats['largest_cast_scratch_bytes'] == 0
    assert stats['mapping_overhead_bytes'] > 0
    assert stats['buffer_reserve_bytes'] >= sum(b.numel()*b.element_size() for b in meta.buffers())


@pytest.mark.parametrize('mutation', ['missing', 'shape', 'extra'])
def test_strict_headers_fail_before_tensor_read(tmp_path, monkeypatch, mutation):
    torch, _, path = tiny_source(tmp_path, 'llama', 'float32')
    _, raw, headers = e._evaluation_preflight(path)
    key = next(iter(headers))
    if mutation == 'missing':
        del headers[key]
    elif mutation == 'shape':
        headers[key] = dict(headers[key], shape=[1])
    else:
        headers['not.a.native.weight'] = dict(headers[key])
    import safetensors
    monkeypatch.setattr(safetensors, 'safe_open', lambda *a, **kw: pytest.fail('unsafe tensor read'))
    with pytest.raises(r.ReconstructionBlocked):
        e._evaluation_inputs(raw, headers, 'float32', torch)


def test_unknown_loader_still_conservative_and_cast_pages_never_freed():
    args = dict(loaded_bytes=1000, source_bytes=2000, workspace_bytes=70,
        runtime_reserve_bytes=0, retained_cast_source_bytes=2000,
        largest_cast_scratch_bytes=300, mapping_overhead_bytes=90, buffer_reserve_bytes=20)
    explicit = d.inference_host_phases(**args, loader_strategy=d.EXPLICIT_NATIVE_LOADER)
    unknown = d.inference_host_phases(**args, loader_strategy='native_sounding_unknown')
    assert explicit['loader_bytes'] == 3410
    assert explicit['forward_bytes'] == 3180
    assert unknown['loader_bytes'] == 3110
    assert unknown['loader_strategy'] == 'unknown_conservative_full_shard'
    same = dict(args, source_bytes=1000, retained_cast_source_bytes=0, largest_cast_scratch_bytes=0)
    assert d.inference_host_phases(**same)['loader_bytes'] == 2110
    assert d.inference_host_phases(**same, loader_strategy=d.EXPLICIT_NATIVE_LOADER)['loader_bytes'] == 1110


def test_runtime_rechecks_available_after_meta_before_any_weights(tmp_path, monkeypatch):
    torch, _, path = tiny_source(tmp_path, 'qwen2', 'bfloat16')
    calls = []
    def available():
        calls.append(True)
        return 2 * 1024**3 if len(calls) == 1 else 32 * 1024**2
    monkeypatch.setattr(r, '_available_ram', available)
    monkeypatch.setattr(r, '_load_source', lambda *a, **kw: pytest.fail('read weights after failed admission'))
    with pytest.raises(r.ReconstructionBlocked, match='memory admission refused'):
        e.NativeGenerator(path, dtype='bfloat16', max_new_tokens=2, max_input_tokens=3)
    assert len(calls) >= 2


@pytest.mark.parametrize('kind', ['qwen2', 't5', 'switch_transformers', 'llama'])
@pytest.mark.parametrize('stored,requested', [('float32','float32'), ('float32','bfloat16'), ('bfloat16','float32')])
def test_actual_tensor_assignment_has_no_same_dtype_clone(tmp_path, monkeypatch, kind, stored, requested):
    torch, _, path = tiny_source(tmp_path, kind, stored)
    before, raw, headers = e._evaluation_preflight(path)
    meta, plan = e._evaluation_inputs(raw, headers, requested, torch)
    import safetensors
    original_open = safetensors.safe_open
    reads, pointers = [], {}
    class ObservedFile:
        def __init__(self, *args, **kwargs):
            self.handle = original_open(*args, **kwargs)
        def __enter__(self):
            self.handle.__enter__()
            return self
        def __exit__(self, *args):
            return self.handle.__exit__(*args)
        def get_tensor(self, key):
            value = self.handle.get_tensor(key)
            reads.append(key)
            pointers[key] = (value.data_ptr(), value.dtype)
            return value
    monkeypatch.setattr(safetensors, 'safe_open', ObservedFile)
    from asea.specialist.recovery import _load_recovery_model
    model = _load_recovery_model(meta, path, headers, plan, torch)
    e._native_router_precision(model, headers)
    assert sorted(reads) == sorted(headers)  # exactly once, including audited routers
    for alias, key in plan['aliases'].items():
        pointer, source_dtype = pointers[key]
        value = model.get_parameter(alias)
        if value.dtype == source_dtype:
            assert value.data_ptr() == pointer, alias
        else:
            assert value.data_ptr() != pointer, alias


def test_only_verified_explicit_loader_can_pass_reduced_memory_bound(monkeypatch):
    # Arithmetic-only host observations, not a resource-availability claim.
    from test_specialist_workflow import inference_config
    headers = {'weight': {'numel': 400_000_000, 'dtype': 'BF16', 'file': 'model.safetensors'}}
    inventory = {'model.safetensors': {'size': 800_001_024}}
    plan = {'strategy': d.EXPLICIT_NATIVE_LOADER, 'target_dtypes': {'weight': 'bfloat16'}}
    monkeypatch.setattr(r, '_available_ram', lambda: 1_750_000_000)
    monkeypatch.setattr(e, '_current_rss', lambda: 200_000_000)
    report = e._admit_loader(headers, inference_config(), 'bfloat16', load_plan=plan, inventory=inventory)
    assert report['admitted']
    for unknown in (None, {'strategy': 'native_unknown'}):
        with pytest.raises(r.ReconstructionBlocked, match='memory admission refused'):
            e._admit_loader(headers, inference_config(), 'bfloat16', load_plan=unknown, inventory=inventory)
    with pytest.raises(r.ReconstructionBlocked, match='memory admission refused'):
        e._admit_loader(headers, inference_config(), 'bfloat16', memory_budget_bytes=1_000_000_000,
                       load_plan=plan, inventory=inventory)
