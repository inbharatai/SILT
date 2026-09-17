"""Exact native shape adapters without importing Torch/Transformers or tensor data."""
from __future__ import annotations

import math
from pathlib import Path

from asea.artifacts import digest, safe_path, model_inventory
from asea.specialist.reconstruction import _headers, _json, _validate_config, _release_file_cache

WIDTHS = {'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2}
TOKENIZER_NAMES = {'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                   'vocab.json', 'merges.txt', 'added_tokens.json', 'spiece.model', 'tokenizer.model'}


def _positive(config, key, default=None):
    n = config.get(key, default)
    if type(n) is not int or n <= 0:
        raise ValueError('missing/invalid native dimension: ' + key)
    return n


def _qwen_shapes(config, retention):
    h, width, layers, vocab = (_positive(config, k) for k in (
        'hidden_size', 'intermediate_size', 'num_hidden_layers', 'vocab_size'))
    heads = _positive(config, 'num_attention_heads')
    kv = _positive(config, 'num_key_value_heads', heads)
    if h % heads or heads % kv:
        raise ValueError('Qwen attention dimensions are not divisible')
    if config.get('head_dim', h // heads) != h // heads or config.get('hidden_act', 'silu') != 'silu':
        raise ValueError('unsupported Qwen head_dim/activation')
    if config.get('attention_bias', True) is not True or config.get('mlp_bias', False) is not False:
        raise ValueError('unsupported Qwen bias policy')
    if type(retention) not in (float, int) or not math.isfinite(retention) or not 0 < retention < 1:
        raise ValueError('retention must be finite and strictly between zero and one')
    kept = math.floor(width * retention / 64) * 64
    if kept < 64 or kept >= width:
        raise ValueError('retention must produce a smaller positive multiple of 64')
    d = h // heads
    source = {'model.embed_tokens.weight': [vocab, h], 'model.norm.weight': [h]}
    aliases = []
    tied = config.get('tie_word_embeddings', False)
    if type(tied) is not bool:
        raise ValueError('invalid embedding tie policy')
    if tied:
        aliases.append(['model.embed_tokens.weight', 'lm_head.weight'])
    else:
        source['lm_head.weight'] = [vocab, h]
    for i in range(layers):
        prefix = 'model.layers.%d.' % i
        for name, shape in {'self_attn.q_proj.weight': [h, h], 'self_attn.q_proj.bias': [h],
                'self_attn.k_proj.weight': [kv*d, h], 'self_attn.k_proj.bias': [kv*d],
                'self_attn.v_proj.weight': [kv*d, h], 'self_attn.v_proj.bias': [kv*d],
                'self_attn.o_proj.weight': [h, h], 'input_layernorm.weight': [h],
                'post_attention_layernorm.weight': [h], 'mlp.gate_proj.weight': [width, h],
                'mlp.up_proj.weight': [width, h], 'mlp.down_proj.weight': [h, width]}.items():
            source[prefix + name] = shape
    target = {k: list(s) for k, s in source.items()}
    for name in target:
        if name.endswith(('.mlp.gate_proj.weight', '.mlp.up_proj.weight')):
            target[name][0] = kept
        elif name.endswith('.mlp.down_proj.weight'):
            target[name][1] = kept
    return source, target, aliases, {'source_intermediate_size': width, 'output_intermediate_size': kept,
        'actual_channel_retention': kept/width, 'operation': 'gate/up rows and down columns only; attention/embedding/norms unchanged',
        'removed_parameters': 3 * layers * h * (width-kept),
        'formula': 'source_unique_parameters - 3*layers*hidden*(intermediate-floor(intermediate*retention/64)*64)'}


def _switch_shapes(config):
    h, width, layers, vocab, heads, d, experts = (_positive(config, k) for k in (
        'd_model', 'd_ff', 'num_layers', 'vocab_size', 'num_heads', 'd_kv', 'num_experts'))
    decoder = _positive(config, 'num_decoder_layers', layers) if config.get('num_decoder_layers') is not None else layers
    if experts < 2 or config.get('dense_act_fn', 'relu') != 'relu' or config.get('tie_encoder_decoder', False):
        raise ValueError('unsupported Switch experts/activation/encoder-decoder ties')
    if config.get('router_dtype', 'float32') not in ('float32', 'bfloat16'):
        raise ValueError('unsupported router dtype')
    buckets = _positive(config, 'relative_attention_num_buckets', 32)
    source = {'shared.weight': [vocab, h]}
    aliases = [['shared.weight', 'encoder.embed_tokens.weight', 'decoder.embed_tokens.weight']]
    if config.get('tie_word_embeddings', True):
        aliases[0].append('lm_head.weight')
    else:
        source['lm_head.weight'] = [vocab, h]
    sparse_count = 0
    for stack, count in [('encoder', layers), ('decoder', decoder)]:
        sparse_layers = config.get('num_sparse_' + stack + '_layers', 3)
        if type(sparse_layers) is not int or not 0 <= sparse_layers <= count:
            raise ValueError('invalid sparse layer count')
        step = count // sparse_layers if sparse_layers else count
        for i in range(count):
            p = '%s.block.%d.layer.' % (stack, i)
            for j, attention in [(0, 'SelfAttention')] + ([(1, 'EncDecAttention')] if stack == 'decoder' else []):
                ap = p + str(j) + '.' + attention + '.'
                for projection in ('q', 'k', 'v'):
                    source[ap + projection + '.weight'] = [heads*d, h]
                source[ap + 'o.weight'] = [h, heads*d]
                source[p + str(j) + '.layer_norm.weight'] = [h]
                if i == 0 and attention == 'SelfAttention':
                    source[ap + 'relative_attention_bias.weight'] = [buckets, heads]
            j = 2 if stack == 'decoder' else 1
            fp = p + str(j) + '.mlp.'
            source[p + str(j) + '.layer_norm.weight'] = [h]
            sparse = (i % step == 1 or step == 1) if step > 0 else False
            if sparse:
                sparse_count += 1
                source[fp + 'router.classifier.weight'] = [experts, h]
                if config.get('router_bias', False):
                    source[fp + 'router.classifier.bias'] = [experts]
                for e in range(experts):
                    source[fp + 'experts.expert_%d.wi.weight' % e] = [width, h]
                    source[fp + 'experts.expert_%d.wo.weight' % e] = [h, width]
            else:
                source[fp + 'wi.weight'] = [width, h]
                source[fp + 'wo.weight'] = [h, width]
        source[stack + '.final_layer_norm.weight'] = [h]
    if not sparse_count:
        raise ValueError('Switch source has no sparse layers to reduce')
    target = {}
    for key, shape in source.items():
        if '.router.classifier.' in key:
            continue
        if '.experts.' in key:
            if '.experts.expert_0.' not in key:
                continue
            key = key.replace('.experts.expert_0.', '.')
        target[key.replace('.mlp.', '.DenseReluDense.')] = list(shape)
    return source, target, aliases, {'experts_per_sparse_layer': 1, 'sparse_layer_count': sparse_count,
        'source_experts_per_sparse_layer': experts, 'actual_expert_retention': 1/experts,
        'retention_argument': 'not_applied: native fixed top1 family; no quality knob changed',
        'operation': 'one equal-shaped expert per sparse layer; remove routers; map native dense T5'}


def _match_headers(expected, aliases, headers):
    canonical = dict(headers)
    for group in aliases:
        found = [k for k in group if k in canonical]
        if len(found) != 1:
            raise ValueError('missing or redundant tied alias storage: ' + ','.join(group))
        entry = canonical.pop(found[0])
        canonical[group[0]] = entry
    if set(canonical) != set(expected):
        raise ValueError('native header key mismatch: missing=%d unexpected=%d' % (
            len(set(expected)-set(canonical)), len(set(canonical)-set(expected))))
    for name, shape in expected.items():
        if canonical[name]['shape'] != shape:
            raise ValueError('native header shape mismatch: ' + name)
    return canonical


def _target_dtype(name, family, requested, source_dtype=None, config=None):
    if '.router.classifier.' in name:
        if source_dtype not in (None,'F32','BF16'):
            raise ValueError('unsupported native Switch router source dtype')
        target = (config or {}).get('router_dtype', 'float32')
        if source_dtype == 'F32' and target != 'float32':
            raise ValueError('router dtype would round original F32 router values')
        return 'F32' if target == 'float32' else 'BF16'
    if family == 't5' and 'wo' in name.split('.'):
        return 'F32'
    return 'F32' if requested == 'float32' else 'BF16'


def shape_metadata(config, retention=0.75, dtype='bfloat16', rank=8, headers=None):
    """Pure shape math; headers=None is explicitly UNBOUND and cannot yield READY."""
    if dtype not in ('float32', 'bfloat16'):
        raise ValueError('unsupported compute dtype; no implicit recast')
    if type(rank) is not int or not 1 <= rank <= 128:
        raise ValueError('unsupported rank')
    family = config.get('model_type')
    if family == 'qwen2':
        source, target, aliases, reduction = _qwen_shapes(config, retention)
        targets = ('q_proj', 'v_proj', 'down_proj')
    elif family == 'switch_transformers':
        source, target, aliases, reduction = _switch_shapes(config)
        targets = ('q', 'v', 'wo')
    else:
        raise ValueError('unsupported model_type; no model-name size guessing')
    canonical = _match_headers(source, aliases, headers) if headers is not None else None
    raw = cast_pages = scratch = source_loaded = f32 = largest = 0
    dtype_totals, exceptions = {}, {}
    for name, shape in source.items():
        n = math.prod(shape)
        source_dtype = canonical[name]['dtype'] if canonical is not None else ('F32' if dtype == 'float32' else 'BF16')
        native = _target_dtype(name, family, dtype, source_dtype, config)
        b, loaded = n * WIDTHS[source_dtype], n * WIDTHS[native]
        raw += b
        source_loaded += loaded
        largest = max(largest, loaded)
        dtype_totals[source_dtype] = dtype_totals.get(source_dtype, 0) + n
        if native != ('F32' if dtype == 'float32' else 'BF16'):
            exceptions[name] = {'source_dtype': source_dtype, 'loaded_dtype': native, 'parameters': n}
        if source_dtype != native:
            cast_pages += b
            scratch = max(scratch, b + loaded)
    base = adapters = largest_target = target_bytes = retained_f32 = 0
    reconstructed = student_cast_pages = student_cast_scratch = 0
    for name, shape in target.items():
        n = math.prod(shape)
        # Switch source does NOT retain wo in F32. Reconstruction reuses its
        # BF16 expert weights; the subsequent native T5 loader keeps wo in F32.
        stored = _target_dtype(name, family, dtype)
        native = _target_dtype(name, 't5' if family=='switch_transformers' else family, dtype)
        b = n * WIDTHS[native]
        original = n * WIDTHS[stored]
        reconstructed += original
        if stored!=native:
            student_cast_pages += original
            student_cast_scratch = max(student_cast_scratch,original+b)
        base += b
        if native == 'F32' and dtype != 'float32':
            retained_f32 += n
        if len(shape) == 2 and name.endswith('.weight') and name.split('.')[-2] in targets:
            adapters += rank * sum(shape)
            largest_target = max(largest_target, n)
            target_bytes += b
    return {'bound_to_headers': canonical is not None, 'model_type': family, 'dtype': dtype,
            'source_parameters': sum(math.prod(s) for s in source.values()),
            'base_parameters': sum(math.prod(s) for s in target.values()),
            'source_raw_tensor_bytes': raw, 'source_loaded_tensor_bytes': source_loaded,
            'base_tensor_bytes': base, 'reconstructed_tensor_bytes': reconstructed,
            'student_cast_source_pages_bytes': student_cast_pages, 'student_cast_scratch_bytes': student_cast_scratch,
            'factor_parameters': adapters, 'factor_tensor_bytes': adapters*4,
            'factor_optimizer_gradient_bytes': adapters*16, 'factor_snapshot_bytes': adapters*4,
            'source_cast_pages_bytes': cast_pages, 'source_cast_scratch_bytes': scratch,
            'source_load_peak_lower_bound_bytes': source_loaded + cast_pages + scratch,
            'source_largest_tensor_bytes': largest, 'base_largest_tensor_parameters': max(math.prod(s) for s in target.values()),
            'largest_lora_target_parameters': largest_target, 'lora_target_tensor_bytes': target_bytes,
            'retained_f32_parameters': retained_f32, 'dtype_exceptions': exceptions, 'raw_dtype_parameter_totals': dtype_totals,
            'source_tensor_count': len(source), 'base_tensor_count': len(target), 'reduction': reduction}


def inspect_source(source_path, retention, dtype, rank):
    root = safe_path(source_path)
    config = _json(root / 'config.json')
    # Structural checks must not consult this process's ML installation: the
    # selected interpreter can differ. Saved/runtime versions are checked by the
    # planner against its probe, then by the native runtime again before load.
    _validate_config({k:v for k,v in config.items() if k!='transformers_version'})
    weight_paths = list(root.glob('*.safetensors'))
    bound = bool(weight_paths)
    if bound:
        inventory = model_inventory(root)
        tokenizer_present = ('tokenizer.json' in inventory or 'spiece.model' in inventory
                             or 'tokenizer.model' in inventory
                             or {'vocab.json','merges.txt'} <= set(inventory))
        if not tokenizer_present:
            raise ValueError('source tokenizer assets missing; no native load/tokenization attempted')
        for name in inventory:
            if name.endswith('.json'):
                item = _json(root/name,64*1024**2 if name=='tokenizer.json' else 16*1024**2)
                if isinstance(item,dict) and any(item.get(k) is not None for k in ('auto_map','base_model_name_or_path','peft_type')):
                    raise ValueError('unsafe custom code or base-model indirection: '+name)
        headers = _headers(root,inventory)
        if config != _json(root/'config.json'):
            raise ValueError('config changed during inventory')
        for name in inventory:
            if name.endswith('.safetensors'):
                _release_file_cache(root/name)
        counts = shape_metadata(config, retention, dtype, rank, headers)
    else:
        from asea.artifacts import file_hash
        inventory = {'config.json': file_hash(root / 'config.json')}
        counts = shape_metadata(config, retention, dtype, rank)
    # Do not copy arbitrary model config strings, credentials, acquisition URLs,
    # hostname, source path or dataset contents into public model identity.
    identity = {'model_type': config['model_type'], 'config_sha256': inventory['config.json']['sha256'],
                'files': inventory, 'inventory_sha256': digest(inventory),
                'checkpoint_bound': bound, 'counts': counts,
                'tensor_file_bytes': sum(v['size'] for k, v in inventory.items() if k.endswith('.safetensors')),
                'tokenizer_asset_bytes': sum(v['size'] for k, v in inventory.items() if k in TOKENIZER_NAMES)}
    return identity, config
