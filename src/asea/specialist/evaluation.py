"""Native local inference and scoped function-IO observations; no admission certificate.

Deliberately does not use Compose's prompt renderer: recovery already supervises
one format instruction in the original user text. Optional ML imports are lazy.
"""
from __future__ import annotations

import ast
import dataclasses
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

from asea.artifacts import file_hash, model_inventory, safe_file, safe_path
from .recovery import _bounded_json, _encoding_contract

MAX_NEW_TOKENS = 384


def representation_inventory(path):
    """Hash the entire deployable representation, never execute artifact runners.

    Bundle metadata/counts are usable only after strict inspect_bundle validation.
    Include the manifest and optional audit report in the same freeze map as all
    base, tokenizer, runner and factor files (the manifest cannot hash itself).
    """
    root = safe_path(path)
    marker = root / "specialist_bundle.json"
    if marker.exists() or marker.is_symlink():
        from .standalone import inspect_bundle
        manifest = inspect_bundle(root)
        names = set(manifest["files"]) | {"specialist_bundle.json"}
        if (root / "recovery_report.json").exists():
            names.add("recovery_report.json")
        return {name: file_hash(safe_file(root / name)) for name in sorted(names)}
    return model_inventory(root)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def load_suite(path):
    from asea.compose.schema import EvaluationSuite
    suite = EvaluationSuite.model_validate(_bounded_json(path if isinstance(path, bytes) else safe_file(path)))
    if any(c.metric != "function_io" or c.input is None or c.input_file is not None for c in suite.cases):
        raise ValueError("FunctionalSuite accepts existing EvaluationSuite text function_io cases only")
    if suite.claims != ["coding"]:
        raise ValueError("FunctionalSuite requires claims=['coding']; claims cover exercised short functions only")
    return suite


def _current_rss():
    """Current resident bytes, not the process-lifetime high-water mark."""
    from .reconstruction import ReconstructionBlocked
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                value = int(line.split()[1]) * 1024
                if value > 0:
                    return value
    except (OSError, ValueError, IndexError) as exc:
        raise ReconstructionBlocked("Cannot establish current process RSS") from exc
    raise ReconstructionBlocked("Cannot establish current process RSS (Linux /proc required)")


def _validate_llama_config(config):
    """Evaluation-only native Llama guard; never widens reconstruction/recovery."""
    from .reconstruction import ReconstructionBlocked
    if not isinstance(config, dict) or config.get("model_type") != "llama":
        raise ReconstructionBlocked("evaluation requires native llama model_type")
    if config.get("architectures") != ["LlamaForCausalLM"]:
        raise ReconstructionBlocked("evaluation requires strict native LlamaForCausalLM architecture")
    for key in ("vocab_size", "hidden_size", "intermediate_size", "num_attention_heads",
                "num_key_value_heads", "max_position_embeddings"):
        if type(config.get(key)) is not int or not 0 < config[key] <= 2 ** 20:
            raise ReconstructionBlocked("invalid or excessive config dimension: " + key)
    if type(config.get("num_hidden_layers")) is not int or not 0 < config["num_hidden_layers"] <= 256:
        raise ReconstructionBlocked("invalid or excessive layer count: num_hidden_layers")
    if any(config.get(k) is not None for k in ("auto_map", "quantization_config", "compression_config",
            "base_model_name_or_path", "peft_type")) or config.get("pruned_heads"):
        raise ReconstructionBlocked("custom, quantized, PEFT, or pruned configuration unsupported")
    if config.get("is_encoder_decoder", False) is not False or config.get("num_experts", 1) != 1:
        raise ReconstructionBlocked("evaluation supports dense causal Llama only")
    if type(config.get("pretraining_tp", 1)) is not int or config.get("pretraining_tp", 1) != 1:
        raise ReconstructionBlocked("evaluation requires native untensorized Llama")
    for key in ("tie_word_embeddings", "attention_bias", "mlp_bias"):
        if key in config and type(config[key]) is not bool:
            raise ReconstructionBlocked("invalid native Llama boolean: " + key)
    if config.get("transformers_version") is not None:
        from importlib.metadata import version
        from packaging.version import Version, InvalidVersion
        try:
            saved, installed = Version(config["transformers_version"]), Version(version("transformers"))
            if saved > installed or saved.major != installed.major:
                raise ReconstructionBlocked("unsupported config transformers_version")
        except (InvalidVersion, TypeError) as exc:
            raise ReconstructionBlocked("invalid config transformers_version") from exc


def _evaluation_preflight(path):
    """Safe local Llama admission, with original preflight unchanged for other kinds.

    No exception fallback: unsupported/malformed configs still fail closed. All
    JSON and safetensors headers are checked before native tokenizer/model loads.
    """
    from .reconstruction import (_artifact_preflight, _json, _headers,
                                 _release_file_cache, ReconstructionBlocked)
    root = safe_path(path)
    config = _json(root / "config.json")
    if not isinstance(config, dict) or config.get("model_type") != "llama":
        return _artifact_preflight(root)
    inventory = model_inventory(root)
    for name in inventory:
        if Path(name).name == "adapter_config.json":
            raise ReconstructionBlocked("PEFT/base-model indirection unsupported")
        if name.endswith(".json"):
            value = _json(root / name, 64 * 1024**2 if name == "tokenizer.json" else 16 * 1024**2)
            if isinstance(value, dict) and any(value.get(k) is not None for k in
                    ("auto_map", "base_model_name_or_path", "peft_type")):
                raise ReconstructionBlocked("unsafe auto_map or base-model indirection: " + name)
    _validate_llama_config(config)
    headers = _headers(root, inventory)
    for name in inventory:
        if name.endswith(".safetensors"):
            _release_file_cache(root / name)
    return inventory, config, headers


def _native_llama_meta(config, headers, torch):
    """Strict native evaluation constructor; shared key/shape/tied-alias proof."""
    from .reconstruction import _strict_header_match
    _validate_llama_config(config)
    from transformers import LlamaConfig, LlamaForCausalLM
    native = LlamaConfig.from_dict(config)
    native._attn_implementation = "eager"
    with torch.device("meta"):
        meta = LlamaForCausalLM(native)
    _strict_header_match(meta, headers)
    return native, LlamaForCausalLM, meta


def _evaluation_inputs(raw, headers, dtype, torch):
    """Strict native class/key/shape/tie proof, then parameter-meta/CPU buffers.

    Evaluation-only Llama does not broaden the reconstruction/recovery family
    gate. No checkpoint tensor is touched until all admission checks pass.
    """
    from accelerate import init_empty_weights
    from .reconstruction import _native_meta, _load_plan, ReconstructionBlocked
    from .devices import NATIVE_BUFFER_RESERVE
    _runtime_dimensions(raw)
    constructor = _native_llama_meta if raw.get("model_type") == "llama" else _native_meta
    config, cls, checked = constructor(raw, headers, torch)
    if sum(b.numel() * b.element_size() for b in checked.buffers()) > NATIVE_BUFFER_RESERVE:
        raise ReconstructionBlocked("native constructor buffers exceed bounded 16MiB")
    del checked
    config.torch_dtype = getattr(torch, dtype)
    with init_empty_weights(include_buffers=False):
        meta = cls(config)
    meta.tie_weights()
    if sum(b.numel() * b.element_size() for b in meta.buffers()) > NATIVE_BUFFER_RESERVE:
        raise ReconstructionBlocked("native constructor buffers exceed bounded 16MiB")
    return meta, _load_plan(meta, headers, dtype)


def _native_router_precision(model, headers):
    """Audit already assigned original F32 routers; never reopen a second mmap.

    _load_plan rejects lossy router casts before weights. _load_source loads the
    exact source F32 storage directly, so HF's old restoration pass is redundant.
    Preserve its bounded digest report without a second simultaneous mapping.
    """
    from .reconstruction import ReconstructionBlocked
    routers = {k: v for k, v in headers.items() if k.endswith("router.classifier.weight")}
    total = sum(v["numel"] * 4 for v in routers.values() if v["dtype"] == "F32")
    if total > 64 * 1024**2:
        raise ReconstructionBlocked("source F32 router restoration exceeds bounded 64MiB")
    restored = []
    for key, entry in sorted(routers.items()):
        if entry["dtype"] != "F32":
            continue
        if entry["numel"] * 4 > 16 * 1024**2:
            raise ReconstructionBlocked("source router tensor exceeds bounded 16MiB")
        parameter = model.get_parameter(key)
        if str(parameter.dtype) != "torch.float32" or list(parameter.shape) != entry["shape"] or not parameter.is_contiguous():
            raise ReconstructionBlocked("invalid canonical source router precision")
        digest = hashlib.sha256(memoryview(parameter.detach().numpy())).hexdigest()
        restored.append(dict(key=key, source_file=entry["file"], dtype="F32",
                             tensor_sha256=digest, bytes=entry["numel"] * 4))
    return {"policy": "canonical original F32 router storage directly from source; audited before forward, no rounded upcast or second mmap",
            "restored": restored, "source_non_f32_routers": {k: v["dtype"] for k, v in routers.items() if v["dtype"] != "F32"}}


def _admit_native_import(memory_budget_bytes):
    # No weights allocated in this phase. A full checkpoint estimate belongs
    # AFTER native class/dtype/alias proof, still strictly before weight reads.
    from .devices import NATIVE_BUFFER_RESERVE
    return _runtime_admission("loader_pre_import", 512 * 1024**2 + NATIVE_BUFFER_RESERVE,
        dict(model_weights_materialized=False, buffer_reserve_bytes=NATIVE_BUFFER_RESERVE,
             formula="512MiB incremental import/meta reserve + bounded native CPU buffers; no weights"),
        memory_budget_bytes)


def _runtime_dimensions(config):
    """No guessed dimensions/default architecture in an allocation formula."""
    from .reconstruction import ReconstructionBlocked, _validate_config
    if isinstance(config, dict) and config.get("model_type") == "llama":
        _validate_llama_config(config)
    else:
        _validate_config(config)
    def dim(key):
        n = config.get(key)
        if type(n) is not int or n <= 0:
            raise ReconstructionBlocked("unknown inference config dimension: " + key)
        return n
    causal = config["model_type"] in ("qwen2", "llama")
    h, f, heads = (dim("hidden_size"), dim("intermediate_size"), dim("num_attention_heads")) if causal else (
        dim("d_model"), dim("d_ff"), dim("num_heads"))
    if causal:
        kv, layers = dim("num_key_value_heads"), dim("num_hidden_layers")
        if h % heads or heads % kv or config.get("head_dim", h // heads) != h // heads:
            raise ReconstructionBlocked("unsupported inference head geometry")
        dhead, decoder_layers = h // heads, layers
    else:
        kv, dhead, layers = heads, dim("d_kv"), dim("num_layers")
        decoder_layers = dim("num_decoder_layers")
    experts = dim("num_experts") if config["model_type"] == "switch_transformers" else 1
    return dict(causal=causal, hidden=h, ffn=f, heads=heads, kv_heads=kv,
        head_dim=dhead, layers=layers, decoder_layers=decoder_layers,
        vocab=dim("vocab_size"), experts=experts)


def _runtime_admission(phase, estimate, fields, memory_budget_bytes):
    from .reconstruction import _memory_budget, ReconstructionBlocked
    # _memory_budget observes host/cgroup remaining *now*. Its operator argument
    # is intentionally not used: inference's operator cap is TOTAL process RSS.
    if memory_budget_bytes is not None and (type(memory_budget_bytes) is not int or memory_budget_bytes <= 0):
        raise ReconstructionBlocked("memory_budget_bytes must be a positive integer or None")
    rss = _current_rss()
    budget = _memory_budget()
    operator_remaining = None if memory_budget_bytes is None else max(0, memory_budget_bytes - rss)
    limit = budget["limit_bytes"] if operator_remaining is None else min(budget["limit_bytes"], operator_remaining)
    report = dict(budget, **fields, phase=phase, current_rss_bytes=rss,
        requested_memory_budget_bytes=memory_budget_bytes, operator_remaining_bytes=operator_remaining,
        limit_bytes=limit, estimated_additional_peak_bytes=estimate,
        estimated_total_process_peak_bytes=rss + estimate,
        budget_policy="min(observed host/cgroup remaining minus max(256MiB,10%), optional TOTAL process cap minus current RSS)",
        admitted=estimate <= limit)
    if estimate > limit:
        exc = ReconstructionBlocked("memory admission refused at %s: additional estimated %d bytes > effective %d bytes" % (phase, estimate, limit))
        exc.memory_admission = report
        raise exc
    return report


def _admit_loader(headers, config, dtype, memory_budget_bytes=None, *, phase="loader_pre_import", native_loaded_bytes=0,
                  load_plan=None, inventory=None, extra_loader_bytes=0):
    from .reconstruction import _DTYPES, _SIZES, ReconstructionBlocked
    dimensions = _runtime_dimensions(config)
    if dtype not in _DTYPES or not headers or any(v.get("dtype") not in _SIZES for v in headers.values()):
        raise ReconstructionBlocked("unknown inference weight dtype/header")
    from .devices import (inference_loaded_bytes, inference_host_phases,
                          explicit_loader_storage, EXPLICIT_NATIVE_LOADER)
    if load_plan is not None and load_plan.get("strategy") == EXPLICIT_NATIVE_LOADER:
        storage = explicit_loader_storage(headers, load_plan, inventory)
        phases = inference_host_phases(**storage, workspace_bytes=0, extra_loader_bytes=extra_loader_bytes)
        return _runtime_admission(phase, phases["loader_bytes"], dict(
            dimensions=dimensions, header_loaded_bytes=storage["loaded_bytes"],
            source_payload_bytes=storage["source_bytes"], loader_storage=storage,
            extra_loader_bytes=extra_loader_bytes,
            source_mmap_and_cast_transient_bytes=storage["retained_cast_source_bytes"] + storage["largest_cast_scratch_bytes"],
            loader_runtime_reserve_bytes=512 * 1024**2,
            retained_cast_source_bytes=storage["retained_cast_source_bytes"],
            unknown_inference_workspace_charged_bytes=0,
            formula="explicit per-tensor: loaded + retained cast source pages + largest cast scratch + mapping + bounded buffers + 512MiB"), memory_budget_bytes)
    loaded = inference_loaded_bytes(headers, dtype, native_loaded_bytes)
    source = sum(v["numel"] * _SIZES[v["dtype"]] for v in headers.values())
    transient, runtime = max(loaded, source), 512 * 1024**2
    retained_cast_source = sum(v['numel'] * _SIZES[v['dtype']] for k,v in headers.items()
        if inference_loaded_bytes({k:v}, dtype) != v['numel'] * _SIZES[v['dtype']])
    phases = inference_host_phases(loaded, source, 0)
    return _runtime_admission(phase, phases["loader_bytes"], dict(
        dimensions=dimensions, header_loaded_bytes=loaded, source_payload_bytes=source,
        source_mmap_and_cast_transient_bytes=transient, loader_runtime_reserve_bytes=runtime,
        retained_cast_source_bytes=retained_cast_source,
        unknown_inference_workspace_charged_bytes=0,
        formula="loaded + max(loaded, source_payload) + 512MiB; no fictitious prompt"), memory_budget_bytes)


def _unresident_weight_mmaps(root, headers):
    """Charge pages of retained checkpoint mappings not yet present in RSS.

    Casting can leave file-backed storage alive; neither mmap length nor fully
    resident weight bytes may simply be assumed zero. Private/COW pages count in
    Rss too. A vanished mapping needs no future fault allowance.
    """
    from .reconstruction import ReconstructionBlocked
    paths = {str(root / v["file"]) for v in headers.values()}
    missing = mappings = 0
    size = resident = None
    selected = False
    try:
        for line in Path("/proc/self/smaps").read_text().splitlines() + ["0-0 sentinel"]:
            parts = line.split(None, 5)
            if parts and "-" in parts[0]:
                if selected:
                    if size is None or resident is None or not 0 <= resident <= size:
                        raise ValueError("missing or inconsistent mapping residency")
                    missing += size - resident
                mappings += len(parts) >= 5
                selected = len(parts) == 6 and parts[5].removesuffix(" (deleted)") in paths
                size = resident = None
            elif selected and line.startswith("Size:"):
                size = int(parts[1]) * 1024
            elif selected and line.startswith("Rss:"):
                resident = int(parts[1]) * 1024
        if not mappings:
            raise ValueError("empty mapping evidence")
    except (OSError, ValueError, IndexError) as exc:
        raise ReconstructionBlocked("Cannot establish checkpoint mmap residency") from exc
    return missing


def _admit_inference(config, dtype, prefill_tokens, max_new_tokens, memory_budget_bytes=None, *, unresident_weight_mmap_bytes=0, factor_rank=0, _estimate_only=False):
    """Batch-one, eager CPU, no autograd, greedy cached native generation only.

    Working tensors are per-layer, not summed over layers; retained KV is summed
    over layers. Conservatively overlap the last prefill logits with decoding.
    Detailed coefficients and assumptions are documented in SPECIALIST_WORKFLOW.
    """
    from .reconstruction import _DTYPES, ReconstructionBlocked
    d = _runtime_dimensions(config)
    if dtype not in _DTYPES or any(type(n) is not int or n <= 0 for n in (prefill_tokens, max_new_tokens)):
        raise ReconstructionBlocked("unknown inference dtype or sequence length")
    if type(unresident_weight_mmap_bytes) is not int or unresident_weight_mmap_bytes < 0:
        raise ReconstructionBlocked("invalid checkpoint mmap residency")
    b, p, g = _DTYPES[dtype], prefill_tokens, max_new_tokens
    h, f, a, k, hd, v = (d[key] for key in ("hidden", "ffn", "heads", "kv_heads", "head_dim", "vocab"))
    t = p + g if d["causal"] else g + 1
    # Full allocated cache bound plus one layer's old/new concatenation overlap.
    cache_sequence = t if d["causal"] else t + p
    kv = 2 * d["decoder_layers"] * k * hd * cache_sequence * b
    concat = 2 * k * hd * t * b
    encoder = 0 if d["causal"] else 2 * p * h * b
    # 16 FP32 hidden-width and 8 FP32 FFN-width temporaries cover eager
    # projections, gated MLP, residuals, RMSNorm/native float32 work, and casts.
    def layer(q, s, cross=False):
        projections = q * (16 * h + 8 * f + 4 * a * hd) * 4
        attention = a * q * s * (2 * b + 8)  # scores, FP32 softmax, cast, bias
        mask = q * s * 8
        router = q * d["experts"] * 16
        return projections + attention * (2 if cross else 1) + mask + router
    prefill_logits = p * v * (b + 4) if d["causal"] else 0
    # Native full-prefill logits (do not assume logits_to_keep optimization),
    # FP32 output cast and two simultaneously live next-token score vectors.
    decode_logits = v * (b + 4)
    scores = v * 16
    prefill_workspace = layer(p, p)
    decode_workspace = layer(1, t + (0 if d["causal"] else p), cross=not d["causal"])
    stages = {"prefill_forward": prefill_workspace + prefill_logits + scores,
        "prefill_decode_overlap": prefill_logits + decode_workspace + decode_logits + scores}
    stage_peak = max(stages.values())
    tokens = 64 * (p + g + 1)  # growing IDs/masks, cache positions and Python traces
    reserve = 64 * 1024**2  # allocator/library workspaces; separate from host safety reserve
    if type(factor_rank) is not int or not 0 <= factor_rank <= 128:
        raise ReconstructionBlocked("invalid validated factor rank")
    factor_workspace = 4 * p * (2 * h + 2 * f + factor_rank) if factor_rank else 0
    estimate = factor_workspace + unresident_weight_mmap_bytes + kv + concat + encoder + tokens + reserve + stage_peak
    fields = dict(
        dimensions=d, dtype_bytes=b, factor_workspace_bytes=factor_workspace, actual_prefill_tokens=p, max_new_tokens=g, cache_tokens=t,
        kv_cache_bytes=kv, cache_concat_overlap_bytes=concat, retained_encoder_bytes=encoder,
        prefill_activation_bytes=prefill_workspace, native_prefill_logits_bytes=prefill_logits,
        decode_activation_bytes=decode_workspace, native_decode_logits_bytes=decode_logits,
        bounded_score_bytes=scores, retained_generation_score_bytes=0,
        stage_additional_bytes=stages, stage_peak_bytes=stage_peak, token_and_mask_bytes=tokens,
        incremental_runtime_reserve_bytes=reserve, resident_weights_charged_bytes=0,
        unresident_weight_mmap_bytes=unresident_weight_mmap_bytes,
        formula="factor_workspace + unresident_mmaps + KV + concat_overlap + encoder + tokens + 64MiB + max(prefill_forward, prefill_decode_overlap)")
    if _estimate_only:
        return dict(fields, estimated_additional_peak_bytes=estimate)
    return _runtime_admission("inference_incremental", estimate, fields, memory_budget_bytes)


def _resources():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"peak_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024),
            "measurement": "process_lifetime_high_water_not_isolated_per_model",
            "allocator_release_guaranteed": False}


def render_prompt(tokenizer, family, prompt):
    """Single rendering contract for profiling AND live generation; no answers."""
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 65536:
        raise ValueError("prompt must be nonblank and <=65536 characters")
    if family == "seq2seq":
        return prompt, True
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True), False
    return prompt + "\n", False


def profile_prompts(tokenizer, family, config, prompts, max_new_tokens):
    """Profile all already-authorized inputs, never sample a suite or trim text."""
    lengths, hashes = [], []
    limits = [n for n in (config.get("max_position_embeddings"),
        getattr(tokenizer, "model_max_length", None)) if type(n) is int and 0 < n < 1000000]
    context = min(limits + [2048])
    for prompt in prompts:
        prefix, specials = render_prompt(tokenizer, family, prompt)
        ids = tokenizer.encode(prefix, add_special_tokens=specials, truncation=False)
        required = len(ids) + max_new_tokens if family == "causal" else max(len(ids), max_new_tokens + 1)
        if not ids or required > context:
            raise ValueError("complete input + generation budget exceeds native context limit; never truncate")
        if tokenizer.eos_token_id is None or (family == "seq2seq" and ids[-1] != tokenizer.eos_token_id):
            raise ValueError("native prompt EOS contract failed")
        lengths.append(len(ids))
        hashes.append(json_hash(ids))
    if not lengths:
        raise ValueError("cannot profile an empty input set")
    return {"max_input_tokens": max(lengths), "input_token_lengths": lengths,
        "input_ids_sha256": hashes, "all_inputs_profiled": True,
        "input_count": len(lengths), "max_new_tokens": max_new_tokens,
        "context_limit": context, "truncation": False}


def profile_local_inputs(model, prompts, max_new_tokens):
    """Safe local tokenizer-only pre-model profile (native or validated bundle)."""
    root = safe_path(model)
    marker = root / "specialist_bundle.json"
    if marker.exists() or marker.is_symlink():
        from .standalone import inspect_bundle
        inspect_bundle(root)
        root = root / "base"
    _, config, _ = _evaluation_preflight(root)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
    family = "causal" if config["model_type"] in ("qwen2", "llama") else "seq2seq"
    return profile_prompts(tokenizer, family, config, prompts, max_new_tokens)


def _generation_policy(model, tokenizer, family, max_new_tokens):
    """Resolve the validated native policy, never inherit checkpoint policy knobs.

    Transformers 4.51.3 merges global-default-valued fields back from the model
    unless use_model_defaults=False is passed to *generate*, not just its config.
    The version-scoped resolver below is also used by that native generate API.
    Its result is the configuration-resolution stage, before native input-length
    bookkeeping and device-local special-token tensors; it is not a live hook.
    """
    import inspect
    import transformers
    from transformers import GenerationConfig
    from transformers.generation.utils import GenerationMixin
    from .reconstruction import ReconstructionBlocked

    if transformers.__version__ != "4.51.3" or "use_model_defaults" not in inspect.signature(GenerationMixin.generate).parameters:
        raise ReconstructionBlocked("generation policy requires validated transformers 4.51.3 use_model_defaults API")
    if family not in ("causal", "seq2seq"):
        raise ReconstructionBlocked("unsupported native generation family")
    if type(model).__module__.startswith("peft."):
        import peft
        if peft.__version__ != "0.15.2" or model.active_peft_config.is_prompt_learning:
            raise ReconstructionBlocked("generation policy requires validated non-prompt PEFT 0.15.2 wrapper")
    saved = getattr(model, "generation_config", None)
    if not isinstance(saved, GenerationConfig) or not isinstance(saved.transformers_version, str):
        raise ReconstructionBlocked("missing native generation configuration/version metadata")
    from packaging.version import Version, InvalidVersion
    try:
        Version(saved.transformers_version)
    except InvalidVersion as exc:
        raise ReconstructionBlocked("invalid native generation configuration/version metadata") from exc

    # Only native special IDs survive from config.json. Saved generation special
    # IDs remain the native resolver's fallback when these are absent. All other
    # knobs start at library defaults, even if config.json itself has drifted.
    native = GenerationConfig.from_model_config(model.config)
    cfg = GenerationConfig()
    special_ids = ("bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id")
    for name in special_ids:
        setattr(cfg, name, getattr(native, name))
    cfg.do_sample, cfg.num_beams, cfg.num_return_sequences = False, 1, 1
    cfg.max_new_tokens, cfg.use_cache = max_new_tokens, True
    cfg.min_length, cfg.min_new_tokens = 0, 0
    cfg.forced_bos_token_id, cfg.forced_eos_token_id = None, None
    cfg.repetition_penalty, cfg.no_repeat_ngram_size = 1.0, 0
    cfg.return_dict_in_generate, cfg.output_scores = False, False
    cfg.output_logits, cfg.output_attentions, cfg.output_hidden_states = False, False, False
    cfg.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    requested = cfg.to_dict()
    resolved, unused = model._prepare_generation_config(cfg, use_model_defaults=False)
    effective = resolved.to_dict()
    if unused or any(effective.get(key) != value for key, value in requested.items() if key not in special_ids):
        raise ReconstructionBlocked("native generation policy drift during configuration resolution")
    for name in special_ids:
        value = getattr(resolved, name)
        expected = requested[name] if requested[name] is not None else getattr(saved, name)
        if value != expected:
            raise ReconstructionBlocked("native special-token policy drift during configuration resolution")
        values = value if name == "eos_token_id" and isinstance(value, list) else [value]
        if value is not None and (not values or any(type(v) is not int or v < 0 for v in values)):
            raise ReconstructionBlocked("invalid native special-token metadata: " + name)
    if resolved.eos_token_id is None or (family == "seq2seq" and resolved.decoder_start_token_id is None):
        raise ReconstructionBlocked("missing native EOS/decoder-start token metadata")
    mode = resolved.get_generation_mode()
    if mode.value != "greedy_search" or resolved.do_sample is not False or resolved.num_beams != 1:
        raise ReconstructionBlocked("native generation did not resolve to greedy search")
    return resolved, {"schema": "specialist_generation_policy_v1",
        "requested_generate_config": requested, "resolved_generate_config": effective,
        "use_model_defaults": False, "generation_mode": mode.value,
        "resolution_method": "transformers_4.51.3._prepare_generation_config",
        "resolution_stage": "before_native_input_length_and_special_token_tensor_preparation",
        "verification": "resolved_config_plus_explicit_generate_use_model_defaults_false",
        "live_generation_mode_observed": False,
        "special_token_policy": "model_config_then_saved_generation_config; tokenizer_pad_or_eos",
        "checkpoint_policy_defaults_disabled": True}


class NativeGenerator:
    """One frozen native model, explicit greedy configuration, exact recovery prefix."""
    def __init__(self, model, dtype="bfloat16", max_new_tokens=256, memory_budget_bytes=None,
                 execution_device="cpu", device_memory_budget_bytes=None, max_input_tokens=None):
        from .devices import validate_device_request
        validate_device_request(execution_device)
        if max_input_tokens is not None and (type(max_input_tokens) is not int or not 1 <= max_input_tokens <= 2048):
            raise ValueError("max_input_tokens must be in [1,2048] or None")
        self.max_input_tokens = max_input_tokens
        self.execution_device_requested = execution_device
        self.execution_device = "cpu"
        self.device_memory_budget_bytes = device_memory_budget_bytes
        if dtype not in ("bfloat16", "float32"):
            raise ValueError("dtype must be bfloat16 or float32")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
            raise ValueError("max_new_tokens must be in [1,384]")
        self.path = safe_path(model)
        from .reconstruction import _artifact_preflight
        self.base_path, self.manifest = self.path, None
        self.source_model_class = None
        self.representation = "native_merged"
        marker = self.path / "specialist_bundle.json"
        if marker.exists() or marker.is_symlink():
            from .standalone import inspect_bundle, load_standalone, _adapter_headers
            self.manifest = inspect_bundle(self.path)
            if dtype != self.manifest["dtype"]:
                raise ValueError("dtype must match bundle; no implicit recast")
            self.before = representation_inventory(self.path)
            self.base_path = self.path / "base"
            _, self.raw, base_headers = _artifact_preflight(self.base_path)
            self.headers = {k: dict(v, file="base/" + v["file"]) for k, v in base_headers.items()}
            self.headers.update({k: dict(v, file="adapter/adapter_model.safetensors") for k, v in
                _adapter_headers(self.path / "adapter/adapter_model.safetensors").items()})
            self.family = "causal" if self.manifest["model_type"] == "qwen2" else "seq2seq"
            self.dtype, self.max_new_tokens = dtype, max_new_tokens
            self.memory_budget_bytes = memory_budget_bytes
            self.pre_import_admission = _admit_native_import(memory_budget_bytes)
            import torch
            self.torch = torch
            meta, base_plan = _evaluation_inputs(self.raw, base_headers, dtype, torch)
            del meta
            plan = dict(base_plan, target_dtypes=dict(base_plan["target_dtypes"]))
            plan["target_dtypes"].update({k: "float32" for k in self.headers if k not in base_headers})
            self.load_plan = plan
            load_inventory = {v["file"]: {"size": safe_file(self.path / v["file"]).stat().st_size}
                              for v in self.headers.values()}
            extra = self.manifest["counts"]["factor_parameters"] * 8  # PEFT init + saved factors
            self.post_import_admission = _admit_loader(self.headers, self.raw, dtype, memory_budget_bytes,
                phase="loader_post_import", load_plan=plan, inventory=load_inventory, extra_loader_bytes=extra)
            self.admission = _admit_loader(self.headers, self.raw, dtype, memory_budget_bytes,
                phase="loader_pre_weights", load_plan=plan, inventory=load_inventory, extra_loader_bytes=extra)
            self._select_execution()
            bundle = load_standalone(self.path, dtype=dtype, local_only=True,
                execution_device=self.execution_device, memory_budget_bytes=memory_budget_bytes,
                device_memory_budget_bytes=device_memory_budget_bytes)
            self.model, self.tokenizer = bundle.model, bundle.tokenizer
            self.base_path, self.manifest = bundle.base_path, bundle.manifest
            self.encoding = _encoding_contract(self.tokenizer, self.family)
            self.representation = "factor_preserving"
            self.source_model_class = self.manifest["model_class"]
            self.router_precision = {"applicable": False, "reason": "bundle has no Switch routers"}
            self.loaded_resources = dict(_resources(), current_rss_bytes=_current_rss(), phase="weights_loaded",
                native_parameter_bytes=sum(p.numel() * p.element_size() for p in self.model.parameters()))
            if representation_inventory(self.path) != self.before:
                self.close()
                raise ValueError("bundle changed during loading")
            return
        self.before, self.raw, self.headers = _evaluation_preflight(self.path)
        if self.raw.get("silt_specialist_component"):
            raise ValueError("incomplete bundle base requires trained factors; load bundle root")
        kind = self.raw.get("model_type")
        if kind not in ("qwen2", "t5", "switch_transformers", "llama"):
            raise ValueError("native qwen2/t5/switch_transformers or evaluation-only llama required")
        self.family = "causal" if kind in ("qwen2", "llama") else "seq2seq"
        self.dtype, self.max_new_tokens = dtype, max_new_tokens
        self.memory_budget_bytes = memory_budget_bytes
        self.pre_import_admission = _admit_native_import(memory_budget_bytes)
        import torch
        from transformers import AutoTokenizer
        from .recovery import _load_recovery_model
        self.torch = torch
        meta, self.load_plan = _evaluation_inputs(self.raw, self.headers, dtype, torch)
        self.post_import_admission = _admit_loader(self.headers, self.raw, dtype, memory_budget_bytes,
            phase="loader_post_import", load_plan=self.load_plan, inventory=self.before)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True, trust_remote_code=False)
        self.encoding = _encoding_contract(self.tokenizer, self.family)
        self.admission = _admit_loader(self.headers, self.raw, dtype, memory_budget_bytes,
            phase="loader_pre_weights", load_plan=self.load_plan, inventory=self.before)
        self._select_execution()
        self.model = _load_recovery_model(meta, self.path, self.headers, self.load_plan, torch)
        self.router_precision = _native_router_precision(self.model, self.headers)
        if self.execution_device != "cpu":
            from .devices import device_admission, move_preserving_dtype
            device_admission(torch, self.execution_device, self.device_load_estimate,
                             device_memory_budget_bytes, phase="native-inference-transfer")
            move_preserving_dtype(self.model, self.execution_device)
        self.source_model_class = type(self.model).__name__
        self.model.eval().requires_grad_(False)
        self.loaded_resources = dict(_resources(), current_rss_bytes=_current_rss(), phase="weights_loaded",
            native_parameter_bytes=sum(p.numel() * p.element_size() for p in self.model.parameters()))

    def _select_execution(self):
        from .devices import resolve_device, inference_envelope
        self.inference_envelope = inference_envelope(self.raw, self.dtype, self.max_new_tokens,
            self.admission["header_loaded_bytes"],
            factor_rank=self.manifest["counts"]["rank"] if self.manifest else 0,
            max_input_tokens=self.max_input_tokens)
        work = self.inference_envelope["workspace"]["estimated_additional_peak_bytes"]
        self.device_load_estimate = self.inference_envelope["device_load_bytes"]
        from .devices import inference_host_phases
        storage = self.admission.get("loader_storage", dict(
            loaded_bytes=self.admission["header_loaded_bytes"], source_bytes=self.admission["source_payload_bytes"],
            retained_cast_source_bytes=self.admission.get("retained_cast_source_bytes", 0)))
        self.host_phase_envelope = inference_host_phases(**storage,
            workspace_bytes=work, extra_loader_bytes=self.admission.get("extra_loader_bytes", 0))
        host = self.host_phase_envelope["estimated_additional_peak_bytes"]
        self.device_selection = resolve_device(self.execution_device_requested, torch=self.torch, dtype=self.dtype,
            host_required_bytes=host, device_required_bytes=self.inference_envelope["device_peak_bytes"],
            memory_budget_bytes=self.memory_budget_bytes, device_memory_budget_bytes=self.device_memory_budget_bytes,
            total_process_cap=True)
        self.execution_device = self.device_selection["device"]

    def generate(self, prompt):
        self.last_trace = {"schema": "specialist_native_generation_v1", "stop_reason": "error",
            "model_kind": self.raw.get("model_type"),
            "generated_token_ids": None, "generated_token_count": None, "generate_config": None,
            "requested": {"max_new_tokens": self.max_new_tokens, "dtype": self.dtype, "do_sample": False,
                "num_beams": 1, "use_cache": True, "use_model_defaults": False}}
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 65536:
            raise ValueError("prompt must be nonblank and <=65536 characters")
        tok, torch = self.tokenizer, self.torch
        prefix, specials = render_prompt(tok, self.family, prompt)
        inputs = tok(prefix, add_special_tokens=specials, truncation=False,
            return_tensors="pt", return_token_type_ids=False)
        prefill = inputs["input_ids"][0].tolist()
        self.last_trace.update(input_token_ids=prefill if len(prefill) <= 2048 else None,
            input_token_count=len(prefill), input_token_ids_sha256=json_hash(prefill),
            input_ids_retention="complete" if len(prefill) <= 2048 else "oversize_rejected_hash_only",
            effective_prefill_sha256=hashlib.sha256(prefix.encode()).hexdigest())
        if self.max_input_tokens is not None and len(prefill) > self.max_input_tokens:
            raise ValueError("input exceeds declared/profiled max_input_tokens envelope; never truncate")
        if tok.eos_token_id is None:
            raise ValueError("native recovery encoding requires EOS")
        if self.family == "seq2seq" and (not prefill or prefill[-1] != tok.eos_token_id):
            raise ValueError("native seq2seq prompt must end in EOS, matching recovery")
        limits = [int(n) for n in (self.raw.get("max_position_embeddings"),
                  getattr(tok, "model_max_length", None)) if type(n) is int and 0 < n < 1000000]
        context_limit = min(limits + [2048])
        required = len(prefill) + self.max_new_tokens if self.family == "causal" else max(len(prefill), self.max_new_tokens + 1)
        if not prefill or required > context_limit:
            raise ValueError("complete input + generation budget exceeds native context limit; never truncate")
        cfg, policy = _generation_policy(self.model, tok, self.family, self.max_new_tokens)
        started = time.monotonic()
        self.last_trace.update(generate_config=policy["resolved_generate_config"], generation_policy=policy)
        try:
            self.generation_admission = _admit_inference(self.raw, self.dtype, len(prefill), self.max_new_tokens,
                memory_budget_bytes=self.memory_budget_bytes,
                unresident_weight_mmap_bytes=_unresident_weight_mmaps(self.path, self.headers),
                factor_rank=self.manifest["counts"]["rank"] if self.manifest else 0)
            self.last_trace["memory_admission"] = self.generation_admission
        except Exception as exc:
            self.last_trace["memory_admission"] = getattr(exc, "memory_admission", None)
            raise
        if self.execution_device != "cpu":
            from .devices import device_admission
            self.generation_admission["device_admission"] = device_admission(torch, self.execution_device,
                self.generation_admission["estimated_additional_peak_bytes"] + 128 * 1024**2,
                self.device_memory_budget_bytes, phase="generation-forward")
            inputs = {k: v.to(self.execution_device) for k, v in inputs.items()}
        with torch.inference_mode():
            result = self.model.generate(**inputs, generation_config=cfg, use_model_defaults=False)
        full = result[0].tolist()
        generated = full[len(prefill):] if self.family == "causal" else full[1:]
        eos = cfg.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        positions = [i for i, token in enumerate(generated) if token in eos_ids]
        stopped = bool(positions) and positions[-1] == len(generated) - 1
        cap = len(generated) >= self.max_new_tokens
        text = tok.decode(generated, skip_special_tokens=True)
        from .devices import runtime_metadata
        execution = runtime_metadata(torch, self.execution_device, self.model)
        return {"schema_version": 1, "status": "completed", "completed": True,
            "output": {"type": "text", "text": text}, "text": text,
            "representation": self.representation, "factor_preserving": self.manifest is not None,
            "standalone_teacher_independent": True, "external_teacher_loaded": False,
            "source_model_class": self.source_model_class,
            "execution_runtime": execution, "cross_device_validation": "unknown",
            "encoding": self.encoding, "router_precision": self.router_precision, "generation": {
                "schema": "specialist_native_generation_v1", "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "effective_prefill_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
                "input_token_ids": prefill, "generated_token_ids": generated,
                "native_sequence_token_ids": full, "input_token_count": len(prefill),
                "generated_token_count": len(generated), "eos_positions": positions,
                "cap_reached": cap, "truncated": cap and not stopped,
                "stop_reason": "eos" if stopped else ("max_new_tokens" if cap else "unknown"),
                "context_limit": context_limit, "greedy": policy["generation_mode"] == "greedy_search",
                "generate_config": policy["resolved_generate_config"], "generation_policy": policy,
                "dtype": self.dtype,
                "model_kind": self.raw.get("model_type"),
                "model_class": type(self.model).__name__, "tokenizer_class": type(tok).__name__,
                "device": str(next(self.model.parameters()).device),
                "attention_implementation": getattr(self.model.config, "_attn_implementation", None),
                "torch_version": torch.__version__,
                "transformers_version": __import__("transformers").__version__,
                "wall_seconds": time.monotonic() - started},
            "resources": dict(_resources(), device=self.execution_device, memory_budget_bytes=self.memory_budget_bytes,
                device_memory_budget_bytes=self.device_memory_budget_bytes, device_selection=self.device_selection,
                inference_envelope=self.inference_envelope, host_phase_envelope=self.host_phase_envelope,
                gpu_measurements={k: v for k, v in execution.items() if k.startswith("gpu_")},
                admission=self.admission, pre_import_admission=self.pre_import_admission,
                post_import_admission=self.post_import_admission, weights_loaded=self.loaded_resources,
                generation_admission=self.generation_admission, current_rss_bytes=_current_rss(),
                phase="generation_complete")}

    def close(self):
        self.model = None
        gc.collect()
        from .devices import release_cuda_cache
        if hasattr(self, "torch"):
            release_cuda_cache(self.torch, self.execution_device)


def infer(model, prompt, *, dtype="bfloat16", max_new_tokens=256, memory_budget_bytes=None,
          execution_device="cpu", device_memory_budget_bytes=None, max_input_tokens=None):
    if max_input_tokens is None:
        max_input_tokens = profile_local_inputs(model, [prompt], max_new_tokens)["max_input_tokens"]
    generator = NativeGenerator(model, dtype, max_new_tokens, memory_budget_bytes=memory_budget_bytes,
                execution_device=execution_device, device_memory_budget_bytes=device_memory_budget_bytes,
                max_input_tokens=max_input_tokens)
    try:
        result = generator.generate(prompt)
        if representation_inventory(generator.path) != generator.before:
            raise ValueError("model store changed during inference")
        result.update(model=str(generator.path), model_files=generator.before)
        return result
    finally:
        generator.close()


def validate(generations, suite, *, trace_policy="digest"):
    """Compare generated code via the unchanged external host oracle, never exec."""
    from asea.certification import extract_source, _function_passed
    from asea.certification.function_oracle import evaluate_functions
    if trace_policy not in ("none", "digest", "value"):
        raise ValueError("trace_policy must be none, digest or value")
    if not hasattr(suite, "cases"):
        suite = load_suite(suite)
    if not isinstance(generations, list):
        raise ValueError("generations must be a list with unique case id and native result")
    indexed = {}
    allowed = {c.id for c in suite.cases}
    for item in generations:
        if not isinstance(item, dict) or item.get("id") not in allowed or item["id"] in indexed:
            raise ValueError("unknown or duplicate generation id")
        indexed[item["id"]] = item
    rows = []
    for case in suite.cases:
        item = indexed.get(case.id, {"id": case.id, "status": "missing"})
        row = {"id": case.id, "group": case.group, "passed": False, "generation": item,
               "failure_stage": None, "oracle": None, "grade_complete": True,
               "generation_completed": item.get("status") == "completed", "oracle_executed": False,
               "operational_failure": False}
        # Offline missing/failed records and observed truncation retain the declared
        # task-failure contract. Live backend failures use explicit BLOCKED records.
        if item.get("status", "").upper() == "BLOCKED" or item.get("operational_failure") is True:
            row.update(grade_complete=False, operational_failure=True,
                failure_stage=item.get("failure_stage", "model_runtime"), error=item.get("error"))
            rows.append(row)
            continue
        try:
            generation = item.get("generation", {})
            if item.get("status") != "completed":
                raise ValueError("generation missing or failed")
            if generation.get("truncated") or generation.get("stop_reason") != "eos":
                raise ValueError("generation truncated or termination not established")
            if not isinstance(item.get("text"), str):
                raise ValueError("generated source is not text")
            source = extract_source(item["text"])
            if not source.strip():
                raise ValueError("empty generated source")
            from asea.certification.sandbox import SandboxLimits
            if len(source.encode("utf-8", "strict")) > SandboxLimits().source_bytes:
                raise ValueError("generated source exceeds oracle source byte budget")
            # Parsing is data-only: malformed candidate syntax is a completed
            # quality failure, never an unavailable oracle or a dropped problem.
            ast.parse(source)
        except Exception as exc:
            row.update(failure_stage="generation_or_preprocessing", error=str(exc))
        else:
            try:
                oracle = evaluate_functions(source, case.function_cases,
                    trace_policy={"schema_version": 1, "return_retention": trace_policy})
                row["oracle"] = oracle
                available = oracle.get("supported") is True and oracle.get("status") in ("PASSED", "FAILED", "TIMEOUT", "RESOURCE_LIMIT")
                row["oracle_executed"] = available
                if not available:
                    row.update(grade_complete=False, operational_failure=True,
                        failure_stage="function_oracle", error=oracle.get("reason", "oracle unavailable"))
                else:
                    row["passed"] = _function_passed(oracle, case.function_cases)
                    if not row["passed"]:
                        row["failure_stage"] = "function_oracle"
            except Exception as exc:
                # A valid source reached the host API but it did not provide a
                # verdict. Do not turn a dependency/setup exception into bad code.
                row.update(grade_complete=False, operational_failure=True,
                    failure_stage="function_oracle", error=str(exc))
        rows.append(row)
    passed = sum(r["passed"] for r in rows)
    blocked = sum(r["operational_failure"] for r in rows)
    graded = sum(r["grade_complete"] for r in rows)
    groups = {}
    for group in ("target", "control"):
        items = [r for r in rows if r["group"] == group]
        n = sum(r["passed"] for r in items)
        unavailable = sum(r["operational_failure"] for r in items)
        groups[group] = {"tasks": len(items), "passed": n, "blocked": unavailable,
            "pass_rate": n / len(items) if items and not unavailable else None}
    return {"schema_version": 1, "status": "BLOCKED" if blocked else "completed", "completed": not blocked,
        "engineering_complete": not blocked, "quality_pass": not blocked and passed == len(rows), "certificate": False,
        "suite": suite.name, "suite_config_sha256": json_hash(suite.model_dump(mode="json")),
        "tasks_total": len(rows), "tasks_passed": passed, "tasks_failed": graded - passed,
        "tasks_blocked": blocked, "tasks_graded": graded, "operational_failures": blocked,
        "generation_completed": sum(r["generation_completed"] for r in rows),
        "oracle_executed": sum(r["oracle_executed"] for r in rows),
        "pass_rate": passed / len(rows) if rows and not blocked else None,
        "observed_pass_fraction": passed / len(rows) if rows else None, "groups": groups, "cases": rows,
        "trace_policy": trace_policy, "return_values_authoritative": False,
        "claim": "Observed short-function IO only; host verdicts authoritative, no universal coding or safety certificate",
        "noninferiority": "insufficient_evidence_no_power_or_universal_claim"}


def oracle_preflight():
    """Trusted cheap containment check before allocating any model weights."""
    from asea.certification.sandbox import probe_code_sandbox
    try:
        result = probe_code_sandbox()
        evidence = dataclasses.asdict(result)
        return dict(evidence, available=result.supported and result.passed)
    except Exception as exc:
        return {"available": False, "status": "BLOCKED", "error": str(exc)}


def evaluate(model, suite, *, dtype="bfloat16", max_new_tokens=256, trace_policy="digest", memory_budget_bytes=None,
             execution_device="cpu", device_memory_budget_bytes=None, max_input_tokens=None):
    suite_path = safe_file(suite)
    suite_hash = file_hash(suite_path)
    loaded = load_suite(suite_path)
    generator, generations, load_error, before = None, [], None, None
    load_admission_error = None
    input_profile = None
    availability = oracle_preflight()
    failure_stage = "oracle_preflight" if not availability["available"] else "model_load"
    try:
        try:
            if not availability["available"]:
                raise RuntimeError("trusted oracle preflight unavailable")
            before = representation_inventory(model)
            input_profile = profile_local_inputs(model, [case.input for case in loaded.cases], max_new_tokens)
            if max_input_tokens is None:
                max_input_tokens = input_profile["max_input_tokens"]
            elif input_profile["max_input_tokens"] > max_input_tokens:
                raise ValueError("suite input exceeds declared max_input_tokens; never truncate")
            generator = NativeGenerator(model, dtype, max_new_tokens, memory_budget_bytes=memory_budget_bytes,
                execution_device=execution_device, device_memory_budget_bytes=device_memory_budget_bytes,
                max_input_tokens=max_input_tokens)
        except Exception as exc:
            load_error = str(exc)
            load_admission_error = getattr(exc, "memory_admission", None)
        for case in loaded.cases:
            try:
                if generator is None:
                    raise RuntimeError(load_error)
                result = generator.generate(case.input)
            except Exception as exc:
                result = {"status": "BLOCKED", "completed": False, "operational_failure": True,
                    "failure_stage": failure_stage if generator is None else "model_runtime", "error": str(exc),
                    "generation": dict(getattr(generator, "last_trace", {}) or {}),
                    "encoding": getattr(generator, "encoding", None),
                    "memory_admission": load_admission_error if generator is None else getattr(exc, "memory_admission", None)}
            generations.append(dict(result, id=case.id))
    finally:
        if generator is not None:
            generator.close()
    report = validate(generations, loaded, trace_policy=trace_policy)
    try:
        unchanged = before is not None and before == representation_inventory(model) and suite_hash == file_hash(suite_path)
    except Exception as exc:
        unchanged = False
        load_error = load_error or str(exc)
    report.update(model=str(safe_path(model)), model_files=before, suite_file=suite_hash,
        dtype=dtype, max_new_tokens=max_new_tokens, max_input_tokens=max_input_tokens,
        input_profile=input_profile, inputs_unchanged=unchanged,
        oracle_preflight=availability, resources=dict(_resources(),
            device=getattr(generator, "execution_device", None),
            execution_device_requested=execution_device,
            device_selection=getattr(generator, "device_selection", None),
            memory_budget_bytes=memory_budget_bytes, device_memory_budget_bytes=device_memory_budget_bytes),
        representation=getattr(generator, "representation", None),
        source_model_class=getattr(generator, "source_model_class", None),
        factor_preserving=(getattr(generator, "manifest", None) is not None) if generator else None,
        external_teacher_loaded=False if generator else None)
    if load_error or not unchanged:
        report.update(status="BLOCKED", completed=False, engineering_complete=False, quality_pass=False,
                      error=load_error or "input changed during evaluation")
    return report
