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
    suite = EvaluationSuite.model_validate(_bounded_json(safe_file(path)))
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


def _admit_loader(headers, config, dtype, memory_budget_bytes=None, *, phase="loader_pre_import", native_loaded_bytes=0):
    from .reconstruction import _DTYPES, _SIZES, ReconstructionBlocked
    dimensions = _runtime_dimensions(config)
    if dtype not in _DTYPES or not headers or any(v.get("dtype") not in _SIZES for v in headers.values()):
        raise ReconstructionBlocked("unknown inference weight dtype/header")
    b = _DTYPES[dtype]
    # Preserve native F32 wo tensors and restored routers, as in reconstruction.
    loaded = max(native_loaded_bytes, sum(v["numel"] * max(b, 4 if k.endswith((".wo.weight", "router.classifier.weight")) or ".lora_" in k else b)
        for k, v in headers.items()))
    source = sum(v["numel"] * _SIZES[v["dtype"]] for v in headers.values())
    transient, runtime = max(loaded, source), 512 * 1024**2
    return _runtime_admission(phase, loaded + transient + runtime, dict(
        dimensions=dimensions, header_loaded_bytes=loaded, source_payload_bytes=source,
        source_mmap_and_cast_transient_bytes=transient, loader_runtime_reserve_bytes=runtime,
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


def _admit_inference(config, dtype, prefill_tokens, max_new_tokens, memory_budget_bytes=None, *, unresident_weight_mmap_bytes=0, factor_rank=0):
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
    return _runtime_admission("inference_incremental", estimate, dict(
        dimensions=d, dtype_bytes=b, factor_workspace_bytes=factor_workspace, actual_prefill_tokens=p, max_new_tokens=g, cache_tokens=t,
        kv_cache_bytes=kv, cache_concat_overlap_bytes=concat, retained_encoder_bytes=encoder,
        prefill_activation_bytes=prefill_workspace, native_prefill_logits_bytes=prefill_logits,
        decode_activation_bytes=decode_workspace, native_decode_logits_bytes=decode_logits,
        bounded_score_bytes=scores, retained_generation_score_bytes=0,
        stage_additional_bytes=stages, stage_peak_bytes=stage_peak, token_and_mask_bytes=tokens,
        incremental_runtime_reserve_bytes=reserve, resident_weights_charged_bytes=0,
        unresident_weight_mmap_bytes=unresident_weight_mmap_bytes,
        formula="factor_workspace + unresident_mmaps + KV + concat_overlap + encoder + tokens + 64MiB + max(prefill_forward, prefill_decode_overlap)"), memory_budget_bytes)


def _resources():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"peak_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024),
            "measurement": "process_lifetime_high_water_not_isolated_per_model",
            "allocator_release_guaranteed": False}


class NativeGenerator:
    """One frozen native model, explicit greedy configuration, exact recovery prefix."""
    def __init__(self, model, dtype="bfloat16", max_new_tokens=256, memory_budget_bytes=None):
        if dtype not in ("bfloat16", "float32"):
            raise ValueError("dtype must be bfloat16 or float32")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
            raise ValueError("max_new_tokens must be in [1,384]")
        self.path = safe_path(model)
        from .reconstruction import _artifact_preflight, _native_meta, _restore_f32_routers
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
            self.pre_import_admission = _admit_loader(self.headers, self.raw, dtype, memory_budget_bytes)
            import torch
            self.torch = torch
            _, _, meta = _native_meta(self.raw, base_headers, torch)
            native_bytes = sum(p.numel() * max(2 if dtype == "bfloat16" else 4,
                4 if k.endswith(".wo.weight") else 0) for k, p in meta.named_parameters())
            native_bytes += self.manifest["counts"]["factor_parameters"] * 4
            del meta
            self.post_import_admission = _admit_loader(self.headers, self.raw, dtype, memory_budget_bytes,
                phase="loader_post_import", native_loaded_bytes=native_bytes)
            self.admission = _admit_loader(self.headers, self.raw, dtype, memory_budget_bytes,
                phase="loader_pre_weights", native_loaded_bytes=native_bytes)
            bundle = load_standalone(self.path, dtype=dtype, local_only=True)
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
        self.pre_import_admission = _admit_loader(self.headers, self.raw, dtype, memory_budget_bytes)
        import torch
        from transformers import AutoTokenizer
        self.torch = torch
        constructor = _native_llama_meta if kind == "llama" else _native_meta
        config, model_cls, meta = constructor(self.raw, self.headers, torch)
        native_bytes = sum(p.numel() * max(2 if dtype == "bfloat16" else 4,
            4 if k.endswith((".wo.weight", "router.classifier.weight")) else 0) for k, p in meta.named_parameters())
        del meta
        self.post_import_admission = _admit_loader(self.headers, self.raw, dtype, memory_budget_bytes,
            phase="loader_post_import", native_loaded_bytes=native_bytes)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True, trust_remote_code=False)
        self.encoding = _encoding_contract(self.tokenizer, self.family)
        self.admission = _admit_loader(self.headers, self.raw, dtype, memory_budget_bytes,
            phase="loader_pre_weights", native_loaded_bytes=native_bytes)
        self.model, loading = model_cls.from_pretrained(str(self.path), local_files_only=True,
            config=config, trust_remote_code=False, use_safetensors=True, torch_dtype=getattr(torch, dtype),
            low_cpu_mem_usage=True, output_loading_info=True)
        self.router_precision = _restore_f32_routers(self.model, self.path, self.headers, torch)
        if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            self.close()
            raise ValueError("native checkpoint has incomplete/incompatible weights")
        self.source_model_class = type(self.model).__name__
        self.model.eval().requires_grad_(False)
        self.loaded_resources = dict(_resources(), current_rss_bytes=_current_rss(), phase="weights_loaded",
            native_parameter_bytes=sum(p.numel() * p.element_size() for p in self.model.parameters()))

    def generate(self, prompt):
        self.last_trace = {"schema": "specialist_native_generation_v1", "stop_reason": "error",
            "model_kind": self.raw.get("model_type"),
            "generated_token_ids": None, "generated_token_count": None, "generate_config": None,
            "requested": {"max_new_tokens": self.max_new_tokens, "dtype": self.dtype, "do_sample": False}}
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 65536:
            raise ValueError("prompt must be nonblank and <=65536 characters")
        tok, torch = self.tokenizer, self.torch
        if self.family == "seq2seq":
            prefix, specials = prompt, True
        elif getattr(tok, "chat_template", None):
            prefix = tok.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
            specials = False
        else:
            prefix, specials = prompt + "\n", False
        inputs = tok(prefix, add_special_tokens=specials, truncation=False,
            return_tensors="pt", return_token_type_ids=False)
        prefill = inputs["input_ids"][0].tolist()
        self.last_trace.update(input_token_ids=prefill if len(prefill) <= 2048 else None,
            input_token_count=len(prefill), input_token_ids_sha256=json_hash(prefill),
            input_ids_retention="complete" if len(prefill) <= 2048 else "oversize_rejected_hash_only",
            effective_prefill_sha256=hashlib.sha256(prefix.encode()).hexdigest())
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
        from transformers import GenerationConfig
        # Fresh config avoids checkpoint sampling/forced-token defaults silently
        # changing this comparison; only native special IDs are inherited.
        cfg = GenerationConfig.from_model_config(self.model.config)
        cfg.do_sample, cfg.num_beams, cfg.num_return_sequences = False, 1, 1
        cfg.max_new_tokens, cfg.use_cache = self.max_new_tokens, True
        cfg.min_length, cfg.min_new_tokens = 0, 0
        cfg.forced_bos_token_id, cfg.forced_eos_token_id = None, None
        cfg.repetition_penalty, cfg.no_repeat_ngram_size = 1.0, 0
        cfg.return_dict_in_generate, cfg.output_scores = False, False
        cfg.output_logits, cfg.output_attentions, cfg.output_hidden_states = False, False, False
        cfg.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        started = time.monotonic()
        self.last_trace["generate_config"] = cfg.to_dict()
        try:
            self.generation_admission = _admit_inference(self.raw, self.dtype, len(prefill), self.max_new_tokens,
                memory_budget_bytes=self.memory_budget_bytes,
                unresident_weight_mmap_bytes=_unresident_weight_mmaps(self.path, self.headers),
                factor_rank=self.manifest["counts"]["rank"] if self.manifest else 0)
            self.last_trace["memory_admission"] = self.generation_admission
        except Exception as exc:
            self.last_trace["memory_admission"] = getattr(exc, "memory_admission", None)
            raise
        with torch.inference_mode():
            result = self.model.generate(**inputs, generation_config=cfg)
        full = result[0].tolist()
        generated = full[len(prefill):] if self.family == "causal" else full[1:]
        eos = cfg.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        positions = [i for i, token in enumerate(generated) if token in eos_ids]
        stopped = bool(positions) and positions[-1] == len(generated) - 1
        cap = len(generated) >= self.max_new_tokens
        text = tok.decode(generated, skip_special_tokens=True)
        return {"schema_version": 1, "status": "completed", "completed": True,
            "output": {"type": "text", "text": text}, "text": text,
            "representation": self.representation, "factor_preserving": self.manifest is not None,
            "standalone_teacher_independent": True, "external_teacher_loaded": False,
            "source_model_class": self.source_model_class,
            "encoding": self.encoding, "router_precision": self.router_precision, "generation": {
                "schema": "specialist_native_generation_v1", "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "effective_prefill_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
                "input_token_ids": prefill, "generated_token_ids": generated,
                "native_sequence_token_ids": full, "input_token_count": len(prefill),
                "generated_token_count": len(generated), "eos_positions": positions,
                "cap_reached": cap, "truncated": cap and not stopped,
                "stop_reason": "eos" if stopped else ("max_new_tokens" if cap else "unknown"),
                "context_limit": context_limit, "greedy": True,
                "generate_config": cfg.to_dict(), "dtype": self.dtype,
                "model_kind": self.raw.get("model_type"),
                "model_class": type(self.model).__name__, "tokenizer_class": type(tok).__name__,
                "device": str(next(self.model.parameters()).device),
                "attention_implementation": getattr(self.model.config, "_attn_implementation", None),
                "torch_version": torch.__version__,
                "transformers_version": __import__("transformers").__version__,
                "wall_seconds": time.monotonic() - started},
            "resources": dict(_resources(), device="cpu", memory_budget_bytes=self.memory_budget_bytes,
                admission=self.admission, pre_import_admission=self.pre_import_admission,
                post_import_admission=self.post_import_admission, weights_loaded=self.loaded_resources,
                generation_admission=self.generation_admission, current_rss_bytes=_current_rss(),
                phase="generation_complete")}

    def close(self):
        self.model = None
        gc.collect()


def infer(model, prompt, *, dtype="bfloat16", max_new_tokens=256, memory_budget_bytes=None):
    generator = NativeGenerator(model, dtype, max_new_tokens, memory_budget_bytes=memory_budget_bytes)
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


def evaluate(model, suite, *, dtype="bfloat16", max_new_tokens=256, trace_policy="digest", memory_budget_bytes=None):
    suite_path = safe_file(suite)
    suite_hash = file_hash(suite_path)
    loaded = load_suite(suite_path)
    generator, generations, load_error, before = None, [], None, None
    load_admission_error = None
    availability = oracle_preflight()
    failure_stage = "oracle_preflight" if not availability["available"] else "model_load"
    try:
        try:
            if not availability["available"]:
                raise RuntimeError("trusted oracle preflight unavailable")
            before = representation_inventory(model)
            generator = NativeGenerator(model, dtype, max_new_tokens, memory_budget_bytes=memory_budget_bytes)
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
        dtype=dtype, max_new_tokens=max_new_tokens, inputs_unchanged=unchanged,
        oracle_preflight=availability, resources=dict(_resources(), device="cpu", memory_budget_bytes=memory_budget_bytes),
        representation=getattr(generator, "representation", None),
        source_model_class=getattr(generator, "source_model_class", None),
        factor_preserving=(getattr(generator, "manifest", None) is not None) if generator else None,
        external_teacher_loaded=False if generator else None)
    if load_error or not unchanged:
        report.update(status="BLOCKED", completed=False, engineering_complete=False, quality_pass=False,
                      error=load_error or "input changed during evaluation")
    return report
