"""Trusted single-device execution admission; no plugin, offload or GPU emulation.

CPU remains the default. CUDA uses a full CPU native load followed by .to(device),
NOT a low-host-memory streaming loader. All tensor dtypes are preserved. Estimates
are conservative admission bounds, not guarantees against concurrent allocation.
"""
from __future__ import annotations

import os
import re


class DeviceRejected(ValueError):
    """Unsupported/unavailable execution backend or insufficient current resources."""


def inference_loaded_bytes(headers, dtype, native_loaded_bytes=0):
    """Shared conservative inference storage count, including native F32 exceptions.

    This matches the loader's admission bound, not a promise that all Switch wo
    weights actually stay F32. Factors are always F32; tied aliases are supplied
    as unique storage by the header/shape preflight.
    """
    byte = {"bfloat16": 2, "float32": 4}[dtype]
    return max(native_loaded_bytes, sum(v["numel"] * max(byte,
        4 if k.endswith((".wo.weight", "router.classifier.weight")) or ".lora_" in k else byte)
        for k, v in headers.items()))


def inference_envelope(config, dtype, max_new_tokens, loaded_bytes, *, factor_rank=0, max_input_tokens=None):
    """Declared/profiled input bound shared by planner and runtime.

    Absent an input profile, retain conservative min(2048, native context).
    An explicit bound is an enforceable envelope, never truncation permission.
    Generation budget and dtype are unchanged. No tensor/hardware operations.
    """
    from .evaluation import _admit_inference
    # These are native architecture defaults already allowed by shape preflight.
    config = dict(config)
    if config["model_type"] in ("qwen2", "llama"):
        config.setdefault("num_key_value_heads", config["num_attention_heads"])
    elif config.get("num_decoder_layers") is None:
        config["num_decoder_layers"] = config["num_layers"]
    context = min(2048, config.get("max_position_embeddings", 2048))
    if max_input_tokens is not None:
        if type(max_input_tokens) is not int or not 1 <= max_input_tokens <= context:
            raise DeviceRejected("max_input_tokens must be a positive integer within native context")
        context = max_input_tokens
    work = _admit_inference(config, dtype, context, max_new_tokens,
        factor_rank=factor_rank, _estimate_only=True)
    transfer_reserve = 128 * 1024**2
    return {"prefill_tokens": context, "max_new_tokens": max_new_tokens,
            "input_bound_kind": "declared_or_profiled" if max_input_tokens is not None else "conservative_unprofiled",
            "oversize_policy": "reject_without_truncation",
            "factor_rank": factor_rank, "loaded_tensor_bytes": loaded_bytes,
            "workspace": work, "device_transfer_reserve_bytes": transfer_reserve,
            "device_load_bytes": loaded_bytes + transfer_reserve,
            "device_peak_bytes": loaded_bytes + transfer_reserve + work["estimated_additional_peak_bytes"],
            "estimate_not_guarantee": True}


# Only this concrete loader has a per-tensor lifetime proof. Unknown strategies
# retain the full-shard conversion allowance, even if their name sounds native.
EXPLICIT_NATIVE_LOADER = "explicit_safetensors_tensor_meta_assign"
NATIVE_BUFFER_RESERVE = 16 * 1024**2


def explicit_loader_storage(headers, plan, inventory):
    """Header-only accounting for the already validated native assignment plan.

    Equal dtype CPU .to aliases mapped storage. Cast source pages can remain
    mapped beside other tensors: charge ALL, plus largest source+target scratch.
    File metadata/alignment and bounded native buffers are not tensor payload.
    """
    from .reconstruction import _SIZES, _DTYPES, ReconstructionBlocked
    if plan.get("strategy") != EXPLICIT_NATIVE_LOADER or set(plan.get("target_dtypes", {})) != set(headers):
        raise ReconstructionBlocked("unproven or incomplete explicit inference load plan")
    loaded = source = retained = scratch = 0
    for key, entry in headers.items():
        target = plan["target_dtypes"][key]
        if target not in _DTYPES or entry["dtype"] not in _SIZES:
            raise ReconstructionBlocked("unknown explicit inference dtype")
        raw, native = entry["numel"] * _SIZES[entry["dtype"]], entry["numel"] * _DTYPES[target]
        source += raw
        loaded += native
        if entry["dtype"] != {"float32": "F32", "bfloat16": "BF16"}[target]:
            retained += raw
            scratch = max(scratch, raw + native)
    files = {entry["file"] for entry in headers.values()}
    mapping = sum(inventory[name]["size"] for name in files) - source + len(files) * 8192
    if mapping < 0:
        raise ReconstructionBlocked("inconsistent explicit inference mapping extent")
    return dict(loaded_bytes=loaded, source_bytes=source,
        retained_cast_source_bytes=retained, largest_cast_scratch_bytes=scratch,
        mapping_overhead_bytes=mapping, buffer_reserve_bytes=NATIVE_BUFFER_RESERVE,
        loader_strategy=EXPLICIT_NATIVE_LOADER)


def inference_host_phases(loaded_bytes, source_bytes, workspace_bytes, *,
                          runtime_reserve_bytes=512 * 1024**2, tokenizer_bytes=0,
                          retained_cast_source_bytes=0, loader_strategy=None,
                          largest_cast_scratch_bytes=0, mapping_overhead_bytes=0,
                          buffer_reserve_bytes=0, extra_loader_bytes=0):
    """Additional allocations before weights: serial loader / forward peaks.

    Current process RSS is NOT a component here: runtime compares this additional
    peak to observed free RAM, and separately to total operator cap minus RSS.
    The 512MiB is an *incremental* library/loader reserve, not a second copy of
    already resident Python/Torch. mmap/cast pages are fully charged at load;
    forward charges native storage once plus any retained cast-source mapping
    allowance (no assumed release credit). Runtime rechecks unresident mmap pages.
    """
    proven = loader_strategy == EXPLICIT_NATIVE_LOADER
    components = (loaded_bytes, source_bytes, workspace_bytes, runtime_reserve_bytes,
        tokenizer_bytes, retained_cast_source_bytes, largest_cast_scratch_bytes,
        mapping_overhead_bytes, buffer_reserve_bytes, extra_loader_bytes)
    if any(type(n) is not int or n < 0 for n in components):
        raise DeviceRejected("invalid inference host allocation component")
    overhead = runtime_reserve_bytes + tokenizer_bytes + mapping_overhead_bytes + buffer_reserve_bytes
    transient = retained_cast_source_bytes + largest_cast_scratch_bytes if proven else max(loaded_bytes, source_bytes)
    load = loaded_bytes + transient + overhead + extra_loader_bytes
    forward = loaded_bytes + retained_cast_source_bytes + workspace_bytes + overhead
    return {"loader_bytes": load, "forward_bytes": forward,
            "loader_strategy": loader_strategy if proven else "unknown_conservative_full_shard",
            "largest_cast_scratch_bytes": largest_cast_scratch_bytes,
            "mapping_overhead_bytes": mapping_overhead_bytes, "buffer_reserve_bytes": buffer_reserve_bytes,
            "extra_loader_bytes": extra_loader_bytes,
            "retained_cast_source_bytes": retained_cast_source_bytes,
            "estimated_additional_peak_bytes": max(load, forward),
            "combination": "max(serial_loader, loaded_weights_plus_forward)",
            "current_rss_included": False, "runtime_reserve_is_incremental": True}


def recovery_workspace(config, dtype, length, response_tokens, *, training=False,
                       factor_rank=0, factor_parameters=0, kd=False):
    """Batch-one native eager workspace, derived from tensor lifetimes (no weights).

    Non-reentrant block checkpointing saves block inputs, not every block's
    attention/MLP temporaries. Recompute/backward has ONE block live. Frozen
    weights referenced by saved-tensor hooks alias model storage and cost zero
    extra here. CE/full-vocabulary KL are top-level tensors, never depth-scaled.
    3x one-block workspace covers recomputed values, their gradients and GEMM/
    softmax backward scratch. 128MiB library reserve is separate from the device
    transfer reserve and free-memory 10%/256MiB admission margin. This is a
    conservative allocation model, not CPU measurements claimed as GPU proof.
    """
    from .evaluation import _admit_inference
    if type(response_tokens) is not int or not 1 <= response_tokens <= length:
        raise DeviceRejected("invalid complete response length")
    if type(factor_parameters) is not int or factor_parameters < 0:
        raise DeviceRejected("invalid factor parameter count")
    config = dict(config)
    if config['model_type'] in ('qwen2', 'llama'):
        config.setdefault('num_key_value_heads', config['num_attention_heads'])
    elif config.get('num_decoder_layers') is None:
        config['num_decoder_layers'] = config['num_layers']
    native = _admit_inference(config, dtype, length, 2, factor_rank=factor_rank, _estimate_only=True)
    d, b = native["dimensions"], native["dtype_bytes"]
    depth = d["layers"] if d["causal"] else d["layers"] + d["decoder_layers"]
    # Shared encoder state plus self/cross-attention in a recomputed decoder.
    block = native["prefill_activation_bytes"] * (1 if d["causal"] else 2)
    block += native["factor_workspace_bytes"]
    # Native RMSNorm/T5 precision and PEFT inputs may be F32 even under BF16.
    saved_inputs = (depth + (1 if d["causal"] else 3)) * length * d["hidden"] * 4 if training else 0
    # Full model output + selected response copy + CE log-softmax/backward;
    # KL adds teacher F32 targets, normalizers and checkpointed chunk scratch.
    full_logits = length * d["vocab"] * max(b, 4)
    response_logits = response_tokens * d["vocab"] * (b + (24 if kd else 20)) if training else response_tokens * d["vocab"] * (b + 4)
    optimizer = factor_parameters * 8 if training else 0  # Adam m/v, F32
    gradients = factor_parameters * 4 if training else 0
    factors = factor_parameters * 4 if training else 0
    # host-only best snapshot is NOT duplicated into device workspace.
    reserve = 128 * 1024**2
    block_peak = block * (3 if training else 1)
    fields = {"checkpoint_saved_inputs_bytes": saved_inputs,
        "one_block_forward_bytes": block, "recompute_backward_bytes": block_peak,
        "full_output_logits_bytes": full_logits, "response_ce_kl_bytes": response_logits,
        "factor_parameter_bytes": factors, "optimizer_state_bytes": optimizer,
        "gradient_bytes": gradients, "library_reserve_bytes": reserve,
        "resident_base_weights_charged_bytes": 0, "saved_weight_reference_extra_bytes": 0,
        "depth": depth, "length": length, "response_tokens": response_tokens,
        "gradient_checkpointing": training, "backward_block_multiplier": 3 if training else 1}
    fields["estimated_additional_peak_bytes"] = (saved_inputs + block_peak + full_logits + response_logits +
        factors + optimizer + gradients + reserve + native["token_and_mask_bytes"])
    fields['persistent_training_buffers_bytes'] = factors + optimizer + gradients
    fields['forward_backward_transient_bytes'] = fields['estimated_additional_peak_bytes'] - factors - optimizer - gradients
    return fields


def recovery_incremental_workspace(envelope, *, resident_factors=0, resident_optimizer=0, resident_gradients=0):
    """Credit only observed live buffer storage (already charged to driver free).

    Never credit cached/reserved allocator blocks, presumed teardown or base
    weights: base storage is excluded from the workspace model from the outset.
    """
    credit = 0
    for amount, component in ((resident_factors, 'factor_parameter_bytes'),
            (resident_optimizer, 'optimizer_state_bytes'), (resident_gradients, 'gradient_bytes')):
        if type(amount) is not int or amount < 0:
            raise DeviceRejected('invalid observed resident training buffers')
        credit += min(amount, envelope[component])
    return envelope['estimated_additional_peak_bytes'] - credit


def validate_device_request(execution_device):
    if not isinstance(execution_device, str):
        raise DeviceRejected("execution_device must be cpu, auto, cuda or cuda:N")
    if execution_device in ("cpu", "auto", "cuda") or re.fullmatch(r"cuda:(0|[1-9][0-9]*)", execution_device):
        return "cuda:0" if execution_device == "cuda" else execution_device
    if any(token in execution_device.lower() for token in ("mps", "rocm", "hip", "offload", "multi", ",")):
        raise DeviceRejected("MPS/ROCm/HIP/multi-GPU/offload are recognized but explicitly unsupported; no fallback")
    raise DeviceRejected("Invalid execution_device; expected cpu, auto or cuda:N (no fallback)")


def _positive_budget(value):
    if value is not None and (type(value) is not int or value <= 0):
        raise DeviceRejected("device_memory_budget_bytes must be a positive integer or None")


def _cuda_probe(torch, device, dtype, *, operators=True):
    """Observe actual NVIDIA runtime and execute tiny operators before model weights.

    Device context restores the caller's current device. No RNG or global TF32 /
    determinism setting is changed. BF16 emulation on pre-Ampere is not admitted.
    """
    if getattr(torch.version, "hip", None) or not getattr(torch.version, "cuda", None):
        raise DeviceRejected("CUDA requires an NVIDIA CUDA torch build, not CPU or HIP/ROCm")
    if not torch.cuda.is_available():
        raise DeviceRejected("NVIDIA CUDA runtime/driver is unavailable; no CPU fallback")
    index = int(device.split(":")[1])
    if index >= torch.cuda.device_count():
        raise DeviceRejected("Requested CUDA device index is unavailable: " + device)
    try:
        with torch.cuda.device(index):
            torch.cuda.init()
            props = torch.cuda.get_device_properties(index)
            capability = tuple(torch.cuda.get_device_capability(index))
            bf16 = capability[0] >= 8 and bool(torch.cuda.is_bf16_supported())
            if dtype == "bfloat16" and not bf16:
                raise DeviceRejected("Native BF16 unsupported on requested NVIDIA device: " + device)
            if operators:
                # Covers dense native projection, FP32 softmax and autograd on the
                # selected device. Architecture-specific kernels are still tested
                # by the real forward and fail closed rather than falling back.
                with torch.enable_grad():
                    x = torch.ones((8, 8), dtype=getattr(torch, dtype), device=device, requires_grad=True)
                    y = (x @ x).float().softmax(-1).square().sum()
                    y.backward()
                    if x.grad is None or not bool(torch.isfinite(x.grad).all()):
                        raise DeviceRejected("CUDA operator probe returned invalid gradients")
                torch.cuda.synchronize(index)
                del x, y
            return {"device": device, "name": props.name, "capability": list(capability),
                    "total_vram_bytes": int(props.total_memory), "native_bfloat16_supported": bf16,
                    "torch_cuda_version": torch.version.cuda, "hip": None,
                    "operator_probe": "matmul_softmax_backward_passed" if operators else "not_repeated"}
    except DeviceRejected:
        raise
    except Exception as exc:
        raise DeviceRejected("CUDA runtime/operator probe failed on %s: %s" % (device, exc)) from exc


def device_admission(torch, device, required_bytes, device_memory_budget_bytes=None, *, phase="runtime"):
    """Reobserve free VRAM on ONE device; never sum GPUs or assume empty_cache freed.

    The optional operator ceiling is total current PyTorch RESERVED bytes plus the
    proposed additional allocation, separate from host RAM and driver free VRAM.
    Free-memory safety reserve is max(256MiB,10%). Cached blocks receive no credit.
    """
    _positive_budget(device_memory_budget_bytes)
    if type(required_bytes) is not int or required_bytes < 0:
        raise DeviceRejected("Invalid device allocation estimate")
    if device == "cpu":
        return {"device": "cpu", "phase": phase, "applicable": False, "admitted": True}
    try:
        with torch.cuda.device(int(device.split(":")[1])):
            free, total = (int(v) for v in torch.cuda.mem_get_info())
            allocated = int(torch.cuda.memory_allocated())
            reserved = int(torch.cuda.memory_reserved())
        if not 0 <= allocated <= reserved <= total or not 0 <= free <= total:
            raise DeviceRejected("Inconsistent NVIDIA memory observations")
    except DeviceRejected:
        raise
    except Exception as exc:
        raise DeviceRejected("Cannot observe per-device CUDA memory: " + str(exc)) from exc
    safety = max(256 * 1024**2, free // 10)
    available = max(0, free - safety)
    cap_remaining = None if device_memory_budget_bytes is None else max(0, device_memory_budget_bytes - reserved)
    limit = available if cap_remaining is None else min(available, cap_remaining)
    report = dict(device=device, phase=phase, applicable=True, free_vram_bytes=free,
                  total_vram_bytes=total, allocated_bytes=allocated, reserved_bytes=reserved,
                  safety_headroom_bytes=safety, requested_device_memory_budget_bytes=device_memory_budget_bytes,
                  operator_remaining_bytes=cap_remaining, limit_bytes=limit,
                  estimated_additional_peak_bytes=required_bytes, admitted=required_bytes <= limit,
                  reclaimed_memory_credit_bytes=0, estimate_not_guarantee=True)
    if not report["admitted"]:
        exc = DeviceRejected("CUDA memory admission refused at %s on %s: %d > %d bytes" %
                             (phase, device, required_bytes, limit))
        exc.memory_admission = report
        raise exc
    return report


def host_admission(required_bytes, memory_budget_bytes=None, *, phase="device-host", total_process_cap=False):
    from .reconstruction import _memory_budget
    if total_process_cap:
        from .evaluation import _runtime_admission
        return _runtime_admission(phase, required_bytes,
            {"loader_policy": "full native CPU allocation before CUDA transfer; no low-host streaming claim"}, memory_budget_bytes)
    budget = _memory_budget(memory_budget_bytes)
    report = dict(budget, phase=phase, estimated_additional_peak_bytes=required_bytes,
                  admitted=required_bytes <= budget["limit_bytes"])
    if not report["admitted"]:
        exc = DeviceRejected("Host memory admission refused at %s: %d > %d bytes" %
                             (phase, required_bytes, budget["limit_bytes"]))
        exc.memory_admission = report
        raise exc
    return report


def resolve_device(execution_device="cpu", *, torch=None, dtype="float32", host_required_bytes=0,
                   device_required_bytes=0, memory_budget_bytes=None, device_memory_budget_bytes=None,
                   total_process_cap=False):
    """Choose a backend only after resource-fit checks; explicit requests never fall back.

    Auto requires known nonzero header/config-derived peaks. Auto can fall back to
    CPU after CUDA capability/fit refusal, but must independently pass host fit.
    """
    request = validate_device_request(execution_device)
    _positive_budget(device_memory_budget_bytes)
    if dtype not in ("float32", "bfloat16"):
        raise DeviceRejected("Only float32/bfloat16 execution is supported")
    if request == "auto" and (host_required_bytes <= 0 or device_required_bytes <= 0):
        raise DeviceRejected("auto requires header/config-derived host and device resource estimates")
    host = host_admission(host_required_bytes, memory_budget_bytes, total_process_cap=total_process_cap)
    if request == "cpu":
        return {"requested": execution_device, "device": "cpu", "host_admission": host,
                "cross_device_validation": "unknown", "fallback": False}
    if torch is None:
        import torch
    candidates = [request]
    if request == "auto":
        # CPU/HIP builds cannot claim CUDA merely because cuda.is_available() is true.
        candidates = (["cuda:%d" % i for i in range(torch.cuda.device_count())]
                      if not getattr(torch.version, "hip", None) and getattr(torch.version, "cuda", None)
                      and torch.cuda.is_available() else [])
    failures = []
    for candidate in candidates:
        try:
            probe = _cuda_probe(torch, candidate, dtype, operators=False)
            device_admission(torch, candidate, max(device_required_bytes, 8 * 1024**2),
                             device_memory_budget_bytes, phase="selection-before-operator-probe")
            probe = _cuda_probe(torch, candidate, dtype)
            admission = device_admission(torch, candidate, device_required_bytes,
                                         device_memory_budget_bytes, phase="selection-before-weights")
            return {"requested": execution_device, "device": candidate, "probe": probe,
                    "host_admission": host, "device_admission": admission, "fallback": False,
                    "cross_device_validation": "unknown",
                    "loader_policy": "full CPU native load then dtype-preserving CUDA transfer"}
        except DeviceRejected as exc:
            if request != "auto":
                raise
            failures.append({"device": candidate, "reason": str(exc)})
    host = host_admission(host_required_bytes, memory_budget_bytes, total_process_cap=total_process_cap)
    return {"requested": execution_device, "device": "cpu", "host_admission": host,
            "cuda_candidates_rejected": failures, "fallback": True, "cross_device_validation": "unknown"}


def runtime_metadata(torch, device, model=None):
    """Actual settings/allocations; GPU peaks and process RSS are separate measures."""
    device = str(device)
    result = {"device": device, "torch_version": str(torch.__version__),
              "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
              "cross_device_validation": "unknown", "cuda_execution_verified": device.startswith("cuda:"),
              "gpu_validation_status": "executed_on_selected_device" if device.startswith("cuda:") else "GPU_UNVERIFIED"}
    if model is not None:
        result.update(parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                      parameter_devices=sorted({str(p.device) for p in model.parameters()}),
                      cache_config=getattr(model.config, "use_cache", None),
                      attention_implementation=getattr(model.config, "_attn_implementation", None))
    if device.startswith("cuda:"):
        result.update(float32_matmul_precision=torch.get_float32_matmul_precision(),
                      cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                      tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                      tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
                      cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                      cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                      gpu_allocated_bytes=int(torch.cuda.memory_allocated(device)),
                      gpu_reserved_bytes=int(torch.cuda.memory_reserved(device)),
                      gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                      gpu_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
                      gpu_peak_scope="process allocator since last external peak reset; not isolated",
                      allocator_release_guaranteed=False)
    return result


def release_cuda_cache(torch, device):
    """Best effort only; preserve caller current-device state and grant zero credit."""
    if str(device).startswith("cuda:"):
        try:
            with torch.cuda.device(int(str(device).split(":")[1])):
                torch.cuda.empty_cache()
        except Exception:
            return False
    return True


def move_preserving_dtype(model, device):
    """Move without recasting native F32 exceptions or F32 factors."""
    before = {name: (p.dtype, tuple(p.shape)) for name, p in model.named_parameters()}
    model.to(device=device)
    after = {name: (p.dtype, tuple(p.shape)) for name, p in model.named_parameters()}
    if before != after or any(str(p.device) != str(device) for p in model.parameters()):
        raise DeviceRejected("Device transfer changed parameter dtype/shape or placement")
    return model
