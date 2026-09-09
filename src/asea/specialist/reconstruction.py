"""Local safetensors -> smaller native HF weights, never a serving-time mask.

Only imports the optional ML stack inside reconstruct(). Source files are read-only.
This is structural initialization, not capability certification or promotion.
"""
import ctypes
from contextlib import contextmanager
import errno
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile

from asea.artifacts import Blocked, atomic_json, file_hash, model_inventory, safe_file, safe_path


class ReconstructionBlocked(Blocked):
    """Unsupported architecture, unsafe input, or resource admission failure."""


_DTYPES = {"bfloat16": 2, "float32": 4}
_SIZES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}


def _unique_json(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ReconstructionBlocked("duplicate JSON key: " + key)
        value[key] = item
    return value


def _json(path, limit=2 * 1024 * 1024):
    path = safe_file(path)
    if path.stat().st_size > limit:
        raise ReconstructionBlocked("JSON input exceeds bounded size: " + str(path))
    try:
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ReconstructionBlocked("JSON input exceeds bounded size")
        def finite_float(value):
            number = float(value)
            if not math.isfinite(number):
                raise ReconstructionBlocked("nonfinite JSON number: " + value)
            return number
        return json.loads(raw, object_pairs_hook=_unique_json, parse_float=finite_float,
                          parse_constant=lambda x: (_ for _ in ()).throw(ReconstructionBlocked("nonfinite JSON: " + x)))
    except (ValueError, UnicodeError) as exc:
        raise ReconstructionBlocked("invalid JSON: " + str(path)) from exc


def _headers(root, inventory):
    """Read shapes, never tensors; reject loose/unindexed extra weight files."""
    single, index = "model.safetensors", "model.safetensors.index.json"
    if single in inventory and index in inventory:
        raise ReconstructionBlocked("ambiguous single-file and sharded model")
    weight_map = None
    if index in inventory:
        weight_map = _json(root / index).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ReconstructionBlocked("invalid weight map")
        names = set(weight_map.values())
    elif single in inventory:
        names = {single}
    else:
        raise ReconstructionBlocked("native model.safetensors or shard index required")
    if names != {n for n in inventory if n.endswith(".safetensors")}:
        raise ReconstructionBlocked("unindexed or extra safetensors files")
    tensors = {}
    for name in sorted(names):
        path = safe_file(root / name)
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ReconstructionBlocked("truncated safetensors header")
            size = struct.unpack("<Q", prefix)[0]
            if not 2 <= size <= 16 * 1024 * 1024:
                raise ReconstructionBlocked("safetensors header exceeds bounded size")
            try:
                raw = stream.read(size)
                if len(raw) != size:
                    raise ReconstructionBlocked("truncated safetensors header")
                header = json.loads(raw, object_pairs_hook=_unique_json)
            except (ValueError, UnicodeError) as exc:
                raise ReconstructionBlocked("invalid safetensors header") from exc
        if not isinstance(header, dict):
            raise ReconstructionBlocked("safetensors header must be an object")
        data_size = path.stat().st_size - 8 - size
        spans = []
        for key, entry in header.items():
            if key == "__metadata__":
                continue
            if not isinstance(entry, dict):
                raise ReconstructionBlocked("safetensors tensor entry must be an object")
            shape = entry.get("shape")
            dtype = entry.get("dtype")
            offsets = entry.get("data_offsets")
            if (key in tensors or not isinstance(shape, list) or len(shape) > 8
                    or any(type(n) is not int or n < 0 for n in shape) or dtype not in _SIZES
                    or not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(n) is not int for n in offsets)):
                raise ReconstructionBlocked("unsupported safetensors tensor metadata: " + key)
            numel = math.prod(shape)
            start, end = offsets
            if start < 0 or end < start or end > data_size or end - start != numel * _SIZES[dtype]:
                raise ReconstructionBlocked("invalid safetensors tensor span: " + key)
            if weight_map is not None and weight_map.get(key) != name:
                raise ReconstructionBlocked("header disagrees with shard index: " + key)
            spans.append((start, end))
            tensors[key] = {"shape": shape, "dtype": dtype, "numel": numel, "file": name}
        cursor = 0
        for start, end in sorted(spans):
            if start != cursor:
                raise ReconstructionBlocked("non-contiguous safetensors data")
            cursor = end
        if cursor != data_size:
            raise ReconstructionBlocked("safetensors trailing or missing data")
    if weight_map is not None and set(weight_map) != set(tensors):
        raise ReconstructionBlocked("shard index has missing tensor keys")
    if not tensors:
        raise ReconstructionBlocked("empty weights")
    return tensors


def _available_ram():
    """Observed host and every visible cgroup ancestor's remaining RAM; no override.

    Read both cgroup layouts and resolve the process hierarchy from mountinfo.
    An inaccessible enclosing host constraint cannot be inferred or bypassed.
    """
    values = []
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                values.append(int(line.split()[1]) * 1024)
    except (OSError, ValueError):
        pass
    locations = {(Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup"), True),
                 (Path("/sys/fs/cgroup/memory"), Path("/sys/fs/cgroup/memory"), False)}
    try:
        groups = [line.split(":", 2) for line in Path("/proc/self/cgroup").read_text().splitlines()]
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            left, right = line.split(" - ", 1)
            fields, fs = left.split(), right.split()
            if fs[0] not in ("cgroup", "cgroup2"):
                continue
            v2 = fs[0] == "cgroup2"
            for _, controllers, relative in groups:
                if (v2 and controllers == "") or (not v2 and "memory" in controllers.split(",") and "memory" in fs[-1].split(",")):
                    mount, root = Path(fields[4]), Path(fields[3])
                    group = Path(relative)
                    if group == root or root in group.parents:
                        locations.add((mount / group.relative_to(root), mount, v2))
    except (OSError, ValueError, IndexError):
        pass
    for location, mount, v2 in locations:
        while location == mount or mount in location.parents:
            limit_name, usage_name = (("memory.max", "memory.current") if v2 else
                                      ("memory.limit_in_bytes", "memory.usage_in_bytes"))
            try:
                limit = (location / limit_name).read_text().strip()
                if limit != "max":
                    used = int((location / usage_name).read_text())
                    values.append(max(0, int(limit) - used))
            except FileNotFoundError:
                pass
            except (OSError, ValueError) as exc:
                raise ReconstructionBlocked("Cannot establish cgroup memory headroom") from exc
            if location == mount:
                break
            location = location.parent
    if not values or min(values) <= 0:
        raise ReconstructionBlocked("Cannot establish available memory headroom")
    return min(values)


def _memory_budget(memory_budget_bytes=None, available=None):
    if memory_budget_bytes is not None and (type(memory_budget_bytes) is not int or memory_budget_bytes <= 0):
        raise ReconstructionBlocked("memory_budget_bytes must be a positive integer or None")
    available = _available_ram() if available is None else available
    reserve = max(256 * 1024**2, available // 10)
    effective = max(0, available - reserve)
    if memory_budget_bytes is not None:
        effective = min(effective, memory_budget_bytes)
    return {"requested_memory_budget_bytes": memory_budget_bytes, "available_host_bytes": available,
            "safety_headroom_bytes": reserve, "limit_bytes": effective,
            "budget_policy": "min(optional operator ceiling, observed host/cgroup available minus max(256MiB,10%)); CPU only",
            "estimate_not_guarantee": True}


def _validate_config(config):
    if not isinstance(config, dict):
        raise ReconstructionBlocked("native config must be an object")
    for key in ("vocab_size", "hidden_size", "intermediate_size", "d_model", "d_ff", "d_kv", "num_heads", "num_attention_heads", "num_key_value_heads", "num_experts", "max_position_embeddings"):
        if key in config and (type(config[key]) is not int or not 0 < config[key] <= 2 ** 20):
            raise ReconstructionBlocked("invalid or excessive config dimension: " + key)
    for key in ("num_hidden_layers", "num_layers", "num_decoder_layers"):
        if key in config and config[key] is not None and (type(config[key]) is not int or not 0 < config[key] <= 256):
            raise ReconstructionBlocked("invalid or excessive layer count: " + key)
    architectures = {"qwen2": "Qwen2ForCausalLM", "switch_transformers": "SwitchTransformersForConditionalGeneration", "t5": "T5ForConditionalGeneration"}
    if config.get("model_type") not in architectures:
        raise ReconstructionBlocked("unsupported native model_type")
    if config.get("architectures") not in (None, [architectures[config["model_type"]]]):
        raise ReconstructionBlocked("unsupported native architecture")
    if any(config.get(k) is not None for k in ("auto_map", "quantization_config", "compression_config", "base_model_name_or_path", "peft_type")) or config.get("pruned_heads"):
        raise ReconstructionBlocked("custom, quantized, PEFT, or pruned configuration unsupported")
    if config.get("num_experts", 1) > 256:
        raise ReconstructionBlocked("unsupported excessive native expert count")
    if config.get("model_type") == "switch_transformers":
        router_dtype = config.get("router_dtype", "float32")
        if not isinstance(router_dtype, str) or router_dtype not in _DTYPES:
            raise ReconstructionBlocked("unsupported native Switch router_dtype; expected float32 or bfloat16")
    if config.get("transformers_version") is not None:
        from packaging.version import Version, InvalidVersion
        try:
            saved = Version(config["transformers_version"])
            installed = Version(importlib.metadata.version("transformers"))
            if saved > installed or saved.major != installed.major:
                raise ReconstructionBlocked("unsupported config transformers_version")
        except (InvalidVersion, TypeError) as exc:
            raise ReconstructionBlocked("invalid config transformers_version") from exc


def _release_file_cache(path, sync=False):
    """Best-effort eviction of this read file's clean cache, never a RAM-limit change.

    Full integrity hashing and large disk banks otherwise leave reclaimable pages
    charged to a small cgroup. Failure is harmless: observed-headroom checks stay
    conservative and can reject. No input file contents or metadata are rewritten.
    """
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return False
    fd = os.open(str(safe_file(path)), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if sync:
            os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def _artifact_preflight(path):
    """Shared admission BEFORE Auto config/tokenizer or real checkpoint loaders."""
    root = safe_path(path)
    inventory = model_inventory(root)
    for name in inventory:
        if Path(name).name == "adapter_config.json":
            raise ReconstructionBlocked("PEFT/base-model indirection unsupported")
        if name.endswith(".json"):
            # Large tokenizer vocabulary JSON is permitted, but still bounded.
            value = _json(root / name, 64 * 1024**2 if name == "tokenizer.json" else 16 * 1024**2)
            if isinstance(value, dict) and any(value.get(k) is not None for k in ("auto_map", "base_model_name_or_path", "peft_type")):
                raise ReconstructionBlocked("unsafe auto_map or base-model indirection: " + name)
    config = _json(root / "config.json")
    _validate_config(config)
    headers = _headers(root, inventory)
    for name in inventory:
        if name.endswith(".safetensors"):
            _release_file_cache(root / name)
    return inventory, config, headers


def _native_meta(config, headers, torch):
    from transformers import (Qwen2Config, Qwen2ForCausalLM, SwitchTransformersConfig,
                              SwitchTransformersForConditionalGeneration, T5Config, T5ForConditionalGeneration)
    classes = {"qwen2": (Qwen2Config, Qwen2ForCausalLM),
               "switch_transformers": (SwitchTransformersConfig, SwitchTransformersForConditionalGeneration),
               "t5": (T5Config, T5ForConditionalGeneration)}
    config_class, model_class = classes[config["model_type"]]
    native = config_class.from_dict(config)
    native._attn_implementation = "eager"
    with torch.device("meta"):
        meta = model_class(native)
    _strict_header_match(meta, headers)
    return native, model_class, meta


def _restore_f32_routers(model, root, headers, torch):
    """Restore original F32 values, NOT an upcast of rounded global-BF16 values."""
    from safetensors import safe_open
    routers = {k: v for k, v in headers.items() if k.endswith("router.classifier.weight")}
    restored = []
    total = sum(v["numel"] * 4 for v in routers.values() if v["dtype"] == "F32")
    if total > 64 * 1024**2:
        raise ReconstructionBlocked("source F32 router restoration exceeds bounded 64MiB")
    for key, entry in sorted(routers.items()):
        if entry["dtype"] != "F32":
            continue
        if entry["numel"] * 4 > 16 * 1024**2:
            raise ReconstructionBlocked("source router tensor exceeds bounded 16MiB")
        with safe_open(str(safe_file(root / entry["file"])), framework="pt", device="cpu") as handle:
            original = handle.get_tensor(key)
        parameter = model.get_parameter(key)
        if list(parameter.shape) != entry["shape"] or original.dtype != torch.float32:
            raise ReconstructionBlocked("invalid source router precision restoration")
        digest = hashlib.sha256(memoryview(original.contiguous().numpy())).hexdigest()
        parameter.data = original.to(device=parameter.device)  # original read-only mmap storage, no second F32 copy
        restored.append({"key": key, "source_file": entry["file"], "dtype": "F32", "tensor_sha256": digest,
                         "bytes": entry["numel"] * 4})
    return {"policy": "preserve/restore header-proven original F32 router values directly from source before any forward; no rounded upcast",
            "restored": restored, "source_non_f32_routers": {k: v["dtype"] for k, v in routers.items() if v["dtype"] != "F32"}}


def _rss_bytes():
    """Current resident pages, not lifetime ru_maxrss; unavailable means no credit."""
    try:
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def _load_plan(meta, headers, dtype):
    """Exact storage after canonical native initialization, before any freeze hash."""
    if not isinstance(dtype, str) or dtype not in _DTYPES:
        raise ReconstructionBlocked("unsupported native load dtype")
    aliases = _strict_header_match(meta, headers)
    if sum(p.numel() for p in meta.parameters()) != sum(h["numel"] for h in headers.values()):
        raise ReconstructionBlocked("native/header unique parameter count disagreement")
    keep = set(getattr(meta, "_keep_in_fp32_modules", None) or [])
    router_dtype, routers = None, {}
    if meta.config.model_type == "switch_transformers":
        import torch
        from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersTop1Router
        router_dtype = meta.config.router_dtype
        if not isinstance(router_dtype, str) or router_dtype not in _DTYPES:
            raise ReconstructionBlocked("unsupported native Switch router_dtype; expected float32 or bfloat16")
        for name, router in meta.named_modules():
            if not isinstance(router, SwitchTransformersTop1Router):
                continue
            if (type(router) is not SwitchTransformersTop1Router or type(router.classifier) is not torch.nn.Linear
                    or router.dtype != getattr(torch, router_dtype)):
                raise ReconstructionBlocked("unsupported native Switch router classifier/dtype")
            # Native _cast_classifier casts the WHOLE Linear on first forward:
            # encoder and decoder, weight and optional bias, regardless of source dtype.
            for parameter_name, parameter in router.classifier.named_parameters():
                key = aliases[name + ".classifier." + parameter_name]
                source_dtype = headers[key]["dtype"]
                if source_dtype not in ("F32", "BF16"):
                    raise ReconstructionBlocked("unsupported native Switch router source dtype: " + key)
                if source_dtype == "F32" and router_dtype != "float32":
                    raise ReconstructionBlocked("native router_dtype would round original F32 router values: " + key)
                routers[key] = {"source_dtype": source_dtype, "loaded_dtype": router_dtype,
                                "numel": parameter.numel()}
        if {aliases[n] for n in aliases if ".router.classifier." in n} != set(routers):
            raise ReconstructionBlocked("unrecognized native Switch router parameters")
    targets = {}
    for key, entry in headers.items():
        names = [alias for alias, stored in aliases.items() if stored == key]
        retained_f32 = dtype == "bfloat16" and any(keep.intersection(n.split(".")) for n in names)
        targets[key] = router_dtype if key in routers else ("float32" if retained_f32 else dtype)
    return {"strategy": "explicit_safetensors_tensor_meta_assign", "target_dtypes": targets,
            "native_keep_in_fp32_modules": sorted(keep), "aliases": aliases,
            "native_router_dtype": router_dtype, "native_router_tensors": routers,
            "native_router_policy": "canonical classifier storage before freeze/hash; original F32 or exact source BF16 values; no forward-time cast allocation"}


def _load_source(meta, root, headers, plan, torch):
    """One tensor .to at a time; same-dtype CPU .to aliases the mmap, no clone.

    The meta constructor has CPU nonpersistent buffers. No initialized random
    parameter may survive: every unique native parameter is assigned from a header.
    """
    from safetensors import safe_open
    aliases = plan["aliases"]
    if any(not isinstance(target, str) or target not in _DTYPES for target in plan["target_dtypes"].values()):
        raise ReconstructionBlocked("unsupported planned native load dtype")
    for key, entry in plan["native_router_tensors"].items():
        if plan["target_dtypes"][key] != entry["loaded_dtype"] or headers[key]["dtype"] != entry["source_dtype"]:
            raise ReconstructionBlocked("native router load plan disagreement: " + key)
    by_storage = {key: [alias for alias, stored in aliases.items() if stored == key] for key in headers}
    for shard in sorted({entry["file"] for entry in headers.values()}):
        with safe_open(str(safe_file(root / shard)), framework="pt", device="cpu") as handle:
            for key in sorted(k for k, h in headers.items() if h["file"] == shard):
                original = handle.get_tensor(key)
                source_dtype = {"BF16": torch.bfloat16, "F16": torch.float16,
                                "F32": torch.float32, "F64": torch.float64}[headers[key]["dtype"]]
                if list(original.shape) != headers[key]["shape"] or original.dtype != source_dtype:
                    raise ReconstructionBlocked("source tensor changed after header admission: " + key)
                value = original.to(dtype=getattr(torch, plan["target_dtypes"][key]), device="cpu")
                parameter = torch.nn.Parameter(value, requires_grad=False)
                for alias in by_storage[key]:
                    module_name, name = alias.rsplit(".", 1) if "." in alias else ("", alias)
                    module = meta.get_submodule(module_name) if module_name else meta
                    if name not in module._parameters:
                        raise ReconstructionBlocked("non-parameter source tensor unsupported: " + alias)
                    setattr(module, name, parameter)
                del original, value, parameter
    meta.tie_weights()
    if any(p.is_meta for p in list(meta.parameters()) + list(meta.buffers())):
        raise ReconstructionBlocked("uninitialized meta tensor after source assignment")
    for alias, key in aliases.items():
        if meta.get_parameter(alias).dtype != getattr(torch, plan["target_dtypes"][key]):
            raise ReconstructionBlocked("materialized native dtype plan disagreement: " + alias)
    return meta


def _admit(headers, config, dtype, max_length, memory_budget_bytes=None, *, load_plan=None,
           runtime_rss_bytes=0, retention=0.75):
    numel = sum(item["numel"] for item in headers.values())
    proven = load_plan is not None and load_plan.get("strategy") == "explicit_safetensors_tensor_meta_assign"
    targets = load_plan["target_dtypes"] if proven else {}
    loaded_bytes = sum(item["numel"] * _DTYPES[targets.get(key, dtype)] for key, item in headers.items())
    source_bytes = sum(item["numel"] * _SIZES.get(item.get("dtype"), _DTYPES[dtype]) for item in headers.values())
    shards = {}
    cast_bytes = largest_cast_tensor = 0
    for key, item in headers.items():
        size = item["numel"] * _SIZES.get(item.get("dtype"), _DTYPES[dtype])
        shards[item.get("file", "unknown")] = shards.get(item.get("file", "unknown"), 0) + size
        if item.get("dtype") != {"bfloat16": "BF16", "float32": "F32"}[targets.get(key, dtype)]:
            cast_bytes += size
            largest_cast_tensor = max(largest_cast_tensor, size)
    # Cast source pages can remain mapped by another same-shard tensor. Until
    # reclamation is guaranteed, reserve ALL cast-source pages, not only one shard.
    transient = cast_bytes if proven else max(loaded_bytes, source_bytes)
    if not proven:
        # Unknown native policy: conservatively allow all parameters to remain F32.
        loaded_bytes = max(loaded_bytes, numel * 4)
        transient = max(transient, loaded_bytes)
        targets = {key: "float32" for key in headers}
    vocab = config.get("vocab_size", 0)
    width = config.get("intermediate_size", config.get("d_ff", 0))
    hidden = config.get("hidden_size", config.get("d_model", 0))
    heads = config.get("num_attention_heads", config.get("num_heads", 1))
    # Full native logits (up to F32), finite mask, attention softmax and layer
    # temporaries. No KV cache, autograd, or simultaneous reference logits.
    forward = max_length * (vocab * 8 + width * 32 + hidden * 32) + max_length**2 * heads * 16
    layer_cast = hidden * width * 12  # up to three F32 MLP matrices
    replacement = 0
    if config.get("model_type") == "qwen2":
        kept = int(math.floor(width * retention / 64)) * 64
        replacement = config.get("num_hidden_layers", 0) * hidden * max(0, kept) * 3 * _DTYPES[dtype]
    # Mmap lifetime spans the shard: pruned original pages may remain resident
    # alongside ALL replacement MLPs. Do not pretend only one replacement survives.
    router_growth = sum(h["numel"] * max(0, 4 - _DTYPES[targets.get(k, dtype)])
                        for k, h in headers.items() if "router.classifier." in k)
    serialization = max(min(loaded_bytes, 256 * 1024**2),
                        max((h["numel"] * _DTYPES[targets.get(k, dtype)] for k, h in headers.items()), default=0))
    workspace = replacement + max(forward, layer_cast, serialization) + router_growth
    runtime_remaining = max(0, 512 * 1024**2 - runtime_rss_bytes)
    estimate = loaded_bytes + transient + workspace + runtime_remaining
    budget = _memory_budget(memory_budget_bytes)
    if estimate > budget["limit_bytes"]:
        raise ReconstructionBlocked("memory admission refused before model allocation: estimated %d bytes > effective %d bytes" % (estimate, budget["limit_bytes"]))
    return {**budget, "estimated_peak_bytes": estimate, "header_numel": numel,
            "header_loaded_bytes": loaded_bytes, "source_payload_bytes": source_bytes,
            "load_strategy": load_plan["strategy"] if proven else "conservative_unknown_loader",
            "native_keep_in_fp32_modules": load_plan.get("native_keep_in_fp32_modules", []) if proven else None,
            "target_dtype_counts": {d: sum(h["numel"] for k, h in headers.items() if targets.get(k, dtype) == d) for d in _DTYPES},
            "dtype_exceptions": {k: {"source_dtype": h.get("dtype"), "loaded_dtype": targets[k], "numel": h["numel"]}
                                 for k, h in headers.items() if proven and targets[k] != dtype},
            "cast_tensor_plan": [{"key": k, "source_dtype": h["dtype"], "loaded_dtype": targets[k],
                                  "source_bytes": h["numel"] * _SIZES[h["dtype"]],
                                  "target_bytes": h["numel"] * _DTYPES[targets[k]]}
                                 for k, h in headers.items() if proven and h["dtype"] !=
                                 {"bfloat16": "BF16", "float32": "F32"}[targets[k]]],
            "forward_workspace_bytes": forward, "layer_cast_workspace_bytes": layer_cast,
            "serialization_workspace_bytes": serialization, "router_upcast_growth_bytes": router_growth,
            "largest_shard_bytes": max(shards.values(), default=0), "largest_cast_tensor_bytes": largest_cast_tensor,
            "load_cast_transient_bytes": transient, "cast_policy": "all cast-source pages; shard reclamation not assumed",
            "replacement_mlp_bytes": replacement, "future_workspace_bytes": workspace,
            "post_import_rss_bytes": runtime_rss_bytes, "runtime_remaining_bytes": runtime_remaining,
            "formula": "future unique source storage + cast-source pages + replacement/workspace + max(0,512MiB-postimport RSS); compare incremental bytes to available"}


def _future_admit(admission, stage, memory_budget_bytes, *, replacements_allocated=False):
    """Weights already loaded are in memory.current/RSS; charge only future work."""
    budget = _memory_budget(memory_budget_bytes)
    future = admission["future_workspace_bytes"] + admission["runtime_remaining_bytes"]
    if replacements_allocated:
        future -= admission["replacement_mlp_bytes"]
    measurement = {"stage": stage, "rss_bytes": _rss_bytes(), "future_incremental_bytes": future, **budget}
    admission.setdefault("stage_measurements", []).append(measurement)
    if future > budget["limit_bytes"]:
        raise ReconstructionBlocked("memory admission refused at %s: future workspace %d > effective %d" % (stage, future, budget["limit_bytes"]))


def _calibration(path, max_samples):
    value = _json(path, 16 * 1024 * 1024)
    if not isinstance(value, dict) or not isinstance(value.get("samples"), list) or not value["samples"]:
        raise ReconstructionBlocked("calibration requires nonempty {samples: [{prompt, response?}]}")
    rows = value["samples"]
    if len(rows) > 10000:
        raise ReconstructionBlocked("calibration exceeds 10000 records")
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("prompt"), str)
                or not row["prompt"].strip() or not isinstance(row.get("response", ""), str)):
            raise ReconstructionBlocked("calibration prompt must be nonempty text; response must be text")
    return rows


def _strict_header_match(meta_model, headers):
    """Only missing tied aliases are legal; no unmatched or shape-mismatched keys."""
    expected = meta_model.state_dict(keep_vars=True)
    extras = set(headers) - set(expected)
    if extras:
        raise ReconstructionBlocked("unmatched source keys: " + repr(sorted(extras)))
    groups = {}
    for key, tensor in expected.items():
        groups.setdefault(id(tensor), []).append(key)
        if key in headers and list(tensor.shape) != headers[key]["shape"]:
            raise ReconstructionBlocked("source shape mismatch: " + key)
    missing = []
    for keys in groups.values():
        stored = set(keys).intersection(headers)
        if len(stored) > 1 and len(keys) > 1:
            raise ReconstructionBlocked("redundant tied-alias storage is unsupported; ambiguous source provenance: " + repr(sorted(stored)))
        if not stored:
            missing.extend(keys)
    if missing:
        raise ReconstructionBlocked("missing source keys (random initialization forbidden): " + repr(sorted(missing)))
    return {key: next(candidate for candidate in groups[id(tensor)] if candidate in headers)
            for key, tensor in expected.items()}


def _count(model):
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "tensor_bytes": sum(p.numel() * p.element_size() for p in model.parameters())}


def _calibration_encoding(tokenizer, row, family, max_length):
    # Runtime import avoids a module-level reconstruction/recovery cycle.
    from .recovery import _encode_details, _json_hash, _encoding_contract, RecoveryRejected
    causal = family == "qwen2"
    try:
        if not causal or row.get("response"):
            effective = dict(row, response=row.get("response") or row["prompt"])
            return _encode_details(effective, tokenizer, "causal" if causal else "seq2seq", max_length)
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                                  tokenize=False, add_generation_prompt=True)
        else:
            text = row["prompt"] + "\n"
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if not ids or len(ids) > max_length:
            raise ReconstructionBlocked("Complete token length exceeds calibration max_length (sample %s); no truncation" % row.get("id", "<unnamed>"))
        return ({"input_ids": ids, "attention_mask": [1] * len(ids)},
                {"id": row.get("id"), "input_tokens": len(ids), "label_tokens": 0,
                 "input_ids_sha256": _json_hash(ids), "prompt_only": True,
                 "template_sha256": _encoding_contract(tokenizer, "causal")["template_sha256"]})
    except RecoveryRejected as exc:
        raise ReconstructionBlocked(str(exc)) from exc


def _batches(tokenizer, rows, family, config, max_length, torch):
    for row in rows:
        example, _ = _calibration_encoding(tokenizer, row, family, max_length)
        batch = {k: torch.tensor([v], dtype=torch.long) for k, v in example.items() if k != "labels"}
        if family != "qwen2":
            start = config.decoder_start_token_id
            if start is None:
                raise ReconstructionBlocked("Switch/T5 decoder_start_token_id must be explicit")
            batch["decoder_input_ids"] = torch.tensor([[start] + example["labels"][:-1]], dtype=torch.long)
            batch["decoder_attention_mask"] = torch.ones_like(batch["decoder_input_ids"])
        yield batch


@contextmanager
def _no_cache(model):
    """Explicit argument AND native config disable cache; restore on failure too."""
    previous = model.config.use_cache
    model.config.use_cache = False
    try:
        yield
    finally:
        model.config.use_cache = previous


def _logit_fingerprint(logits, torch):
    """Full native-vocabulary parity, streamed finite check, no F32 full copy.

    Digest equality is bitwise parity (stricter than torch.equal for signed zero),
    with the usual SHA-256 collision assumption, not a numerical tolerance test.
    """
    flat = logits.detach().contiguous().view(-1)
    for start in range(0, flat.numel(), 1024 * 1024):
        if not torch.isfinite(flat[start:start + 1024 * 1024]).all():
            raise ReconstructionBlocked("non-finite native calibration logits")
    return {"shape": list(logits.shape), "dtype": str(logits.dtype),
            "sha256": hashlib.sha256(memoryview(flat.view(torch.uint8).numpy())).hexdigest()}


def _observe(model, tokenizer, rows, family, max_length, scores, torch):
    """Native read-only hooks; never patch forward. Verify exact hooked/unhooked logits."""
    handles = []
    counts = {key: 0 for key in scores}
    batches = _batches(tokenizer, rows, family, model.config, max_length, torch)
    first = next(batches)
    with _no_cache(model), torch.inference_mode():
        reference_logits = model(**first, use_cache=False, output_hidden_states=False, output_attentions=False, return_dict=True).logits
        reference = _logit_fingerprint(reference_logits, torch)
        del reference_logits  # never hold two full input_length*vocab probes
        try:
            if family == "qwen2":
                for index, layer in enumerate(model.model.layers):
                    key = "model.layers.%d.mlp" % index
                    def capture(module, args, key=key):
                        # Input to native down_proj IS silu(gate_proj(x))*up_proj(x).
                        values = args[0].detach().float().reshape(-1, args[0].shape[-1])
                        scores[key].add_(values.abs().sum(dim=0))
                        counts[key] += values.shape[0]
                        return None
                    handles.append(layer.mlp.down_proj.register_forward_pre_hook(capture))
            else:
                for name, module in model.named_modules():
                    if name in scores:
                        def capture(module, args, output, key=name):
                            mask, probability, _ = output
                            scores[key].add_((mask.float() * probability.float()).sum(dim=(0, 1)))
                            counts[key] += mask.shape[0] * mask.shape[1]
                            return None
                        handles.append(module.router.register_forward_hook(capture))
            observed = model(**first, use_cache=False, output_hidden_states=False, output_attentions=False, return_dict=True).logits
            if reference != _logit_fingerprint(observed, torch):
                raise ReconstructionBlocked("native calibration hooks changed source outputs")
            del reference, observed
            for batch in batches:
                model(**batch, use_cache=False, output_hidden_states=False, output_attentions=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
    for key in scores:
        if not counts[key] or not torch.isfinite(scores[key]).all():
            raise ReconstructionBlocked("invalid or unobserved calibration scores: " + key)
        scores[key].div_(counts[key])
    return {"native_forward_unchanged": True, "hook_parity": "exact_logits_first_sample",
            "parity_comparison": "sequential full-vocabulary shape/dtype/SHA256 bitwise equality; finite logits",
            "cache_during_observation": False,
            "observed_tokens_by_layer": counts, "samples_used": len(rows)}


def _rank(score, count, torch):
    if not torch.isfinite(score).all():
        raise ReconstructionBlocked("non-finite importance scores")
    # Stable ties favour the original low channel/expert index, independent of RNG.
    return torch.sort(torch.argsort(score, descending=True, stable=True)[:count]).values


def _qwen(model, tokenizer, rows, method, retention, max_length, torch):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    width = model.config.intermediate_size
    kept = int(math.floor(width * retention / 64)) * 64
    if kept < 64 or kept >= width:
        raise ReconstructionBlocked("retention must produce a strictly smaller positive intermediate_size multiple of 64")
    if model.config.hidden_act != "silu":
        raise ReconstructionBlocked("Qwen2 requires native SiLU SwiGLU")
    scores = {}
    for index, layer in enumerate(model.model.layers):
        if type(layer.mlp) is not Qwen2MLP or any(getattr(layer.mlp, p).bias is not None for p in ("gate_proj", "up_proj", "down_proj")):
            raise ReconstructionBlocked("unsupported non-native Qwen2 MLP")
        scores["model.layers.%d.mlp" % index] = torch.zeros(width, dtype=torch.float32)
    verification = {"native_forward_unchanged": True, "hook_parity": "not_applicable_baseline", "samples_used": 0}
    if method == "activation":
        verification = _observe(model, tokenizer, rows, "qwen2", max_length, scores, torch)
    selected, provenance = {}, {}
    for key, value in model.state_dict().items():
        provenance[key] = {"source_key": key, "operation": "retained", "source_shape": list(value.shape), "output_shape": list(value.shape)}
    with torch.no_grad():
        for index, layer in enumerate(model.model.layers):
            name = "model.layers.%d.mlp" % index
            mlp = layer.mlp
            down_norm = torch.linalg.vector_norm(mlp.down_proj.weight.float(), dim=0)
            if method == "activation":
                score = scores[name] * down_norm
            elif method == "magnitude":
                score = torch.linalg.vector_norm(mlp.gate_proj.weight.float(), dim=1) * torch.linalg.vector_norm(mlp.up_proj.weight.float(), dim=1) * down_norm
            else:
                score = None
            indices = (_rank(score, kept, torch) if score is not None else
                       torch.div(torch.arange(kept) * width, kept, rounding_mode="floor"))
            selected[name] = indices.tolist()
            for projection, axis in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)):
                linear = getattr(mlp, projection)
                linear.weight = torch.nn.Parameter(linear.weight.detach().index_select(axis, indices).contiguous(), requires_grad=False)
                if axis == 0:
                    linear.out_features = kept
                else:
                    linear.in_features = kept
                key = name + "." + projection + ".weight"
                provenance[key].update(operation="index_select", axis=axis, selected_indices_ref=name, output_shape=list(linear.weight.shape))
            mlp.intermediate_size = kept
    model.config.intermediate_size = kept
    return model, selected, provenance, [], verification, {
        "source_intermediate_size": width, "output_intermediate_size": kept,
        "actual_channel_retention": kept / width,
        "score": {"activation": "mean_abs(native_down_proj_input) * L2(down_proj_column)",
                  "magnitude": "L2(gate_row) * L2(up_row) * L2(down_column)",
                  "uniform": "floor(i * source_width / retained_width)"}[method],
        "initialization": "structured_weight_pruning", "teacher_output_parity_claimed": False}


def _switch(model, tokenizer, rows, method, max_length, seed, torch):
    from transformers import T5Config, T5ForConditionalGeneration
    from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersSparseMLP, SwitchTransformersDenseActDense
    if model.config.dense_act_fn != "relu" or model.config.num_experts < 2:
        raise ReconstructionBlocked("Switch-to-T5 supports native ReLU with at least two experts only")
    sparse = {name: module for name, module in model.named_modules() if type(module) is SwitchTransformersSparseMLP}
    if not sparse:
        raise ReconstructionBlocked("Switch source has no sparse layers to reduce")
    scores = {name: torch.zeros(model.config.num_experts, dtype=torch.float32) for name in sparse}
    verification = {"native_forward_unchanged": True, "hook_parity": "not_applicable_baseline", "samples_used": 0}
    if method == "activation":
        verification = _observe(model, tokenizer, rows, "switch_transformers", max_length, scores, torch)
    chosen = {}
    for layer_index, (name, module) in enumerate(sparse.items()):
        if list(module.experts) != ["expert_%d" % i for i in range(model.config.num_experts)]:
            raise ReconstructionBlocked("unsupported Switch expert layout")
        for expert in module.experts.values():
            if type(expert) is not SwitchTransformersDenseActDense:
                raise ReconstructionBlocked("unsupported Switch expert class")
        if method == "magnitude":
            scores[name] = torch.tensor([float(torch.linalg.vector_norm(e.wi.weight.float()) * torch.linalg.vector_norm(e.wo.weight.float())) for e in module.experts.values()])
        if method == "uniform":
            chosen[name] = [(seed + layer_index) % model.config.num_experts]
        else:
            chosen[name] = _rank(scores[name], 1, torch).tolist()
    fields = ("vocab_size", "d_model", "d_kv", "d_ff", "num_layers", "num_decoder_layers", "num_heads",
              "relative_attention_num_buckets", "relative_attention_max_distance", "dropout_rate",
              "layer_norm_epsilon", "initializer_factor", "pad_token_id", "eos_token_id", "bos_token_id",
              "decoder_start_token_id", "tie_word_embeddings", "tie_encoder_decoder", "use_cache")
    config = T5Config(**{key: getattr(model.config, key) for key in fields}, feed_forward_proj="relu")
    # Meta constructor allocates no model weights; assign reuses the SAME retained
    # source tensors. There is never a second resident dense weight copy.
    with torch.device("meta"):
        target = T5ForConditionalGeneration(config)
    source_state = model.state_dict()
    mapped, provenance, removed = {}, {}, []
    for key, tensor in source_state.items():
        new_key = key
        sparse_name = next((name for name in sparse if key.startswith(name + ".")), None)
        if sparse_name is not None:
            suffix = key[len(sparse_name) + 1:]
            selected_prefix = "experts.expert_%d." % chosen[sparse_name][0]
            allowed_router = {"router.classifier.weight"}
            if model.config.router_bias:
                allowed_router.add("router.classifier.bias")
            allowed_experts = {"experts.expert_%d.%s.weight" % (i, p) for i in range(model.config.num_experts) for p in ("wi", "wo")}
            if suffix in allowed_router:
                removed.append({"source_key": key, "reason": "router_omitted"})
                continue
            if suffix not in allowed_experts:
                raise ReconstructionBlocked("unsupported Switch state key: " + key)
            if not suffix.startswith(selected_prefix):
                removed.append({"source_key": key, "reason": "unselected_expert"})
                continue
            new_key = sparse_name.rsplit(".mlp", 1)[0] + ".DenseReluDense." + suffix[len(selected_prefix):]
        elif ".mlp." in key:
            new_key = key.replace(".mlp.", ".DenseReluDense.")
        if new_key in mapped:
            raise ReconstructionBlocked("duplicate mapped target key: " + new_key)
        mapped[new_key] = tensor
        provenance[new_key] = {"source_key": key, "operation": "retained" if key == new_key else "renamed", "source_shape": list(tensor.shape), "output_shape": list(tensor.shape)}
    expected = target.state_dict()
    if set(expected) != set(mapped):
        raise ReconstructionBlocked("Switch/T5 key mapping mismatch; missing=%r unexpected=%r" % (sorted(set(expected) - set(mapped)), sorted(set(mapped) - set(expected))))
    if any(expected[key].shape != value.shape for key, value in mapped.items()):
        raise ReconstructionBlocked("Switch/T5 shape mapping mismatch")
    target.load_state_dict(mapped, strict=True, assign=True)
    target.tie_weights()
    if any(p.is_meta for p in target.parameters()):
        raise ReconstructionBlocked("uninitialized meta tensor after Switch conversion")
    target.generation_config = model.generation_config
    target.eval()
    return target, chosen, provenance, removed, verification, {
        "source_intermediate_size": model.config.d_ff, "output_intermediate_size": config.d_ff,
        "experts_per_sparse_layer": 1, "source_experts_per_sparse_layer": model.config.num_experts,
        "actual_expert_retention": 1 / model.config.num_experts,
        "retention_argument": "not_applied: fixed top1 family; no concat/expansion or new weights",
        "score": {"activation": "mean accepted router probability mass per expert (capacity mask applied)",
                  "magnitude": "L2(expert.wi) * L2(expert.wo)", "uniform": "(seed + sparse_layer_index) modulo num_experts"}[method],
        "initialization": "structural_initialized_requires_repair_and_evaluation",
        "teacher_output_parity_claimed": False,
        "router_caveat": "Native source routers use configured router_dtype (normally float32). Routers, input-dependent probabilities, capacity overflow and unselected experts are removed. Dense T5 does not preserve Switch forward gating or logits."}


def _publish(stage, destination):
    """Linux atomic rename with NOREPLACE, refusing overwrite even in a race."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise ReconstructionBlocked("atomic no-replace publication requires Linux renameat2")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(stage), -100, os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise ReconstructionBlocked("output appeared during reconstruction; refusing replacement")
        raise OSError(code, os.strerror(code), str(destination))
    fd = os.open(str(destination.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def reconstruct(source_dir, output_dir, calibration_path, *, family="auto", method="activation", retention=0.75, dtype="bfloat16", max_length=256, max_samples=32, seed=17, memory_budget_bytes=None) -> dict:
    """Reconstruct once on CPU; return the persisted reconstruction_manifest.json dict.

    family: auto | qwen2 | switch_transformers. method: activation | magnitude |
    uniform. Switch always retains exactly one expert per sparse layer. All inputs
    must be local; output must not exist. See docs/SPECIALIST_RECONSTRUCTION.md.
    """
    if family not in {"auto", "qwen2", "switch_transformers"}:
        raise ReconstructionBlocked("unsupported family: " + str(family))
    if method not in {"activation", "magnitude", "uniform"} or dtype not in _DTYPES:
        raise ReconstructionBlocked("unsupported method or dtype")
    if not isinstance(retention, (int, float)) or isinstance(retention, bool) or not math.isfinite(retention) or not 0 < retention < 1:
        raise ReconstructionBlocked("retention must be finite and strictly between zero and one")
    if type(max_length) is not int or not 1 <= max_length <= 2048 or type(max_samples) is not int or not 1 <= max_samples <= 4096 or type(seed) is not int:
        raise ReconstructionBlocked("invalid bounded calibration options or seed")
    source, output, calibration = safe_path(source_dir), safe_path(output_dir), safe_file(calibration_path)
    if not source.is_dir() or output.exists() or source == output or source in output.parents or output in source.parents:
        raise ReconstructionBlocked("source must exist; output must be new and disjoint from source")
    if not output.parent.is_dir():
        raise ReconstructionBlocked("output parent must already exist")
    inventory, raw_config, headers = _artifact_preflight(source)
    if not isinstance(raw_config, dict) or any(raw_config.get(k) for k in ("auto_map", "quantization_config", "compression_config", "pruned_heads")):
        raise ReconstructionBlocked("custom, quantized, or pruned-head configuration unsupported")
    detected = raw_config.get("model_type")
    if detected not in {"qwen2", "switch_transformers"} or (family != "auto" and family != detected):
        raise ReconstructionBlocked("unsupported family or family/config mismatch: " + str(detected))
    family = detected
    _memory_budget(memory_budget_bytes)  # validate before any allocation
    rows = _calibration(calibration, max_samples)
    calibration_hash = file_hash(calibration)
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM, SwitchTransformersConfig, SwitchTransformersForConditionalGeneration
    except ImportError as exc:
        raise ReconstructionBlocked("install the optional localmodels dependencies for reconstruction") from exc
    config_class, model_class = ((Qwen2Config, Qwen2ForCausalLM) if family == "qwen2" else (SwitchTransformersConfig, SwitchTransformersForConditionalGeneration))
    if raw_config.get("architectures") not in (None, [model_class.__name__]):
        raise ReconstructionBlocked("unsupported source architecture")
    config = config_class.from_dict(raw_config)
    config._attn_implementation = "eager"
    post_import_rss = _rss_bytes()
    from accelerate import init_empty_weights
    # Unlike torch.device(meta), this leaves small native rotary buffers on CPU.
    # Only parameters are empty; re-tie after accelerate's registration wrapper.
    with init_empty_weights(include_buffers=False):
        meta = model_class(config)
    meta.tie_weights()
    plan = _load_plan(meta, headers, dtype)
    alias_sources = plan["aliases"]
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, trust_remote_code=False)
    encoding_audit = [_calibration_encoding(tokenizer, row, family, max_length)[1] for row in rows]
    rows = rows[:max_samples]
    actual_length = max(max(e["input_tokens"], e["label_tokens"]) for e in encoding_audit)
    admission = _admit(headers, raw_config, dtype, actual_length, memory_budget_bytes,
                       load_plan=plan, runtime_rss_bytes=post_import_rss, retention=retention)
    admission["stage_measurements"] = [{"stage": "pre_load", "rss_bytes": _rss_bytes(),
                                       "future_incremental_bytes": admission["estimated_peak_bytes"]}]
    requested_dtype = getattr(torch, dtype)
    model = _load_source(meta, source, headers, plan, torch)
    del meta, plan
    if "generation_config.json" in inventory:
        model.generation_config = transformers.GenerationConfig.from_pretrained(str(source), local_files_only=True)
    router_precision = _restore_f32_routers(model, source, headers, torch)
    model.requires_grad_(False)
    model.eval()
    before = _count(model)
    admission["materialized_source_count"] = before
    if before["parameters"] != admission["header_numel"] or before["tensor_bytes"] != admission["header_loaded_bytes"]:
        raise ReconstructionBlocked("materialized source count disagrees with admission dtype/alias plan")
    _future_admit(admission, "post_load_pre_observation", memory_budget_bytes)
    if family == "qwen2":
        result = _qwen(model, tokenizer, rows, method, retention, max_length, torch)
    else:
        result = _switch(model, tokenizer, rows, method, max_length, seed, torch)
    model, selected, provenance, removed, verification, details = result
    del result
    gc.collect()
    after = _count(model)
    if after["parameters"] >= before["parameters"] or after["tensor_bytes"] >= before["tensor_bytes"]:
        raise ReconstructionBlocked("reconstruction did not physically reduce model weights")
    _future_admit(admission, "post_transform_pre_forward", memory_budget_bytes, replacements_allocated=True)
    output_state = model.state_dict()
    for key, value in provenance.items():
        value["source_storage_key"] = alias_sources[value["source_key"]]
        value["source_file"] = headers[value["source_storage_key"]]["file"]
        value["output_dtype"] = str(output_state[key].dtype).removeprefix("torch.")
    del output_state
    model.config._name_or_path = ""
    model.config.architectures = [type(model).__name__]
    model.config.torch_dtype = requested_dtype
    # Validate a real native forward of the final structure before serialization.
    with _no_cache(model), torch.inference_mode():
        batch = next(_batches(tokenizer, rows[:1], family, model.config, max_length, torch))
        logits = model(**batch, use_cache=False, output_hidden_states=False, output_attentions=False, return_dict=True).logits
        if not torch.isfinite(logits).all():
            raise ReconstructionBlocked("non-finite reconstructed model logits")
        verification["reconstructed_forward_finite"] = True
        del logits
    stage = Path(tempfile.mkdtemp(prefix="." + output.name + ".stage-", dir=str(output.parent)))
    try:
        model.save_pretrained(str(stage), safe_serialization=True, max_shard_size="256MB")
        tokenizer.save_pretrained(str(stage))
        admission["stage_measurements"].append({"stage": "post_save", "rss_bytes": _rss_bytes(),
                                                 "measurement": "current RSS, not sampled peak"})
        # Re-hash all source/calibration bytes before publishing: no mutation or
        # accidental serving dependency. Hashes are provenance, not copied teacher.
        if model_inventory(source) != inventory or file_hash(calibration) != calibration_hash:
            raise ReconstructionBlocked("source or calibration changed during reconstruction")
        output_inventory = model_inventory(stage)
        _strict_header_match(model, _headers(stage, output_inventory))
        verification["serialized_state_shapes_and_keys_strict"] = True
        source_weight_bytes = sum(inventory[name]["size"] for name in inventory if name.endswith(".safetensors"))
        output_weight_bytes = sum(output_inventory[name]["size"] for name in output_inventory if name.endswith(".safetensors"))
        manifest = {
            "schema_version": 1, "artifact_kind": "standalone_native_hf_specialist", "status": "RECONSTRUCTED_UNVALIDATED",
            "source": {"family": family, "architecture": model_class.__name__, "files": inventory, "config": raw_config},
            "output": {"architecture": type(model).__name__, "model_type": model.config.model_type,
                       "config": _json(stage / "config.json"), "files": output_inventory, "standalone": True,
                       "teacher_required_at_serve": False, "safe_serialization": True, "max_shard_size": "256MB"},
            "calibration": {"file": calibration.name, **calibration_hash, "samples_available_used": len(rows),
                            "max_samples": max_samples, "max_length": max_length,
                            "text_policy": "shared complete native recovery encoding; causal response when supplied, otherwise user generation prefix; no truncation",
                            "encoding_audit": encoding_audit, "actual_max_length": actual_length},
            "method": {"name": method, "requested_retention": retention, "dtype": dtype, "seed": seed, **details},
            "selected_indices": selected, "weights_provenance": provenance, "removed_source_tensors": removed,
            "counts": {"source": {**before, "safetensors_bytes": source_weight_bytes},
                       "output": {**after, "safetensors_bytes": output_weight_bytes}},
            "reduction": {"parameters_removed": before["parameters"] - after["parameters"],
                          "parameter_fraction": 1 - after["parameters"] / before["parameters"],
                          "tensor_bytes_removed": before["tensor_bytes"] - after["tensor_bytes"],
                          "safetensors_bytes_removed": source_weight_bytes - output_weight_bytes},
            "router_precision": router_precision, "memory_admission": admission, "verification": {**verification, "source_files_unchanged": True,
                "source_loading_strict": True, "capability_evaluated": False, "promotion_performed": False},
            "runtime": {"torch": torch.__version__, "transformers": transformers.__version__, "device": "cpu"},
        }
        atomic_json(stage / "reconstruction_manifest.json", manifest)
        safe_path(output)
        _publish(stage, output)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)
