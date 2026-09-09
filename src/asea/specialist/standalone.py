"""Trusted, local-only loader for teacher-independent factor-preserving specialists.

The root is NOT a Hugging Face native checkpoint. Never dispatch model code from
bundle metadata. inspect_bundle reads bounded JSON/headers and hashes, not weights.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import os
import sys
import json
import math
from pathlib import Path
import re
import struct
from typing import Any

from asea.artifacts import safe_path, safe_file
from .reconstruction import _artifact_preflight, _json, _unique_json
from .recovery import _file_hash, _require

SCHEMA = "silt_factor_preserving_v1"
PATHS = {"base": "base", "adapter": "adapter", "tokenizer": "base", "runner": "run_specialist.py"}
COMPONENT = "INCOMPLETE_BASE_REQUIRES_LOCAL_TRAINED_FACTORS"
TARGETS = {"qwen2": ["down_proj", "q_proj", "v_proj"], "t5": ["q", "v", "wo"]}
CLASSES = {"qwen2": "Qwen2ForCausalLM", "t5": "T5ForConditionalGeneration"}
ADAPTER_KEYS = {"base_model_name_or_path", "peft_type", "task_type", "r", "lora_alpha",
                "lora_dropout", "bias", "target_modules", "inference_mode"}
COUNT_KEYS = {"base_parameters", "factor_parameters", "total_parameters", "rank",
              "base_safetensors_bytes", "factor_safetensors_bytes", "total_safetensors_bytes",
              "original_teacher_parameters", "original_teacher_safetensors_bytes"}
# This is a convenience copy of a fixed trusted runner, not a plug-in loaded by API.
RUNNER_SOURCE = '''"""Requires installed SILT, Torch, Transformers, PEFT and safetensors."""
import argparse
from pathlib import Path
import torch
from asea.specialist.standalone import load_standalone

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(Path(__file__).resolve().parent))
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args()
    bundle = load_standalone(args.model)
    model, tokenizer = bundle.model, bundle.tokenizer
    causal = bundle.manifest["model_type"] == "qwen2"
    if causal:
        prefix = (tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}],
                  tokenize=False, add_generation_prompt=True) if tokenizer.chat_template
                  else args.prompt + "\\n")
    else:
        prefix = args.prompt
    inputs = tokenizer(prefix, add_special_tokens=not causal, truncation=False,
                       return_tensors="pt", return_token_type_ids=False)
    with torch.no_grad():
        ids = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
    if causal:
        ids = ids[:, inputs["input_ids"].shape[-1]:]
    print(tokenizer.decode(ids[0], skip_special_tokens=True))
'''


@dataclass
class StandaloneModel:
    model: Any
    tokenizer: Any
    manifest: dict
    base_path: Path


def _adapter_headers(path):
    """Validate a single adapter safetensors file, without materializing tensors."""
    path = safe_file(path)
    with path.open("rb") as stream:
        prefix = stream.read(8)
        _require(len(prefix) == 8, "Truncated adapter header")
        size = struct.unpack("<Q", prefix)[0]
        _require(2 <= size <= 16 * 1024**2, "Adapter header exceeds bound")
        raw = stream.read(size)
        _require(len(raw) == size, "Truncated adapter header")
    header = json.loads(raw, object_pairs_hook=_unique_json)
    _require(isinstance(header, dict), "Invalid adapter header")
    tensors, spans = {}, []
    for name, entry in header.items():
        if name == "__metadata__":
            _require(isinstance(entry, dict) and all(isinstance(k, str) and isinstance(v, str)
                     for k, v in entry.items()), "Invalid safetensors metadata")
            continue
        _require(isinstance(entry, dict) and set(entry) == {"dtype", "shape", "data_offsets"},
                 "Invalid adapter tensor schema")
        shape, offsets = entry["shape"], entry["data_offsets"]
        _require(entry["dtype"] == "F32" and isinstance(shape, list) and len(shape) == 2
                 and all(type(n) is int and n > 0 for n in shape), "Adapter requires F32 rank-2 tensors")
        _require(isinstance(offsets, list) and len(offsets) == 2
                 and all(type(n) is int and n >= 0 for n in offsets)
                 and offsets[1] - offsets[0] == math.prod(shape) * 4, "Invalid adapter offsets")
        spans.append(tuple(offsets))
        tensors[name] = {"shape": shape, "numel": math.prod(shape), "dtype": "F32"}
    cursor = 0
    for start, end in sorted(spans):
        _require(start == cursor, "Overlapping/gapped adapter spans")
        cursor = end
    _require(tensors and cursor == path.stat().st_size - 8 - size, "Adapter payload size mismatch")
    return tensors


def _inventory(root):
    files = {}
    _require(not any(p.is_symlink() for p in root.iterdir()), "Symlink bundle entry forbidden")
    _require(set(p.name for p in root.iterdir()) <=
             {"base", "adapter", "specialist_bundle.json", "run_specialist.py", "recovery_report.json"},
             "Unexpected bundle root entry")
    for directory in (root / "base", root / "adapter"):
        _require(directory.is_dir() and not directory.is_symlink(), "Missing/unsafe bundle component")
        for path in directory.iterdir():
            safe_file(path)
            _require(path.is_file() and not path.is_symlink(), "Nested/unsafe bundle component")
            if directory.name == "base":
                _require(path.name in TOKENIZER_ASSETS | {"config.json", "generation_config.json", "model.safetensors.index.json"}
                         or re.fullmatch(r"model(?:-[0-9]{5}-of-[0-9]{5})?\.safetensors", path.name),
                         "Unexpected native base asset")
            relative = path.relative_to(root).as_posix()
            files[relative] = {"sha256": _file_hash(path), "bytes": path.stat().st_size}
    runner = safe_file(root / PATHS["runner"])
    _require(runner.stat().st_size == len(RUNNER_SOURCE.encode()), "Unrecognized runner source size")
    with runner.open("rb") as stream:
        _require(stream.read(len(RUNNER_SOURCE.encode()) + 1) == RUNNER_SOURCE.encode(), "Unrecognized runner source")
    files[PATHS["runner"]] = {"sha256": _file_hash(runner), "bytes": runner.stat().st_size}
    if (root / "recovery_report.json").exists():
        _json(root / "recovery_report.json", 16 * 1024**2)  # audit only, never a load directive
    return files


def inspect_bundle(path) -> dict:
    """Return validated manifest; no model allocation, tensor loading, or source access.

    Hashes detect corruption, not authenticity. The optional recovery report and
    teacher provenance are audit only. Exact runtime versions are checked by load.
    """
    root = safe_path(path)
    _require(root.is_dir(), "Bundle directory required")
    manifest = _json(root / "specialist_bundle.json", 16 * 1024**2)
    required = {"schema", "export_mode", "artifact_kind", "paths", "model_type", "model_class",
                "dtype", "runtime_versions", "counts", "files", "teacher_source_provenance"}
    _require(isinstance(manifest, dict) and set(manifest) == required, "Unknown bundle schema fields")
    _require(manifest["schema"] == SCHEMA and manifest["export_mode"] == "factor_preserving"
             and manifest["artifact_kind"] == "native_base_plus_required_local_lora", "Unknown bundle kind")
    _require(manifest["paths"] == PATHS, "Bundle paths must be exact local relative paths")
    kind = manifest["model_type"]
    _require(kind in CLASSES and manifest["model_class"] == CLASSES[kind], "Unsupported bundle model class")
    _require(manifest["dtype"] in ("float32", "bfloat16"), "Unsupported bundle dtype")
    versions = manifest["runtime_versions"]
    _require(isinstance(versions, dict) and set(versions) == {"torch", "transformers", "peft", "safetensors"}
             and all(isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9.+_-]{1,100}", v)
                     for v in versions.values()), "Invalid runtime versions")
    files = _inventory(root)
    _require(manifest["files"] == files, "Bundle file hash/size/inventory mismatch")
    _require({p.name for p in (root / "adapter").iterdir()} ==
             {"adapter_config.json", "adapter_model.safetensors"}, "Unexpected adapter files")
    _, config, base_headers = _artifact_preflight(root / "base")
    _require(config["model_type"] == kind and config.get("silt_specialist_component") == COMPONENT,
             "Missing categorical incomplete-base marker")
    _require(config.get("_name_or_path", "") == "", "External base source path forbidden")
    # v1 records the compute dtype, not permission to recast a saved base. The
    # known T5 class keeps wo in F32; Qwen has no native F32 exception. Reject
    # conflicting headers before tensor reads, even if file hashes were updated.
    for key, entry in base_headers.items():
        expected_dtype = ("F32" if manifest["dtype"] == "float32" or
                          (kind == "t5" and "wo" in key.split(".")) else "BF16")
        _require(entry["dtype"] == expected_dtype,
                 "Bundle base header dtype disagrees with native manifest policy: " + key)
    adapter = _json(root / "adapter" / "adapter_config.json")
    _require(isinstance(adapter, dict) and set(adapter) == ADAPTER_KEYS, "Unknown adapter configuration fields")
    rank = adapter["r"]
    _require(type(rank) is int and 1 <= rank <= 128, "Invalid adapter rank")
    _require(adapter == {"base_model_name_or_path": "../base", "peft_type": "LORA",
              "task_type": "CAUSAL_LM" if kind == "qwen2" else "SEQ_2_SEQ_LM",
              "r": rank, "lora_alpha": 2 * rank, "lora_dropout": 0.0, "bias": "none",
              "target_modules": TARGETS[kind], "inference_mode": True},
             "Unsupported adapter options or external base_model_name_or_path")
    factors = _adapter_headers(root / "adapter" / "adapter_model.safetensors")
    expected = {}
    for key, entry in base_headers.items():
        if key.endswith(".weight") and key.split(".")[-2] in TARGETS[kind]:
            _require(len(entry["shape"]) == 2, "Nonlinear LoRA target")
            out_features, in_features = entry["shape"]
            prefix = "base_model.model." + key[:-len(".weight")]
            expected[prefix + ".lora_A.weight"] = [rank, in_features]
            expected[prefix + ".lora_B.weight"] = [out_features, rank]
    _require(expected and {k: v["shape"] for k, v in factors.items()} == expected,
             "Adapter keys/targets/rank shapes do not match native base")
    counts = manifest["counts"]
    _require(isinstance(counts, dict) and set(counts) == COUNT_KEYS and
             all(type(v) is int and v > 0 for v in counts.values()), "Invalid bundle counts")
    base_bytes = sum(v["bytes"] for k, v in files.items() if k.startswith("base/") and k.endswith(".safetensors"))
    factor_bytes = files["adapter/adapter_model.safetensors"]["bytes"]
    base_count, factor_count = sum(v["numel"] for v in base_headers.values()), sum(v["numel"] for v in factors.values())
    _require(all(counts[k] == v for k, v in {
        "base_parameters": base_count, "factor_parameters": factor_count, "rank": rank,
        "total_parameters": base_count + factor_count, "base_safetensors_bytes": base_bytes,
        "factor_safetensors_bytes": factor_bytes, "total_safetensors_bytes": base_bytes + factor_bytes}.items()),
        "Bundle counts disagree with validated headers/files")
    provenance = manifest["teacher_source_provenance"]
    _require(isinstance(provenance, dict) and set(provenance) ==
             {"audit_only", "model_type", "parameters", "safetensors_bytes", "files_sha256"}
             and provenance["audit_only"] is True and provenance["model_type"] in ("qwen2", "t5", "switch_transformers")
             and provenance["parameters"] == counts["original_teacher_parameters"]
             and provenance["safetensors_bytes"] == counts["original_teacher_safetensors_bytes"], "Invalid teacher audit provenance")
    _require(isinstance(provenance["files_sha256"], dict) and provenance["files_sha256"] and
             all(isinstance(k, str) and not Path(k).is_absolute() and ".." not in Path(k).parts
                 and "\\" not in k and isinstance(v, str) and re.fullmatch(r"[a-f0-9]{64}", v)
                 for k, v in provenance["files_sha256"].items()), "Invalid teacher audit hashes")
    return manifest


def load_standalone(path, dtype=None, local_only=True) -> StandaloneModel:
    """Load verified local base + native PEFT factors, never merge or fetch teacher.

    dtype=None preserves the recorded graph; an explicit dtype must match it.
    local_only=False is rejected. Return .model, .tokenizer, .manifest, .base_path.
    Only the recorded exact dependency versions are admitted for arithmetic parity.
    """
    _require(local_only is True, "Standalone loader is local-only")
    manifest = inspect_bundle(path)
    root = safe_path(path)
    import torch
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from .recovery import _recovery_inputs, _load_recovery_model
    actual = {name: importlib.metadata.version(name) for name in manifest["runtime_versions"]}
    _require(actual == manifest["runtime_versions"], "Exact bundle runtime versions required")
    requested = str(dtype).removeprefix("torch.") if dtype is not None else manifest["dtype"]
    _require(requested == manifest["dtype"], "dtype must match bundle; recasting changes trained arithmetic")
    _, config, headers = _artifact_preflight(root / "base")
    meta, plan = _recovery_inputs(config, headers, requested, torch)
    native_dtypes = {"float32": "F32", "bfloat16": "BF16"}
    _require(all(headers[key]["dtype"] == native_dtypes[target]
                 for key, target in plan["target_dtypes"].items()),
             "Bundle base dtype policy disagrees with known native loader plan")
    # Exactly the training loader, including T5's F32 wo and CPU rotary buffers.
    # HF from_pretrained(BF16) can silently round saved F32 wo tensors; it is not
    # interchangeable with this computation graph. No saved base tensor is cast.
    model = _load_recovery_model(meta, root / "base", headers, plan, torch)
    # PEFT reads model.name_or_path and overwrites its constructor config with it.
    # Supply only this verified local component, never a provenance/source path.
    model.name_or_path = str(root / "base")
    # Use only constructor values we validated; never pass an untrusted config to
    # AutoPeftModel or follow PEFT's base_model_name_or_path resolution.
    rank = manifest["counts"]["rank"]
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, bias="none",
                     target_modules=TARGETS[manifest["model_type"]],
                     task_type="CAUSAL_LM" if manifest["model_type"] == "qwen2" else "SEQ_2_SEQ_LM",
                     inference_mode=True, base_model_name_or_path=str(root / "base"))
    model = get_peft_model(model, cfg)
    state = load_file(str(root / "adapter" / "adapter_model.safetensors"), device="cpu")
    expected = get_peft_model_state_dict(model)
    _require(set(state) == set(expected) and all(state[k].shape == expected[k].shape
             and state[k].dtype == expected[k].dtype and bool(torch.isfinite(state[k]).all()) for k in state),
             "Adapter state differs from native PEFT graph")
    info = set_peft_model_state_dict(model, state)
    _require(not info.unexpected_keys and not any("lora_" in k for k in info.missing_keys), "Incomplete adapter load")
    _require(all(torch.equal(v, state[k]) for k, v in get_peft_model_state_dict(model).items()),
             "Adapter reload changed trained factors")
    model.requires_grad_(False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(root / "base", local_files_only=True, trust_remote_code=False)
    return StandaloneModel(model=model, tokenizer=tokenizer, manifest=manifest, base_path=root / "base")


# Serialization has no tensor-sized host allocation: the CPU contiguous storage
# is already resident. Shards bound files, not RAM; an indivisible tensor may be
# larger than the target (e.g. a 272 MiB embedding in a 256 MiB-target shard).
EXPORT_CHUNK_BYTES = 1024**2
EXPORT_SHARD_BYTES = 256 * 1024**2
EXPORT_HEADER_BYTES = 16 * 1024**2
TOKENIZER_ASSETS = frozenset({"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt", "spiece.model", "tokenizer.model",
    "chat_template.jinja"})


def _tensor_bytes(value):
    import torch
    _require(sys.byteorder == "little", "Streaming safetensors requires little-endian storage")
    _require(value.device.type == "cpu" and value.is_contiguous() and value.layout == torch.strided
             and value.dtype in (torch.float32, torch.bfloat16) and not value.is_meta,
             "Bounded export requires contiguous native F32/BF16 CPU tensors; no implicit staging/cast")
    # BF16 has no NumPy scalar dtype. View raw uint8 storage, never .float() or
    # .tobytes(); memoryview slices handed to write/hash remain zero-copy.
    return memoryview(value.detach().reshape(-1).view(torch.uint8).numpy())


def _stream_safetensors(path, tensors, guard=None):
    """Write standard safetensors header + contiguous raw storage, bounded I/O.

    No full-model clone, tensor conversion, mmap output, or Python payload bytes.
    Reader acceptance is checked by the normal strict header/native loader.
    """
    from .reconstruction import _release_file_cache
    if guard is not None:
        guard()
    header, offset, header_bound = {"__metadata__": {"format": "pt"}}, 0, 64
    _require(tensors and len(tensors) <= 100000, "Invalid streaming tensor count")
    for key, value in sorted(tensors.items()):
        _require(isinstance(key, str) and 0 < len(key) <= 1024 and key != "__metadata__",
                 "Invalid streaming tensor key")
        header_bound += 6 * len(key) + 512  # escaped JSON key, shape, 64-bit offsets
        _require(header_bound <= EXPORT_HEADER_BYTES, "Streaming header exceeds bound")
        shape = list(value.shape)
        _require(1 <= len(shape) <= 8 and all(type(n) is int and 0 < n <= 2**31 for n in shape),
                 "Invalid streaming tensor shape")
        raw = _tensor_bytes(value)
        size = math.prod(shape) * value.element_size()
        _require(raw.nbytes == size and offset + size < 2**63, "Streaming tensor span overflow")
        header[key] = {"dtype": "F32" if value.element_size() == 4 else "BF16",
                       "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
        del raw
    encoded = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    encoded += b" " * (-len(encoded) % 8)
    _require(len(encoded) <= EXPORT_HEADER_BYTES, "Streaming header exceeds bound")
    digests = {}
    with path.open("xb", buffering=0) as stream:
        def write_all(view):
            while view:
                written = stream.write(view)
                _require(written is not None and written > 0, "Short streaming write")
                view = view[written:]
        write_all(memoryview(struct.pack("<Q", len(encoded))))
        write_all(memoryview(encoded))
        pending = len(encoded) + 8
        for key, value in sorted(tensors.items()):
            raw, digest = _tensor_bytes(value), hashlib.sha256()
            for offset in range(0, raw.nbytes, EXPORT_CHUNK_BYTES):
                chunk = raw[offset:offset + EXPORT_CHUNK_BYTES]
                digest.update(chunk)
                write_all(chunk)
                pending += len(chunk)
                if pending >= 8 * 1024**2:
                    # Bound outstanding dirty output pages as well as Python
                    # buffers. Advice is not credit: reobserve before continuing.
                    os.fsync(stream.fileno())
                    _release_file_cache(path)
                    pending = 0
                    if guard is not None:
                        guard()
            digests[key] = digest.hexdigest()
            del raw
        stream.flush()
        os.fsync(stream.fileno())
    _release_file_cache(path)
    return digests


def _stream_base(base, native, plan, guard=None):
    """Normalize PEFT keys and save exactly one admitted native tied alias."""
    import torch
    state = {}
    for key, value in native.state_dict().items():  # shallow detached views only
        if ".lora_" in key:
            continue
        key = key.replace(".base_layer.", ".")
        _require(key not in state and "lora_" not in key and "base_layer" not in key,
                 "Unexpected/duplicate normalized PEFT state key")
        state[key] = value
    _require(set(state) == set(plan["aliases"]), "Native streaming keys differ from admitted aliases")
    # Canonical stored keys are those already admitted from the input native
    # checkpoint. Every dropped tied key must be an exact view of that storage.
    for alias, stored in plan["aliases"].items():
        value, canonical = state[alias], state[stored]
        _require(value.shape == canonical.shape and value.dtype == canonical.dtype
                 and value.data_ptr() == canonical.data_ptr() and value.stride() == canonical.stride(),
                 "Native tied alias differs at export: " + alias)
    unique = {key: state[key] for key in plan["target_dtypes"]}
    _require(all(v.dtype == getattr(torch, plan["target_dtypes"][k]) for k, v in unique.items()),
             "Native streaming dtype differs from admitted runtime policy")
    shards, current, size = [], {}, 0
    for key, value in sorted(unique.items()):
        count = value.numel() * value.element_size()
        if current and size + count > EXPORT_SHARD_BYTES:
            shards.append(current)
            current, size = {}, 0
        current[key] = value
        size += count
    if current:
        shards.append(current)
    weight_map, digests = {}, {}
    for index, shard in enumerate(shards, 1):
        name = ("model.safetensors" if len(shards) == 1 else
                "model-%05d-of-%05d.safetensors" % (index, len(shards)))
        digests.update(_stream_safetensors(base / name, shard, guard))
        weight_map.update({key: name for key in shard})
    if len(shards) > 1:
        (base / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": sum(v.numel() * v.element_size() for v in unique.values())},
            "weight_map": weight_map}, indent=2) + "\n")
    return digests


def _copy_tokenizer_assets(source, base, expected, byte_bound):
    """Copy only explicit native tokenizer assets; no links/manifests/custom code."""
    copied = 0
    for name in sorted(TOKENIZER_ASSETS.intersection(expected)):
        path = safe_file(source / name)
        before = path.stat()
        _require(copied + before.st_size <= byte_bound, "Tokenizer source exceeds admitted asset bytes")
        digest = hashlib.sha256()
        with path.open("rb") as reader, (base / name).open("xb") as writer:
            remaining = before.st_size
            while remaining:
                chunk = reader.read(min(EXPORT_CHUNK_BYTES, remaining))
                _require(chunk, "Tokenizer source truncated during copy")
                digest.update(chunk)
                writer.write(chunk)
                remaining -= len(chunk)
            _require(not reader.read(1), "Tokenizer source grew during copy")
        copied += before.st_size
        after = path.stat()
        _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                 (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                 and digest.hexdigest() == expected[name], "Tokenizer source changed during bounded copy: " + name)
    _require((base / "tokenizer_config.json").is_file(), "Missing copied tokenizer configuration")
    cfg = _json(base / "tokenizer_config.json")
    cfg.pop("name_or_path", None)
    cfg.pop("_name_or_path", None)
    for key, value in cfg.items():
        if key.endswith("_file") and value is not None:
            _require(value in TOKENIZER_ASSETS and (base / value).is_file(), "Unexpected tokenizer asset reference")
    (base / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2) + "\n")


def _write_bundle(root, student, tokenizer, report, deployment_cache, source):
    """Capture trained factors BEFORE any merge; serialize current frozen base bytes."""
    from peft import get_peft_model_state_dict
    from .recovery import _parameter_hash, _phase_guard
    import torch
    base, adapter = root / "base", root / "adapter"
    base.mkdir()
    adapter.mkdir()
    native = student.get_base_model()
    native.config.use_cache = deployment_cache
    native.config._name_or_path = ""
    native.config.silt_specialist_component = COMPONENT
    def frozen_hash():
        return _parameter_hash((n, p) for n, p in student.named_parameters() if "lora_" not in n)
    _require(frozen_hash() == report["frozen_parameter_hash_after"], "Frozen base changed before streaming")
    def stream_guard():
        _phase_guard(report["resources"], "factor-stream-write", report["resources"]["export_io_and_metadata_bytes"],
                     report["hyperparameters"]["memory_budget_bytes"])
    base_digests = _stream_base(base, native, report["resources"]["student"]["load_plan"], stream_guard)
    # Serialize the small live native config, including restored use_cache and
    # family-specific flags; never copy a stale source config or model metadata.
    native.config.architectures = [type(native).__name__]
    native.config.save_pretrained(base)
    if getattr(native, "generation_config", None) is not None:
        native.generation_config.save_pretrained(base)
    for name in ("config.json", "generation_config.json"):
        if (base / name).is_file():
            cfg = _json(base / name)
            cfg.pop("name_or_path", None)
            if "_name_or_path" in cfg:
                cfg["_name_or_path"] = ""
            (base / name).write_text(json.dumps(cfg, indent=2) + "\n")
    _copy_tokenizer_assets(source, base, report["student_store_hashes_before"],
                           report["resources"]["student"]["tokenizer_asset_bytes"])
    factors = get_peft_model_state_dict(student)
    _require(all(v.dtype == torch.float32 for v in factors.values()), "Expected native PEFT F32 trained factors")
    factor_digests = _stream_safetensors(adapter / "adapter_model.safetensors", factors, stream_guard)
    _require(frozen_hash() == report["frozen_parameter_hash_after"], "Frozen base changed during streaming")
    _require(_parameter_hash((n, p) for n, p in student.named_parameters() if "lora_" in n)
             == report["adapter_hash_after"], "Trained factors changed during streaming")
    report["bounded_export"] = {"strategy": "native_cpu_contiguous_storage_stream_v1",
        "io_chunk_bytes": EXPORT_CHUNK_BYTES, "shard_target_bytes": EXPORT_SHARD_BYTES,
        "sync_and_dynamic_recheck_interval_bytes": 8 * 1024**2,
        "oversized_single_tensor_shards_allowed": True, "tensor_staging_bytes": 0,
        "whole_model_clone": False, "source_weights_copied": False,
        "base_payload_sha256": base_digests, "factor_payload_sha256": factor_digests,
        "base_and_factor_live_hashes_verified_before_after": True,
        "tokenizer_asset_policy": "explicit allowlist, private bounded copy and source digest/stat verification"}
    kind, rank = native.config.model_type, report["hyperparameters"]["rank"]
    cfg = {"base_model_name_or_path": "../base", "peft_type": "LORA",
           "task_type": "CAUSAL_LM" if kind == "qwen2" else "SEQ_2_SEQ_LM", "r": rank,
           "lora_alpha": 2 * rank, "lora_dropout": 0.0, "bias": "none",
           "target_modules": TARGETS[kind], "inference_mode": True}
    (adapter / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (root / PATHS["runner"]).write_text(RUNNER_SOURCE)
    files = _inventory(root)
    _, _, base_headers = _artifact_preflight(base)
    base_count = sum(v["numel"] for v in base_headers.values())
    factor_count = sum(v.numel() for v in factors.values())
    base_bytes = sum(v["bytes"] for k, v in files.items() if k.startswith("base/") and k.endswith(".safetensors"))
    factor_bytes = files["adapter/adapter_model.safetensors"]["bytes"]
    source = report["resources"]["teacher"]
    manifest = {"schema": SCHEMA, "export_mode": "factor_preserving",
        "artifact_kind": "native_base_plus_required_local_lora", "paths": PATHS,
        "model_type": kind, "model_class": CLASSES[kind], "dtype": report["hyperparameters"]["dtype"],
        "runtime_versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "safetensors")},
        "files": files,
        "counts": {"base_parameters": base_count, "factor_parameters": factor_count, "rank": rank,
            "total_parameters": base_count + factor_count, "base_safetensors_bytes": base_bytes,
            "factor_safetensors_bytes": factor_bytes, "total_safetensors_bytes": base_bytes + factor_bytes,
            "original_teacher_parameters": source["parameters"], "original_teacher_safetensors_bytes": source["file_bytes"]},
        "teacher_source_provenance": {"audit_only": True, "model_type": report["teacher_model_type"],
            "parameters": source["parameters"], "safetensors_bytes": source["file_bytes"],
            "files_sha256": report["source_hashes_before"]}}
    _require(base_count + factor_count == report["total_parameters_with_adapter"] and
             factor_count == report["trainable_parameters"], "Bundle count mismatch against trained network")
    (root / "specialist_bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return inspect_bundle(root)
