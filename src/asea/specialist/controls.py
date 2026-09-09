"""Explicit random-MLP negative control; never a reconstruction fallback.

No optional ML imports at import time. Only a native Qwen2 META model is
constructed for strict shape/alias verification; no real model/teacher loads.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile

from asea.artifacts import Blocked, atomic_json, model_inventory, safe_file, safe_path
from asea.specialist.reconstruction import (
    ReconstructionBlocked, _artifact_preflight, _headers, _memory_budget,
    _native_meta, _publish, _strict_header_match, _unique_json,
)


class ControlBlocked(ReconstructionBlocked):
    """Unsafe, unsupported, or mechanically invalid control request."""


_CHUNK_ELEMENTS = 262144
_HASH_CHUNK = 1024 * 1024
_MANIFEST = "random_mlp_control_manifest.json"
_PUBLIC_ASSETS = {
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "vocab.json", "vocab.txt", "merges.txt", "tokenizer.model", "spiece.model",
    "chat_template.jinja",
}
_TORCH_DTYPES = {"F32": "float32", "BF16": "bfloat16"}


def _spans(root, headers):
    """Offsets from headers already validated by shared _headers()."""
    result = {}
    for filename in sorted({entry["file"] for entry in headers.values()}):
        with safe_file(root / filename).open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ControlBlocked("source header changed during preflight")
            size = struct.unpack("<Q", prefix)[0]
            if not 2 <= size <= 16 * 1024 * 1024:
                raise ControlBlocked("source header changed during preflight")
            raw = stream.read(size)
        header = json.loads(raw, object_pairs_hook=_unique_json)
        for key, entry in header.items():
            if key == "__metadata__":
                continue
            expected = headers.get(key)
            if (not expected or expected["file"] != filename
                    or entry["shape"] != expected["shape"]
                    or entry["dtype"] != expected["dtype"]):
                raise ControlBlocked("source header changed during preflight")
            start, end = entry["data_offsets"]
            result[key] = (8 + size + start, end - start)
    if set(result) != set(headers):
        raise ControlBlocked("source header changed during preflight")
    return result


def _span_hash(path, offset, size):
    digest = hashlib.sha256()
    with safe_file(path).open("rb") as stream:
        stream.seek(offset)
        remaining = size
        while remaining:
            block = stream.read(min(remaining, _HASH_CHUNK))
            if not block:
                raise ControlBlocked("truncated tensor payload")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _tensor_seed(seed, key):
    # Stable per storage key: independent of filenames, sharding, or global RNG.
    value = ("silt-random-mlp-v1\0" + str(seed) + "\0" + key).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") & ((1 << 63) - 1)


def _replace_tensor(path, offset, entry, std, seed, key, torch):
    generator = torch.Generator(device="cpu").manual_seed(_tensor_seed(seed, key))
    dtype = getattr(torch, _TORCH_DTYPES[entry["dtype"]])
    with safe_file(path).open("r+b") as stream:
        stream.seek(offset)
        remaining = entry["numel"]
        while remaining:
            count = min(remaining, _CHUNK_ELEMENTS)
            # Draw FP32 Gaussian, then round once to the ORIGINAL storage dtype.
            values = torch.empty(count, dtype=torch.float32, device="cpu")
            values.normal_(mean=0.0, std=std, generator=generator)
            values = values.to(dtype)
            if not bool(torch.isfinite(values).all()):
                raise ControlBlocked("nonfinite randomized MLP tensor: " + key)
            # uint8 view avoids unsupported BF16 -> numpy conversions.
            payload = values.contiguous().view(torch.uint8).numpy().tobytes()
            if stream.write(payload) != len(payload):
                raise ControlBlocked("short write of randomized tensor")
            remaining -= count
        stream.flush()
        os.fsync(stream.fileno())


def randomize_mlp_control(input_student, output_dir, seed=17):
    """Return/persist BASELINE_RANDOM_MLP_UNVALIDATED, with unchanged architecture.

    The input must be an existing local, standalone native Qwen2 safetensors
    checkpoint. Output must be new, disjoint, and have an existing parent.
    F32 and BF16 tensor storage are supported without dtype conversion of the
    inherited backbone. No teacher, dataset, optimizer, or quality run is used.
    """
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ControlBlocked("seed must be an integer in [0, 2**63)")
    if sys.byteorder != "little":
        raise ControlBlocked("safetensors payload writing requires little-endian host")
    source, output = safe_path(input_student), safe_path(output_dir)
    if (not source.is_dir() or output.exists() or source == output
            or source in output.parents or output in source.parents):
        raise ControlBlocked("input must exist; output must be new and disjoint")
    if not output.parent.is_dir():
        raise ControlBlocked("output parent must already exist")
    inventory, raw_config, headers = _artifact_preflight(source)
    if raw_config.get("model_type") != "qwen2":
        raise ControlBlocked("random MLP control supports native Qwen2 only; no fallback")
    std = raw_config.get("initializer_range")
    if (type(std) not in (int, float) or not math.isfinite(std) or std <= 0):
        raise ControlBlocked("explicit finite positive config.initializer_range required")
    if any(entry["dtype"] not in _TORCH_DTYPES for entry in headers.values()):
        raise ControlBlocked("control requires original F32/BF16 tensor storage")
    if "tokenizer_config.json" not in inventory or not any(
            name in inventory for name in ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json", "vocab.txt")):
        raise ControlBlocked("standalone local tokenizer assets required")
    admission = _memory_budget()
    estimated = 256 * 1024 ** 2 + _CHUNK_ELEMENTS * 16 + _HASH_CHUNK * 2
    if estimated > admission["limit_bytes"]:
        raise ControlBlocked("memory admission refused for bounded control workspace")
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ControlBlocked("install optional localmodels dependencies") from exc
    # Meta construction cannot allocate real model parameters or draw global RNG.
    native, model_class, meta = _native_meta(raw_config, headers, torch)
    aliases = _strict_header_match(meta, headers)
    logical_shapes = {key: list(tensor.shape) for key, tensor in meta.state_dict().items()}
    parameters = sum(parameter.numel() for parameter in meta.parameters())
    mlp_keys = {
        "model.layers.%d.mlp.%s.weight" % (layer, projection)
        for layer in range(native.num_hidden_layers)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    observed_mlp = {key for key in logical_shapes if ".mlp." in key}
    if observed_mlp != mlp_keys or not mlp_keys.issubset(headers):
        raise ControlBlocked("native MLP ownership is not exactly gate/up/down weights")
    if any(aliases[key] != key for key in mlp_keys):
        raise ControlBlocked("MLP weights must have independent storage ownership")
    spans = _spans(source, headers)
    weight_files = {entry["file"] for entry in headers.values()}
    copied = set(weight_files) | (set(inventory) & _PUBLIC_ASSETS)
    if "model.safetensors.index.json" in inventory:
        copied.add("model.safetensors.index.json")
    copied.update(name for name in inventory if name.startswith("chat_templates/") and name.endswith(".jinja"))
    # Do NOT carry forward stale reconstruction/recovery/evaluation manifests.
    omitted = sorted(set(inventory) - copied)
    copy_bytes = sum(inventory[name]["size"] for name in copied)
    if shutil.disk_usage(output.parent).free < copy_bytes + 16 * 1024 ** 2:
        raise ControlBlocked("disk admission refused for independent checkpoint copy")
    stage = Path(tempfile.mkdtemp(prefix="." + output.name + ".control-stage-", dir=str(output.parent)))
    try:
        for name in sorted(copied):
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            # A real copy: never hardlink or symlink mutable output to the source.
            shutil.copyfile(safe_file(source / name), destination)
        # Detect copy/source races before interpreting or patching any offset.
        if model_inventory(stage) != {name: inventory[name] for name in copied}:
            raise ControlBlocked("checkpoint copy changed during staging")
        provenance = {}
        for key, entry in sorted(headers.items()):
            offset, size = spans[key]
            before = _span_hash(source / entry["file"], offset, size)
            randomized = key in mlp_keys
            if randomized:
                _replace_tensor(stage / entry["file"], offset, entry, float(std), seed, key, torch)
            after = _span_hash(stage / entry["file"], offset, size)
            if randomized == (before == after):
                raise ControlBlocked("MLP must change and shared backbone must not change: " + key)
            provenance[key] = {
                "source_storage_key": key, "output_storage_key": key,
                "source_file": entry["file"], "output_file": entry["file"],
                "owner": "random_mlp_control" if randomized else "inherited_shared_backbone",
                "operation": "normal_reinitialize" if randomized else "byte_identical_copy",
                "shape": entry["shape"], "dtype": entry["dtype"],
                "parameters": entry["numel"], "tensor_bytes": size,
                "before_sha256": before, "after_sha256": after,
                "changed": randomized,
                "tensor_seed": _tensor_seed(seed, key) if randomized else None,
            }
        output_inventory = model_inventory(stage)
        if _headers(stage, output_inventory) != headers:
            raise ControlBlocked("serialized architecture/shapes/dtypes/sharding changed")
        _strict_header_match(meta, _headers(stage, output_inventory))
        for name in copied:
            if output_inventory[name]["size"] != inventory[name]["size"]:
                raise ControlBlocked("checkpoint file size changed: " + name)
            if name not in weight_files and output_inventory[name] != inventory[name]:
                raise ControlBlocked("config/tokenizer bytes changed: " + name)
        if model_inventory(source) != inventory:
            raise ControlBlocked("source changed during control creation")
        count = {
            "parameters": parameters,
            "stored_tensors": len(headers), "logical_tensors": len(aliases),
            "tensor_bytes": sum(size for _, size in spans.values()),
            "safetensors_bytes": sum(inventory[name]["size"] for name in weight_files),
        }
        manifest = {
            "schema_version": 1,
            "artifact_kind": "standalone_native_hf_random_mlp_control",
            "status": "BASELINE_RANDOM_MLP_UNVALIDATED",
            "study_role": "NEGATIVE_CONTROL",
            "method": {
                "name": "random_mlp", "seed": seed,
                "distribution": "Normal(mean=0, std=config.initializer_range); FP32 draws then original-dtype rounding",
                "initializer_range": std, "scope": "every gate_proj/up_proj/down_proj MLP weight only",
                "rng": "CPU torch.Generator per tensor; SHA256(silt-random-mlp-v1 NUL seed NUL storage_key), first8 little-endian masked to 63bits",
                "chunk_elements": _CHUNK_ELEMENTS,
                "fallback_for_requested_model": False,
                "skill_extraction_claimed": False,
            },
            "source": {"files": inventory, "config": raw_config},
            "output": {
                "files": output_inventory, "config": raw_config,
                "architecture": model_class.__name__, "model_type": "qwen2",
                "standalone": True, "teacher_required_at_serve": False,
                "safe_serialization": True,
                "file_inventory_excludes": [_MANIFEST],
            },
            "omitted_source_ancillary_files": omitted,
            "counts": {"source": dict(count), "output": dict(count),
                "randomized_mlp_tensors": len(mlp_keys),
                "randomized_mlp_parameters": sum(headers[key]["numel"] for key in mlp_keys),
                "preserved_shared_tensors": len(headers) - len(mlp_keys)},
            "tensor_provenance": provenance,
            "logical_tensor_storage_ownership": aliases,
            "memory_admission": dict(admission, estimated_workspace_bytes=estimated,
                policy="meta-only native validation; bounded payload chunks; no full model allocation"),
            "verification": {
                "strict_native_meta_keys_shapes_and_aliases": True,
                "source_files_unchanged": True, "all_mlp_tensors_changed": True,
                "all_shared_tensors_byte_identical": True,
                "parameter_count_and_tensor_bytes_equal": True,
                "safetensors_file_sizes_equal": True,
                "configuration_and_tokenizer_bytes_unchanged": True,
                "checkpoint_load_performed": False, "teacher_loaded": False,
                "forward_or_quality_run_performed": False,
                "capability_evaluated": False, "quality_pass": False,
                "promotion_performed": False, "certificate": False,
            },
            "runtime": {"torch": torch.__version__, "transformers": transformers.__version__, "device": "cpu"},
            "interpretation": "Unvalidated initialization ablation only. Compare inherited-MLP and random-MLP students under identical recovery data, teacher KD, 64 steps, seeds and evaluation protocol. Not evidence of skill extraction or quality promotion.",
        }
        atomic_json(stage / _MANIFEST, manifest)
        safe_path(output)
        _publish(stage, output)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create an explicit unvalidated native random-MLP control; never certify quality.")
    parser.add_argument("--input-student", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    try:
        manifest = randomize_mlp_control(args.input_student, args.output, args.seed)
    except (Blocked, OSError, ValueError, ImportError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc), "certificate": False}), file=sys.stderr)
        return 2
    print(json.dumps({"status": manifest["status"], "output": str(safe_path(args.output)), "certificate": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
