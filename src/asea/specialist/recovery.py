"""Local, fail-closed recovery of an already reconstructed native student.

This is an experimental training primitive, not a quality/promotion gate.
Heavy libraries are imported only by recover(); checkpoints must be safetensors.
"""
from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import struct
import tempfile
import time
import weakref
from typing import Any, Dict


class RecoveryRejected(ValueError):
    """Invalid or unsafe recovery request."""


def _require(condition, message):
    if not condition:
        raise RecoveryRejected(message)


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _store_hash(path):
    """Hash every file, not just weights; never write into an input store."""
    from asea.artifacts import model_inventory
    from .reconstruction import _release_file_cache
    inventory = model_inventory(path)
    for name in inventory:
        if name.endswith(".safetensors"):
            _release_file_cache(path / name)
    return {name: entry["sha256"] for name, entry in inventory.items()}


def _family(teacher_type, student_type):
    if (teacher_type, student_type) == ("qwen2", "qwen2"):
        return "causal"
    if teacher_type in ("switch_transformers", "t5") and student_type == "t5":
        return "seq2seq"
    raise RecoveryRejected("Unsupported teacher/student model_type pair: %s/%s" %
                           (teacher_type, student_type))


MAX_DATA_BYTES = 16 * 1024 * 1024
MAX_DATA_ROWS = 10000


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def _bounded_json(path):
    # Bound bytes BEFORE allocating/decoding JSON, including a growing-file race.
    _require(path.is_file() and path.stat().st_size <= MAX_DATA_BYTES,
             "Dataset exceeds 16 MiB byte limit or is not a file")
    with path.open("rb") as handle:
        raw = handle.read(MAX_DATA_BYTES + 1)
    _require(len(raw) <= MAX_DATA_BYTES, "Dataset exceeds 16 MiB byte limit")
    text = raw.decode("utf-8")
    # Lightweight lexical prepass: count every array's elements before json.loads
    # builds objects. Strings (including escaped delimiters) are single tokens.
    stack = []
    lex = r'"(?:[^"\\\x00-\x1f]|\\.)*"|[\[\]{},:]|[^\s\[\]{},:]+'
    for token in re.finditer(lex, text):
        value = token.group()
        if stack and stack[-1][0] == "[" and stack[-1][2] and value != "]":
            stack[-1][1] += 1
            stack[-1][2] = False
            _require(stack[-1][1] <= MAX_DATA_ROWS, "Dataset exceeds 10000 array entries/records")
        if value in ("[", "{"):
            stack.append([value, 0, value == "["])
            _require(len(stack) <= 32, "Dataset nesting exceeds 32 levels")
        elif value in ("]", "}"):
            _require(stack and stack[-1][0] == ("[" if value == "]" else "{"), "Invalid JSON nesting")
            stack.pop()
        elif value == "," and stack and stack[-1][0] == "[":
            stack[-1][2] = True
    def nonfinite(value):
        raise RecoveryRejected("Nonfinite JSON value: " + value)
    def finite_float(value):
        number = float(value)
        _require(math.isfinite(number), "Nonfinite JSON number: " + value)
        return number
    return json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=nonfinite, parse_float=finite_float)


def _sample_families(rows):
    families = set()
    for row in rows:
        for key in ("family", "family_id", "task_family"):
            if key in row:
                value = row[key]
                _require(isinstance(value, str) and value and value == value.strip(),
                         "Invalid sample family: " + key)
                families.add(value)
    return families


def _read_samples(path):
    data = _bounded_json(path)
    _require(isinstance(data, dict) and isinstance(data.get("samples"), list),
             "Data must be an object containing samples")
    rows = data["samples"]
    _require(rows and len(rows) <= MAX_DATA_ROWS, "Dataset empty or exceeds 10000 records")
    ids, hashes = set(), set()
    for row in rows:
        _require(isinstance(row, dict) and all(isinstance(row.get(k), str)
                 for k in ("id", "prompt", "response")), "Samples need string id/prompt/response")
        identifier = row["id"]
        _require(identifier and len(identifier) <= 256 and identifier == identifier.strip()
                 and not any(ord(c) < 32 or ord(c) == 127 for c in identifier), "Invalid sample id")
        _require(row["prompt"].strip() and row["response"].strip(), "Empty prompt or response")
        _require(len(row["prompt"]) <= 65536 and len(row["response"]) <= 65536,
                 "Sample text exceeds 65536 characters")
        digest = _json_hash([row["prompt"], row["response"]])
        _require(identifier not in ids and digest not in hashes, "Duplicate sample id/content")
        ids.add(identifier)
        hashes.add(digest)
    _sample_families(rows)
    return rows, ids, hashes


def _recovery_inputs(raw, headers, dtype, torch):
    """Validate native aliases, then create parameter-only meta with CPU buffers.

    No checkpoint tensor is read here. Keep construction identical to the proven
    reconstruction loader; torch.device(meta) alone would strand Qwen rotary buffers.
    """
    from accelerate import init_empty_weights
    from .reconstruction import _native_meta, _load_plan
    config, cls, checked = _native_meta(raw, headers, torch)
    del checked
    config.torch_dtype = getattr(torch, dtype)
    with init_empty_weights(include_buffers=False):
        meta = cls(config)
    meta.tie_weights()
    return meta, _load_plan(meta, headers, dtype)


def _load_recovery_model(meta, directory, headers, plan, torch):
    """Execute the admitted per-tensor loader, not an opaque HF shard conversion."""
    from .reconstruction import _load_source
    from transformers import GenerationConfig
    model = _load_source(meta, directory, headers, plan, torch)
    if (directory / "generation_config.json").is_file():
        model.generation_config = GenerationConfig.from_pretrained(directory, local_files_only=True)
    model.eval()
    return model


def _header_stats(directory, rank, targets, dtype="bfloat16", prepared=None):
    from .reconstruction import _artifact_preflight, _SIZES
    if prepared is None:
        import torch
        inventory, raw, headers = _artifact_preflight(directory)
        meta, plan = _recovery_inputs(raw, headers, dtype, torch)
        del meta
    else:
        inventory, headers, plan = prepared
    _require(plan["strategy"] == "explicit_safetensors_tensor_meta_assign", "Unproven recovery loader")
    parameters = adapters = largest = largest_target = source_bytes = retained_f32 = 0
    loaded = shared = cast_source = cast_target = scratch = lora_target_bytes = 0
    groups, casts = {}, []
    widths, native = {"float32": 4, "bfloat16": 2}, {"float32": "F32", "bfloat16": "BF16"}
    for key, entry in headers.items():
        shape, count = entry["shape"], entry["numel"]
        target = plan["target_dtypes"][key]
        source_size, target_size = count * _SIZES[entry["dtype"]], count * widths[target]
        parameters += count
        source_bytes += source_size
        loaded += target_size
        largest = max(largest, count)
        if target == "float32" and dtype != "float32":
            retained_f32 += count
        group = groups.setdefault(entry["dtype"] + "->" + target,
                                 {"parameters": 0, "source_bytes": 0, "loaded_bytes": 0})
        group["parameters"] += count
        group["source_bytes"] += source_size
        group["loaded_bytes"] += target_size
        if entry["dtype"] == native[target]:
            shared += target_size  # CPU .to of identical dtype/device aliases storage
        else:
            cast_source += source_size
            cast_target += target_size
            scratch = max(scratch, source_size + target_size)
            casts.append({"key": key, "source_dtype": entry["dtype"], "loaded_dtype": target,
                          "source_bytes": source_size, "target_bytes": target_size})
        names = [alias for alias, stored in plan["aliases"].items() if stored == key]
        if len(shape) == 2 and any(n.endswith(".weight") and n.split(".")[-2] in targets for n in names):
            adapters += rank * sum(shape)
            lora_target_bytes += target_size
            largest_target = max(largest_target, count)
    # Mapped cast-source pages may stay live beside same-shard shared tensors.
    # Reserve ALL of them (across shards), plus a conservative one-tensor scratch.
    # Header/alignment pages are additional to payload, never negative "free" credit.
    files = [v["size"] for k, v in inventory.items() if k.endswith(".safetensors")]
    mapping_overhead = sum(files) - source_bytes + len(files) * 8192
    from .standalone import TOKENIZER_ASSETS
    return {"parameters": parameters, "adapter_parameters": adapters,
            "tensor_count": len(headers),
            "tokenizer_asset_bytes": sum(v["size"] for k, v in inventory.items() if k in TOKENIZER_ASSETS),
            "largest_tensor_parameters": largest, "largest_lora_target_parameters": largest_target,
            "file_bytes": sum(files), "largest_shard_bytes": max(files),
            "source_payload_bytes": source_bytes, "retained_f32_parameters_upper_bound": retained_f32,
            "dtypes": sorted({v["dtype"] for v in headers.values()}),
            "load_plan": plan, "dtype_groups": groups, "loaded_bytes": loaded,
            "lora_target_loaded_bytes": lora_target_bytes,
            "shared_mmap_payload_bytes": shared, "cast_target_allocation_bytes": cast_target,
            "cast_source_pages_bytes": cast_source, "largest_cast_tensor_scratch_bytes": scratch,
            "mapping_overhead_bytes": mapping_overhead, "cast_tensor_plan": casts,
            "load_peak_bytes": loaded + cast_source + scratch + mapping_overhead}


def _available_ram():
    from .reconstruction import _available_ram as observed
    return observed()


def _preflight(teacher, student, output_parent, config, rank, targets, width, max_length, kd, torch,
               memory_budget_bytes=None, teacher_mode="cached", max_response_tokens=None,
               total_response_tokens=0, bank_rows=0, teacher_config=None, export_mode="native_merged",
               teacher_prepared=None, student_prepared=None):
    """CPU-only, serial cached teacher/student phases. No hardware fit guarantee."""
    from .reconstruction import _memory_budget
    dtype = "bfloat16" if width == 2 else "float32"
    source = _header_stats(teacher, rank, targets, dtype, teacher_prepared)
    candidate = _header_stats(student, rank, targets, dtype, student_prepared)
    _require(candidate["adapter_parameters"] > 0, "No native LoRA target matrices found")
    response = max_length if max_response_tokens is None else max_response_tokens
    vocabulary = int(config.vocab_size)
    def activation(cfg):
        hidden = int(getattr(cfg, "hidden_size", getattr(cfg, "d_model", 512)))
        layers = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "num_layers", 12)))
        return max(128 * 1024**2, max_length * hidden * layers * width * 12)
    source_loaded, student_loaded = source["loaded_bytes"], candidate["loaded_bytes"]
    adapter_bytes = candidate["adapter_parameters"] * 16
    snapshot_bytes = candidate["adapter_parameters"] * 4
    merge_bytes = candidate["largest_lora_target_parameters"] * 8 if export_mode == "native_merged" else 0
    activations = activation(config)
    # Keep full vocabulary; response positions only in the verified F32 bank.
    logits_bytes = max_length * vocabulary * width + response * vocabulary * (width + (24 if kd else 20))
    teacher_logits = vocabulary * (max_length * max(width, 4) + response * (width + 4))
    teacher_workspace = teacher_logits + activation(teacher_config or config) if kd else 0
    teacher_phase = source["load_peak_bytes"] + teacher_workspace if kd else 0
    weight_bytes = student_loaded + (source_loaded if kd and teacher_mode == "resident" else 0)
    # Native merge can dirty every targeted mmap base matrix. Its original file
    # pages may still be charged alongside COW destinations; do not count them free.
    merge_cow = candidate["lora_target_loaded_bytes"] if export_mode == "native_merged" else 0
    student_workspace = adapter_bytes + snapshot_bytes + logits_bytes + activations + merge_bytes + merge_cow
    # Resident teacher forward work can overlap student graphs; keep its bound too.
    resident_bytes = (source["load_peak_bytes"] + teacher_workspace
                      if kd and teacher_mode == "resident" else 0)
    student_incremental = candidate["load_peak_bytes"] + student_workspace
    student_phase = student_incremental + resident_bytes
    # Native-merged admission is deliberately unchanged. The factor path uses
    # a verified zero-staging writer, followed by a fresh same-dtype tensor loader
    # only AFTER weakref-verified training teardown. No largest-embedding cast and
    # no duplicate model serialization are performed by that path.
    export_workspace = max(activations + logits_bytes,
                           candidate["largest_tensor_parameters"] * 8) + adapter_bytes
    export_reload = student_loaded * 2 + export_workspace
    export_write_peak = export_reload
    export_io = export_probe = export_reload_weights = 0
    if export_mode == "factor_preserving":
        from .standalone import EXPORT_CHUNK_BYTES, EXPORT_HEADER_BYTES
        # JSON/header Python objects and native tokenizer reload, not just file
        # bytes. The original tokenizer remains resident during verification.
        export_io = (64 * 1024**2 + 2 * EXPORT_CHUNK_BYTES + 8 * candidate["tokenizer_asset_bytes"]
                     + 8192 * candidate["tensor_count"])
        hidden = int(getattr(config, "hidden_size", getattr(config, "d_model", 512)))
        layers = int(getattr(config, "num_hidden_layers", getattr(config, "num_layers", 12)))
        decoder_layers = int(getattr(config, "num_decoder_layers", layers))
        # Full forward (not response-only) + retained before/after logits and
        # F32 subtraction/abs diagnostics. Two-token greedy generation also runs
        # while the retained forward is live; cache and score buffers included.
        full_logits = (max_length + 2) * vocabulary * max(width, 4)
        kv_cache = (layers + decoder_layers) * (max_length + 2) * hidden * width * 4
        export_probe = activations + 6 * full_logits + kv_cache
        export_workspace = export_io + export_probe + adapter_bytes
        # At most one file per tensor; each shard header is bounded and total
        # header descriptors are bounded by key length/native tensor count.
        mapping = candidate["tensor_count"] * (8192 + 2048) + EXPORT_HEADER_BYTES
        export_reload_weights = student_loaded + mapping + adapter_bytes
        export_reload = export_reload_weights + export_io + export_probe
        export_write_peak = candidate["load_peak_bytes"] + adapter_bytes + snapshot_bytes + export_workspace
    estimated = max(teacher_phase, student_phase, export_write_peak, export_reload)
    budget = _memory_budget(memory_budget_bytes, available=_available_ram())
    bank_tensor_bytes = total_response_tokens * vocabulary * 4 if kd and teacher_mode == "cached" else 0
    # Safetensors headers + per-row provenance and manifest, bounded before writes.
    bank_disk_bound = bank_tensor_bytes + bank_rows * 16384 + 1024**2 if bank_tensor_bytes else 0
    # Includes private base serialization, F32 factors, tokenizer/config/headers,
    # and duplicate-file scratch; source stores are not hard-linked or rewritten.
    factor_disk = candidate["adapter_parameters"] * 8 if export_mode == "factor_preserving" else 0
    disk_required = student_loaded * 2 + factor_disk + 64 * 1024**2 + bank_disk_bound
    if export_mode == "factor_preserving":
        disk_required += candidate["tokenizer_asset_bytes"] + candidate["tensor_count"] * 2048
    disk_free = shutil.disk_usage(output_parent).free
    _require(disk_free >= disk_required, "Insufficient disk headroom before model loading/cache")
    _require(estimated <= budget["limit_bytes"], "Insufficient estimated training memory headroom: %d > %d" % (estimated, budget["limit_bytes"]))
    return {**budget, "teacher": source, "student": candidate, "weight_bytes": weight_bytes,
            "lora_optimizer_training_bytes": adapter_bytes, "adapter_snapshot_host_bytes": snapshot_bytes,
            "logits_workspace_bytes": logits_bytes, "activation_estimate_bytes": activations,
            "merge_workspace_bytes": merge_bytes, "teacher_phase_estimated_bytes": teacher_phase,
            "student_phase_estimated_bytes": student_phase, "estimated_peak_bytes": estimated,
            "student_phase_incremental_bytes": student_incremental,
            "teacher_workspace_bytes": teacher_workspace, "student_workspace_bytes": student_workspace,
            "merge_cow_source_pages_bytes": merge_cow,
            "export_workspace_bytes": export_workspace, "export_reload_estimated_bytes": export_reload,
            "export_write_peak_estimated_bytes": export_write_peak,
            "export_io_and_metadata_bytes": export_io, "export_full_probe_workspace_bytes": export_probe,
            "export_reload_weights_and_factors_bytes": export_reload_weights,
            "export_strategy": ("native_cpu_contiguous_storage_stream_v1" if export_mode == "factor_preserving"
                                else "native_hf_serialization_conservative_two_copy"),
            "runtime_phase_checks": [],
            "available_device_bytes": budget["limit_bytes"], "device": "cpu", "teacher_mode": teacher_mode,
            "actual_max_complete_length": max_length, "actual_max_response_tokens": response,
            "bank_tensor_bytes_bound": bank_tensor_bytes, "bank_disk_bytes_bound": bank_disk_bound,
            "disk_required_bytes": disk_required, "disk_free_bytes": disk_free,
            "factor_export_disk_bytes_bound": factor_disk, "export_mode": export_mode,
            "assumptions": "explicit per-tensor CPU meta assignment; class-specific native F32 exceptions; identical-dtype .to shares mmap; ALL cast-source pages plus largest source+target tensor scratch; header/alignment overhead; native merge COW pages; full-vocab response KL; dynamic observed headroom with unchanged reserve/operator ceiling; no assumed cache reclamation"}


def _phase_guard(resources, phase, required, memory_budget_bytes):
    """Check additional allocations against current free RAM, never credit a free.

    Full phase peaks already passed the operator ceiling at admission. Existing
    weights/pages remain in observed usage, not an invented subtraction from it.
    """
    from .reconstruction import _memory_budget
    budget = _memory_budget(memory_budget_bytes, available=_available_ram())
    check = {"phase": phase, "additional_required_bytes": required, **budget}
    resources["runtime_phase_checks"].append(check)
    _require(required <= budget["limit_bytes"],
             "Insufficient %s memory headroom: %d > %d" % (phase, required, budget["limit_bytes"]))
    return check


def _assert_collected(references, resources, phase):
    """Prove no model/parameter Python references survive before one-model reload.

    This is not reclaimed-RSS credit: file pages, allocator arenas and any other
    process remain charged in the following dynamic MemAvailable/cgroup check.
    """
    gc.collect()
    surviving = [name for name, reference in references if reference() is not None]
    evidence = {"phase": phase, "tracked_objects": len(references),
                "surviving_objects": surviving, "all_collected": not surviving,
                "reclaimed_memory_credit_bytes": 0}
    resources.setdefault("release_checks", []).append(evidence)
    _require(not surviving, "Training references retained before %s: %s" % (phase, ", ".join(surviving[:8])))
    return evidence


def _encoding_contract(tokenizer, family):
    if family == "seq2seq":
        template = "native-seq2seq: prompt and response independently; add_special_tokens=True; truncation=False"
        name = "native_seq2seq_v1"
    elif getattr(tokenizer, "chat_template", None):
        template = tokenizer.get_chat_template()
        name = "native_chat_response_only_v1"
    else:
        template = "prompt + '\\n' + response + eos_token; add_special_tokens=False; truncation=False"
        name = "flat_newline_response_only_v1"
    return {"format": name, "template": template,
            "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "truncation": False, "boundary_policy": "reject incompatible prefix or boundary-spanning token",
            "terminal_eos_positions": 1}


def _encode_details(row, tokenizer, family, max_length):
    _require(family in ("causal", "seq2seq"), "Unsupported encoding family")
    prompt, response = row["prompt"], row["response"]
    eos = tokenizer.eos_token_id
    _require(eos is not None, "EOS token is required")
    contract = _encoding_contract(tokenizer, family)
    if family == "seq2seq":
        ids = tokenizer.encode(prompt, add_special_tokens=True, truncation=False)
        labels = tokenizer.encode(response, add_special_tokens=True, truncation=False)
        _require(labels and labels[-1] == eos and labels.count(eos) == 1,
                 "Incompatible seq2seq tokenizer: native response must have exactly one terminal EOS")
        _require(ids and ids[-1] == eos, "Incompatible seq2seq tokenizer: missing native prompt EOS")
        prefix_ids = ids
        response_tokens = len(labels) - 1
        _require(response_tokens > 0, "Response must contain content before terminal EOS")
        rendered_hash = _json_hash([prompt, response])
        prefill_text = prompt
        boundary_engine = "native_seq2seq_special_tokens"
    else:
        eos_text = getattr(tokenizer, "eos_token", None)
        _require(isinstance(eos_text, str) and eos_text, "EOS token text is required")
        if getattr(tokenizer, "chat_template", None):
            user = [{"role": "user", "content": prompt}]
            prefix = tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(user + [{"role": "assistant", "content": response}],
                                                 tokenize=False, add_generation_prompt=False)
            empty = tokenizer.apply_chat_template(user + [{"role": "assistant", "content": ""}],
                                                  tokenize=False, add_generation_prompt=False)
            _require(full.startswith(prefix + response) and empty.startswith(prefix + eos_text),
                     "Incompatible chat template: full must start with identical user generation prefix and unmodified response")
            suffix = empty[len(prefix + eos_text):]
            _require(suffix in ("", "\n", "\r\n") and full == prefix + response + eos_text + suffix,
                     "Incompatible chat template: response rewriting or unsupported role suffix")
        else:
            prefix = prompt + "\n"
            full = prefix + response + eos_text
        start, end = len(prefix), len(prefix) + len(response)
        terminal_end = end + len(eos_text)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False, truncation=False)
        _require(prefix_ids, "Incompatible template: empty effective prefill")
        try:
            encoded = tokenizer(full, add_special_tokens=False, truncation=False,
                                return_offsets_mapping=True, return_attention_mask=False)
            ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        except (TypeError, NotImplementedError, KeyError):
            _require(not getattr(tokenizer, "chat_template", None),
                     "Incompatible chat tokenizer: complete offset mapping required")
            # Encode-only test doubles/slow flat tokenizers: never concatenate IDs
            # for training. Independently verify exact equality to the full encoding.
            ids = tokenizer.encode(full, add_special_tokens=False, truncation=False)
            response_ids = tokenizer.encode(response, add_special_tokens=False, truncation=False)
            _require(ids == prefix_ids + response_ids + [eos],
                     "Incompatible flat template: strict joint/prefix token stability failed")
            offsets = None
        _require(ids[:len(prefix_ids)] == prefix_ids,
                 "Incompatible template: inference prefix IDs differ from full training prefix IDs")
        boundary_engine = "strict_joint_prefix_id_equality" if offsets is None else "offset_mapping"
        if offsets is None:
            labels = [-100] * len(prefix_ids) + ids[len(prefix_ids):]
            response_tokens = len(labels) - len(prefix_ids) - 1
        else:
            _require(len(ids) == len(offsets), "Invalid complete token offsets")
            labels = [-100] * len(ids)
            response_tokens = eos_positions = 0
            for index, (left, right) in enumerate(offsets):
                _require(0 <= left <= right <= len(full), "Invalid complete token offset bounds")
                # Never supervise a prompt/role-cap character or silently lose a
                # response character: a boundary token cannot satisfy both promises.
                crosses = any(left < boundary < right for boundary in (start, end, terminal_end))
                _require(not crosses, "Incompatible template: boundary-spanning token would mask response characters")
                if start <= left < right <= end:
                    _require(index >= len(prefix_ids), "Incompatible template: response overlaps inference prefix")
                    labels[index] = ids[index]
                    response_tokens += 1
                elif left == end and right == terminal_end and ids[index] == eos:
                    labels[index] = eos
                    eos_positions += 1
            _require(eos_positions == 1, "Incompatible template: terminal EOS must be one mapped token")
        _require(labels.count(eos) == 1 and response_tokens > 0,
                 "Response must contain content and exactly one supervised terminal EOS; text is never rewritten")
        _require(all(value == -100 for value in labels[:len(prefix_ids)]), "Prompt target leakage")
        rendered_hash = hashlib.sha256(full.encode()).hexdigest()
        prefill_text = prefix
    _require(len(ids) <= max_length and len(labels) <= max_length,
             "Complete token length exceeds max_length (sample %s: input=%d, labels=%d, limit=%d); no truncation" %
             (row.get("id", "<unnamed>"), len(ids), len(labels), max_length))
    example = {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}
    evidence = {"id": row.get("id"), "response_characters": len(response),
                "response_utf8_bytes": len(response.encode()),
                "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                "response_content_tokens": response_tokens, "supervised_tokens": sum(x != -100 for x in labels),
                "input_tokens": len(ids), "label_tokens": len(labels), "terminal_eos_positions": 1,
                "effective_prefill_tokens": len(prefix_ids), "effective_prefill_ids_sha256": _json_hash(prefix_ids),
                "effective_prefill_text_sha256": hashlib.sha256(prefill_text.encode()).hexdigest(),
                "label_mask_sha256": _json_hash([x != -100 for x in labels]),
                "input_ids_sha256": _json_hash(ids), "labels_sha256": _json_hash(labels),
                "rendered_sha256": rendered_hash, "masked_boundary_tokens": 0, "boundary_engine": boundary_engine,
                "template_sha256": contract["template_sha256"]}
    return example, evidence


def _encode(row, tokenizer, family, max_length):
    """Complete encoding only; reject instead of trimming/truncating/relabeling."""
    return _encode_details(row, tokenizer, family, max_length)[0]


def _response_logits(logits, labels, family):
    if family == "causal":
        logits, labels = logits[:, :-1], labels[:, 1:]
    mask = labels != -100
    _require(bool(mask.any()), "No supervised response tokens")
    return logits[mask], labels[mask]


def _full_kl(student, teacher, chunk_size=4096):
    """Exact forward KL T||S, full vocabulary, T=1; vocab-chunked arithmetic.

    Chunk recomputation avoids retaining every chunk's fp32 probabilities for backward.
    No top-k approximation and no teacher gradients.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    s_norm = torch.logsumexp(student.float(), dim=-1)
    t_norm = torch.logsumexp(teacher.float(), dim=-1)

    def term(s, t, sn, tn):
        tlp = t.float() - tn[:, None]
        slp = s.float() - sn[:, None]
        return (tlp.exp() * (tlp - slp)).sum()

    result = student.new_zeros((), dtype=torch.float32)
    for offset in range(0, student.shape[-1], chunk_size):
        args = (student[:, offset:offset + chunk_size], teacher[:, offset:offset + chunk_size],
                s_norm, t_norm)
        if torch.is_grad_enabled() and student.requires_grad:
            result = result + checkpoint(term, *args, use_reentrant=False)
        else:
            result = result + term(*args)
    return result / student.shape[0]


def _native_batch(example, native, family, torch, device):
    batch = {k: torch.tensor([v], dtype=torch.long, device=device) for k, v in example.items()}
    labels = batch.pop("labels")
    if family == "seq2seq":
        batch["decoder_input_ids"] = native._shift_right(labels)
        batch["decoder_attention_mask"] = torch.ones_like(labels)
    return batch, labels


class _TeacherBank:
    """Private single-run bank. Cannot open or adopt an external cache/manifest.

    Every read verifies in-memory provenance plus full file hash before loading
    one row. F32 storage losslessly preserves native BF16/F32 teacher logits.
    """
    def __init__(self, directory, provenance, vocabulary, tensor_bound, disk_bound):
        from asea.artifacts import safe_path
        self.directory = safe_path(directory)
        self.directory.mkdir(mode=0o700, exist_ok=False)
        self.provenance = dict(provenance)
        self.vocabulary = vocabulary
        self.tensor_bound, self.disk_bound = tensor_bound, disk_bound
        self.tensor_bytes = self.file_bytes = 0
        self.entries = {}

    def write(self, example, selected, split, sample_id):
        import torch
        from safetensors.torch import save_file
        from asea.artifacts import safe_file
        _require(selected.dtype in (torch.float32, torch.bfloat16), "Unsupported native teacher logit dtype")
        tensor = selected.detach().to(dtype=torch.float32, device="cpu").contiguous()
        _require(tensor.ndim == 2 and tensor.shape[1] == self.vocabulary and bool(torch.isfinite(tensor).all()),
                 "Invalid full-vocabulary teacher bank tensor")
        size = tensor.numel() * tensor.element_size()
        _require(self.tensor_bytes + size <= self.tensor_bound, "Teacher bank tensor bound exceeded")
        _require(id(example) not in self.entries, "Teacher bank row already written")
        path = self.directory / ("row-%06d.safetensors" % len(self.entries))
        _require(not path.exists() and not path.is_symlink(), "Teacher bank row path must be exclusive")
        metadata = {**self.provenance, "encoding_sha256": _json_hash(example), "split": split,
                    "sample_id": sample_id, "native_logits_dtype": str(selected.dtype), "storage_dtype": "float32"}
        metadata = {k: str(v) for k, v in metadata.items()}
        save_file({"logits": tensor}, path, metadata=metadata)
        safe_file(path)
        file_bytes = path.stat().st_size
        _require(self.file_bytes + file_bytes <= self.disk_bound, "Teacher bank disk bound exceeded")
        entry = {"file": path.name, "sha256": _file_hash(path), "tensor_bytes": size,
                 "file_bytes": file_bytes, "shape": list(tensor.shape), "metadata": metadata}
        self.entries[id(example)] = entry
        self.tensor_bytes += size
        self.file_bytes += file_bytes
        from .reconstruction import _release_file_cache
        _release_file_cache(path, sync=True)

    def read(self, example, device):
        from asea.artifacts import safe_file
        from safetensors import safe_open
        entry = self.entries.get(id(example))
        _require(entry is not None and entry["metadata"]["encoding_sha256"] == _json_hash(example),
                 "Unverified teacher bank row/encoding")
        path = safe_file(self.directory / entry["file"])
        _require(path.stat().st_size == entry["file_bytes"] and _file_hash(path) == entry["sha256"],
                 "Teacher bank integrity failure")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            _require(handle.metadata() == entry["metadata"] and list(handle.keys()) == ["logits"],
                     "Teacher bank provenance mismatch")
            tensor = handle.get_tensor("logits").clone()
        _require(list(tensor.shape) == entry["shape"] and str(tensor.dtype) == "torch.float32",
                 "Teacher bank shape/dtype mismatch")
        from .reconstruction import _release_file_cache
        _release_file_cache(path)
        return tensor.to(device=device)

    def receipt(self):
        return {"path": str(self.directory), "provenance": self.provenance, "rows": list(self.entries.values()),
                "actual_tensor_bytes": self.tensor_bytes, "actual_file_bytes": self.file_bytes,
                "storage_dtype": "float32", "full_vocabulary": True, "response_positions_only": True,
                "external_cache_accepted": False, "required_at_serve": False}


def _parameter_hash(named_parameters):
    import torch
    h = hashlib.sha256()
    for name, parameter in named_parameters:
        h.update(name.encode())
        flat = parameter.detach().reshape(-1)
        for offset in range(0, flat.numel(), 1024 * 1024):
            # Bound CPU staging even for a huge embedding on CUDA; avoid tobytes copy.
            part = flat[offset:offset + 1024 * 1024].cpu().contiguous().view(torch.uint8).numpy()
            h.update(memoryview(part))
    return h.hexdigest()


def _publish_no_replace(source, destination):
    # Linux atomic rename with RENAME_NOREPLACE: unlike Path.rename, cannot clobber
    # even an empty destination created by a racing process. Fail closed elsewhere.
    libc = ctypes.CDLL(None, use_errno=True)
    _require(hasattr(libc, "renameat2"), "Atomic no-replace export requires Linux renameat2")
    function = libc.renameat2
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def recover(teacher_dir, student_dir, output_dir, training_path, validation_path, *,
            steps=64, learning_rate=0.0001, rank=8, max_length=256, dtype="bfloat16",
            seed=17, kd_weight=0.7, method="lora_kd", max_samples=256,
            memory_budget_bytes=None, teacher_mode="cached", export_mode="native_merged") -> dict:
    """Train only LoRA; explicitly export merged native or factor-preserving bundle.

    Rejections return status='rejected', artifact_admitted=False and no output.
    See docs/SPECIALIST_RECOVERY.md for the bounded, single-process contract.
    """
    report: Dict[str, Any] = {"schema_version": 1, "status": "rejected", "artifact_admitted": False,
                              "method": method, "export_mode": export_mode, "requested_steps": steps, "actual_steps": 0,
                              "teacher_forward_calls": 0, "validation_teacher_forward_calls": 0,
                              "seed": seed, "quality_claim": "none; loss metrics are not code quality"}
    started = time.monotonic()
    staging = None
    teacher = student = optimizer = bank = None
    try:
        from asea.artifacts import safe_path, safe_file, model_inventory
        from .reconstruction import (_artifact_preflight, _native_meta, _strict_header_match,
                                     _restore_f32_routers, _memory_budget, _headers, _release_file_cache)
        _require(export_mode in ("native_merged", "factor_preserving"),
                 "export_mode must be native_merged or factor_preserving; no implicit fallback")
        _require(teacher_mode in ("cached", "resident"), "teacher_mode must be cached or resident")
        _memory_budget(memory_budget_bytes, available=_available_ram())
        _require(type(steps) is int and 1 <= steps <= 4096, "steps must be in [1,4096]")
        _require(type(rank) is int and 1 <= rank <= 128, "rank must be in [1,128]")
        _require(type(max_length) is int and 4 <= max_length <= 2048, "max_length must be in [4,2048]")
        _require(type(max_samples) is int and 1 <= max_samples <= 4096, "max_samples must be in [1,4096]")
        _require(type(seed) is int and 0 <= seed < 2**32, "seed must be a uint32")
        _require(math.isfinite(learning_rate) and 0 < learning_rate <= 0.1, "Invalid learning_rate")
        _require(math.isfinite(kd_weight) and 0 <= kd_weight <= 1, "Invalid kd_weight")
        _require(method in ("lora_kd", "supervised_lora"), "Unsupported recovery method")
        _require(method != "supervised_lora" or kd_weight == 0,
                 "supervised_lora requires kd_weight=0")
        _require(dtype in ("bfloat16", "float32"), "Only bfloat16 and float32 are supported")
        use_kd = method == "lora_kd" and kd_weight > 0
        source, candidate = safe_path(teacher_dir), safe_path(student_dir)
        raw_output = safe_path(output_dir)
        _require(not raw_output.exists() and not raw_output.is_symlink(), "Output must be a new exclusive path")
        output = raw_output.resolve()
        _require(source.is_dir() and candidate.is_dir(), "Input model directories must exist")
        _require(source != candidate, "Teacher and student stores must be distinct")
        for store in (source, candidate):
            _require(not output.is_relative_to(store) and not store.is_relative_to(output),
                     "Output must not overlap teacher/student stores")
        _require(output.parent.is_dir(), "Output parent must already exist")
        train_path, dev_path = safe_file(training_path), safe_file(validation_path)
        _require(train_path != dev_path, "Training and validation paths must differ")
        train, train_ids, train_hashes = _read_samples(train_path)
        dev, dev_ids, dev_hashes = _read_samples(dev_path)
        _require(not train_ids & dev_ids, "Training/validation id leakage")
        _require(not train_hashes & dev_hashes, "Training/validation exact content leakage")
        train_families, dev_families = _sample_families(train), _sample_families(dev)
        _require(not train_families & dev_families, "Training/validation family leakage")
        report["data"] = {"training_sha256": _file_hash(train_path), "validation_sha256": _file_hash(dev_path),
                          "training_file_samples": len(train), "validation_file_samples": len(dev),
                          "training_families": sorted(train_families), "validation_families": sorted(dev_families),
                          "family_check": "available row family/family_id/task_family fields; external manifests checked by parent",
                          "split_checks": "id, exact prompt/response, available family and complete tokenized content disjoint across entire files"}
        source_inventory, source_raw, source_headers = _artifact_preflight(source)
        candidate_inventory, candidate_raw, candidate_headers = _artifact_preflight(candidate)
        import torch
        import transformers
        import peft
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
        from peft import LoraConfig, TaskType, get_peft_model
        import torch.nn.functional as F
        torch.manual_seed(seed)
        report["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                              "peft": peft.__version__}
        report["determinism"] = "seeded CPU; no GPU validation claim"
        teacher_meta, teacher_plan = _recovery_inputs(source_raw, source_headers, dtype, torch)
        student_meta, student_plan = _recovery_inputs(candidate_raw, candidate_headers, dtype, torch)
        tc, sc = teacher_meta.config, student_meta.config
        native_student_class = type(student_meta)
        report["input_admission"] = {"inventories_checked": True, "bounded_json_and_headers": True,
                                     "native_shapes_keys_tied_aliases_before_weight_load": True}
        family = _family(tc.model_type, sc.model_type)
        report["family"] = family
        report["teacher_model_type"] = tc.model_type
        _require(not (candidate / "adapter_config.json").exists(), "Student must be reconstructed native weights, not adapter")
        teacher_tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(candidate, local_files_only=True, trust_remote_code=False)
        _require(teacher_tokenizer.get_vocab() == tokenizer.get_vocab(), "Teacher/student tokenizer vocabulary mismatch")
        for key in ("eos_token_id", "pad_token_id", "bos_token_id"):
            _require(getattr(teacher_tokenizer, key) == getattr(tokenizer, key), "Tokenizer special token mismatch: " + key)
        _require(tc.vocab_size == sc.vocab_size and max(tokenizer.get_vocab().values()) < sc.vocab_size,
                 "Model logit vocabularies must match tokenizer IDs")
        if family == "seq2seq":
            _require(tc.decoder_start_token_id == sc.decoder_start_token_id and sc.decoder_start_token_id is not None,
                     "Seq2seq decoder start tokens must match")
        contract = _encoding_contract(tokenizer, family)
        _require(contract == _encoding_contract(teacher_tokenizer, family), "Teacher/student encoding template mismatch")
        report["encoding"] = contract
        # Validate EVERY complete train/dev example before weights, optimizer, or
        # random subsampling. Invalid unselected rows must not escape admission.
        encoded_splits = []
        for rows in (train, dev):
            complete = []
            for row in rows:
                example, evidence = _encode_details(row, tokenizer, family, max_length)
                _require(example == _encode(row, teacher_tokenizer, family, max_length),
                         "Teacher/student complete tokenization mismatch")
                complete.append((row, example, evidence))
            encoded_splits.append(complete)
        token_hashes = [{_json_hash(item[1]) for item in split} for split in encoded_splits]
        _require(not token_hashes[0] & token_hashes[1], "Training/validation complete tokenized content leakage")
        report["data"]["tokenized_split_disjoint"] = True
        report["data"]["all_rows_length_checked_before_weight_load"] = True
        report["data"]["max_complete_input_tokens"] = max(item[2]["input_tokens"] for split in encoded_splits for item in split)
        report["data"]["max_complete_label_tokens"] = max(item[2]["label_tokens"] for split in encoded_splits for item in split)
        report["data"]["encoding_audit"] = {name: [item[2] for item in split]
                                              for name, split in zip(("training", "validation"), encoded_splits)}
        rng = random.Random(seed)
        for split in encoded_splits:
            rng.shuffle(split)
        chosen_train, chosen_dev = [split[:max_samples] for split in encoded_splits]
        train, dev = [item[0] for item in chosen_train], [item[0] for item in chosen_dev]
        encoded_train, encoded_dev = [item[1] for item in chosen_train], [item[1] for item in chosen_dev]
        report["data"].update(training_samples=len(train), validation_samples=len(dev),
                              training_ids=[r["id"] for r in train], validation_ids=[r["id"] for r in dev])
        targets = ["q_proj", "v_proj", "down_proj"] if family == "causal" else ["q", "v", "wo"]
        actual_length = max(max(len(e["input_ids"]), len(e["labels"])) for e in encoded_train + encoded_dev)
        response_counts = [sum(v != -100 for v in (e["labels"][1:] if family == "causal" else e["labels"]))
                           for e in encoded_train + encoded_dev]
        report["resources"] = _preflight(source, candidate, output.parent, sc, rank, targets,
                                          2 if dtype == "bfloat16" else 4, actual_length, use_kd, torch,
                                          memory_budget_bytes, teacher_mode, max(response_counts),
                                          sum(response_counts), len(response_counts), tc, export_mode,
                                          (source_inventory, source_headers, teacher_plan),
                                          (candidate_inventory, candidate_headers, student_plan))
        report["source_hashes_before"] = {k: v["sha256"] for k, v in source_inventory.items()}
        report["student_store_hashes_before"] = {k: v["sha256"] for k, v in candidate_inventory.items()}
        device = torch.device("cpu")
        report["device"] = str(device)
        report["hyperparameters"] = {"learning_rate": learning_rate, "rank": rank, "max_length": max_length,
                                       "dtype": dtype, "kd_weight": kd_weight, "max_samples": max_samples,
                                       "batch_size": 1, "gradient_clip_norm": 1.0, "temperature": 1.0,
                                       "teacher_mode": teacher_mode, "memory_budget_bytes": memory_budget_bytes}
        model_class = AutoModelForCausalLM if family == "causal" else AutoModelForSeq2SeqLM
        load = dict(local_files_only=True, trust_remote_code=False, use_safetensors=True,
                    torch_dtype=getattr(torch, dtype), low_cpu_mem_usage=True, device_map={"": "cpu"}, attn_implementation="eager")
        def load_native(path):
            nonlocal teacher_meta, student_meta
            if path in (source, candidate):
                is_teacher = path == source
                meta = teacher_meta if is_teacher else student_meta
                headers = source_headers if is_teacher else candidate_headers
                plan = teacher_plan if is_teacher else student_plan
                model = _load_recovery_model(meta, path, headers, plan, torch)
                if is_teacher:
                    teacher_meta = None  # no hidden reference surviving cached teardown
                    report["teacher_router_precision"] = _restore_f32_routers(model, source, source_headers, torch)
                else:
                    student_meta = None
                expected = report["resources"]["teacher" if is_teacher else "student"]["loaded_bytes"]
                _require(sum(p.numel() * p.element_size() for p in model.parameters()) == expected,
                         "Materialized recovery dtype/alias plan disagreement")
                return model
            # Independent native export validation deliberately retains the prior
            # HF loader, tolerances and computation graph; not a training input.
            model, information = model_class.from_pretrained(path, output_loading_info=True, **load)
            _require(not any(information.get(key) for key in
                             ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
                     "Native checkpoint is incomplete/incompatible: " + str(information))
            return model.to(device)

        staging = Path(tempfile.mkdtemp(prefix="." + output.name + ".recovery-", dir=output.parent))
        # Teacher-only phase. No student weights, adapters, gradients or optimizer exist.
        if use_kd:
            _phase_guard(report["resources"], "teacher-load", report["resources"]["teacher_phase_estimated_bytes"], memory_budget_bytes)
            teacher = load_native(source)
            teacher.requires_grad_(False)
            teacher.eval()
            teacher.config.use_cache = False
            teacher_hash = _parameter_hash(teacher.named_parameters())
            if teacher_mode == "cached":
                provenance = {"model_files_sha256": _json_hash(report["source_hashes_before"]),
                              "training_sha256": report["data"]["training_sha256"],
                              "validation_sha256": report["data"]["validation_sha256"],
                              "encoding_contract_sha256": _json_hash(contract),
                              "loaded_teacher_parameters_sha256": teacher_hash}
                bank = _TeacherBank(staging / ".teacher-bank", provenance, int(sc.vocab_size),
                                    report["resources"]["bank_tensor_bytes_bound"], report["resources"]["bank_disk_bytes_bound"])
                for split, rows, examples in (("training", train, encoded_train), ("validation", dev, encoded_dev)):
                    for row, example in zip(rows, examples):
                        _require(time.monotonic() - started < 86400, "Teacher-cache wall-clock budget exhausted")
                        _phase_guard(report["resources"], "teacher-cache-forward", report["resources"]["teacher_workspace_bytes"], memory_budget_bytes)
                        batch, labels = _native_batch(example, teacher, family, torch, device)
                        with torch.no_grad():
                            teacher_output = teacher(**batch, use_cache=False)
                            selected, _ = _response_logits(teacher_output.logits, labels, family)
                            bank.write(example, selected, split, row["id"])
                        del teacher_output, selected, batch, labels
                        report["teacher_forward_calls" if split == "training" else "validation_teacher_forward_calls"] += 1
                _require(all(not p.requires_grad and p.grad is None for p in teacher.parameters()), "Teacher acquired gradients")
                _require(teacher_hash == _parameter_hash(teacher.named_parameters()), "Teacher weights changed during cache generation")
                _require(report["source_hashes_before"] == _store_hash(source), "Teacher files changed during cache generation")
                report["teacher_bank"] = bank.receipt()
                report["teacher_frozen_no_grad"] = True
                teacher_refs = [("teacher", weakref.ref(teacher))] + [
                    ("teacher." + n, weakref.ref(p)) for n, p in teacher.named_parameters()]
                del teacher
                teacher = None
                _assert_collected(teacher_refs, report["resources"], "cached-teacher-release")
                for name in source_inventory:
                    if name.endswith(".safetensors"):
                        _release_file_cache(source / name)
                report["teacher_bank"]["file_cache_policy"] = "fsync private writes and best-effort POSIX_FADV_DONTNEED; cloned one-row reads; observed limits still enforced"
                # Return free CPU heap pages when libc supports it; never alter limits.
                libc = ctypes.CDLL(None)
                if hasattr(libc, "malloc_trim"):
                    libc.malloc_trim(0)
                report["teacher_unloaded_before_student"] = True
        # Reobserve free RAM after teacher deletion, not just at initial admission.
        report["resources"]["student_phase_recheck"] = _phase_guard(
            report["resources"], "student-phase", report["resources"]["student_phase_incremental_bytes"], memory_budget_bytes)
        student = load_native(candidate)
        deployment_cache = getattr(student.config, "use_cache", True)
        student.config.use_cache = False
        student = get_peft_model(student, LoraConfig(r=rank, lora_alpha=2 * rank,
                                 lora_dropout=0.0, bias="none", target_modules=targets,
                                 task_type=TaskType.CAUSAL_LM if family == "causal" else TaskType.SEQ_2_SEQ_LM))
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()
        adapters = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
        _require(adapters and all("lora_" in n for n, _ in adapters), "Non-adapter parameter marked trainable")
        initial = {n: p.detach().float().cpu().clone() for n, p in adapters}
        report["adapter_hash_before"] = _parameter_hash(adapters)
        frozen = [(n, p) for n, p in student.named_parameters() if not p.requires_grad]
        report["frozen_parameter_hash_before"] = _parameter_hash(frozen)
        native_target_before = {name: _parameter_hash([("weight", module.base_layer.weight)])
                                for name, module in student.get_base_model().named_modules()
                                if hasattr(module, "lora_A") and hasattr(module, "base_layer")}
        report["trainable_parameters"] = sum(p.numel() for _, p in adapters)
        report["total_parameters_with_adapter"] = sum(p.numel() for p in student.parameters())
        optimizer = torch.optim.AdamW([p for _, p in adapters], lr=learning_rate, weight_decay=0.0, foreach=False)

        def losses(example, validation=False):
            _require(time.monotonic() - started < 86400, "One-day recovery wall-clock budget exhausted")
            _phase_guard(report["resources"], "student-forward-backward",
                         report["resources"]["student_workspace_bytes"] +
                         (report["resources"]["teacher_workspace_bytes"] if teacher is not None else 0), memory_budget_bytes)
            batch, labels = _native_batch(example, student.get_base_model(), family, torch, device)
            teacher_selected = bank.read(example, device) if bank is not None else None
            if teacher is not None:
                with torch.no_grad():
                    teacher_output = teacher(**batch, use_cache=False)
                    teacher_selected, _ = _response_logits(teacher_output.logits, labels, family)
                    del teacher_output
                key = "validation_teacher_forward_calls" if validation else "teacher_forward_calls"
                report[key] += 1
            output_logits = student(**batch, use_cache=False).logits
            selected, target = _response_logits(output_logits, labels, family)
            ce = F.cross_entropy(selected.float(), target, reduction="mean")
            kl = _full_kl(selected, teacher_selected) if teacher_selected is not None else None
            objective = ce if kl is None else (1 - kd_weight) * ce + kd_weight * kl
            _require(bool(torch.isfinite(objective)) and bool(torch.isfinite(ce)) and
                     (kl is None or bool(torch.isfinite(kl))), "Nonfinite loss")
            return objective, ce, kl, int(target.numel())

        def evaluate():
            student.eval()
            ce_sum = kl_sum = objective_sum = tokens = 0
            with torch.no_grad():
                for example in encoded_dev:
                    objective, ce, kl, count = losses(example, validation=True)
                    tokens += count
                    ce_sum += float(ce) * count
                    kl_sum += (float(kl) if kl is not None else 0) * count
                    objective_sum += float(objective) * count
            return {"response_tokens": tokens, "supervised_ce": ce_sum / tokens,
                    "forward_kl": kl_sum / tokens if use_kd else None,
                    "objective": objective_sum / tokens, "aggregation": "response-token-weighted mean"}

        report["validation_pre"] = evaluate()
        student.train()
        history = []
        report["training_history"] = history
        for step in range(steps):
            _require(time.monotonic() - started < 86400, "One-day recovery wall-clock budget exhausted")
            optimizer.zero_grad(set_to_none=True)
            objective, ce, kl, count = losses(encoded_train[step % len(encoded_train)])
            objective.backward()
            grads = [p.grad for _, p in adapters if p.grad is not None]
            _require(grads and all(bool(torch.isfinite(g).all()) for g in grads), "Missing/nonfinite adapter gradients")
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in adapters], 1.0, error_if_nonfinite=True)
            _require(float(norm) > 0, "Zero adapter gradient norm")
            optimizer.step()
            _require(all(bool(torch.isfinite(p).all()) for _, p in adapters), "Nonfinite updated adapter")
            report["actual_steps"] += 1
            history.append({"step": step + 1, "sample_id": train[step % len(train)]["id"],
                            "response_tokens": count, "objective": float(objective.detach()),
                            "supervised_ce": float(ce.detach()), "forward_kl": float(kl.detach()) if kl is not None else None,
                            "gradient_norm_pre_clip": float(norm)})
        report["training_history"] = history
        token_total = sum(x["response_tokens"] for x in history)
        report["training"] = {"response_tokens": token_total,
                              "supervised_ce": sum(x["supervised_ce"] * x["response_tokens"] for x in history) / token_total,
                              "forward_kl": sum(x["forward_kl"] * x["response_tokens"] for x in history) / token_total if use_kd else None}
        report["validation_post"] = evaluate()
        report["validation_post"]["model_state"] = "trained_adapter_premerge; not reloaded deployment validation"
        report["adapter_hash_after"] = _parameter_hash(adapters)
        delta = sum(float((p.detach().float().cpu() - initial[n]).square().sum()) for n, p in adapters)
        report["adapter_delta_l2"] = math.sqrt(delta)
        report["frozen_parameter_hash_after"] = _parameter_hash(frozen)
        _require(report["frozen_parameter_hash_before"] == report["frozen_parameter_hash_after"], "Frozen base weights changed")
        _require(delta > 0 and report["adapter_hash_before"] != report["adapter_hash_after"], "No adapter parameter update")
        report["source_hashes_after"] = _store_hash(source)
        report["student_store_hashes_after"] = _store_hash(candidate)
        _require(report["source_hashes_before"] == report["source_hashes_after"], "Teacher store changed during training")
        _require(report["student_store_hashes_before"] == report["student_store_hashes_after"], "Student source store changed during training")
        report["source_unchanged"] = report["student_store_unchanged"] = True
        report["teacher_frozen_no_grad"] = teacher is None or all(
            not p.requires_grad and p.grad is None for p in teacher.parameters())
        _require(report["teacher_frozen_no_grad"], "Teacher acquired gradients")
        if teacher is not None:
            _require(teacher_hash == _parameter_hash(teacher.named_parameters()), "Resident teacher weights changed")
        optimizer.zero_grad(set_to_none=True)
        if export_mode == "factor_preserving":
            training_refs = [("student", weakref.ref(student)), ("optimizer", weakref.ref(optimizer))]
            training_refs += [("student." + n, weakref.ref(p)) for n, p in student.named_parameters()]
            if teacher is not None:
                training_refs += [("teacher", weakref.ref(teacher))]
                training_refs += [("teacher." + n, weakref.ref(p)) for n, p in teacher.named_parameters()]
        del optimizer, teacher, initial
        optimizer = teacher = None
        student.gradient_checkpointing_disable()
        if hasattr(student, "disable_input_require_grads"):
            student.disable_input_require_grads()
        # One held-out teacher-forced position, full vocabulary, eval/no-cache.
        # This is a numerical merge/export probe, not generation/quality parity.
        probe_example = encoded_dev[0]
        def parity_logits(model, native):
            batch = {key: torch.tensor([value], dtype=torch.long, device=device)
                     for key, value in probe_example.items() if key != "labels"}
            labels = torch.tensor([probe_example["labels"]], dtype=torch.long, device=device)
            if family == "seq2seq":
                batch["decoder_input_ids"] = native._shift_right(labels)
                batch["decoder_attention_mask"] = torch.ones_like(labels)
            model.eval()
            with torch.no_grad():
                logits = model(**batch, use_cache=False).logits
                selected, _ = _response_logits(logits, labels, family)
                return selected[-1].float().cpu().clone()
        _phase_guard(report["resources"], "export-write",
                     report["resources"]["export_workspace_bytes"] + report["resources"]["merge_workspace_bytes"] +
                     report["resources"]["merge_cow_source_pages_bytes"], memory_budget_bytes)
        if export_mode == "factor_preserving":
            from .standalone import _write_bundle, load_standalone
            if bank is not None:
                shutil.rmtree(bank.directory)
                report["teacher_bank"]["removed_before_export"] = True
                bank = None
            student.config.use_cache = deployment_cache
            # Full teacher-forced forward and short deterministic generation on
            # permitted dev data only. Neither probe is a quality measurement.
            if family == "causal":
                prefix = (tokenizer.apply_chat_template([{"role": "user", "content": dev[0]["prompt"]}],
                          tokenize=False, add_generation_prompt=True) if tokenizer.chat_template
                          else dev[0]["prompt"] + "\n")
            else:
                prefix = dev[0]["prompt"]
            generation_inputs = tokenizer(prefix, add_special_tokens=family == "seq2seq",
                truncation=False, return_tensors="pt", return_token_type_ids=False)
            def factor_probe(model):
                model.eval()
                batch, _ = _native_batch(probe_example, model.get_base_model(), family, torch, device)
                with torch.no_grad():
                    # CPU no-grad output owns its storage; retaining it needs no
                    # second full-logit clone. No model/parameter view is returned.
                    full = model(**batch, use_cache=False).logits.detach().cpu()
                    generated = model.generate(**generation_inputs, do_sample=False, max_new_tokens=2,
                                               use_cache=deployment_cache).detach().cpu().clone()
                return full, generated
            before_logits, before_generated = factor_probe(student)
            manifest = _write_bundle(staging, student, tokenizer, report, deployment_cache, candidate)
            report["trained_artifact_capture"] = {"before_merge": True, "merge_called": False,
                "adapter_parameters_sha256": report["adapter_hash_after"],
                "adapter_file_sha256": manifest["files"]["adapter/adapter_model.safetensors"]["sha256"]}
            report["bundle_counts"] = manifest["counts"]
            report["size_comparison"] = {
                "parameter_ratio_to_original_teacher": manifest["counts"]["total_parameters"] / manifest["counts"]["original_teacher_parameters"],
                "safetensors_ratio_to_original_teacher": manifest["counts"]["total_safetensors_bytes"] / manifest["counts"]["original_teacher_safetensors_bytes"],
                "smaller_parameters_than_teacher": manifest["counts"]["total_parameters"] < manifest["counts"]["original_teacher_parameters"],
                "smaller_safetensors_than_teacher": manifest["counts"]["total_safetensors_bytes"] < manifest["counts"]["original_teacher_safetensors_bytes"]}
            report["output_weight_sha256"] = {k: v["sha256"] for k, v in manifest["files"].items() if k.endswith(".safetensors")}
            # Release every training weight/gradient reference before fresh load.
            del student, frozen, adapters, objective, ce, kl, grads
            student = None
            release = _assert_collected(training_refs, report["resources"], "factor-export-reload")
            # Mapped training weights can prevent earlier cache advice taking
            # effect. Retry only after verified unmapping; still grant zero free
            # credit and make the following dynamic observation authoritative.
            for directory, inventory in ((source, source_inventory), (candidate, candidate_inventory)):
                for name in inventory:
                    if name.endswith(".safetensors"):
                        _release_file_cache(directory / name)
            libc = ctypes.CDLL(None)
            if hasattr(libc, "malloc_trim"):
                libc.malloc_trim(0)
            release["post_collection_cache_and_heap_release"] = "best-effort only; no credited bytes"
            _phase_guard(report["resources"], "export-reload", report["resources"]["export_reload_estimated_bytes"], memory_budget_bytes)
            bundle = load_standalone(staging, dtype=dtype)
            merged = bundle.model  # common publication code; this model is NOT merged
            reloaded_tokenizer = bundle.tokenizer
            _require(_parameter_hash((n, p) for n, p in merged.named_parameters() if "lora_" in n)
                     == report["adapter_hash_after"], "Reload changed trained adapter parameters")
            _require(_parameter_hash((n, p) for n, p in merged.named_parameters() if "lora_" not in n)
                     == report["frozen_parameter_hash_after"], "Reload changed frozen base parameters")
            _require(reloaded_tokenizer.get_vocab() == tokenizer.get_vocab() and
                     _encoding_contract(reloaded_tokenizer, family) == contract, "Bundle tokenizer changed")
            for key in ("eos_token_id", "pad_token_id", "bos_token_id"):
                _require(getattr(reloaded_tokenizer, key) == getattr(tokenizer, key), "Bundle special IDs changed")
            for row, example in ((item[0], item[1]) for split in encoded_splits for item in split):
                _require(_encode(row, reloaded_tokenizer, family, max_length) == example,
                         "Bundle complete encoding changed")
            _phase_guard(report["resources"], "factor-reload-forward",
                         report["resources"]["export_full_probe_workspace_bytes"], memory_budget_bytes)
            after_logits, after_generated = factor_probe(merged)
            forward_equal = bool(torch.equal(before_logits, after_logits))
            generation_equal = bool(torch.equal(before_generated, after_generated))
            report["standalone_reload_probe"] = {"passed": forward_equal and generation_equal,
                "computation_graph": "native base plus native PEFT LoRA; no weight merge",
                "scope": "one complete dev teacher-forced forward and two-token greedy generation fixture; not quality validation",
                "sample_id": dev[0]["id"], "atol": 0.0, "rtol": 0.0, "exact_equality_required": True,
                "forward_bitwise_equal": forward_equal, "generation_bitwise_equal": generation_equal,
                "max_abs_error": float((before_logits.float() - after_logits.float()).abs().max()),
                "forward_shape": list(before_logits.shape), "generation_max_new_tokens": 2,
                "generation_input_ids": generation_inputs["input_ids"].tolist(),
                "generation_before_ids": before_generated.tolist(), "generation_after_ids": after_generated.tolist(),
                "cache_restored": merged.config.use_cache, "quality_claim": False}
            _require(forward_equal and generation_equal, "Factor-preserving fresh reload parity failed")
            _require(merged.config.use_cache == deployment_cache, "Bundle deployment cache changed")
            report["standalone_tokenizer_verified"] = True
            report["standalone_inventory_and_native_keys_verified"] = True
        else:
            before_merge_logits = parity_logits(student, student.get_base_model())
            merged = student.merge_and_unload(safe_merge=True)
            native_target_after = {name: _parameter_hash([("weight", merged.get_submodule(name).weight)])
                                   for name in native_target_before}
            changed = sum(native_target_before[name] != native_target_after[name] for name in native_target_before)
            report["native_target_hashes_before"] = native_target_before
            report["native_target_hashes_after"] = native_target_after
            report["merged_native_matrices_changed"] = changed
            _require(changed > 0, "Merged native weights unchanged; adapter update rounded away at export dtype")
            merged.config.use_cache = deployment_cache
            merged.config._name_or_path = ""
            merged.eval()
            after_merge_logits = parity_logits(merged, merged)
            atol, rtol = ((0.02, 0.05) if dtype == "bfloat16" else (2e-5, 2e-5))
            merge_close = bool(torch.allclose(before_merge_logits, after_merge_logits, atol=atol, rtol=rtol))
            report["merge_parity_probe"] = {"sample_id": dev[0]["id"], "scope": "last supervised teacher-forced position, full vocabulary",
                                             "atol": atol, "rtol": rtol, "passed": merge_close,
                                             "max_abs_error": float((before_merge_logits - after_merge_logits).abs().max()),
                                             "exact_equality_required": False, "generation_parity_claim": False}
            _require(merge_close, "Pre/post-merge numerical parity probe exceeded dtype tolerance")
            _require(type(merged) is native_student_class, "Merged model is not the native student architecture")
            _require(not getattr(merged, "_hf_peft_config_loaded", False), "Adapter-only save flag remains set")
            _require(not any("lora_" in n for n, _ in merged.named_parameters()), "Merged native model retains adapter parameters")
            if bank is not None:
                shutil.rmtree(bank.directory)
                report["teacher_bank"]["removed_before_export"] = True
                bank = None
            merged.save_pretrained(staging, safe_serialization=True, max_shard_size="1GB")
            tokenizer.save_pretrained(staging)
            _require(not (staging / "adapter_config.json").exists(), "Export retained adapter dependency")
            report["output_weight_sha256"] = {p.name: _file_hash(p) for p in sorted(staging.glob("*.safetensors"))}
            _require(report["output_weight_sha256"], "Export has no native safetensors")
            export_inventory, export_raw, export_headers = _artifact_preflight(staging)
            _strict_header_match(merged, export_headers)
            reloaded_tokenizer = AutoTokenizer.from_pretrained(staging, local_files_only=True, trust_remote_code=False)
            _require(reloaded_tokenizer.get_vocab() == tokenizer.get_vocab(), "Standalone tokenizer vocabulary changed")
            _require(_encoding_contract(reloaded_tokenizer, family) == contract, "Standalone tokenizer template changed")
            for key in ("eos_token_id", "pad_token_id", "bos_token_id"):
                _require(getattr(reloaded_tokenizer, key) == getattr(tokenizer, key), "Standalone tokenizer special IDs changed")
            for row, example in ((item[0], item[1]) for split in encoded_splits for item in split):
                _require(_encode(row, reloaded_tokenizer, family, max_length) == example, "Standalone complete encoding changed")
            report["standalone_tokenizer_verified"] = True
            report["standalone_inventory_and_native_keys_verified"] = True
            # Release base-weight references before a separate standalone reload; no
            # second resident full model solely for export verification.
            del student, merged, frozen, adapters, objective, ce, kl, grads
            student = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            _phase_guard(report["resources"], "export-reload", report["resources"]["export_reload_estimated_bytes"], memory_budget_bytes)
            merged = load_native(staging)
            _require(merged.config.use_cache == deployment_cache, "Saved deployment use_cache was not restored")
            _require(not any("lora_" in name for name, _ in merged.named_parameters()), "Reload retained LoRA dependency")
            reloaded_logits = parity_logits(merged, merged)
            reload_close = bool(torch.allclose(after_merge_logits, reloaded_logits, atol=atol, rtol=rtol))
            report["standalone_reload_probe"] = {"passed": reload_close, "atol": atol, "rtol": rtol,
                                                  "cache_restored": merged.config.use_cache,
                                                  "max_abs_error": float((after_merge_logits - reloaded_logits).abs().max()),
                                                  "scope": "same teacher-forced position; not full post-export validation"}
            _require(reload_close, "Standalone reload numerical parity probe exceeded tolerance")
        # Include merging/serialization in the immutability window (important for mmap-backed loads).
        report["source_hashes_after"] = _store_hash(source)
        report["student_store_hashes_after"] = _store_hash(candidate)
        _require(report["source_hashes_before"] == report["source_hashes_after"], "Teacher store changed during export")
        _require(report["student_store_hashes_before"] == report["student_store_hashes_after"], "Student source store changed during export")
        _require(time.monotonic() - started < 86400, "One-day recovery wall-clock budget exhausted")
        report.update(status="completed", artifact_admitted=True, output_dir=str(output),
                      standalone_native=export_mode == "native_merged", standalone_teacher_independent=True,
                      merged=export_mode == "native_merged", factor_preserving=export_mode == "factor_preserving",
                      cache_restored=merged.config.use_cache,
                      elapsed_seconds=time.monotonic() - started)
        # 'artifact_admitted' means successfully exported, never promotion/quality admission.
        (staging / "recovery_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        _publish_no_replace(staging, output)
        staging = None
    except Exception as exc:
        report.update(status="rejected", artifact_admitted=False, standalone_native=False,
                      error={"type": type(exc).__name__, "message": str(exc)},
                      elapsed_seconds=time.monotonic() - started)
        report.pop("output_dir", None)
        if "source_hashes_before" in report:
            try:
                report["source_hashes_after"] = _store_hash(source)
                report["student_store_hashes_after"] = _store_hash(candidate)
                report["source_unchanged"] = report["source_hashes_before"] == report["source_hashes_after"]
                report["student_store_unchanged"] = report["student_store_hashes_before"] == report["student_store_hashes_after"]
            except Exception:
                report["source_unchanged"] = report["student_store_unchanged"] = None
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    return report
