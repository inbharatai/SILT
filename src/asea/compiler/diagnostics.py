"""Pass-2 Switch research audits, deliberately separate from v1 certification.

The coordinator never loads a model. Each native/wrapper/save/reload arm runs in
its own fresh interpreter, with only one local checkpoint resident at a time.
Short teacher logits live in temporary .npy files for exact bounded comparison;
only scalar summaries and token traces survive in the public report.
"""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import selectors
import signal
import shutil
import subprocess
import sys
import tempfile
import time

from .core import (CompilerError, RoutingTelemetry, apply_selection, canonical_hash,
                   digest, fail, generate, inspect_model, inventory, load_model,
                   loaded_structure, observed_resources, preflight, read_json,
                   restore_router_fp32, runtime, runtime_versions, safe_path,
                   sparse_layers, validate_precision_lineage, write_json)

SCHEMA = "switch-span-diagnostics-v2"
SCOPE = "audit_only_span_denoising_NOT_quality_or_coding_certification"
NORMALIZER = "whitespace-collapse-v1; no punctuation removal; no substring matching"
MAX_REPORT_BYTES = 32 * 1024 * 1024
DEFAULT_WORKER_TIMEOUT_SECONDS = 300
DEFAULT_WORKER_OUTPUT_LIMIT_BYTES = 1024 * 1024
# Virtual address space, including mapped checkpoint files; NOT RSS/aggregate.
DEFAULT_WORKER_ADDRESS_SPACE_MIB = 8192
_TOKENIZER_EOS = object()
DEFAULT_SUITE = {"schema": SCHEMA, "split": "diagnostic", "cases": [
    {"id": "default-mechanics-only", "prompt": "The <extra_id_0> walks in the park.",
     "target": "<extra_id_0> man <extra_id_1>", "expected_spans": ["man"],
     "references": ["<extra_id_0> man <extra_id_1>"], "provenance": "mechanics_only_not_heldout"}]}


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    return value


def validate_suite(data):
    if not isinstance(data, dict) or data.get("schema") != SCHEMA or data.get("split") != "diagnostic":
        fail("INVALID_DIAGNOSTIC_SUITE", "Require schema=switch-span-diagnostics-v2 and split=diagnostic; not a v1 heldout suite")
    rows = data.get("cases")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 32:
        fail("INVALID_DIAGNOSTIC_SUITE", "Require 1..32 diagnostic cases")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or row["id"] in seen:
            fail("INVALID_DIAGNOSTIC_SUITE", "Require unique nonempty string ids")
        seen.add(row["id"])
        for field in ("prompt", "target"):
            if not isinstance(row.get(field), str) or not row[field].strip() or len(row[field]) > 4096:
                fail("INVALID_DIAGNOSTIC_SUITE", "Require nonempty prompt and teacher target <=4096 characters")
        spans = row.get("expected_spans")
        if not isinstance(spans, list) or not 1 <= len(spans) <= 8 or any(not isinstance(s, str) or len(s) > 1024 for s in spans):
            fail("INVALID_DIAGNOSTIC_SUITE", "Require expected_spans: 1..8 strings in sentinel order")
        refs = row.get("references", [row["target"]])
        if not isinstance(refs, list) or not 1 <= len(refs) <= 32 or any(not isinstance(s, str) or len(s) > 4096 for s in refs):
            fail("INVALID_DIAGNOSTIC_SUITE", "references must be original strings")
        if row.get("group", "target") not in ("target", "control"):
            fail("INVALID_DIAGNOSTIC_SUITE", "group must be target or control")
        pattern = r"<extra_id_(\d+)>"
        prompt_order = [int(n) for n in re.findall(pattern, row["prompt"])]
        chunks = re.split(pattern, row["target"])
        target_order = [int(n) for n in chunks[1::2]]
        if (prompt_order != list(range(len(spans))) or target_order != list(range(len(spans) + 1)) or
                chunks[0].strip() or chunks[-1].strip() not in ("", "</s>") or
                [normalize(s) for s in chunks[2:-1:2]] != [normalize(s) for s in spans]):
            fail("INVALID_DIAGNOSTIC_SUITE", "Prompt/target sentinel order and teacher target must exactly match expected_spans")
    if len(json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")) > 256 * 1024:
        fail("INVALID_DIAGNOSTIC_SUITE", "Diagnostic suite exceeds 256 KiB serialized ceiling")
    return data


def diagnostic_bounds(dtype, max_length, max_new_tokens, teacher_tokens, atol, rtol):
    if dtype not in ("float16", "bfloat16"):
        fail("INVALID_DTYPE", "Research audits support float16 or bfloat16 only; full FP32 is not a 4GB control")
    if not 1 <= max_length <= 64 or not 1 <= max_new_tokens <= 32 or not 1 <= teacher_tokens <= 32:
        fail("INVALID_ARGUMENT", "Diagnostics require max_length 1..64, max_new_tokens 1..32, teacher_tokens 1..32")
    if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 or v > 1 for v in (atol, rtol)):
        fail("INVALID_ARGUMENT", "Numeric tolerances must be finite in [0,1] and prespecified")


def destination(output, roots):
    dest = safe_path(output, False)
    if dest.exists():
        fail("OUTPUT_EXISTS", "Audit output must not exist")
    if not dest.parent.is_dir():
        fail("UNSAFE_OUTPUT", "Audit output parent must already exist")
    for root in roots:
        if root is not None:
            source = safe_path(root)
            if source == dest or source in dest.parents:
                fail("UNSAFE_OUTPUT", "Audit output must be outside every source model tree")
    return dest


def normalize(text):
    return " ".join(text.split())


def sentinel_ids(tokenizer):
    return {int(idx): int(re.fullmatch(r"<extra_id_(\d+)>", token).group(1))
            for token, idx in tokenizer.get_vocab().items() if re.fullmatch(r"<extra_id_(\d+)>", token)}


def parse_spans(ids, tokenizer, expected_count, generation_eos_ids=_TOKENIZER_EOS,
                decoder_start_token_id=None, require_decoder_start=True):
    """Strict decoder-start, ordered closed spans, terminal EOS, optional pads.

    `valid` is the backwards-compatible span-structure flag, NOT full correctness.
    Text after the closing sentinel is retained and always fails full correctness.
    """
    vocab = set(tokenizer.get_vocab().values())
    malformed = not isinstance(ids, list) or any(type(t) is not int or t not in vocab for t in ids)
    ids = ids if isinstance(ids, list) else []
    mapping = sentinel_ids(tokenizer)
    eos = tokenizer.eos_token_id if generation_eos_ids is _TOKENIZER_EOS else generation_eos_ids
    eos = eos if isinstance(eos, list) else ([] if eos is None else [eos])
    start_id = tokenizer.pad_token_id if decoder_start_token_id is None else decoder_start_token_id
    structural = set(getattr(tokenizer, "all_special_ids", [])) - set(mapping)
    structural.update(t for t in [tokenizer.pad_token_id, tokenizer.eos_token_id, *eos] if t is not None)
    def decode(tokens):
        if any(type(t) is not int or t not in vocab for t in tokens):
            return "<invalid-token-ids>"
        return tokenizer.decode(tokens, skip_special_tokens=False)
    boundaries = [(pos, mapping[token]) for pos, token in enumerate(ids) if type(token) is int and token in mapping]
    order = [number for _, number in boundaries]
    expected = list(range(expected_count + 1))
    prefix = ids[:boundaries[0][0]] if boundaries else ids
    trailing = ids[boundaries[-1][0] + 1:] if boundaries else []
    prefix_valid = prefix == ([start_id] if require_decoder_start else [])
    eos_positions = [i for i, token in enumerate(ids) if token in eos and not (require_decoder_start and i == 0 and token == start_id)]
    premature_eos = bool(eos_positions and (not boundaries or eos_positions[0] <= boundaries[-1][0]))
    post_eos = ids[eos_positions[0] + 1:] if eos_positions else []
    terminal_valid = (len(eos_positions) == 1 and not premature_eos and
                      all(t == tokenizer.pad_token_id for t in post_eos)) if eos else False
    trailing_content = [t for t in trailing if t not in eos and t != tokenizer.pad_token_id]
    # Padding is permitted ONLY after terminal EOS, never between close and EOS.
    tail_exact = bool(trailing and trailing[0] in eos and terminal_valid)
    spans = []
    for (start, number), (end, following) in zip(boundaries, boundaries[1:]):
        tokens = ids[start + 1:end]
        spans.append({"sentinel": number, "next_sentinel": following, "token_ids": tokens,
                      "text": decode(tokens),
                      "contains_structural_token": any(type(t) is not int or t in structural for t in tokens)})
    valid = (not malformed and order == expected and prefix_valid and not premature_eos and
             not any(s["contains_structural_token"] for s in spans))
    complete = valid and tail_exact and not trailing_content
    return {"parser_version": "t5-token-sentinel-spans-v3", "valid": valid,
            "span_structure_valid": valid, "complete_generation_well_formed": complete,
            "malformed_token_ids": malformed, "decoder_start_prefix_valid": prefix_valid,
            "premature_eos": premature_eos, "terminal_grammar_valid": terminal_valid,
            "content_after_eos": any(t != tokenizer.pad_token_id for t in post_eos),
            "generation_eos_token_ids": eos,
            "sentinel_order": order, "expected_sentinel_order": expected,
            "missing_sentinels": [i for i in expected if i not in order],
            "extra_sentinels": [i for i in order if i not in expected],
            "misordered_or_duplicate": order != sorted(set(order)),
            "spans": spans, "prefix_token_ids": prefix, "trailing_token_ids": trailing,
            "trailing_text": decode(trailing_content),
            "full_output_well_formed": complete}


def token_trace(ids, tokenizer, expected_spans, max_new_tokens, generation_eos_ids=_TOKENIZER_EOS, decoder_start_token_id=None):
    if not isinstance(ids, list) or any(type(t) is not int or t not in tokenizer.get_vocab().values() for t in ids):
        fail("INVALID_GENERATION_TOKENS", "Generation must contain only known integer token IDs")
    # Native seq2seq generate returns decoder-start plus new tokens.
    generated = ids[1:]
    eos = tokenizer.eos_token_id
    actual_eos = eos if generation_eos_ids is _TOKENIZER_EOS else generation_eos_ids
    actual_eos = actual_eos if isinstance(actual_eos, list) else ([] if actual_eos is None else [actual_eos])
    eos_positions = [i for i, token in enumerate(ids) if token in actual_eos]
    eos_emitted = any(token in actual_eos for token in generated)
    longest = current = 0
    previous = None
    for token in generated:
        current = current + 1 if token == previous else 1
        previous, longest = token, max(longest, current)
    repeated = {}
    for n in (2, 3, 4):
        counts = {}
        for i in range(len(generated) - n + 1):
            gram = tuple(generated[i:i + n])
            counts[gram] = counts.get(gram, 0) + 1
        repeated[str(n)] = sum(count - 1 for count in counts.values() if count > 1)
    parsed = parse_spans(ids, tokenizer, len(expected_spans), actual_eos, decoder_start_token_id)
    by_number = {span["sentinel"]: span["text"] for span in parsed["spans"]}
    span_matches = [parsed["valid"] and normalize(by_number.get(i, "")) == normalize(expected)
                    for i, expected in enumerate(expected_spans)]
    return {"raw_token_ids": ids, "full_special_decode": tokenizer.decode(ids, skip_special_tokens=False),
            "original_skip_special_decode": tokenizer.decode(ids, skip_special_tokens=True),
            "parsed": parsed, "span_exact_match": span_matches, "all_spans_correct": all(span_matches),
            "expected_spans_correct": all(span_matches),
            "complete_output_correct": all(span_matches) and parsed["complete_generation_well_formed"],
            "metric": SCHEMA, "normalizer": NORMALIZER,
            "eos_token_id": eos, "generation_eos_token_ids": actual_eos,
            "eos_positions": eos_positions, "eos_emitted": eos_emitted,
            "generated_token_count": len(generated), "max_new_tokens_reached": len(generated) >= max_new_tokens,
            "termination": ("eos" if parsed["terminal_grammar_valid"] else "invalid_eos_order") if eos_emitted else ("max_new_tokens" if len(generated) >= max_new_tokens else "other"),
            "longest_repeated_token_run": longest, "repeated_ngrams": repeated,
            "raw_trace_sha256": canonical_hash(ids)}


def numeric_summary(values, torch):
    finite = torch.isfinite(values)
    clean = values[finite].float()
    return {"shape": list(values.shape), "dtype": str(values.dtype),
            "summary_accumulation_dtype": "float32", "values": values.numel(),
            "finite_values": int(finite.sum()), "nan_values": int(torch.isnan(values).sum()),
            "inf_values": int(torch.isinf(values).sum()), "all_finite": bool(finite.all()),
            "finite_fraction": float(finite.sum()) / max(1, values.numel()),
            "finite_min": float(clean.min()) if clean.numel() else None,
            "finite_max": float(clean.max()) if clean.numel() else None,
            "finite_mean": float(clean.mean()) if clean.numel() else None,
            "finite_l2": float(torch.linalg.vector_norm(clean)) if clean.numel() else None}


def direct_load(root, info, dtype, memory_budget_mib, preserve_router_fp32):
    """Independent native HF load; no compiler load_model call or AutoModel dispatch."""
    plan = preflight(info, dtype, memory_budget_mib)
    torch, cls, tokenizer_cls = runtime()
    if read_json(Path(root) / "tokenizer_config.json").get("auto_map"):
        fail("UNSUPPORTED_CONFIG", "Tokenizer remote code is forbidden")
    tokenizer = tokenizer_cls.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
    loaded, loading = cls.from_pretrained(str(root), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, torch_dtype=getattr(torch, dtype), low_cpu_mem_usage=True, output_loading_info=True)
    if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        fail("INVALID_WEIGHTS", "Native checkpoint key mismatch: %s" % loading)
    plan["router_precision"] = (restore_router_fp32(loaded, root, torch) if preserve_router_fp32 else
                                {"mode": "native_global_cast_then_router_runtime_upcast"})
    loaded.eval()
    return loaded, tokenizer, torch, plan


def precision_binding(root, dtype, preserve, reference=None, info=None):
    """Bind exact precision and inventories without granting admission or overrides."""
    info = info if info is not None else inspect_model(root)
    for filename in ("compiler_manifest.json", "roundtrip_manifest.json"):
        path = Path(root) / filename
        if path.exists():
            manifest = read_json(path)
            policy = manifest.get("experiment_config", {})
            if bool(policy.get("preserve_router_fp32")) != preserve:
                fail("ROUTER_PRECISION_POLICY_MISMATCH", "Explicit router precision arm differs from artifact manifest")
            if filename == "compiler_manifest.json":
                validate_precision_lineage(info, reference, manifest, dtype)
            else:
                if manifest.get("dtype") != dtype:
                    fail("DTYPE_MISMATCH", "Roundtrip artifact must be audited in its recorded dtype")
                actual = {k: v for k, v in info["files"].items() if k not in
                          ("roundtrip_manifest.json", "roundtrip_diagnostics.json")}
                if actual != manifest.get("candidate_files"):
                    fail("LINEAGE_MISMATCH", "Roundtrip artifact differs from its bound inventory")
                if reference is not None and (manifest.get("source_files") != reference["files"] or
                        manifest.get("source_artifact_sha256") != reference["artifact_sha256"]):
                    fail("REFERENCE_MISMATCH", "Reference must match exact original roundtrip source")
    artifact = any((Path(root) / name).exists() for name in ("compiler_manifest.json", "roundtrip_manifest.json"))
    return {"recorded_dtype_verified": artifact, "requested_dtype": dtype, "reference_binding":
            "exact_original_inventory_and_sha256" if reference is not None and artifact
            else "independent_checkpoint_comparison_not_pruning_only"}


def validate_tokenized_case(row, tokenizer, full_input_ids, input_ids, target_ids):
    mapping = sentinel_ids(tokenizer)
    expected_order = list(range(len(row["expected_spans"])))
    if any([mapping[t] for t in ids if t in mapping] != expected_order for ids in (full_input_ids, input_ids)):
        fail("INVALID_DIAGNOSTIC_SUITE", "Prompt tokenization/truncation must retain every required sentinel in order")
    parsed = parse_spans(target_ids, tokenizer, len(expected_order), require_decoder_start=False)
    expected = [normalize(tokenizer.decode(tokenizer(s, add_special_tokens=False)["input_ids"],
                                          skip_special_tokens=False)) for s in row["expected_spans"]]
    if (not parsed["complete_generation_well_formed"] or
            [normalize(s["text"]) for s in parsed["spans"]] != expected):
        fail("INVALID_DIAGNOSTIC_SUITE", "Tokenized teacher target must be complete and equal canonical expected_spans")


def run_cases(loaded, tokenizer, torch, config, prefix):
    import numpy as np
    cases = []
    gen_seconds = 0.0
    prepared = []
    # Validate EVERY teacher objective and prompt before the first inference.
    for row in config["suite"]["cases"]:
        encoded = tokenizer(row["prompt"], return_tensors="pt", truncation=True, max_length=config["max_length"])
        full_input = tokenizer(row["prompt"], return_tensors="pt")["input_ids"]
        labels = tokenizer(row["target"], return_tensors="pt", truncation=False)["input_ids"]
        if labels.numel() > config["teacher_tokens"]:
            fail("DIAGNOSTIC_TARGET_TOO_LONG", "Case %s target has %d tokens, ceiling %d; shorten target or explicitly increase teacher-tokens" % (row["id"], labels.numel(), config["teacher_tokens"]))
        # Include EOS explicitly even for local mechanics tokenizers without a template.
        if tokenizer.eos_token_id is not None and int(labels[0, -1]) != tokenizer.eos_token_id:
            labels = torch.cat((labels, torch.tensor([[tokenizer.eos_token_id]])), dim=1)
        if labels.numel() > config["teacher_tokens"]:
            fail("DIAGNOSTIC_TARGET_TOO_LONG", "Target including EOS exceeds teacher-token ceiling")
        validate_tokenized_case(row, tokenizer, full_input[0].tolist(), encoded["input_ids"][0].tolist(), labels[0].tolist())
        prepared.append((row, encoded, full_input, labels))
    if loaded.config.vocab_size > 131072:
        fail("DIAGNOSTIC_VOCAB_LIMIT", "Short-logit audit supports vocabulary <=131072")
    for index, (row, encoded, full_input, labels) in enumerate(prepared):
        gen_numeric = {"calls": 0, "values": 0, "nonfinite_values": 0, "nan_values": 0, "dtype_value_counts": {}}
        def head_hook(module, inputs, output):
            dtype_name = str(output.dtype)
            gen_numeric["dtype_value_counts"][dtype_name] = gen_numeric["dtype_value_counts"].get(dtype_name, 0) + output.numel()
            gen_numeric["calls"] += 1
            gen_numeric["values"] += output.numel()
            gen_numeric["nonfinite_values"] += int((~torch.isfinite(output)).sum())
            gen_numeric["nan_values"] += int(torch.isnan(output).sum())
        with RoutingTelemetry(loaded, torch, reap=True) as routes:
            handle = loaded.lm_head.register_forward_hook(head_hook)
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    if config["loader"] == "native":
                        output = loaded.generate(input_ids=encoded["input_ids"], attention_mask=encoded.get("attention_mask"),
                            do_sample=False, num_beams=1, max_new_tokens=config["max_new_tokens"])
                        ids = output[0].tolist()
                        del output
                    else:
                        ids = generate(loaded, tokenizer, torch, row["prompt"], config["max_length"],
                                       config["max_new_tokens"], return_token_ids=True)
            finally:
                handle.remove()
            elapsed = time.perf_counter() - started
            generation_routing = routes.result(require_finite=False)
        gen_seconds += elapsed
        trace = token_trace(ids, tokenizer, row["expected_spans"], config["max_new_tokens"], loaded.generation_config.eos_token_id,
                            loaded.generation_config.decoder_start_token_id)
        trace["numeric"] = {**gen_numeric, "all_finite": gen_numeric["nonfinite_values"] == 0,
                            "finite_fraction": 1.0 - gen_numeric["nonfinite_values"] / max(1, gen_numeric["values"]),
                            "scope": "all_lm_head_outputs_during_generation"}
        trace["generation_wall_seconds"] = elapsed
        trace["seconds_per_generated_token"] = elapsed / max(1, trace["generated_token_count"])
        with RoutingTelemetry(loaded, torch, reap=True) as routes:
            with torch.inference_mode():
                output = loaded(input_ids=encoded["input_ids"], attention_mask=encoded.get("attention_mask"),
                                labels=labels, use_cache=False)
            logits = output.logits[0].detach()
            summary = numeric_summary(logits, torch)
            floats = logits.float()
            nll = torch.nn.functional.cross_entropy(floats, labels[0], reduction="none")
            top = floats.topk(min(2, floats.shape[-1]), dim=-1)
            path = Path(config["workdir"]) / (prefix + "-%d.npy" % index)
            np.save(str(path), floats.cpu().numpy(), allow_pickle=False)
            target_ids = labels[0].tolist()
            mapping = sentinel_ids(tokenizer)
            def slice_nll(predicate):
                selected = [float(value) for token, value in zip(target_ids, nll) if predicate(token)]
                return sum(selected) / len(selected) if selected else None
            teacher = {"target_token_ids": target_ids, "decoder_prefix_token_ids": loaded._shift_right(labels)[0].tolist(),
                "target_full_special_decode": tokenizer.decode(target_ids, skip_special_tokens=False),
                "numeric": summary, "mean_target_token_nll": float(nll.mean()),
                "nll_per_token": nll.tolist(), "nll_accumulation_dtype": "float32",
                "sentinel_nll": slice_nll(lambda token: token in mapping),
                "eos_nll": slice_nll(lambda token: token == tokenizer.eos_token_id),
                "content_nll": slice_nll(lambda token: token not in mapping and token != tokenizer.eos_token_id),
                "argmax_token_ids": top.indices[:, 0].tolist(),
                "top1_minus_top2_margins": (top.values[:, 0] - top.values[:, 1]).tolist(),
                "logits_sha256": digest(path), "comparison_storage_dtype": "float32",
                "_logits_file": str(path), "scope": "short_teacher_forced_logits_direct_cross_entropy_no_router_aux_loss"}
            teacher_routing = routes.result(require_finite=False)
            del logits, floats, output, nll, top
        references = row.get("references", [row["target"]])
        cases.append({"id": row["id"], "prompt": row["prompt"], "target": row["target"],
            "group": row.get("group"), "references": references, "expected_spans": row["expected_spans"],
            "input_token_ids": encoded["input_ids"][0].tolist(), "input_truncated": full_input.numel() > encoded["input_ids"].numel(),
            "generation": trace, "teacher_forced": teacher,
            "literal_exact_match_text_proxy": trace["original_skip_special_decode"] in references,
            "routing": {"generation_native_cache_config": generation_routing, "teacher_forced": teacher_routing}})
    return finite_json(cases), gen_seconds


def worker(config):
    started = time.perf_counter()
    root = config["model"]
    info = inspect_model(root)
    if info["files"] != config["expected_model_files"]:
        fail("INPUT_CHANGED", "Model inventory changed between coordinator preflight and worker load")
    precision_binding(root, config["dtype"], config["preserve_router_fp32"], info=info)
    preflight(info, config["dtype"], config["memory_budget_mib"])
    torch, _, _ = runtime()
    torch.manual_seed(0)
    # Tiny dtype kernel smoke test, not a performance claim or memory-guard bypass.
    probe = torch.ones((2, 2), dtype=getattr(torch, config["dtype"]))
    if not bool(torch.isfinite(probe @ probe).all()):
        fail("DTYPE_KERNEL_FAILURE", "CPU dtype matmul produced nonfinite values")
    del probe
    loader = direct_load if config["loader"] == "native" else load_model
    loaded, tokenizer, torch, plan = loader(root, info, config["dtype"], config["memory_budget_mib"], config["preserve_router_fp32"])
    before_structure = loaded_structure(loaded, config["dtype"])
    # Do not silently override source generation configuration. Both paths use
    # the same native greedy kwargs; all inherited defaults are recorded.
    graph = {"modules": {name: type(module).__name__ for name, module in loaded.named_modules()},
             "parameter_shapes": {name: list(p.shape) for name, p in loaded.named_parameters()},
             "parameter_dtypes_before_forward": {name: str(p.dtype) for name, p in loaded.named_parameters()},
             "router_dtype": loaded.config.router_dtype,
             "router_precision": plan["router_precision"]["mode"], "requested_dtype": config["dtype"]}
    semantic = {"class": type(tokenizer).__name__, "vocab_sha256": canonical_hash(tokenizer.get_vocab()),
                "sentinel_ids": sentinel_ids(tokenizer), "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id, "decoder_start_token_id": loaded.config.decoder_start_token_id,
                "generation_config": loaded.generation_config.to_dict()}
    cases, generation_seconds = run_cases(loaded, tokenizer, torch, config, "before")
    result = {"loader": config["loader"], "model_sha256": info["artifact_sha256"], "cases": cases,
              "preflight": plan, "structure_before_forward": before_structure,
              "loaded_structure": loaded_structure(loaded, config["dtype"]),
              "graph_manifest": graph, "graph_sha256": canonical_hash(graph),
              "tokenizer_and_generation": semantic, "runtime": runtime_versions()}
    if config.get("save_to"):
        n = loaded.config.num_experts
        identity_stats = {name: {"probability_sum": [1.0] * n} for name, _ in sparse_layers(loaded)}
        result["identity_mapping"] = apply_selection(loaded, torch, identity_stats, n)
        result["identity_cases"], seconds = run_cases(loaded, tokenizer, torch, config, "identity")
        generation_seconds += seconds
        result["identity_structure"] = loaded_structure(loaded, config["dtype"])
        loaded.save_pretrained(config["save_to"], safe_serialization=True, max_shard_size="256MB")
        tokenizer.save_pretrained(config["save_to"])
    del loaded, tokenizer
    gc.collect()
    if inventory(safe_path(root)) != info["files"]:
        fail("SOURCE_CHANGED", "Model changed during audit worker")
    result["resources"] = {**observed_resources(started, generation_seconds), "pid": os.getpid(),
        "rss_scope": "fresh_worker_process_lifetime_one_checkpoint_only",
        "wall_scope": "worker_inspection_load_probes_optional_save_release",
        "fresh_process": True, "torch_threads": torch.get_num_threads()}
    return finite_json(result)


def worker_limits(timeout, output_limit, address_space_mib):
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0.05 <= timeout <= 3600:
        fail("INVALID_ARGUMENT", "worker timeout must be finite in [0.05,3600] seconds")
    if type(output_limit) is not int or not 1024 <= output_limit <= 16 * 1024 * 1024:
        fail("INVALID_ARGUMENT", "worker combined output limit must be 1024..16777216 bytes")
    if type(address_space_mib) is not int or not 512 <= address_space_mib <= 16384:
        fail("INVALID_ARGUMENT", "worker address space must be 512..16384 MiB")
    return {"worker_timeout_seconds": timeout, "worker_output_limit_bytes": output_limit,
            "worker_address_space_mib": address_space_mib}


def bounded_process(command, env, timeout, output_limit):
    """POSIX host deadline + streaming combined pipe cap, never capture_output.

    Trusted native diagnostic workers only; process-group cleanup is not isolation.
    No log content is re-emitted into the public command's JSON transport.
    """
    started = time.monotonic()
    totals = {"stdout": 0, "stderr": 0}
    hashes = {name: hashlib.sha256() for name in totals}
    stderr_tail = b""
    proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True)
    try:
        with selectors.DefaultSelector() as selector:
            for name, pipe in (("stdout", proc.stdout), ("stderr", proc.stderr)):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, name)
            while selector.get_map() or proc.poll() is None:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    fail("DIAGNOSTIC_WORKER_TIMEOUT", "Fresh worker exceeded finite wall deadline")
                for key, _ in selector.select(min(remaining, .05)):
                    block = os.read(key.fileobj.fileno(), 8192)
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    name = key.data
                    totals[name] += len(block)
                    if sum(totals.values()) > output_limit:
                        fail("DIAGNOSTIC_WORKER_OUTPUT_LIMIT", "Fresh worker exceeded combined stdout/stderr byte ceiling")
                    hashes[name].update(block)
                    if name == "stderr":
                        stderr_tail = (stderr_tail + block)[-2048:]
            return proc.returncode, stderr_tail.decode("utf-8", errors="replace"), {
                name: {"bytes": totals[name], "sha256": hashes[name].hexdigest()} for name in totals}
    finally:
        # Also terminate surviving children holding no pipe, not merely the leader.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)
        proc.stdout.close()
        proc.stderr.close()


def bounded_write_json(path, value):
    # Encode incrementally to bound both serialized artifacts and parent allocation.
    encoder = json.JSONEncoder(sort_keys=True, ensure_ascii=False, allow_nan=False)
    size = 0
    path = Path(path)
    try:
        with path.open("xb") as handle:
            for piece in encoder.iterencode(value):
                encoded = piece.encode("utf-8")
                size += len(encoded)
                if size + 1 > MAX_REPORT_BYTES:
                    fail("DIAGNOSTIC_REPORT_LIMIT", "Diagnostic JSON exceeds 32 MiB artifact ceiling")
                handle.write(encoded)
            handle.write(b"\n")
    except Exception:
        # Only remove our incomplete exclusive file, never an existing destination.
        if size:
            path.unlink(missing_ok=True)
        raise


def invoke_worker(config, directory, label):
    work = Path(directory) / label
    work.mkdir()
    config = {**config, "workdir": str(work)}
    request, response = work / "request.json", work / "response.json"
    write_json(request, config)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
    package_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, "-m", "asea.compiler.diagnostics", str(request), str(response)]
    limits = worker_limits(config["worker_timeout_seconds"], config["worker_output_limit_bytes"], config["worker_address_space_mib"])
    # No shell, remote code, checkpoint downloads, or concurrent model loads.
    returncode, stderr_tail, logs = bounded_process(command, env, limits["worker_timeout_seconds"], limits["worker_output_limit_bytes"])
    if not response.exists():
        fail("DIAGNOSTIC_WORKER_FAILED", "Fresh %s worker exit %s; %s" % (label, returncode, stderr_tail))
    result = read_json(response, max_bytes=MAX_REPORT_BYTES)
    if returncode or not result.get("ok"):
        error = result.get("error", {})
        fail(error.get("type", "DIAGNOSTIC_WORKER_FAILED"), "%s: %s" % (label, str(error.get("message", stderr_tail))[:2048]))
    result = result["result"]
    if result.get("model_sha256") != canonical_hash(config["expected_model_files"]):
        fail("LINEAGE_MISMATCH", "Worker loaded a different inventory than coordinator preflight")
    result["execution"] = {"label": label, "fresh_interpreter": True, "returncode": returncode,
        **limits, "logs": logs, "response_bytes": response.stat().st_size, "response_sha256": digest(response),
        "memory_scope": "RLIMIT_AS per worker process, not RSS or aggregate; preflight remains a heuristic",
        "offline": True, "requested_thread_ceiling": 2, "command_template": "python -m asea.compiler.diagnostics <internal-request> <internal-response>"}
    return result


def compare_cases(left, right, atol, rtol):
    import numpy as np
    if [row["id"] for row in left] != [row["id"] for row in right]:
        fail("DIAGNOSTIC_CASE_MISMATCH", "Comparison cases differ")
    compared = []
    for a, b in zip(left, right):
        x = np.load(a["teacher_forced"]["_logits_file"], mmap_mode="r", allow_pickle=False)
        y = np.load(b["teacher_forced"]["_logits_file"], mmap_mode="r", allow_pickle=False)
        compatible = (x.shape == y.shape and a["input_token_ids"] == b["input_token_ids"] and
                      a["teacher_forced"]["target_token_ids"] == b["teacher_forced"]["target_token_ids"])
        numeric = {"compatible_tokens_and_shapes": compatible, "all_finite": False, "within_tolerance": False}
        if compatible:
            finite = np.isfinite(x) & np.isfinite(y)
            all_finite = bool(finite.all())
            delta = np.abs(np.asarray(x)[finite].astype(np.float64) - np.asarray(y)[finite].astype(np.float64))
            baseline = np.abs(np.asarray(x)[finite].astype(np.float64))
            numeric.update({"all_finite": all_finite, "compared_finite_values": int(finite.sum()),
                "max_abs_error": float(delta.max()) if delta.size else None,
                "rms_error": float(np.sqrt(np.mean(delta ** 2))) if delta.size else None,
                "max_relative_error_denominator_floor_1e_8": float((delta / np.maximum(baseline, 1e-8)).max()) if delta.size else None,
                "within_tolerance": all_finite and bool(np.all(delta <= atol + rtol * baseline))})
        ta, tb = a["teacher_forced"], b["teacher_forced"]
        nll_a, nll_b = ta["mean_target_token_nll"], tb["mean_target_token_nll"]
        compared.append({"id": a["id"], "generation_token_ids_equal": a["generation"]["raw_token_ids"] == b["generation"]["raw_token_ids"],
            "teacher_argmax_equal": compatible and ta["argmax_token_ids"] == tb["argmax_token_ids"],
            "teacher_next_token_agreement": (sum(i == j for i, j in zip(ta["argmax_token_ids"], tb["argmax_token_ids"])) / len(ta["argmax_token_ids"])) if compatible else None,
            "mean_target_nll_right_minus_left": nll_b - nll_a if nll_a is not None and nll_b is not None else None,
            "logits": numeric})
        del x, y
    return {"atol": atol, "rtol": rtol, "reference_direction": "left baseline, right comparator",
            "all_generation_ids_equal": all(c["generation_token_ids_equal"] for c in compared),
            "all_short_logits_within_tolerance": all(c["logits"]["within_tolerance"] for c in compared),
            "cases": compared, "quality_certification": False}


def clean_private(value):
    if isinstance(value, dict):
        return {k: clean_private(v) for k, v in value.items() if not k.startswith("_")}
    if isinstance(value, list):
        return [clean_private(v) for v in value]
    return value


def base_report(command, config, source, suite_hash):
    return {"schema": SCHEMA, "command": command, "status": "AUDIT_ONLY", "pruned": False,
            "admission": "UNADMITTED", "certifies_coding_skills": False, "quality_certification": False,
            "evidence_scope": SCOPE, "suite": config["suite"], "suite_sha256": suite_hash,
            "source_artifact_sha256": source["artifact_sha256"], "source_files": source["files"],
            "dtype": config["dtype"], "preserve_router_fp32": config["preserve_router_fp32"],
            "worker_limits": {key: config[key] for key in
                ("worker_timeout_seconds", "worker_output_limit_bytes", "worker_address_space_mib")},
            "report_byte_ceiling": MAX_REPORT_BYTES,
            "memory_enforcement_scope": "per-process RLIMIT_AS; no aggregate/cgroup/RSS enforcement; resident preflight remains heuristic",
            "parser_implementation_sha256": digest(Path(__file__)), "normalizer": NORMALIZER,
            "normalizer_sha256": canonical_hash(NORMALIZER), "core_implementation_sha256": digest(Path(__file__).with_name("core.py")),
            "generation_parameters": {"do_sample": False, "num_beams": 1, "batch_size": 1,
                "max_length": config["max_length"], "max_new_tokens": config["max_new_tokens"],
                "teacher_tokens_ceiling_including_eos": config["teacher_tokens"]},
            "limitations": ["Exploratory diagnostics, not heldout noninferiority or admission evidence",
                "No original T5X/conversion-equivalence proof; no coding capability claim",
                "REAP/usage summaries are tiny coverage diagnostics, not causal importance",
                "Fresh-process RSS is a whole worker peak, including runtime/probes; not tensor-only memory",
                "Generation numeric checks cover lm_head, routers and dispatched expert outputs, not every intermediate",
                "At most 32 teacher-forced positions; NLL excludes router auxiliary loss and is not generative quality"]}


def make_config(model, suite, dtype, max_length, max_new_tokens, teacher_tokens, memory_budget_mib, preserve_router_fp32):
    return {"model": str(safe_path(model)), "suite": suite, "dtype": dtype, "max_length": max_length,
            "max_new_tokens": max_new_tokens, "teacher_tokens": teacher_tokens,
            "memory_budget_mib": memory_budget_mib, "preserve_router_fp32": preserve_router_fp32}


def check_disk(parent, info, config, workers, save=False):
    vocab = info["config"].get("vocab_size", 32128)
    logits = len(config["suite"]["cases"]) * config["teacher_tokens"] * vocab * 4 * workers
    checkpoint = info["weights"]["stored_parameters"] * 2 if save else 0
    required = logits + checkpoint + 256 * 1024 ** 2
    free = shutil.disk_usage(parent).free
    if required > free:
        fail("DISK_PREFLIGHT", "Need estimated %d bytes for bounded logits/checkpoint plus margin; only %d free" % (required, free))
    return {"required_bytes_estimate": required, "free_bytes_before": free, "heuristic_not_guarantee": True}


def diagnose_model(model, suite, output, dtype, reference_model=None, preserve_router_fp32=False,
                   max_length=64, max_new_tokens=16, teacher_tokens=16, memory_budget_mib=None, atol=0.001, rtol=0.001,
                   worker_timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS,
                   worker_output_limit_bytes=DEFAULT_WORKER_OUTPUT_LIMIT_BYTES,
                   worker_address_space_mib=DEFAULT_WORKER_ADDRESS_SPACE_MIB):
    diagnostic_bounds(dtype, max_length, max_new_tokens, teacher_tokens, atol, rtol)
    limits = worker_limits(worker_timeout_seconds, worker_output_limit_bytes, worker_address_space_mib)
    dest = destination(output, (model, reference_model))
    suite_path = safe_path(suite)
    suite_hash = digest(suite_path)
    data = validate_suite(read_json(suite_path))
    config = make_config(model, data, dtype, max_length, max_new_tokens, teacher_tokens, memory_budget_mib, preserve_router_fp32)
    info = inspect_model(model)
    config.update(limits)
    config["expected_model_files"] = info["files"]
    precision_binding(model, dtype, preserve_router_fp32, info=info)
    preflight(info, dtype, memory_budget_mib)
    disk = check_disk(dest.parent, info, config, 3 if reference_model else 2)
    report = base_report("diagnose", config, info, suite_hash)
    ref_info = inspect_model(reference_model) if reference_model else None
    if ref_info is not None:
        preflight(ref_info, dtype, memory_budget_mib)
        precision_binding(reference_model, dtype, preserve_router_fp32, info=ref_info)
        report["comparison_provenance"] = precision_binding(model, dtype, preserve_router_fp32, reference=ref_info, info=info)
        report["reference_artifact_sha256"] = ref_info["artifact_sha256"]
    with tempfile.TemporaryDirectory(prefix=".pass2-audit-", dir=str(dest.parent)) as work:
        native = invoke_worker({**config, "loader": "native"}, work, "native")
        wrapper = invoke_worker({**config, "loader": "wrapper"}, work, "wrapper")
        report["runs"] = {"native": native, "wrapper": wrapper}
        report["comparisons"] = {"native_vs_wrapper": compare_cases(native["cases"], wrapper["cases"], atol, rtol)}
        if reference_model:
            reference = invoke_worker({**config, "model": str(safe_path(reference_model)), "loader": "wrapper",
                                       "expected_model_files": ref_info["files"]}, work, "reference")
            report["runs"]["reference"] = reference
            report["reference_files"] = ref_info["files"]
            report["comparisons"]["reference_vs_model"] = compare_cases(reference["cases"], wrapper["cases"], atol, rtol)
            if inventory(safe_path(reference_model)) != ref_info["files"]:
                fail("SOURCE_CHANGED", "Reference changed during diagnostic")
        report = clean_private(report)
    if digest(suite_path) != suite_hash or inventory(safe_path(model)) != info["files"]:
        fail("INPUT_CHANGED", "Suite or model changed during diagnostic")
    report["source_unchanged"] = True
    report["disk_preflight"] = disk
    report["evidence_sha256"] = canonical_hash(report)
    bounded_write_json(dest, report)
    return {**report, "output": str(dest)}


def roundtrip_model(model, output, dtype, suite=None, preserve_router_fp32=False,
                    max_length=64, max_new_tokens=16, teacher_tokens=16, memory_budget_mib=None, atol=0.001, rtol=0.001,
                   worker_timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS,
                   worker_output_limit_bytes=DEFAULT_WORKER_OUTPUT_LIMIT_BYTES,
                   worker_address_space_mib=DEFAULT_WORKER_ADDRESS_SPACE_MIB):
    diagnostic_bounds(dtype, max_length, max_new_tokens, teacher_tokens, atol, rtol)
    limits = worker_limits(worker_timeout_seconds, worker_output_limit_bytes, worker_address_space_mib)
    dest = destination(output, (model,))
    data = validate_suite(read_json(suite)) if suite else DEFAULT_SUITE
    suite_hash = digest(safe_path(suite)) if suite else canonical_hash(data)
    config = make_config(model, data, dtype, max_length, max_new_tokens, teacher_tokens, memory_budget_mib, preserve_router_fp32)
    info = inspect_model(model)
    config.update(limits)
    config["expected_model_files"] = info["files"]
    precision_binding(model, dtype, preserve_router_fp32, info=info)
    preflight(info, dtype, memory_budget_mib)
    disk = check_disk(dest.parent, info, config, 4, save=True)
    report = base_report("roundtrip", config, info, suite_hash)
    with tempfile.TemporaryDirectory(prefix=".pass2-roundtrip-", dir=str(dest.parent)) as work:
        staging = Path(work) / "checkpoint"
        staging.mkdir()
        native = invoke_worker({**config, "loader": "native"}, work, "native")
        saved = invoke_worker({**config, "loader": "wrapper", "save_to": str(staging)}, work, "save")
        reload_run = invoke_worker({**config, "model": str(staging), "loader": "wrapper",
                                    "expected_model_files": inventory(staging)}, work, "fresh-reload")
        report["runs"] = {"native": native, "source_and_identity_save": saved, "fresh_reload": reload_run}
        report["comparisons"] = {
            "native_vs_wrapper": compare_cases(native["cases"], saved["cases"], atol, rtol),
            "source_vs_identity_in_memory": compare_cases(saved["cases"], saved["identity_cases"], atol, rtol),
            "saved_in_memory_vs_fresh_reload": compare_cases(saved["identity_cases"], reload_run["cases"], atol, rtol),
            "source_vs_fresh_reload": compare_cases(saved["cases"], reload_run["cases"], atol, rtol)}
        report["identity_mapping"] = saved["identity_mapping"]
        counts = [run["loaded_structure"]["parameters"] for run in (native, saved, reload_run)]
        if len(set(counts)) != 1 or saved["identity_structure"]["parameters"] != counts[0]:
            fail("IDENTITY_STRUCTURE_MISMATCH", "Identity roundtrip changed unique parameter count")
        report["source_parameters"] = counts[0]
        report["roundtrip_parameters"] = counts[-1]
        report["removed_parameters"] = 0
        report["num_experts"] = info["num_experts"]
        report["disk_preflight"] = disk
        if inventory(safe_path(model)) != info["files"] or (suite and digest(safe_path(suite)) != suite_hash):
            fail("INPUT_CHANGED", "Source/suite changed during roundtrip")
        report["source_unchanged"] = True
        report = clean_private(report)
        report["evidence_sha256"] = canonical_hash(report)
        manifest = {"schema": "switch-identity-roundtrip-v2", "status": "AUDIT_ONLY", "pruned": False,
            "admission": "UNADMITTED", "source_files": info["files"], "source_artifact_sha256": info["artifact_sha256"],
            "candidate_files": inventory(staging), "dtype": dtype, "num_experts": info["num_experts"],
            "parameters": counts[0], "removed_parameters": 0, "identity_mapping": saved["identity_mapping"],
            "experiment_config": {"preserve_router_fp32": preserve_router_fp32, "audit_only": True},
            "graph_sha256": reload_run["graph_sha256"], "diagnostics_sha256": report["evidence_sha256"]}
        bounded_write_json(staging / "roundtrip_manifest.json", manifest)
        bounded_write_json(staging / "roundtrip_diagnostics.json", report)
        if dest.exists():
            fail("OUTPUT_EXISTS", "Output appeared during roundtrip")
        staging.rename(dest)
    return {**report, "output": str(dest), "manifest": str(dest / "roundtrip_manifest.json"),
            "report": str(dest / "roundtrip_diagnostics.json")}


def diagnostic_receipt(report):
    """Compact JSON stdout; detailed bounded report stays exclusively on disk."""
    path = Path(report.get("report", report["output"]))
    receipt = {k: report[k] for k in ("schema", "command", "status", "admission", "pruned",
        "quality_certification", "certifies_coding_skills", "source_artifact_sha256", "source_unchanged",
        "dtype", "preserve_router_fp32", "output", "evidence_sha256")}
    receipt.update({"stdout_contract": "bounded_receipt_v1", "publication": "PUBLISHED",
                    "report": str(path), "report_bytes": path.stat().st_size, "report_sha256": digest(path),
                    "report_byte_ceiling": MAX_REPORT_BYTES, "logit_arrays_in_stdout": False})
    receipt["comparisons"] = {name: {k: result[k] for k in
        ("all_generation_ids_equal", "all_short_logits_within_tolerance", "quality_certification")}
        for name, result in report["comparisons"].items()}
    for field in ("manifest", "source_parameters", "roundtrip_parameters", "removed_parameters", "num_experts"):
        if field in report:
            receipt[field] = report[field]
    if "manifest" in receipt:
        receipt["manifest_sha256"] = digest(receipt["manifest"])
    return receipt


def worker_main():
    """Private transport, not a new certification or generic execution API."""
    from contextlib import redirect_stdout
    if len(sys.argv) != 3:
        print("Internal worker: use silt-compile diagnose/roundtrip", file=sys.stderr)
        return 2
    response = safe_path(sys.argv[2], False)
    try:
        config = read_json(sys.argv[1])
        limits = worker_limits(config["worker_timeout_seconds"], config["worker_output_limit_bytes"], config["worker_address_space_mib"])
        requested = limits["worker_address_space_mib"] * 1024 ** 2
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        effective = min([requested] + [v for v in (soft, hard) if v != resource.RLIM_INFINITY])
        resource.setrlimit(resource.RLIMIT_AS, (effective, effective))
        if resource.getrlimit(resource.RLIMIT_AS) != (effective, effective):
            fail("DIAGNOSTIC_RESOURCE_SETUP_FAILED", "Could not verify worker RLIMIT_AS")
        with redirect_stdout(sys.stderr):
            result = worker(config)
        result["address_space_limit_bytes"] = effective
        bounded_write_json(response, {"ok": True, "result": result})
        return 0
    except Exception as exc:
        bounded_write_json(response, {"ok": False, "error": {"type": exc.code if isinstance(exc, CompilerError) else "DIAGNOSTIC_RUNTIME_ERROR",
                                                             "message": str(exc)[:2048], "exception": type(exc).__name__}})
        return 2


if __name__ == "__main__":
    sys.exit(worker_main())
