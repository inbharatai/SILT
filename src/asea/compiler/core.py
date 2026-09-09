"""Compiler v1: one deliberately narrow, auditable model-family adapter."""
import gc
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from importlib.metadata import version
from pathlib import Path
import struct
import tempfile
from datetime import datetime, timezone

SCHEMA = "asea.compiler.v1"
ARCH = "SwitchTransformersForConditionalGeneration"
SCOPE = "seq2seq_text_proxy_not_coding_capability"


class CompilerError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def fail(code, message):
    raise CompilerError(code, message)


def safe_path(value, must_exist=True):
    path = Path(os.path.abspath(os.path.expanduser(str(value))))
    for part in (path,) + tuple(path.parents):
        if part.is_symlink():
            fail("UNSAFE_PATH", "Symlinks are forbidden: %s" % part)
    if must_exist and not path.exists():
        fail("NOT_FOUND", "Path does not exist: %s" % path)
    return path


def read_json(path, max_bytes=16 * 1024 * 1024):
    path = safe_path(path)
    if not path.is_file() or path.stat().st_size > max_bytes:
        fail("INVALID_INPUT", "Expected bounded JSON file: %s" % path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        fail("INVALID_JSON", str(exc))


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def inventory(root):
    root = safe_path(root)
    if not root.is_dir():
        fail("INVALID_MODEL", "Model must be a local directory")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            fail("UNSAFE_PATH", "Model tree contains symlink: %s" % path)
        if path.is_file():
            result[path.relative_to(root).as_posix()] = {"sha256": digest(path), "bytes": path.stat().st_size}
        elif not path.is_dir():
            fail("UNSAFE_PATH", "Special files are forbidden")
    return result


def validate_config(config):
    if not isinstance(config, dict) or config.get("model_type") != "switch_transformers" or config.get("architectures") != [ARCH]:
        fail("UNSUPPORTED_ARCHITECTURE", "Only native SwitchTransformersForConditionalGeneration is supported")
    if config.get("auto_map") or config.get("quantization_config"):
        fail("UNSUPPORTED_CONFIG", "Remote code and quantized checkpoints are unsupported")
    if type(config.get("num_experts")) is not int or config["num_experts"] < 1:
        fail("INVALID_MODEL", "Invalid num_experts")


def weight_metadata(root):
    """Read only safetensors headers; do not import torch or deserialize pickle."""
    files = sorted(root.glob("*.safetensors"))
    if not files:
        fail("UNSAFE_WEIGHTS", "No local safetensors weights found; pickle is never loaded")
    tensors = {}
    dtype_width = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}
    total = 0
    for path in files:
        safe_path(path)
        size = path.stat().st_size
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                fail("INVALID_WEIGHTS", "Truncated safetensors header")
            length = struct.unpack("<Q", prefix)[0]
            if length > 64 * 1024 * 1024 or length + 8 > size:
                fail("INVALID_WEIGHTS", "Invalid safetensors header length")
            try:
                header = json.loads(handle.read(length))
            except (ValueError, UnicodeError):
                fail("INVALID_WEIGHTS", "Invalid safetensors JSON header")
        if not isinstance(header, dict):
            fail("INVALID_WEIGHTS", "Invalid safetensors metadata")
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            if name in tensors or not isinstance(spec, dict) or spec.get("dtype") not in dtype_width:
                fail("INVALID_WEIGHTS", "Duplicate tensor or unsupported non-floating dtype")
            shape = spec.get("shape")
            offsets = spec.get("data_offsets")
            if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
                fail("INVALID_WEIGHTS", "Invalid tensor shape")
            count = math.prod(shape)
            if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int for n in offsets) or not (0 <= offsets[0] <= offsets[1] <= size - 8 - length) or offsets[1] - offsets[0] != count * dtype_width[spec["dtype"]]:
                fail("INVALID_WEIGHTS", "Invalid tensor offsets")
            tensors[name] = {"shape": shape, "dtype": spec["dtype"], "file": path.name}
            total += count
    index = root / "model.safetensors.index.json"
    if index.exists():
        data = read_json(index)
        mapping = data.get("weight_map", {})
        if not mapping or set(mapping) != set(tensors) or any(tensors[k]["file"] != v for k, v in mapping.items()):
            fail("INVALID_WEIGHTS", "Shard index does not match local tensor files")
    elif len(files) != 1 or files[0].name != "model.safetensors":
        fail("INVALID_WEIGHTS", "Expected model.safetensors or indexed shards")
    return {"stored_parameters": total, "weight_bytes": sum(p.stat().st_size for p in files), "tensor_count": len(tensors), "weight_files": [p.name for p in files]}


def inspect_model(model):
    root = safe_path(model)
    config = read_json(root / "config.json")
    validate_config(config)
    files = inventory(root)
    metadata = weight_metadata(root)
    lineage = root / "compiler_manifest.json"
    if lineage.exists():
        manifest = read_json(lineage)
        expected = manifest.get("candidate", {}).get("files")
        actual = {k: v for k, v in files.items() if k != "compiler_manifest.json"}
        if expected != actual:
            fail("LINEAGE_MISMATCH", "Candidate content differs from its lineage manifest")
    return {"schema": SCHEMA, "command": "inspect", "model": str(root), "architecture": ARCH,
            "num_experts": config["num_experts"], "config": config, "files": files,
            "artifact_sha256": canonical_hash(files), "weights": metadata, "evidence_scope": SCOPE,
            "admission": "UNADMITTED", "repair_training": {"supported": False, "reason": "No repair training or Phase 6 dense reconstruction implemented"}}


def runtime():
    try:
        import torch
        import transformers
        import accelerate  # noqa: F401 -- HF low_cpu_mem_usage loader dependency
        from transformers import SwitchTransformersForConditionalGeneration, AutoTokenizer
    except ImportError as exc:
        fail("MISSING_DEPENDENCY", "Install torch 2.6 CPU, transformers==4.51.3, accelerate, safetensors and tokenizer dependencies: %s" % exc)
    if transformers.__version__ != "4.51.3":
        fail("UNSUPPORTED_RUNTIME", "Compiler v1 requires transformers==4.51.3")
    if tuple(int(n) for n in torch.__version__.split("+")[0].split(".")[:2]) < (2, 6):
        fail("UNSUPPORTED_RUNTIME", "Compiler v1 requires torch>=2.6")
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    return torch, SwitchTransformersForConditionalGeneration, AutoTokenizer


def preflight(info, dtype="float32", memory_budget_mib=None):
    if dtype not in ("float32", "float16", "bfloat16"):
        fail("INVALID_DTYPE", "Use float32, float16 or bfloat16 on CPU")
    # Conservative two target-weight copies plus 512 MiB for bounded activations/runtime.
    estimate = info["weights"]["stored_parameters"] * (4 if dtype == "float32" else 2) * 2 + 512 * 1024 ** 2
    budget = 3584 * 1024 ** 2
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                budget = min(budget, int(line.split()[1]) * 1024)
        cap = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        current = int(Path("/sys/fs/cgroup/memory.current").read_text())
        if cap != "max":
            # Hashing large checkpoints warms reclaimable file cache. Do not count
            # inactive file pages as permanently resident model allocations.
            reclaimable = 0
            try:
                stat = dict(line.split() for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines())
                reclaimable = min(current, int(stat.get("inactive_file", "0")))
            except (OSError, ValueError):
                pass
            budget = min(budget, max(0, int(cap) - current + reclaimable))
    except (OSError, ValueError):
        pass
    if memory_budget_mib is not None:
        if memory_budget_mib <= 0:
            fail("INVALID_ARGUMENT", "Memory budget must be positive")
        budget = min(budget, memory_budget_mib * 1024 ** 2)
    if estimate > budget:
        fail("MEMORY_PREFLIGHT", "Estimated %d MiB exceeds %d MiB available budget; try float16 or a larger host" % (estimate // 1024 ** 2, budget // 1024 ** 2))
    return {"estimated_peak_bytes": estimate, "budget_bytes": budget, "dtype": dtype, "heuristic_not_guarantee": True}


def restore_router_fp32(model, root, torch):
    """Restore only original F32 classifier tensors, never the whole state dict.

    Called AFTER HF's global dtype load. Upcasting rounded half weights is not
    restoration. Non-F32 source routers retain the declared native load behavior.
    """
    from safetensors import safe_open
    restored = {}
    parameters = dict(model.named_parameters())
    wanted = {name for name in parameters if ".router.classifier." in name}
    if any(layer.router.dtype != torch.float32 for _, layer in sparse_layers(model)):
        fail("ROUTER_DTYPE_MISMATCH", "Precision-preserving arm requires native router_dtype=float32")
    for path in sorted(Path(root).glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in wanted.intersection(handle.keys()):
                view = handle.get_slice(name)
                if view.get_dtype() != "F32":
                    continue
                if math.prod(view.get_shape()) > 4 * 1024 * 1024:
                    fail("ROUTER_SIZE_LIMIT", "Router restoration exceeds bounded tensor limit")
                value = handle.get_tensor(name)
                parameter = parameters[name]
                if list(value.shape) != list(parameter.shape):
                    fail("INVALID_WEIGHTS", "Router tensor shape mismatch: %s" % name)
                with torch.no_grad():
                    parameter.data = value.clone()
                restored[name] = {"source_dtype": "F32", "loaded_dtype": "float32",
                                  "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()}
    if not restored:
        fail("NO_F32_ROUTERS", "No original F32 router classifier tensors found")
    return {"mode": "source_f32_safetensors_after_global_load", "tensors": restored}


def load_model(root, info, dtype, memory_budget_mib, preserve_router_fp32=False):
    for filename in ("compiler_manifest.json", "roundtrip_manifest.json"):
        path = Path(root) / filename
        if path.exists():
            policy = read_json(path).get("experiment_config")
            if policy is not None and bool(policy.get("preserve_router_fp32")) != preserve_router_fp32:
                fail("ROUTER_PRECISION_POLICY_MISMATCH", "Requested loader precision arm differs from artifact manifest; use diagnose with explicit matching flag")
    plan = preflight(info, dtype, memory_budget_mib)
    torch, cls, tokenizer_cls = runtime()
    tokenizer_config = read_json(Path(root) / "tokenizer_config.json")
    if tokenizer_config.get("auto_map"):
        fail("UNSUPPORTED_CONFIG", "Tokenizer remote-code mappings are forbidden")
    tokenizer = tokenizer_cls.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
    model, loading = cls.from_pretrained(str(root), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, torch_dtype=getattr(torch, dtype), low_cpu_mem_usage=True,
        output_loading_info=True)
    if loading.get("missing_keys") or loading.get("unexpected_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        fail("INVALID_WEIGHTS", "Checkpoint keys do not match native architecture: %s" % loading)
    plan["router_precision"] = (restore_router_fp32(model, root, torch) if preserve_router_fp32 else
                                {"mode": "native_global_cast_then_router_runtime_upcast"})
    model.eval()
    model.config._name_or_path = ""
    return model, tokenizer, torch, plan


def load_samples(path, kind):
    data = read_json(path)
    key = "samples" if kind == "calibration" else "cases"
    rows = data.get(key) if isinstance(data, dict) else None
    maximum = 256 if kind == "calibration" else 4096
    if not isinstance(rows, list) or not 1 <= len(rows) <= maximum:
        fail("INVALID_DATASET", "%s must contain 1..%d %s" % (kind, maximum, key))
    ids = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not row["prompt"].strip() or len(row["prompt"]) > 32768:
            fail("INVALID_DATASET", "Each row requires a nonempty prompt <=32768 characters")
        if kind == "calibration":
            if not isinstance(row.get("target"), str) or not row["target"].strip() or len(row["target"]) > 32768:
                fail("INVALID_DATASET", "Calibration requires nonempty teacher-forced target <=32768 characters")
        else:
            if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
                fail("INVALID_DATASET", "Evaluation requires unique string case ids")
            ids.add(row["id"])
            if "group" in row and row["group"] not in ("target", "control"):
                fail("INVALID_DATASET", "Optional group must be target or control")
            refs = row.get("references")
            if not isinstance(refs, list) or not refs or any(not isinstance(r, str) or len(r) > 32768 for r in refs):
                fail("INVALID_DATASET", "Each evaluation case requires string references")
    if kind != "calibration" and data.get("split") != "heldout":
        fail("INVALID_DATASET", "Evaluation suite must declare split=heldout")
    return data, rows


def sample_hash(row):
    return canonical_hash({"prompt": row["prompt"]})


def bounds(max_length, max_new_tokens=1):
    if not 1 <= max_length <= 256 or not 1 <= max_new_tokens <= 128:
        fail("INVALID_ARGUMENT", "max_length must be 1..256; max_new_tokens must be 1..128")


def sparse_layers(model):
    from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersSparseMLP
    layers = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, SwitchTransformersSparseMLP)]
    if not layers:
        fail("UNSUPPORTED_CONFIG", "No Switch sparse MLP layers found")
    n = model.config.num_experts
    for name, layer in layers:
        if list(layer.experts) != ["expert_%d" % i for i in range(n)] or layer.router.classifier.out_features != n:
            fail("INVALID_MODEL", "Unexpected expert/router layout at %s" % name)
    return layers


class RoutingTelemetry:
    """Native post-cast top1/dispatch; optional conditional output-norm evidence.

    Analytic FP32 mass is kept separate from actual post-cast selection. Expert
    hooks see the real post-capacity output before exactly one gate multiply.
    Only the current invocation mask/gates and scalar accumulators are retained.
    """
    def __init__(self, model, torch, reap=False):
        self.torch, self.reap = torch, reap
        self.layers = sparse_layers(model)
        self.n = model.config.num_experts
        self.handles, self.pending = [], {}
        self.stats = {name: {
            "probability_sum": torch.zeros(self.n, dtype=torch.float64),
            "analytic_fp32_top1_count": torch.zeros(self.n, dtype=torch.int64),
            "top1_count": torch.zeros(self.n, dtype=torch.int64),
            "dispatched_count": torch.zeros(self.n, dtype=torch.int64),
            "reap_weighted_l2_sum": torch.zeros(self.n, dtype=torch.float64),
            "expert_l2_sum": torch.zeros(self.n, dtype=torch.float64),
            "expert_output_count": torch.zeros(self.n, dtype=torch.int64),
            "tokens": 0, "postcast_vs_analytic_disagreements": 0,
            "nonfinite_router_values": 0, "nonfinite_expert_values": 0,
        } for name, _ in self.layers}

    def __enter__(self):
        torch, n = self.torch, self.n
        def router_hook(name):
            def collect(module, inputs, output):
                mask, gate, logits = output
                entry = self.stats[name]
                # This mirrors native softmax(dtype=router.dtype).to(input_dtype),
                # not the independently recomputed FP32 analytic argmax.
                actual = torch.softmax(logits.detach(), -1, dtype=module.dtype).to(inputs[0].dtype)
                analytic = torch.softmax(logits.detach().float(), -1)
                flat = analytic.reshape(-1, n)
                pre = actual.reshape(-1, n).argmax(-1)
                entry["probability_sum"] += flat.sum(0).double().cpu()
                entry["analytic_fp32_top1_count"] += torch.bincount(flat.argmax(-1).cpu(), minlength=n)
                entry["top1_count"] += torch.bincount(pre.cpu(), minlength=n)
                entry["dispatched_count"] += mask.detach().reshape(-1, n).sum(0).long().cpu()
                entry["postcast_vs_analytic_disagreements"] += int((pre != flat.argmax(-1)).sum())
                entry["nonfinite_router_values"] += int((~torch.isfinite(logits)).sum()) + int((~torch.isfinite(gate)).sum())
                entry["tokens"] += flat.shape[0]
                if self.reap:
                    self.pending[name] = (mask.detach().bool(), gate.detach().squeeze(-1))
            return collect
        def expert_hook(name, index):
            def collect(module, inputs, output):
                mask, gates = self.pending[name]
                selected_gates = gates[mask[:, :, index]].float()
                values = output.detach().float()
                if values.shape[0] != selected_gates.numel():
                    fail("INVALID_ROUTING", "Expert/gate dispatch shape mismatch at %s" % name)
                entry = self.stats[name]
                entry["nonfinite_expert_values"] += int((~torch.isfinite(values)).sum())
                norms = torch.linalg.vector_norm(values, dim=-1)
                entry["reap_weighted_l2_sum"][index] += (norms * selected_gates).double().sum().cpu()
                entry["expert_l2_sum"][index] += norms.double().sum().cpu()
                entry["expert_output_count"][index] += values.shape[0]
            return collect
        for name, layer in self.layers:
            self.handles.append(layer.router.register_forward_hook(router_hook(name)))
            if self.reap:
                for i in range(n):
                    self.handles.append(layer.experts["expert_%d" % i].register_forward_hook(expert_hook(name, i)))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.pending.clear()

    def result(self, require_finite=True):
        result = {}
        for name, entry in self.stats.items():
            bad = entry["nonfinite_router_values"] + entry["nonfinite_expert_values"]
            if require_finite and (entry["tokens"] == 0 or bad or
                    not self.torch.isfinite(entry["probability_sum"]).all()):
                fail("INVALID_ROUTING", "No finite routing evidence at %s" % name)
            row = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in entry.items()}
            row["top1_semantics"] = "native_postcast_pre_capacity"
            row["dropped_positions"] = row["tokens"] - sum(row["dispatched_count"])
            row["drop_fraction"] = row["dropped_positions"] / row["tokens"] if row["tokens"] else None
            row["reap_dispatched"] = [total / count if count else None for total, count in
                zip(row["reap_weighted_l2_sum"], row["expert_output_count"])] if self.reap else None
            row["ean_dispatched"] = [total / count if count else None for total, count in
                zip(row["expert_l2_sum"], row["expert_output_count"])] if self.reap else None
            result[name] = row
        return result


def collect_routing(model, tokenizer, torch, rows, max_length, scorer="probability_mass"):
    with RoutingTelemetry(model, torch, reap=scorer == "reap_dispatched") as collector:
        with torch.inference_mode():
            for row in rows:
                encoded = tokenizer(row["prompt"], return_tensors="pt", truncation=True, max_length=max_length)
                labels = tokenizer(row["target"], return_tensors="pt", truncation=True, max_length=max_length)["input_ids"]
                model(input_ids=encoded["input_ids"], attention_mask=encoded.get("attention_mask"), labels=labels, use_cache=False)
    return collector.result()


def selection_scores(stats, scorer="probability_mass", minimum_expert_observations=1):
    if scorer not in ("probability_mass", "reap_dispatched"):
        fail("INVALID_SCORER", "Use probability_mass or reap_dispatched")
    if type(minimum_expert_observations) is not int or minimum_expert_observations < 1:
        fail("INVALID_COVERAGE_POLICY", "minimum_expert_observations must be >=1")
    scores = {}
    for name, row in stats.items():
        if scorer == "reap_dispatched":
            counts = row.get("expert_output_count", [])
            unknown = [i for i, count in enumerate(counts) if count < minimum_expert_observations]
            if not counts or unknown or counts != row.get("dispatched_count"):
                fail("INSUFFICIENT_EXPERT_COVERAGE", "REAP-Switch-dispatched-adaptation blocked at %s; under-observed expert indices %s; require >=%d actual outputs per expert (unknown is not zero importance)" % (name, unknown, minimum_expert_observations))
            values = row.get("reap_dispatched")
        else:
            values = row.get("probability_sum")
        if not values or any(v is None or not math.isfinite(v) for v in values):
            fail("INVALID_ROUTING", "Nonfinite/undefined selection scores at %s" % name)
        scores[name] = values
    return scores


def apply_selection(model, torch, stats, keep, scorer="probability_mass", minimum_expert_observations=1):
    selected = {}
    scores = selection_scores(stats, scorer, minimum_expert_observations)
    for name, layer in sparse_layers(model):
        # Probability mass is usage evidence, not causal expert importance.
        rank = sorted(range(model.config.num_experts), key=lambda i: (-scores[name][i], i))
        indices = sorted(rank[:keep])
        old = layer.router.classifier
        new = torch.nn.Linear(old.in_features, keep, bias=old.bias is not None, device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight.copy_(old.weight[indices])
            if old.bias is not None:
                new.bias.copy_(old.bias[indices])
        layer.experts = torch.nn.ModuleDict({"expert_%d" % j: layer.experts["expert_%d" % i] for j, i in enumerate(indices)})
        layer.router.classifier = new
        layer.router.num_experts = keep
        selected[name] = {"retained_source_indices": indices, "new_to_source": {str(j): i for j, i in enumerate(indices)}, "routing": stats[name]}
        tokens = stats[name].get("tokens", 0)
        if tokens:
            selected[name]["analytic_counterfactual"] = {
                "formula": "p_retained_i / sum(p_retained_j); fixed original input, not an observed post-prune gate",
                "mean_retained_fp32_probability_mass": sum(stats[name]["probability_sum"][i] for i in indices) / tokens,
                "removed_native_postcast_winner_fraction": sum(count for i, count in enumerate(stats[name]["top1_count"]) if i not in indices) / tokens,
                "mean_gate_amplification": None,
                "limitation": "Nonlinear mean amplification cannot be recovered from probability sums; do not invert the mean mass"}
    model.config.num_experts = keep
    # Stacks own copied configs in HF; update them too (top-level config is serialized).
    model.encoder.config.num_experts = keep
    model.decoder.config.num_experts = keep
    model.eval()
    return selected


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def prune_model(model, output, keep_experts, calibration, dtype="float32", max_length=128, memory_budget_mib=None,
                preserve_router_fp32=False, scorer="probability_mass", minimum_expert_observations=1):
    start = time.perf_counter()
    bounds(max_length)
    selection_scores({}, scorer, minimum_expert_observations)
    research_only = dtype == "bfloat16" or preserve_router_fp32 or scorer == "reap_dispatched"
    source = safe_path(model)
    target = safe_path(output, False)
    if target.exists():
        fail("OUTPUT_EXISTS", "Output path must not exist")
    if source == target or source in target.parents:
        fail("UNSAFE_OUTPUT", "Output must not be under source")
    if not target.parent.is_dir():
        fail("UNSAFE_OUTPUT", "Output parent must already exist")
    before = inspect_model(source)
    inherited_research = any(research_artifact(read_json(source / name)) for name in
        ("compiler_manifest.json", "roundtrip_manifest.json") if (source / name).exists())
    research_only = research_only or inherited_research
    n = before["num_experts"]
    if type(keep_experts) is not int or not 1 <= keep_experts < n:
        fail("INVALID_KEEP", "keep_experts must be >=1 and less than source num_experts")
    _, rows = load_samples(calibration, "calibration")
    calibration_hash = digest(safe_path(calibration))
    loaded, tokenizer, torch, plan = load_model(source, before, dtype, memory_budget_mib, preserve_router_fp32)
    source_parameters = sum(p.numel() for p in loaded.parameters())
    stats = collect_routing(loaded, tokenizer, torch, rows, max_length, scorer)
    selection = apply_selection(loaded, torch, stats, keep_experts, scorer, minimum_expert_observations)
    candidate_parameters = sum(p.numel() for p in loaded.parameters())
    if candidate_parameters >= source_parameters:
        fail("STRUCTURAL_FAILURE", "No physical parameter reduction")
    # Atomic final rename: never expose partial candidate as a completed artifact.
    import shutil
    staging = Path(tempfile.mkdtemp(prefix=".compiler-", dir=str(target.parent)))
    try:
        loaded.save_pretrained(str(staging), safe_serialization=True, max_shard_size="256MB")
        tokenizer.save_pretrained(str(staging))
        version = {"torch": torch.__version__}
        import transformers
        version["transformers"] = transformers.__version__
        del loaded, tokenizer
        gc.collect()
        after = inventory(source)
        if after != before["files"]:
            fail("SOURCE_CHANGED", "Source content changed during compilation")
        if digest(safe_path(calibration)) != calibration_hash:
            fail("CALIBRATION_CHANGED", "Calibration file changed during compilation")
        candidate_files = inventory(staging)
        manifest = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
            "admission": "UNADMITTED", "architecture": ARCH, "evidence_scope": SCOPE,
            "status": "AUDIT_ONLY" if research_only else "UNADMITTED",
            "source": {"path_for_audit_only": str(source), "artifact_sha256": before["artifact_sha256"], "files": before["files"], "parameters": source_parameters},
            "source_after_sha256": canonical_hash(after), "source_unchanged": True,
            "candidate": {"artifact_sha256": canonical_hash(candidate_files), "files": candidate_files, "parameters": candidate_parameters},
            "calibration": {"sha256": calibration_hash, "sample_prompt_sha256": [sample_hash(r) for r in rows], "samples": len(rows), "batch_size": 1, "max_length": max_length, "decoder_mode": "teacher_forced_targets", "padding": "none; EOS and decoder-start tokens included"},
            "selection_method": ("REAP-Switch-dispatched-adaptation" if scorer == "reap_dispatched" else "per_layer_sum_router_probability_routing_usage_NOT_causal_importance"),
            "experiment_config": {"scorer": scorer, "minimum_expert_observations": minimum_expert_observations,
                                  "preserve_router_fp32": preserve_router_fp32,
                                  "source_research_only": inherited_research, "audit_only": research_only},
            "original_num_experts": n, "num_experts": keep_experts, "layers": selection,
            "preflight": plan, "runtime": version, "repair_training": {"supported": False},
            "preservation": "Only sparse experts and corresponding router rows replaced; shared dense modules retained; requested load dtype conversion applies globally"}
        write_json(staging / "compiler_manifest.json", manifest)
        if target.exists():
            fail("OUTPUT_EXISTS", "Output appeared during compilation")
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"schema": SCHEMA, "command": "prune", "output": str(target), "manifest": str(target / "compiler_manifest.json"),
            "manifest_sha256": digest(target / "compiler_manifest.json"), "admission": "UNADMITTED",
            "status": "AUDIT_ONLY" if research_only else "UNADMITTED", "audit_only": research_only,
            "source_parameters": source_parameters, "candidate_parameters": candidate_parameters,
            "removed_parameters": source_parameters - candidate_parameters, "num_experts": keep_experts, "source_unchanged": True, "evidence_scope": SCOPE,
            "resources": {**observed_resources(start, 0), "pid": os.getpid(),
                          "wall_scope": "prune_including_hashes_calibration_save",
                          "fresh_process_required_for_independent_peak": True}}


def generate(loaded, tokenizer, torch, prompt, max_length, max_new_tokens, return_token_ids=False):
    with torch.inference_mode():
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        output = loaded.generate(input_ids=tokens["input_ids"], attention_mask=tokens.get("attention_mask"),
                                 do_sample=False, num_beams=1, max_new_tokens=max_new_tokens)
    return output[0].tolist() if return_token_ids else tokenizer.decode(output[0], skip_special_tokens=True)


def runtime_versions():
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{name: version(name) for name in ("torch", "transformers", "accelerate", "safetensors", "tokenizers")}}


def observed_resources(start, generation_seconds):
    # ru_maxrss is a process lifetime high-water mark, NOT a resettable per-model peak.
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result = {"wall_seconds": float(time.perf_counter() - start),
              "generation_wall_seconds": float(generation_seconds),
              "process_peak_rss_bytes": peak * (1.0 if sys.platform == "darwin" else 1024.0)}
    if any(not math.isfinite(v) or v < 0 for v in result.values()):
        fail("INVALID_RESOURCE_METRICS", "Resource measurements must be finite and nonnegative")
    return {**result, "rss_scope": "process_lifetime_high_water_sequential_not_isolated_model_peak",
            "wall_scope": "load_generate_release_excludes_inventory_hashing", "observed": True}


def loaded_structure(loaded, dtype):
    hist = {}
    mismatches = []
    count = actual_bytes = 0
    for name, parameter in loaded.named_parameters():
        n = parameter.numel()
        actual = str(parameter.dtype).replace("torch.", "")
        hist[actual] = hist.get(actual, 0) + n
        count += n
        actual_bytes += n * parameter.element_size()
        if actual != dtype and not (".router." in name and actual == "float32"):
            mismatches.append(name)
    return {"parameters": count, "actual_parameter_bytes": actual_bytes,
            "normalized_parameter_bytes": count * (2 if dtype in ("float16", "bfloat16") else 4),
            "parameter_dtype_counts": hist, "requested_dtype": dtype,
            "unexpected_dtype_parameters": mismatches}


def infer_model(model, prompt, max_new_tokens=32, dtype="float32", max_length=128, memory_budget_mib=None):
    bounds(max_length, max_new_tokens)
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 32768:
        fail("INVALID_ARGUMENT", "Prompt must be nonempty and <=32768 characters")
    info = inspect_model(model)
    start = time.perf_counter()
    loaded, tokenizer, torch, plan = load_model(model, info, dtype, memory_budget_mib)
    structure = loaded_structure(loaded, dtype)
    generation_start = time.perf_counter()
    text = generate(loaded, tokenizer, torch, prompt, max_length, max_new_tokens)
    generation_seconds = time.perf_counter() - generation_start
    del loaded, tokenizer
    gc.collect()
    resources = observed_resources(start, generation_seconds)
    if inventory(safe_path(model)) != info["files"]:
        fail("MODEL_CHANGED", "Model changed during inference")
    return {"schema": SCHEMA, "command": "infer", "model_sha256": info["artifact_sha256"], "prompt": prompt, "generation": text,
            "generation_sha256": canonical_hash(text), "max_length": max_length, "max_new_tokens": max_new_tokens, "preflight": plan, "evidence_scope": SCOPE,
            "resources": resources, "runtime": runtime_versions(), "loaded_structure": structure,
            "model_unchanged": True, "admission": "UNADMITTED"}


def score_generations(rows, texts):
    if not rows or len(rows) != len(texts) or any(not isinstance(t, str) for t in texts):
        fail("INVALID_EVIDENCE", "Require one generation per nonempty case")
    outputs = []
    for row, text in zip(rows, texts):
        outputs.append({"id": row["id"], "prompt": row["prompt"], "references": row["references"], "generation": text,
                        "exact_match": text in row["references"], "generation_sha256": canonical_hash(text),
                        **({"group": row["group"]} if "group" in row else {})})
    return {"metric": "literal_exact_match_text_proxy", "exact_match": sum(o["exact_match"] for o in outputs) / len(outputs), "cases": outputs}


def evaluate_model(model, suite, reference_model=None, dtype="float32", max_length=128, max_new_tokens=32, memory_budget_mib=None, evidence_output=None):
    bounds(max_length, max_new_tokens)
    _, rows = load_samples(suite, "suite")
    suite_hash = digest(safe_path(suite))
    info = inspect_model(model)
    manifest_path = safe_path(model) / "compiler_manifest.json"
    heldout = "user_declared_only_no_calibration_manifest"
    if manifest_path.exists():
        calibration = read_json(manifest_path).get("calibration", {})
        overlap = set(calibration.get("sample_prompt_sha256", [])) & {sample_hash(row) for row in rows}
        if overlap or calibration.get("sha256") == suite_hash:
            fail("CALIBRATION_LEAKAGE", "Evaluation reuses a calibration file or prompt")
        heldout = "user_declared_plus_exact_prompt_disjoint_from_calibration"
    reference_info = inspect_model(reference_model) if reference_model is not None else None
    def run(root, inspected):
        start = time.perf_counter()
        loaded, tokenizer, torch, plan = load_model(root, inspected, dtype, memory_budget_mib)
        structure = loaded_structure(loaded, dtype)
        generation_start = time.perf_counter()
        texts = [generate(loaded, tokenizer, torch, row["prompt"], max_length, max_new_tokens) for row in rows]
        generation_seconds = time.perf_counter() - generation_start
        del loaded, tokenizer
        gc.collect()
        resources = observed_resources(start, generation_seconds)
        if inventory(safe_path(root)) != inspected["files"]:
            fail("MODEL_CHANGED", "Model files changed during evaluation")
        return score_generations(rows, texts), plan, resources, structure
    result, plan, resources, structure = run(model, info)
    reference = None
    if reference_model is not None:
        reference_result, reference_plan, reference_resources, reference_structure = run(reference_model, reference_info)
        reference = {"model_sha256": reference_info["artifact_sha256"], "result": reference_result,
                     "model_files": reference_info["files"], "preflight": reference_plan,
                     "resources": reference_resources, "loaded_structure": reference_structure,
                     "candidate_minus_reference_exact_match": result["exact_match"] - reference_result["exact_match"],
                     "generation_agreement": sum(a["generation"] == b["generation"] for a, b in zip(result["cases"], reference_result["cases"])) / len(rows)}
    for root, inspected in ((model, info), (reference_model, reference_info)):
        if root is not None and inventory(safe_path(root)) != inspected["files"]:
            fail("SOURCE_CHANGED" if root == reference_model else "MODEL_CHANGED", "Content changed during paired evaluation")
    if digest(safe_path(suite)) != suite_hash:
        fail("SUITE_CHANGED", "Suite changed during evaluation")
    evidence = {"schema": SCHEMA, "command": "evaluate", "created_at": datetime.now(timezone.utc).isoformat(),
        "admission": "UNADMITTED", "certifies_coding_skills": False, "evidence_scope": SCOPE,
        "model_sha256": info["artifact_sha256"], "model_files": info["files"], "suite_sha256": suite_hash,
        "heldout_check": heldout, "result": result, "reference": reference,
        "resources": resources, "loaded_structure": structure, "runtime": runtime_versions(),
        "model_unchanged": True, "reference_unchanged": True if reference else None,
        "generation_parameters": {"do_sample": False, "num_beams": 1, "max_length": max_length, "max_new_tokens": max_new_tokens},
        "preflight": plan, "dtype": dtype, "limitations": ["No code execution or functional coding tests", "No admission, causal-importance or performance claim", "Exact prompt disjointness does not prove semantic heldout independence"]}
    evidence["evidence_sha256"] = canonical_hash(evidence)
    if evidence_output:
        dest = safe_path(evidence_output, False)
        if dest.exists():
            fail("OUTPUT_EXISTS", "Evidence output already exists")
        for root in (model, reference_model):
            if root and (safe_path(root) == dest or safe_path(root) in dest.parents):
                fail("UNSAFE_OUTPUT", "Evidence must be written outside model directories")
        with dest.open("x", encoding="utf-8") as handle:
            json.dump(evidence, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        evidence["evidence_output"] = str(dest)
    return evidence


# Fixed before observing results: eight one-sided exact bounds, Bonferroni FWER 5%.
# No bootstrap randomness, fitted variance, or zero-variance automatic pass.
ADMISSION_ALPHA = 0.05 / 8


def admission_policy(minimum_cases=20, max_loss=0.02, minimum_reference_score=0.5,
                     minimum_candidate_score=0.5, minimum_size_reduction=0.1):
    if type(minimum_cases) is not int or not 20 <= minimum_cases <= 4096:
        fail("INVALID_POLICY", "minimum_cases must be an integer in 20..4096")
    values = {"max_loss": (max_loss, 0.0, 0.05),
              "minimum_reference_score": (minimum_reference_score, 0.5, 1.0),
              "minimum_candidate_score": (minimum_candidate_score, 0.5, 1.0),
              "minimum_size_reduction": (minimum_size_reduction, 0.1, 1.0)}
    for name, (value, low, high) in values.items():
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            fail("INVALID_POLICY", "%s must be finite in [%s,%s]" % (name, low, high))
    return {"minimum_cases": minimum_cases, **{k: float(v[0]) for k, v in values.items()},
            "confidence": 0.95, "tail_alpha": ADMISSION_ALPHA,
            "method": "paired_gain_lower_minus_loss_upper_exact_binomial_bonferroni",
            "random_seed": None, "randomness": "none_exact_deterministic_bounds",
            "scope": SCOPE, "version": "compiler-admission-v1"}


def binomial_upper(k, n, alpha=ADMISSION_ALPHA):
    """Clopper-Pearson one-sided upper confidence bound, including k=0."""
    if n < 1 or not 0 <= k <= n:
        fail("INVALID_EVIDENCE", "Invalid binomial counts")
    if k == n:
        return 1.0
    if k == 0:
        return -math.expm1(math.log(alpha) / n)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in range(k + 1)]
    low, high = k / n, 1.0
    for _ in range(60):
        p = (low + high) / 2
        cdf = math.fsum(math.exp(c + i * math.log(p) + (n - i) * math.log1p(-p)) for i, c in enumerate(logs))
        if cdf > alpha:
            low = p
        else:
            high = p
    return high


def binomial_lower(k, n):
    return 1.0 - binomial_upper(n - k, n)


def admission_suite(rows, policy):
    if len(rows) < policy["minimum_cases"]:
        fail("INSUFFICIENT_CASES", "Certification requires at least minimum_cases (hard floor 20)")
    if any(r.get("group") not in ("target", "control") for r in rows):
        fail("MISSING_GROUPS", "Every certification case requires group=target or control")
    if {r["group"] for r in rows} != {"target", "control"}:
        fail("MISSING_CONTROLS", "Require nonempty distinct target and control groups")
    prompts = [r["prompt"].strip() for r in rows]
    if len(set(prompts)) != len(prompts):
        fail("DUPLICATE_CASES", "Repeated prompts cannot inflate independent sample counts or serve as controls")
    if any(any(not ref.strip() for ref in r["references"]) for r in rows):
        fail("INVALID_DATASET", "Certification references must be nonempty text")


def quality_decision(rows, candidate, reference, policy):
    """Pure statistical helper for tests; NOT a JSON certificate ingestion API."""
    admission_suite(rows, policy)
    expected = [r["id"] for r in rows]
    for result in (candidate, reference):
        cases = result.get("cases", [])
        if [c.get("id") for c in cases] != expected:
            fail("INVALID_EVIDENCE", "Paired case identities or lengths differ")
        for row, case in zip(rows, cases):
            if any(case.get(k) != row.get(k) for k in ("prompt", "references", "group")):
                fail("INVALID_EVIDENCE", "Scored case differs from heldout suite")
            if type(case.get("exact_match")) is not bool or not isinstance(case.get("generation"), str) or case["exact_match"] != (case["generation"] in row["references"]):
                fail("INVALID_EVIDENCE", "Literal score inconsistent with generation")
        score = result.get("exact_match")
        if type(score) not in (int, float) or not math.isfinite(score) or score != sum(c["exact_match"] for c in cases) / len(rows):
            fail("INVALID_EVIDENCE", "Invalid aggregate exact match")
    reasons, groups = [], {}
    for group in ("target", "control"):
        pairs = [(a["exact_match"], b["exact_match"]) for row, a, b in zip(rows, candidate["cases"], reference["cases"]) if row["group"] == group]
        n = len(pairs)
        c, r = sum(a for a, _ in pairs), sum(b for _, b in pairs)
        gains = sum(a and not b for a, b in pairs)
        losses = sum(b and not a for a, b in pairs)
        lower = binomial_lower(gains, n) - binomial_upper(losses, n)
        c_lower, r_lower = binomial_lower(c, n), binomial_lower(r, n)
        groups[group] = {"cases": n, "candidate_score": c / n, "reference_score": r / n,
                         "candidate_score_lower": c_lower, "reference_score_lower": r_lower,
                         "paired_gains": gains, "paired_losses": losses,
                         "candidate_minus_reference": (c - r) / n, "noninferiority_lower": lower}
        for condition, code, message in (
            (r_lower < policy["minimum_reference_score"], "REFERENCE_SCORE_FLOOR", "Reference absolute quality lower bound is insufficient"),
            (c_lower < policy["minimum_candidate_score"], "CANDIDATE_SCORE_FLOOR", "Candidate absolute quality lower bound is insufficient"),
            (lower < -policy["max_loss"], "NONINFERIORITY_UNPROVEN", "Paired one-sided lower bound does not establish noninferiority")):
            if condition:
                reasons.append({"code": code, "group": group, "message": message})
    return groups, reasons


def certificate_destination(output, model, reference_model):
    dest = safe_path(output, False)
    for root in (safe_path(model), safe_path(reference_model)):
        if dest == root or root in dest.parents:
            fail("UNSAFE_OUTPUT", "Certificate must be outside both model directories")
    if dest.exists():
        fail("OUTPUT_EXISTS", "Certificate output must not exist")
    if not dest.parent.is_dir():
        fail("UNSAFE_OUTPUT", "Certificate parent must already exist")
    return dest


def validate_precision_lineage(candidate, source, manifest, dtype):
    """Non-admitting provenance checks shared by research and v1 certification.

    source=None binds only the candidate, never asserts a source-relative result.
    Research artifacts are intentionally accepted here, not by validate_lineage.
    """
    if manifest.get("preflight", {}).get("dtype") != dtype:
        fail("DTYPE_MISMATCH", "Artifact must be loaded using its recorded prune dtype")
    files = {k: v for k, v in candidate["files"].items() if k != "compiler_manifest.json"}
    if manifest.get("candidate", {}).get("files") != files or manifest.get("candidate", {}).get("artifact_sha256") != canonical_hash(files):
        fail("LINEAGE_MISMATCH", "Candidate inventory/hash differs from manifest")
    original = manifest.get("source", {})
    if not isinstance(original.get("files"), dict) or original.get("artifact_sha256") != canonical_hash(original["files"]):
        fail("INVALID_LINEAGE", "Original source inventory/hash must be internally consistent")
    if manifest.get("source_unchanged") is not True or manifest.get("source_after_sha256") != original["artifact_sha256"]:
        fail("SOURCE_CHANGED", "Manifest must establish original source immutability")
    if source is not None and (original.get("artifact_sha256") != source["artifact_sha256"] or original.get("files") != source["files"]):
        fail("REFERENCE_MISMATCH", "Reference must match exact original source inventory and digest")


def research_artifact(manifest):
    experiment = manifest.get("experiment_config", {})
    return bool(experiment.get("audit_only") or experiment.get("preserve_router_fp32") or
                experiment.get("source_research_only") or experiment.get("scorer") == "reap_dispatched" or
                manifest.get("preflight", {}).get("dtype") == "bfloat16" or
                manifest.get("status") == "AUDIT_ONLY" or manifest.get("schema") == "switch-identity-roundtrip-v2")


def validate_lineage(candidate, source, manifest, dtype):
    if research_artifact(manifest):
        fail("AUDIT_ONLY_ARTIFACT", "Research BF16/router-precision/REAP/roundtrip arms cannot enter v1 certification")
    if dtype not in ("float32", "float16"):
        fail("INVALID_DTYPE", "v1 certification supports float32 or float16 only")
    if manifest.get("schema") != SCHEMA or manifest.get("architecture") != ARCH or manifest.get("evidence_scope") != SCOPE:
        fail("INVALID_LINEAGE", "Require compiler v1 seq2seq lineage manifest")
    validate_precision_lineage(candidate, source, manifest, dtype)
    if manifest.get("repair_training", {}).get("supported") is not False:
        fail("UNSUPPORTED_REPAIR", "Repair training remains unsupported")
    if not 1 <= candidate["num_experts"] < source["num_experts"] or manifest.get("num_experts") != candidate["num_experts"] or manifest.get("original_num_experts") != source["num_experts"] or not manifest.get("layers"):
        fail("STRUCTURAL_FAILURE", "Require physical expert reduction and recorded layer selection")


def size_decision(candidate, source, manifest, evidence, policy, dtype):
    result = {}
    for label, info, observed, lineage in (
        ("candidate", candidate, evidence["loaded_structure"], manifest["candidate"]),
        ("reference", source, evidence["reference"]["loaded_structure"], manifest["source"])):
        if observed.get("requested_dtype") != dtype or observed.get("unexpected_dtype_parameters") != []:
            fail("DTYPE_MISMATCH", "Actual loaded parameters must use prune dtype (native FP32 routers excepted)")
        if type(observed.get("parameters")) is not int or observed["parameters"] <= 0 or observed["parameters"] != lineage.get("parameters"):
            fail("STRUCTURAL_FAILURE", "Loaded unique parameter count differs from compiler manifest")
        width = 2 if dtype in ("float16", "bfloat16") else 4
        if observed.get("normalized_parameter_bytes") != observed["parameters"] * width:
            fail("STRUCTURAL_FAILURE", "Invalid normalized loaded parameter size")
        result[label] = {**observed, "stored_parameters": info["weights"]["stored_parameters"],
                         "normalized_weight_bytes": info["weights"]["stored_parameters"] * width,
                         "on_disk_weight_bytes": info["weights"]["weight_bytes"],
                         "on_disk_artifact_bytes": sum(f["bytes"] for f in info["files"].values())}
    reductions = {}
    reasons = []
    for field in ("parameters", "normalized_parameter_bytes", "normalized_weight_bytes"):
        denominator = result["reference"][field]
        if type(denominator) not in (int, float) or denominator <= 0:
            fail("STRUCTURAL_FAILURE", "Invalid source size")
        reductions[field] = 1.0 - result["candidate"][field] / denominator
        if reductions[field] < policy["minimum_size_reduction"]:
            reasons.append({"code": "INSUFFICIENT_SIZE_REDUCTION", "metric": field,
                            "message": "Dtype-normalized structural reduction is below policy"})
    result["reductions"] = reductions
    return result, reasons


def certify_model(model, reference_model, suite, certificate_output, minimum_cases=20, max_loss=0.02,
                  minimum_reference_score=0.5, minimum_candidate_score=0.5, minimum_size_reduction=0.1,
                  dtype="float32", max_length=128, max_new_tokens=32, memory_budget_mib=None):
    """Rerun this compiler's evaluator; never accept supplied evaluation JSON."""
    if dtype not in ("float32", "float16"):
        fail("INVALID_DTYPE", "v1 certification supports float32 or float16 only; BF16 is research-only")
    policy = admission_policy(minimum_cases, max_loss, minimum_reference_score,
                              minimum_candidate_score, minimum_size_reduction)
    bounds(max_length, max_new_tokens)
    dest = certificate_destination(certificate_output, model, reference_model)
    if safe_path(model) == safe_path(reference_model):
        fail("REFERENCE_MISMATCH", "Candidate and original source must be distinct")
    suite_hash = digest(safe_path(suite))
    _, rows = load_samples(suite, "suite")
    admission_suite(rows, policy)
    candidate, source = inspect_model(model), inspect_model(reference_model)
    manifest = read_json(safe_path(model) / "compiler_manifest.json")
    for name in ("compiler_manifest.json", "roundtrip_manifest.json"):
        path = safe_path(reference_model) / name
        if path.exists() and research_artifact(read_json(path)):
            fail("AUDIT_ONLY_ARTIFACT", "Research artifacts cannot be v1 certification reference ancestors")
    validate_lineage(candidate, source, manifest, dtype)
    # Deliberately direct internal invocation. No evaluation input file or external attestation.
    evidence = evaluate_model(model=model, reference_model=reference_model, suite=suite, dtype=dtype,
                              max_length=max_length, max_new_tokens=max_new_tokens,
                              memory_budget_mib=memory_budget_mib)
    for root, before, code in ((model, candidate, "MODEL_CHANGED"), (reference_model, source, "SOURCE_CHANGED")):
        if inventory(safe_path(root)) != before["files"]:
            fail(code, "Model/source changed during certification")
    if digest(safe_path(suite)) != suite_hash or evidence["suite_sha256"] != suite_hash:
        fail("SUITE_CHANGED", "Suite changed during certification")
    if evidence["model_sha256"] != candidate["artifact_sha256"] or evidence["reference"]["model_sha256"] != source["artifact_sha256"]:
        fail("LINEAGE_MISMATCH", "Evaluated model hashes differ from certified inventories")
    for observation in (evidence, evidence["reference"]):
        measured = observation.get("resources", {})
        if measured.get("observed") is not True:
            fail("INVALID_RESOURCE_METRICS", "Require observed evaluator resource measurements")
        for key in ("wall_seconds", "generation_wall_seconds", "process_peak_rss_bytes"):
            v = measured.get(key)
            if type(v) is not float or not math.isfinite(v) or v <= 0:
                fail("INVALID_RESOURCE_METRICS", "Require positive finite float resource measurements")
    if not evidence.get("runtime") or any(not isinstance(v, str) or not v for v in evidence["runtime"].values()):
        fail("INVALID_RESOURCE_METRICS", "Require recorded runtime versions")
    groups, reasons = quality_decision(rows, evidence["result"], evidence["reference"]["result"], policy)
    sizes, size_reasons = size_decision(candidate, source, manifest, evidence, policy, dtype)
    reasons.extend(size_reasons)
    certificate = {"schema": SCHEMA, "command": "certify", "status": "REJECTED" if reasons else "CERTIFIED",
        "admission": "UNADMITTED" if reasons else "ADMITTED_SEQ2SEQ_TEXT_ONLY", "reasons": reasons,
        "created_at": datetime.now(timezone.utc).isoformat(), "evidence_scope": SCOPE,
        "certifies_coding_skills": False, "autoactivated": False, "external_attestation": False,
        "repair_training": {"supported": False}, "source_unchanged": True, "candidate_unchanged": True,
        "source_sha256": source["artifact_sha256"], "candidate_sha256": candidate["artifact_sha256"],
        "source_inventory": source["files"], "candidate_inventory": candidate["files"],
        "manifest_sha256": candidate["files"]["compiler_manifest.json"]["sha256"],
        "suite_sha256": suite_hash, "policy": policy, "policy_sha256": canonical_hash(policy),
        "groups": groups, "sizes": sizes, "evaluation": evidence,
        "limitations": ["Local reproducible evidence, not a signature or external attestation",
                        "Seq2seq literal text proxy only; no coding capability certified",
                        "User-curated heldout independent-case assumption; exact prompt checks do not prove independence",
                        "Sequential RSS is process-lifetime high water, not isolated per-model peaks or a speed/memory improvement claim",
                        "No activation or mutation of either model; no repair training"]}
    certificate["certificate_sha256"] = canonical_hash(certificate)
    # Revalidate path immediately before exclusive creation; never touch model directories.
    dest = certificate_destination(dest, model, reference_model)
    try:
        with dest.open("x", encoding="utf-8") as handle:
            json.dump(certificate, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError:
        fail("OUTPUT_EXISTS", "Certificate output appeared during evaluation")
    return {**certificate, "certificate_output": str(dest)}
