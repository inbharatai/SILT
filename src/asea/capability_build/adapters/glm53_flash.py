"""GLM-5.3-Flash adapter (``glm5_next_moe``) -- WORKER-ONLY.

This adapter runs ONLY inside the isolated GLM worker
(:mod:`workers.glm53.worker`) because it requires the Transformers 5.x
runtime. It is NEVER imported into the main SILT environment, whose
``localmodels`` extra pins ``transformers==4.51.3`` exactly (rule 13/14:
the GLM runtime is isolated; the global pin is untouched). The main-package
imports below are stdlib-only.

Exact-architecture detection is mandatory BEFORE any behavior: the loaded
checkpoint must match the GLM-5.3-Flash shape this adapter was built
against (45 layers, 288 routed + 1 shared expert, 8 experts/token, sigmoid
router scoring, first 3 MLP layers dense, hidden 4096, ~1M positions, BF16
text weights, vision tower present). Anything else fails with a typed
error -- never a best-effort guess on a different architecture.

Honesty notes that bind here:

  * ``~18B active parameters`` is a per-token compute statement about the
    full model; it is NEVER treated as an identifiable 18B subset that
    could be extracted.
  * Router logits under sigmoid scoring give per-expert selection
    probabilities that do not sum to 1; ``collect_routing`` records
    sigmoid masses and top-k counts as USAGE evidence only, never causal
    importance.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from ..errors import BlockedResource, InterventionInvalid
from .base import MoEArchitectureAdapter

#: The exact architecture this adapter supports. Detection is exact-shape,
#: not "closest match". Field names follow the real GLM-5.3-Flash config
#: (zai-org/GLM-5.3-Flash, config.json: model_type "glm5_next"; expert
#: counts under ``n_routed_experts``/``n_shared_experts``; sigmoid router
#: scoring under ``scoring_func``; first-3-MLP-dense under
#: ``first_k_dense_replace``). Transformers pin: the config's
#: ``transformers_version`` field = 5.16.0.
EXPECTED = {
    "model_type": "glm5_next",
    "num_hidden_layers": 45,
    "n_routed_experts": 288,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "scoring_func": "sigmoid",
    "first_k_dense_replace": 3,
    "hidden_size": 4096,
    "max_position_embeddings_floor": 1_000_000,
}

#: Class names accepted for the text backbone (ConditionalGeneration
#: wraps the same backbone).
_ACCEPTED_CLASSES = (
    "Glm5NextForConditionalGeneration",
    "Glm5NextForCausalLM",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InterventionInvalid("GLM-5.3-Flash detection failed: %s" % message)


def detect_architecture(model, config) -> Dict[str, Any]:
    """Exact-shape detection. Returns the inspection inventory or raises
    :class:`InterventionInvalid` naming the mismatch. The MoE values live
    under ``text_config`` in the composite ConditionalGeneration config;
    both nesting shapes are read, everything else fails closed."""
    text = getattr(config, "text_config", None) or config
    _require(
        type(model).__name__ in _ACCEPTED_CLASSES,
        "class %r is not %s" % (type(model).__name__, "/".join(_ACCEPTED_CLASSES)),
    )
    _require(
        getattr(config, "model_type", None) == EXPECTED["model_type"]
        or getattr(text, "model_type", None) == EXPECTED["model_type"],
        "model_type %r != %r"
        % (getattr(config, "model_type", None), EXPECTED["model_type"]),
    )
    _require(
        int(getattr(text, "num_hidden_layers", -1)) == EXPECTED["num_hidden_layers"],
        "num_hidden_layers %r != 45" % getattr(text, "num_hidden_layers", None),
    )
    _require(
        int(getattr(text, "n_routed_experts", getattr(text, "num_routed_experts", -1)))
        == EXPECTED["n_routed_experts"],
        "n_routed_experts %r != 288"
        % getattr(text, "n_routed_experts", None),
    )
    _require(
        int(getattr(text, "n_shared_experts", getattr(text, "num_shared_experts", -1)))
        == EXPECTED["n_shared_experts"],
        "n_shared_experts %r != 1" % getattr(text, "n_shared_experts", None),
    )
    _require(
        int(getattr(text, "num_experts_per_tok", -1)) == EXPECTED["num_experts_per_tok"],
        "num_experts_per_tok %r != 8" % getattr(text, "num_experts_per_tok", None),
    )
    _require(
        str(getattr(text, "scoring_func", getattr(text, "moe_router_score_type", ""))).lower()
        == EXPECTED["scoring_func"],
        "router scoring_func %r is not sigmoid"
        % getattr(text, "scoring_func", None),
    )
    _require(
        int(getattr(text, "first_k_dense_replace", -1)) == EXPECTED["first_k_dense_replace"],
        "first_k_dense_replace %r != 3 (first 3 MLP layers must be dense)"
        % getattr(text, "first_k_dense_replace", None),
    )
    _require(
        int(getattr(text, "hidden_size", -1)) == EXPECTED["hidden_size"],
        "hidden_size %r != 4096" % getattr(text, "hidden_size", None),
    )
    positions = int(getattr(text, "max_position_embeddings", 0) or 0)
    _require(
        positions >= EXPECTED["max_position_embeddings_floor"],
        "max_position_embeddings %r < 1M" % positions,
    )
    dtype = str(getattr(text, "torch_dtype", getattr(text, "dtype", "")))
    _require(
        "bfloat16" in dtype,
        "text weights are %r, expected bfloat16" % dtype,
    )
    has_vision = any(
        "vision" in name.lower() for name, _ in model.named_modules()
    )
    _require(has_vision, "no vision tower module found (GLM-5.3-Flash has one)")
    quantized = bool(getattr(model, "is_quantized", False))
    return {
        "arch": "glm5_next_moe",
        "class": type(model).__name__,
        "detected": True,
        "quantization": quantized,
        "quantization_note": (
            "quantized checkpoints change the parameter-hash contract; "
            "interventions on quantized weights are refused"
            if quantized
            else None
        ),
        "expected": dict(EXPECTED),
        "active_parameters_note": (
            "~18B active is a per-token compute statement about the FULL "
            "model; it is not an identifiable 18B subset and no extraction "
            "claim is made or implied"
        ),
    }


class Glm53FlashAdapter(MoEArchitectureAdapter):
    arch = "glm5_next_moe"

    def __init__(self, model, torch, config=None) -> None:
        config = config or getattr(model, "config", None)
        self.inspection = detect_architecture(model, config)
        self.model = model
        self.torch = torch
        self.config = config
        self.text = getattr(config, "text_config", None) or config
        if self.inspection["quantization"]:
            raise BlockedResource(
                requirement=(
                    "GLM-5.3-Flash loaded in a QUANTIZED state (%s); "
                    "temporary interventions on packed quantized tensors "
                    "cannot be provably restored bit-identically"
                    % self.inspection["quantization"]
                ),
                remedy="load the BF16 checkpoint (zai-org/GLM-5.3-Flash-BF16) "
                "in the worker on a box with sufficient memory, or run the "
                "behavioural (cloud) evidence class instead",
            )
        self.layers = self._find_sparse_layers()
        self.n_experts = int(
            getattr(self.text, "n_routed_experts", getattr(self.text, "num_routed_experts"))
        )
        self.experts_per_tok = int(getattr(self.text, "num_experts_per_tok"))
        self._telemetry = None
        self._baseline_hashes: Optional[Dict[str, str]] = None
        self._masked: Dict[str, Dict[str, Any]] = {}

    def _decoder_blocks(self):
        """Locate the decoder layer list on any nesting of the composite
        Glm5NextForConditionalGeneration wrapper, without hard-coding one
        transformers internals layout."""
        candidate_paths = (
            ("model", "layers"),               # CausalLM shape
            ("model", "model", "layers"),      # ConditionalGeneration shapes
            ("model", "language_model", "layers"),
            ("model", "model", "language_model", "layers"),
            ("language_model", "layers"),
        )
        for path in candidate_paths:
            node = self.model
            try:
                for attribute in path:
                    node = getattr(node, attribute)
                blocks = list(node)
                if blocks and all(hasattr(b, "named_modules") for b in blocks):
                    return blocks
            except (AttributeError, TypeError):
                continue
        raise InterventionInvalid(
            "could not locate the GLM decoder layer list on this wrapper; "
            "refusing to guess (transformers %s)"
            % getattr(__import__("transformers"), "__version__", "unknown")
        )

    def _find_sparse_layers(self) -> List[Any]:
        """MoE decoder layers = every layer index >= first_k_dense_replace
        whose modules include a routed-expert container."""
        dense = int(getattr(self.text, "first_k_dense_replace"))
        sparse = []
        for index, block in enumerate(self._decoder_blocks()):
            if index < dense:
                continue
            names = [name for name, _ in block.named_modules()]
            if any("experts" in name for name in names):
                sparse.append((index, block))
        _require(
            len(sparse) == int(getattr(self.text, "num_hidden_layers")) - dense,
            "found %d sparse layers, expected %d"
            % (len(sparse), int(getattr(self.text, "num_hidden_layers")) - dense),
        )
        return sparse

    # -- read-only -----------------------------------------------------------

    def inspect_model(self) -> Dict[str, Any]:
        return dict(self.inspection)

    def parameter_inventory(self) -> Dict[str, Any]:
        grand = 0
        by_kind: Dict[str, int] = {}
        for _, parameter in self.model.named_parameters():
            count = int(parameter.numel())
            grand += count
            kind = parameter.dtype
            by_kind[str(kind)] = by_kind.get(str(kind), 0) + count
        return {
            "total_parameters": grand,
            "by_dtype": by_kind,
            "note": "counts are about tensor storage, not capability",
        }

    def enumerate_sparse_layers(self) -> List[int]:
        return [index for index, _ in self.layers]

    # -- telemetry ----------------------------------------------------------

    def _router_of(self, block):
        for name, module in block.named_modules():
            if name.endswith("router") or "gate" in name.rsplit(".", 1)[-1]:
                return name, module
        raise InterventionInvalid("no router module found in a GLM sparse layer")

    def register_router_hooks(self) -> None:
        """Sigmoid-mass + top-k usage telemetry (USAGE EVIDENCE ONLY)."""
        torch = self.torch
        n = self.n_experts
        k = self.experts_per_tok
        self._stats: Dict[int, Dict[str, Any]] = {}

        def hook(layer_index, router_name):
            def collect(module, inputs, output):
                logits = output[0] if isinstance(output, tuple) else output
                entry = self._stats.setdefault(
                    layer_index,
                    {
                        "sigmoid_mass": [0.0] * n,
                        "topk_count": [0] * n,
                        "tokens": 0,
                        "nonfinite_router_values": 0,
                    },
                )
                scores = torch.sigmoid(logits.detach().float())
                flat = scores.reshape(-1, n)
                entry["tokens"] += int(flat.shape[0])
                mass = flat.sum(0).cpu().tolist()
                for expert in range(n):
                    entry["sigmoid_mass"][expert] += float(mass[expert])
                top = flat.topk(k, dim=-1).indices.reshape(-1).cpu().tolist()
                for expert in top:
                    entry["topk_count"][expert] += 1
                entry["nonfinite_router_values"] += int(
                    (~torch.isfinite(logits.detach())).sum()
                )
            return collect

        self._telemetry = []
        for index, block in self.layers:
            router_name, router = self._router_of(block)
            self._telemetry.append(
                router.register_forward_hook(hook(index, router_name))
            )

    def collect_routing(self, reset: bool = True) -> Dict[str, Any]:
        _require(self._stats is not None, "register_router_hooks first")
        result = {
            str(index): {
                "sigmoid_mass": entry["sigmoid_mass"],
                "topk_count": entry["topk_count"],
                "tokens": entry["tokens"],
                "usage_evidence_not_causal_importance": True,
                "router_scoring": "sigmoid_top%d" % self.experts_per_tok,
            }
            for index, entry in self._stats.items()
        }
        for index, entry in self._stats.items():
            if entry["nonfinite_router_values"]:
                raise InterventionInvalid(
                    "nonfinite router values observed at layer %d" % index
                )
        if reset:
            self._stats = {}
        return result

    def collect_expert_outputs(self, reset: bool = True) -> Dict[str, Any]:
        # REAP-style expert-output statistics for GLM are collected inside
        # the worker via per-expert hooks; the in-package adapter records
        # router usage only until that path is exercised live. Absence is
        # stated, not faked.
        return {"note": "expert_output_collection_not_implemented_in_adapter"}

    # -- intervention protocol -----------------------------------------------

    def freeze_baseline(self) -> Dict[str, str]:
        self._baseline_hashes = self._current_hashes()
        return self._baseline_hashes

    def _current_hashes(self) -> Dict[str, str]:
        hashes: Dict[str, str] = {}
        with self.torch.inference_mode():
            for name, parameter in self.model.named_parameters():
                raw = (
                    parameter.detach()
                    .to("cpu", copy=True)
                    .contiguous()
                    .view(self.torch.uint8)
                )
                hashes[name] = hashlib.sha256(raw.numpy().tobytes()).hexdigest()
        return hashes

    def verify_unchanged(self) -> Dict[str, Any]:
        if self._baseline_hashes is None:
            self.freeze_baseline()
            return {"unchanged": True, "detail": "baseline frozen on first use"}
        current = self._current_hashes()
        changed = [
            name
            for name, digest in self._baseline_hashes.items()
            if current.get(name) != digest
        ]
        changed.extend(name for name in current if name not in self._baseline_hashes)
        return {
            "unchanged": not changed,
            "detail": "hash-identical to baseline"
            if not changed
            else "modified parameters: %s" % sorted(changed)[:16],
        }

    def temporary_mask(self, target, *, scale: float = 0.0) -> Dict[str, Any]:
        _require(0 <= target.layer < len(self.layers), "layer index out of range")
        index, block = self.layers[target.layer]
        router_name, router = self._router_of(block)
        gate = None
        for name, module in router.named_modules():
            if isinstance(module, self.torch.nn.Linear):
                gate = module
                break
        _require(gate is not None, "no router Linear found at layer %d" % index)
        if target.kind == "expert":
            indices = [target.expert_id]
            _require(
                0 <= target.expert_id < self.n_experts,
                "expert %d out of range 0..%d" % (target.expert_id, self.n_experts - 1),
            )
        else:
            indices = list(range(self.n_experts))
        registry_key = "%d:%s" % (index, target.key)
        _require(
            registry_key not in self._masked,
            "component already masked; restore before re-masking",
        )
        saved = {"weight": gate.weight.data[indices].clone()}
        if gate.bias is not None:
            saved["bias"] = gate.bias.data[indices].clone()
        with self.torch.no_grad():
            # Sigmoid(x) -> 0 for very negative x: the expert is never in
            # the top-k selection, without touching expert weights.
            gate.weight.data[indices] = -30.0
            if gate.bias is not None:
                gate.bias.data[indices] = -30.0
        self._masked[registry_key] = {
            "indices": indices,
            "saved": saved,
            "gate": gate,
        }
        return {"key": registry_key, "layer": index, "indices": indices}

    def restore_mask(self, target) -> Dict[str, Any]:
        _require(0 <= target.layer < len(self.layers), "layer index out of range")
        index, block = self.layers[target.layer]
        registry_key = "%d:%s" % (index, target.key)
        record = self._masked.pop(registry_key, None)
        _require(
            record is not None,
            "no recorded mask; cannot restore what was never masked",
        )
        gate = record["gate"]
        indices = record["indices"]
        with self.torch.no_grad():
            gate.weight.data[indices] = record["saved"]["weight"]
            if gate.bias is not None and "bias" in record["saved"]:
                gate.bias.data[indices] = record["saved"]["bias"]
        return {"key": registry_key, "restored": True}

    # -- reduction -----------------------------------------------------------

    def structural_reduce(self, keep_components) -> Dict[str, Any]:
        """GLM-5.3-Flash structural reduction runs inside the worker with
        its own memory admission; the main-package adapter refuses to
        reduce on this side (the host cannot hold the teacher)."""
        raise BlockedResource(
            requirement=(
                "GLM-5.3-Flash structural reduction must run inside the "
                "isolated worker next to the loaded teacher"
            ),
            remedy="invoke the worker's reduce operation; do not attempt "
            "in-process reduction of a 320B-parameter teacher",
        )