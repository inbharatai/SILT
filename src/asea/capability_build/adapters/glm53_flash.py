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

Routing contract (audit 2026-09-19, corrected against the REAL implementation
in transformers 5.16.1, ``src/transformers/models/glm5_next/modeling_glm5_next.py``,
Apache-2.0): ``Glm5NextTextTopkRouter`` computes ``sigmoid(logits)`` scores,
adds an ``e_score_correction_bias`` buffer to the SELECTION scores only,
runs group-based top-k (top-2 within each group, summed; top ``topk_group``
groups kept), gathers the routing weights from the PRE-bias scores,
renormalises when ``norm_topk_prob`` and scales by ``routed_scaling_factor``.
The router returns the full decision ``(router_logits, topk_weights,
topk_indices)``. This adapter therefore captures the ACTUAL dispatched
expert ids and weights from the router's own output -- never a re-derived
approximation -- and the mask hook recomputes the selection under the same
equation (see :func:`_glm5_next_selection`). The earlier code refused a
config boolean named ``e_score_correction_bias`` (a field the real config
never had) and derived its own plain sigmoid top-k, which silently
misattributed usage on the real (bias-corrected, group top-k) router; both
defects were found by the 2026-09-19 external audit.

Honesty notes that bind here:

  * ``~18B active parameters`` is a per-token compute statement about the
    full model; it is NEVER treated as an identifiable 18B subset that
    could be extracted.
  * Router logits under sigmoid scoring give per-expert selection
    probabilities that do not sum to 1; ``collect_routing`` records
    sigmoid masses AND the actual dispatched counts/weights as USAGE
    evidence only, never causal importance.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

from ..errors import BlockedResource, InterventionInvalid
from .base import MoEArchitectureAdapter

#: The exact architecture this adapter supports. Detection is exact-shape,
#: not "closest match". Field names follow the real GLM-5.3-Flash config
#: (zai-org/GLM-5.3-Flash, config.json: model_type "glm5_next"; expert
#: counts under ``n_routed_experts``/``n_shared_experts`` (aliased to
#: ``num_local_experts`` by the transformers config); first-3-MLP-dense
#: under ``first_k_dense_replace``). The config declares NO ``scoring_func``
#: field: sigmoid scoring is structural to ``Glm5NextTextTopkRouter``, and
#: detection verifies the ROUTER MODULE CONTRACT (weight matrix shape,
#: ``e_score_correction_bias`` buffer, group/scaling attributes), not a
#: nonexistent config field. Runtime pin: transformers 5.16.1 -- the model
#: card's own ``transformers_version`` field says 5.16.0, but the 5.16.0
#: wheel does NOT contain ``models/glm5_next`` (verified: the directory
#: exists at tag v5.16.1 and 404s at v5.16.0); 5.16.1 is the minimal
#: release that actually ships the architecture.
EXPECTED = {
    "model_type": "glm5_next",
    "num_hidden_layers": 45,
    "n_routed_experts": 288,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "first_k_dense_replace": 3,
    "hidden_size": 4096,
    "max_position_embeddings_floor": 1_000_000,
}

#: The class the real checkpoint loads as (via AutoModelForMultimodalLM /
#: AutoModelForImageTextToText; there is no Glm5NextForCausalLM in
#: transformers 5.16.1).
_ACCEPTED_CLASSES = (
    "Glm5NextForConditionalGeneration",
    "Glm5NextForCausalLM",  # tolerated wrapper alias, kept for older stacks
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InterventionInvalid("GLM-5.3-Flash detection failed: %s" % message)


def _glm5_next_selection(
    torch, router, logits, suppress: FrozenSet[int] = frozenset()
) -> Tuple[Any, Any, Any]:
    """Replicates ``Glm5NextTextTopkRouter.forward`` (transformers 5.16.1,
    Apache-2.0, huggingface/transformers@v5.16.1,
    ``src/transformers/models/glm5_next/modeling_glm5_next.py``) EXACTLY,
    starting from the already-computed router logits, with ONE extension:
    experts in ``suppress`` are forced to ``-inf`` in the selection scores
    BEFORE the group stage -- the suppression point the mask semantics
    require. Without it a masked expert's logit of ``-1e9`` gives
    ``sigmoid(-1e9) = 0`` but a nonzero ``e_score_correction_bias`` added
    afterwards could still push it into the top-k -- silent
    under-suppression, the exact defect class this project refuses.

    Returns ``(scores, topk_indices, topk_weights)`` where ``scores`` are the
    PRE-bias sigmoid scores the real router gathers weights from.
    """
    top_k = int(getattr(router, "top_k"))
    num_experts = int(getattr(router, "num_experts"))
    num_group = int(getattr(router, "num_group"))
    topk_group = int(getattr(router, "topk_group"))
    norm_topk_prob = bool(getattr(router, "norm_topk_prob"))
    scaling = float(getattr(router, "routed_scaling_factor"))
    bias = getattr(router, "e_score_correction_bias", None)
    scores = torch.sigmoid(logits.detach().float())
    scores_for_choice = (
        scores + bias.to(scores.dtype)
        if torch.is_tensor(bias) else scores
    )
    if suppress:
        scores_for_choice = scores_for_choice.masked_fill(
            torch.zeros_like(scores_for_choice, dtype=torch.bool)
            .index_fill_(1, torch.tensor(sorted(suppress), device=scores.device), True),
            float("-inf"),
        )
    group_scores = (
        scores_for_choice.view(-1, num_group, num_experts // num_group)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, num_group, num_experts // num_group)
        .reshape(-1, num_experts)
    )
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    topk_indices = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_indices)
    if norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denominator
    topk_weights = topk_weights * scaling
    return scores, topk_indices, topk_weights


def _router_output_contract(torch, output, layer_index: int, router_name: str):
    """Validate the v5.16.1 router return contract ``(router_logits,
    topk_weights, topk_indices)`` and unpack it. A revision that changes the
    contract is a typed detection failure, never a silent misread."""
    if not (isinstance(output, tuple) and len(output) == 3):
        raise InterventionInvalid(
            "GLM-5.3-Flash detection failed: router %r at layer %d returned "
            "%s, not the (router_logits, topk_weights, topk_indices) tuple "
            "of the v5.16.1 Glm5NextTextTopkRouter contract; this transformers "
            "revision's routing cannot be read faithfully"
            % (router_name, layer_index, type(output).__name__)
        )
    logits, weights, indices = output
    for name, tensor in (("router_logits", logits), ("topk_weights", weights),
                         ("topk_indices", indices)):
        if not torch.is_tensor(tensor):
            raise InterventionInvalid(
                "GLM-5.3-Flash detection failed: router %r at layer %d "
                "returned a non-tensor %s (%r); the routing contract is "
                "broken by this transformers revision"
                % (router_name, layer_index, name, type(tensor).__name__)
            )
    if indices.dtype.is_floating_point:
        raise InterventionInvalid(
            "GLM-5.3-Flash detection failed: router %r at layer %d returned "
            "floating-point topk_indices; expert ids must be integers"
            % (router_name, layer_index)
        )
    return logits, weights, indices


def _validate_router_module(router, layer_index: int) -> Dict[str, Any]:
    """The router MODULE contract (not config fields): a ``weight`` matrix of
    shape (num_experts, hidden), an ``e_score_correction_bias`` buffer over
    the experts, and the group/scaling attributes of the v5.16.1 router."""
    _require(
        hasattr(router, "weight"),
        "router at layer %d has no weight matrix" % layer_index,
    )
    shape = tuple(getattr(router.weight, "shape", ()))
    _require(
        len(shape) == 2 and shape[0] == EXPECTED["n_routed_experts"]
        and shape[1] == EXPECTED["hidden_size"],
        "router weight shape %r at layer %d != (288, 4096)"
        % (shape, layer_index),
    )
    bias = getattr(router, "e_score_correction_bias", None)
    _require(
        bias is not None and tuple(getattr(bias, "shape", ())) == (EXPECTED["n_routed_experts"],),
        "router at layer %d carries no e_score_correction_bias buffer over "
        "288 experts; this is not the v5.16.1 Glm5NextTextTopkRouter contract"
        % layer_index,
    )
    for attribute in ("top_k", "num_experts", "num_group", "topk_group",
                      "norm_topk_prob", "routed_scaling_factor"):
        _require(
            hasattr(router, attribute),
            "router at layer %d lacks %r (v5.16.1 router contract)"
            % (layer_index, attribute),
        )
    _require(
        int(router.top_k) == EXPECTED["num_experts_per_tok"],
        "router top_k %r != 8 at layer %d" % (router.top_k, layer_index),
    )
    _require(
        int(router.num_experts) == EXPECTED["n_routed_experts"],
        "router num_experts %r != 288 at layer %d"
        % (router.num_experts, layer_index),
    )
    return {
        "num_group": int(router.num_group),
        "topk_group": int(router.topk_group),
        "norm_topk_prob": bool(router.norm_topk_prob),
        "routed_scaling_factor": float(router.routed_scaling_factor),
    }


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
        or getattr(text, "model_type", None) in (EXPECTED["model_type"], "glm5_next_text"),
        "model_type %r != %r"
        % (getattr(config, "model_type", None), EXPECTED["model_type"]),
    )
    _require(
        int(getattr(text, "num_hidden_layers", -1)) == EXPECTED["num_hidden_layers"],
        "num_hidden_layers %r != 45" % getattr(text, "num_hidden_layers", None),
    )
    n_routed = getattr(text, "n_routed_experts", None)
    if n_routed is None:
        n_routed = getattr(text, "num_local_experts",
                           getattr(text, "num_routed_experts", None))
    _require(
        n_routed is not None and int(n_routed) == EXPECTED["n_routed_experts"],
        "n_routed_experts %r != 288" % n_routed,
    )
    n_shared = getattr(text, "n_shared_experts", None)
    if n_shared is None:
        n_shared = getattr(text, "num_shared_experts", None)
    _require(
        n_shared is not None and int(n_shared) == EXPECTED["n_shared_experts"],
        "n_shared_experts %r != 1" % n_shared,
    )
    _require(
        int(getattr(text, "num_experts_per_tok", -1)) == EXPECTED["num_experts_per_tok"],
        "num_experts_per_tok %r != 8" % getattr(text, "num_experts_per_tok", None),
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
        "router_contract": "v5.16.1 Glm5NextTextTopkRouter: sigmoid scores + "
                           "e_score_correction_bias on selection scores only, "
                           "group top-k, pre-bias weight gather, routed scaling",
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
        self.n_experts = EXPECTED["n_routed_experts"]
        self.experts_per_tok = EXPECTED["num_experts_per_tok"]
        # The ROUTER MODULE contract is verified on every sparse layer: the
        # real v5.16.1 router (weight shape, e_score_correction_bias buffer,
        # group/scaling attributes) is what the telemetry and mask hooks
        # rely on; a stack that merely looks like GLM is refused here
        # rather than misinstrumented (audit 2026-09-19).
        self.router_params: Dict[int, Dict[str, Any]] = {}
        for index, block in self.layers:
            _, router = self._router_of(block)
            self.router_params[index] = _validate_router_module(router, index)
        self._telemetry = None
        self._stats: Optional[Dict[int, Dict[str, Any]]] = None
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
            ("model", "text_model", "layers"),
            ("model", "model", "language_model", "layers"),
            ("language_model", "layers"),
            ("text_model", "layers"),
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
        inspection = dict(self.inspection)
        inspection["router_params"] = dict(self.router_params)
        return inspection

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

    def active_masks(self) -> List[str]:
        """The live suppression set (registry keys "layer_index:target").
        Read-only reporting used by the worker's generate op so a caller
        can never attribute a completion to the wrong mask state."""
        return sorted(self._masked)

    # -- telemetry ----------------------------------------------------------

    def _router_of(self, block):
        """The v5.16.1 router module, located by its CONTRACT (weight matrix
        + e_score_correction_bias buffer), never by name guessing: the MoE
        also contains ``gate_up_proj`` expert projections whose names
        contain ``gate`` but which are not routers."""
        candidates = []
        for name, module in block.named_modules():
            if hasattr(module, "e_score_correction_bias") and hasattr(module, "weight"):
                candidates.append((name, module))
        _require(
            len(candidates) == 1,
            "expected exactly one e_score_correction_bias router module per "
            "sparse layer, found %d" % len(candidates),
        )
        return candidates[0]

    def register_router_hooks(self) -> None:
        """Sigmoid-mass + ACTUAL DISPATCH telemetry (USAGE EVIDENCE ONLY).

        The v5.16.1 router returns ``(router_logits, topk_weights,
        topk_indices)`` -- the dispatch decision itself. The hook records
        the analytic sigmoid mass AND the real dispatched expert ids /
        weight sums straight from the router's own output, so the recorded
        usage is the ACTUAL routing under the correction-bias + group-top-k
        equation, never a re-derived approximation.

        Idempotent (audit 2026-09-18): a previous registration's hook
        handles are removed first. Re-registering used to reassign
        ``self._telemetry`` without removing the old handles, leaking
        live hooks whose entry-pointers wrote into a dict nobody read
        anymore -- a second trace on the same adapter silently doubled
        ``tokens`` in the stale dict while the fresh one stayed empty."""
        for handle in self._telemetry or []:
            handle.remove()
        if self._masked:
            raise InterventionInvalid(
                "cannot register telemetry while masks are active (%s): the "
                "mask hook would race the telemetry hook on the same router "
                "output and record the UNMASKED dispatch as if masked; "
                "restore all masks first" % sorted(self._masked)
            )
        torch = self.torch
        n = self.n_experts
        self._stats: Dict[int, Dict[str, Any]] = {}

        def hook(layer_index, router_name):
            def collect(module, inputs, output):
                logits, weights, indices = _router_output_contract(
                    torch, output, layer_index, router_name)
                if logits.shape[-1] != n:
                    raise InterventionInvalid(
                        "router %r at layer %d returned %d logits, expected "
                        "%d" % (router_name, layer_index, logits.shape[-1], n)
                    )
                if weights.shape != indices.shape:
                    raise InterventionInvalid(
                        "router %r at layer %d returned weights %r and "
                        "indices %r of different shapes"
                        % (router_name, layer_index, tuple(weights.shape),
                           tuple(indices.shape))
                    )
                if indices.dim() != 2 or indices.shape[-1] != self.experts_per_tok:
                    raise InterventionInvalid(
                        "router %r at layer %d returned topk_indices %r; "
                        "the v5.16.1 contract is [tokens, %d]"
                        % (router_name, layer_index, tuple(indices.shape),
                           self.experts_per_tok)
                    )
                if int(indices.max()) >= n or int(indices.min()) < 0:
                    raise InterventionInvalid(
                        "router %r at layer %d dispatched an expert id outside "
                        "0..%d" % (router_name, layer_index, n - 1)
                    )
                entry = self._stats.setdefault(
                    layer_index,
                    {
                        "sigmoid_mass": [0.0] * n,
                        "dispatched_count": [0] * n,
                        "dispatched_weight_sum": [0.0] * n,
                        "tokens": 0,
                        "nonfinite_router_values": 0,
                        "correction_bias_nonzero": bool(
                            module.e_score_correction_bias.abs().sum() > 0
                        ),
                    },
                )
                scores = torch.sigmoid(logits.detach().float())
                flat = scores.reshape(-1, n)
                entry["tokens"] += int(flat.shape[0])
                mass = flat.sum(0).cpu().tolist()
                for expert in range(n):
                    entry["sigmoid_mass"][expert] += float(mass[expert])
                # ACTUAL dispatch: bincount over the router's own topk ids
                # ([tokens, experts_per_tok]), and per-expert weight sums
                # from its own weights (the post-bias, post-group,
                # post-scaling values in effect).
                flat_ids = indices.reshape(-1).long()
                flat_weights = weights.reshape(-1).float()
                counts = torch.bincount(flat_ids, minlength=n).cpu().tolist()
                weight_sums = (
                    torch.zeros(n).scatter_add(0, flat_ids, flat_weights)
                    .cpu()
                    .tolist()
                )
                for expert in range(n):
                    entry["dispatched_count"][expert] += int(counts[expert])
                    entry["dispatched_weight_sum"][expert] += float(weight_sums[expert])
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

    def remove_router_hooks(self) -> bool:
        """Remove the telemetry hooks if any are live (idempotent, returns
        whether live handles were removed). The worker calls this in a
        ``finally`` so a routing pass that raised never leaves live hooks
        attached to a teacher whose parameter hashes still verify clean --
        the exact contamination ``active_masks`` reporting exists to catch
        (audit 2026-09-18)."""
        removed = False
        for handle in self._telemetry or []:
            handle.remove()
            removed = True
        self._telemetry = None
        return removed

    def collect_routing(self, reset: bool = True) -> Dict[str, Any]:
        _require(self._stats is not None, "register_router_hooks first")
        for index, entry in self._stats.items():
            if entry["nonfinite_router_values"]:
                raise InterventionInvalid(
                    "nonfinite router values observed at layer %d" % index
                )
        result = {
            str(index): {
                "sigmoid_mass": entry["sigmoid_mass"],
                "dispatched_count": entry["dispatched_count"],
                "dispatched_weight_sum": entry["dispatched_weight_sum"],
                "tokens": entry["tokens"],
                "correction_bias_nonzero": entry["correction_bias_nonzero"],
                "router_params": self.router_params.get(int(index), {}),
                "usage_evidence_not_causal_importance": True,
                "router_scoring": "sigmoid_top%d_bias_corrected_group_topk"
                                  % self.experts_per_tok,
            }
            for index, entry in self._stats.items()
        }
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
        current = self._current_hashes()
        changed = [
            name
            for name, digest in self._baseline_hashes.items()
            if current.get(name) != digest
        ]
        changed.extend(name for name in current if name not in self._baseline_hashes)
        # Parameter hashes CANNOT see suppression hooks (masks rewrite the
        # router OUTPUT at forward time and never touch a parameter), so the
        # active-mask registry rides along: a masked teacher must never
        # verify as "unchanged" for the purposes of a NEW intervention even
        # though its weights are untouched (audit 2026-09-18).
        return {
            "unchanged": not changed,
            "detail": "hash-identical to baseline"
            if not changed
            else "modified parameters: %s" % sorted(changed)[:16],
            "active_masks": sorted(self._masked),
        }

    def temporary_mask(self, target, *, scale: float = 0.0) -> Dict[str, Any]:
        """Suppress the masked expert's ROUTING DECISION, weights untouched.

        A weight-row rewrite is mathematically UNRELIABLE under this
        router and was removed: writing a constant ``-c`` into an expert's
        router rows makes its logit ``-c * sum(hidden)``, which is
        STRONGLY POSITIVE whenever the hidden-state sum is negative --
        the masked expert would become MORE likely to enter the top-8,
        the exact opposite of the intervention.

        A forward hook on the router module rewrites the FULL router
        output. The masked expert's logit becomes ``-1e9`` for every
        token, input-independent, AND its selection score
        (``sigmoid(logit) + e_score_correction_bias``) is forced to
        ``-inf`` before the group stage, so the expert can never enter
        the top-``experts_per_tok`` selection even with a nonzero
        correction bias; the returned ``(topk_weights, topk_indices)``
        are recomputed under the exact v5.16.1 routing equation
        (:func:`_glm5_next_selection`). Router weights, expert weights
        and every other parameter are never modified, so
        ``verify_unchanged`` holds during the mask window trivially and
        honestly."""
        if scale != 0.0:
            raise InterventionInvalid(
                "partial-scale masking (scale=%r) is not implemented; only "
                "full suppression is, and this adapter refuses to pretend "
                "otherwise" % scale
            )
        _require(
            target.kind == "expert",
            "masking kind %r is not supported: masking EVERY expert does not "
            "disable top-%d routing (top-k still fires over all -1e9 logits) "
            "and would silently under-suppress"
            % (target.kind, self.experts_per_tok),
        )
        _require(0 <= target.layer < len(self.layers), "layer index out of range")
        _require(
            0 <= target.expert_id < self.n_experts,
            "expert %d out of range 0..%d" % (target.expert_id, self.n_experts - 1),
        )
        if self._telemetry:
            raise InterventionInvalid(
                "cannot mask while telemetry hooks are live: the telemetry "
                "hook registered first would record the router's UNMASKED "
                "dispatch; call remove_router_hooks() first"
            )
        index, block = self.layers[target.layer]
        router_name, router = self._router_of(block)
        registry_key = "%d:%s" % (index, target.key)
        _require(
            registry_key not in self._masked,
            "component already masked; restore before re-masking",
        )
        indices = [target.expert_id]

        def suppress(module, inputs, output):
            torch = self.torch
            logits, weights, indices_out = _router_output_contract(
                torch, output, index, router_name)
            masked_logits = logits.clone()
            for expert in indices:
                masked_logits[..., expert] = -1.0e9
            _, new_indices, new_weights = _glm5_next_selection(
                torch, module, masked_logits, suppress=frozenset(indices)
            )
            return (masked_logits, new_weights, new_indices)

        handle = router.register_forward_hook(suppress)
        self._masked[registry_key] = {
            "indices": indices,
            "handle": handle,
            "router_name": router_name,
            "layer_index": index,
        }
        return {
            "key": registry_key,
            "layer": index,
            "indices": indices,
            "mechanism": "router_output_forward_hook_full_tuple",
            "weights_modified": False,
        }

    def restore_mask(self, target) -> Dict[str, Any]:
        _require(0 <= target.layer < len(self.layers), "layer index out of range")
        index, block = self.layers[target.layer]
        registry_key = "%d:%s" % (index, target.key)
        record = self._masked.get(registry_key)
        _require(
            record is not None,
            "no recorded mask; cannot restore what was never masked",
        )
        # Remove the hook BEFORE dropping the registry record (audit
        # 2026-09-18): if handle.remove() raises, the record stays so the
        # adapter keeps reporting the mask ACTIVE. The old order popped
        # first -- a failed remove left the suppression hook live while the
        # adapter believed the teacher restored (and parameter hashes
        # cannot see hooks), silently corrupting every later measurement.
        record["handle"].remove()
        self._masked.pop(registry_key)
        return {
            "key": registry_key,
            "restored": True,
            "mechanism": "router_output_forward_hook_removed",
            "weights_modified": False,
        }

    # -- reduction -----------------------------------------------------------

    def structural_reduce(self, keep_components) -> Dict[str, Any]:
        """GLM-5.3-Flash structural reduction is NOT implemented anywhere in
        this build: the worker exposes no ``reduce`` op, and reducing a
        ~320B-parameter teacher in-process on the host is out of the
        question. The honest refusal names the gap instead of directing the
        operator to a nonexistent handler (audit 2026-09-19 correction:
        the old remedy pointed at "the worker's reduce operation", which
        does not exist)."""
        raise BlockedResource(
            requirement=(
                "GLM-5.3-Flash structural reduction is not implemented: no "
                "component of this build can prune the teacher, and no "
                "reduce operation exists in the worker; a reduction claim "
                "would be fabricated"
            ),
            remedy="implement the worker-side physical reduction (memory-"
                   "admitted, expert renumbering like the compiler's "
                   "apply_selection) and execute it on a box that can hold "
                   "the teacher before any reduction is reported",
        )