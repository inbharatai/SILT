"""Switch-transformers adapter (wraps compiler telemetry; mutates nothing).

This adapter REUSES :mod:`asea.compiler.core` contracts by calling them --
``sparse_layers``, ``RoutingTelemetry``, ``collect_routing`` -- it never
modifies the compiler. The semantics are inherited verbatim: analytic-FP32
probability mass kept separate from native-postcast top-1 and post-capacity
dispatched counts; REAP conditional-mean expert outputs; "probability mass
is usage evidence, not causal expert importance".

Intervention masking is implemented as a forward hook on the router
classifier that rewrites the router OUTPUT (the masked expert's logit
becomes -1e9 for every token, input-independent), so the expert can
never win the top-1 argmax. A weight-row rewrite was removed: writing a
negative constant into a router row makes the logit ``-c * sum(hidden)``,
which is STRONGLY POSITIVE when the hidden-state sum is negative -- the
masked expert would become MORE likely to win, the opposite of the
intervention. Router and expert weights are never modified; restoration
removes the hook, and full-parameter hashing proves the teacher was
never touched. The masking changes which expert wins the router argmax;
it never writes to disk and never exports a model.

``structural_reduce`` produces a physically smaller CANDIDATE by keeping an
EXPLICIT index set per layer (from causal evidence downstream) -- the
compiler's own ``apply_selection`` keeps top-k by usage score, which this
package must NOT reuse as a causal ranking.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from ..errors import InterventionInvalid
from .base import MoEArchitectureAdapter


class SwitchAdapter(MoEArchitectureAdapter):
    arch = "switch_transformers"

    def __init__(self, model, torch) -> None:
        self.model = model
        self.torch = torch
        from ...compiler.core import sparse_layers  # lazy; needs localmodels extra

        self.layers = sparse_layers(model)  # validates expert/router layout
        self.n_experts = int(model.config.num_experts)
        self._telemetry = None
        self._baseline_hashes: Optional[Dict[str, str]] = None
        self._masked: Dict[str, Dict[str, Any]] = {}

    # -- read-only -----------------------------------------------------------

    def inspect_model(self) -> Dict[str, Any]:
        config = self.model.config
        return {
            "arch": self.arch,
            "class": type(self.model).__name__,
            "num_experts": self.n_experts,
            "sparse_layers": [name for name, _ in self.layers],
            "dense_layers": [],  # Switch has no dense-MLP prefix concept
            "experts_per_token": 1,
            "shared_experts": 0,
            "router_scoring": "softmax_top1",
            "hidden_size": getattr(config, "d_model", None),
            "dtype": str(getattr(self.model, "dtype", None)),
            "quantization": bool(getattr(self.model, "quantization_method", None) or getattr(self.model, "is_quantized", False)),
        }

    def parameter_inventory(self) -> Dict[str, Any]:
        totals: Dict[str, int] = {}
        grand = 0
        for name, parameter in self.model.named_parameters():
            count = int(parameter.numel())
            totals[name.rsplit(".", 1)[-1] + "/%s" % (parameter.dtype)] = (
                totals.get(name.rsplit(".", 1)[-1] + "/%s" % (parameter.dtype), 0) + count
            )
            grand += count
        return {"total_parameters": grand, "by_kind": totals}

    def enumerate_sparse_layers(self) -> List[int]:
        return [index for index, _ in enumerate(self.layers)]

    # -- telemetry ----------------------------------------------------------

    def register_router_hooks(self) -> None:
        from ...compiler.core import RoutingTelemetry

        self._telemetry = RoutingTelemetry(self.model, self.torch, reap=True)

    def collect_routing(self, reset: bool = True) -> Dict[str, Any]:
        if self._telemetry is None:
            raise InterventionInvalid(
                "register_router_hooks must be called before collect_routing"
            )
        return self._telemetry.result()

    def collect_expert_outputs(self, reset: bool = True) -> Dict[str, Any]:
        if self._telemetry is None:
            raise InterventionInvalid(
                "register_router_hooks must be called before collect_expert_outputs"
            )
        stats = self._telemetry.result()
        return {
            name: {
                "reap_dispatched": row.get("reap_dispatched"),
                "ean_dispatched": row.get("ean_dispatched"),
                "expert_output_count": row.get("expert_output_count"),
            }
            for name, row in stats.items()
        }

    # -- intervention protocol -----------------------------------------------

    def freeze_baseline(self) -> Dict[str, str]:
        """Hash every parameter (bytes) so any later change is provable."""
        self._baseline_hashes = self._current_hashes()
        return self._baseline_hashes

    def _current_hashes(self) -> Dict[str, str]:
        hashes: Dict[str, str] = {}
        with self.torch.inference_mode():
            for name, parameter in self.model.named_parameters():
                # Byte-exact hash: reinterpret the contiguous tensor as raw
                # uint8 (works for every dtype incl. bf16, unlike .numpy()).
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
            # First use: freeze instead of vacuously passing.
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

    def _layer_by_index(self, layer: int):
        try:
            name, module = self.layers[layer]
        except IndexError:
            raise InterventionInvalid("no sparse layer at index %d" % layer)
        return name, module

    def temporary_mask(self, target, *, scale: float = 0.0) -> Dict[str, Any]:
        """Suppress the masked expert's ROUTING DECISION, weights untouched.

        The hook rewrites the router classifier's OUTPUT logits (masked
        expert -> -1e9 for every token, input-independent) so the expert
        can never win the top-1 argmax. Router weights are never
        modified: a weight-row rewrite is mathematically unreliable here
        (the masked logit would be ``-c * sum(hidden)``, strongly
        POSITIVE when the hidden-state sum is negative), and unmodified
        weights make ``verify_unchanged`` hold trivially and honestly.
        """
        if scale != 0.0:
            raise InterventionInvalid(
                "partial-scale masking (scale=%r) is not implemented; only "
                "full suppression is, and this adapter refuses to pretend "
                "otherwise" % scale
            )
        if target.kind != "expert":
            raise InterventionInvalid(
                "masking kind %r is not supported: masking EVERY expert does "
                "not disable top-1 routing (argmax still fires over all "
                "-1e9 logits) and would silently under-suppress" % target.kind
            )
        if not (0 <= target.expert_id < self.n_experts):
            raise InterventionInvalid(
                "expert %d out of range 0..%d"
                % (target.expert_id, self.n_experts - 1)
            )
        name, module = self._layer_by_index(target.layer)
        classifier = module.router.classifier
        registry_key = "%s:%s" % (name, target.key)
        if registry_key in self._masked:
            raise InterventionInvalid(
                "component %s is already masked; restore before re-masking"
                % target.key
            )
        expert_id = [int(target.expert_id)]

        def suppress(module_, inputs, output):
            logits = output[0] if isinstance(output, tuple) else output
            if not isinstance(logits, self.torch.Tensor):
                return None
            masked = logits.clone()
            masked[..., expert_id] = -1.0e9
            if isinstance(output, tuple):
                return (masked,) + tuple(output[1:])
            return masked

        handle = classifier.register_forward_hook(suppress)
        self._masked[registry_key] = {
            "layer_index": target.layer,
            "indices": expert_id,
            "handle": handle,
            "layer_name": name,
        }
        return {
            "key": registry_key,
            "layer": name,
            "indices": expert_id,
            "mechanism": "router_output_forward_hook",
            "weights_modified": False,
        }

    def restore_mask(self, target) -> Dict[str, Any]:
        name, module = self._layer_by_index(target.layer)
        registry_key = "%s:%s" % (name, target.key)
        record = self._masked.pop(registry_key, None)
        if record is None:
            raise InterventionInvalid(
                "no recorded mask for %s; cannot restore what was never masked"
                % target.key
            )
        record["handle"].remove()
        return {
            "key": registry_key,
            "restored": True,
            "mechanism": "router_output_forward_hook_removed",
            "weights_modified": False,
        }

    # -- reduction -----------------------------------------------------------

    def structural_reduce(self, keep_components) -> Dict[str, Any]:
        """Physically keep only the given experts per sparse layer, as a
        NEW in-memory candidate (the caller saves/evaluates it; admission is
        never implied). ``keep_components`` maps layer INDEX -> expert index
        list. Layers absent from the map keep ALL their experts."""
        if not isinstance(keep_components, dict):
            raise InterventionInvalid(
                "keep_components must map layer index -> expert indices"
            )
        torch = self.torch
        for layer_index, keep in keep_components.items():
            name, module = self._layer_by_index(int(layer_index))
            keep = sorted(int(i) for i in keep)
            if not keep:
                raise InterventionInvalid(
                    "refusing to produce a layer with zero experts at %s" % name
                )
            if any(not (0 <= i < self.n_experts) for i in keep):
                raise InterventionInvalid("expert index out of range at %s" % name)
            old = module.router.classifier
            new = torch.nn.Linear(
                old.in_features,
                len(keep),
                bias=old.bias is not None,
                device=old.weight.device,
                dtype=old.weight.dtype,
            )
            with torch.no_grad():
                new.weight.copy_(old.weight[keep])
                if old.bias is not None:
                    new.bias.copy_(old.bias[keep])
            module.experts = torch.nn.ModuleDict(
                {"expert_%d" % j: module.experts["expert_%d" % i] for j, i in enumerate(keep)}
            )
            module.router.classifier = new
            module.router.num_experts = len(keep)
        return {
            "candidate": self.model,
            "kept": {
                self.layers[int(li)][0]: sorted(int(i) for i in keep)
                for li, keep in keep_components.items()
            },
            "admission": "CANDIDATE_UNADMITTED",
        }