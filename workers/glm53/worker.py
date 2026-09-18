"""Isolated GLM-5.3-Flash worker (runs INSIDE the Docker image; never in the
main SILT environment).

Owns the Transformers 5.x runtime (pinned to the model card's
``transformers_version`` = 5.16.0 -- see requirements.lock) so the main
SILT environment keeps its ``transformers==4.51.3`` pin untouched. Reads
versioned JSONL frames from stdin, writes validated JSONL frames to stdout;
diagnostics go to stderr only. The teacher weights at ``GLM_CHECKPOINT``
are opened READ-ONLY: the worker never writes into the checkpoint tree.

Hardware honesty (binding): before the model is touched, ``preflight``
admits or refuses on the same formula the compiler uses
(stored_parameters x dtype_bytes x 2 + 512 MiB vs available memory, plus
the cgroup ceiling). A ~320B-parameter BF16 teacher needs ~640 GiB before
headroom; a laptop is refused with the EXACT blocker and remedy, never a
swap-death or a fabricated trace. FP8/quantized checkpoints are refused
for interventions (unrestorable bit-identically) -- the BF16 repository
variant is the remedy.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

from asea.capability_build import worker_protocol as proto
from asea.capability_build.errors import BlockedResource, InterventionInvalid

_CHECKPOINT = os.environ.get("GLM_CHECKPOINT", "")
_HEADROOM_MIB = 512
_DTYPE_BYTES = 2  # BF16


def _stderr(*parts) -> None:
    print(*parts, file=sys.stderr, flush=True)


def _available_bytes() -> int:
    """Available memory = min(MemAvailable, cgroup ceiling) -- the compiler
    preflight discipline."""
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            mem_available = next(
                int(line.split()[1]) * 1024
                for line in handle
                if line.startswith("MemAvailable")
            )
    except (OSError, StopIteration):
        mem_available = 0
    for cgroup in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(cgroup, encoding="ascii") as handle:
                limit = int(handle.read().strip())
            if 0 < limit:
                mem_available = min(mem_available, limit) if mem_available else limit
        except (OSError, ValueError):
            continue
    return mem_available


def preflight(parameter_count: int) -> dict:
    """Memory admission BEFORE any weight is loaded."""
    required = parameter_count * _DTYPE_BYTES * 2 + _HEADROOM_MIB * 1024 * 1024
    available = _available_bytes()
    if available <= 0 or required > available:
        raise BlockedResource(
            requirement=(
                "GLM-5.3-Flash BF16 needs ~%.0f GiB resident "
                "(parameters %d x %d bytes x 2 + %d MiB headroom); "
                "this container sees %s bytes available"
                % (
                    required / (1024 ** 3),
                    parameter_count,
                    _DTYPE_BYTES,
                    _HEADROOM_MIB,
                    available,
                )
            ),
            remedy="run this worker on a GPU box / host with enough memory "
            "for the ~320B-parameter teacher, or use the behavioural "
            "(cloud) evidence class on small hosts",
        )
    return {"required_bytes": required, "available_bytes": available}


def main() -> int:
    if not _CHECKPOINT:
        print(json.dumps({
            "protocol": proto.PROTOCOL, "protocol_version": proto.PROTOCOL_VERSION,
            "op": "hello", "id": "boot", "ok": False,
            "error": {"kind": "blocked",
                      "requirement": "GLM_CHECKPOINT env var is not set",
                      "remedy": "mount the BF16 checkpoint and pass -e "
                                "GLM_CHECKPOINT=/checkpoint"},
        }), flush=True)
        return 2

    torch = None
    adapter = None

    def hello(payload: dict) -> dict:
        import transformers

        config = transformers.AutoConfig.from_pretrained(
            _CHECKPOINT, trust_remote_code=False
        )
        text = getattr(config, "text_config", None) or config
        # Config-arithmetic admission BEFORE any weights are touched.
        preflight_result = preflight(_config_parameter_estimate(text))
        return {
            "transformers": transformers.__version__,
            "torch": getattr(torch_module(), "__version__", None),
            "checkpoint": _CHECKPOINT,
            "model_type": getattr(config, "model_type", None),
            "preflight": preflight_result,
        }

    def torch_module():
        nonlocal torch
        if torch is None:
            import torch as _torch

            torch = _torch
        return torch

    def load_adapter() -> dict:
        nonlocal adapter
        if adapter is not None:
            return {"already_loaded": True}
        tm = torch_module()
        import transformers

        # Config-side preflight first: refuse before touching safetensors.
        config = transformers.AutoConfig.from_pretrained(
            _CHECKPOINT, trust_remote_code=False
        )
        text = getattr(config, "text_config", None) or config
        estimate = _config_parameter_estimate(text)
        preflight(estimate)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            _CHECKPOINT, torch_dtype=tm.bfloat16, device_map="cpu",
            low_cpu_mem_usage=True, trust_remote_code=False,
        )
        from asea.capability_build.adapters.glm53_flash import Glm53FlashAdapter

        adapter = Glm53FlashAdapter(model, tm, config=config)
        return {"loaded": True,
                "inspect": adapter.inspect_model(),
                "parameters": adapter.parameter_inventory()}

    def _config_parameter_estimate(text) -> int:
        """Config-arithmetic parameter count (no weights touched): a lower
        bound used only for the pre-admission refusal."""
        hidden = int(getattr(text, "hidden_size", 0))
        layers = int(getattr(text, "num_hidden_layers", 0))
        dense = int(getattr(text, "first_k_dense_replace", 0))
        routed = int(getattr(text, "n_routed_experts", 0))
        shared = int(getattr(text, "n_shared_experts", 0))
        expert_params = getattr(text, "moe_intermediate_size", None) or getattr(
            text, "intermediate_size", 0
        )
        per_expert = 3 * hidden * int(expert_params)  # gate/up/down
        moe = (routed + shared) * per_expert * max(0, layers - dense)
        dense_mlp = layers * 3 * hidden * int(getattr(text, "intermediate_size", 0))
        vocab = int(getattr(text, "vocab_size", 0))
        return moe + dense_mlp + vocab * hidden + 4 * layers * hidden * hidden

    def routing(payload: dict) -> dict:
        if adapter is None:
            load_adapter()
        prompts = payload.get("prompts")
        if not isinstance(prompts, list) or not prompts:
            raise InterventionInvalid("routing payload needs a non-empty 'prompts' list")
        max_length = int(payload.get("max_length", 2048))
        adapter.register_router_hooks()
        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained(_CHECKPOINT)
        per_prompt = []
        tm = torch_module()
        try:
            with tm.inference_mode():
                for item in prompts:
                    encoded = tokenizer(
                        item["prompt"], return_tensors="pt", truncation=True,
                        max_length=max_length,
                    )
                    adapter.model(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded.get("attention_mask"),
                        use_cache=False,
                    )
                    per_prompt.append({
                        "sample_id": item.get("sample_id"),
                        "group": item.get("group"),
                        "routing": adapter.collect_routing(reset=True),
                    })
        finally:
            if adapter._telemetry:
                for handle in adapter._telemetry:
                    handle.remove()
                adapter._telemetry = None
        return {
            "per_prompt": per_prompt,
            "usage_evidence_not_causal_importance": True,
        }

    def mask(payload: dict) -> dict:
        if adapter is None:
            load_adapter()
        target = payload["target"]
        if target.get("kind") not in ("expert", "layer"):
            raise InterventionInvalid("mask target kind must be expert or layer")
        from asea.capability_build.intervention import InterventionTarget

        component = InterventionTarget(
            target["kind"], int(target["layer"]),
            int(target.get("expert_id", -1)),
        )
        return adapter.temporary_mask(component)

    def restore(payload: dict) -> dict:
        if adapter is None:
            raise InterventionInvalid("nothing is masked; load first")
        target = payload["target"]
        from asea.capability_build.intervention import InterventionTarget

        component = InterventionTarget(
            target["kind"], int(target["layer"]),
            int(target.get("expert_id", -1)),
        )
        result = adapter.restore_mask(component)
        result.update(adapter.verify_unchanged())
        return result

    def verify(payload: dict) -> dict:
        if adapter is None:
            raise InterventionInvalid("model not loaded; nothing to verify")
        return adapter.verify_unchanged()

    handlers = {
        "hello": hello,
        "inspect": lambda payload: (
            load_adapter()
            if adapter is None
            else {"inspect": adapter.inspect_model(),
                  "parameters": adapter.parameter_inventory()}
        ),
        "routing": routing,
        "mask": mask,
        "restore": restore,
        "verify": verify,
        "shutdown": lambda payload: {"bye": True},
    }

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = proto.decode(line.strip().encode("utf-8"))
        except ValueError as exc:
            print(proto.encode({
                "protocol": proto.PROTOCOL,
                "protocol_version": proto.PROTOCOL_VERSION,
                "op": "unknown", "id": "unknown", "ok": False,
                "error": {"kind": "invalid_frame", "message": str(exc)},
            }).decode("utf-8"), flush=True)
            continue
        operation = request.get("op")
        if operation == "shutdown":
            print(proto.encode(proto.make_response(request, ok=True,
                                                   result={"bye": True})).decode("utf-8"),
                  flush=True)
            return 0
        try:
            handler = handlers.get(operation)
            if handler is None:
                raise InterventionInvalid("unknown op: %r" % operation)
            result = handler(request.get("payload") or {})
            response = proto.make_response(request, ok=True, result=result)
        except BlockedResource as exc:
            response = proto.blocked_response(request, exc.requirement, exc.remedy)
        except InterventionInvalid as exc:
            response = proto.make_response(request, ok=False, error={
                "kind": "arch_mismatch", "message": str(exc)})
        except Exception as exc:  # Never crash the loop on one bad request.
            _stderr(traceback.format_exc())
            response = proto.make_response(request, ok=False, error={
                "kind": "worker_error",
                "type": type(exc).__name__, "message": str(exc)})
        print(proto.encode(response).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())