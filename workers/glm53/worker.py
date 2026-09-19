"""Isolated GLM-5.3-Flash worker (runs INSIDE the Docker image; never in the
main SILT environment).

Owns the Transformers 5.x runtime pinned to **5.16.1** (see requirements.lock
-- the GLM-5.3-Flash model card declares ``transformers_version`` 5.16.0,
but the 5.16.0 wheel does NOT ship ``models/glm5_next`` (verified against
the upstream tag: the directory exists at v5.16.1 and 404s at v5.16.0), so
5.16.1 is the minimal release that actually contains the architecture; the
worker records both the declared and the installed version in ``hello``
rather than pretending the pin matches the card). The main SILT
environment keeps its ``transformers==4.51.3`` pin untouched. Reads
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

import hashlib
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
    # Frames are UTF-8 JSONL by protocol contract, but the interpreter's
    # default stdio encoding follows the container locale (a C-locale or
    # cp1252 image would make a non-ASCII prompt fail mid-frame with
    # UnicodeEncodeError -- a protocol fault that looks like a crash).
    # Reconfigure BEFORE the loop touches a single frame.
    for stream in (sys.stdin, sys.stdout):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError, OSError):
            # Pre-3.7 or unsupported stream: the Docker image pins a modern
            # Python, so this never fires in practice; the encode/decode
            # layer below still validates every frame.
            pass

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
            "transformers_declared_by_checkpoint":
                getattr(config, "transformers_version", None),
            "transformers_pin_note": (
                "installed %s; the model card declares %s -- 5.16.0 does not "
                "ship models/glm5_next, 5.16.1 is the minimal release that does"
                % (transformers.__version__,
                   getattr(config, "transformers_version", None))
            ),
            "torch": getattr(torch_module(), "__version__", None),
            "checkpoint": _CHECKPOINT,
            "checkpoint_identity": _checkpoint_identity(),
            "model_type": getattr(config, "model_type", None),
            "preflight": preflight_result,
        }

    def _checkpoint_identity() -> dict:
        """sha256 of the checkpoint's structural files (config.json,
        generation_config.json, model.safetensors.index.json), recorded in
        hello/inspect so every downstream artifact can state EXACTLY which
        checkpoint revision produced it. Weight shards are NOT hashed here
        (~320B params: the reader would run for tens of minutes); the
        identity pins the manifest that defines the shard set + digests."""
        import os as _os

        identity = {}
        for name in ("config.json", "generation_config.json",
                     "model.safetensors.index.json"):
            path = _os.path.join(_CHECKPOINT, name)
            try:
                with open(path, "rb") as handle:
                    identity[name] = hashlib.sha256(handle.read()).hexdigest()
            except OSError as exc:
                identity[name] = None
                _stderr("checkpoint identity: cannot read %s: %s" % (path, exc))
        return identity

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

        # ARCH-PRESENCE CHECK before any config/weights are touched: this
        # build must refuse on a transformers that lacks glm5_next rather
        # than failing cryptically inside from_pretrained (audit 2026-09-19).
        try:
            from transformers.models.glm5_next import modeling_glm5_next  # noqa: F401
        except ImportError as exc:
            raise InterventionInvalid(
                "this transformers %s does not contain models/glm5_next "
                "(import failed: %s); GLM-5.3-Flash needs transformers "
                ">= 5.16.1 -- pin the worker's requirements.lock, do not "
                "guess at a partial install" % (transformers.__version__, exc)
            )
        from transformers.models.auto import modeling_auto

        mapping = getattr(modeling_auto,
                           "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES", None)
        if mapping is None or "glm5_next" not in mapping:
            raise InterventionInvalid(
                "transformers %s ships models/glm5_next but does not map it "
                "onto AutoModelForMultimodalLM; the worker refuses to load "
                "through a class this revision does not support"
                % transformers.__version__
            )

        # Config-side preflight first: refuse before touching safetensors.
        config = transformers.AutoConfig.from_pretrained(
            _CHECKPOINT, trust_remote_code=False
        )
        text = getattr(config, "text_config", None) or config
        estimate = _config_parameter_estimate(text)
        preflight(estimate)
        # GLM-5.3-Flash is a vision-language model: the class the real
        # checkpoint loads as is Glm5NextForConditionalGeneration, reached
        # via AutoModelForMultimodalLM (there is no Glm5NextForCausalLM in
        # transformers 5.16.1; AutoModelForCausalLM would either fail or
        # silently pick a wrong mapping -- audit 2026-09-19). device_map is
        # accelerate-backed, hence the accelerate pin in requirements.lock.
        model = transformers.AutoModelForMultimodalLM.from_pretrained(
            _CHECKPOINT, torch_dtype=tm.bfloat16, device_map="cpu",
            low_cpu_mem_usage=True, trust_remote_code=False,
        )
        from asea.capability_build.adapters.glm53_flash import Glm53FlashAdapter

        adapter = Glm53FlashAdapter(model, tm, config=config)
        return {"loaded": True,
                "checkpoint_identity": _checkpoint_identity(),
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
            # Public adapter API, not private internals: a routing pass that
            # raised must never leave live telemetry hooks on a teacher whose
            # parameter hashes still verify clean.
            adapter.remove_router_hooks()
        return {
            "per_prompt": per_prompt,
            "usage_evidence_not_causal_importance": True,
        }

    def generate(payload: dict) -> dict:
        """Generate completions under whatever masks are currently active --
        the measurement primitive of the causal-intervention protocol
        (mask -> generate -> judge -> restore -> hash-verify). The host
        judges the returned completions through the REAL host oracle; this
        op only generates and reports the live mask set so the caller can
        refuse to attribute an outcome to a masked teacher whose masks
        were not the ones it believes (audit 2026-09-19, item 8)."""
        if adapter is None:
            load_adapter()
        prompts = payload.get("prompts")
        if not isinstance(prompts, list) or not prompts:
            raise InterventionInvalid("generate payload needs a non-empty 'prompts' list")
        max_new_tokens = int(payload.get("max_new_tokens", 256))
        if max_new_tokens <= 0:
            raise InterventionInvalid("max_new_tokens must be positive")
        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained(_CHECKPOINT)
        tm = torch_module()
        per_prompt = []
        with tm.inference_mode():
            for item in prompts:
                prompt = item.get("prompt")
                if not isinstance(prompt, str) or not prompt:
                    raise InterventionInvalid(
                        "each generate prompt needs a non-empty 'prompt' string"
                    )
                encoded = tokenizer(prompt, return_tensors="pt",
                                    truncation=True)
                output = adapter.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
                new_tokens = int(
                    output.shape[-1] - encoded["input_ids"].shape[-1]
                )
                text = tokenizer.decode(
                    output[0, encoded["input_ids"].shape[-1]:],
                    skip_special_tokens=True,
                )
                per_prompt.append({
                    "sample_id": item.get("sample_id"),
                    "group": item.get("group"),
                    "completion": text,
                    "new_tokens": new_tokens,
                })
        return {
            "per_prompt": per_prompt,
            # The mask/restore registry is deliberately not touched here: a
            # mid-batch exception leaves active masks exactly as they were,
            # reported here, so the caller never confuses a torn batch with
            # a clean one.
            "active_masks": adapter.active_masks(),
            "generation": {"greedy": True, "max_new_tokens": max_new_tokens},
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
            else {"checkpoint_identity": _checkpoint_identity(),
                  "inspect": adapter.inspect_model(),
                  "parameters": adapter.parameter_inventory()}
        ),
        "routing": routing,
        "generate": generate,
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
                # The protocol defines its own error kind for this: an
                # unknown op is an invalid_op, not an arch_mismatch -- kind
                # mapping that lies makes the client's diagnostics lie too
                # (audit 2026-09-18).
                response = proto.make_response(request, ok=False, error={
                    "kind": "invalid_op",
                    "message": "unknown op: %r (known: %s)"
                               % (operation, ", ".join(sorted(handlers))),
                })
            else:
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