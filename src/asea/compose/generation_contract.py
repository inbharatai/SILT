"""Versioned generation instructions and bounded, observational evidence (no ML imports).

Nothing here repairs model output or promises visibility into HF's internal cache.
"""
import contextlib
import contextvars
import hashlib
import json
import math

TRACE_SCHEMA = "generation_trace_v1"
MAX_TOKEN_IDS = 512
MAX_METADATA_BYTES = 8192
MAX_TRACE_BYTES = 16384
MAX_RUN_TRACE_BYTES = 131072
_TRACE_SINK = contextvars.ContextVar("generation_trace_sink", default=None)
_INSTRUCTIONS = {
    "raw_python": "Return only raw Python source code. Do not include Markdown fences or explanatory prose.",
    "fenced_python": "Return exactly one Markdown code block labeled python, containing only Python source code. Do not include prose outside the code block.",
}


def _contract(node):
    contract = getattr(node, "output_contract", None)
    return contract.model_dump(mode="json") if hasattr(contract, "model_dump") else contract


def preflight_node(node, declarations=()):
    """Raise ValueError before loading for unsupported/conflicting structured settings.

    declarations is an iterable of explicit format strings from other structured
    sources (e.g. suite/gate); natural-language task text must NOT be passed here.
    """
    contract = _contract(node)
    formats = list(declarations)
    if contract is not None:
        if set(contract) != {"schema_version", "format"} or type(contract["schema_version"]) is not int or contract["schema_version"] != 1 or contract["format"] not in _INSTRUCTIONS:
            raise ValueError("unsupported output contract")
        if node.component.kind != "hf_text":
            raise ValueError("output_contract supports only hf_text (not speech, vision or fixtures)")
        if getattr(node, "prompt_prefix", ""):
            raise ValueError("strict output_contract cannot combine with nonempty legacy prompt_prefix; use migration_preview")
        formats.append(contract["format"])
    if any(item not in _INSTRUCTIONS for item in formats) or len(set(formats)) > 1:
        raise ValueError("conflicting or unsupported structured output format declarations")
    policy = getattr(node, "cache_policy", None)
    if policy not in (None, "model_default", "enabled", "disabled"):
        raise ValueError("unsupported cache_policy")
    if policy is not None and node.component.kind not in {"hf_text", "hf_asr", "hf_vision"}:
        raise ValueError("cache_policy is not supported by this adapter")
    # The validated vision path only supports forced false; omission may enable
    # unverified model-default caching, so model_default is blocked too.
    if node.component.kind == "hf_vision" and policy in {"enabled", "model_default"}:
        raise ValueError("vision cache policy is unverified; only disabled is supported")


def render_prompt(node, task_text):
    """One renderer, before the existing chat template; task bytes untouched."""
    preflight_node(node)
    contract = _contract(node)
    if contract is None:
        return getattr(node, "prompt_prefix", "") + task_text
    return _INSTRUCTIONS[contract["format"]] + "\n\n" + task_text


def migration_preview(node, output_format):
    """Read-only preview; never edits prefix/spec/task or repairs fences.

    A nonempty prefix requires a human to decide what task prose to retain.
    The caller must construct and validate a new Node after that review.
    """
    if output_format not in _INSTRUCTIONS:
        raise ValueError("unsupported output format")
    prefix = getattr(node, "prompt_prefix", "")
    return {"schema_version": 1, "read_only": True,
            "supported_adapter": node.component.kind == "hf_text",
            "proposed_output_contract": {"schema_version": 1, "format": output_format},
            "instruction": _INSTRUCTIONS[output_format], "existing_prompt_prefix": prefix,
            "requires_prefix_review": bool(prefix), "applied": False}


def cache_kwargs(node):
    preflight_node(node)
    policy = getattr(node, "cache_policy", None)
    if policy == "model_default":
        return {}
    if policy is not None:
        return {"use_cache": policy == "enabled"}
    if node.component.kind == "hf_text":
        return {"use_cache": getattr(node, "use_cache", False)}
    if node.component.kind == "hf_vision":
        return {"use_cache": False}
    return {}


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def new_trace(node, value):
    binding = {key: value[key] for key in ("type", "sha256", "size", "sample_rate", "duration_seconds") if key in value}
    if "text" in value:
        payload = value["text"].encode("utf-8")
        binding.update(text_sha256=hashlib.sha256(payload).hexdigest(), text_bytes=len(payload))
    return {"schema": TRACE_SCHEMA, "node": node.id, "status": "started", "stage": "preflight",
            "requested": {"output_contract": _contract(node), "cache_policy": getattr(node, "cache_policy", None),
                          "legacy_use_cache": getattr(node, "use_cache", False), "dtype": node.dtype,
                          "max_new_tokens": node.max_new_tokens},
            "input_binding": binding, "forwarded": {}, "resolved": {},
            "input_token_count": None, "output_token_count": None, "generated_token_count": None,
            "generated_token_ids": None, "eos_positions": None, "cap_reached": None,
            "stop_reason": "unknown"}


# Closed labels: never read exception class attributes (including metaclass
# descriptors), stringify exceptions, or inspect arbitrary exception properties.
_EXCEPTION_TYPES = ((BaseException, "BaseException"), (Exception, "Exception"),
                    (RuntimeError, "RuntimeError"), (ValueError, "ValueError"),
                    (TypeError, "TypeError"), (KeyError, "KeyError"),
                    (OSError, "OSError"), (MemoryError, "MemoryError"),
                    (KeyboardInterrupt, "KeyboardInterrupt"), (SystemExit, "SystemExit"))


def exception_type(exc):
    cls = type(exc)
    return next((label for known, label in _EXCEPTION_TYPES if cls is known), "opaque")


def exception_is(exc, classes):
    # Invoke type's native MRO descriptor directly, not metaclass attribute lookup
    # or isinstance's fallback to an instance's possibly hostile __class__ property.
    mro = type.__dict__["__mro__"].__get__(type(exc))
    return any(base is known for base in mro for known in classes)


def failure_metadata(exc):
    return {"exception_type": exception_type(exc),
            "message_availability": "not_collected_no_exception_hooks"}


def primary_error(exc, trusted_type):
    """Preserve common legacy errors only when formatting cannot invoke user code.

    Arbitrary exception types/arguments use a declared fallback, not a fabricated
    message or hash. Even BaseException.__str__ may invoke an argument's repr.
    """
    cls = type(exc)
    safe = (BaseException, Exception, RuntimeError, ValueError, TypeError,
            MemoryError, KeyboardInterrupt, SystemExit, trusted_type)
    if any(cls is known for known in safe):
        args = BaseException.__dict__["args"].__get__(exc)
        if type(args) is tuple and (not args or (len(args) == 1 and type(args[0]) is str)):
            label = "Blocked" if cls is trusted_type else exception_type(exc)
            return (args[0] if args and args[0] else label), False
    return "exception message unavailable (unsafe formatting omitted)", True


def observe(trace, operation, *args, **kwargs):
    """An observation failure must not replace an inference result/exception."""
    if trace is None:
        return
    try:
        operation(trace, *args, **kwargs)
    except BaseException as exc:
        try:
            trace["observation_error"] = exception_type(exc)
        except BaseException:
            pass  # Diagnostics must not replace either results or primary errors.


def _small(value):
    if type(value) is float and not math.isfinite(value):
        return {"unrecorded_type": "nonfinite_float"}
    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) in (list, tuple):
        if len(value) <= 32:
            return [_small(v) for v in value]
        if all(type(v) is int for v in value):
            return {"values": list(value[:32]), "total": len(value), "truncated": True,
                    "sha256": _hash(value), "hash_basis": "ordered_integer_list_json"}
        return {"unrecorded_type": "sequence", "total": len(value), "truncated": True}
    return {"unrecorded_type": "opaque"}


def observe_call(trace, model, inputs, kwargs, node):
    trace["stage"] = "generation"
    trace["forwarded"] = {"generate_kwargs": {k: _small(v) for k, v in kwargs.items()},
                          "input_tensors": tensor_metadata(inputs)}
    if "input_ids" in inputs:
        trace["input_token_count"] = int(inputs["input_ids"].shape[-1])
    flags = ("use_cache", "eos_token_id", "pad_token_id", "bos_token_id", "decoder_start_token_id",
             "do_sample", "num_beams", "max_length", "max_new_tokens", "forced_eos_token_id",
             "return_dict_in_generate", "output_scores", "output_logits")
    resolved = {}
    for name in ("config", "generation_config"):
        config = getattr(model, name, None)
        resolved[name] = {key: _small(getattr(config, key)) for key in flags if hasattr(config, key)}
        nested = getattr(config, "text_config", None)
        if nested is not None:
            resolved[name]["text_config"] = {key: _small(getattr(nested, key)) for key in flags if hasattr(nested, key)}
    resolved["cache_internal_state"] = "not_observed"
    trace["resolved"] = resolved
    trace["call_metadata_sha256"] = _hash({"forwarded": trace["forwarded"], "resolved": resolved})


def tensor_metadata(inputs):
    # Read shape/dtype only; never copy full tensors, logits or scores.
    return {str(key)[:80]: {"shape": [int(n) for n in value.shape[:8]], "dtype": str(value.dtype)[:80]}
            for key, value in list(inputs.items())[:32] if hasattr(value, "shape") and hasattr(value, "dtype")}


def observe_tokens(trace, generated, model, node, input_count=0):
    """Read actual returned IDs, never tokenize decoded text.

    Encoder-decoder sequences include a decoder prefix: only subtract a verified
    one-token decoder-start prefix. Unknown/custom prefixes => unknown new count.
    ASR uses custom generation; stop_reason remains unknown even at an EOS/cap.
    """
    sequence = getattr(generated, "sequences", generated)[0]
    count = int(sequence.shape[-1])
    causal = node.component.kind in {"hf_text", "hf_vision"} and node.component.task != "seq2seq"
    if causal and count < input_count:
        raise ValueError("returned sequence is shorter than the observed input; new token count unknown")
    start = input_count if causal else 0
    new_count_known = causal
    gc = getattr(model, "generation_config", None)
    config = getattr(model, "config", None)
    decoder_start = getattr(gc, "decoder_start_token_id", None)
    if decoder_start is None:
        decoder_start = getattr(config, "decoder_start_token_id", None)
    if not causal and node.component.kind == "hf_text" and count and decoder_start is not None and int(sequence[0]) == decoder_start:
        start, new_count_known = 1, True
    ids = [int(v) for v in sequence[start:start + MAX_TOKEN_IDS].tolist()]
    eos = getattr(gc, "eos_token_id", None)
    eos_source = "model.generation_config.eos_token_id"
    if eos is None:
        eos = getattr(config, "eos_token_id", None)
        eos_source = "model.config.eos_token_id"
    if eos is None:
        eos_source = "not_configured"
        eos = []
    elif type(eos) is int:
        eos = [eos]
    elif type(eos) not in (list, tuple) or not all(type(v) is int for v in eos):
        raise ValueError("EOS configuration is not an observable integer or integer sequence")
    # Snapshot the complete declaration before classification; never modify the
    # model configuration or infer HF's private effective stopping configuration.
    eos = list(eos)
    eos_hash = _hash(eos)
    trace.update(output_token_count=count - (input_count if causal else 0),
                 generated_token_count=count - start if new_count_known else None,
                 generated_token_ids=ids if new_count_known else None,
                 returned_sequence_token_ids=ids if not new_count_known else None,
                 token_ids_truncated=count - start > MAX_TOKEN_IDS,
                 eos_token_ids=eos[:32], eos_token_ids_total=len(eos), eos_token_ids_truncated=len(eos) > 32,
                 eos_configuration={"source": eos_source, "sha256": eos_hash,
                                    "hash_basis": "ordered_integer_list_json",
                                    "observation": "post_generation_model_configuration",
                                    "effective_stopping_configuration": "not_verified"},
                 eos_positions=[i for i, token in enumerate(ids) if token in eos],
                 matched_eos_token_ids=sorted({token for token in ids if token in eos}),
                 eos_positions_basis="generated_token_ids" if new_count_known else "returned_sequence_token_ids",
                 cap_reached=(count - start >= node.max_new_tokens) if new_count_known else None,
                 stop_reason="unknown")
    trace["stage"] = "decode"


def observe_prompt(trace, prompt, templated):
    payload = prompt.encode("utf-8")
    trace["rendered_prompt"] = {"sha256": hashlib.sha256(payload).hexdigest(),
                                "utf8_bytes": len(payload), "chat_template_applied": bool(templated)}


def observe_media(trace, **facts):
    trace.setdefault("media", {}).update(facts)


def bounded_trace(trace):
    """Hard per-node bound; hash omitted metadata rather than silently truncate it."""
    result = dict(trace)
    metadata = {key: result.get(key) for key in ("forwarded", "resolved", "media")}
    if len(_bytes(metadata)) > MAX_METADATA_BYTES:
        for key in metadata:
            result.pop(key, None)
        result["metadata_omitted"] = {"reason": "byte_limit", "sha256": _hash(metadata)}
    if len(_bytes(result)) > MAX_TRACE_BYTES:
        result = {"schema": TRACE_SCHEMA, "node": trace["node"], "status": trace["status"],
                  "stage": trace["stage"], "evidence_omitted": "node_byte_limit", "sha256": _hash(trace)}
    return result


@contextlib.contextmanager
def capture_traces(sink):
    """Scope a list sink without changing the historic five-argument adapter API."""
    token = _TRACE_SINK.set(sink)
    try:
        yield sink
    finally:
        _TRACE_SINK.reset(token)


def append_trace(trace):
    sink = _TRACE_SINK.get()
    if sink is not None:
        result = bounded_trace(trace)
        # Reserve a tiny marker for every possible node (graph maximum = 16).
        if len(_bytes(sink + [result])) > MAX_RUN_TRACE_BYTES - 4096:
            result = {"schema": TRACE_SCHEMA, "node": trace["node"], "status": trace["status"],
                      "stage": trace["stage"], "evidence_omitted": "run_byte_limit", "sha256": _hash(trace)}
        if len(_bytes(sink + [result])) <= MAX_RUN_TRACE_BYTES:
            sink.append(result)
