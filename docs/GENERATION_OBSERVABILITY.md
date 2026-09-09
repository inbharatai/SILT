# Generation output contracts and bounded run observations

This scoped change adds opt-in machine-format instructions for **new local HF text runs**, explicit adapter cache policy, and bounded observations on new runs. It does not modify model weights, model generation defaults, consumed runs, old records, evaluation gates, certificates, CLI or Studio. Unit spies/native CPU tensors are not pretrained quality or performance evidence.

## New Node fields

```json
{
  "output_contract": {"schema_version": 1, "format": "raw_python"},
  "cache_policy": "disabled"
}
```

- `Node.output_contract: OutputContract | None = None`: schema version integer `1`; closed `format` enum `raw_python | fenced_python`. Explicit contracts are supported only by `hf_text` (`causal` or `seq2seq`). ASR, Vits, vision and fixtures reject them before model loading.
- `Node.cache_policy: Literal['model_default', 'enabled', 'disabled'] | None = None`: distinct from the existing `use_cache` field. An explicit policy governs forwarding; the old flag remains serialized and recorded, not deleted or rewritten.
- No explicit contract means **exactly** the old `prompt_prefix + task_text`, followed by the same existing chat-template/tokenizer path.
- An explicit contract requires an empty legacy `prompt_prefix`. Even whitespace-only prefixes are nonempty. There is no heuristic extraction, auto-removal of prefix prose, fence repair or task rewriting.
- The contract requests a format; it does **not** certify that the response follows it. Gate/oracle integration must independently inspect the unchanged response.

`generation_contract.render_prompt(node, task_text)` is the sole renderer of contract instructions. It prepends the selected fixed instruction and two newline characters to the task text, without changing any task characters. The runtime subsequently applies its existing chat template, if present. Natural-language mentions of competing formats are **not** interpreted as structured declarations.

## Preflight and read-only migration API (parent integration)

Import from `asea.compose.generation_contract`:

- `preflight_node(node, declarations=()) -> None`: validates adapter support, prefix conflict, and cache support. `declarations` is an iterable of **explicit structured format strings** from other sources, e.g. suite/gate configuration. Unsupported strings or competing formats raise `ValueError`. Parent CLI/gate code must call this with its other structured declarations **before registering/loading models**; do not pass task prose. Node validation invokes it automatically with no external declarations. Runtime preparation and direct adapter invocation recheck it, including model-copy/construct bypasses; runtime converts its errors to `Blocked`.
- `render_prompt(node, task_text) -> str`: validates and renders, without mutating the node/task. Do not additionally render the same instruction in CLI/UI.
- `migration_preview(node, output_format) -> dict`: returns `schema_version`, `read_only: true`, `supported_adapter`, `proposed_output_contract`, `instruction`, `existing_prompt_prefix`, `requires_prefix_review`, `applied: false`. It never returns an automatically rewritten task or applies a change. A human must decide which prefix prose to retain as task content, then construct and validate a **new** Node/spec. Do not use `model_copy(update=...)` as validation.
- `cache_kwargs(node) -> dict`: the exact adapter cache keyword override/omission described below, after preflight.

No migration or old-record backfill is performed.

## Adapter-specific cache support

| Adapter | Legacy `cache_policy=None` | `model_default` | `enabled` | `disabled` |
|---|---|---|---|---|
| HF text causal/seq2seq | Forward old `use_cache` (default false) | Omit `use_cache` | Forward true | Forward false |
| HF ASR Whisper | Omit `use_cache`, regardless of old flag | Omit `use_cache` | Forward true | Forward false |
| HF vision SmolVLM | Force false, regardless of old flag | Blocked: unverified model-default path | Blocked: unverified enabled path | Forward false |
| Vits TTS / fixture | Not applicable; omit | Blocked | Blocked | Blocked |

Omission means **defer to model configuration**, not an assertion that internal caching was enabled/disabled. Observations distinguish:

1. `requested`: explicit contract, cache policy, legacy flag, requested dtype and token cap.
2. `forwarded.generate_kwargs`: scalar/list metadata of actual explicit generation keyword arguments; omission stays omission. `forwarded.input_tensors` records shapes/dtypes of actual tensor kwargs. Vits instead records `model_call_kwargs`, never a fictitious generate call.
3. `resolved.config`, `resolved.generation_config`, and nested `text_config` where present: observed model configuration default flags at call time. These are not hidden internal execution state. `resolved.cache_internal_state` is always `not_observed`.

Existing max-new-token cap (1–512), dtypes, deterministic `do_sample=False`, `num_beams=1`, pad fallback, seeds, resampling, resizing/processor behavior and loading kwargs are preserved. No logits/scores or return-dictionary mode is requested for observation. Hooks do not generate a second response or change RNG state.

## Canonical serialization / fresh graph binding

Import `dump_spec(value)` from `asea.compose.schema`; it accepts a spec or Node and returns the JSON-compatible canonical model dump. `Node` also has a compatible Pydantic wrap serializer, so nested `model_dump()`, `model_dump(mode='json')` and `model_dump_json()` callers inherit the compatibility rule:

- Omit **only the two new fields when None** (including explicit null).
- Retain every existing legacy field/default/null under ordinary dumping.
- Serialize and hash all new non-null settings, including explicit `model_default` or `disabled` even when its immediate behavior matches a legacy path.

Do not replace this with blanket `exclude_none`, `exclude_unset` or `exclude_defaults`: those would remove old bound fields. Runtime candidate/config/graph preparation uses `dump_spec`; old unset graphs retain old canonical serialization and runtime version. New explicit settings create different graph/config bindings. New trace schema markers are independent of the unchanged legacy runtime version. This change does not re-sign or reconstruct old records.

## New-run evidence API and schema

New `run_spec` results have:

```json
{
  "generation_trace_schema": "generation_trace_v1",
  "generation_trace_v1": [
    {"schema": "generation_trace_v1", "node": "node_id", "status": "succeeded", "stage": "complete"}
  ]
}
```

The trace list contains bounded **per-node** evidence, including a failing adapter's entry. Existing `node_results`, output text, waveform generation and output digest semantics remain unchanged. No trace is added inside the adapter's old return object. Old records lacking this marker mean **not observed**, not zero tokens/no cache.

`capture_traces(sink)` is a context manager for direct adapter unit/integration use:

```python
traces = []
try:
    with capture_traces(traces):
        result = runtime._adapter(node, value, output_path, limits, allow_fixtures)
except Exception:
    # traces already contains the failing adapter's evidence; do not invent counts.
    raise
```

The historic five-argument `_adapter` API is unchanged. `new_trace(node, value)`, `observe(trace, operation, ...)`, `bounded_trace(trace)` and `append_trace(trace)` are helper APIs; the runtime creates/appends traces automatically. Do not append twice. `capture_traces` uses a context-local sink, not a process-global mutable record.

### Token facts

- `input_token_count`: actual tokenized input length when an `input_ids` tensor exists; null for feature-only audio inputs.
- `output_token_count`: returned output sequence length after removing a known causal input prefix; encoder-decoder counts include its returned decoder prefix.
- `generated_token_count`, `generated_token_ids`: actual returned continuation IDs and count for causal/vision. For standard text seq2seq, a one-token decoder-start prefix is removed **only when its ID matches the observed model default**; otherwise the generated count/IDs remain null.
- `returned_sequence_token_ids`: actual returned IDs when a reliable continuation boundary is unknown, notably custom Whisper generation. These may include decoder/language/task prefix tokens; do not report their length as new generated tokens.
- `token_ids_truncated`: whether more actual IDs existed than the retention cap. Retained IDs are at most 512; counts come from the tensor shape, not retained-list length or retokenized response text.
- `eos_token_ids`, `eos_positions`, `eos_positions_basis`: observed configuration EOS IDs and their zero-based locations in the retained ID sequence. Positions beyond the retained cap are not available.
- `cap_reached`: observed continuation count >= requested cap when the boundary is known; null otherwise. Reaching the cap does not prove length was the stopping cause.
- `stop_reason`: conservatively `unknown` in v1. Returned tokens/config alone do not prove a unique stopping cause, especially EOS-at-cap or custom generation. No internal HF stop-event claim is made.

The original decode path is untouched. In particular, seq2seq decoding still receives the original returned full sequence, and malformed fences are returned unmodified.

### Failure facts

`status` is `failed`, `blocked` or `cancelled`; `stage` is the last attempted phase (`preflight`, `loading`, `generation`, `decode`, or `complete`). `failure` stores bounded `exception_type` and a SHA-256 of the exception message, not an arbitrarily large/sensitive traceback. If generation throws before returning IDs, all output count/ID/EOS/cap fields remain **null**, while already observed input/config facts remain valid. If decoding fails after a generated tensor was returned, its actual token facts may legitimately be present. Observation failures are labeled with `observation_error` and do not replace a model result/exception. Hard process kills and failures before entering an adapter cannot fabricate an adapter trace; existing pending-run recovery remains unchanged.

### Media and bindings

- `input_binding`: input text UTF-8 hash/byte count and/or original media file hash/byte count, with available rate/duration metadata. Raw tasks are not duplicated in traces.
- `rendered_prompt`: actual post-template prompt UTF-8 hash/byte count and whether chat templating was applied; no full prompt copy.
- `config_files` and `config_metadata_sha256`: observed hashes/sizes of bounded known local configuration files (`config.json`, generation/tokenizer/preprocessor/processor config where present), never weights. `call_metadata_sha256` binds forwarded kwargs/tensor metadata and observed model defaults.
- WAV observations record actual decoded PCM byte hash/count, channel count, input sample count/rate/duration, and processed mono/resampled sample count/rate/duration. Original input binding and source-file metadata are distinguished from actually decoded PCM bytes; no full waveform is copied for telemetry.
- Vits records actual returned waveform sample count/rate/duration even though output-format contracts/cache policies are unsupported. There are no fake generated text-token counts.
- Processor outputs record only shapes/dtypes (bounded 32 entries, up to 8 dimensions), never full tensors. Vision records actual loaded image dimensions. When the native 4-D `pixel_attention_mask` exists, `observed_nonempty_image_tile_count` counts nonempty tile masks (padded empty slots excluded); otherwise tile count is absent, not guessed from model name or image dimensions.

### Hard bounds

- `MAX_TOKEN_IDS = 512`.
- `MAX_METADATA_BYTES = 8192` for forwarded/resolved/media metadata combined per node. Oversized metadata is omitted with a reason and digest.
- `MAX_TRACE_BYTES = 16384` per node; oversized entries fall back to a small omission marker with node/status/stage/digest.
- `MAX_RUN_TRACE_BYTES = 131072` for the compact-JSON trace list. Space is reserved for omission markers for all 16 possible graph nodes. Run-limit omissions remain labeled, not silently treated as complete traces.

These limits are fixed runtime safety bounds, not oracle retention settings. Parent-owned CLI/gate/UI can add retention controls separately. Studio/export should display omission/unknown markers and must not expand tensors, infer counts from text, or promise unobserved HF cache internals.

## Focused local validation

Run from the copied project root, without downloads:

```sh
PYTHONPATH=/agent/workspace/silt-observability/src /agent/workspace/silt-venv/bin/python -m pytest tests/test_generation_observability.py tests/test_vision_runtime.py tests/test_composition_v1.py -q
```

Tests cover legacy nested serialization/fingerprints, explicit-field binding, read-only migration/conflicts, adapter support/preload rejection, actual scalar kwargs and tensor shapes, unchanged malformed output, failure telemetry, EOS/cap ambiguity, seq2seq prefix accounting, audio resampling facts, Vits waveform facts, vision nonempty tiles, byte limits and observation-on/off spy parity. Existing vision tests use tiny native loading fixtures only. No real pretrained or final evaluation runs are performed or consumed.

### Scoped implementation validation log

- Focused command above: **95 passed in 5.01s** (48 new generation tests plus 47 existing vision/composition tests).
- Broader regression command additionally included `test_crash_recovery.py`, `test_certification_hardening.py`, and `test_integration_contracts.py`: **203 passed, 1 failed**. The failed existing `test_real_sandbox_metric_integration` recorded `stale_implementation_fingerprint: implementation or environment changed; reevaluate` while other scoped agents were editing this shared copy. Its isolated rerun passed (1 passed in 5.63s). Do not call the combined regression green; rerun after all parallel edits are complete.
- No certificate/gate/CLI source was edited to suppress the fingerprint guard.

### Parent integration checklist

1. Add `compose/generation_contract.py` to any explicit **implementation-fingerprint file inventory** (certificate code is outside this task's ownership). Existing inventories include schema/runtime but did not yet include this new behavior-bearing module at validation time.
2. Use `dump_spec` or the compatible ordinary Pydantic dump, preserving every old bound field. Bind every new explicit setting in fresh candidates; never backfill old records.
3. Route structured format declarations through `preflight_node` before loading; call `migration_preview` for a read-only human review, not an auto-edit. Keep the runtime as the sole instruction renderer.
4. Surface the per-node `generation_trace_v1` entries, failure/null fields, and omission markers. Absence in old records is not a count or cache-state claim. Apply parent-owned oracle retention policy separately.
5. Rerun integration/fingerprint tests after the shared copy stops changing.
