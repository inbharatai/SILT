# Function oracle observability v1

This is additive **evidence**, not a new correctness/admission oracle. The host
still validates structured response frames and performs exact-type `_equal`
comparisons against host-only expectations. `cases` remains exactly
`[{"id": ..., "matched": true|false}]`; candidate verdicts/counters never control
it. All candidate errors, malformed frames, incomplete executions and abnormal
exits remain failures. Full retained values are actual validated response values,
including wrong outputs and objects containing misleading `passed` fields.

## API and privacy

```python
from asea.certification.function_oracle import evaluate_functions, replay_returns

# Existing positional/keyword calls still work. Default retains no return content.
r = evaluate_functions(source, cases, limits=None)
r = evaluate_functions(source, cases, trace_policy={
    "schema_version": 1, "return_retention": "value"})
# Expectations supplied independently by an authorized host caller, never in trace.
replay = replay_returns(r["return_trace"], cases)
```

`trace_policy` is an exact two-key dict; `None` means version 1 / `none`.
- `none`: per-observed-return identity/phase and `reason: policy_none`; no value,
  type or digest. A failed request with no validated return has no return entry;
  its requested case and failure event are in `diagnostic`.
- `digest`: JSON value type and SHA-256 of canonical UTF-8 JSON; no value. Low
  entropy values may be guessed from their hashes: **digest is not anonymization**.
- `value`: type, digest and complete JSON value. Explicit opt-in for **trusted
  local storage and authorized output review only**. A return can contain PII,
  echoed arguments or text posing as instructions. Treat it as untrusted data.

Policy is host-side only; neither policy nor trace nor expectations are staged in
the candidate root, appended to requests, or sent back to the candidate.
The existing source/input/suite hashes and bounded stdout/stderr remain unchanged;
`none` is not a promise that stdout/stderr or case IDs are non-sensitive. No new
exception message or exception traceback is retained. No host `repr`/`str` of a
candidate exception, object deserialization, pickle or candidate code evaluation
is used.

The CLI preserves `--source`, `--cases`, `--output`, and `--help`, and adds:

```text
--trace-policy none|digest|value   (default: none)
```

`value` emits sensitive actual values to stdout and, if requested, `--output`.
Do not redirect these into public logs. Value-policy files are newly created with
owner-only mode `0600` (subject to a more restrictive umask); exclusive creation
still refuses existing paths/symlinks. The library does not choose a storage
location, upload anything or authorize export. Parent UI/export controls must
honor `evidence_flags: ["sensitive_actual_value_trace"]`. That flag is set whenever
value retention was requested, including blocked runs with empty traces, so it
cannot disappear just because no return arrived.

`return_trace` is a host-generated **output-only evidence registry key**. Input
case keys are unchanged: exactly `id,function,args,kwargs,expected`. No trace,
registry, expectation, resource or verdict fields from input cases are accepted.

## Bounds and replay semantics

All existing containment/preflight settings and bounded transport defaults remain:
64 KiB response body including its JSON-RPC envelope; 1 MiB cumulative response
stream including 4-byte frame headers; 64 KiB request body; 1 MiB cumulative
requests; 128 cases; 16 depth; 4096 nodes; 16 KiB UTF-8 per string; signed 64-bit
integers. Only exact null/bool/int/string/list/dict values are accepted. The same
Unicode scalar, escape-byte, depth, duplicate-key and preallocation checks apply
regardless of retention policy. Traces never truncate a rejected response into a
passing value. Diagnostic errors use the **same response channel and same caps**,
not a new unlimited side channel. Existing stdout/stderr output cap is unchanged.

Trace metadata is not part of the candidate response envelope and does not reduce
the prior response budget. A serialized evidence document also includes metadata
and may use ASCII escapes; its entire file size is not the transport frame size.
The retained values must still fit the original canonical response envelopes and
cumulative framed-response cap. Replay reconstructs those envelopes with the
same fixed 32-character transport ID width and enforces both limits again.

`replay_returns(trace, cases)` verifies ordered case identity, closed entry shape,
retention/phase/type, canonical digest and bounded values, then **independently**
compares each retained value to the caller's current expectation using `_equal`.
It does not read stored `matched`, `passed`, counters or expected answers. An
incomplete trace returns its verified prefix with `available: false`. None/digest
policies return `values_not_retained`; malformed value traces raise `ValueError`.
Replay does not launch candidates, execute source or certify execution. Hashes
are not signatures: an attacker able to replace both a trace and digest can forge
an artifact. Authenticate the trusted local store before relying on replay.
`available: true` means all values were replayable, **not** all values matched and
not that the original process exited normally. Replay must never replace the
execution gate.

## Exception diagnostics and exit evidence

The adapter attempts a standard JSON-RPC error subtype with fixed numeric code,
fixed message and only bounded phase plus exact-builtin exception type code.
An exact identity whitelist is captured before candidate import; unknown types
(including custom subclasses and codec `ProtocolError`) become `opaque`.
Exception messages, arguments, attributes and overridden `__str__`/`__repr__` are
not inspected. The adapter can be modified after import, so **every reported phase
and exception code remains explicitly `untrusted_candidate`**, even `MemoryError`.
A forged well-shaped error is still an execution failure, never a passed return or
proof of resource exhaustion. Extra fields, counters, custom type strings,
messages and simultaneous result/error envelopes are rejected.

`diagnostic.host_phase` describes host-observed setup/request/response state.
`candidate_error.phase` is only an untrusted claim of import/call/serialize.
Response EOF without a report cannot establish which candidate phase failed.
`requested_case_id` comes from the host's currently issued case, not a candidate
ID field; opaque transport IDs are correlated against the host outstanding ID.

`termination.observed_exit` is the first poll **before cleanup/grace**. If the
process is not yet reaped, it is null with explicit `unknown` state; it is never
backfilled with the later forced `-9`. A grace of at most 50 ms, limited to the
remaining wall deadline, permits monitor/child exit propagation.
`exit_before_signal` is a second poll after that grace. Cleanup action, signal and
reason are recorded separately from `final_exit`; a normal exit 1 remains 1.
No minimum wait extends the deadline. At an exhausted deadline the final exit may
still be unknown even after SIGKILL was sent. Compatibility `returncode` is the
final observed code, not the original observed code. `oom_proven` is always false:
SIGKILL, `MemoryError` claims and the existing generic resource-signal category do
not prove OOM. No peak-memory, CPU or aggregate-resource measurements are invented.

## Exact additive JSON schemas

Existing top-level result schema/oracle mode remain version 1 /
`host_data_only_functions_v1`; these fields are additive. The following Draft
2020-12 schema describes the four new top-level fields and reusable wire/API
subtypes. Existing result fields are intentionally permitted in `properties`'s
parent object. Closed child schemas reject unspecified fields. JSON Schema cannot
express the byte/node budgets, exact Python builtin identity, digest computation
or identity/order correlation; those procedural constraints above are normative.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["return_trace", "diagnostic", "termination", "evidence_flags"],
  "properties": {
    "return_trace": {"$ref": "#/$defs/trace"},
    "diagnostic": {"$ref": "#/$defs/diagnostic"},
    "termination": {"$ref": "#/$defs/termination"},
    "evidence_flags": {
      "type": "array", "maxItems": 1, "uniqueItems": true,
      "items": {"const": "sensitive_actual_value_trace"}
    }
  },
  "$defs": {
    "caseId": {"oneOf": [
      {"type": "string", "minLength": 1},
      {"type": "integer", "minimum": -9223372036854775808, "maximum": 9223372036854775807}
    ]},
    "value": {"oneOf": [
      {"type": "null"}, {"type": "boolean"},
      {"type": "integer", "minimum": -9223372036854775808, "maximum": 9223372036854775807},
      {"type": "string"},
      {"type": "array", "items": {"$ref": "#/$defs/value"}},
      {"type": "object", "additionalProperties": {"$ref": "#/$defs/value"}}
    ]},
    "policy": {
      "type": "object", "additionalProperties": false,
      "required": ["schema_version", "return_retention"],
      "properties": {
        "schema_version": {"const": 1},
        "return_retention": {"enum": ["none", "digest", "value"]}
      }
    },
    "entry": {
      "type": "object", "additionalProperties": false,
      "required": ["id", "phase", "retention", "reason"],
      "properties": {
        "id": {"$ref": "#/$defs/caseId"}, "phase": {"const": "response"},
        "retention": {"enum": ["none", "digest", "value"]},
        "reason": {"enum": ["policy_none", "retained"]},
        "value_type": {"enum": ["null", "bool", "int", "string", "list", "dict"]},
        "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "value": {"$ref": "#/$defs/value"}
      },
      "oneOf": [
        {
          "properties": {"retention": {"const": "none"}, "reason": {"const": "policy_none"}},
          "not": {"anyOf": [{"required": ["value_type"]}, {"required": ["sha256"]}, {"required": ["value"]}]}
        },
        {
          "properties": {"retention": {"const": "digest"}, "reason": {"const": "retained"}},
          "required": ["value_type", "sha256"], "not": {"required": ["value"]}
        },
        {
          "properties": {"retention": {"const": "value"}, "reason": {"const": "retained"}},
          "required": ["value_type", "sha256", "value"]
        }
      ]
    },
    "trace": {
      "type": "object", "additionalProperties": false,
      "required": ["schema_version", "policy", "returns"],
      "properties": {
        "schema_version": {"const": 1}, "policy": {"$ref": "#/$defs/policy"},
        "returns": {"type": "array", "maxItems": 128, "items": {"$ref": "#/$defs/entry"}}
      }
    },
    "errorData": {
      "type": "object", "additionalProperties": false,
      "required": ["schema_version", "phase", "type_code"],
      "properties": {
        "schema_version": {"const": 1},
        "phase": {"enum": ["import", "call", "serialize"]},
        "type_code": {"enum": ["opaque", "ValueError", "TypeError", "RuntimeError", "KeyError", "IndexError", "ZeroDivisionError", "ImportError", "ModuleNotFoundError", "AttributeError", "NameError", "SyntaxError", "MemoryError", "RecursionError", "AssertionError", "OSError", "SystemExit"]}
      }
    },
    "wireError": {
      "type": "object", "additionalProperties": false,
      "required": ["jsonrpc", "id", "error"],
      "properties": {
        "jsonrpc": {"const": "2.0"}, "id": {"type": "string", "pattern": "^[0-9a-f]{32}$"},
        "error": {
          "type": "object", "additionalProperties": false,
          "required": ["code", "message", "data"],
          "properties": {
            "code": {"const": -32000}, "message": {"const": "candidate error"},
            "data": {"$ref": "#/$defs/errorData"}
          }
        }
      }
    },
    "candidateError": {
      "type": "object", "additionalProperties": false,
      "required": ["trust", "schema_version", "phase", "type_code"],
      "properties": {
        "trust": {"const": "untrusted_candidate"}, "schema_version": {"const": 1},
        "phase": {"enum": ["import", "call", "serialize"]},
        "type_code": {"$ref": "#/$defs/errorData/properties/type_code"}
      }
    },
    "diagnostic": {
      "type": "object", "additionalProperties": false,
      "required": ["schema_version", "host_phase", "event", "requested_case_id", "candidate_error"],
      "properties": {
        "schema_version": {"const": 1},
        "host_phase": {"enum": ["preflight", "setup", "request", "response"]},
        "event": {"enum": ["not_executed", "not_started", "request_queued", "awaiting_response", "setup_eof", "response_eof", "response_eof_before_setup", "return_validated", "candidate_error", "transport_error", "wall_timeout"]},
        "requested_case_id": {"oneOf": [{"type": "null"}, {"$ref": "#/$defs/caseId"}]},
        "candidate_error": {"oneOf": [{"type": "null"}, {"$ref": "#/$defs/candidateError"}]}
      }
    },
    "exit": {"type": ["integer", "null"]},
    "termination": {
      "type": "object", "additionalProperties": false,
      "required": ["schema_version", "observed_exit", "observed_exit_state", "exit_before_signal", "cleanup_action", "cleanup_signal", "cleanup_reason", "grace_seconds", "final_exit", "final_exit_state", "oom_proven"],
      "properties": {
        "schema_version": {"const": 1},
        "observed_exit": {"$ref": "#/$defs/exit"},
        "observed_exit_state": {"enum": ["known", "unknown"]},
        "exit_before_signal": {"$ref": "#/$defs/exit"},
        "cleanup_action": {"enum": ["none", "kill_process_group", "process_group_already_gone"]},
        "cleanup_signal": {"enum": [null, "SIGKILL"]},
        "cleanup_reason": {"type": "string"},
        "grace_seconds": {"type": "number", "minimum": 0, "maximum": 0.05},
        "final_exit": {"$ref": "#/$defs/exit"},
        "final_exit_state": {"enum": ["known", "unknown"]},
        "oom_proven": {"const": false}
      }
    },
    "matchedCase": {
      "type": "object", "additionalProperties": false, "required": ["id", "matched"],
      "properties": {"id": {"$ref": "#/$defs/caseId"}, "matched": {"type": "boolean"}}
    },
    "replay": {
      "type": "object", "additionalProperties": false,
      "required": ["schema_version", "available", "reason", "cases"],
      "properties": {
        "schema_version": {"const": 1}, "available": {"type": "boolean"},
        "reason": {"enum": ["values_not_retained", "replayed", "incomplete_trace"]},
        "cases": {"type": "array", "maxItems": 128, "items": {"$ref": "#/$defs/matchedCase"}}
      }
    }
  }
}
```

Policy/entry retention must agree; known/unknown exit states must agree with
non-null/null codes. Value policy and the sensitive flag must agree. These are
also enforced by the host construction. The CLI `INVALID_INPUT` envelope remains
its existing minimal shape and is not an evaluation result covered by this schema.

## Parent integration / gate compatibility

- **No `_function_passed` change required.** Keep its exact `cases` comparison and
  all existing execution/isolation/resource/counter checks. Do not add trace
  fields inside case verdict entries or use replay availability as an admission
  result. New host-side fields do not change the existing mode/version.
- To expose value retention through higher-level certification, thread the
  explicit validated policy to `evaluate_functions` as an optional host argument;
  do not accept arbitrary candidate/input keys or weaken defaults.
- Register `return_trace` as optional output evidence only. Treat
  `sensitive_actual_value_trace` as authorization/UI/export-sensitive. Redact or
  omit full trace values from public views/exports unless separately authorized;
  never infer authorization from candidate metadata. Existing stdout/stderr may
  also be sensitive. This file does not implement parent export controls.
- Store evidence with provenance/integrity binding in a trusted local registry.
  The oracle hash changes because implementation changes; old expected pin sets
  must not be silently reused or represented as unchanged.
- Display original observed exit, post-grace exit and cleanup/final code separately.
  Label candidate error type/phase as claims. Never label SIGKILL as proven OOM.

## Validation evidence (synthetic only)

Final combined run: **134 passed in 60.39 seconds**, covering the new suite
and all pre-existing function oracle tests, with actual Linux isolation enabled
(no skips). Prior incremental runs: 130 passed / 62.77 seconds and 133 passed /
60.32 seconds. The embedded JSON Schema parsed successfully (13 definitions).
Command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/agent/workspace/silt-observability/src /agent/workspace/silt-venv/bin/python -m pytest -p no:cacheprovider -q /agent/workspace/silt-observability/tests/test_oracle_observability.py /agent/workspace/silt-observability/tests/test_function_oracle.py
```

The unqualified `python` launcher was absent and system `python3` lacked pytest;
the existing workspace venv was used without installing packages. Tests cover
wrong structured values, all retention modes, independent replay/type traps,
forged errors/counters, no exception text/repr, Unicode byte limits, EOF/crash and
cleanup races, original exact-frame/total-limit/isolation regressions, and CLI
compatibility/privacy. New synthetic seed-shaped correct/wrong-output fixtures
exercise the strict parent gate. No consumed benchmark cases, reference code,
models, final outputs or prior implementation roots were executed or modified.
Independent parent review remains required; passing tests are not production
admission evidence.
