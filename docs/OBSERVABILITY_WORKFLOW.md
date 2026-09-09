# Composition observability workflow (`obs_v1`)

This is **observational evidence, not stronger admission**. No retained return,
exception report, EOS position, cache request, or diagnostic can authorize a pass.
The existing independent host comparison, exact counters, successful contained
execution, target/control coverage, graph/artifact bindings, resource checks,
implementation fingerprint and activation checks remain authoritative.

This work does not retune models, change pilot prompts, inspect frozen final
answers, run pretrained evaluations, or convert old results into fresh evidence.
Tests use newly created synthetic fixtures; those are control-flow tests, not
trained-model quality measurements.

## 1. Preview a new strict spec without changing anything

```sh
python -m asea.compose preview --spec OLD_SPEC.json --output-format raw_python
python -m asea.compose preview --spec OLD_SPEC.json --output-format fenced_python
```

`preview` is the only compose command that does not require `--workspace`. It also
accepts an existing `--workspace PATH` argument but never constructs a Workspace,
creates the directory, registers artifacts, inventories model paths, or loads a
model. Missing model files are therefore not a problem for a valid spec preview.
JSON spec reading/validation still applies, including the existing file size,
symlink, duplicate-key and closed-schema rules.

The JSON result is:

```json
{
  "schema_version": 1,
  "read_only": true,
  "applied": false,
  "output_format": "raw_python",
  "nodes": [
    {
      "node_id": "text",
      "adapter": "hf_text",
      "support": "supported",
      "schema_version": 1,
      "read_only": true,
      "supported_adapter": true,
      "proposed_output_contract": {"schema_version": 1, "format": "raw_python"},
      "instruction": "Return only raw Python source code. Do not include Markdown fences or explanatory prose.",
      "existing_prompt_prefix": "",
      "requires_prefix_review": false,
      "applied": false
    }
  ]
}
```

Every node gets a proposal, including intermediate nodes. Non-`hf_text` adapters
are clearly labeled `support: unsupported_adapter` and `supported_adapter: false`;
a proposal does not imply that it is valid to apply there. Nonempty legacy
prefixes are copied into the read-only preview and marked
`requires_prefix_review: true`. Review potentially sensitive task prose before
sharing that preview. Nothing removes or rewrites a prefix automatically.

After human review, author a **new** spec. Do not edit old frozen/pilot specs.
Strict format instructions are supported only by `hf_text` and cannot coexist
with a nonempty `prompt_prefix`. The renderer prepends the fixed instruction and
a blank line, preserves the task bytes, then uses the existing chat-template path.
It never infers a declaration from natural-language task prose, repairs fences,
or rewrites candidate output.

## 2. Declare an evaluation case's required format explicitly

A case can opt into:

```json
{"output_format": "raw_python"}
```

or `fenced_python`. This field is optional. Its default `None` is omitted from
case/suite serialization; no other legacy null/default field is removed. Legacy
case hashes and payload shapes therefore remain unchanged when this field is
absent. Declaring a format requires an explicit strict `output_contract` on the
selected output node. A legacy node is not silently upgraded.

Before the **first inference**, the evaluator checks every case's structured
format declaration against the selected output node and every other explicitly
contracted node contributing to it. Conflicting case declarations, a conflicting
intermediate contract, unsupported adapters, or a missing required output
contract block the evaluation. The existing whole-suite input and resource
preflights also remain in force. Preview alone is not approval or evaluation.

## 3. Choose return retention independently of admission

```sh
python -m asea.compose --workspace NEW_WS evaluate \
  --spec NEW_SPEC.json --suite NEW_SUITE.json --trace-policy none
```

`--trace-policy none|digest|value` defaults to `none`. Existing resource-profile,
memory, timeout, input and other defaults are unchanged.

Python API:

```python
evaluate(workspace, spec, suite, trace_policy="digest")
evaluate(workspace, spec, suite,
         trace_policy={"schema_version": 1, "return_retention": "value"})
```

Omitting `trace_policy` or passing `None` selects the exact default oracle policy.
The low-level host oracle receives the validated dict. It retains only fully
host-validated actual returns, without expected values/arguments in return-trace
entries. The source and public function inputs still go to the contained
candidate; expected answers are not echoed back to it. No host-side candidate
`repr` is called by this integration.

| Retention | Return entry | Historical return-value replay |
|---|---|---|
| `none` | id, phase, retention, `reason: policy_none` | unavailable; values were not retained |
| `digest` | same identity plus validated type and SHA-256 | unavailable; a digest is not an actual return |
| `value` | same identity/type/hash plus exact bounded actual value | possible using the oracle codec and separate expected cases; never an execution/admission credential |

The original oracle `cases: [{id, matched}]` shape and `_function_passed` criteria
are unchanged. The value policy is sensitive even if an attempt never completes.
Wrong answers can have valid bounded actual-value traces; their retention never
makes the answer correct.

## 4. Evidence fields for consumers

New evaluations keep `schema_version: 1`, the existing critical report fields,
and add:

- `evidence_schema: "obs_v1"`.
- `trace_policy: {schema_version: 1, return_retention: none|digest|value}`.
- `trace_policy_hash`: SHA-256 of the canonical policy JSON.
- `evidence_flags`: `[]` or `["sensitive_observations"]`, derived from actual
  original observation payloads, including signed run traces and blocked attempts.
- Each completed case row has `obs_v1` and `obs_v1_hash` in addition to all its
  original fields. `obs_v1.schema` is `obs_v1`.
- `obs_v1.generation` binds the original signed run's generation trace hash, run
  ID, graph hash, input digest and output hash. It also lists `missing_nodes`.
- `obs_v1.oracle` binds the retained-return trace hash and, when available,
  diagnostic and termination hashes. Diagnostic/termination status is reported
  independently of retained-return availability.
- Blocked host-oracle attempts are retained as `blocked_oracle_evidence` with
  `case_id`, `run_id`, `oracle_evidence`, `oracle_evidence_hash` and a generation
  binding. They never create a passing completed case row.

Consumers must display **original unknown**, not "no error", "no OOM", "passed",
or "not generated", when evidence was not captured. New per-case bindings use
`status: original_unknown` and explicit reasons such as
`legacy_generation_trace_unavailable`, `generation_trace_unavailable`,
`legacy_return_trace_unavailable`, `oracle_trace_unavailable` or
`metric_has_no_return_trace`. Missing diagnostic/termination status is also
`original_unknown`. Historical records without these extensions remain absent:
readers must not backfill them by rerunning a candidate or retokenizing text.

Generation traces themselves remain in signed runs under
`generation_trace_schema: generation_trace_v1` and `generation_trace_v1: [...]`.
See [GENERATION_OBSERVABILITY.md](GENERATION_OBSERVABILITY.md) for requested vs
forwarded vs resolved settings, token/metadata bounds, actual token IDs, media
bindings, unsupported cache settings, and unknown stop reasons. The evaluation's
`--trace-policy` controls function return retention, **not** generation-token
observation; token IDs can be sensitive even with `--trace-policy none`.

See [ORACLE_OBSERVABILITY.md](ORACLE_OBSERVABILITY.md) for oracle `return_trace`,
`diagnostic`, `termination` and `evidence_flags`. Oracle value retention uses
`sensitive_actual_value_trace`. Raw stdout/stderr and candidate-reported error
metadata are untrusted and potentially sensitive. A host-observed process exit
is distinct from a later cleanup signal; neither unknown nor cleanup SIGKILL is
proof of OOM.

## 5. Activation validates originals and separately reruns correctness

The implementation fingerprint now explicitly includes
`compose/generation_contract.py`; changing a renderer invalidates prior admission
evidence just like changing the oracle/runtime. It does not silently bless old
results under a new rendering contract.

Activation authenticates original workspace records, verifies trace policy/hash,
flags, generation graph/settings/hash/token bounds, return identity/type/hash and
codec bounds, known/unknown termination consistency, and per-row bindings.
Value replay can detect inconsistency with original host comparisons, but cannot
replace host counters or make a failing execution pass. Existing functional
correctness re-execution still occurs with the selected policy. Nondeterministic
timing, cleanup grace durations, diagnostic text and exception messages are **not
compared between runs for a verdict**. Fresh diagnostic observations never replace
or mutate original captured evidence.

These checks are integrity/consistency checks within the existing signed-store
trust model, not authentication of an arbitrary unsigned trace, proof of cache
internals, algorithm identity or universal correctness.

## 6. Export is fail-closed by default

```sh
# Default: refuse potentially sensitive observation data.
python -m asea.compose --workspace NEW_WS export \
  --deployment DEPLOYMENT_ID --output OUTSIDE_WORKSPACE.zip

# Explicit privacy opt-in, AFTER independently reviewing the evidence:
python -m asea.compose --workspace NEW_WS export \
  --deployment DEPLOYMENT_ID --output OUTSIDE_WORKSPACE.zip \
  --include-sensitive-traces
```

Python API: `export_deployment(..., include_sensitive_traces=False)`; this new
keyword-only boolean defaults to `False`. `include_dependencies` remains the
existing positional/optional argument. Opt-in never bypasses admission, current
fingerprint, artifact, destination, dependency integrity or deployment approval
gates. Blocked, rejected, cancelled and fixture-only evaluations cannot export as
admitted deployments, regardless of the opt-in.

**Chosen compatibility strategy: refuse, do not redact.** The version-one bundle
format and import validation expect original evidence. To avoid silently invalid
legacy bundles or changing signed originals, sensitive default export raises the
typed `SENSITIVE_TRACE_EXPORT_REQUIRES_OPT_IN` error before creating the output
archive. CLI returns its normal JSON error envelope and exit code 2. The error
message contains no sensitive payload.

The scan checks actual policy/value presence, not merely evidence flags. It is
conservative and case-insensitive for privacy labels/policies, so malformed legacy
capitalization cannot hide sensitivity. It also checks generation token IDs,
stdout/stderr, candidate errors and diagnostic containers; therefore even a
`none`/`digest` function evaluation may require explicit export opt-in. Ordinary
non-sensitive text-only mock evidence can still use the unchanged default bundle
path. This is not a general PII detector for all old suite/input/source fields;
existing bundles already contain those fields and must be reviewed accordingly.

Explicit opt-in exports the exact original evidence in the unchanged bundle
format. No redacted copies are emitted in this implementation; consequently no
redacted copy is falsely labeled original-HMAC-replayable. Workspace evaluation,
run and deployment originals are never rewritten during either refused or
successful export. As before, an integrity bundle is not a portable trust
credential; importing does not confer activation authority.

## Offline verification

Run with the source tree selected, without downloading or running pretrained
models:

```sh
PYTHONPATH=src python -m pytest -q tests/test_observability_integration.py \
  tests/test_generation_observability.py tests/test_oracle_observability.py \
  tests/test_function_oracle.py tests/test_pass2_integration.py \
  tests/test_certification_hardening.py tests/test_composition_v1.py \
  tests/test_bundle_import.py
```

OS-contained oracle tests probe actual namespace support and explicitly skip or
block when unavailable; no weaker candidate execution fallback is introduced.
The broader existing focused suite may use freshly generated tiny random models,
never pretrained quality runs. Integration logs for this change are stored outside
the repository under `/agent/workspace/silt-observability-evidence/`.
