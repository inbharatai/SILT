# Experimental Studio observability glue

This is a **trusted single-user loopback CLI bridge**, not a hosted multi-user
service, an evidence authenticator, an admission rule, or a model experiment.
The implementation changes only `src/asea/studio/experimental.py` and its
`static/experimental.html`; focused tests are in `tests/test_observability_studio.py`.
The public Compose CLI already owns preview, retention and export semantics.
See [OBSERVABILITY_WORKFLOW.md](OBSERVABILITY_WORKFLOW.md),
[GENERATION_OBSERVABILITY.md](GENERATION_OBSERVABILITY.md), and
[ORACLE_OBSERVABILITY.md](ORACLE_OBSERVABILITY.md).

## Public command mapping and typed operator intents

All jobs retain the named-operator and `local_files_confirmed: true` requirements.
The API schema remains strict (`extra=forbid`, strict types); booleans are JSON
booleans, not `1` or strings. There is no arbitrary-flags field.

| Studio operation / request fields | Public CLI arguments |
|---|---|
| Compose `preview`, `spec`, `output_format: raw_python|fenced_python` | `python -m asea.compose preview --spec=FILE --output-format=FORMAT` |
| Compose `evaluate`, `trace_policy: none|digest|value` | `--trace-policy=POLICY` |
| Compose `evaluate`, `value_trace_consent: true` | Studio-only local storage consent; **no new CLI flag** |
| Compose `export`, `include_sensitive_traces: true` | `--include-sensitive-traces` |
| Compose `export`, `sensitive_export_confirmed: true` | Studio-only separate export confirmation; **no new CLI flag** |

`trace_policy` defaults to `none`. When omitted from an API request the existing
argv is preserved and the public CLI supplies its `none` default. The browser
sends its selected policy explicitly. Value retention without the strict boolean
consent is rejected **before queueing/spawning** as `CONFIRM_VALUE_TRACE`.
The visible warning says value traces **contain return values**; this includes
wrong outputs, echoed inputs, PII, and untrusted instruction-like strings.
Digest retention is not anonymization.

The export boolean and its separate confirmation must agree; mismatches return
`CONFIRM_SENSITIVE_EXPORT`. Non-default new controls on another action are
`INVALID_ARGUMENT`; unknown fields or invalid enums/types are `INVALID_REQUEST`.
Explicit default-valued fields remain accepted in full legacy `model_dump()`
requests, but are never forwarded on an irrelevant operation. Value consent on a
non-value policy is rejected. `cache_policy`, `output_contract`, `arbitrary_flags`,
and other unsupported top-level UI request fields are rejected rather than ignored.

Preview requires a format. It never passes `--workspace`, registers an artifact,
or produces a rewritten file. Successful preview envelopes must have
`read_only: true` and `applied: false`; malformed successful envelopes fail the
existing CLI-envelope check. The selected format, proposed contract, original
prefix, and prefix-review flag are visible in the privacy-filtered result.

**No generation/cache mutation controls:** after reviewing a preview, author a
**new user-owned spec**. This UI never edits JSON, migrates prefixes, repairs
fences, interprets task prose as a declaration, or retunes a model. The public
runtime remains the only format-instruction renderer. Unsupported adapters stay
explicitly unsupported in the public preview result.

### Preview write boundary

Standalone public `compose preview` is read-only even if a caller supplies an
unused `--workspace PATH`: no Workspace construction, model inventory, loading,
registration, or workspace writes. Spec validation and safe spec reads still occur.
Tests prove both formats work with missing model directories, with and without
that workspace argument, and leave the input bytes/directory contents unchanged.

Studio itself must initialize its private outer root, lock, queue and job metadata.
A Studio preview therefore creates an **outer job log**, not a Compose workspace.
The distinction is visible in guidance/summary. Job metadata is mode `0600` in
mode `0700` directories, as before. No claim of zero outer Studio filesystem writes
is made.

## Three independent privacy decisions

1. **Creation/storage:** `value_trace_consent` authorizes one newly submitted
   evaluation's trusted local value retention. New jobs record
   `observability_authorization` with schema, `source: studio_operator`, operator,
   policy, storage consent, export intent and separate export confirmation. This
   is operator metadata, not cryptographic identity. Historical jobs without it
   remain `unavailable / original_unknown`; no historical consent is invented.
2. **Viewer reveal:** a separate unchecked `#reveal-sensitive` control authorizes
   viewing the selected job's raw streams, token IDs and appropriately flagged
   retained values, and downloading a deployment archive. The authenticated
   request sends `X-SILT-Reveal-Sensitive: true`; absent/`false` means no consent,
   and other spellings return typed `INVALID_ARGUMENT` (422). The header is never
   a URL token. Reveal state is not stored server-side and resets when switching
   jobs/workflows. Unchecking clears the displayed/downloadable copy immediately;
   stale in-flight responses from another job or reveal state are discarded.
3. **Export creation:** `include_sensitive_traces` plus
   `sensitive_export_confirmed` is a separate intentional disclosure decision.
   Storage or viewer consent never sets these fields. Operation changes reset the
   storage/export checkboxes and return policy to safe defaults.

### Projection endpoint behavior

- All API routes retain the session-token, loopback, same-origin and fetch-site
  checks; responses remain `no-store`. There is no generic workspace reader.
- Job list responses use only bounded job identity/status metadata, not historical
  raw stdout, stderr, result, argv, summaries or error strings.
- Submission/cancel responses and ordinary GET job responses are privacy-filtered
  copies. Original job and signed workspace records are not rewritten.
- Raw top-level and nested stdout/stderr are hidden by default, including invalid
  or truncated CLI output; generation continuation/returned-sequence IDs are also
  redacted in the default projection. Safe counts/configuration facts remain.
- Full return values require **both** explicit viewer reveal and the containing
  oracle envelope's `sensitive_actual_value_trace` evidence flag, together with
  `return_retention: value` and entry `retention: value`. The UI does not infer this
  flag from the generic evaluation `sensitive_observations` flag, a candidate
  claim, or historical metadata. Unflagged/malformed value entries stay redacted
  even after reveal, and their alternate raw streams stay withheld.
- Permitted value JSON is retained exactly, including nested objects and wrong
  outputs. It is rendered using `textContent`, never HTML or evaluation. Candidate
  value objects are not recursively interpreted as telemetry schemas.
- Once permitted, raw stdout is the original captured string, not reconstructed
  JSON. The separate `#projected-result` shows a clearly labeled display copy.
- "Save job evidence JSON" saves **the current projection** (including its privacy
  markers), not a portable signed original. An authorized revealed copy can be
  sensitive. Review it before sharing.
- Deployment archive download independently requires the viewer header, including
  inherited historical exports. Existing non-export artifact roles preserve their
  established transport behavior; this is not a general PII detector or a new
  blanket license to publish source/input/model artifacts.

The display explicitly reports `original_integrity_validated: false`. An evidence
flag is a privacy label, **not proof of HMAC authenticity**. The UI does not weaken
or substitute for CLI activation/export integrity and admission checks. Persisted
job records retain the existing trusted-local-owner storage model.

## Read-only observation panels

`#generation-observations` shows each available `generation_trace_v1` entry with
separate requested, forwarded and resolved metadata, token counts, observed EOS
IDs/positions/basis, cap reach, token-ID truncation, stop reason, media metadata,
and the original trace projection (including omission/failure markers).
Forwarded omission stays omission, not `use_cache=false`. Resolved configuration
is not hidden HF execution state. Unknown counts remain null; missing fields are
`unavailable / original_unknown`, never backfilled as zero or false. No count is
inferred from text and no EOS-at-cap observation is labeled a proven stopping cause.

`#oracle-observations` separately labels host diagnostics and **UNTRUSTED candidate**
exception phase/type claims. Original observed exit and its unknown state are
separate from post-grace exit, cleanup signal/reason and final exit. SIGKILL or a
candidate MemoryError claim never proves OOM. Trace policy and retained/redacted
actual return entries remain available for inspection, not admission.

`#observation-bindings` displays existing per-case `obs_v1` bindings. Panels traverse
only the selected job's already captured CLI result (bounded nesting/entry count).
They do not follow local paths, open referenced run records, expand tensors, load
models, read datasets, or replay candidates. Thus an evaluate envelope may contain
only a generation binding while full node details live in its signed run record;
the UI marks those details unavailable here. A Compose run job whose CLI envelope
contains full generation traces can display them. This deliberate limitation
avoids a new unauthenticated filesystem reader or Workspace side effects.

No field changes a return status, evaluation routing, exact host comparisons,
resource gates, activation, or pass/fail behavior. A completed preview or model
process is not a quality verdict. Legacy records stay historical.

## Export meaning: refuse, not redact

The public exporter is fail-closed by default: sensitive observation payloads cause
`SENSITIVE_TRACE_EXPORT_REQUIRES_OPT_IN` before creating the archive. Opt-in exports
**exact originals** in the existing bundle format; the UI never emits a redacted
bundle and calls it HMAC-replayable. The scan includes actual value policy/content,
generation token IDs and potentially sensitive diagnostic/log containers, not only
flags. Even `none` or `digest` return policy can require explicit opt-in.

A successful default export means the existing exporter found no sensitive
observation content under its defined scan, **not** a guarantee that all original
source/input/suite fields contain no secrets or PII. Review those independently.
Opt-in never bypasses existing deployment approval, artifact binding, integrity,
dependency, destination, resource or implementation-fingerprint checks. Refused
exports leave signed originals unchanged. Importing a bundle does not grant
activation authority.

## Selectors for parent smoke validation

- `#mode`, `#operation`; Compose operations include `preview`.
- `input[name=spec]`, `#output-format` (`raw_python`, `fenced_python`).
- `#trace-policy` (`none`, `digest`, `value`), `#value-trace-consent`.
- `#include-sensitive-traces`, `#sensitive-export-confirmed`.
- `#reveal-sensitive`, `#privacy-status`, `#projected-result`, `#evidence` (raw),
  `#stderr`, `#argv`.
- `#generation-observations`, `#oracle-observations`, `#observation-bindings`.
- `#save-evidence`, `#cancel`, `#submit` retain existing behavior/routing.

## Focused validation (no models or frozen datasets)

With the copied source tree and existing venv:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/agent/workspace/silt-observability/src \
 /agent/workspace/silt-venv/bin/python -m pytest -p no:cacheprovider -q \
 /agent/workspace/silt-observability/tests/test_observability_studio.py \
 /agent/workspace/silt-observability/tests/test_experimental_studio.py \
 /agent/workspace/silt-observability/tests/test_pass2_studio.py
```

Result: **93 passed**, one pre-existing FastAPI `on_event` deprecation warning.
An additional selected `test_integration_contracts.py` run with
`-k 'ui_action or legacy_server or public_cli_help or select_'` yielded
**15 passed, 21 deselected**, with the same warning. The selection avoids any model
or dataset tests while exercising the real mounted app, legacy opt-in routing,
auth boundaries, public CLI help/errors and exact action/selector contracts.
The inline JavaScript also passed `node --check -` (Node v24.14.1).

Logs are outside the code tree:
`/agent/workspace/silt-observability-evidence/studio-contract-tests-final.log` and
`studio-integration-contracts.log`. Tests use fresh synthetic metadata, missing
model paths, tiny transport subprocess doubles and the existing model-free UI
contracts. They do not claim a real pretrained smoke, browser visual review,
quality measurement or end-to-end sensitive admitted-deployment export. Those
remain parent-owned validation. No Git, prior-root edits, model runs, retuning,
frozen-final dataset reads or spec mutations were performed by this UI task.
