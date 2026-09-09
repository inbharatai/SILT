# Pass-2 experimental Studio: fixed public CLI forms

This is an additive bridge in `src/asea/studio/experimental.py` and its existing
`static/experimental.html`, not a replacement Studio or a new evaluator.
Read contracts first: [PASS2_WORKFLOWS.md](PASS2_WORKFLOWS.md),
[COMPILER_DIAGNOSTICS.md](COMPILER_DIAGNOSTICS.md),
[RESOURCE_CONTROLS.md](RESOURCE_CONTROLS.md), and
[VALIDATION_GOVERNANCE.md](VALIDATION_GOVERNANCE.md).

## Start and trust boundary

Use the existing installed Studio launcher/operator flow documented in
[EXPERIMENTAL_STUDIO.md](EXPERIMENTAL_STUDIO.md), with
`SILT_ENABLE_EXPERIMENTAL=1` and a private `SILT_EXPERIMENTAL_ROOT` outside legacy
`.studio`. Open `/experimental` on a loopback host. The parent/operator installs
the NEWPASS package; this bridge never installs packages or downloads models.
All processes use the Studio interpreter, `sys.executable`, and only fixed public
modules. Existing Compose and compiler routes/actions remain available.

The existing token, same-origin and loopback checks protect **all** job/cancel
writes and artifact reads. No arbitrary executable, module, command-string,
unknown flag, private execution fixture, or execution `run` is exposed. Inputs
beginning with `-` stay values via `--flag=value`. Inputs retain local-owner,
symlink and special-file checks; output is an exclusive new filename/directory
inside that job's `artifacts/`, never an absolute user-selected destination.
The permission checkbox authorizes reads of paths nested in supplied manifests;
this remains a trusted same-UID operator tool, not a hostile-manifest sandbox.

## Selectors and API fields

The existing endpoint is `POST /api/experimental/jobs`. Requests use the closed,
strict `JobRequest` schema exposed by authenticated `GET /api/experimental/schema`.
Always provide `mode`, `operation`, nonblank `operator`, and
`local_files_confirmed:true`. Whole-job `timeout_seconds` stays 1–900, default 300.
The process-group timeout/cancel behavior, one serial worker, three waiting jobs,
20 retained records and combined 10 MiB capture ceiling are unchanged.

Browser controls:

* `#mode`, `#operation`, `#operator`, `#submit`, `#refresh`, `#cancel` retain their IDs.
* `#guidance` explains the selected action and its evidence limitations.
* Every field selector is `[name="FIELD"]`; its visibility wrapper is
  `[data-field="FIELD"]`. Disabled hidden inputs are not sent.
* `#summary` shows process status, interpretation and produced path/size/SHA-256
  references. `#evidence` retains the original public CLI envelope (collapsed by
  default); `#stderr` and `#argv` preserve original logs and exact argv.
* `#artifact-list` has authenticated download buttons. `#save-evidence` saves the
  Studio job record, not a fabricated certificate.

### Compose / evaluate

Added optional fields:

| Field selector | Exact public flag | Validation / semantics |
|---|---|---|
| `[name="resource_profile"]` | `--resource-profile` | `observe_only`, `process_as`, `cgroup_v2` |
| `[name="memory_mib"]` | `--memory-mib` | Integer 32–1048576; UI 512 |
| `[name="case_timeout"]` | `--case-timeout` | Finite seconds >0 and <=86400; UI 60 |

These fields are accepted **only** for Compose evaluate. Omitted API fields emit
no new flags, preserving the old CLI defaults and argv. The guided UI supplies
explicit default requests. `observe_only` means no new hard memory/time limits;
`process_as` means fresh per-case virtual-address-space-limited workers, not hard
RSS or aggregate-memory bounds. `cgroup_v2` currently blocks; **never retry in a
weaker profile or silently fall back to observe-only**.

Exact shape (paths are placeholders, not evidence):

```sh
PY -m asea.compose --workspace ROOT/compose evaluate \
  --spec=/absolute/spec.json --suite=/absolute/suite.json \
  --resource-profile=process_as --memory-mib=1024 --case-timeout=60
```

### Compiler / diagnose and roundtrip

`mode=compiler`, `operation=diagnose|roundtrip` adds the public research actions.
Use existing `[name="model"]`, `[name="suite"]`, `[name="output"]`,
`[name="dtype"]`, `[name="max_length"]`, `[name="max_new_tokens"]`, and
`[name="memory_budget_mib"]`. Diagnose requires a diagnostic suite; roundtrip
allows omission for its explicitly mechanics-only probe. Diagnose alone permits
`reference_model`. Both emit `--output` to a contained new output. Roundtrip writes
a **full checkpoint directory**, not a prune artifact: review free disk first.

New fields are `[name="preserve_router_fp32"]` -> `--preserve-router-fp32` and
`[name="worker_timeout_seconds"]` -> `--worker-timeout-seconds`. Worker timeout
is finite 0.05–900 and must not exceed the whole-job deadline. Omission emits
`min(120, timeout_seconds)`. That deadline is per diagnostic worker; the outer
Studio deadline covers the entire coordinator plus sequential workers.

Diagnostics permit `float16` or `bfloat16`, max input length <=64 and new tokens
<=32. Omitted API dtype/length/token fields use the diagnostic defaults
float16/64/16; explicit invalid values are **rejected, not coerced**. The form
shows these action-specific defaults. Existing action defaults remain float32,
128 input tokens, 32 new tokens. Existing Studio memory preflight budget bounds
remain 128–3072 MiB, default 2048; this is not a kernel RSS limit. The diagnostic
CLI additionally retains its own bounded output/address-space worker controls.

```sh
PY -m asea.compiler diagnose --model=/absolute/source \
  --suite=/absolute/span-suite-v2.json --output=JOB/artifacts/diagnostics.json \
  --dtype=bfloat16 --max-length=64 --max-new-tokens=16 \
  --memory-budget-mib=2048 --worker-timeout-seconds=120 --preserve-router-fp32

PY -m asea.compiler roundtrip --model=/absolute/source \
  --suite=/absolute/span-suite-v2.json --output=JOB/artifacts/roundtrip \
  --dtype=float16 --max-length=64 --max-new-tokens=16 \
  --memory-budget-mib=2048 --worker-timeout-seconds=120
```

**Research audits are never admitted.** Successful completion must retain
`AUDIT_ONLY`, `UNADMITTED`, `quality_certification:false` and
`stdout_contract:bounded_receipt_v1`. Exit 0 is not a parity/quality pass.
The bridge rejects old or falsely green success envelopes for these actions,
never upgrading old evidence. Detailed reports remain on disk; public stdout is
the bounded receipt with report path/hash/size and comparison summaries. Studio
registers produced artifacts and streams file SHA-256 calculation in 1 MiB chunks;
it does not load checkpoint bytes or full detailed diagnostic JSON into the page.

### Compiler / prune additions

Only prune accepts `[name="scorer"]` (`probability_mass` or `reap_dispatched`) and
`[name="minimum_expert_observations"]` (integer 1–1000000; optional). They map
exactly to `--scorer` and `--minimum-expert-observations`. Omission preserves the
public default probability-mass scorer and minimum 1. A supplied observation
minimum does not create a REAP coverage guarantee for probability-mass scoring.
Prune also accepts `preserve_router_fp32`; neither field is forwarded to generic
inference/evaluate/certify. Dtype bfloat16 is available where the public parser
accepts it; `certify` refuses it instead of substituting float16.

```sh
PY -m asea.compiler prune --model=/absolute/source \
  --dtype=bfloat16 --max-length=64 --memory-budget-mib=2048 \
  --output=JOB/artifacts/candidate --keep-experts=7 \
  --calibration=/absolute/calibration.json --scorer=reap_dispatched \
  --minimum-expert-observations=4 --preserve-router-fp32
```

### Execution / probe only

Choose `#mode=execution`, `#operation=probe`. No model/resource-run fields appear.
There are no optional application arguments:

```sh
PY -m asea.execution probe
```

The public module reads capabilities only. It neither writes cgroups nor tests
memory allocation enforcement. Studio job bookkeeping still writes local job
records. The original probe envelope has no `ok` boolean; Studio preserves it,
without manufacturing an execution guarantee or translating it to quality PASS.

### Validation / voice-check and export-listen-batch

`#mode=validation` exposes exactly these two actions; no register/freeze/finish,
check-review, ratings submission, signing, upload or listening approval.

Voice fields:

| Field selector | Exact public flag |
|---|---|
| `[name="audio_file"]` | `--audio` (existing local PCM WAV) |
| `[name="text"]` | `--text` (exact original script) |
| `[name="language"]` | `--language` (nonempty; e.g. `en`) |
| `[name="model_manifest"]` | `--model-manifest` (local JSON) |
| `[name="output"]` | `--output` (new contained JSON; default `voice-evidence.json`) |

Batch export uses `[name="manifest"]` -> `--manifest`; use the authoritative
ListeningBatch JSON schema, with existing local clips and model manifests. Output
defaults to `listening-packet.json`. This is only a local coordinator packet.

Both actions optionally record `[name="use_purpose"]` as
`noncommercial_experimental` or `commercial` with the operator in the **Studio job
record only**. The current public voice CLI has **no `--purpose` / `--use` flag**.
Studio therefore never invents one or claims the selection enforces license/use
policy. Selection grants no commercial rights, consent, or authorization.
Original manifest restrictions and raw public rights evidence remain unchanged.

```sh
PY -m asea.validation --workspace ROOT/validation voice-check \
  --audio=/absolute/output.wav --text='Exact expected speech' --language=en \
  --model-manifest=/absolute/voice-model.json --output=JOB/artifacts/voice-evidence.json

PY -m asea.validation --workspace ROOT/validation export-listen-batch \
  --manifest=/absolute/listening-batch.json --output=JOB/artifacts/listening-packet.json
```

The validation prefix differs from Compose: global `--workspace ROOT/validation`
is before the subcommand. These voice-only commands allow omission and do not
open the workspace, registry or old artifact store; the bridge supplies this
fixed new-root path consistently, never a legacy workspace.

Original success envelopes have `status:pending_human`, `quality_admission:false`,
not `ok:true`. Blocked operations emit JSON on **stderr** with exit 2. Studio
understands both without rewriting raw stdout/stderr. A completed packet remains
pending human intelligibility, naturalness and pronunciation review. Synthetic
waveform facts, scripts, populated templates, hash matches, and process exit 0
are not a human approval. The UI never autoplays/listens, fabricates ratings, or
signs. Qualified real humans must perform separately governed listening.

## Human controls and history

Changing workflow/operation clears the currently selected envelope, summary,
logs, argv and download buttons. Historical jobs stay selectable but are not
presented as evidence for an unsubmitted new form. Old records lacking summary
or hashes are labeled historical/missing, not upgraded. Jobs say
`completed · not a quality verdict`, never green PASS. Examples are placeholders;
the operator must supply actual local paths and inspect the exact generated argv.
Activation/rollback remain existing explicitly named, confirmed actions; new
research/probe/voice routes never invoke them automatically.

## NEWPASS-only engineering checks

Executed with the existing venv, explicitly selecting this NEWPASS source tree:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/agent/workspace/silt-pass2/src \
  /agent/workspace/silt-venv/bin/python -m pytest -q -p no:cacheprovider \
  /agent/workspace/silt-pass2/tests/test_pass2_studio.py \
  /agent/workspace/silt-pass2/tests/test_experimental_studio.py
```

Result: **59 passed** (10.87 seconds), one existing FastAPI `on_event` deprecation
warning. Inline UI JavaScript passed `node --check` without errors. Coverage
includes unchanged old argv, exact public diagnostic parser matching, scope and
bounds rejection, token/origin/loopback gates for every new action, actual
NEWPASS missing-input CLI failures for Compose/diagnose/roundtrip/voice/batch,
actual read-only public probe, and synthetic silent-PCM/batch preparation that
remains `pending_human`. Fake-green/old diagnostic envelopes fail closed. Tests
use no heavy models, no downloads, no real listeners and no quality admission.
Browser human-control acceptance and package reinstall remain parent/operator
steps; parser and synthetic tests are not real-model acceptance evidence.
