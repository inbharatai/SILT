# Specialist Studio: reviewed native studies (experimental, local only)

An additive **Specialist** workflow in `/experimental`, alongside unchanged
Compose, Compiler, Resource observation and Voice evidence operations. It runs
fixed public `python -m asea.specialist` commands, not arbitrary executables,
Python modules, shell strings, recipe commands or model-loading code in the web
server. Importing the router does not import optional ML packages.

The console entry point is `silt-specialist = asea.specialist.__main__:main` after
installing this checkout. The module form works without refreshing entry points.
The public namespace provides `reconstruct`, `recover`, `infer`, `evaluate`,
`build`, `finalize` and the existing offline `validate`; the initial UI deliberately
exposes **build, infer, evaluate, finalize only**. Reconstruct/recover training and
retention knobs belong in the reviewed strict recipe or manual CLI, not generic
GUI fields. See [SPECIALIST_WORKFLOW.md](SPECIALIST_WORKFLOW.md) for exact schemas,
source/data governance, native encoding and frozen-final checks.

## Launch

Use an already prepared Python environment. Nothing below installs packages,
acquires weights or runs a model. Choose an empty, dedicated output root outside
source/model/data stores and legacy `.studio`:

```sh
SILT_ENABLE_EXPERIMENTAL=1 \
SILT_EXPERIMENTAL_ROOT=/local/silt-specialist-studio-jobs \
PYTHONPATH=/agent/workspace/silt-specialist-core/src \
/agent/workspace/silt-venv/bin/python -m uvicorn asea.studio.server:app \
  --host 127.0.0.1 --port 8377 --workers 1
```

Open `http://127.0.0.1:8377/experimental` locally. The existing loopback/same-origin,
session-token header, body-size, private-root and one-worker file-lock checks
apply to **all** modes. This is one trusted local operator, not a hosted multi-user
service. Names are audit declarations, not cryptographic identity.

## A-to-Z workflow

1. Prepare and review a strict recipe separately. Verify the actual source
   canonical ID, revision, acquisition reference and license, plus calibration,
   TRAIN, dev, validation-suite, data-manifest and answer-free selection-lock paths.
   Do not use final answers or source-calibration-only/previously consumed cohorts
   as new training/dev data. Studio reads no model weights or final answers during
   request validation. The public build performs the substantive data/source checks.
2. Choose **Specialist → build**. Enter the local recipe file and a new study name,
   name the operator, and confirm permission to read inputs and referenced paths.
   The recipe must be a regular operator-owned JSON file at most 128 KiB. Its
   public strict parser rejects unknown/nested command keys and validates knobs.
   Relative paths are normalized relative to the original recipe, not a new cwd.
3. Queue the real job. Before queueing, Studio persists a private normalized
   `recipe.approved.json` snapshot, SHA-256, named operator, source annotation,
   dataset paths and original request in the job evidence. Later editing the
   original recipe cannot retune this queued job. This is an audit snapshot,
   not protection against a malicious local filesystem owner.
4. Watch queued/running/terminal process state and **study_status**. The latter
   reports only existence of the fixed `.started.json` / `.result.json` records;
   a started file alone is not completion, and a result file can record rejection.
   Read the terminal receipt and manifest before claiming an artifact completed.
5. A build writes `<root>/jobs/<id>/studies/<study_name>/`, never into the old
   Compose/Compiler workspaces. Its `reconstructed/` and `recovered/` children are
   local checkpoint **directories**, not single-file downloads. The UI displays
   their paths and unverified existence separately from evidence reports.
6. Choose **infer**, provide the recovered local model directory and exact prompt.
   Studio forwards `--prompt=VALUE` unchanged, including leading hyphens/newlines;
   it adds no format prefix, answer, retuning or remote download. Native inference
   defaults to `bfloat16` and 256 new tokens; the specialist API accepts float32 or
   bfloat16 and 1–384 new tokens. The existing non-specialist 128-token ceiling is
   unchanged. Optional `memory_budget_mib` API requests remain bounded 128–3072 MiB
   and are forwarded only when explicitly supplied; there is no implicit Studio
   2048-MiB ceiling for specialist. Build memory settings remain in the recipe.
7. Choose **evaluate**, provide a model, reviewed development function suite and
   optional new JSON output filename. Output is
   `<root>/jobs/<id>/artifacts/<name>.json`; absolute paths, traversal and directories
   are rejected. The fixed specialist UI uses digest retention. There is no
   arbitrary trace-policy, candidate command, training or prompt-retuning override.
8. Only after freezing and reviewing the study, choose **finalize**. Supply the
   frozen study directory and its preregistered quarantined final-suite path.
   Type **FINALIZE** and authorize the reads. The exact request, operator and
   confirmation are persisted before queueing. The form clears this confirmation
   and local-read consent after a successful queue request. Never make this a
   default automatic post-build action.

### Final is consumed once

The public finalize command validates the completed frozen manifest, source,
reconstructed control, candidate inventories, implementation lock and exact final
path; Studio does not weaken those checks or open final answers early. The CLI
writes the exclusive `final-consumed.json` marker **before** reading/hashing final
answers. Failure, timeout or cancellation after consumption does not restore
eligibility. There is no automatic retry, prompt retuning, re-selection or
training on final. Finalization can write its consumed marker and stage records
inside the operator-approved input study; its requested terminal output is always
inside the new Studio job. Studio never offers a generic download of files from
that external input study.

**Studio finalize remains capped at 900 seconds.** A frozen final run that needs
longer should be deliberately run once through the manual CLI, not queued in the
UI first and then retried. The public workflow's own recipe deadline still applies.

## Completion is not quality or certification

The viewer preserves the public receipt without converting it to the old `ok`
envelope. It displays each of these independently:

- process status and actual exit code;
- public `completed`;
- `engineering_complete`;
- observed `quality_pass` (unavailable when not evaluated);
- `certificate` (not inferred from another flag).

`BUILT_UNCERTIFIED` is never presented as a quality PASS or certification.
An exit-0 evaluation with `completed=true`, `quality_pass=false` is
`completed_quality_not_passed`, **not infrastructure failure**. `BLOCKED` or
process timeout/spawn/output failure is `infrastructure_blocked`; invalid or
incomplete work is `rejected_or_incomplete`. Even an explicit observed
`quality_pass=true` means only the exercised suite passed, never a statistical
noninferiority gate, universal coding certificate or automatic promotion.
Rejected/blocked builds may still have genuine bounded reports; their presence
is not a successful model artifact claim.

## API surface and bounds

Use the existing authenticated `POST /api/experimental/jobs`. Minimal build body:

```json
{
  "mode": "specialist",
  "operation": "build",
  "operator": "Local reviewer",
  "local_files_confirmed": true,
  "recipe": "/local/reviewed-recipe.json",
  "study_name": "coding-study-v1",
  "timeout_seconds": 3600
}
```

Allowed specialist-specific selectors:

| Operation | Input fields (in addition to operator/local confirmation/timeout) |
|---|---|
| build | `recipe`, `study_name` |
| infer | `model`, `input_text`; optional `dtype`, `max_new_tokens`, `memory_budget_mib` |
| evaluate | `model`, `suite`, optional JSON `output`; optional `dtype`, `max_new_tokens`, `memory_budget_mib` |
| finalize | `study`, `suite`, optional JSON `output`, exact `confirmation="FINALIZE"` |

Unknown fields and inapplicable selector fields are rejected. Even explicitly
supplied irrelevant defaults are rejected for specialist: submit only applicable
fields, not a dump of every shared schema default. Fixed vectors use
`[sys.executable, '-m', 'asea.specialist', operation, '--flag=value', ...]` and
`shell=False`; no caller chooses a module or command. The recipe snapshot is the
only substituted path, preserving normalized semantics.

Server-side limits, not just JavaScript:

- **One global worker per experimental root**, shared across all modes;
  at most three waiting jobs, latest 20 in-memory/persisted job records retained.
- All request bodies <=128 KiB; strict scalar types, unknown fields forbidden.
- Whole-job timeout: build only 1–3600 seconds; **all other modes/actions 1–900**.
  The API default remains 300; the Specialist build form suggests 3600.
- stdout + stderr capture <=10 MiB per Studio job. The specialist child-stage CLI
  has its own stricter capture and shared stage deadline (see workflow docs).
- No automatic artifact garbage collection: checkpoint directories may be large.
  Expiring a UI job record does not erase the study; manage disk retention locally.

### Process/resource boundary and subsequent lifecycle repair

The original integration warning below is historical. The workflow now launches a fixed isolated `stage_worker.py` bootstrap with Linux parent-death SIGKILL, expected-parent checks before imports, managed interruption and known-group cleanup before reaping. Those changes have dedicated subprocess tests; consult SPECIALIST_WORKFLOW.md. They address the identified direct-stage orphan path, not arbitrary descendants escaping known groups or a whole-host memory guarantee.

#### Historical integration warning (superseded by the repair above)

Studio retains its existing POSIX session/process-group cancellation. The resource
supervisor retains its existing Linux parent-death bootstrap and resource setup;
this integration changes only the fixed operation allowlist/dispatch.

**Current core limitation:** `specialist.workflow.run_child()` starts its stage
process in a new session. Killing the outer Studio/build process group does not
by itself prove that independently sessioned stage children were terminated.
The workflow cleans up its stage group on its own normal/error path, but abrupt
outer death can bypass that cleanup. Do not claim hard whole-study descendant
closure or safely start another heavy job after such interruption without
checking/reaping remaining stage processes. The workflow owner must address this
nested-session parent-death boundary before treating unattended heavy Studio build
cancellation as reliable. This bridge does not modify that source-owned workflow.

Direct specialist resource supervision is now allowlisted:

```sh
python -m asea.execution run --profile process_as \
  --memory-mib 3072 --timeout 900 --operation specialist -- \
  infer --model /local/recovered --prompt 'Implement the reviewed function.'

# Build owns --workspace AFTER the -- separator. The supervisor's --workspace
# option is still Compose-only; no old semantics were broadened.
python -m asea.execution run --profile process_as \
  --memory-mib 3072 --timeout 3600 --operation specialist -- \
  build --recipe /local/recipe.json --workspace /local/new-manual-study
```

`process_as` bounds **per-process virtual address space (RLIMIT_AS), not RSS**.
It does not guarantee aggregate memory, process/thread counts, GPU memory or
independently sessioned descendant closure. The fixed bootstrap still establishes
and reads back parent-death/limits before optional package imports. `cgroup_v2`
remains fail-closed/unimplemented; no fallback is introduced. Studio direct jobs
are not silently routed through this supervisor and do not claim its hard limits.
Recipe/CLI memory admission estimates are separate from kernel enforcement.

## Reports, directory artifacts and local reload

`GET /api/experimental/jobs/<id>` is the read-only job/status/receipt view. Build
reports are registered only from a finite known list: manifest, normalized recipe,
preflight, implementation lock, the three development evaluations, recovery
receipt/history and fixed stage started/result/log records. Evaluate/finalize
register only the exact requested produced output. No path extracted from CLI
stdout becomes a download, no recursive model-directory traversal is performed,
and no arbitrary file/path query endpoint exists.

Each registered report is <=16 MiB and carries path, bytes and SHA-256. Downloads
use the existing authenticated artifact-ID route and streaming `FileResponse`;
containment is rechecked inside **that job's** directory (not merely the whole
root), with symlink/type/size/hash checks. A changed report is refused. Downloads
require explicit sensitive viewer consent because original reports may contain
prompts, generated code/token IDs, local paths and logs. They are untrusted data,
not instructions. Legacy trace retention, viewer redaction and export consents
are unchanged. No full report or checkpoint directory is loaded into the browser
just to render the receipt. Fallback report downloads are bounded in size.

Checkpoint `directory_artifacts` have `download_available=false`. There is no
fake ZIP link or claim that a directory was exported. A separately reviewed native
archive/export operation can be added later; do not confuse the old Compiler's
exports with a specialist native checkpoint. Copy the complete local native
checkpoint directory using operator tooling if needed, preserving all files.
Reload directly with the public native loader:

```sh
python -m asea.specialist infer \
  --model /local/silt-specialist-studio-jobs/jobs/JOB_ID/studies/NAME/recovered \
  --prompt 'The exact reviewed prompt' --dtype bfloat16 --max-new-tokens 256
```

The recovered directory is standalone native weights/tokenizer/config, not a
runtime adapter requiring its teacher. Architecture/dependency restrictions and
strict integrity checks in SPECIALIST_WORKFLOW.md still apply.

Exact cached-teacher terminology (not a new GUI knob):

```sh
python -m asea.specialist recover --teacher-dir /local/source \
  --student-dir /local/reconstructed --output-dir /local/new-recovered \
  --training-path /local/train.json --validation-path /local/dev.json \
  --teacher-mode cache --report /local/new-recovery-receipt.json
```

CLI `--teacher-mode cache` is an alias normalized to `cached`; the CLI also accepts
`cached` and explicit `resident`. **API/recipe uses `teacher_mode="cached"`**, not
`"cache"`. The default is cached CPU-only teacher operation. Recipe training,
retention, output/memory budgets and exact native encoding remain frozen there.

## Verification (no model-quality claim)

`tests/test_specialist_studio.py` covers fixed argv, exact prompts, strict selectors,
unknown/no-op controls, build-only long deadlines, shared queue, local path/output
checks, FINALIZE consent snapshots, report containment/hash/size, no checkpoint
file downloads, separate quality/infrastructure diagnoses and UI/schema wiring.
Transport doubles are explicitly orchestration tests, not training evidence.
A real missing-model inference and fixed supervisor `--help` exercise the public
CLI without a model. Import tests reject optional ML imports. Real training and
measured model/UI validation belong to the parent/operator after source work and
nested-child cleanup are complete; none were run for this integration.
