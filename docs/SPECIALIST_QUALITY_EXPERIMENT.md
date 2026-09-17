# Specialist model-quality experiment automation

This guide describes the reproducible larger-model quality experiment route for
SILT's public specialist CLI/API. It is an orchestration layer around existing
SILT commands, not a separate implementation of reconstruction, recovery,
inference, grading or admission.

## Current evidence and scope

This is framework-adaptive **capacity admission**, not an any-hardware guarantee.
Architecture/header dimensions, exact native dtypes, phase lifetimes and observed
headroom drive decisions; GPU model names do not. Linux CPU has actual tiny native
loading, two-step recovery, export and reload evidence. CUDA execution is implemented
but **GPU_UNVERIFIED** on the current CPU-only host: all six actual NVIDIA pipeline
tests skip. Native Windows, macOS/MPS, ROCm/HIP, multi-GPU and offload are blocked.
A primitive CUDA probe is necessary but not pipeline validation.

The separately authorized real pretrained CPU **two-step engineering run completed**
all five stages, strict reload and fresh inference through the public CLI. It used
one selected TRAIN row, one supervised DEV row, two calibration samples and a
32-token generation cap. Its status is `BUILT_NOT_FINALIZED` / `BUILT_UNCERTIFIED`,
not quality approval; no final suite was consumed. This is engineering evidence,
not a new quality-improvement result. The prior
[V5 comparison](EXPERIMENTAL_RELEASE_2026_09.md) is unchanged: original source
**12/16**, unrepaired activation reconstruction **8/16**, recovered **8/16**, compact
baseline **8/16**. Recovery had no aggregate improvement; these consumed final tasks
must not become a new unseen test. Independent local regression recorded 2,447
passed, 14 skipped and 92 warnings; six skips require real NVIDIA CUDA hardware.
This is not remote CI, CUDA validation or a successful 3B execution.

The concrete reasons for confidence are bounded, real mechanisms: CPU mmap-assignment
and dtype/tie tests; native logits/generation parity under the documented precision
policy; actual tiny full-CE/KL forward/backward storage measurements and two-step
export/reload tests; forbidden-inode read instrumentation; and tiny real Linux
process-supervision regressions. These are not merely interface existence, but
neither are they large-model, GPU, quality or power-failure proofs. Details and the
precision exception are in [hardware planning](HARDWARE_ADAPTIVE.md).

## Historical local admission result (before adaptive integration)

Candidate proposed first: `Qwen/Qwen2.5-Coder-3B-Instruct`.

Status on that earlier probed machine: **BLOCKED before training**. This is not a current hardware admission or a CUDA validation claim.

Reasons observed on 2026-09-12:

- Windows has a CUDA-capable NVIDIA RTX 5050 Laptop GPU and Windows PyTorch
  reports `torch.cuda.is_available() == True`.
- That revision was Linux CPU-oriented. The adaptive integration keeps reconstruction
  on CPU and introduces explicitly selected CPU/CUDA execution for recovery and
  evaluation. Actual CUDA pipeline validation remains separate and must not be inferred
  from device discovery or fixture tests; native Windows execution is not admitted.
- WSL2 Ubuntu is available and can see the NVIDIA device through `nvidia-smi`,
  but the actual WSL Python environment used for regression does not have
  `torch`, `transformers`, `peft`, `accelerate`, `safetensors` or
  `sentencepiece` installed.
- WSL reported about 11.75 GB available RAM during the probe. Specialist
  reconstruction/recovery admission is based on observed Linux host/cgroup
  headroom and actual local source weights; no 3B run should be started until
  SILT's own admission accepts the acquired checkpoint, recipe and data.
- No approved fresh 3B recipe, local source checkpoint, compact baseline
  checkpoint or fresh final set was supplied in the repository state used for
  this probe.

This is an infrastructure/admission block, not a failed quality score. No final
data was consumed.

## Automation entry point

Use:

```sh
python scripts/run_specialist_quality_experiment.py --help
```

The script records hardware/runtime evidence, hashes the key local docs, captures
`python -m asea.specialist --help`, and invokes only the public specialist CLI:

```sh
python -m asea.specialist build --recipe /local/fresh-recipe.json --workspace /local/new-study
python -m asea.specialist finalize --study /local/new-study --suite /local/fresh-final-suite.json --output /local/final-result.json
```

### Hardware and authorized data profiling first

The wrapper invokes `SELECTED_PYTHON -m asea.hardware plan --recipe PATH
--workspace-parent PARENT`. It never imports a planner from a different wrapper
Python environment. A recipe-bearing `--preflight-only` request performs inventory
and planning, **never build or final**. Without a recipe, the receipt explicitly
says `NOT_READY_FOR_EXECUTION`; hardware discovery alone is not admission.

Default planning reads and hashes reviewed TRAIN/DEV and metadata, validates the
complete supervised rows, and tokenizes authorized text with the pinned native
local tokenizer before selection. All rendered DEV suite inputs are profiled;
DEV oracle/reference fields are hash-bound with the reviewed file but are not
sent to the tokenizer worker. Planning does not open/hash/tokenize FINAL or read
calibration contents. Runtime separately validates calibration as an exact TRAIN
subset. `no_execution` means no model construction, weight materialization,
generation or training, **not no dataset access**. Hardware CLI
`--no-data-profile` produces header-only, non-READY planning instead.

The hardware API accepts `inference_scope="dev"` or `"final"`. DEV uses the complete
reviewed input maximum (101 tokens in the bound 42+8-row 0.5B example), not a fixed
2K or training `max_length` surrogate. Final uses a separate conservative unseen
scope with `final_input_fit_proven=false`, before consumption and even for CPU.
The current hardware CLI has no scope flag; finalize selects it internally.
DEV READY cannot prove final fit. No reviewed fixed final-input envelope has been
supplied; it must be explicitly protocol/recipe/API-bound in future, not borrowed
from DEV. Larger independently admitted resources may meet the conservative
estimate but do not prove unseen input lengths. The recorded conservative final
snapshot is BLOCKED. A READY plan remains a volatile feasibility snapshot, not
training permission, quality certification, legal approval or reserved resources.

```sh
python scripts/run_specialist_quality_experiment.py \
  --python /local/venv/bin/python \
  --recipe /local/fresh-qwen3b-recipe.json \
  --workspace /local/new-qwen3b-study \
  --preflight-only --report /local/new-preflight.json
```

`--candidate` is an optional expected identity, not a label override. It must equal
the recipe's canonical source ID (or resolved local source path if no ID is
provided). The planner records the actual checkpoint inventory/config and hashes;
a metadata label alone does not prove remote model provenance.

### Execution devices and unchanged quality gates

`recover`, `infer` and `evaluate` expose `--device cpu|auto|cuda:N`; omitted means
CPU. `reconstruct` intentionally has no device flag. Build recipes accept
`execution_device` (default `cpu`). A selected plan is frozen into an exclusive
`<report>.execution-recipe.json`; only the device is resolved and relative input
paths made absolute. No training, scoring or budget settings are silently changed.
The workflow sends the same concrete device to source, reconstructed-control and
recovered evaluations and to recovery. Reconstruction stays on CPU. Final uses
the frozen device, rechecks capability, and never silently falls back. Hardware
and device source files are included in the implementation lock; adding a new
module also invalidates an old final lock. Use canonical `cuda:0`, not bare `cuda`,
in public CLI/recipes. `memory_budget_bytes` is the host budget;
`device_memory_budget_bytes` is independent. The matching CLI flags are
`--memory-budget-mib` (reconstruct/recover/infer/evaluate) and
`--device-memory-budget-mib` (recover/infer/evaluate only). CLI MiB transport is
exact to integral bytes. Recovery/reconstruction host caps are additional-allocation
ceilings; inference/standalone caps are total-process RSS ceilings, not extra
headroom. Device caps include observed PyTorch reserved bytes plus the additional
allocation, independently of free-VRAM admission.

Native inference constructs meta parameters with real bounded CPU buffers and
assigns weights tensor by tensor, preserving tied aliases, native F32 T5 `wo` and
original Switch router precision. Same-dtype mmap assignment avoids an unnecessary
full duplicate model; unknown loader strategies retain conservative allowances.
BF16 T5 is intentionally not bitwise equivalent to old raw-HF BF16 `wo`; parity is
against the native F32-`wo` policy. CUDA still performs a **full host native load
before device transfer**, plus host export staging: this is not out-of-core loading,
direct GPU streaming or offload. CPU capacity must fit even with ample VRAM.

### Current public API and approval bindings

The legacy positional call forms remain valid; optional controller guards are:

```python
build(recipe, workspace, expected_recipe_sha256=None,
      expected_data_binding_file=None, expected_data_binding_sha256=None)
finalize(study, suite, output, expected_recipe_sha256=None,
         expected_config_sha256=None, expected_data_binding_sha256=None)
```

Public `build` exposes `--expected-recipe-sha256`,
`--expected-data-binding-file` and `--expected-data-binding-sha256` (the data pair
must be supplied together). Public `finalize` exposes `--expected-recipe-sha256`,
`--expected-config-sha256` and `--expected-data-binding-sha256`; it reopens the
original approved sidecar from the study. The wrapper supplies these guards rather
than treating an editable manifest label as approval.

The wrapper hashes intended exclusive execution-recipe bytes **before publication**;
build verifies then parses those same recipe bytes before model effects. Complete
normalized effective config, its digest, source path/inventory and planner bindings
must match the returned build, not only selected labels. Expected recipe/config
bindings are checked again immediately before final.

An exclusive `<report>.data-binding.json` carries the approved five-file
path/SHA256/size/device/inode map: TRAIN, supervised DEV, designated manifest,
selection lock and reviewed DEV suite. Its intended bytes are SHA-bound before
publication. Public build checks approved snapshots and validates those same read
bytes. A consistent replacement TRAIN/DEV/manifest cannot redefine the reviewed
plan. Returned data pins must match before final eligibility. Final reopens the
approved sidecar, verifies its expected digest, compares exact maps, and rechecks
actual built-data pins before final replanning, immediately before the consumed
marker and before/after final stages. Legacy hash-less READY objects do not meet
this controller approval contract.

Quarantine checks use stat-only known final aliases, single-link designated metadata,
`O_NOFOLLOW`/`O_NONBLOCK`, and descriptor `fstat` identity/regular-file/size checks
before first read, followed by bounded-read stability and SHA verification. Pins
repeat admission rather than blindly reopening a path. Core calibration can be a
distinct file with an exact TRAIN subset. These are trusted-owner immutable-file
rules, not an atomic filesystem or adversarial sandbox: same-owner in-place changes
may only be detected after reading, and native stages can race edits after guards.
Keep inputs immutable; no claim is made to prevent all user alterations.

A build without final consumption:

```sh
python scripts/run_specialist_quality_experiment.py \
  --python /local/venv/bin/python \
  --recipe /local/fresh-qwen3b-recipe.json \
  --workspace /local/new-qwen3b-study \
  --report /local/new-qwen3b-build-report.json
```

### Durable transport and output ownership

All output parents must already exist and be writable. The report, immutable
terminal snapshot, effective recipe, workspace, final output, and source/input
paths must not collide or alias. Symlink components are rejected; the wrapper
never creates a workspace merely to hold a report.

Before any effectful build/final command, an exclusive private report reservation
is durable. Progress/summary writes use only the held owned inode, not
`replace(path)`, which could race and overwrite a user's replacement. The
**authoritative terminal result is `<report>.terminal.json`**, atomically
published by an exclusive hard link from fully written/fsynced private bytes.
The primary report contains its pointer. If interrupted during a summary rewrite,
use the immutable terminal snapshot if present; otherwise the attempt is incomplete,
never successful. Existing outputs are never overwritten.

A linked terminal snapshot saying COMPLETE is **not its own durability
acknowledgement**. It records
`final_acknowledgement: {state: "REQUIRES_SEPARATE_ACK", path: "...ack.json"}`.
The separate exclusive `<report>.ack.json` is published only after terminal-directory
and summary-file fsync return. It binds the terminal SHA, records
`state: "TERMINAL_AND_SUMMARY_FSYNC_RETURNED"` and `power_failure_guarantee: false`.
Missing ack means the acknowledgement is absent, not that durability was proven.
Fsync failure or late target collision raises rather than silently succeeding or
overwriting. This non-atomic acknowledgement does **not** guarantee arbitrary
power-loss survival. Public recover/reconstruct full receipts similarly require
successful stdout as the final acknowledgement; a snapshot cannot certify that its
own last fsync returned. Report/history/output targets are preadmitted before
backend effects; a preliminary receipt remains incomplete until history and compact
pointers are validated. History failure cannot leave it claiming completed.

All four critical controller review bug classes have repair evidence: **executed
recipe/config binding**, **outer-process containment**, **public recover/reconstruct
report admission**, and **classification/durability truth**. Subsequent DATA binding
adds the independently pinned reviewed-data boundary above. Containment uses a fixed
Linux worker launched with `sys.executable -I -S`, PDEATHSIG and parent/session checks,
kill-group-before-leader-reap and bounded inherited-pipe draining. It contains the
known process group/direct child, not escaped descendants. Non-Linux containment
explicitly blocks. These are scoped regression repairs, not a release-wide proof.

Stdout and stderr capture are bounded to 1 MiB each per command while the whole
streams are drained and hashed. Reports record raw byte counts, truncation and
UTF-8 replacement, retaining a bounded base64 sample for invalid bytes. Timeouts,
signals and launch `OSError` become durable blocked receipts; credentials/environment
variables are not dumped. A zero exit alone is insufficient: the command-specific
schema, exact source CLI status, boolean completion and engineering flags must
agree. Top-level `BLOCKED`/`REJECTED`, errors, nonzero exits and malformed receipts
cannot become completion. Raw CLI status and a separate stable classification
reason code are retained. Only the exact unavailable-local-file diagnostic is
reclassified as infrastructure; corruption mentioning config or missing tensor
keys remains rejected. Exact `StageBlocked` is operational BLOCKED, not quality
REJECTED; a disappearing frozen GPU blocks before the final consumed marker.
Quality rejection remains REJECTED. These distinctions do not imply that every
interruption can publish a receipt: absent terminal/ack evidence must stay absent.

Operational limits are explicit, not unlimited: `--timeout-seconds` defaults to
3600 and cannot exceed 3600; recovery steps cannot exceed 4096. Primitive probes
have their own bounded waits. No cgroup/resource-limit override, cache-drop trick,
hidden precision reduction or timeout-completion guarantee is provided.

### Final approval is explicit and engineering-only

`--run-final` alone is insufficient. This wrapper does **not implement a compact
baseline evaluation arm** and cannot issue a quality certificate. Its baseline
status is always `BASELINE_NOT_RUN`; merely passing `--compact-baseline` does not
execute it. Do not relabel excluded/not-run baseline or other comparison arms as
passed. Automatic final is available only under an
explicitly reviewed **engineering-only** policy waiving that missing comparison;
otherwise final is blocked. Core comparison and scientific gates are not silently
made stricter by CUDA selection.

Required flags are `--reviewed-plan-sha256`, `--acceptance-record`,
`--exclude-ledger`, `--final-suite`, and `--final-output`. Supply `--reviewed-plan`
with a saved READY plan or preflight report so live volatile RAM observations can
be rechecked without changing the reviewed recipe/inventory/device bindings.
The saved plan hash is verified; the fresh plan must independently be READY.
An acceptance record has the following explicit shape (placeholders are not approval):

```json
{
  "schema_version": 1,
  "status": "APPROVED",
  "policy": "engineering_all_pass_v1",
  "plan_sha256": "REVIEWED_PLAN_SHA256",
  "recipe_sha256": "ORIGINAL_RECIPE_FILE_SHA256",
  "compact_baseline": "not_run_engineering_only",
  "exclude_ledger_sha256": "AUTHORITATIVE_LEDGER_FILE_SHA256"
}
```

This chosen policy requires source qualification and all-pass recovered development
quality before final, and all-pass recovered final quality for wrapper success.
Core `BUILT_UNCERTIFIED` receipts may still mean engineering completion with failed
quality; their original status is preserved. Proposed recovery-delta/nonregression
criteria below are **not** implemented as extra hidden gates by this wrapper.

Before final, `--deployment-receipt` must supply a separately reviewed exercised
complete-bundle test with the teacher unavailable. Required fields are
`schema_version: 1`, `status: "REVIEWED"`, `teacher_unavailable: true`,
`passed: true`, positive integer `tasks_exercised`, the exact `execution_device`,
and the study's `frozen_models_sha256`. This exercise is **not automated here**;
without a matching receipt the wrapper says `AWAITING_DEPLOYMENT_RECEIPT` and
never consumes final. Loader existence, reload parity or metadata is not substituted
for an exercised task. The supplied receipt is recorded as reviewed external
evidence, not claimed as an independently verified test by the wrapper.

## Fresh data requirement

Do not reuse `data/specialist-v1/final.json` or
`data/specialist-v1/final-suite.json` as a new unseen final set. The repository
documents that set as consumed. A new 3B study needs a fresh governed data root
with:

- `manifest.json`
- `selection-lock.json`
- `calibration.json`
- `train.json`
- `validation.json`
- `validation-suite.json`
- quarantined `final.json`
- quarantined `final-suite.json`

The build workflow reads only permitted training/development files and reviewed
metadata before model work. Declared final hashes may be present in that metadata;
planning/build do **not** obtain them by opening or hashing final files. Final
answers are opened only by separately authorized `finalize` after its one-shot
consumption marker is written and the pre-final guards have passed.

The authoritative exclude ledger is a separately maintained JSON object:
`{"schema_version": 1, "authoritative": true, "consumed": [{"id": "...", "family": "..."}]}`.
Its file hash is bound in the reviewed acceptance record. The wrapper compares
all answer-free selection IDs/families against it before build and again before
final; a self-declared `prior_consumed_selection` array alone is not authoritative.
Calibration is intentionally an exact TRAIN subset, not a fourth disjoint split.
Final answers are not parsed for freshness checks. Freeze/ledger/one-shot guards
are intended to prevent **consumed-set reuse under the recorded policy**, not all
possible user alterations or same-owner dishonesty. JSON `APPROVED`/`REVIEWED`
records are explicit external assertions with hashes, not authenticated human
signatures. The engineering waiver explicitly excludes the unimplemented baseline
comparison; it cannot prove the proposed four-arm quality experiment or certify
superiority.

## Proposed 3B budget correction — not owner approval

The Qwen 3B template now explicitly proposes a **6 GiB** output ceiling
(`6442450944` bytes), replacing the infeasible 4 GiB example for the retained base
plus factors and complete bundle. This is a **PROPOSED template default**, not a
user-approved budget change or a guarantee of feasibility. Existing explicit
recipe ceilings remain intact; the legacy core default is not silently raised.
`memory_budget_bytes: null` means observed host admission, not a fixed 4 GiB
operator cap. For larger models the planner derives the necessary base/factor/
bundle and working-memory budget from model metadata, reports infeasibility and
proposed requirements, and never increases user limits implicitly. Review a new
recipe and its plan before experiments. The unchanged legacy output default is
**4 GiB**. Accepted positive integral `output_budget_bytes` values end at
`2**63 - 1`, a representational limit **not a 64 GiB hardware cap**. Actual disk,
plan, complete-bundle and emitted-output checks remain authoritative.

The latest native-loading planning snapshots admit only the small engineering
recipe on that snapshot (2,462,955,856 B additional CPU); the original default
recipe is BLOCKED against later observed headroom (2,679,379,760 B needed). Earlier
READY plans do not reserve RAM. The 3B config-only proposal lacks a local pinned
checkpoint and 3B tokenizer profile, so is not READY; no large experiment follows
from a proposed output ceiling. Its provisional 32 GiB host / single 16 GiB
native-BF16 NVIDIA recommendation requires actual inventory, data, disk and GPU
validation. Neither the superseded CUDA multiplier nor a generic double-base-load
estimate is the current explicit-loader proof; see [phase arithmetic](HARDWARE_ADAPTIVE.md).

## Proposed acceptance criteria before training

These thresholds must be approved before a real run. They are recorded as
proposal metadata by the automation, not silently treated as owner approval.

- Source qualification: evaluate the source first on the complete development
  suite. The default recipe `source_quality_floor` is `1.0`; if changed, record
  the reason before running. A below-floor source may continue only as research,
  not as success.
- Quality: every declared development and final task must be accounted for.
  Operational blocks remain `BLOCKED`; syntax, timeout-with-supported-oracle,
  wrong return and truncation remain failed graded tasks in the original
  denominator.
- Comparison arms: evaluate the source/teacher, unrepaired reconstruction,
  recovered model and a compact local baseline under the same suite,
  max-new-token budget, dtype and trace policy. SmolLM2-360M-Instruct is the
  documented compact Llama-family example when separately acquired as a local
  safetensors checkpoint.
- Development gate: recovered quality must beat the unrepaired reconstruction
  and must not regress controls before finalization is considered under that proposed
  scientific policy. Those comparison criteria require a separately implemented and
  reviewed policy; the current wrapper's explicit engineering-only policy enforces
  source qualification and all-pass development/final gates, not a quality certificate.
- Final gate: run `finalize` once after the candidate is frozen. Do not retune,
  alter prompts, change channel selection, retrain, or relax gates based on
  final results.
- Compression: the recovered whole bundle must fit the predeclared output budget
  and its actual safetensors bytes must be less than the source safetensors
  bytes. Record complete bundle size, base size, factor size and hashes.
- Resource evidence: record CPU, RAM, GPU/VRAM, SSD space, WSL status, actual
  PyTorch/CUDA availability, SILT memory-admission fields, wall time, dtype,
  max length, max new tokens and dependency versions.
- Deployment: before final evaluation, verify complete-bundle loading through
  SILT's trusted public loader with the original teacher unavailable.
- Integrity: preserve all receipts, hashes, logs, skips and failures. A failed
  quality result remains a failure.

## Device-local Codex runbook

These are **future user-device instructions**, not commands executed during the
active CPU engineering run. Use the user's own Linux/WSL checkout and selected
CUDA-enabled virtual environment, not an `/agent` container path or native Windows
Python. Do not modify source, templates, frozen data or budgets to obtain a pass.
Choose new evidence/report paths with existing writable parents; never overwrite
an earlier attempt. Install missing reviewed dependencies in that environment
before probing; `nvidia-smi` or Windows CUDA availability alone is insufficient.

1. **Probe the actual selected interpreter first**, and inspect its real NVIDIA
   CUDA availability, logical index, native dtype support, host/cgroup and free
   VRAM/disk observations. A missing/unsupported backend is a stop, not a simulated
   profile or CPU-fallback success. Example placeholders must be replaced locally:

   ```sh
   cd "$HOME/path/to/your/SILT-checkout"
   export PYTHONPATH="$PWD/src:$PWD"
   PY="$HOME/path/to/your/cuda-venv/bin/python"
   "$PY" -m asea.hardware probe --python "$PY" \
     --path /local/models/source --path /local/output-parent \
     --output /local/output-parent/gpu-probe-new.json
   ```

2. **Complete the genuine tiny GPU suite before any large experiment.** On that
   exact interpreter/device run the existing native-device tests; preserve their
   complete log and skip reasons. The six `test_real_cuda*` cases must actually
   execute successfully, not skip or use fabricated device responses. They cover
   two-step teacher/student training, CPU export, same-CUDA reload/parity and
   native-merged/factor paths. The rest includes CPU/negative-path tests; its mere
   pass is not GPU validation. A dtype skip still leaves that requested dtype
   unverified. Do not relax tolerances; GPU/CPU equivalence is not established by
   same-device parity.

   ```sh
   "$PY" -m pytest -q -rs tests/test_specialist_devices.py
   ```

3. Only after those probes/tests succeed, inventory the actual local checkpoint,
   freeze/review fresh TRAIN/DEV and metadata with final quarantined, review the
   scientific recipe and explicit budgets, then produce a new bound plan:

   ```sh
   "$PY" -m asea.hardware plan --python "$PY" \
     --recipe /local/reviewed-recipe.json --device cuda:0 \
     --workspace-parent /local/output-parent \
     --output /local/output-parent/reviewed-plan-new.json
   ```

   `hardware plan --python` is supported. Set the reviewed recipe's
   `execution_device` to canonical `cuda:0` (or deliberately reviewed `auto`),
   not bare `cuda`. Require live READY and review the CPU staging/export and
   full-vocabulary teacher-bank disk estimates, not just VRAM. The wrapper's
   own preflight must independently agree in the selected environment.

4. Obtain actual owner permission for the experiment, use a new study root, and
   execute through the public wrapper/CLI. Keep baseline `BASELINE_NOT_RUN` until
   an actual comparison arm exists. Do not pass final flags for an ordinary
   engineering smoke. Before separately approved final, require the reviewed
   plan/acceptance/exclusion/deployment bindings above and conservative final
   admission. Preserve raw results, terminal and ack JSON. Stop on BLOCKED or
   REJECTED; do not retune after final or promote a completed engineering run into
   a quality certificate.

Exact-version standalone runtime validation remains in force, including Torch
wheel/build version; a CPU `+cpu` bundle is not automatically portable to a different
CUDA wheel. A GPU-produced bundle likewise retains its runtime identity contract.
The returned standalone `.model` is native; arbitrary unbounded forwards are the
caller's responsibility. Use guarded `NativeGenerator`/`infer` for prompt admission.

## Adaptive integration regression scope

`tests/test_specialist_quality_experiment_runner.py` and
`tests/test_hardware_dispatch.py` use private fixture roots and explicitly fake
planner/CLI responses for negative-path and dispatch mechanics. Existing core
workflow tests are retained. These tests are not quality evidence and do not
validate an actual GPU pipeline. No new quality or final run is authorized by
this documentation.
