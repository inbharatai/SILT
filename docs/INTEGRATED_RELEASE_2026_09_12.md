# Integrated release snapshot — 2026-09-12

This summary describes the combined source package: hardware-adaptive specialist
execution and safeguards, numeric/state-restoration fixes, and Studio product
repairs alongside SILT's existing transfer portfolio. This publication is source-only,
not a wheel release, deployment confirmation, hosted model service or quality certificate.

**Generation-policy correction included in this source snapshot.** Public deployment is verified separately; publication adds no model weights or new quality result. Fresh governed data and a frozen, verified effective-policy protocol are required for new quality claims; the consumed final set is not a retry set. See the [generation-policy correction](GENERATION_POLICY_NOTICE.md).

The local specialist generation-policy fix is verified on tiny models and a retained recovered-Qwen nonfinal CPU prompt under seeds 0 and 1; this is mechanics evidence, not quality. It requires **Transformers 4.51.3** and, for factor wrappers, **non-prompt PEFT 0.15.2**. Production receipts distinguish requested/resolved policy and do not claim live-mode instrumentation. Subsequent local readiness verification rebuilt the wheel and sdist and confirmed payload equality with the previously tested installed wheel. This is packaging integrity evidence, not new model-quality or CUDA evidence; rebuild and reverify packaged resources after further source/document edits.

**Independent integrated local verification (historical pre-readiness-fix snapshot):** 2,695 passed, 18 skipped, 92 warnings,
zero failures/errors. The completed run used the pinned Python 3.9 CPU environment;
this is not a clean-install, CUDA or remote CI claim. Historical component results
are listed separately below and are not added to this total.

**Later local readiness snapshot:** the byte-identical 421-file source recorded **2,816 passed, 18 skipped and 92 warnings** on Linux CPU: ten GPU-required skips (six specialist, four streamer), six optional pretrained-model skips and two host-enforcement templates. These retained results were not rerun by the publication reviewer; they establish neither model quality nor remote CI success.

[README](../README.md) · [Capability catalog](CAPABILITIES.md) ·
[Hardware contract](HARDWARE_ADAPTIVE.md) ·
[Experiment and device-local runbook](SPECIALIST_QUALITY_EXPERIMENT.md)

## What is integrated

Core L3 packets, Double Gate, packet-derived LoRA, SiltStream, ZeroForge,
SiltSpring, composition, compiler and experimental specialist reconstruction and
recovery remain distinct paths. Existing transfer gates, defaults, measured scores
and the earlier legal notices are unchanged. The earlier unmerged experiment
wrapper's binding, containment, report-admission and classification/durability
issues are remedied in this source package; this is not a claim that every failure
mode is eliminated. No consumed-final case was retuned and no score was changed.

The current package adds `asea.hardware` feasibility planning, native loading,
phase-lifetime memory estimates, explicit devices, reviewed data bindings and
controller report safeguards. Studio fixes improve navigation, small-screen forms,
README rendering and the distinction between app availability and model readiness.
No model weights are included in the public source repository. No model or endpoint
is automatically activated by installation, import, planning, build or this release.

## Hardware, memory and cache boundaries

Linux CPU is the supported and exercised adaptive specialist target. Single-NVIDIA-
CUDA execution code is implemented but **GPU_UNVERIFIED**: no real NVIDIA pipeline
success is established by the CPU evidence. A primitive probe or synthetic VRAM
fixture is not pipeline validation. An operator-reported Windows RTX5050 inventory
is not an independently retained execution receipt.

`python -m asea.hardware probe` and `plan` discover resources and estimate native
architecture/dtype/phase capacity. A plan's `READY` means estimated feasibility,
not owner approval, a resource reservation, native-loading success or quality.
Config-only proposals without actual checkpoint inventory and tokenizer/data
bindings cannot be READY. The Qwen2.5-Coder-3B template is a proposal, not an approved
or completed 3B experiment. Its proposed budget does not establish device fit.

Specialist `recover`/`infer`/`evaluate` accept `cpu`, `auto` or canonical `cuda:N`.
Explicit unavailable devices block rather than silently falling back. Reconstruction
remains CPU-only; CUDA recovery/inference still require independently sufficient
host memory. CPU staging for CUDA is **not out-of-core execution**. Native Windows,
macOS/MPS, ROCm/HIP, multi-GPU and offload are unsupported for this adaptive path.

Strict native inventory loading uses meta parameters with real bounded CPU buffers,
validated keys/shapes/ties, and explicit payload loading. Native T5 precision rules
remain significant: preserving F32 `wo` for BF16 T5 is a precision correction, not
blanket bitwise equivalence to an older HF loader. Phase estimates account for
model/factor lifetimes, activations, full-vocabulary teacher-logit cache/workspace,
export and reload; they are not universal peak-RSS, allocator or kernel-workspace
bounds. Host RAM, VRAM and disk are separate constraints. Source and input files
must remain immutable during the trusted-owner workflow.

See [hardware planning and evidence](HARDWARE_ADAPTIVE.md),
[native-loading tests](../tests/test_native_evaluation_loading.py),
[memory-envelope tests](../tests/test_specialist_envelopes.py) and
[device tests](../tests/test_specialist_devices.py). Those regression definitions
are not additional execution receipts.

## Planning, data and controller truth

Header-only planning does not tokenize datasets. The separate authorized profile
may read permitted TRAIN/DEV and designated metadata, under bounded admission;
`no_execution` means no model generation/training, not no dataset access. Unseen
FINAL remains quarantined. Reviewed data bindings are mandatory in the controller:
a five-file sidecar pins TRAIN, supervised DEV, designated manifest, selection lock
and reviewed DEV suite by path, hash, size and filesystem identity. Expected recipe,
normalized config and data pins are rechecked before effects and final consumption.
A hash-less legacy READY object or editable manifest is not approval.

A fixed Linux worker supplies scoped process-group containment, bounded output
capture and timeout handling. It does not contain every escaped descendant or
turn trusted local input into a hostile-filesystem sandbox. Exact CLI status,
completion booleans, recipe/config/data bindings and command-specific receipts must
agree; zero exit alone cannot convert missing, blocked or rejected evidence into
success. Output ownership/admission is checked before backend effects.

The authoritative terminal report is `<report>.terminal.json`; its separate
`<report>.ack.json` binds the terminal digest and records that terminal-directory
and summary-file fsync returned. A terminal snapshot is not its own acknowledgement.
Missing acknowledgement remains missing evidence. This is not an arbitrary
power-loss survival guarantee, and interruption may prevent any final receipt.
See the [controller contract](SPECIALIST_QUALITY_EXPERIMENT.md),
[wrapper](../scripts/run_specialist_quality_experiment.py),
[workflow](../src/asea/specialist/workflow.py),
[controller regressions](../tests/test_controller_integrity_review.py) and
[data-binding regressions](../tests/test_data_binding_review.py).

The wrapper does **not** implement a compact-baseline evaluation arm. Its status
remains `BASELINE_NOT_RUN`, even when `--compact-baseline` is supplied. Final needs
explicit reviewed engineering-only acceptance waiving that missing comparison,
current bound READY admission, an exclusion ledger and a separate reviewed,
exercised teacher-unavailable deployment receipt. This is not automatic four-arm
quality certification. The historical six-arm comparison below was a separate
recorded study, not evidence that this wrapper runs all six arms.

## Numeric validation and restoration fixes

Both real-HF `certify_hf_states` and toy `SpringModel.certify` now require finite
real tolerances and reference/candidate losses, validate suite mappings, and reject
non-finite derived deltas or ratios. Booleans, strings and complex values are not
accepted as numeric loss evidence. Failed toy recertification clears stale/partial
certificates. Earlier static cautions about missing finite guards are superseded.

Valid finite behavior stays unchanged, including library tolerance .02, Studio
.05 and `(loss - reference) / max(abs(reference), 1e-12)`. The real report still
measures next-token prompt loss after a full resident reference. Full supplied
suites are not automatically filtered to heldout-only cases. These numeric fixes
do not turn prompt-loss certificates into task-quality evidence or attach toy
serving/selection semantics to real HF reports.

`HFStreamer` exit restores banked parameters **and persistent buffers** from the
original full bank, retaining native dtypes, pre-offload devices and registered
object ties. It does not roll back live LoRA updates or nonpersistent runtime
caches and is not a transaction for arbitrary forward side effects, failed entry
or storage failure. See [HF implementation](../src/asea/deepapply/backends/siltstream_vendor/hf_real.py),
[toy implementation](../src/asea/deepapply/backends/siltstream_vendor/spring.py),
[numeric tests](../tests/test_spring_nonfinite_guards.py) and
[restoration tests](../tests/test_streamer_state_restore.py). Scripted scalar losses
and tiny tensor tests are engineering evidence, not pretrained model results.

## Studio and readiness

Tab-scoped navigation, mobile form layout and the bounded README table renderer
are repaired without a new frontend runtime. Health/discovery reports separate
code availability, startup-mounted experimental routes and current enablement;
it does not load models, run a hardware probe or create an experimental manager.
“App available” is not a claim that selected models, dependencies, local assets or
quality have been verified. Remote/cloud-backed connectors can still send data
off-host. Experimental Studio requires exactly `SILT_ENABLE_EXPERIMENTAL=1` before
startup; an off-start needs a restart. The public setup page is documentation,
not hosted compute or a public-to-local token bridge. See
[product regression definitions](../tests/test_studio_product_fixes.py).

## Evidence: keep these snapshots separate

- **Historical hardware branch:** 2,447 passed / 14 skipped / 92 warnings, recorded
  locally. Six skips require actual NVIDIA CUDA pipeline execution. This is not
  remote CI, GPU validation or a combined-source result.
- **Historical product branch:** 2,325 passed / 12 skipped, a separate local source
  and runtime snapshot. It is not additive with the hardware count.
- **Separate pretrained CPU engineering smoke:** all five construction stages,
  two optimizer updates, one selected TRAIN row, one supervised DEV row, two
  calibration samples and a 32-token generation cap; strict reload and fresh
  inference. No final suite was consumed. This establishes bounded engineering
  execution, not quality retention, a new benchmark score or 3B success.
- **Historical frozen V5 quality:** source 12/16; activation-unrepaired 8/16;
  recovered 8/16; random-MLP control 0/16; uniform-unrepaired control 0/16;
  SmolLM2-360M-Instruct 8/16. All six arms completed. In these unchanged historical
  observations, recovery did not improve the aggregate or beat the compact baseline.
  **Policy correction:** four derived arms have archived sampling-default overrides,
  despite requested-greedy receipts; source-Qwen and SmolLM2 logs do not. Actual
  historical per-case modes were not instrumented. This does not establish
  matched-greedy retention or a controlled competitive comparison; four source-only
  passes cannot be attributed solely to compression/training. Those final tasks
  remain consumed. No corrected score or final rerun is reported. See the
  [full correction and preserved evidence](GENERATION_POLICY_NOTICE.md).
  See the [original bounded release evidence](EXPERIMENTAL_RELEASE_2026_09.md).
- **Integrated local full-suite verification:** 2,695 passed, 18 skipped and 92
  warnings, zero failures/errors. Ten skips require actual GPU hardware, six are
  pretrained opt-ins and two are pending host-enforcement implementations. The
  existing pinned Python 3.9 CPU environment was used with offline flags; dependency
  closure and a pip dry-run passed, but this is not a clean-install claim. Remote
  CI runs independently and must be read separately. No pretrained/final experiment
  was rerun, and no new model-quality or CUDA-validation result follows.

The generation-policy correction leaves teacher-forced CE/KL and direct
`do_sample=False` strict recovery export/reload probes outside this specific
default-merge defect. Three archived algorithmic output defects remain valid
static diagnoses, not causal compression proof. Later bounded nonfinal fix
checks do not revise historical scores or the full-suite counts above.

No teacher-level quality preservation, new 3B result, universal hardware fit,
throughput gain or clean-machine portability follows. For a larger experiment,
use the [device-local runbook](SPECIALIST_QUALITY_EXPERIMENT.md#device-local-codex-runbook)
with fresh governed data and actual device evidence, not a consumed final set.

[LICENSE](../LICENSE), [NOTICE](../NOTICE) and [PATENT.md](../PATENT.md) are unchanged.
LoRA and teacher-guided training are established prior art; this integration makes
no new novelty or patent-coverage claim.
