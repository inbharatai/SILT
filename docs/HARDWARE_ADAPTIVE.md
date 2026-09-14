# Hardware-aware specialist feasibility planning

`asea.hardware` is a **local feasibility planning module with bounded authorized TRAIN/DEV profiling**, not a model loader,
training runner, quality evaluator, approval service, or reservation system.

**Evidence boundary (2026-09-12):** framework-adaptive capacity is based on native
architecture, tensor dtypes, phase lifetimes and observed resources, never GPU
product-name checks. Linux CPU has actual tiny native loading/training/export
validation. NVIDIA CUDA is implemented but **GPU_UNVERIFIED** here; all six actual
NVIDIA pipeline tests skip on this CPU-only host. Native MPS/macOS, ROCm/HIP,
Windows, multi-GPU and offload execution are blocked. There is no any-hardware
fit, numerical-equivalence, throughput or quality guarantee. A separate real
pretrained CPU engineering run completed all five stages, two optimizer updates,
strict reload and fresh inference. It used one selected TRAIN row and one supervised
DEV row, two calibration samples and a 32-token generation cap; no final suite was
consumed and no new quality-retention result is claimed. Prior V5 quality remains source **12/16** versus recovered
**8/16**, with no aggregate recovery improvement; see the
[recorded comparison](EXPERIMENTAL_RELEASE_2026_09.md).
Importing it does not import Torch, Transformers, or PEFT. There are no new app or
external tool API capabilities implied by this module.

```python
from asea.hardware import probe_hardware, plan_specialist

hardware = probe_hardware(
    python_executable="/path/to/venv/bin/python",
    paths=("/local/models/source", "/local/output-parent"),
)
plan = plan_specialist(
    "/local/recipe.json", hardware,
    workspace_parent="/local/output-parent", requested_device="auto",
)
```

## Public contract

- `probe_hardware(python_executable=None, paths=()) -> dict`
- `plan_specialist(recipe, hardware=None, *, workspace_parent=None,
  requested_device=None, python_executable=None, profile_data=True, inference_scope="dev") -> dict`
- Recipe can be a dictionary or local JSON path. Both use the **same strict
  `asea.specialist.workflow.load_recipe`** as the workflow. Relative dictionary
  paths are relative to the working directory; file recipe paths are relative to
  the recipe's directory. Unknown fields and invalid dtypes remain errors.
- An explicit `requested_device` overrides device selection in the recipe. If
  neither is explicit, the standalone planner searches `auto`. The legacy
  workflow may normalize an omitted execution device to `cpu`; an already
  normalized recipe consequently preserves that CPU request.
- Schema version is 1. Core fields include `status`, `execution_device`, `reasons`,
  `model`, `hardware`, `phases`, `estimates`, `bindings`, `plan_sha256`,
  `data_profile`, `no_execution`, `no_execution_semantics`, `quality_approval`,
  `reservation`, `runtime_guards_authoritative`, and `evidence_level`.
- `status` is `READY`, `BLOCKED`, or `PLANNING_ONLY`. Only a local safetensors
  checkpoint with a validated native shape inventory can be `READY`.
  A config-only calculation can be `PLANNING_ONLY` (or `BLOCKED` when a limit
  fails), **never READY**. A missing source cannot be inferred from its name.
- `execution_device` is `cpu`, `cuda:N`, or null. Null means there is no admitted
  execution choice. Phase device choices in a blocked plan are estimates only.
- `no_execution` is always true; `quality_approval` and `reservation` always false.
  READY means estimated resource feasibility, **not** quality, certification,
  legal/license approval, data correctness, or successful native loading.
- `plan_sha256` hashes canonical JSON of every root field except itself.
  `bindings` contains normalized recipe, inventory and hardware hashes, requested
  device, workspace parent and permitted data SHA256/size/device/inode metadata. These are
  change-detection bindings, not signatures or transferable admission tokens.

## Discovery and its limits

Host discovery is stdlib-first. Linux RAM is the intersection of `MemAvailable`
and the **remaining** memory of every visible cgroup ancestor (`limit - current
usage`), across cgroup v1 and v2. CPU capacity intersects logical CPUs, affinity,
cpusets and ancestor quota/period ratios. Mount roots are resolved through
`/proc/self/mountinfo` and `/proc/self/cgroup`, including namespace-relative
mounts. No cgroup tuning, quota changes, cache dropping, or memory-limit overrides
are performed. Inaccessible enclosing constraints cannot be inferred; malformed
visible constraints fail closed rather than inventing headroom.

Windows memory uses read-only `GlobalMemoryStatusEx`. WSL kernel or launcher
presence is descriptive only; distributions are not enumerated or started.
macOS uses read-only `sysctlbyname(hw.memsize)`; available RAM remains unknown
rather than equating installed memory with free memory. Those native platforms
are discovered but not admitted for this execution backend.

Every requested disk path gets its own `shutil.disk_usage` observation at its
actual existing ancestor. A future output directory is measured on its parent
volume. The planner never borrows source-volume free space for an output volume.
Caller-supplied snapshots must contain a disk entry whose `path` exactly matches
the resolved `workspace_parent`; missing or ambiguous entries block admission.

A fixed script in the **selected Python interpreter** imports Torch in an
isolated subprocess. Output is limited to 64 KiB and process waiting to 20 seconds
(with bounded cleanup). Only tiny 2×2 arithmetic checks and device discovery run;
no model, tokenizer, dataset, tensor checkpoint or network loader is called. The
snapshot records Torch/dependency versions, CPU dtype primitives, CUDA device
index/name/free/total memory/native BF16 support/dtype primitives, HIP version,
and MPS availability. It does not collect hostnames, serials, GPU UUIDs, machine
IDs, environment dumps or credentials. If Torch is absent or times out, host
observations survive and execution feasibility fails closed.

`profile_kind: synthetic` labels fabricated test profiles. Supplied profiles are
not authenticated hardware evidence, including a dictionary previously obtained
from a real probe. `evidence_level` does not upgrade them to hardware validation.
No synthetic GPU name or result is a claim that such hardware was detected.

## Native metadata adapters

The planner reads bounded local JSON and bounded safetensors **headers**, checks
exact key/shape coverage and unique tied-alias storage, and streams file hashes.
Hashing necessarily reads checkpoint bytes, but never deserializes or materializes
checkpoint tensors. Native safety helpers refuse symlinks, unsafe/pickle model
assets, remote-code/base-model indirection, ambiguous shard layouts, duplicate
keys, invalid spans, and unsupported architectures. Native tokenizer assets must
be present; default planning loads the native local tokenizer and validates/encodes authorized TRAIN/DEV text without constructing a model.
Metadata/header parser bounds are safety bounds, not a general parameter-count
ceiling. Model sizes and GPU product names are not dispatch rules.

**Qwen2:** only gate/up projection rows and down projection columns are pruned.
The retained intermediate width is

```text
kept = floor(intermediate_size * retention / 64) * 64
base_parameters = source_unique_parameters
                - 3 * num_hidden_layers * hidden_size * (intermediate_size - kept)
```

The retained width must be positive, at least 64 and strictly smaller. Attention,
Q/K/V biases, GQA dimensions, embeddings (including exact tying), norms and LM
head are counted independently and are not multiplied by the retention fraction.

For the **config-only dimension example** `hidden=2048`, `intermediate=11008`,
`layers=36`, `vocab=151936`, `heads=16`, `KV heads=2`, tied embeddings and
`retention=.75`, kept width is 8256 and base parameters are 2,477,240,320.
BF16 base tensors alone are **4,954,480,640 bytes**, before F32 LoRA factors,
tokenizer and serialization overhead. A 4 GiB output cap (4,294,967,296 bytes)
therefore blocks before loading. This is a verified shape formula, not a claim
that this example's full checkpoint was loaded or its quality validated.

**Switch Transformers:** native ReLU, equal-shaped experts, one expert per sparse
layer, router removal and native dense-T5 mapping. Retention is explicitly marked
not applied for this existing fixed-top-1 family. The planner does not pretend to
implement concatenation, expansion, multiple-expert selection or quality repair.
Switch source router dtype is preserved (original F32 cannot be silently rounded).
Switch source `wo` is not assumed F32; the subsequent native T5 recovery loader's
F32 `wo` exception and its BF16-to-F32 cast source pages/scratch are counted
separately. The exported base/factor budget uses those native recovery dtypes.

Default planning opens and hashes the designated `manifest.json`, answer-free
`selection-lock.json`, full `train.json` and `validation.json`, and reviewed
`validation-suite.json`. Complete rows are validated and encoded with the local
native tokenizer before seeded selection. Only DEV suite `input` fields enter
tokenization; the complete suite (including DEV expected results) is hash-bound.
Calibration contents are not read by the planner; the approved manifest binds
its expected hash, and runtime validates an exact TRAIN subset (a separate file
is legitimate). No FINAL answers/suites are opened, hashed, or tokenized by planning.
Freshness uses IDs/families in the answer-free lock and authoritative operator ledger.

`no_execution=True` means no model construction, tensor materialization,
generation or training, **not no dataset access**. Use `--no-data-profile` on
`python -m asea.hardware plan`, or `profile_data=False`, for header-only planning;
this does not read TRAIN/DEV contents or tokenize them and cannot return READY.
Source metadata and checkpoint hashing still occur in header-only mode.

The controller sends the approved plan's permitted-data SHA/size/device/inode
map as a SHA-bound `--expected-data-binding-file` plus
`--expected-data-binding-sha256` to public `asea.specialist build`. Core admission
checks open-once bounded snapshots before model stages and validates those same
bytes. Build receipts and final eligibility retain these approved identities.
Runtime checks actual immutable inputs before/after each model stage; changes
stop the study, and unchanged-data failures are never suppressed.

Quarantine admission uses stat-only final-alias checks, O_NOFOLLOW and descriptor
identity checks **before reading**. Designated operator metadata (recipes,
manifests, locks, approval/ledger/receipt inputs) must have exactly one hardlink,
including before final paths are known. Non-weight native config/tokenizer assets
have the same conservative constraint; weight-cache hardlinks are not globally
banned. Permitted TRAIN/DEV hardlinks to known final paths are rejected by stat.
These are trusted-owner immutable-file constraints, **not an adversarial sandbox
or atomic host guarantee**: malicious same-owner in-place replacement of readable
contents can only be detected after a read, and edits after a stage guard can
race a stage. Keep all study inputs immutable throughout the run.

## DEV and unseen FINAL are separate admission scopes

`inference_scope="dev"` profiles every rendered input in the reviewed complete
DEV suite through the same native `render_prompt`/tokenizer path used by actual
generation. TRAIN and supervised DEV response lengths separately size recovery
and the full-vocabulary teacher bank; recovery `max_length` is not an inference
input bound. The bound is neither sampled nor silently fixed to 2048. The reviewed
local 0.5B example has 42 TRAIN + 8 DEV rows and DEV input lengths
`[101, 62, 67, 71, 67, 64, 69, 80]`: the DEV inference maximum is **101**. This does
not change responses, full KL vocabulary, BF16 precision or generation budget.

`inference_scope="final"` is a separate conservative **unseen** scope: no final
file is opened, hashed or tokenized, and `final_input_fit_proven=false`. It budgets
only final evaluation plus report disk, not a second build. `finalize` requests
this scope before its durable consumed marker or final reads, including on CPU.
DEV READY does not establish final fit. In the recorded envelope snapshot final
was BLOCKED; a reviewed bound and/or sufficiently larger, independently admitted
resources are needed. More RAM alone does not prove the lengths of unseen inputs.
A future fixed final-input envelope must be explicitly reviewed and bound through
protocol/recipe/API; no such owner-approved envelope was supplied or silently
added. The public hardware CLI currently plans DEV; final scope is exposed by the
Python API and used internally by finalize, not a CLI `--inference-scope` flag.

`NativeGenerator`, `infer` and `evaluate` accept Python API
`max_input_tokens=None` (a declared value is currently bounded to 1..2048).
`infer` profiles its actual prompt before weights when omitted; `evaluate`
profiles every authorized suite input and checks any supplied bound. Each
`generate` still checks actual tokens, native context, RAM and selected-device
memory. It refuses oversize rather than truncating or reducing generation.
There is no public specialist `--max-input-tokens` option in this revision.

## Explicit native loading and phase lifetimes

Evaluation now constructs native Qwen2/T5/Switch (plus separately guarded,
evaluation-only Llama) with meta **parameters** and real CPU buffers via
`init_empty_weights(include_buffers=False)`. Strict native key/shape/tie planning
precedes per-tensor mmap assignment; no opaque HF whole-model load is assumed.
CPU buffers remain on CPU and bounded, tied aliases retain shared storage,
T5 `wo` stays F32, and original F32 Switch routers are audited without a lossy
round-trip or second router read. Missing, extra or malformed headers fail closed.
Local generation configuration is retained; explicit greedy generation policy is
unchanged. After loading, no parameter or buffer remains meta.

For this proven loader strategy, same-dtype/device assignment aliases mmap storage;
a dtype conversion allocates its native target. Host estimates charge exact loaded
storage, retained source pages for **all** cast tensors, mapping overhead, and the
largest source-plus-target cast scratch. Retained pages receive no presumed release
credit. Shared `inference_host_phases` takes the maximum of loader and forward
lifetimes, not their sum; loaded base storage is not counted twice. Unknown loader
strategies retain conservative duplicate-storage allowance rather than borrowing
this proof. Pre-import admission charges a 512 MiB incremental library/meta reserve
and bounded 16 MiB buffers, then post-import/pre-weight admission reobserves actual
remaining memory before weights. Factor-preserving loading additionally accounts
for factor initialization and saved-factor copies.

**CUDA still loads the full native model on the host before `.to(cuda:N)`** and
stages full host serialization. Per-tensor CPU assignment is not out-of-core
execution, offload, direct-to-GPU streaming or a way to omit host capacity. More
VRAM cannot waive CPU reconstruction, host loading or export requirements.

Recovery workspace separately counts F32 factors, Adam m/v, gradients,
checkpointed block inputs, one recomputed eager block, full-vocabulary output/CE/KL
and a 128 MiB reserve. Saved inputs scale with depth; attention in the active block
scales quadratically with sequence. LM-head/CE/KL work is not multiplied by depth,
and frozen-weight autograd aliases add no duplicate weights. Three times one-block
forward workspace covers recomputed values, gradients and backward scratch; it is
a conservative modeled constant, not universal measured peak RSS. GPU guards
subtract only observed live factor/optimizer/gradient storage already counted in
free/reserved observations, bounded by its modeled component; no allocator-cache
release is assumed. The superseded full-inference-times-depth estimate is not
current CUDA feasibility evidence.

## Resource policy and phases

All phases are recorded: source evaluation, CPU reconstruction, cached teacher
bank generation (when applicable), adaptation, export, and inference/evaluation.
Estimates explicitly separate raw tensor bytes from native loaded storage and
RSS/workspace allowances. Counts include:

- Source dtype groups; all cast-source mmap pages and largest cast-tensor scratch.
- Complete replacement Qwen MLP matrices alongside mapped source pages.
- F32 full-vocabulary logits, eager attention, activation and generation/KV bounds.
- F32 LoRA factors, gradients, Adam states and snapshots; merged-export COW and
  matrix windows or factor-preserving writer/reload verification.
- Native runtime allowance of 512 MiB and tokenizer allowance of
  `max(64 MiB, 8 * tokenizer_asset_bytes)`, plus header/mapping overhead.
- Cached full-vocabulary F32 teacher-bank disk, reconstructed/output copies,
  export scratch, factors, tokenizer and headers. Resident teacher mode keeps
  teacher storage/workspaces in the overlapping adaptation phase instead.

Host RAM, output-volume disk and CUDA VRAM reserve
`max(256 MiB, floor(10% of observed free))`. The recipe's
`memory_budget_bytes` remains a **host** ceiling, never a VRAM setting.
Reconstruction/recovery use their additional-allocation ceiling semantics;
evaluation/standalone use a total-process-RSS ceiling. The planner records the
selected tokenizer interpreter's RSS and deducts that baseline for evaluation's
operator cap; runtime reobserves current RSS. Observed host/cgroup free memory
already excludes resident use, so RSS is not subtracted from that headroom twice.

`device_memory_budget_bytes` is separate: current PyTorch reserved storage plus
proposed additional allocation must fit it independently of driver free VRAM
minus reserve. `output_budget_bytes` is authoritative for the complete emitted
bundle and is a positive integral byte value at most `2**63 - 1` (a numeric
representation bound, not a 64 GiB hardware cap). The legacy default stays **4 GiB**;
the 3B template's **6 GiB** is PROPOSED, not owner-approved. Actual output disk,
workspace and emitted-byte checks still apply. No fixed host-RAM or parameter-count
ceiling is introduced, but this is not unlimited operation: the controller timeout
is at most 3600 seconds, recovery at most 4096 steps, and native token/context and
parser bounds remain explicit. Equality with an effective estimate/cap is accepted;
one byte below is blocked. Runtime rechecks actual resources.

`estimates.proposed_output_budget_bytes` is an **explicit proposal**, not an edit
to the recipe, resource limits, retention, rank, dtype, model, or context. Tensor
lower bounds alone are not used as a claim about RSS or total working disk.
Conservative estimates can reject a run that a more tightly proven implementation
could support; timing/throughput and timeout completion are not predicted.

## Backend choice and evidence

Reconstruction remains **CPU-only**, even when another phase is planned on CUDA.
The CPU dtype primitive, host reconstruction and current CPU admission checks
must fit independently of VRAM. For `auto`, choose the first feasible single
NVIDIA CUDA device by logical index whose native dtype capability and primitive
probe are valid; otherwise choose feasible CPU. Explicit absent/unsupported
requests block instead of falling back. No dtype conversion to make a GPU fit.

CPU execution is implemented and has existing Linux evidence. Single NVIDIA
CUDA execution is implemented by the device backend but **requires actual
pipeline validation on NVIDIA hardware**; a successful 2×2 primitive probe is not
that validation. This work's real probe observed two CPU cores, roughly 4 GiB
host RAM, Torch 2.6.0+cpu and **no CUDA devices**. It provides no CUDA throughput,
training success or all-hardware portability proof. Windows native execution,
macOS/MPS, ROCm, multi-GPU execution and offload are explicitly unsupported.
Discovery alone never fabricates execution capability.

## Standalone CLI

```bash
python -m asea.hardware probe --python /path/to/venv/bin/python \
  --path /local/models/source --path /local/output-parent --output hardware.json

python -m asea.hardware plan --python /path/to/venv/bin/python \
  --recipe recipe.json --device auto \
  --workspace-parent /local/output-parent --output plan.json
```

`plan` also accepts `--python` to select the interpreter used by its probe. JSON
always carries explicit no-execution flags. `--output` publishes exclusively and
never overwrites an existing report. Exit code is 0 for a probe or READY plan,
2 for blocked/planning-only plans. Neither command starts a workflow.

Public specialist recipes and `recover`/`infer`/`evaluate --device` accept `cpu`,
`auto`, or canonical `cuda:N`; use `cuda:0`, **not bare `cuda`**, at these interfaces.
Reconstruction has no device or device-budget flag. Host `--memory-budget-mib`
applies to reconstruct/recover/infer/evaluate; separate
`--device-memory-budget-mib` applies only to recover/infer/evaluate. CLI MiB values
must convert to positive integral bytes; APIs/recipes use byte integers. See the
[current controller API and receipt contract](SPECIALIST_QUALITY_EXPERIMENT.md).

## Evidence: confidence from bounded real mechanics, not a feature list

The native-loading handoff records actual tiny CPU Qwen2/Llama/tied-Switch/F32-T5
bitwise parameter/logit/generation-ID parity, same-dtype mmap pointer sharing,
cast storage separation and one payload read per tensor. **BF16 T5 is a native
precision correction, not old-HF bitwise equivalence:** HF 4.51.3 loaded `wo` BF16;
the native path preserves F32. Raw-HF logits differ; bitwise parity holds after
canonicalizing the comparison model's `wo` to F32. Original F32 checkpoint bits
are separately SHA-checked. Strict unsupported untied Switch fixtures still fail
rather than permit randomly initialized missing embeddings.

Envelope evidence includes actual tiny Qwen/T5 CPU forward/backward unique-storage
measurements across depth 2/4 and sequence 16/32, checkpointing, LoRA, CE and full
KL, plus unchanged two-step F32/BF16 export/reload parity tests. These support the
specific storage/lifetime model, **not** all-size RSS/kernel-workspace bounds or
GPU numerical correctness. Synthetic hardware/cgroup/VRAM boundary tests establish
selection arithmetic and fail-closed behavior only. All six real NVIDIA tests
(four family/dtype factor-recovery cases and two native-merged cases) remain skipped
here, not counted as GPU success. No new whole-suite CI claim is made.

Latest **planning-only** local snapshots after native loading:

| Recipe | Overall additional CPU estimate | Recorded admission |
|---|---:|---|
| Separate small engineering recipe | 2,462,955,856 B; reconstruction dominates | READY against 2,787,014,247 B effective headroom |
| Original default recipe | 2,679,379,760 B; export dominates | BLOCKED against 2,622,685,594 B effective headroom |

Small source evaluation is 1,824,103,760 B; default source evaluation is
1,826,985,296 B. Their CUDA estimates remain respectively 1,751,988,272 B and
2,273,937,200 B, **unmeasured GPU_UNVERIFIED estimates**. Earlier READY snapshots
cannot override later free-memory observations. No allocation reservation, completed
real two-step engineering run, or default-fit claim follows from this table.

A config-only 3B proposal is likewise not READY: no local pinned checkpoint or
actual 3B tokenizer profile was supplied. Its recorded conservative example gives
~12.25 GiB additional host and ~9.52 GiB CUDA, suggesting 32 GiB host + a single
16 GiB native-BF16 NVIDIA device for review, not certifying a card. Full-vocabulary
teacher-bank workspace can dominate disk: the 512-row configured-token example
needs ~94.9 GB workspace before reserve (proposal: at least 120 GB free beyond
source assets). Replace config-only assumptions with actual bound inventory/data
and replan; do not transfer the 0.5B DEV lengths to another tokenizer/model.

The authoritative engineering inputs for this documentation are the native-loading,
envelope, data-binding and controller-repair handoffs maintained outside the Git
checkout. Final independent verification recorded 2,447 passed, 14 skipped and
92 warnings; six skips require actual NVIDIA CUDA. The real pretrained CPU smoke
and fresh eight-token inference passed, with no external teacher loaded for serving.
The test/verification reports are supplied in the local handoff, not a remote CI claim.
For a user-owned Linux/WSL checkout, follow the
[device-local validation runbook](SPECIALIST_QUALITY_EXPERIMENT.md#device-local-codex-runbook)
before any larger experiment. Do not substitute an agent-container checkout or a
successful discovery probe for genuine GPU pipeline tests.
