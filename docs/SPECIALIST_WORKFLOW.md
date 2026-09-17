# Public specialist workflow (experimental, local-only)

This is a **new namespace**, not a rename or replacement of `silt-compile`,
Compose, certification, reconstruction, or recovery APIs:

```sh
PYTHONPATH=/agent/workspace/silt-specialist-core/src \
 /agent/workspace/silt-venv/bin/python -m asea.specialist --help
```

The executable path above is an example of the existing local environment. Install
optional native-model and PEFT dependencies separately if absent. No command
acquires pretrained weights, installs dependencies, runs arbitrary shell commands,
promotes a model, or manufactures a quality certificate.

## Commands

```sh
# All existing reconstruct() arguments are exposed, with their API defaults.
python -m asea.specialist reconstruct \
  --source-dir /local/teacher --output-dir /local/new-pruned \
  --calibration-path /local/data/calibration.json \
  --family auto --method activation --retention 0.75 \
  --dtype bfloat16 --max-length 256 --max-samples 32 --seed 17

# All existing recover() arguments are exposed, with their API defaults.
python -m asea.specialist recover \
  --teacher-dir /local/teacher --student-dir /local/new-pruned \
  --output-dir /local/new-recovered \
  --training-path /local/data/train.json --validation-path /local/data/validation.json \
  --steps 64 --rank 8 --learning-rate 0.0001 --kd-weight 0.7 \
  --method lora_kd --teacher-mode cache --dtype bfloat16 --max-length 256 --max-samples 256 --seed 17 \
  --report /local/new-recovery-receipt.json

python -m asea.specialist infer --model /local/new-recovered \
  --prompt 'Return only raw Python source code. Implement the requested function ...' \
  --dtype bfloat16 --max-new-tokens 256

python -m asea.specialist evaluate --model /local/new-recovered \
  --suite /local/data/validation-suite.json --output /local/new-evaluation.json \
  --dtype bfloat16 --max-new-tokens 256 --trace-policy digest

python -m asea.specialist validate --generations /local/native-generations.json \
  --suite /local/data/validation-suite.json --output /local/new-validation.json \
  --trace-policy digest

python -m asea.specialist build --recipe /local/recipe.json --workspace /local/new-study
python -m asea.specialist finalize --study /local/new-study \
  --suite /local/data/final-suite.json --output /local/new-final-result.json
```

The reproducible operator wrapper in
[`../scripts/run_specialist_quality_experiment.py`](../scripts/run_specialist_quality_experiment.py)
collects hardware/runtime evidence and then invokes the same public commands.
It does not load models or grade outputs itself. See
[`SPECIALIST_QUALITY_EXPERIMENT.md`](SPECIALIST_QUALITY_EXPERIMENT.md) and
[`../configs/specialist_qwen3b_quality_template.json`](../configs/specialist_qwen3b_quality_template.json)
for the Qwen2.5-Coder-3B recipe template, pre-training acceptance criteria and
current local admission boundary.

`reconstruct` supports `family=auto|qwen2|switch_transformers`,
`method=activation|magnitude|uniform`. `recover` supports
`method=lora_kd|supervised_lora`; supervised-only requires `kd_weight=0`.
Both expose `--report` for an additional exclusive receipt file. A returned
recovery rejection is **not completed** and its full rejected report is retained
inside the receipt; no model artifact is asserted. Both public CLI training and
calibration bounds default to `max_length=256`, aligned
with the recipe and complete specialist-v1 native encodings. `--teacher-mode cache`
is a CLI alias for API/recipe `teacher_mode="cached"`; `resident` is explicit.
The default cached teacher policy is CPU-only. No GPU capability is claimed.

`reconstruct`, `recover`, `infer`, and `evaluate` expose optional
`--memory-budget-mib`, converted exactly to API `memory_budget_bytes`. Recipes use
`memory_budget_bytes` (integer bytes or null). Null means **no hardcoded 4 GiB RAM
ceiling**, not unlimited memory: admission uses readable host/cgroup headroom
minus its reserve and intersects it with any operator ceiling. A larger requested
ceiling never overrides observed headroom. Reconstruction/recovery retain their
existing allocation guards. Native inference uses the stage-specific accounting
below rather than reusing the reconstruction full-model formula after loading.
All estimates remain estimates, not allocation guarantees.

### Evaluation-only compact native Llama alternative

`infer` and `evaluate` also accept an **existing local safetensors** directory with
`model_type="llama"` and exactly `architectures=["LlamaForCausalLM"]`. This narrowly
adds native CPU evaluation for a separately acquired compact deployment alternative
such as **SmolLM2-360M-Instruct**. It does **not** add Llama reconstruction, recovery,
training, factor-bundle support, or a generic remote/custom-model loader.
Reconstruction remains `auto|qwen2|switch_transformers`; recovery family pairs are
unchanged. No command in this adapter downloads weights.

The separate evaluation-only constructor uses installed `LlamaConfig` and
`LlamaForCausalLM`, eager attention, local-only tokenizer/weights, and no remote
code. Admission retains safe inventory checks, bounded JSON and safetensors header
reads, dimension/version and architecture guards, and the shared exact native
key/shape/tied-alias validation **before** tokenizer or checkpoint loading. Missing
untied tensors, redundant tied storage, unsafe files/indirection, malformed headers,
quantization and unsupported head geometry are rejected; no random-initialization
or alternate-loader fallback is allowed. Llama uses the existing causal loader,
prefill, KV-cache and total-process memory checks, not reconstruction admission.

The original case input is forwarded unchanged into that tokenizer's exact native
chat template (`add_generation_prompt=True`, then `add_special_tokens=False`).
No extra task instructions or teacher/student-specific prompt rewriting are added.
The existing flat-newline causal fallback is unchanged when no chat template exists.
Runtime traces identify `model_kind="llama"` and `LlamaForCausalLM`, retain actual
input/generated token IDs, and report real resources; they are not quality claims.

Tiny **random**, tied and untied safetensors fixtures test local CLI inference,
float32/BF16 generation, memory refusal and suite input forwarding plus unsafe
config/header/key/alias rejection. They neither train a
model nor establish pretrained SmolLM2 quality, successful acquisition, or that its
measured deployment footprint fits any particular learned student. A fair size
comparison must separately inventory the acquired competitor and complete learned
student representation under the chosen dtype, retain the same suite inputs and
generation policy, and report size/quality observations without certification.
Finish this implementation revision **before** creating a new implementation lock;
never edit locked source, rewrite prior locks, or reuse an old study workspace.

### Bounded reconstruction/recovery receipt transport

Both model commands accept `--receipt-mode full|compact`, default **`full`** for
backward compatibility. `full` preserves the entire reconstruction CLI result;
with `--report`, recovery retains its existing history-externalization behavior.
The Python `reconstruct()` and `recover()` APIs and their full model evidence are
unchanged. `receipt_mode` is CLI transport only, never an API/training parameter
or a recipe field; unknown modes and booleans are rejected.

Use `--receipt-mode compact --report /local/new-receipt.json` for large models.
Compact mode requires a new report **outside** the output directory before any
model work. Build automatically requests compact mode for **both** reconstruction
and recovery, at `<study>/reconstruct.receipt.json` and
`<study>/recover.receipt.json`. No existing study is rewritten or upgraded.

A compact receipt is one JSON object with `stdout_contract` equal to
`reconstruct_compact_v1` or `recover_compact_v1`. It retains status/completion and
engineering/failure flags, and absolute `full_receipt_file`,
`full_receipt_sha256`, `full_receipt_bytes` references to the complete immutable
receipt. Successful reconstruction also has `reconstruction_manifest_*` pointing
to `reconstruction_manifest.json`; successful recovery has `recovery_manifest_*`
pointing to `recovery_report.json`. These audit references are not deployment or
teacher dependencies, signatures, quality certificates, or support guarantees.

Reconstruction's `result` projects counts/reduction, method/runtime, source and
output inventory/config hashes, bounded calibration/verification fields, selected
channel group/count/hash, and provenance/removal counts. **Exact selected indices,
per-tensor provenance, raw configurations/inventories, per-layer observations and
encoding audits remain in the full report and model manifest.** Optional proof
fields appear only when present; no success boolean is synthesized. Recovery uses
one allowlisted `result` map of status, actual/requested steps, export/admission,
standalone/parity flags, counts and source-store digests. Teacher-bank rows,
logit/token vectors, target maps and raw histories are never copied to compact
stdout. Full recovery evidence and the `.history.json` sidecar are preserved,
including partial-step rejection histories; compact history references include
absolute path, size, SHA-256 and row count.

The exact UTF-8 compact stdout payload (including newline) is capped at **64 KiB**.
Unexpected summary growth, NaN, missing/unsafe pointers or publication failures
produce a small non-success transport receipt, not truncated proof or a huge
fallback result. Already published full evidence is retained and referenced when
available. No artifact is admitted merely because a report was written.

The child transport verifies the full-receipt pointer against the **expected argv
report path**, then reads at most the existing **16 MiB** limit and hashes those
same bytes. It recreates the projection from that verified full report and checks
all claimed fields and model/history references; arbitrary claimed paths and
success flags cannot replace disk evidence. Selection arrays may exceed 10,000
channels (this is not a dataset row reader). Stage terminal summaries retain only
the verified compact projection and full-evidence pointers, not duplicate arrays.
Failure statuses, nonzero exits and `engineering_complete=false` still block or
reject the stage. All model-stage logs and failure records remain durable.

The **1 MiB aggregate stdout+stderr capture cap** and **16 MiB report limit** are
unchanged. Legacy full stdout can correctly fail that cap, even if model export
succeeded; compact stdout does not prevent unrelated stderr floods. Synthetic
10+ MiB manifest and Qwen-shaped selection tests exercise transport without native
weights. They do not establish large-model compatibility, RAM/disk fit, training
quality, or successful pretrained-model execution. Reports larger than 16 MiB
need a separate bounded format, not a raised capture default.

The new implementation lock hashes both changed executable modules plus this
workflow contract and `tests/test_specialist_workflow.py`. Preserve previous study
records and locks; a new source-frozen build must use a **new workspace**.

### Explicit factor-preserving export (no implicit fallback)

`recover --export-mode native_merged|factor_preserving` and recipe
`recovery.export_mode` default to **`native_merged`**, retaining the existing
native merge behavior and numerical tolerances. A merge parity failure remains a
rejection: there is no automatic factor fallback, dtype change or tolerance increase.
To train and keep the exact factor arithmetic, explicitly select:

```sh
python -m asea.specialist recover \
  --teacher-dir /local/teacher --student-dir /local/new-pruned \
  --output-dir /local/new-factor-bundle \
  --training-path /local/data/train.json --validation-path /local/data/validation.json \
  --steps 64 --rank 8 --learning-rate 0.0001 --kd-weight 0.7 \
  --method lora_kd --teacher-mode cached --export-mode factor_preserving \
  --dtype bfloat16 --max-length 256 --max-samples 256 --seed 17 \
  --report /local/new-factor-recovery.receipt.json
```

For the staged path, copy the recipe into a **new** recipe file, set
`"recovery": { ... , "export_mode": "factor_preserving" }`, and invoke
`build --recipe /local/new-factor-recipe.json --workspace /local/new-factor-study`.
Keep the same permitted source/data/hyperparameters for a controlled rerun. Deleted
trained factors from a rejected/cleaned-up native-merge run cannot be reconstructed
from that run's receipts: this requires fresh real training, not renaming the old
artifact. This implementation change itself does not claim a completed 64-step run.

A factor artifact contains `specialist_bundle.json`, a self-contained native
`base/` (including tokenizer), required trained F32 LoRA files in `adapter/`, a
fixed convenience `run_specialist.py`, and `recovery_report.json`. It is **not** a
Hugging Face native merged checkpoint. `base/` is categorically incomplete without
its factors; passing it alone to specialist inference is refused. `infer` and
`evaluate` take the **bundle root** with the same flags as native inference:

```sh
python -m asea.specialist infer --model /local/new-factor-bundle \
  --dtype bfloat16 --max-new-tokens 256 --prompt 'Your original recovery-format prompt'
```

Bundle detection precedes the generic native inventory (which intentionally rejects
artifact Python and adapter indirection). The installed trusted `inspect_bundle`
validates exact local paths, base/factor headers, adapter options, file hashes and
recorded counts; installed `load_standalone(..., local_only=True)` then loads one
base plus the required factors. No artifact runner is executed/imported, no
arbitrary PEFT artifact loader is dispatched, and no external teacher is loaded.
Requested dtype must exactly match the bundle, and loader dependency versions must
match its recorded versions. The exact recovery chat/flat/seq2seq prefix is reused.
Inference records `representation="factor_preserving"`, native `source_model_class`,
the actual PEFT runtime model class, `factor_preserving=true`, and
`external_teacher_loaded=false`. The source/freeze inventory jointly hashes **all**
base/factor/tokenizer/runner files plus the manifest and optional recovery report.

Preflight uses strictly validated base **and factor** headers; F32 factors are
charged at four bytes even for a BF16 base. Meta validation allocates no full
weights. The existing loader bound includes loaded base+factors plus their
mmap/cast transient, not a second resident full model. Generation separately charges
unresident base and adapter mappings and a conservative F32 LoRA workspace
`4*prefill_tokens*(2*hidden + 2*ffn + rank)`; resident weights are not charged again.

Build requires real requested steps, export admission, teacher independence and
passed standalone reload parity. Native mode additionally requires
`standalone_native=true`; factor mode instead requires `factor_preserving=true`
and the matching explicit export mode, **not** `standalone_native=true`.
Factor reload checks require exact forward and generation parity; they remain
engineering checks, not functional-quality evidence. Build compares **base plus
factor safetensors bytes** with the original source and charges the **whole bundle**
against the output budget (`recovered_total_bundle_bytes`). Finalization loads the
source-native, pruned-native and recovered-bundle arms under the same strict hash
freeze, one consumed marker and no training on final. The executable implementation
lock includes `standalone.py` and `controls.py` before candidate freeze.

`--report` and its `.history.json` sidecar **must be outside the output directory**;
inside-output paths are rejected before training. Staged builds already use
`<study>/recover.receipt.json` and `<study>/recover.receipt.json.history.json`, outside
`<study>/recovered/`. Engineering completion stays separate from quality; unsupported
oracle, operational failure and candidate-failure scoring statuses are unchanged.

### Native inference allocation accounting (CPU, batch one)

This corrects two accounting errors, **not a host-permission failure**: using a
fictitious 256-token prefill during loading, and charging the entire model again
against *remaining* RAM before generation. The first source-validation attempt in
`/agent/workspace/silt-specialist-study-075` rejected an estimate of
3,526,188,544 bytes against 3,109,209,293 effective bytes before model allocation.
That study's source recipe, failure receipts and implementation lock remain
immutable. This patch neither retries nor overwrites that workspace, and does not
assert that a new real-model run passed. A previously reported roughly 1.7 GB RSS
for BF16 Qwen is motivating empirical context, **not a fresh measurement produced
by this accounting patch** and not a substitute for admission.

**Dynamic limits, at every phase.** Let `A` be the minimum of current host
`MemAvailable` and every readable enclosing cgroup's remaining memory, measured by
the unchanged reconstruction `_available_ram` helper. Let `R` be current process
`VmRSS` from `/proc/self/status`, not `ru_maxrss`. The additional allocation limit
is `min(A - max(256 MiB, floor(A/10)), operator_total_cap - R)` with terms clamped
at zero and the operator term omitted when no cap was requested. Thus
`infer/evaluate --memory-budget-mib` is a **total process memory ceiling**, not a
fresh per-prompt allowance. Imported libraries, resident weights, retained
allocator pages, tokenizer and earlier results consume that ceiling. Linux RSS
or host/cgroup evidence unavailable means refusal, never guessed extra headroom.
Kernel limits and permissions are never raised, checks are never disabled, and
host reserves are not reclaimed to force a fit. The optional ceiling is an
admission policy, not an OS-enforced promise about an unbounded native allocator.

**Loader phases.** `loader_pre_import`, `loader_post_import` (after native meta
shape/tied-key validation), and `loader_pre_weights` (after tokenizer loading)
each observe fresh RSS/headroom. The additional peak estimate is:

```
loaded_native_weights + max(loaded_native_weights, source_payload) + 512 MiB
```

Safe header tensor sizes/dtypes bound source payload; target bytes include native
retained-F32 `.wo.weight` and restored F32 router handling. After the meta check,
use at least the native parameter-count bound. The full source-mmap/cast transient
is retained even with `low_cpu_mem_usage=True`: mappings/casts are not assumed free
or nonoverlapping. The 512 MiB reserve covers loader/tokenizer/runtime work, not an
unknown prompt. There are **no inference buffers based on a made-up input length**
at this stage. Safe local inventories, public-source safetensors header checks,
strict native shape matching and no-remote-code rules remain unchanged.

**Incremental generation phase.** Render the original native recovery prefix and
tokenize it exactly once without truncation. Reject context overflow rather than
shortening input or output, then use actual prefill `P` and requested generation
`G`. The helper requires explicit validated dimensions: unknown/missing geometry
is refused, not silently replaced with zero. No model weight count appears in the
incremental formula. However, checkpoint storage retained as mmap is inspected in
`/proc/self/smaps`: `Size - Rss` for this model's weight mappings is charged as
additional page-fault residency. Already resident mapped/private/COW weight pages
are not charged twice. A source mapping that no longer exists needs no future
residency allocation. File cache remaining charged to cgroups is not assumed to
have been released. This is deliberately conservative.

For Qwen causal generation, `T = P + G`; KV storage is
`2 * layers * KV_heads * head_dim * T * dtype_bytes` (K and V, **GQA KV heads**, not
query heads). Decoder concat overlap adds one layer's full K/V storage. For
T5/Switch, `T = G + 1`, and decoder cache includes `T + P` for self- plus encoder
cross-attention; two encoder hidden tensors are retained as well. Query-head
attention work still uses all query heads, including expanded GQA keys/values.

The conservative **per-layer**, no-autograd eager workspace for query length `Q`
and key length `S` is:

```
Q * (16*hidden + 8*ffn + 4*query_heads*head_dim) * 4
+ query_heads * Q * S * (2*dtype_bytes + 8)
+ Q*S*8 + Q*experts*16
```

The first term reserves FP32-sized projections, residuals, gated FFN products,
RMSNorm/native float32 computations and casts even for BF16 weights. The attention
term covers native scores, FP32 softmax work, bias and dtype casts; mask work is
separate. Seq2seq decoder attention doubles the attention term and conservatively
uses the sum of self/cross key lengths. Activations are not multiplied by depth
because autograd and hidden/attention retention are off; KV **is** multiplied by
decoder depth. These are auditable engineering upper-envelope coefficients for
the supported native eager implementations, not a universal allocator proof.

Full native causal prefill logits are charged as `P*vocab*(dtype_bytes+4)`; no
last-token-only optimization is presumed. Decode logits charge
`vocab*(dtype_bytes+4)` plus `16*vocab` for bounded live FP32 score vectors. A
fresh greedy `GenerationConfig` explicitly disables retained scores, logits,
attentions and hidden states; scores are not accumulated for all `G` steps.
Take the maximum of prefill-forward workspace and **prefill-logits overlapping
the next decoder forward**, rather than either summing all sequential stages or
forgetting the transition peak. Add full KV, concat overlap, encoder retention,
`64*(P+G+1)` for tokens/masks/traces, unfaulted mapped weights, and a separate
64 MiB incremental native/allocator workspace reserve. The independent host
safety reserve above still applies. No dtype, generation cap or prompt is changed
to make admission pass.

**Evidence and scope.** Success reports retain phase, all formula components,
actual prefill/cache lengths, current RSS and dynamic limits; failures preserve
the rejecting admission in the load error or generation trace. Successful native
runs record fresh RSS after loading and after generation, plus lifetime peak RSS
labelled as such (not an isolated per-model peak). A parent-run retry must collect
those fresh real-model observations before claiming fit. Tiny RANDOM-model tests
and allocation-only dimension fixtures verify mechanics: weights are not charged
twice, long-prefill attention grows quadratically and KV/logits linearly, custom
total caps and shrinking host headroom refuse before forward, missing geometry
refuses, and source mmap/cast/native-F32 accounting is not omitted. These tests
are not model-quality or real-source memory-capacity evidence.

There is no invented `--encoding-mode` switch. Native chat when a template exists,
flat-newline fallback otherwise, and native seq2seq are the actual existing API
encoding contracts. The recipe records `encoding="native_recovery_v1"`.

### CLI/Studio response contract

Stdout is exactly one compact JSON receipt, e.g.:

```json
{"schema_version":1,"command":"build","status":"BUILT_UNCERTIFIED","completed":true,"workspace":"/local/new-study","manifest":"/local/new-study/manifest.json","result":{"engineering_complete":true,"quality_pass":false,"certificate":false,"comparison":{},"error":null}}
```

This is a **schema illustration**, not a measured build result. Diagnostics go to
stderr. `infer` returns its native result in `receipt.result` (including
`text`, `output={type:"text",text:...}`, encoding and token trace). `evaluate` and
`validate` publish full result files and return their path, SHA-256 and task-count
summary. `build` returns the manifest path. `recover`/`reconstruct` preserve their
complete
API result in `--report`; recovery history also gets a separate `.history.json`
sidecar and stdout links to it with the row count (64 rows for 64 completed steps).
Without `--report`, the low-level command retains its full API result in stdout.
No UI changes are required by this namespace.

Exit 0 means the operation completed, **not that model quality passed**. Failed
candidate syntax/format/wrong-return tasks and supported sandbox task timeouts
are included in completed evaluation reports with exit 0; inspect
`quality_pass`, task failures, oracle status and scope. Invalid requests, failed
model loading, failed source integrity, rejected recovery, process timeout and
uncompleted builds return exit 1 with `completed=false`. Argparse usage errors
retain argparse exit 2. Files and directories are never overwritten by the public
workflow. Receipt files are private mode 0600; study directories start mode 0700.

## Strict build recipe

Paths are local and resolved relative to the recipe file, never the process cwd.
Unknown top-level or nested option keys, booleans masquerading as numbers,
nonfinite numbers, duplicate JSON keys and excessive requests are rejected.

```json
{
  "schema_version": 1,
  "source_path": "/local/Qwen2.5-Coder-0.5B-Instruct",
  "calibration": "/local/data/specialist-v1/calibration.json",
  "training": "/local/data/specialist-v1/train.json",
  "validation_data": "/local/data/specialist-v1/validation.json",
  "validation_suite": "/local/data/specialist-v1/validation-suite.json",
  "data_manifest": "/local/data/specialist-v1/manifest.json",
  "selection_lock": "/local/data/specialist-v1/selection-lock.json",
  "source_metadata": {
    "canonical_id": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "revision": "REPLACE_WITH_ACTUAL_ACQUIRED_REVISION",
    "acquisition_pr": "REPLACE_WITH_ACTUAL_ORIGINAL_ACQUISITION_REFERENCE"
  },
  "dtype": "bfloat16",
  "encoding": "native_recovery_v1",
  "seed": 17,
  "max_length": 256,
  "max_new_tokens": 256,
  "output_budget_bytes": 4294967296,
  "memory_budget_bytes": null,
  "source_quality_floor": 1.0,
  "timeout_seconds": 3600,
  "reconstruction": {"family":"auto","method":"activation","retention":0.75,"max_samples":32},
  "recovery": {"steps":64,"rank":8,"learning_rate":0.0001,"kd_weight":0.7,"method":"lora_kd","max_samples":256,"teacher_mode":"cached","export_mode":"native_merged"}
}
```

Only `source_path`, `calibration`, `training`, `validation_data` and
`validation_suite` are required recipe keys; all other settings above default as
shown, except `source_metadata` defaults empty. Data manifest/selection-lock paths
are discovered beside `training` if omitted. They **must exist**: build is the
governed specialist dataset path, not a way to silently use unknown old samples.
Low-level commands remain available for separate experiments/tiny mechanics.
Provide the **real** source identity and acquisition reference rather than leaving
example placeholders. Metadata is provenance supplied by the caller, not a
verified upstream acquisition attestation. Accepted source metadata keys are
`canonical_id`, `revision`, `acquisition_pr`, `acquisition_url`, and `license`.
Original source config and full file inventory are retained unchanged in the
manifest, alongside all provided acquisition metadata. Nothing renames/removes
an original acquisition record from the source store.

Recipe limits: timeout 1–3600 seconds; seed uint32; max_length 4–2048;
max_new_tokens 1–384 (default 256); output budget 1 byte–64 GiB; steps 1–4096;
rank 1–128; max_samples 1–4096; learning_rate `(0,0.1]` (recipe minimum 1e-12);
kd_weight `[0,1]`; retention strictly `(0,1)`; source-quality floor `[0,1]`;
optional memory ceiling 1 through `2**63-1` bytes. The recovered **whole inventory**
must fit the output budget, while its actual safetensors bytes must also be less
than the source's safetensors bytes. Retention is an MLP-channel request for Qwen,
not a total-model reduction claim. Switch retains one native expert per sparse
layer, independently of the recorded retention request.

### Actual stages and immutable evidence

1. Hash the complete safe source inventory and preserve native config and source
   acquisition metadata. Reuse existing artifact safety rules (no symlinks,
   traversal, pickle, remote custom code, adapter/base indirection or escaping
   tokenizer/shard dependencies).
2. Read permitted calibration/TRAIN/dev files and the answer-free selection lock.
   Verify the manifest's exact hashes for **only those permitted files**. Require
   each file in the dataset manifest's directory under its designated split name.
   Compare entire split ID/family sets, metadata fingerprints, prior consumed
   IDs/families, TRAIN-only exact calibration subset and exact validation prompt
   equality. Validate source tokenizer pins. Copy final **metadata hashes and IDs**
   only; do not open or hash `final.json`, `final-suite.json` or final provenance.
3. Evaluate the **source first** on the whole dev function suite (8 tasks in the
   supplied specialist-v1 study: 6 targets, 2 controls). This is an observed source
   capability baseline, not an assumption the teacher can code correctly.
4. Run real native reconstruction. Require its actual successful result, then
   evaluate the reconstructed native model on the same dev suite.
5. Run real recovery with the requested teacher, student and disjoint train/dev
   files. Require `status=completed`, `artifact_admitted=true`, and the requested
   actual optimizer-step count. Preserve failures without claiming an output.
6. Evaluate recovered weights on dev, compare all three observed task pass rates,
   verify source and pre-recovery control hashes, verify output size, and freeze
   the complete source/pruned/repaired inventories.

Each heavyweight stage is a **fresh process** invoking the same fixed public CLI
operations (`reconstruct`, `recover`, `evaluate`) through the trusted absolute
`specialist/stage_worker.py` launcher. The actual vector begins
`[sys.executable, '-I', '-S', fixed_worker_path, expected_parent_pid]`; the remaining
arguments are only public specialist operation/options. No shell, recipe-supplied
executable/module, environment override or import-path override is supported.
Before **any site or ML import**, the Linux bootstrap installs and reads back
`PR_SET_PDEATHSIG(SIGKILL)`, checks the expected parent PID captured before spawn
(to close the parent-exited-before-prctl race), and verifies its own session and
process-group identity. The public CLI is imported only afterward from the fixed
installation; dependency directories are derived from the interpreter installation,
not caller paths. `.pth`, `sitecustomize`, and user-site hooks are not executed;
bytecode writes are disabled. Existing kernel resource limits are inherited
unchanged; this launcher does not invent a larger RAM allowance or replace the
optional memory admission policy. Child ML thread environment defaults to one CPU
thread. Linux main-thread supervision is required; unavailable contracts fail closed.

Stdout+stderr capture is capped at **1 MiB combined**; JSON evidence files have a
separate 16 MiB bound. Child pipes are drained with selectors. On timeout, output
overflow, `BaseException`, or managed SIGTERM/SIGINT, the parent kills the known
stage process group **before reaping its leader**, avoiding a PID-reuse kill race.
Even successful leaders retain their identity until same-group leftovers are
killed. The shared remaining deadline covers capture and bounded reap (a small
cleanup reserve is taken from that budget, never a fresh hour per stage).
A descendant retaining inherited pipes cannot extend that deadline.

If the build CLI parent is killed with SIGKILL, the kernel kills the stage leader
without depending on parent cleanup code. This is a **known-process-group lifecycle
contract**, not containment of arbitrary child processes: parent-death signals do
not propagate through arbitrary forks, and children that deliberately escape the
known group are not claimed to be cleaned up. Parent integrity hashing/publication
is synchronous; stalled filesystem I/O or uninterruptible kernel waits are not a
kernel-level whole-process SLA. External service/cgroup supervision is appropriate
when stronger process-tree containment or a hard wall-clock bound is required.

`<stage>.started.json` records `RUNNING`, logical public CLI argv/hash and fixed
launcher path/flags; `<stage>.result.json` records actual `COMPLETED`, `BLOCKED`, or
`REJECTED`, return code and receipt, worker PID and known process group. Bounded raw
stdout/stderr are separate `<stage>.logs.json` files. Timeout/output-overflow and
managed-interruption evidence retains the actual post-reap return code (or null
if the bounded reap could not finish), partial logs, byte counts/hashes, elapsed
time, capture bound, and any parsed receipt. An interrupted training receipt is
never admitted as a successful stage, even if it printed successful JSON before
interruption. Managed interruptions record `interrupted=true`, block the stage
and build, and stop later training/evaluation and candidate freezing. Separate
started/result files make history append-only, not a replaced optimistic status.
Uncatchable SIGKILL or interruption during publication can still leave a started
record without a terminal record; this is **not** completion. `preflight.json`, `recipe.json`, all
three full evaluation files and terminal `manifest.json` persist. A rejected
manifest does not erase an earlier reconstruction or failed recovery receipt.
The workflow never resumes into/overwrites an existing study directory.

Build status is `BUILT_UNCERTIFIED`, `BLOCKED` (unavailable execution), or
`REJECTED` (invalid/integrity/recovery admission failure).
`engineering_complete` means the requested study operations completed, while
`quality_pass` means the recovered model passed **all observed suite tasks**.
Neither implies statistical noninferiority, a universal coding certificate,
power≥0.98, safety, automatic promotion, or suitability for unseen domains.
Observed retention ratio is null if the source passed zero tasks; no fabricated
ratio or division-by-zero score is used. A few-case pilot is reported as
`insufficient_evidence_no_power98_gate`, not silently promoted or denied the
ability to run a real build because a large powered study is absent.

### Source qualification, output roles, and source lock

`source-validation.json` is a required **observed** baseline on the complete dev
suite, not an assumed property of a checkpoint name. The frozen recipe's
`source_quality_floor` is in `[0,1]` (default 1.0); `qualified_source` means only
that this observed validation pass rate met that declared floor. It is never a
teacher certificate. A below-floor source may continue real reconstruction and
recovery as **RESEARCH**, with `qualified_source=false` and `certificate=false`.
Missing powered evidence or an incomplete theoretical domain claim alone does
not skip real training. An unavailable source model/oracle blocks the stage and
stops subsequent heavy operations; it does not establish a low source score.

The manifest separately names source control, reconstructed control, recovered
candidate, and development validation output roles. `candidate_frozen` is set
only after all real model stages, optimizer-step evidence, integrity checks and
size checks pass. `implementation-lock.json` records exact source hashes for the
specialist (including `stage_worker.py`), artifact and oracle code plus actual API signatures **before the first
model stage**. Each stage checks the frozen implementation; finalization refuses
changed code before consuming final. Do not start heavy runs while sibling code
repairs are still in progress. Freezing local hashes is reproducibility evidence,
not a signature against a malicious filesystem owner.

Source-calibration-only B data, previously used cohorts (including any known
source-filter cohort), and final answers are not new TRAIN/dev material. A fresh
training dataset needs its own valid manifest, source filters, and selection lock
listing all prior consumed IDs/families. `data_preflight` rejects those consumed
IDs/families, cross-split families/content, changed hashes, and arbitrary split
path aliases. Only the exact designated `train.json`, TRAIN-only
`calibration.json`, `validation.json`, and `validation-suite.json` in the locked
manifest directory may enter build. A path with a `final*` or `source_calibration_only*` component (hyphens normalize
to underscores) cannot be passed in their place. A manifest explicitly labeled
`source_calibration_only=true` or `usage="source_calibration_only"` is refused for
training. Metadata/provenance supplied by a caller is not an upstream
attestation; the implementation cannot discover undeclared historical use.
Source-only reference calibration must remain separately labeled and must never
be merged into training or selection after inspecting final outputs.

## Native inference and evaluation contract

Only native Qwen2 causal models and native Switch/T5 encoder-decoder models are
supported. Local safetensors/config headers are checked against the native model
on the meta device, then the actual checkpoint is loaded with strict missing,
unexpected and mismatched-weight checks. No random replacement tensors or
pretrained downloads are allowed. T5/Switch support is an architectural mechanics
path, **not evidence that those source checkpoints have coding capability**.

- Qwen with chat template: exactly
  `apply_chat_template([{role:'user',content:prompt}], tokenize=False,
  add_generation_prompt=True)` followed by tokenizer `add_special_tokens=False`,
  `truncation=False`. No extra system message, Compose format-contract prefix,
  rewritten task, reference answer, family label or host expected value.
- Causal tokenizer without chat template: exact `prompt + '\n'`, no extra special
  tokens. This matches recovery's explicit flat fallback.
- Switch/T5: full prompt with native `add_special_tokens=True`, no causal chat or
  newline prefix; native decoder-start handling. Decoder start is separately
  retained in `native_sequence_token_ids`, not counted as generated output.

A fresh native GenerationConfig requests greedy decoding (`do_sample=false`, one
beam and one return sequence), explicit new-token cap, use_cache=true, no forced
EOS/BOS or repetition penalty. Full forwarded config, dtype, exact input token
IDs, native output IDs, generated-only IDs, EOS positions, cap status, stop reason,
encoding/template hash, prefill hash and elapsed generation time are retained.
The MediaIR text output is `{type:'text',text:...}`; trace metadata is separate.
Decode only generated causal IDs, not the user prompt. Reaching the token cap
without EOS is a **truncation failure**, not a successful partial function. EOS on
the final allowed token is a real EOS completion, not automatically a truncation.
Unknown termination also fails task validation.

Admission is not a truncation policy. Causal `input_tokens + max_new_tokens` must
fit the native context bound; seq2seq encoder input and decoder start+generation
are separately bounded. The workflow has an additional 2048-token ceiling.
A 256-token training admission bound is **not** a 256-token generation context
limit. Specialist-v1 responses were admitted at <=192 response tokens, and inputs
can exceed 128 tokens; the default generation budget remains 256. Oversized
inputs are rejected intact, never silently cropped. Existing conservative model
memory guards are reused, with an actual generation-length recheck before forward.

`infer` frees its model on exit. `evaluate` keeps only one model resident for the
suite, then frees it; build uses fresh child processes for every model stage.
Python/allocator cache release does not guarantee RSS reclamation. Resource
reports explicitly label `ru_maxrss` as **process-lifetime high-water, not isolated
per-model memory**. They do not claim a measured aggregate-memory guarantee.

### FunctionSuite and oracle boundaries

The suite is the existing `asea.compose.schema.EvaluationSuite`:
`claims=['coding']`, all cases `metric='function_io'`, text `input` only,
`threshold=1.0`, both target and control groups. No new incompatible suite schema
is introduced. This namespace does not allow legacy unittest strings, fixtures,
proxy text metrics or multimodal data to be reported as functional code quality.

`evaluate` generates every task. `validate` accepts an array of records with
`id`, `status`, `text`, and the native `generation` trace (e.g. take each
`cases[*].generation` from a prior full evaluation). Duplicate/unknown IDs are
invalid input; offline missing/failed records, observed token-cap/unknown-stop
outputs, and unparseable task outputs are
**failed tasks in the original denominator**, not skipped tests. Preserve any
provided token/config evidence; externally supplied trace metadata is not a
cryptographic attestation of how a model generated the text.

Source extraction uses the **existing** certification source extractor; bounded
AST parsing identifies invalid syntax without executing candidate code. Candidate
code never runs in the host interpreter. `evaluate_functions` performs external
sandboxed function calls; `_function_passed` applies the **unchanged host verdict
rules**. The result retains the complete host oracle record. A return value,
`passed` string in generated code, replay or untrusted candidate diagnostic never
sets the workflow verdict. Unavailable isolation/profile/dependency or model
load/runtime failure is **BLOCKED execution**, not a model-quality-zero grade or a
fallback to host execution. A trusted cheap oracle containment preflight runs
before allocating the model; every real oracle call still performs its mandatory
fresh per-execution preflight.

Aggregate accounting is explicit:

- `tasks_total` always includes every declared problem.
- `tasks_graded = tasks_passed + tasks_failed`; syntax, malformed source, wrong
  return, observed truncation, and supported candidate timeout/resource-limit
  outcomes remain complete grades.
- `tasks_blocked = operational_failures`; these are unavailable model/oracle
  executions, not failed quality grades. `tasks_graded + tasks_blocked = tasks_total`.
- `generation_completed` and `oracle_executed` report distinct observed coverage.
  Per-case `grade_complete`, `generation_completed`, `oracle_executed`,
  `operational_failure`, and `failure_stage` preserve the distinction.
- If any task is blocked: `completed=false`, `engineering_complete=false`,
  `quality_pass=false`, `status=BLOCKED`, and `pass_rate=null` (not invented zero).
  `observed_pass_fraction` retains passed/declared-total as partial coverage only.
  A completed all-real-runs evaluation/build may have `quality_pass=false`.

The parent validates full evaluation-file coverage and counts as well as the CLI
receipt. An exit-zero receipt cannot override a blocked or incomplete result file.

Trace policy is explicit `none|digest|value`, with **digest the workflow default**
(the oracle's own default is unchanged). Current-value host return traces are
preserved when `value` is explicitly requested. These are diagnostic observations,
not authority; host comparisons and established isolation determine task passes.
Digest traces can expose low-entropy known values to guessing. Do not assume a
digest provides confidentiality; value traces can contain sensitive data and must
not be broadly shared. Build/finalize always request digest for the known public
short-function dataset.

## One-shot finalization, not final-driven selection

`finalize` is separate. It requires a `BUILT_UNCERTIFIED` manifest, frozen candidate
flag, matching frozen-map hash, exact current source/pruned/repaired full-file
inventories, and the frozen executable implementation/API lock. It accepts only the preregistered final-suite path. It writes and
fsyncs **exclusive `final-consumed.json` before opening/hashing/loading the final
suite or making any final evaluation call**. Failures after consumption retain the
marker and a rejected final report. A second call is refused; no retry/reset API
is provided. Deleting markers or editing studies outside the workflow voids this
process evidence; it is not protection against a malicious filesystem owner.

Only after consumption does it verify final-suite hash and IDs and evaluate the
three frozen controls in separate child processes. There is no final optimizer,
channel selection, repair, prompt retuning or automatic promotion. Final results
remain `BUILT_UNCERTIFIED`, with scoped observations and insufficient statistical
evidence statements even if every exercised case passes.

## Tests and evidence scope

```sh
PYTHONPATH=/agent/workspace/silt-specialist-core/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /agent/workspace/silt-venv/bin/python -m pytest -q \
 /agent/workspace/silt-specialist-core/tests/test_specialist_workflow.py
```

Tests cover strict recipe defaults and rejection, exclusive reports, supplied
TRAIN/dev metadata preflight without final reads, original task denominators,
non-authoritative returned values, actual stage status/order, preserved failed
recovery, frozen controls, once-consumed final marker ordering, fixed public child
rejection, and a tiny **random** Qwen native reconstruction → real one-step recovery
→ standalone native-chat inference. Tiny original orchard/harbor text records are
mechanics fixtures, not pretrained coding-quality evidence. No actual pretrained
build or final benchmark is run as part of these implementation checks.

### Lightweight audit regression command

The build-gate repair was checked without loading/training a native model:

```sh
PYTHONPATH=/agent/workspace/silt-specialist-core/src \
 /agent/workspace/silt-venv/bin/python -m pytest -q \
 /agent/workspace/silt-specialist-core/tests/test_specialist_workflow.py -k 'not tiny_random'
```

The excluded tiny-random integration test is a separate mechanics check. The
lightweight set exercises candidate-vs-infrastructure outcomes, actual bounded
child termination, memory/API wiring, shared headroom admission, immutable
full-grade build gates, history sidecars and final-consumption ordering. These
are engineering regression checks, not evidence of trained network capability.
