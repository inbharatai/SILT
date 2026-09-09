# Native student recovery and factor-preserving export (experimental, local-only)

`asea.specialist.recovery.recover` performs **real teacher-guided training**, not a
simulated score update. It operates on an **already reconstructed native student**;
it does not choose pruning channels, perform reconstruction, alter existing
`deepapply` trainers, or decide promotion gates. No model downloads occur.

## API

```python
from asea.specialist.recovery import recover

report = recover(
    teacher_dir, student_dir, output_dir, training_path, validation_path,
    steps=64,
    learning_rate=0.0001,
    rank=8,
    max_length=256,
    dtype="bfloat16",
    seed=17,
    kd_weight=0.7,
    method="lora_kd",
    max_samples=256,
    memory_budget_bytes=None,  # bounded by observed CPU RAM minus reserve
    teacher_mode="cached",    # cached (default) | resident
    export_mode="native_merged",  # default, unchanged failure-closed semantics
    # export_mode="factor_preserving",  # explicit alternative; NEVER merges
)
if report["status"] != "completed":
    raise RuntimeError(report["error"])
```

All paths are local. Output must be a **new, non-overlapping path** whose parent
already exists. Input stores are never written to. The source teacher is frozen,
`eval()` and `no_grad()`; only student LoRA adapter parameters enter AdamW.
Before **any Auto config/tokenizer loader**, both original path spellings pass
`safe_path`/`safe_file`, full `model_inventory`, bounded JSON/config/header checks,
and native meta-model exact keys/shapes/tied-alias checks shared with reconstruction.
Both teacher and student reject PEFT/base indirection, symlinks/traversal, unsafe
files, tokenizer/model `auto_map` (including empty maps), quantization/compression,
future/incompatible-major config versions, extra/unindexed/native-unmatched keys,
and missing or shape-mismatched weights. There is no random-weight replacement.
The same policy checks the staged export before publication.

Supported config `model_type` pairs:

| Teacher | Reconstructed student | PEFT targets |
|---|---|---|
| `qwen2` | `qwen2` (including a smaller native MLP width) | `q_proj`, `v_proj`, `down_proj` |
| `switch_transformers` | `t5` (dense reconstructed student) | `q`, `v`, `wo` |
| `t5` | `t5` (including an off-the-shelf baseline) | `q`, `v`, `wo` |

Other pairs fail closed. This is not a generic arbitrary-architecture trainer.
Changed MLP widths must already be represented by native Linear modules and a
loadable native config. Widths do not prevent LoRA injection. No custom remote
model code, quantized adapters, pickle `.bin` checkpoints or external shard paths
are supported. Safetensors may be single-file or indexed shards. Both tokenizers
must have **identical token-to-ID vocabularies and EOS/pad/BOS IDs**, and both logit
vocabulary widths must match. Seq2seq decoder-start IDs must also match. Selected
chat-template text must match, and actual complete teacher/student encodings are
compared on **every** train/dev row (matching vocabularies alone is insufficient).

## Data and response-only objective

Both JSON files use this schema:

```json
{"samples": [{"id": "unique-train-id", "prompt": "Task text", "response": "Licensed reference answer"}]}
```

Supply ground-truth responses you have permission to use. The trainer does **not**
generate synthetic labels, relabel dev as training, or trim, truncate, normalize
terminal markers, or rewrite the reference text. Native tokenizer normalization
is still part of the tokenizer's contract (particularly T5); input text is not
preprocessed by recovery. Extra provenance/schema fields are allowed.

Before JSON object construction, each file is bounded to **16 MiB**, arrays to
**10000 entries**, and nesting to 32 levels. A bounded binary read also guards a
growing-file race. Duplicate JSON keys at any depth and nonfinite values
(including overflowing exponents) are rejected. Each split needs 1–10000 samples;
IDs must be nonempty, unpadded, control-character-free strings of at most 256
characters. Prompt/response must be nonblank strings, each at most 65536
characters. This does not authorize multi-GB JSON allocation.

Within each file, duplicate IDs and exact prompt/response pairs are rejected.
Across **entire files, before subsampling or weight loading**, IDs, exact pair
hashes, and complete tokenized examples must be disjoint. When present, row
`family`, `family_id`, and `task_family` values must be nonempty unpadded strings;
no value may occur in both splits. Recovery does not discover adjacent manifests
or open quarantined final data: **the parent validates manifest/file hashes,
family-disjoint selection locks, final quarantine, and provenance**. Row family
checks are additional enforcement, not a claim to detect all semantic leakage.

Every train/dev row is completely encoded and length-checked before **any weight
load, optimizer creation, or gradient**, including rows that will not be selected.
Only afterward a local `random.Random(seed)` shuffles train then dev and chooses
at most `max_samples` per split. Training cycles through train only; dev is
`eval()`/`no_grad()` before/after training and never enters the optimizer.

### Encoding and inference contract (versioned in `report.encoding`)

**`native_chat_response_only_v1` — causal Qwen with a chat template:**

1. Use the loaded tokenizer's actual `get_chat_template()`; do not replace it
   with a handwritten Qwen approximation or insert an extra system message.
2. Render `prefix = apply_chat_template([{"role":"user","content":prompt}],
   tokenize=False, add_generation_prompt=True)`.
3. Render `full` with the same user message plus
   `{"role":"assistant","content":response}`, `tokenize=False`, and
   `add_generation_prompt=False`. It must equal
   **`prefix + response + eos_token + suffix`**, where the empty-assistant render
   establishes the suffix and only `""`, `"\n"`, or `"\r\n"` are supported.
   Content-rewriting, unsupported role caps, and prefix-incompatible templates
   receive a named incompatibility rejection.
4. Tokenize the **complete rendered full text once**, `add_special_tokens=False`,
   `truncation=False`, `return_offsets_mapping=True`. Independently encode the
   inference prefix and require its IDs equal the corresponding full-ID prefix.
5. Positional labels supervise only tokens wholly inside response content plus
   **one** mapped terminal EOS token. All prompt/system/user/assistant role caps
   and any post-EOS template newline are `-100`. A newline belonging to the
   original response **is** supervised. EOS/pad sharing never suppresses EOS.
6. A token spanning a prompt/response/EOS/suffix boundary cannot simultaneously
   preserve prompt masking and every response character. **Reject it explicitly**
   rather than mask it and silently lose the first response character. This
   includes whitespace/BPE merges across the generation-prefix boundary. No
   approved example has a masked boundary token.
7. Require the complete full-ID length **including masked template suffix** to
   be `<= max_length`; otherwise reject, never crop. No BOS or extra EOS is added
   outside the rendered native template.

**Parent inference must use exactly the same user-chat prefix:**

```python
messages = [{"role": "user", "content": prompt}]  # prompt is not preformatted chat
prefix = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(prefix, add_special_tokens=False, truncation=False,
                   return_tensors="pt", return_token_type_ids=False)
# Validate complete prefill length against the deployment limit; never crop.
# Generate from these IDs; decode only IDs after inputs["input_ids"].shape[-1].
# Choose an explicit max_new_tokens budget large enough for a COMPLETE response
# plus EOS; max_length=256 here is a training admission bound, not a generation
# new-token budget. Do not append the reference answer during inference.
```

**`flat_newline_response_only_v1` — tokenizer without a chat template:**
full text is literally `prompt + "\n" + response + eos_token`; inference prefill is
`prompt + "\n"`. Encode full jointly without special-token insertion/truncation;
apply the same offsets/prefix-stability/mask rules. Encode-only test doubles or
slow flat tokenizers can use a strict fallback: independently verify that the
complete joint IDs equal prefix IDs + response IDs + one EOS, then use the
**joint** IDs, never substitute concatenated encodings. An unstable boundary is
rejected. This fallback is explicitly reported, not mislabeled as native chat.

**`native_seq2seq_v1` — native Switch/T5:** independently encode the full prompt
and full response with `add_special_tokens=True, truncation=False`; do not append
manual EOS or use the causal chat template. Native encoding must supply prompt
EOS and exactly one terminal response EOS. Both complete sequences must be
`<= max_length`. Labels are response IDs only; both models receive identical
native `_shift_right(labels)` decoder inputs using the checked
`model.config.decoder_start_token_id`. Parent inference encodes the full prompt
with `add_special_tokens=True, truncation=False` and lets native generation
handle the decoder start. It must not add a causal newline/chat prefix.

All masks are positional, not based on pad-token equality. Causal logits/labels
are shifted by one token. Batch size one avoids dynamic padding; the
response-logit helper respects `-100` padding labels.

`report.encoding` records the exact template text and its UTF-8 SHA-256, format,
boundary policy, and no-truncation flag. `data.encoding_audit.training/validation`
covers **all** input rows: reference character/UTF-8/token counts, reference hash,
complete input and label counts/hashes, supervised token count, one terminal EOS,
mask hash, rendered-text hash, and effective prefill text/ID hashes and token
count. Structured hashes use SHA-256 of compact sorted-key UTF-8 JSON
(`ensure_ascii=False`); the mask is a Boolean array (`label != -100`). The report
also includes maximum complete lengths, selected IDs and the effective RNG seed.

With `T=1`, the loss is:

```
(1 - kd_weight) * response_cross_entropy
+ kd_weight * KL(teacher_distribution || student_distribution)
```

KL is the **exact full-vocabulary forward KL**, normalized across response target
tokens. Vocabulary chunks of 4096 compute the contribution using full log-sum-exp
normalizers. Checkpointed chunk recomputation avoids retaining every probability
chunk for backward. This is **not top-k distillation**. Both models still produce
native full logits; chunking is not a claim that the logits allocation disappears.
CE is always computed and reported, even when its objective coefficient is zero.

Available controls/baselines:

- `method="lora_kd", kd_weight=0.7`: combined supervised + teacher-guided recovery.
- `method="supervised_lora", kd_weight=0`: supervised LoRA baseline. No teacher
  model is loaded or forwarded; teacher metadata/vocabulary/store validation still
  occurs because this API retains a mandatory teacher path.
- `method="lora_kd", kd_weight=0`: also skips all teacher forwards.
- `method="lora_kd", kd_weight=1`: distillation-only objective. For an
  off-the-shelf-student control, pass that independently prepared native student
  store and the same permitted parent training/dev data. This API does not acquire
  or construct the control automatically.

There is no novelty or universal recovery claim. Loss metrics are **not code
quality, functional correctness, or evidence of retained general capability**.

## Memory, execution, and real-training evidence

Recovery is **explicit CPU only**, not automatically CUDA and not a GPU validation
claim. The optional `memory_budget_bytes=None` automatically uses observed host/
cgroup available RAM minus `max(256 MiB, 10% available)` headroom. A positive integer
sets an additional operator ceiling; the effective limit is the smaller of that
ceiling and observed headroom. Raising a caller budget never changes/bypasses host
restrictions. Both cgroup layouts and visible process ancestors are checked. No
bounded observation means rejection. Pass `4 * 1024**3` for an explicit conservative
4 GiB operator cap; **None is not a hard model-size ceiling**. Larger supported
models require genuinely larger verified host headroom, not a hardware promise.

### Cached teacher (default, `teacher_mode="cached"`)

1. Validate full TRAIN and validation encodings, masks, source artifacts, CPU phase
   estimates, and disk bounds **before any real weight load**.
2. Load only the frozen, `eval()` source teacher using validated native meta
   parameters and per-tensor safetensors assignment. Normalize **all encoder and
   decoder Switch router classifiers (weight/bias)** to native `router_dtype`
   **before the freeze/cache hash**: retain original F32 bits or exactly upcast
   source BF16 values. Native first-forward `_cast_classifier` then changes no
   parameter bytes/dtypes; the full dtype-aware hash remains mandatory.
   Switch `wo` is **not** a T5 F32 exception.
   The shared bounded router verification/restoration helper still runs **before
   logits** and reports exact tensor hashes/files and originally non-F32 routers.
   F32 values never pass through a rounded BF16 intermediate.
3. In one teacher-only phase, forward every selected TRAIN **and** validation row
   with precisely the same native IDs, shifted decoder inputs and positional mask
   used by student training. Store **all vocabulary logits at supervised response
   positions** (including positional terminal EOS), never top-k or prompt logits.
4. Each row is a safetensors file in a private `.teacher-bank` under the owned output
   staging directory. F32 storage losslessly preserves native BF16/F32 logit values;
   original logit dtype is declared. Metadata binds the complete encoded example,
   dataset hashes, encoding-contract hash, and full model-file/loaded-parameter
   hashes. Private bank writes are fsynced and receive best-effort clean-file-cache
   eviction advice; one-row reads clone just that row and release its file cache.
   This prevents the entire disk bank unnecessarily remaining charged as file cache
   on small cgroups. Unsupported/ineffective advice never bypasses RAM admission. The report contains each row's shape, actual tensor/file bytes and full
   file SHA-256 plus bank totals and its explicit temporary path.
5. Verify teacher parameters and all source files unchanged, `requires_grad=False`
   and no teacher gradients. **Delete/unload the teacher and collect/release CPU
   memory before loading the student.** Reobserve student-phase RAM headroom.
6. Each student loss reads only its one verified row; full file hash, expected
   shape/dtype and in-memory provenance must match before use. There is no API to
   adopt an external bank or resume from an unverified manifest. Exact full-vocab
   KL is unchanged, with the same floating values as resident mode.
7. Delete the bank before native export. Failure removes the entire owned stage,
   including partial cache. No teacher, bank, adapter, or source directory is
   needed for deployment; provenance hashes are not serving dependencies.

`teacher_mode="resident"` is an optional, separately guarded comparison mode:
teacher and student coexist, the teacher forwards each training/validation loss,
and the memory forecast includes both models. Supervised-only controls load no
teacher and create no bank in either mode. Teacher-forward counters report actual
forwards: cached counts are selected train rows / selected validation rows, not
training steps / repeated validation passes.

### CPU phase and disk estimates

Teacher and student each have an independent, header-derived **executed loader
plan**, not a family-wide dtype guess. Known native classes are built with empty
parameters and CPU nonpersistent buffers (including Qwen rotary buffers); exact
keys/shapes/tied aliases are checked before any checkpoint tensor is read. The
shared reconstruction loader assigns one tensor at a time to every proven alias.
At BF16, T5 keeps its native `wo` matrices F32; Switch has no `wo` F32 exception
and loads every native router classifier at configured `router_dtype` (normally
F32). Supported router dtypes are `float32`/`bfloat16`; other settings, non-F32/BF16
router sources, and F32-source→BF16-router rounding reject. The plan records each
router's source/runtime dtype. This is native initialization, not optimization or
a config override. Qwen has no native F32 exception; its policy is unchanged.

For each model, `resources.teacher/student` records source/loaded dtype groups,
unique loaded bytes, shared mmap payload, cast targets, all cast-source pages,
per-tensor cast plans, and mapping overhead. Identical-dtype CPU `.to` aliases
safetensors storage: it does not allocate a second full shard. Load peak is
`loaded_bytes + cast_source_pages_bytes + largest_cast_tensor_scratch_bytes +
mapping_overhead_bytes`. Scratch conservatively includes the largest converted
source-plus-target tensor; overhead includes file header/alignment bytes plus
8 KiB per shard. **All** cast-source payload pages remain reserved even if some
could be reclaimed. Switch source-F32 routers share mmap storage; source-BF16
routers configured F32 charge their exact cast target, source pages and one-tensor
scratch. Neither case falsely charges F32 expert `wo` or another whole shard.
This is verified loader accounting, not an assumed cache eviction.

Let `L` be the **actual maximum complete selected sequence length**, `R` maximum
supervised response tokens, `V` full vocabulary size, and `b` requested dtype
bytes. All rows are checked against `max_length` first; no cropping or padding
is used to change admission.

- Teacher-only phase: teacher load peak +
  `V * (L * max(b,4) + R * (b+4))` logits/save workspace + activation allowance.
- Cached student phase: student load peak + 16 bytes per LoRA parameter
  (F32 weights, gradients, two Adam moments) + 4-byte adapter snapshot +
  `L*V*b + R*V*(b+24)` full-vocab CE/KL/gradient workspace + activation allowance.
- Native merged mode additionally reserves eight bytes per element of the largest
  adapted matrix and copy-on-write source pages for **all** targeted base matrices.
  Explicit factor mode does not merge and does not charge merge scratch/COW.
- Resident mode adds the full teacher load peak **and teacher forward workspace**
  to the student phase, since it can coexist with the student's graph.
- **Native-merged export is unchanged:** conservative two-student-copy reload,
  largest-tensor serialization scratch and probe/factor allowances remain strict.
- **Factor-preserving export executes `native_cpu_contiguous_storage_stream_v1`:**
  write peak is the admitted student load footprint plus adapter/snapshot allowance
  and bounded I/O, metadata, tokenizer and complete-forward/generation probe
  workspace. There is no second base serialization buffer or embedding-to-F32
  conversion. Reload reserves **one** native same-dtype base, conservative shard
  header/alignment overhead, F32 factor initialization/load workspace, tokenizer
  reload and the full probe workspace. It is permitted only after weak references
  prove the training model, all its base/factor parameters, optimizer and any
  resident teacher model/parameters were collected. Unexpected references reject
  before the fresh loader; they never receive assumed free-memory credit.
- Factor I/O/metadata allowance is 64 MiB plus two 1 MiB I/O chunks, eight times
  permitted tokenizer asset bytes and 8 KiB per native tensor descriptor. The
  complete-forward probe reserves activation space, six full-vocabulary buffers
  of `(L+2)*V*max(b,4)` bytes (including retained before/after logits and F32 error
  diagnostics), plus short-generation KV cache. Full logits—not just the response
  slice—are counted. The live CPU no-grad forward result is retained without a
  redundant full-logit clone.
- Overall admission is the maximum of teacher, student, export-write and reload
  phase peaks. `resources.export_*` records the executed strategy and each bound.
  Additional-allocation guards freshly observe available RAM at teacher load,
  cache forwards, student load/training, export-write, export-reload and the
  factor-reload forward. `resources.release_checks` records collection evidence.
  Existing allocations remain charged in observed usage; **no hypothetical
  reclamation is subtracted**, including teacher bank/mmap pages or allocator
  arenas. After verified collection, clean source-weight cache advice is retried
  and free libc heap pages are returned where supported; the subsequent dynamic
  check remains authoritative. Collected objects do not establish that their RSS
  has been reclaimed.
- Activations retain `max(128 MiB, L*hidden*layers*b*12)`. Existing host/cgroup
  headroom and caller ceilings are unchanged; no bypass or reserve reduction.
- Bank tensor bound: `sum(selected supervised tokens) * V * 4`. Disk admission
  adds 16 KiB per row plus 1 MiB manifest allowance, twice native student bytes,
  64 MiB serialization headroom, and eight bytes per factor parameter in factor
  mode. Factor mode additionally charges the actual permitted tokenizer asset
  bytes and 2 KiB per tensor for output metadata. Actual bank sizes must stay below
  bounds. No GPU staging is assumed.

These are estimates, not peak-RSS measurements or a guarantee that 4 GiB fits.
They exclude already resident interpreter/library memory because observed *free*
RAM already accounts for it. Other workers, allocator fragmentation, page cache,
and native backend scratch still matter. Each phase reports assumptions and
requested/effective limits. A caught OOM rejects; OS kills need parent process
supervision and cannot produce a Python success receipt.

**Historical tokenizer/header-only projection (not a real-weight training run;
pre-explicit-loader accounting, not a current admission quote):** the permitted
specialist-v1 Qwen TRAIN/validation files contain 42/8 rows; complete native-chat
maxima are 220/169 tokens, maximum supervised count 150, total supervised count
2216. Retention 0.75 projects 415586176 student parameters and rank-8 q/v/down
LoRA has 1413120 parameters. With BF16 and all 50 rows selected:

| Quantity | Bytes | GiB (approx.) |
|---|---:|---:|
| Teacher-only phase | 1392729344 | 1.297 |
| Cached student phase / overall peak | 1679203584 | 1.564 |
| Resident-mode overall peak | 2667269120 | 2.484 |
| Full-F32 response bank tensor bound | 1346760704 | 1.254 |
| Bank-inclusive output disk admission | 3078082048 | 2.867 |

These historical projections superseded the old simultaneous-teacher/student,
configured-L256 3.134 GiB formula. Use the current report's per-source loader and
phase estimates instead for admission. Neither projection proves real
training RSS or functional quality.
A tokenizer/header-only check observed reconstruction refusal when its 2948355584
byte estimate exceeded then-current effective headroom; no limit was overridden
and no real source tensors were loaded or trained in that check. Source-integrity
hashing also now releases clean read-file cache where supported; future admission
still uses a fresh observation, never treats cache advice as a fit guarantee.

PEFT adapters remain F32. AdamW is non-foreach, adapters-only, zero weight decay,
with gradient clipping at norm 1.0. Student checkpointing uses non-reentrant
recomputation and `use_cache=False`; original deployment cache is restored.

Hard request limits: `steps` 1–4096, `rank` 1–128, `max_length` 4–2048,
`max_samples` 1–4096 per split, input files at most 16 MiB/10000 records each,
`learning_rate` in (0, 0.1], `kd_weight` in [0, 1], uint32 seed. There is a one-day
wall-clock budget checked before every forward/optimizer iteration. This is a
**per-call** limit, not a global daily quota; a hung native kernel cannot be
interrupted by those checks, so use a parent process timeout for a strict SLA.
There is no infinite optimization, external scheduler, or resumable training checkpoint. The private logit bank is ephemeral and not a resume mechanism.

Every accepted step executes `backward()` and `optimizer.step()`. Nonfinite loss,
nonfinite parameters/gradients, missing or zero total gradient norm, and zero
adapter delta are rejected. Reports include actual step counts, sampled training
IDs, incrementally retained per-step loss history (also on caught mid-loop failure), pre-clip gradient norm, adapter SHA-256 before/after
and L2 parameter delta. Frozen student parameters are hashed before/after and
must remain bit-identical before merge. Teacher `requires_grad=False`/absent
parameter gradients are verified. Every input-store file is hashed before and
after training. After merge, adapted native matrices must have changed hashes;
if BF16 rounding erased the entire update, export is rejected rather than claiming
successful recovery.

## Export and report contract

`export_mode` is an explicit keyword-only choice: `"native_merged"` (default) or
`"factor_preserving"`. Unknown modes reject before training. There is **no silent
fallback** and no tolerance override. Both modes exclusively publish a new sibling
staging directory, preserve source stores, and delete staging on rejection.

### Native merged mode (unchanged default)

PEFT `merge_and_unload(safe_merge=True)` folds adapter updates into the reconstructed
student's native parameters. A sibling staging directory receives config,
standalone safetensors, tokenizer and `recovery_report.json`. No `adapter_config`, teacher-bank payload,
or external `base_model` dependency is permitted. The merged class must be the
expected native class, no LoRA parameters may remain, and the adapter-only-save
flag must be false. Staged inventory/strict native header validation and a fresh
tokenizer reload verify vocabulary, special IDs, template and every permitted
complete input encoding before publication. Linux `renameat2` with
`RENAME_NOREPLACE` atomically publishes the directory and rejects even a racing
empty destination. Unsupported no-replace platforms fail closed. On caught
failure, staging is removed and the returned report has `status="rejected"`,
`artifact_admitted=false` and structured `error.type` / `error.message`.
Input stores are not rolled back or overwritten if an external writer changes
them; the change is detected and rejected. Do not modify stores concurrently.

Successful reports include:

- `status="completed"`, `artifact_admitted=true`, `output_dir`, `merged=true`,
  `standalone_native=true`, and `cache_restored`;
- `requested_steps`, `actual_steps`, `method`, `family`, `seed`, `device`,
  `hyperparameters`, library `versions`, determinism caveat and elapsed time;
- `resources` estimates, trainable/total parameter counts;
- `data` split hashes, selected IDs/counts and disjointness evidence;
- `validation_pre`, `validation_post`: response-token-weighted `supervised_ce`,
  `forward_kl` and combined `objective`; KL is `null` with no teacher forward;
- `training_history`, token-weighted training aggregates, train/dev teacher forward
  counts, adapter hashes/delta, frozen base hashes, native pre/post-merge target
  hashes and `merged_native_matrices_changed`;
- all teacher/student-store pre/post hashes and unchanged flags, exported native
  weight hashes, and `teacher_frozen_no_grad`.

`artifact_admitted` means training/export completed, **not promotion admission**.
Validation post-loss is measured with trained adapters **before merge** and
explicitly labeled `validation_post.model_state`; it is **not** deployment
validation. `merge_parity_probe` compares the last supervised teacher-forced dev
position over the full vocabulary before/after folding. Tolerances are FP32
`atol=rtol=2e-5`, BF16 `atol=0.02, rtol=0.05`; the maximum absolute difference and
pass/fail are reported. Numerical equality is not required, and this single
position does **not** guarantee generated-text parity or full-split quality.

Before publication, in-memory base references are released and the staged native
checkpoint is independently reloaded using the native AutoModel class. Its
saved `config.use_cache` must equal the original deployment setting, it must
contain no LoRA parameters, and `standalone_reload_probe` repeats the same
numerical comparison against the merged model. This is a bounded export check,
not a second full dev evaluation. The parent must separately evaluate the
**freshly reloaded deployment** before quality/promotion decisions. Inference
only needs the output directory, not the teacher or source student. No claim is
made that validation must improve. External task evaluation remains mandatory.

### Factor-preserving mode: trained arithmetic without a merge

```python
report = recover(
    teacher_dir, reconstructed_student_dir, NEW_output_dir, training_path, validation_path,
    steps=64, learning_rate=0.0001, rank=8, max_length=256,
    dtype="bfloat16", seed=17, kd_weight=0.7, teacher_mode="cached",
    export_mode="factor_preserving",  # deliberate choice, not an automatic rescue
)
```

The trained network is **the reconstructed/pruned native Qwen2 or T5 base PLUS
its required local trained LoRA factors**. After backward training, the exporter
captures F32 factors before any merge, saves current frozen base tensors under
native keys without rounding/recasting them, and does **not call
`merge_and_unload`**. The deployed computation remains native PEFT's base Linear
plus LoRA forward arithmetic. This avoids replacing `xW + LoRA(x)` with
`x(W + delta)` in BF16, which is not an arithmetically equivalent operation.
It is not an adapter package requiring the original teacher or student directory.

#### Actual bounded serialization (not a relaxed guard)

The factor exporter no longer calls model `save_pretrained(state_dict=...)` or
`safetensors.torch.save_file` for base/factors. A shallow dict contains only views
of already-live weights. PEFT `.base_layer.` names are normalized, LoRA keys are
excluded from the base, and the exact input-admitted native alias map chooses one
stored key per tied parameter. Every dropped alias must have identical shape,
dtype, pointer and stride; native runtime dtypes are checked before writing.

A standard little-endian safetensors writer constructs a bounded JSON header,
8-byte header-length prefix, aligned header and contiguous tensor spans. It writes
raw CPU storage in **1 MiB memoryview slices**, using
`tensor.detach().reshape(-1).view(torch.uint8).numpy()` to support BF16 without
NumPy BF16 conversion. It never creates a full `bytes` payload, clones the base,
casts an embedding to F32, or creates a second initialized base. Noncontiguous,
non-CPU or unsupported dtype storage fails closed instead of silently staging it.
Partial writes are completed. Every 8 MiB of payload (plus the bounded header),
files are fsynced, clean output-cache eviction is requested and dynamic headroom
is rechecked before continuing. This also bounds outstanding dirty output pages;
eviction advice is best-effort, never assumed reclaimed-memory credit. Native
reader/header validation remains mandatory.

Base shards target **256 MiB**; one indivisible larger tensor is allowed in its
own shard (a 272 MiB embedding is not split or cloned). An ordinary HF shard index
maps exact native keys. F32 factors use the same streaming writer. Per-tensor raw
payload SHA-256s, chunk/shard limits and zero-staging evidence are recorded under
`bounded_export`; frozen-base and factor hashes are checked around writing and
again after fresh reload. No original weight file is byte-copied on an assumption
that source dtype equals live dtype. In particular, source-BF16 T5 `wo` is stored
as its actual native F32 runtime values, not silently rounded back to BF16.

Only explicitly permitted tokenizer assets are privately copied from the hashed
candidate root, in bounded chunks with admitted total bytes and source digest/stat
checks. The allowlist covers native tokenizer JSON/config/special/added tokens,
vocab/merges, SentencePiece models and `chat_template.jinja`. No hardlinks,
`reconstruction_manifest`, custom code, nested assets or unexpected files are
copied. Missing/unexpected tokenizer dependencies reject. Small live model and
generation configs are rendered separately, preserving family flags, dropout and
restored `use_cache`; external name/source fields are removed. The strict bundle
inventory admits only these native assets, expected weights/index and factors.
Original source file hashes are still checked through the end of export.

```text
output/
  specialist_bundle.json          # strict versioned manifest, required factors
  base/
    config.json                   # reconstructed architecture, incomplete-base marker
    model.safetensors             # actual frozen pruned weights; or indexed shards
    tokenizer_config.json         # local tokenizer files + optional generation config
    tokenizer.json                # and any other native tokenizer resources
  adapter/
    adapter_config.json           # exact supported LoRA options; ../base only
    adapter_model.safetensors     # required trained F32 factors
  run_specialist.py               # fixed trusted convenience runner source copy
  recovery_report.json            # audit-only evidence, never loading instructions
```

There is deliberately **no root `config.json`**. `AutoModel.from_pretrained(root)`
is **unsupported** and must not be used for a factor bundle. The base config carries
`silt_specialist_component="INCOMPLETE_BASE_REQUIRES_LOCAL_TRAINED_FACTORS"`.
Loading `base/` intentionally can construct the unrecovered base, but is **not the
registered recovered specialist**, cannot inherit its evidence, and ignores the
trained factors. The parent public loader must detect `specialist_bundle.json`
**before** its native AutoModel path; it must never silently fall through to `base/`.
This is not a claim of a plain merged checkpoint, direct GGUF compatibility, or
quality equivalence from loading the base alone.

#### Public trusted loader and metadata API

```python
from asea.specialist.standalone import inspect_bundle, load_standalone

header = inspect_bundle(output_dir)  # bounded JSON + safetensors headers + file hashes
# No real model/tensor allocation and no source/teacher access in inspect_bundle.
print(header["counts"])              # includes base, factors, total, rank, original teacher
print(header["paths"]["tokenizer"])  # exact relative directory: "base"

bundle = load_standalone(output_dir, dtype=None, local_only=True)
model = bundle.model                # native PEFT inference model, eval/frozen/CPU
tokenizer = bundle.tokenizer        # local base tokenizer, same native templates
```

`StandaloneModel` returns `.model`, `.tokenizer`,
`.manifest`, and `.base_path` (the validated absolute local `base/` path).
`dtype=None` preserves the recorded dtype. An explicit string or `torch.dtype`
must **match** it; requesting recasting rejects because it changes the trained
arithmetic. The recorded dtype also constrains **every saved base header**: F32
requests require F32 throughout; BF16 Qwen requires BF16 throughout; BF16 T5
requires native `wo` weights in F32 and all other base weights BF16. Conflicting
headers reject before tensor reads even when checksums were recomputed. The
known-class meta/alias plan is then checked against those headers and executed
with the **same per-tensor loader as recovery**, without recasting saved tensors.
This avoids HF BF16 loading silently rounding saved T5 F32 `wo`. Generation config,
dropout metadata, tied aliases and CPU rotary buffers are retained. Qwen v1 bundle
schema/runner/files need no migration; this change does not rewrite old bundles.
`local_only=False` rejects. The loader requires exact recorded Torch,
Transformers, PEFT and safetensors versions for this schema. Install those normal
dependencies and SILT; standalone means **teacher-independent, not dependency-free**.

The loader never executes bundle-supplied model code and never delegates base
resolution to AutoPeftModel. It validates fixed root entries and relative paths,
all deployment-file SHA-256s and sizes, known model class/type, base safetensors
and native shape/key/tied-alias compatibility before native weights load, the exact
LoRA schema/options, fixed Qwen/T5 target sets, rank 1–128, F32 factor dtypes, exact
A/B names and dimension-dependent shapes, and counts derived from headers. It
constructs native PEFT from explicit validated arguments bound to the local base,
loads only local safetensors, and checks exact factor values after assignment.
No pickle, remote code, arbitrary adapter options, external base fields, unexpected
runner code, symlinks, traversal, extra/missing factors, or unknown root config are
admitted. `inspect_bundle` validates headers/hashes/options without allocating a
model; native meta-model compatibility is additionally checked by `load_standalone`
before real weights. Hashes are corruption checks, **not signatures/authenticity**;
the bundle must not be concurrently modified during loading.

The manifest schema `silt_factor_preserving_v1` includes exact `paths`, `model_class`,
`runtime_versions`, `export_mode`, deployment `files` hashes/sizes, and `counts`:
`base_parameters`, `factor_parameters`, `total_parameters`, `rank`,
`base_safetensors_bytes`, `factor_safetensors_bytes`, `total_safetensors_bytes`,
`original_teacher_parameters`, `original_teacher_safetensors_bytes`.
Teacher model type, input-file hashes and original counts appear under
`teacher_source_provenance` with `audit_only=true`; **no original paths are used or
needed at runtime**. Audit provenance cannot prove source identity without the
separate original receipt. The optional recovery receipt is not a loader directive.

The report includes `bundle_counts`, `size_comparison` ratios/booleans against the
original input teacher, and `trained_artifact_capture` binding the trained adapter
hash to its stored factor hash. Factor count is adaptive: rank times the sum of
input/output dimensions over every actual target matrix, not a fixed overhead.
Counts include **all required factors**, so a larger rank may remove a small
fixture's nominal compression benefit; the report never hides that.
`merged=false`, `standalone_native=false`, `factor_preserving=true`, and
`standalone_teacher_independent=true` distinguish this categorical artifact type.
The `standalone_native=false` field means "not a plain merged native root", not
"depends on the teacher". Training loss still has no deployment-quality meaning.

CPU preflight sets merge workspace to zero for this branch and adds **eight disk
bytes per factor parameter** (F32 factors plus duplicate-file scratch) to the
existing two-base-copy, tokenizer/config, and teacher-bank allowances. The output
contains one full pruned base plus factors; **base bytes may be duplicated** from
the input store, never an external symlink or a dependency on that input. For the
previous header-only 415586176-parameter, rank-8 example, pure BF16 base payload
plus 1413120 F32 factors is 836824832 bytes (~0.84 decimal GB), before safetensors
headers/tokenizer/configs, and 416999296 parameters. This is a projection, **not a
new real-run measurement**; use actual manifest counts and byte ratios. No heavy
real recovery was run for this export repair.

Before publishing, the exporter releases training weight references, reloads from
only the staged bundle through `load_standalone`, and verifies exact trained
adapter and frozen-base parameter hashes. It checks a **complete teacher-forced
forward** on one permitted dev row and a **two-new-token greedy generation** using
that row's actual native inference prefix. The **same PEFT graph** must be
bitwise-equal (`atol=rtol=0`) across fresh reload for both FP32 and BF16; failures
reject, not widen tolerances. `standalone_reload_probe` records shapes, input and
output token IDs, max-absolute error, cache setting and separate forward/generation
bitwise-equality results. These are short numerical fixtures, **not generated-code
quality, general text parity or full validation-set claims**. No final data is read.

Minimal executable runner (source copied into each bundle, requires installed SILT):

```sh
python /path/to/output/run_specialist.py --prompt 'Your task' --max-new-tokens 64
```

It uses the public trusted API, actual chat prefix (or the documented flat newline
fallback), and native seq2seq tokenization. The parent should apply deployment
input-length admission and full external task evaluation separately; the example
is not a server, benchmark, resume facility, or promotion certificate.

### Failed v3 receipt: preserve evidence, rerun explicitly

The reported v3 run executed 64 steps: dev CE **0.91757 → 0.70982**, forward KL
**0.13316 → 0.13045**, and adapter delta L2 **1.6176**. Its BF16 native merge probe
had maximum absolute error **0.15625** and failed the existing
`allclose(atol=0.02, rtol=0.05)` check; the output was deleted. These are supplied
prior-run observations, not new validation from this patch. They demonstrate
training activity and an export rejection, **not a recovered deployable artifact**.
The original failed `recover.receipt.json` must remain unchanged. If those trained
factors were not saved, they cannot be reconstructed from the receipt, metrics or
hashes. A **new run with the same fixed permitted recipe/data locks/seed and a new
output/receipt path**, explicitly choosing `export_mode="factor_preserving"`, is
required. Do not use final data, fabricate a rescue result, silently change the
recipe, or relax native merge tolerances. No old receipt is opened or overwritten
by this API.

## Offline mechanics tests

```
/agent/workspace/silt-venv/bin/python -m pytest -q tests/test_specialist_recovery.py
```

Factor-preserving coverage includes actual **two-step PEFT training** for the full Qwen2/T5 × FP32/BF16 × cached/resident matrix, exact full
forward and short-generation reload parity, original stores renamed unavailable,
a separate offline subprocess, and the copied public runner. Adversarial cases
rehash malformed adapter configurations/tensors to ensure schema validation is
not merely a checksum check; source pointers, ranks, targets, unknown options,
missing/extra/wrong-shaped factors, unsafe metadata, versions, paths and runner
code reject before weights. Injected merge corruption still rejects in default
native mode; injected factor reload corruption rejects and removes staging.
Typed-loader tests cover Qwen/Switch/T5, both requested dtypes, single/indexed
shards, original tensor values, tied aliases and same-handle mmap identity.
Additional regressions verify Switch router-only F32 accounting (no full-shard
double charge), fresh pressure refusal with unchanged reserves, Qwen old/new
loader exact one-step training equality, manifest/base-header dtype disagreement
before tensor reads, and T5 nondefault dropout/generation-config reload parity.
Final test totals and current source hashes are recorded in the external repair
report rather than treated as a permanent property of this document.

Tests construct tiny **random** native Qwen2 and Switch/T5 models and local
WordLevel tokenizers. The Switch fixture copies a compatible expert to a dense
T5 using a test-local independent name/shape mapping. Both families run actual
backward steps, merge, rename away both input stores, then reload and generate
from only the export. Additional checks cover BF16 distillation-only training,
supervised no-teacher behavior, dense-reference KL and gradients, EOS/prompt/pad
masking, split leakage, tokenizer mismatch, corrupt checkpoint rejection,
resource rejection, injected optimizer failure and atomic no-overwrite.
Additional contract tests cover joint native-chat/flat-newline encoding,
whitespace-crossing prefix refusal, response bytes/newline/EOS masks, incompatible
templates, complete all-row admission before weights, bounded duplicate-key and
nonfinite JSON parsing, family leakage, full-256 vocabulary workspace, and
premerge/merged/reloaded numerical probes. These are mechanics tests, **not
pretrained recovery-quality evidence**. No real pretrained weights were loaded
or quality training performed for this integration repair; only local tokenizer
metadata and tiny randomly initialized model fixtures were used. Do not weaken
memory admission to force a real model into a 4 GB budget.
