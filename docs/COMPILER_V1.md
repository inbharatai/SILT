# Structural expert compiler v1: local Switch only

## Scope and status

This is an actual model-family structural transformation, not a mock adapter or
an invocation of the existing SILT core. Supported architecture:
`SwitchTransformersForConditionalGeneration`, native HF `model_type=switch_transformers`.
The public config of `google/switch-base-8` has this architecture, 8 experts,
12 encoder and 12 decoder layers, and alternating sparse feed-forward layers.
It is a **seq2seq text model, not a frontier coding model**. Routing calibration,
cloze exact match, and random-model mechanics do not establish coding skills.

Config verified without downloading weights:
- https://huggingface.co/google/switch-base-8/raw/main/config.json
- Adapter layout checked against the pinned runtime source:
  https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/switch_transformers/modeling_switch_transformers.py

The original implementation acquired no pretrained weights. A later local run
produced structural and evaluation evidence (see the baseline below), but **did not
establish quality**. Basic `inspect`, `prune`, `evaluate`, and `infer` outputs remain
**UNADMITTED**. Only a new `certify` run can issue narrowly scoped
`ADMITTED_SEQ2SEQ_TEXT_ONLY` evidence; it never activates or rewrites a model.
There is no automatic promotion, training, fine-tuning, distillation,
Phase 6 dense reconstruction, capability transplantation, or performance promise.
`prune --repair-training` returns `supported=false`, `UNSUPPORTED_REPAIR`, exit 2.

## Runtime and trust boundary

Python >=3.9. Importing this subpackage or inspecting/validating a model does not
import torch/transformers; the existing package's pydantic dependency still applies.
Model operations need:

```sh
# In the operator-managed environment, not by this compiler:
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.51.3 accelerate safetensors sentencepiece
```

`accelerate` is required for the single-model `low_cpu_mem_usage=True` loader.
Runtime rejects transformers versions other than 4.51.3, and torch older than 2.6.
Torch 2.6 CPU is the intended target; newer torch is not a tested compatibility claim.
Only CPU, batch size 1, `float32` (default) or `float16` are supported. Router
selective FP32 precision follows the native HF implementation. BF16 source tensors
can be loaded into the requested dtype. Global load dtype conversion affects dense
weights too; preservation means no dense structural rewrite, not byte equality
across dtype conversion.

Models must be **fully materialized local directories** with `config.json`,
`tokenizer_config.json`, complete tokenizer files, and `model.safetensors` or
`model.safetensors.index.json` plus indexed root-level safetensors shards. The
compiler uses `local_files_only=True`, `trust_remote_code=False`,
`use_safetensors=True`; it never downloads models or loads pickle. Native
architecture and tokenizer mappings must not request remote code. Quantized and
nonfloating checkpoints are rejected. All symlinks in model trees and ancestor
paths are rejected (including standard symlink-based HF snapshot layouts: copy
files to a materialized directory). Special files are rejected. Extra files are
hashed but not executed. Unknown/missing/unexpected model tensor keys cause failure.

The compiler is not a security sandbox against a hostile local process concurrently
editing paths, and hashes are content integrity/lineage evidence, **not signatures
or external provenance authentication**. Header inspection validates safetensors
metadata but is not a full native architecture load or quality validation.

## CLI (run from checkout with `PYTHONPATH=src`, or an installed checkout)

```sh
python -m asea.compiler inspect --model /models/switch-base-8
python -m asea.compiler prune --model /models/switch-base-8 \
  --output /models/switch-base-8-keep4 --keep-experts 4 \
  --calibration /data/calibration.json --dtype float16 --max-length 128
python -m asea.compiler evaluate --model /models/switch-base-8-keep4 \
  --suite /data/heldout.json --reference-model /models/switch-base-8 \
  --dtype float16 --max-new-tokens 32 --evidence-output /evidence/evaluation.json
python -m asea.compiler infer --model /models/switch-base-8-keep4 \
  --prompt 'The capital of France is <extra_id_0>.' --max-new-tokens 16 --dtype float16
```

Exact options:

| Command | Required | Optional |
|---|---|---|
| inspect | `--model PATH` | none |
| prune | `--model PATH --output PATH --keep-experts INT --calibration FILE` | common runtime options, `--repair-training` (rejected) |
| evaluate | `--model PATH --suite FILE` | common runtime options, `--reference-model PATH --evidence-output FILE --max-new-tokens INT` |
| infer | `--model PATH --prompt TEXT` | common runtime options, `--max-new-tokens INT` |
| certify | `--model PATH --reference-model PATH --suite FILE --certificate-output FILE` | common runtime options, `--max-new-tokens INT`, policy options below |

Common runtime options: `--dtype {float32,float16}` default float32;
`--max-length INT` default 128, valid 1..256;
`--memory-budget-mib INT` optional positive cap, cannot override detected available RAM.
Generation `--max-new-tokens` defaults 32, valid 1..128. This is a bound on generated
tokens, not total sequence length. Inputs/targets truncate to max-length. No batching
option is exposed: it is fixed at 1. Greedy decoding explicitly uses
`do_sample=false,num_beams=1`.

Every operational command prints one JSON object to stdout. Dependency logging is
redirected to stderr. `--help` retains standard argparse human-readable help.
Exit 0 has `ok=true,schema="asea.compiler.v1",command=...`; validation/unsupported
errors exit 2 with `ok=false,error={type,message},supported=false`; unexpected
runtime/loader failures exit 3 with `error.type="RUNTIME_ERROR"` and exception class.
Exception: **all `certify` refusals/failures exit 2**, with `ok=false`,
`status="REJECTED"`, `admission="UNADMITTED"`, and structured `reasons` containing
stable `code` values. Statistical reasons also name their `group`. A completed
rejected evaluation writes its rejection certificate; preflight, argument, path,
or runtime failures return reasons on stdout without writing a certificate.
Successful certification exits 0 with `status="CERTIFIED"`; basic command `ok=true`
means successful execution, never admission.
OS termination/OOM cannot be caught reliably. Important typed failures include
`UNSAFE_PATH`, `UNSAFE_OUTPUT`, `OUTPUT_EXISTS`, `INVALID_KEEP`,
`UNSUPPORTED_ARCHITECTURE`, `UNSUPPORTED_CONFIG`, `UNSAFE_WEIGHTS`,
`INVALID_WEIGHTS`, `MISSING_DEPENDENCY`, `UNSUPPORTED_RUNTIME`, `MEMORY_PREFLIGHT`,
`INVALID_DATASET`, `CALIBRATION_LEAKAGE`, `LINEAGE_MISMATCH`, `SOURCE_CHANGED`.

## Calibration and pruning semantics

Calibration is a JSON object with 1..256 samples. Each has a nonempty prompt and
nonempty **teacher-forced decoder target**, each <=32768 characters:

```json
{"samples":[
  {"prompt":"The capital of France is <extra_id_0>.",
   "target":"<extra_id_0> Paris <extra_id_1>"}
]}
```

The example is a format demonstration, not a recommended sufficient calibration set.
Prompt-only calibration is rejected: it cannot adequately exercise decoder experts.
Tokenization uses one sample at a time, no padding; actual EOS/decoder-start positions
are included. At every sparse encoder/decoder router a temporary forward hook records:
1. summed all-expert softmax probability mass (recomputed in FP32 from router logits);
2. pre-capacity top-1 selection counts;
3. actual post-capacity dispatched counts;
4. token count.

Per layer, keep the N highest probability-mass experts (ties: lower original index).
This is **routing usage, not causal importance**. Survivors are ordered by original
index and remapped to consecutive `expert_0 ... expert_(N-1)`. The compiler physically
replaces each ModuleDict with surviving module references (not masks), creates a
smaller router classifier with corresponding weight/bias rows, and updates router,
top-level, encoder and decoder expert counts. Shared dense layers and all other
modules remain structurally untouched. Expert capacity is unchanged; changed routing,
renormalized probabilities, and capacity drops can substantially harm outputs.
A single N applies to every sparse layer because this HF family has a global
`num_experts` config. `1 <= N < source.num_experts` is required.

Source is loaded only once, with no teacher/student deepcopy. After routing selection,
removed expert references are released; candidate is saved in 256MB safetensors
shards with the tokenizer, then the model is released. A sibling temporary directory
is renamed to the requested output only after successful save and source rehash.
The output must not exist; its parent must exist, and it cannot be inside the source.
Failure cleans staging. The source is never saved or rewritten.

A CPU-memory heuristic checks `2 * requested_dtype_parameter_bytes + 512 MiB`
against available RAM, cgroup headroom, a 3584 MiB default ceiling, and any smaller
operator cap. This is a refusal guard, not a peak-RSS guarantee. Router FP32 weights,
activations, tokenizer/runtime overhead, page cache, and allocator retention matter.
FP32 loading of a large source may be refused on a ~4GiB sandbox; try float16 rather
than disabling safety checks. Model/reference evaluation loads **sequentially**.

## Lineage and exact output schema

`inspect` returns `architecture,num_experts,config,files,artifact_sha256,weights`,
plus scope/admission/repair status. `files` maps every relative filename to
`{sha256,bytes}`. `weights` has stored parameter count, tensor count, total serialized
weight bytes, and filenames. Stored counts exclude HF tied aliases omitted on disk;
prune's `source_parameters`/`candidate_parameters` count actual unique model parameters.

`prune` returns output path, manifest path/hash, unique source/candidate/removed
parameter counts, retained expert count, source-unchanged=true, and UNADMITTED status.
`compiler_manifest.json` records:
- complete source file inventory + canonical inventory hash, including weights,
  config, tokenizer and any other files; audit-only source path;
- post-operation source hash and equality result;
- candidate file inventory + canonical hash (excluding the manifest itself);
- calibration file SHA256, prompt hashes, counts, length/batch/decoder settings;
- selection method, per-layer retained source indices, new-to-source mapping,
  probability sums, top-1/dispatch counts, tokens;
- parameter counts, runtime versions, load dtype/memory plan and limitations.

The manifest cannot hash itself recursively. Prune's separate `manifest_sha256`
binds it; subsequent `inspect` includes the manifest in the full artifact inventory.
Subsequent model commands reject changes to candidate files versus its manifest.
The audit source path is never dereferenced to reload the candidate: fresh-process
inference needs only the candidate directory.

## Held-out evaluation: text evidence, not coder certification

Suite format (1..4096 cases; unique nonempty string IDs; references required).
`group` is optional for ordinary evaluation and, if present, must be `target` or
`control`. Old ungrouped suites still work. Certification requires both groups.
The upper bound was expanded to permit sufficiently powered uncertainty tests:

```json
{"split":"heldout","cases":[
  {"id":"h001","prompt":"The capital of Germany is <extra_id_0>.",
   "references":["<extra_id_0> Berlin <extra_id_1>","Berlin"]}
]}
```

Decoding uses `skip_special_tokens=true`; references must reflect that decoding.
No normalization is applied. Whitespace, case, punctuation differences fail literal
exact match. Provide real heldout reference data; the example is not benchmark evidence.
The suite must declare heldout. For a compiled candidate, identical calibration
file hashes or overlapping exact prompt hashes are rejected. This does not detect
semantic paraphrase leakage and does not establish independent test-set curation.

Evaluation JSON contains:
- `model_sha256,model_files,suite_sha256,heldout_check`;
- `result={metric:"literal_exact_match_text_proxy",exact_match:0..1,cases:[...]}`;
- each case: `id,prompt,references,generation,exact_match,generation_sha256`, plus
  `group` when provided;
- optional `reference={model_sha256,result,candidate_minus_reference_exact_match,
  generation_agreement}` (same cases, sequential model load), otherwise null;
- `generation_parameters,dtype,preflight,created_at,evidence_scope,limitations`;
- `admission:"UNADMITTED",certifies_coding_skills:false,evidence_sha256`.

The evidence hash is SHA256 of canonical sorted-key compact JSON **before adding
`evidence_sha256` or the optional returned `evidence_output` path**. Individual text
hashes likewise hash canonical JSON strings, not bare UTF-8. File hashes hash raw bytes.
Stdout always writes the complete evidence and raw generations; optionally
`--evidence-output` writes the same evidence to a new file outside model directories.
It does not overwrite files. No generated code is executed. There are **no functional
coding tests** in v1; even code-shaped prompts get a clearly labeled text proxy.

`infer` returns the prompt, decoded generation/hash, model hash, token bounds, memory
plan, and scope. It does not infer admission or reference-model dependence.
Both inference and evaluation now include `resources`, `runtime`,
`loaded_structure`, and post-inference model-immutability checks. Reference
operations have separately collected `resources` and `loaded_structure` fields.
Resource numbers are actual finite floats: monotonic-clock `wall_seconds`
(load, generation, release; excludes inventory hashing),
`generation_wall_seconds`, and `process_peak_rss_bytes` from
`resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` (platform units converted).
The RSS is a **process-lifetime high-water mark**: sequential candidate/reference
measurements can be equal because the peak cannot reset, and must not be described
as isolated model peaks or an exact memory saving. These are one-pass observations,
not repeated isolated benchmarks. The conservative preflight estimate remains
separate and is never substituted for observed metrics. Runtime captures Python,
platform, torch, transformers, accelerate, safetensors, and tokenizers versions.

## Compiler admission gate: seq2seq text only

```sh
python -m asea.compiler certify --model /models/switch-base-8-keep4 \
  --reference-model /models/switch-base-8 --suite /data/admission-heldout.json \
  --certificate-output /evidence/switch-admission.json --dtype float16 \
  --minimum-cases 20 --max-loss 0.02 \
  --minimum-reference-score 0.5 --minimum-candidate-score 0.5 \
  --minimum-size-reduction 0.1
```

`certify` **directly calls this compiler's evaluator** on the supplied local model
and exact source. It does not accept evaluation JSON, scores, signatures, HMAC
keys, or an external attestation as inputs. It does not change model admission
metadata, activate a candidate, or implement repair training. A certificate is a
local audit result tied to these exact files, suite, runtime, and policy, not a
portable proof against a malicious operator editing code or generating new hashes.

| Policy option | Default | Allowed |
|---|---:|---|
| `--minimum-cases` | 20 | Integer 20..4096, total cases |
| `--max-loss` | 0.02 | Finite 0..0.05, absolute score difference |
| `--minimum-reference-score` | 0.5 | Finite 0.5..1.0 |
| `--minimum-candidate-score` | 0.5 | Finite 0.5..1.0 |
| `--minimum-size-reduction` | 0.1 | Finite 0.1..1.0 |

Unsafe policy values are rejected, not clamped. Common runtime options are the
same as evaluation. The requested dtype **must equal prune's manifest dtype**;
there is no separate reference dtype. Both models are actually loaded at that
dtype and their unique parameter counts and dtype histograms are measured.
Native selectively-FP32 router parameters are permitted and reported.

Every case must declare `"group":"target"` or `"group":"control"`; both groups
must be nonempty. IDs and trimmed prompts must be unique across the entire suite,
so the same prompt cannot serve as target and control or inflate the sample count.
All certification references must be nonempty text. Calibration exact-prompt/hash
leakage checks remain in force. Labeling a suite heldout cannot establish actual
statistical independence: operators must preselect independent cases and avoid
repeated tuning against the same suite. No prompt syntax turns this into a coding
benchmark.

### Conservative paired uncertainty

For **each** target/control group, pair candidate and source literal exact-match
outcomes on the same cases. Let gains count candidate-correct/source-wrong and
losses count candidate-wrong/source-correct. The gate computes:

- one-sided exact Clopper–Pearson lower bound for gain probability;
- one-sided exact upper bound for loss probability;
- paired noninferiority lower bound = gain lower bound − loss upper bound;
- exact one-sided lower bounds on **both models' absolute scores**.

All eight one-sided bounds (four per group) use tail alpha `0.05/8 = 0.00625`,
Bonferroni familywise confidence at least 95%, under independent-case sampling.
The deterministic binomial inversion uses no bootstrap and no random seed; the
policy records `random_seed:null` and `randomness:"none_exact_deterministic_bounds"`.
Admission requires the paired lower bound ≥ `-max_loss` **in both groups**, and
both absolute-score lower bounds ≥ their respective policy floors in both groups.
This deliberately conservative rule does not infer positive quality from 0/0,
aggregate equality, generation agreement, or zero observed variance.

**Twenty is an eligibility floor, not enough evidence for automatic admission.**
Even 20 perfectly matching correct cases split 10/10 fail the default loss bound.
With zero paired losses, the upper bound is `1 - alpha^(1/n)` rather than zero;
with identical fully correct outcomes, 300 independent cases per group clear the
default 0.02 loss threshold. Other outcome patterns may require more cases or fail
entirely. The gate never automatically raises tolerance to make a candidate pass.

### Structural, resource, and output requirements

The candidate must have a compiler-v1 manifest whose source inventory and exact
original artifact digest match `--reference-model`, whose source-after digest
proves the recorded source was unchanged, and whose candidate inventory still
matches all candidate files. Expert counts must decrease, recorded layer
selection must exist, and manifest parameter counts must equal the actual loaded
unique counts. The source path written in a manifest is audit-only: the caller
must supply the matching source explicitly.

The certificate reports actual on-disk weight and full-artifact byte counts,
actual loaded parameter bytes, requested-dtype normalized loaded parameter bytes,
and normalized serialized-weight bytes (`stored_parameters * requested_width`).
Admission requires the minimum reduction in **unique parameters AND normalized
parameter bytes AND normalized weight bytes**. An FP32-to-FP16-only disk reduction
cannot pass. Disk/header/tokenizer overhead is reported, not confused with physical
parameter removal. Observed resource metrics and nonempty runtime versions are
required, but are not used to claim latency or memory improvement.

Both inventories are rehashed after both inference runs; suite hashes are also
rechecked. Any source/candidate/suite mutation rejects certification.
`--certificate-output` is required, must be outside **both** model trees with an
existing parent, and is opened exclusively with no overwrite. It must not already
exist. Symlinks are rejected. **Never save the certificate inside the candidate:**
doing so would invalidate its immutable compiler manifest. The output binds full
source/candidate inventories, source/candidate/manifest/suite/policy hashes,
per-group bounds, normalized structural metrics, complete fresh evaluation and
observed resources. `certificate_sha256` hashes canonical JSON before adding itself
or the returned `certificate_output` path; it is not a signature. Rejected completed
runs retain evidence and reasons in the output file for diagnosis.

### Existing local baseline remains unadmitted

The separately recorded local files were inspected while implementing this gate:
`/agent/workspace/silt-evidence/switch-prune.json`,
`switch-comparison-evidence.json`, and
`/agent/workspace/silt-models/switch-keep4/compiler_manifest.json`.
The 8→4 expert run records source 619,339,008 parameters, candidate 392,809,728,
and 226,529,280 removed; the manifest records float16 loading and source unchanged.
The source weight file is 1,238,805,814 bytes; four candidate weight shards total
785,734,056 bytes. Neither raw file reduction nor parameter removal proves quality.

The recorded comparison has **four ungrouped cases**, candidate exact match 0.0,
reference exact match 0.0, difference 0.0, and generation agreement 0.0. For example,
reference `Paris.` fails references `Paris` and the sentinel form; candidate
`Paris...........` also fails. The literal metric was not retroactively normalized
or its references edited to produce a pass. This tiny, ungrouped, zero-score run
and its heuristic-only memory estimates are **not admission evidence**. Existing
artifacts and recorded results remain UNADMITTED; fresh properly designed heldout
cases and actual measurements are needed. This gate implementation itself loads no
large pretrained model and makes no new real-model quality claim.

## Tests and limitations

```sh
python -m pytest -q tests/test_compiler_v1.py tests/test_compiler_admission.py
```

Lightweight tests exercise path/output/architecture/keep/data/memory validation,
header inspection, lineage tamper rejection, error envelopes and exact-match semantics.
Header fixtures are explicitly not usable checkpoints or real-model evidence.
Admission tests use synthetic scores and a labeled synthetic loader to exercise
zero/zero refusal, missing controls, absolute score lower bounds, small-sample
uncertainty, paired loss bounds, dtype-only reduction refusal, required direct
evaluation, immutable inventories, observed-resource schema, and exclusive output.
No synthetic pass is presented as a pretrained model admission.
Optional tests build a tiny **RANDOMLY INITIALIZED** native Switch and local tokenizer:
check physical module and router-row retention, dense weight equality, serialization,
source digests, new-process inference after hiding the source, heldout generation,
and calibration-overlap refusal. They skip without ML dependencies. Passing these
proves mechanics only. Real pretrained pruning, output quality, acceptable retention,
and latency/memory comparisons require separately recorded runs. No repair training
is implemented, and no Phase 6 dense reconstruction claim is made.
