# Local composition v1 (additive, experimental)

This subsystem does not alter SILT's legacy packets, promotion, extraction, CLI,
Studio or model-training flows. It implements local component registration,
closed typed graph execution, reference-fixture evaluation, distinct admission,
manual activation/rollback and deployment export. **A successful run is only a
candidate run, never certification or activation.** No real-model inference or
quality result is claimed by the accompanying unit tests.

## Supported scope

| Node kind | Task | Input → output | Local loader |
|---|---|---|---|
| `hf_text` | `causal` | text → text | AutoModelForCausalLM |
| `hf_text` | `seq2seq` | text → text | AutoModelForSeq2SeqLM |
| `hf_asr` | `whisper` | audio → text | WhisperForConditionalGeneration |
| `hf_tts` | `vits` | text → audio | VitsModel |

Models must already exist locally. Inference uses `local_files_only=True`,
`trust_remote_code=False`, `use_safetensors=True` on weight loaders, CPU float32,
at most two Torch compute threads, and deterministic greedy text generation.
There are no user-defined loaders, shell commands, subprocess adapters, Python
plugins, model downloads, remote-code execution or GPU execution. Whisper input
is at most 30 seconds. WAV is 16-bit PCM mono/stereo at 8–96 kHz; non-16-kHz ASR
input is resampled lazily through SciPy. MP3 and compressed audio are rejected.
Vits output is a PCM WAV under the workspace's `outputs/<run-id>/` directory.

The graph is a fixed DAG with one typed input per node and one selected output.
V1 requires every node to contribute to the output, so its supported shape is a
**linear pipeline**, not arbitrary branching, joins or multimodal reasoning.
For speech → transcription → text coder → speech, use three nodes:

1. `transcribe`: `input: "$input"`, kind `hf_asr`, task `whisper`.
2. `coder`: `input: "transcribe"`, kind `hf_text`, task `causal` or `seq2seq`.
3. `speak`: `input: "coder"`, kind `hf_tts`, task `vits`.

Set root `input_type: "audio"`, `output_node: "speak"`; provide each component's
actual local model directory, risk and provenance. No TTS or ASR model is
implicitly downloaded or assumed installed. A model is loaded only for its
node and its references released and garbage-collected before the next node.
Returned audio never contains an executable command. Generated code is text,
not executed. This pipeline can *produce* speech but cannot be admitted as a
speech-output composition by the current text-only quality evaluator.

## Installation and local directories

The base API and CLI import with Python 3.9+ and Pydantic 2 only. For real local
inference, separately provision CPU Torch >=2.6, Transformers **4.51.x**, NumPy,
Safetensors and model-specific tokenizer dependencies (possibly SentencePiece,
`uroman` for certain Vits tokenizers). SciPy is needed for ASR resampling.
This implementation does not install these dependencies. Torch/Transformers
imports happen only inside a real adapter call.

Copy/provision regular model files, **not symlinked Hugging Face cache snapshots**.
Directories containing symlinks, traversal, Python source/bytecode or pickle
formats (`.bin`, `.pt`, `.pth`, `.pkl`, `.pickle`, `.ckpt`) are refused, even when
safetensors also exist. Keep only the model's required safe configuration,
tokenizer assets and safetensors weights. All included files are hashed with
streaming SHA-256; the full file inventory participates in registration.

The example `configs/compose_text_example.json` is directly accepted by the
schema and CLI. Its concrete local path is
`/agent/workspace/silt-models/SmolLM2-135M-Instruct`; provision safe regular files
there or change `model_path` to your existing directory. No placeholder syntax,
variable substitution, environment interpolation or implicit model resolution
exists. Paths are interpreted relative to the process working directory, not
the spec file; absolute paths are recommended.

## Strict JSON schemas

All objects reject unknown keys; fields use strict Pydantic types. JSON duplicate
keys, nonfinite constants and documents larger than 2 MiB are rejected. Explicit
unsupported kinds such as `hf_vision`, arbitrary code nodes, and model/task
mismatches fail validation. Full machine-readable schemas are available without
ML dependencies:

```python
from asea.compose import ComponentManifest, CompositionSpec, EvaluationSuite
print(CompositionSpec.model_json_schema())
```

`ComponentManifest`:

- `schema_version`: 1 (default).
- `kind`: one of the table's kinds. Reserved `fixture_text` exists solely for
  explicitly opted-in Python unit tests; public CLI rejects it and production
  admission refuses it even when its metrics pass.
- `task`: `causal`, `seq2seq`, `whisper`, `vits` (must match kind).
- `model_path`: existing local directory.
- `risk`: `low`, `medium`, `high`.
- `provenance`: nonempty list of `{source, risk, license}` strings/risk.
  Effective risk is the maximum of the component and every provenance entry;
  graph risk is the maximum across all components. Provenance/license strings
  are declarations, not independently verified claims.

`CompositionSpec`:

- `schema_version`: 1; `name`: nonempty string; `input_type`: `text` or `audio`.
- `nodes`: 1–16 `{id, component, input, prompt_prefix?, max_new_tokens?}` objects.
  IDs match `[A-Za-z][A-Za-z0-9_-]{0,63}`. `input` is `$input` or a node ID.
  `prompt_prefix` defaults to empty. `max_new_tokens` is 1–512, default 128,
  used by text/ASR generation; Vits instead has fixed input token/character caps.
- `output_node`: known node ID. Duplicate IDs, cycles, missing edges, type
  mismatches and dead nodes are refused.
- `limits`: optional object with these defaults:
  `max_input_chars=8192`, `max_output_chars=16384`, `max_audio_seconds=30`,
  `max_input_bytes=16777216`, `max_seconds=300.0`,
  `max_peak_rss_mb=3800.0`, `threads=2`. Schema bounds permit at most 65536 text
  characters, 120 audio seconds (Whisper remains 30), 32 MiB input, 3600 seconds,
  65536 MiB configured RSS and two compute threads.

The graph fingerprint binds the **entire normalized spec**, including all limits,
paths, prompts, risk/provenance, and registered component inventories. Altering
config or model files makes a different candidate. Files are freshly verified
before/after inference and before activation/rollback/export. Registration
references local directories by default (no multi-GB duplication). The Python
`Workspace.import_directory(source)` helper optionally copies a safe directory
once by content inventory, streaming and verifying its copy. It does not unzip
archives. Use it under `with workspace.writer():`.

## Manual CLI workflow

Always use a separate empty directory (or an existing composition workspace),
never a legacy registry or project source directory. Stdout is exactly one JSON
object for every command including errors/help. Diagnostics go to stderr. Exit
0 means success; nonzero means blocked/rejected/error. The public Python
`asea.compose.main(argv)` returns an integer rather than exiting its caller.

```sh
python -m asea.compose --workspace /tmp/silt-compose-demo inspect --spec configs/compose_text_example.json
python -m asea.compose --workspace /tmp/silt-compose-demo run --spec configs/compose_text_example.json --input 'Add two integers.'
# Audio graph supplied separately as described above:
python -m asea.compose --workspace /tmp/silt-compose-demo run --spec /path/speech.json --input-file /path/request.wav
python -m asea.compose --workspace /tmp/silt-compose-demo evaluate --spec configs/compose_text_example.json --suite /path/references.json
python -m asea.compose --workspace /tmp/silt-compose-demo list
```

The run command accepts optional `--input`; audio uses `--input-file` and absent
or empty `--input`. Text file input and images are deliberately unsupported.

A reference suite is explicit, not a claimed promotion status. Example format
(not a claim that any model passes):

```json
{
  "schema_version": 1,
  "name": "independent code-text references",
  "reference_source": "Separately authored, reviewed reference fixtures",
  "cases": [
    {"id":"addition", "group":"target", "input":"Add two integers.",
     "reference":"def add(a, b):\n    return a + b", "metric":"text_similarity_proxy", "threshold":0.9},
    {"id":"identity", "group":"control", "input":"Return the input unchanged.",
     "reference":"def identity(x):\n    return x", "metric":"text_similarity_proxy", "threshold":0.9}
  ]
}
```

`EvaluationSuite` requires a name, nonempty `reference_source`, 2–64 cases,
nonempty target **and** control groups, unique case IDs, independently supplied
nonempty reference strings and **explicit** thresholds in [0,1]. Each case has
exactly one `input` or `input_file`. Reference strings are loaded before runs;
file fixtures are hashed before/after runs and checked again at admission.
No model-generated self-reference is synthesized. Human independence and
representativeness of supplied fixtures cannot be established mechanically.

Metrics: `text_exact` (1/0, >= threshold); `text_similarity_proxy`
(character SequenceMatcher score, >= threshold, <=8192 chars); `word_error_rate`
(case-sensitive whitespace tokenization, <= threshold, <=2048 words; scores can
exceed 1). **None is functional code correctness.** `functional_code` is a
recognized task metric but always BLOCKED before inference: no secure code
sandbox is implemented and no unsafe subprocess fallback exists. Audio-output
quality evaluation is BLOCKED, not replaced by duration or waveform checks.

Only a full passing real non-fixture suite with measured in-budget runs produces
an immutable `admitted` evaluation. Its scope remains the named reference text
metrics. Candidates remain candidates; admission does not mutate them and does
not activate a deployment. Admission is local evidence gating, not an external
security certification, proof of generalization or broad model-quality claim.

After reviewing a returned **admitted** evaluation ID:

```sh
python -m asea.compose --workspace /tmp/silt-compose-demo activate --evaluation evaluation-RETURNED_ID
# Required separately if max provenance risk is high:
python -m asea.compose --workspace /tmp/silt-compose-demo activate --evaluation evaluation-RETURNED_ID --approve-high-risk
python -m asea.compose --workspace /tmp/silt-compose-demo rollback --deployment deployment-PREVIOUSLY_RETURNED_ID
python -m asea.compose --workspace /tmp/silt-compose-demo export --deployment deployment-RETURNED_ID --output /tmp/bundle.zip
# Opt in only when you want large local model files copied into the ZIP:
python -m asea.compose --workspace /tmp/silt-compose-demo export --deployment deployment-RETURNED_ID --output /tmp/full-bundle.zip --include-dependencies
```

IDs above represent actual IDs from JSON responses, not literal runnable IDs.
Activation revalidates reference thresholds, stored outputs, candidate/graph
bindings, per-run measured resources, suite fingerprint and current model
hashes. High-risk activation requires explicit approval. Rollback accepts only
an existing valid known deployment and revalidates it; unknown/stale/corrupt
records never change the pointer. `active.json` is replaced atomically with
file and directory fsync. Activation is a local selection pointer, not a
long-running inference server; CLI `run` always uses the explicit spec.

Export is a distinct `asea-composition-deployment`, `format_version: 1` ZIP,
not a legacy SILT packet. It contains spec, manifests, run/evaluation/deployment
records. Dependencies are excluded by default, with their digest inventories
and source paths retained; therefore that default bundle is not self-contained.
With `--include-dependencies`, model bytes stream through 1 MiB buffers into
Zip64 entries; digest verification occurs during and after copying. It never
exports the workspace integrity secret. Cross-workspace ZIP import, portable
trust credentials and automatic model-path rebasing are **not implemented**.
Raw input WAV files are not included; evaluation records retain their digest
and original path. Exports contain reference/input/output text: review privacy
before sharing them. ZIP SHA-256/size are returned after writing.

## Resource and trust boundaries / omissions

- Sequential loading, bounded tokens/text/audio/cases, no GPU, and two Torch
  compute threads target small-model 4 GiB/2-CPU hosts. **No guarantee any chosen
  model fits 4 GiB.** Model allocation and native kernels are in process, not an
  OS-enforced memory/CPU sandbox. Wall budgets are cooperative checks between
  nodes and after completion, not kill-timeouts. Out-of-memory or a hung native
  model call can terminate/hang the process. Use an external cgroup/container
  when hard ceilings are required. Inspect/register hash models but do not
  predict inference RAM.
- Resource evidence is actual elapsed monotonic time and process-lifetime
  high-water RSS (`getrusage`), conservatively contaminated by previous calls
  in the same process; it is not per-model incremental RSS. It is checked
  against explicit spec limits. Thread count is the configured Torch compute
  bound, not a measurement of every library/OS thread. Model quality and
  resource evidence from unit fixtures are never production-admissible.
- `select_measured(workspace, evaluation_ids, input_type, output_type,
  max_wall_seconds, max_peak_rss_mb, suite_hash=None)` selects least measured peak RSS then
  latency among the supplied already-admitted, type-compatible, in-budget
  bundles evaluated on the same reference suite. Mixed suites require an explicit
  `suite_hash` filter. Otherwise it returns `no_feasible_bundle`. It does not construct/rewrite
  graphs, search models, prove global optimality, or promise performance on
  different hardware/input distributions. This is a Python API, not a CLI
  selection command.
- Immutable records use a workspace HMAC secret to detect ordinary edits such
  as changing status to `admitted`; this is **not protection from the workspace
  owner**, who can read that secret or change the Python implementation. Trust
  the local owner/filesystem. No signatures from a third-party authority,
  cryptographic provenance verification, encrypted storage, malicious-owner
  defense, or complete filesystem TOCTOU race isolation are claimed.
- Import rejects symlinks, traversal, unsafe pickle/code assets and nonregular
  files, and verifies streaming hashes. A hostile actor concurrently replacing
  ancestor directories is outside the trusted-local-filesystem model. Native
  library/model parser vulnerabilities are not eliminated by safetensors.
- A single `O_EXCL` workspace writer lock covers operations. Runs receive one
  immutable terminal record and one per-run JSONL audit event even on caught
  adapter errors. Interrupted processes may leave a stale lock or orphan output;
  inspect/remove the lock manually only after confirming its writer is dead.
  This is not a crash-recovery daemon or tamper-proof distributed audit ledger.
- Vision, dynamic DAGs, multimodal joins, training, weight merging, model
  optimization, speech-quality metrics, adversarial correctness proofs, external
  certificate authorities, packaged UI and runtime package downloads are not
  implemented. None is silently simulated.

## Focused tests

```sh
PYTHONPATH=src python -m pytest -q tests/test_composition_v1.py
```

Tests use the explicitly labeled `fixture_text` adapter and require neither
Torch nor Transformers. They exercise schema/type/cycle rejection, content and
config fingerprints, maximum risk, run/admission separation, fixture refusal,
threshold coverage, blocked functional-code metrics, import defenses, tamper
evidence, atomic pointer replacement, known-deployment refusal, single-writer
locking, JSON CLI, measured output caps and lazy imports. They do not establish
that real ML adapters work with any particular model, measure real-model RAM,
or demonstrate an admitted real deployment. User real-model testing remains
required before such a claim.
modal joins, training, weight merging, model
  optimization, speech-quality metrics, functional-code execution, external
  certificate authorities, packaged UI and runtime package downloads are not
  implemented. None is silently simulated.

## Focused tests

```sh
PYTHONPATH=src python -m pytest -q tests/test_composition_v1.py
```

Tests use the explicitly labeled `fixture_text` adapter and require neither
Torch nor Transformers. They exercise schema/type/cycle rejection, content and
config fingerprints, maximum risk, run/admission separation, fixture refusal,
threshold coverage, blocked functional-code metrics, import defenses, tamper
evidence, atomic pointer replacement, known-deployment refusal, single-writer
locking, JSON CLI, measured output caps and lazy imports. They do not establish
that real ML adapters work with any particular model, measure real-model RAM,
or demonstrate an admitted real deployment. User real-model testing remains
required before such a claim.
usal, single-writer
locking, JSON CLI, measured output caps and lazy imports. They do not establish
that real ML adapters work with any particular model, measure real-model RAM,
or demonstrate an admitted real deployment. User real-model testing remains
required before such a claim.
loyment refusal, single-writer
locking, JSON CLI, measured output caps and lazy imports. They do not establish
that real ML adapters work with any particular model, measure real-model RAM,
or demonstrate an admitted real deployment. User real-model testing remains
required before such a claim.
