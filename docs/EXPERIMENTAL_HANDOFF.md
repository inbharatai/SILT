# Local experimental handoff — SILT Phases 0–5

This is a locally implemented experimental checkpoint, not a claim that every Phase 0–5 acceptance criterion is satisfied. Phase 6 has not begun. No commits, Git remotes, pushes, public deployment or model redistribution were performed.

Baseline: inbharatai/SILT commit 39c9743fe64b7b9094ca7a86bfe1df51b2e172c7. The original checkout remains unchanged. The legacy packet-transfer core, protocol, Gate 1, Gate 2, CLI, deepapply and Spring source are retained. Narrow changes repair storage/export safety, add optional dependencies/entrypoints, and mount an experimental router only when explicitly enabled.

## What is implemented
- Storage-ID/bucket and ZIP-path confinement; safe snapshot validation before restore; staged packet/JSONL/bundle writes; shared same-root thread locking and recoverable failures.
- Separate versioned component/graph records, exact asset digests, candidate/evaluation/deployment states, local integrity checks and crash-released single-writer locking.
- Functional Python evaluation in a Linux OS-isolated runner; non-vacuous admission thresholds, named high-risk approval and recovery of interrupted activation. Existing gates are not relaxed.
- Native local text, Whisper ASR, Vits TTS and matched SmolVLM vision adapters, typed linear pipelines, resource observations and user-facing Compose CLI.
- Native Switch expert telemetry, structural pruning/router rebuilding, standalone safetensors export, fresh-process inference, and a separate source-relative text-retention certification gate. Unsupported repair training is explicitly refused.
- Suite-bound measured selection, bounded-memory dependency ZIP export, safe relocation/import as UNADMITTED with fresh local evaluation required.
- Opt-in loopback Studio workbench using real public CLI subprocesses, authenticated local writes/reads, bounded queue/logs, cancellation, evidence display and download.

## Real-model evidence, not promises
All model operations below were executed through SILT's public commands; Studio demonstrations used visible Chromium form controls. The coordinator did not supply model answers, manually mark evaluations passed or run a separate pruning implementation.

1. A pinned Switch-base-8 checkpoint (619,339,008 parameters) was physically pruned to 392,809,728 parameters by keeping four of eight experts. Serialized weight bytes: 1,238,805,814 to 785,734,056. Source unchanged. A fresh CLI process inferred while the original source-model path was temporarily unavailable, then that path was restored.
2. **Quality retention was NOT established.** On four predeclared cloze diagnostic cases, literal exact-match was 0/4 for both source and candidate. The metric penalized source punctuation (`Paris.` vs `Paris`) and the source also made substantive errors. Candidate degeneration was visible. Equal zero scores are not retention. The new certification CLI rejected the insufficient four-case suite. No compiled model was promoted.
3. Qwen2.5-Coder-0.5B-Instruct, BF16 CPU, passed all six hand-authored function tasks (four target, two control) via contained unit checks, in CLI and Studio. This is a diagnostic-only reference contract, not HumanEval/SWE-bench, a population accuracy estimate, or broad coding certification. The FP32 attempt was budget-blocked; an FP16 attempt exceeded the outer execution timeout; logs are preserved rather than replaced.
4. The MMS TTS → Whisper ASR → SmolLM2 assistant → MMS TTS path ran end-to-end on generated test speech. The voice chain's reported wall time was ~7.89 seconds and process high-water RSS ~1,144 MiB in this specific run. This is mechanism evidence on one synthetic utterance, not real-speaker/accent/noise accuracy or perceptual voice evaluation. Audio-output quality admission remains blocked.
5. Matched SmolVLM-256M answered `Red` on a synthetic red-rectangle image. Reported ~21.63 seconds and ~1,982 MiB high-water RSS. One color test is not screenshot, OCR or visual-grounding validation.
6. A dependency-inclusive Qwen diagnostic bundle was exported and imported with rebound model paths. Import correctly invalidated old admission. A fresh run succeeded while the old model directory was unavailable. Same host/runtime, NOT separate-machine or Windows validation.
7. Actual Studio form workflows exercised inspect, compiler inference, coding evaluation, explicit activation, export, rollback, constrained selection and a deliberately missing-model rejection. No JavaScript page errors were recorded. The remote browser could not reach sandbox loopback; local Chromium was used, with no public tunnel.

The raw evidence and final test totals are supplied with the separate evidence archive. Historical reviewer documents report earlier source snapshots; see the dated addendum in LOCAL_IMPLEMENTATION_STATUS.md rather than interpreting old findings as unfixed without checking their resolution.

## Models and provenance
The acquisition CLI requires an immutable 40-hex commit, inert native assets, safe paths, byte budgets and metadata-backed hashes. It never falls back to pickle or downloaded Python. Download identity is not an assertion of model quality or commercial-use rights.

| Local folder | Repository | Revision | Important restriction |
|---|---|---|---|
| SmolLM2-135M-Instruct | HuggingFaceTB/SmolLM2-135M-Instruct | 12fd25f77366fa6b3b4b768ec3050bf629380bac | Verify publisher license |
| Qwen2.5-Coder-0.5B-Instruct | Qwen/Qwen2.5-Coder-0.5B-Instruct | ea3f2471cf1b1f0db85067f1ef93848e38e88c25 | Verify publisher license |
| switch-base-8 | google/switch-base-8 | 4f9b6f805db423183c0e7c3cda7611e43364acac | Pinned safetensors conversion PR #5, NOT main; conversion equivalence to main was not independently established |
| whisper-tiny.en | openai/whisper-tiny.en | 87c7102498dcde7456f24cfd30239ca606ed9063 | English ASR demo only |
| mms-tts-eng | facebook/mms-tts-eng | c71de0fe7204c83f1c10820a7d696d0b450048ba | CC-BY-NC-4.0: noncommercial experiment; not a commercial PAI voice choice |
| SmolVLM-256M-Instruct | HuggingFaceTB/SmolVLM-256M-Instruct | 7e3e67edbbed1bf9888184d9df282b700a323964 | Matched encoder/projector/decoder only |

No weights are included in the source ZIP. The 619M source is the largest parameter-count model actually exercised here; this is not a proof that every larger model is infeasible. Hardware exposed 2 CPUs, about 4.18 GiB physical RAM and no NVIDIA GPU. Do not extrapolate to a multi-billion-parameter GLM compiler.

## Setup: existing Windows SILT
Existing core/Studio paths remain available with Windows Python (`python`, not the Microsoft Store `python3` stub):

```sh
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -e '.[dev,studio]'
python -m pytest -q -rs
silt-studio
```

Run from a disposable copy first. Never place experimental model bundles in legacy memory/ or .studio job buckets.

## Setup: new experimental workbench
The tested experimental execution target is Linux/Python 3.9. Native Windows experimental workers and the code sandbox intentionally fail closed. WSL2 is a possible environment to validate next, **not something validated in this run**. Legacy Windows support is not being removed.

From the extracted source on Linux, using a supported installed Python:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch==2.6.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,studio,localmodels]' 'peft==0.15.2'
python -m pytest -q -rs
```

The existing CUDA-guard test first resolves `HuggingFaceTB/SmolLM2-135M` config. When running fully offline in a new cache, prefetch only that public config using the official HF CLI before enabling offline flags; evidence documents this environmental prerequisite. Do not edit the test to hide the dependency.

Example explicit safe acquisition:

```sh
mkdir -p "$HOME/silt-models"
silt-artifacts download \
  --repo Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --revision ea3f2471cf1b1f0db85067f1ef93848e38e88c25 \
  --output "$HOME/silt-models/Qwen2.5-Coder-0.5B-Instruct"
python scripts/setup_local_validation.py \
  --models-root "$HOME/silt-models" --output "$HOME/silt-validation"
```

Provision only the models needed for a chosen graph. setup_local_validation.py writes fixture JSON; it neither runs models nor creates results. The default coder spec uses FP32. To reproduce the successful BF16 candidate, set `nodes[0].dtype` to `bfloat16` in a new copy of coder.json; this changes the graph fingerprint and requires fresh evaluation. Never change thresholds or reference answers to force admission.

```sh
silt-compose --workspace "$HOME/silt-runs" inspect --spec "$HOME/silt-validation/coder.json"
silt-compose --workspace "$HOME/silt-runs" evaluate \
  --spec "$HOME/silt-validation/coder.json" \
  --suite "$HOME/silt-validation/coding-diagnostic.json"
SILT_ENABLE_EXPERIMENTAL=1 SILT_EXPERIMENTAL_ROOT="$HOME/silt-experiments" \
  python -m uvicorn asea.studio.server:app --host 127.0.0.1 --port 8377
```

Open http://127.0.0.1:8377/experimental. Enter operator name, choose the operation, supply your local paths, authorize local files, and queue a job. Read raw CLI evidence and exit status. Explicit activation requires an admitted evaluation; import requires reevaluation. No result is sent to a remote service. Do not bind to 0.0.0.0 or run multiple workers on a shared root.

## Boundaries still preventing full acceptance
- No retained-quality coding specialist produced by structural pruning. Switch is seq2seq, not the target frontier coder; meaningful matched-budget quantization/compact/KD baselines remain outstanding.
- No repair training, arbitrary architecture support, nonlinear multi-input fusion, or Phase 6 neural alignment.
- Audio-output quality admission, diverse real-speaker/vision tests and broad multilingual evaluation remain unproven.
- Model RAM checks are measured/cooperative and Studio has process timeout/cancellation; they are not kernel-enforced RSS limits. Cold-load/end-to-end timing and allocator effects limit cross-model comparisons.
- Functional test harness shares an interpreter with candidate code. OS containment passed adversarial checks, but this is not an adversarial correctness oracle against code designed to forge unittest behavior.
- Reference suites are operator-authored. Six examples and a runtime version label are not independent population certification or protected final-test governance.
- Multi-process hostile filesystem interference, machine power-loss durability, native Windows experimental support, actual GPU/hardware tests and a separate-machine redeployment were not proved.

Therefore: preserve the successful mechanisms and failed evidence, keep new features opt-in, do not market this checkpoint as bulletproof or all Phases 0–5 proven, and do not proceed to Phase 6 yet.
