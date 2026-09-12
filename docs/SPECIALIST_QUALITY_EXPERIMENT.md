# Specialist model-quality experiment automation

This guide describes the reproducible larger-model quality experiment route for
SILT's public specialist CLI/API. It is an orchestration layer around existing
SILT commands, not a separate implementation of reconstruction, recovery,
inference, grading or admission.

## Current local admission result

Candidate proposed first: `Qwen/Qwen2.5-Coder-3B-Instruct`.

Status on the probed machine: **BLOCKED before training**.

Reasons observed on 2026-09-12:

- Windows has a CUDA-capable NVIDIA RTX 5050 Laptop GPU and Windows PyTorch
  reports `torch.cuda.is_available() == True`.
- The specialist reconstruction/recovery workflow is Linux CPU-oriented and does
  not claim native Windows or CUDA recovery support.
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

`finalize` runs only when `--run-final` is supplied. That flag exists because
SILT finalization writes `final-consumed.json` before reading the final suite and
there is no retry/reset API. A final attempt must be deliberate.

Preflight-only receipt:

```sh
python scripts/run_specialist_quality_experiment.py \
  --preflight-only \
  --candidate Qwen/Qwen2.5-Coder-3B-Instruct \
  --report /local/new-preflight.json
```

Full build without final consumption:

```sh
python scripts/run_specialist_quality_experiment.py \
  --recipe /local/fresh-qwen3b-recipe.json \
  --workspace /local/new-qwen3b-study \
  --report /local/new-qwen3b-build-report.json
```

Full build plus one-shot final:

```sh
python scripts/run_specialist_quality_experiment.py \
  --recipe /local/fresh-qwen3b-recipe.json \
  --workspace /local/new-qwen3b-study \
  --final-suite /local/fresh-data/final-suite.json \
  --final-output /local/new-qwen3b-final-result.json \
  --run-final \
  --report /local/new-qwen3b-a-to-z-report.json
```

If the public SILT CLI returns `BLOCKED`, `REJECTED` or incomplete evidence, the
automation preserves the raw CLI receipt. It classifies missing local
checkpoints/data as an operator `BLOCKED` condition, while keeping the CLI's own
status field visible in the report. It does not convert missing dependencies,
resource refusal, recovery rejection or task failures into a pass.

When `--run-final` is supplied, the automation still refuses to invoke
`finalize` unless the build receipt reports both `engineering_complete: true`
and `quality_pass: true`. A completed build with failed development quality is
reported as `REJECTED` and the final suite remains unconsumed.

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

The build workflow reads only the permitted training/development files and final
metadata hashes before model work. Final answers are opened only by `finalize`
after the one-shot consumption marker is written.

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
  and must not regress controls before finalization is considered. The wrapper
  requires the public build receipt to report `quality_pass: true` before it
  will invoke the one-shot final command.
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

## Regression evidence from the implementation pass

Native Windows regression command:

```powershell
python -m venv .\work\silt-regression-venv
.\work\silt-regression-venv\Scripts\python.exe -m pip install --upgrade pip
.\work\silt-regression-venv\Scripts\python.exe -m pip install -r .\work\SILT\requirements.txt
cd .\work\SILT
..\silt-regression-venv\Scripts\python.exe -m pytest tests/ -q
```

Result: `4 skipped, 11 collection errors in 6.76s`, caused by POSIX-only
`resource`/`SIGKILL` tests and missing Studio extras. This confirms native
Windows is not the supported specialist workflow path.

Supported WSL/Linux regression command:

```sh
cd /var/tmp/silt-regression-20260912-1119/SILT
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  /var/tmp/silt-regression-20260912-1119/venv/bin/python -m pytest tests/ -q
```

Initial baseline result after correcting child-visible `dist-packages`
dependency placement: `1755 passed, 153 skipped, 16 warnings in 295.28s`.
Final branch-state result after adding this automation, strengthening the
pre-final guard, and normalizing edited text files to LF:
`1762 passed, 153 skipped, 16 warnings in 136.70s`.

Tier 0 mock examples:

```sh
cd /var/tmp/silt-regression-20260912-1119/SILT/examples
PYTHONPATH=../src /var/tmp/silt-regression-20260912-1119/venv/bin/python run_all.py
```

Result: all four mock flows completed. The mock flows are pipeline mechanics,
not model quality evidence.
