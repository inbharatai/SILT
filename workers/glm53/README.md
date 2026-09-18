# Isolated GLM-5.3-Flash worker

Runs the `internal_open_weight` evidence class for GLM-5.3-Flash
(zai-org/GLM-5.3-Flash, ~320B total / ~18B active MoE) **inside its own
runtime**, so the main SILT environment keeps its `transformers==4.51.3`
pin untouched and never imports Transformers 5.x.

## What it is

- `worker.py` — JSONL request loop over stdin/stdout using
  `asea.capability_build.worker_protocol` (protocol `silt.capability_worker.v1`).
  Ops: `hello`, `inspect`, `routing`, `mask`, `restore`, `verify`, `shutdown`.
- `Dockerfile` — image with the pinned Transformers 5.16.0 runtime (the model
  card's own `transformers_version` field).
- `requirements.lock` — the worker's own pins; **never** installed into the
  main environment.

## Honesty contract

- **Hardware preflight before anything loads.** The `hello` op admits or
  refuses with the compiler's memory formula (parameters x dtype x 2 +
  512 MiB vs available/cgroup). A ~320B BF16 teacher needs ~640 GiB; a
  laptop receives a `blocked` frame with the exact requirement + remedy —
  never a swap-death, never a fabricated trace.
- **Quantized checkpoints are refused for interventions.** The main
  zai-org/GLM-5.3-Flash repo ships FP8; packed quantized tensors cannot be
  provably restored bit-identically after a temporary mask. The remedy is
  the BF16 repository variant (zai-org/GLM-5.3-Flash-BF16).
- **Read-only teacher.** The checkpoint is mounted `:ro`; interventions are
  in-memory, hash-verified, and restored before any result is reported.
- **Usage is not causation.** Routing telemetry is recorded as usage
  evidence only; causal claims come exclusively from mask-measure-restore
  interventions with recorded seeds.

## Usage

```
docker build -f workers/glm53/Dockerfile -t silt-glm53-worker .

printf '%s\n' '{"protocol":"silt.capability_worker.v1","protocol_version":1,"op":"hello","id":"h1","payload":{}}' \
  | docker run --rm -i -v /path/to/GLM-5.3-Flash-BF16:/checkpoint:ro \
      -e GLM_CHECKPOINT=/checkpoint silt-glm53-worker
```

Expected on a small host: a `{"ok": false, "error": {"kind": "blocked",
"requirement": "GLM-5.3-Flash BF16 needs ~640 GiB ...", "remedy": ...}}`
frame. That is the correct, honest outcome — not a failure to fix.