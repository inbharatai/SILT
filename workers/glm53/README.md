# Isolated GLM-5.3-Flash worker

Runs the `internal_open_weight` evidence class for GLM-5.3-Flash
(zai-org/GLM-5.3-Flash, ~320B total / ~18B active MoE) **inside its own
runtime**, so the main SILT environment keeps its `transformers==4.51.3`
pin untouched and never imports Transformers 5.x.

## What it is

- `worker.py` — JSONL request loop over stdin/stdout using
  `asea.capability_build.worker_protocol` (protocol `silt.capability_worker.v1`).
  Ops: `hello`, `inspect`, `routing`, `generate`, `mask`, `restore`,
  `verify`, `shutdown`.
- `Dockerfile` — image with the pinned Transformers 5.16.1 runtime plus a
  BUILD-TIME smoke test (the `glm5_next` architecture module, the
  `AutoModelForMultimodalLM` mapping and `accelerate` must all be present
  in the exact pinned wheels — the build fails instead of the first real
  run). The model card DECLARES `transformers_version` 5.16.0, but that
  wheel does not ship `models/glm5_next` (verified against the upstream
  tags); 5.16.1 is the minimal release that does. The worker records BOTH
  the declared and the installed version in its `hello` frame instead of
  pretending the pin matches the card (external audit 2026-09-19).
- `requirements.lock` — the worker's own pins
  (`transformers==5.16.1`, `accelerate==1.15.0`, wheel sha256 digests in
  the comments); **never** installed into the main environment.

## What was corrected in the 2026-09-19 audit pass

- **Model class**: the worker loads through
  `AutoModelForMultimodalLM` (there is no `Glm5NextForCausalLM` in
  transformers 5.16.1; `AutoModelForCausalLM` cannot load the composite
  vision-language checkpoint). An arch-presence check refuses by name
  BEFORE any config or weights are touched, instead of a cryptic
  `from_pretrained` failure.
- **Checkpoint identity**: `hello`/`inspect` report the sha256 of the
  checkpoint's structural files (config.json, generation_config.json,
  model.safetensors.index.json) so every downstream artifact states
  exactly which revision produced it. Weight shards are not hashed (the
  reader would run for tens of minutes); the identity pins the manifest
  that defines the shard set.
- **`generate` op**: completions under the CURRENT mask state, reported
  together with the live mask set — the measurement primitive of the
  causal-intervention protocol (mask → generate → judge → restore →
  hash-verify). The old build honestly refused the intervention loop
  ("generation+judging stage not wired"); the loop is now wired in the
  CLI (`silt-capability intervene`), with judging through the host
  oracle.
- **Router contract**: the adapter captures the ACTUAL dispatched expert
  ids/weights from the router's own
  `(router_logits, topk_weights, topk_indices)` output (sigmoid scores +
  `e_score_correction_bias` on the selection scores only, group top-k,
  pre-bias weight gather, routed scaling), and the mask rewrites the
  full tuple with the masked expert forced to `-inf` BEFORE the group
  stage — a nonzero correction bias can no longer push a masked expert
  back into the top-8.

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
- **Still open (honestly not executed anywhere in this build)**: the
  worker-side structural `reduce` operation (no reduction has been
  executed on GLM; the adapter refuses by name rather than pointing at a
  nonexistent handler), and any live run of the worker itself on
  teacher-sized hardware.

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