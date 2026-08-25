# SILT — Running it on your laptop

Three tiers, in the order you should attempt them.

---

## Tier 0 — verify the harness (2 minutes, no models)

```bash
cd adaptive-skill-extraction-adapter
python3 -m pip install -r requirements.txt      # pydantic + pytest only
python3 -m pytest tests/ -q                     # all pass; 4 skipped (real-weight tests)
cd examples && python3 run_all.py               # 4 mock flows
```

The 4 skips are the real-weight tests (3 in `test_real_connectors.py`, 1 in
`test_studio.py`); set `ASEA_RUN_REAL=1` to enable them. Everything else is
offline and deterministic. If this passes, the adapter itself is sound and any later problem
is a model or connector problem — a useful thing to have established first.

---

## Tier 1 — real models via Ollama (recommended)

Ollama holds the weights in its own process, does its own quantisation, and
exposes HTTP. Your Python process never loads a tensor, so a 7B or 14B receiver
is comfortable on a 16 GB laptop.

```bash
# install ollama from https://ollama.com, then:
ollama serve                      # leave running
ollama pull qwen2.5:7b-instruct
```

```bash
pip install torch transformers sentencepiece    # for the NLLB teacher

cd examples
ASEA_RECEIVER=ollama \
ASEA_RECEIVER_MODEL=qwen2.5:7b-instruct \
python3 flow_real_assamese.py
```

Or declaratively:

```bash
PYTHONPATH=src python3 -m asea.cli run \
  --config configs/real_assamese_ollama.json --workspace .work-real
PYTHONPATH=src python3 -m asea.cli report --workspace .work-real
```

**Check Ollama first** — the connector has a `health()` method and the example
calls it, so a missing model fails immediately with `ollama pull ...` rather than
after ten minutes of inference.

### Model choices that actually matter

| Role | Model | Why |
|---|---|---|
| Receiver | `qwen2.5:7b-instruct` | the smallest Qwen worth conclusions |
| Receiver | `gemma2:9b-instruct` | second receiver, tests generality |
| Receiver | `qwen2.5:14b-instruct` | if you have ≥32 GB |
| Teacher | `facebook/nllb-200-distilled-600M` | genuinely covers Assamese + Manipuri |
| Teacher | `ai4bharat/indictrans2-indic-en-dist-200M` | stronger on Indic, needs IndicTransToolkit |

**Do not use a 0.5B receiver for conclusions.** It is a plumbing test.

---

## Tier 2 — in-process HuggingFace weights

Use this when you want a GPU, full control, or a path to LoRA later.

```bash
pip install torch transformers sentencepiece accelerate

cd examples
ASEA_RECEIVER=hf \
ASEA_RECEIVER_MODEL=Qwen/Qwen2.5-7B-Instruct \
python3 flow_real_assamese.py
```

The connector auto-selects device (`cuda` → `mps` → `cpu`) and dtype (`float16`
on GPU, `bfloat16` on CPU). Override with `device=` / `dtype=`.

If sender and receiver will not co-reside in RAM, call `sender.unload()` between
phases — both HF connectors implement it.

---

## Turn on real semantic similarity

The single highest-value upgrade. The bundled lexical metric scores
`"I rice eat"` against `"I eat rice"` at 1.0 because token-F1 ignores order.

```bash
pip install sentence-transformers          # optional; transformers alone works
ASEA_SIMILARITY=embedding python3 flow_real_assamese.py
```

**Recalibrate your thresholds afterwards.** Embedding cosine for unrelated text
sits around 0.3–0.5, not near 0, so a `sender_correctness_floor` of 0.75 that was
strict under edit distance becomes permissive. Re-tune `RelevancePolicy` and
`PromotionPolicy` or you will promote noise.

---

## Hardware reality

| | This sandbox | 16 GB laptop | 32 GB + GPU |
|---|---|---|---|
| Receiver | Qwen 0.5B (toy) | Qwen 7B via Ollama | Qwen 14B, or 7B fine-tuned |
| Teacher | NLLB-600M bf16 | NLLB-600M fp32 | IndicTrans2-1B |
| Embeddings | doesn't fit alongside | MiniLM | LaBSE |
| Per generation | 3–10 s | ~1 s | <0.3 s |
| Full real flow | ~15 min | ~2 min | seconds |

---

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `ASEA_RECEIVER` | `hf` | `hf` or `ollama` |
| `ASEA_RECEIVER_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | model id or Ollama tag |
| `ASEA_SENDER_DTYPE` | `bfloat16` | `float32` if you have RAM |
| `ASEA_SIMILARITY` | `lexical` | `embedding` for real semantics |
| `ASEA_WORKSPACE` | `.work-real` | where memory/audit are written |
| `ASEA_RUN_REAL` | unset | `1` enables real-weight tests |

---

## Before you believe any number

1. Receiver is ≥7B, not 0.5B.
2. `ASEA_SIMILARITY=embedding`, thresholds re-tuned.
3. `strict_no_mock` left at its default `True` — with real connectors it never
   needs relaxing, and if a run only passes with it off, something is a mock.
4. At least one regression suite covering a capability you are *not* targeting.
5. Native-speaker review of the Assamese data. The shipped sample is unreviewed
   and exists to exercise the pipeline.
6. Read the per-case diff, not just the aggregate delta. A packet can lift the
   average while breaking cases that already worked — this is observed behaviour
   with a real model, not a hypothetical.

## Where to look when it fails

| Symptom | Cause |
|---|---|
| `cannot reach Ollama` | `ollama serve` not running |
| `model_present: false` | `ollama pull <model>` |
| `no actionable gap` | receiver already scores above `receiver_ceiling` (0.85) |
| Everything rejected: `no_mock_provenance` | you are still using mock modules |
| Rejected: `benchmark_improvement` | the packet genuinely did not help — this is the system working |
| Rejected: `no_regression` | packet helped its target and broke a control |
| `Bodo is not covered by NLLB-200` | correct: use IndicTrans2 for `brx` |
| OOM | `dtype="bfloat16"`, or Ollama for the receiver, or `unload()` |
