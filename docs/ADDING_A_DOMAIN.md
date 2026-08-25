# Adding a new domain — the universality guide

SILT is built to be **domain-agnostic**: the pipeline core has no `if domain ==
...` branches and no `if modality == ...` branches. Adding a new domain — say,
*a medical-expert AI teaching a weaker medical assistant* — is mostly writing
**data files**, not editing core code. This guide walks through it concretely
with the medical example, reusing the scaffolding that is already wired.

> Read alongside [`connector_authoring.md`](connector_authoring.md) (how to add
> a *connector* / a new *modality*) and the main [`../README.md`](../README.md)
> for the pipeline overview.

## The two extension axes

SILT has two independent extension axes. Know which one you are on:

| Axis | What you change | When | Code edits? |
|---|---|---|---|
| **New domain / task** (e.g. medical triage, legal, finance) | A `CapabilityKey`, a benchmark suite, maybe a corpus, two catalog entries | You want to transfer a skill in a field SILT doesn't ship a suite for | **Almost none** — reuse an existing modality |
| **New modality** (e.g. OCR, audio-ASR) | An `Extractor` + `Distiller` + optional `MetricPlugin`, registered in the plugin registry | Your data does not fit TEXT / CODE / SPEECH_TTS / STRUCTURED | Yes — see `connector_authoring.md` §"Adding a new modality" |

**Most new domains are the first axis.** Medical is: it reuses
`Modality.STRUCTURED`, which already has `StructuredExtractor` +
`StructuredDistiller` registered and a `MEDICAL/LEGAL/FINANCE` system-prompt
branch. The medical scaffolding already ships — this guide mostly shows you how
to *reuse* it for your own medical sub-task.

## What already ships for medical (reuse, don't rebuild)

| Artifact | Path | What it gives you |
|---|---|---|
| `Domain.MEDICAL` + `HIGH_RISK_DOMAINS` | `src/asea/core/protocol.py:46,62` | Medical is declared high-risk → the gate's `human_approval` check fires automatically (hard, non-configurable) |
| `TRIAGE` CapabilityKey | `src/asea/studio/catalog.py:31-34` | A template structured-medical capability |
| `StructuredExtractor` / `StructuredDistiller` | `src/asea/extraction/extractors.py:137`, `src/asea/distill/strategies.py:226` | Already registered for `Modality.STRUCTURED`; the distiller emits `PacketType.RULE` payloads |
| `system_for_capability` medical branch | `src/asea/modules/real/prompting.py:78-85` | Cautious triage-assistant instruction fires automatically for any `Domain.MEDICAL` capability |
| `render_skills` rules branch | `src/asea/modules/real/prompting.py:112-115` | Formats `if [condition] then: action` for RULE packets |
| `CorpusSender` | `src/asea/modules/real/corpus.py` | File-backed real sender, `is_mock=False`, `synthetic_depth=0` — the safest teacher provenance for high-risk domains |
| `medical_triage.json` suite | `data/benchmarks/medical_triage.json` | Suite schema template with extraction/heldout splits |
| `triage_redflags.json` corpus | `data/corpora/triage_redflags.json` | Corpus schema template |
| `flow_real_medical.py` | `examples/flow_real_medical.py` | End-to-end real AI→AI medical example with calibrated `relevance_floor` and the human-approval step |
| `triage-corpus` catalog entry | `src/asea/studio/catalog.py` (`_triage_corpus`) | Studio entry wiring `CorpusSender` against the triage corpus |
| Gate `human_approval` check | `src/asea/promotion/gate.py:284-295` | Hard, non-bypassable; fires for any high-risk domain packet |

## Worked example: a medical-expert AI teaches a weaker medical assistant

You have a strong medical model (e.g. a big instruct model that gives good
red-flag guidance) and a smaller assistant you want to improve at recognising
red-flag presentations. Here is the whole procedure.

### Step 1 — Declare the capability (usually no enum edit)

A `CapabilityKey` is `task_type / modality / domain / language`. For a medical
red-flag task, reuse `Domain.MEDICAL` and `Modality.STRUCTURED`; only `task_type`
is yours (a free-form string):

```python
# in src/asea/studio/catalog.py, near the existing TRIAGE helper:
def _medical_redflag() -> CapabilityKey:
    return CapabilityKey(
        task_type="symptom_redflag",      # free-form, yours
        modality=Modality.STRUCTURED,     # reuse — already has plugins
        domain=Domain.MEDICAL,            # reuse — already high-risk
        language="en",
    )
```

**You only edit the enums** (`src/asea/core/protocol.py:39-62`) if your domain is
genuinely not in the `Domain` list, and you only add it to `HIGH_RISK_DOMAINS`
if it should force human approval. Medical already is — no edit.

### Step 2 — Plugins (skip — already registered)

The plugin registry is keyed by `Modality` only
(`src/asea/core/plugins.py:70`). `StructuredExtractor` + `StructuredDistiller`
are already registered for `Modality.STRUCTURED`. **No registration needed.**
(If you were on the *new modality* axis instead, you would subclass
`BaseExtractor`/`BaseDistiller` and call `register_extractor`/`register_distiller`
— see `connector_authoring.md`.)

### Step 3 — System prompt (usually no edit)

`system_for_capability` already has a branch for
`domain in (MEDICAL, LEGAL, FINANCE)` that emits a cautious triage-assistant
instruction (`prompting.py:78-85`). Your `Domain.MEDICAL` capability gets it
automatically. Add a new branch **only** if your domain needs a distinct
instruction (e.g. legal disclaimers).

### Step 4 — Write a benchmark suite

Create `data/benchmarks/<your_suite>.json`. The harness enforces a strict
pydantic schema (`src/asea/benchmarks/harness.py:30-57`, `extra="forbid"`):

```json
{
  "suite_id": "symptom_redflag_v1",
  "description": "SAMPLE DATA, NOT CLINICALLY REVIEWED. ...",
  "task_type": "symptom_redflag",
  "modality": "structured",
  "domain": "medical",
  "language": "en",
  "cases": [
    {"case_id": "rf01", "prompt": "crushing chest pain, sweating, nausea",
     "expected": "Red flag for cardiac event. Seek emergency care now.",
     "split": "extraction", "meta": {"human_verified": true, "category": "cardiac"}},
    {"case_id": "rh01",
     "prompt": "A 60-year-old reports crushing chest pain with sweating and nausea.",
     "expected": "Red flag for cardiac event. Seek emergency care now.",
     "split": "heldout", "meta": {"human_verified": true}}
  ]
}
```

Rules:
- `split` must be one of `extraction`, `heldout`, `regression`
  (`harness.py:27`). The pipeline scores on `extraction` (gap + extraction),
  evaluates on `heldout` (baseline vs candidate), and sweeps `regression` for
  non-targeted control capabilities.
- `meta.human_verified: true` makes the extractor tag the packet
  `OriginKind.CURATED_CORPUS` with `synthetic_depth=0` — the honest provenance.
  `meta.category` surfaces as the rule's `category` after distillation.
- Mark it `SAMPLE DATA, NOT CLINICALLY REVIEWED` in the description — the
  shipped medical suite does, and so should yours unless a clinician reviewed it.

### Step 5 — (Optional, recommended for high-risk) write a corpus teacher

For high-risk domains the safest teacher is a **reviewed file corpus**, not a
model — `CorpusSender` has `is_mock=False`, `synthetic_depth=0`, and returns
`"<not-in-corpus>"` for misses rather than hallucinating. Write
`data/corpora/<your_corpus>.json`:

```json
{
  "corpus_id": "symptom_redflag_corpus_v1",
  "reviewed_by": null,
  "note": "SAMPLE DATA, NOT CLINICALLY REVIEWED.",
  "records": [
    {"prompt": "crushing chest pain", "answer": "Red flag for cardiac event. Seek emergency care now.", "confidence": 0.9}
  ]
}
```

If you *do* want a model teacher (e.g. a big expert instruct model), skip the
corpus and use `OllamaConnector`/`HFCausalConnector` as the sender in step 6.
That is real and supported — just note the model teacher's output flows through
the relevance filter's `sender_correctness_floor`, which may need lowering (step 7).

### Step 6 — Wire the sender and receiver into the catalog

Add two entries to `CATALOG` in `src/asea/studio/catalog.py`, mirroring the
existing `tts-teacher-as` / `tts-learner-zero` pair and the `_ollama(...)` factory
pattern. **Additive only** — do not change existing entries:

```python
"medical-expert": {
    "factory": _ollama("your-medical-expert-model", roles=["sender"],
                       capabilities=[_medical_redflag()]),
    "roles": ["sender"],
    "description": "Strong medical instruct model as the red-flag teacher; real weights",
    "requires": "ollama serve + ollama pull <your-medical-expert-model>",
},
"medical-learner": {
    "factory": _ollama("your-smaller-assistant", roles=["receiver"],
                       capabilities=[_medical_redflag()]),
    "roles": ["receiver"],
    "description": "Weaker assistant that learns red-flag rules from the expert",
    "requires": "ollama serve + ollama pull <your-smaller-assistant>",
},
```

If you used a corpus teacher instead, wire it like `triage-corpus`:
`CorpusSender(corpus_path=CORPORA / "<your_corpus>.json", capabilities=[_medical_redflag()], ...)`.

> `build()` (`catalog.py:build`) structurally refuses mocks — so a real teacher is
> enforced, never a lookup table.

### Step 7 — Tune `relevance_floor` (the one per-request knob)

The relevance filter drops a packet as `sender_incorrect` if the teacher's output
similarity to the expected answer is below `sender_correctness_floor`, **default
0.75** (`src/asea/filters/relevance.py:27`). Medical/triage answers are verbose —
a correct "Red flag. Arrange urgent emergency assessment and contact emergency
services" scores *low* on embedding cosine against a short expected string, so
the default 0.75 drops nearly everything as `sender_incorrect`. The shipped
medical example lowers it to **0.35** (`examples/flow_real_medical.py:72-85`).

This is a **per-request** parameter, not a policy edit. Pass it when you start the
transfer:

```bash
curl -X POST localhost:8377/api/transfers \
  -H 'content-type: application/json' \
  -d '{"sender":"medical-expert","receiver":"medical-learner",
       "suites":["symptom_redflag"],"similarity":"embedding",
       "relevance_floor":0.35}'
```

**Do not** touch `receiver_ceiling` / `min_headroom` (`core/gap.py:27-39`) or
any promotion threshold (`promotion/gate.py:29-55`). They are per-Pipeline / not
exposed per-request in the Studio, and the defaults are correct. The
`relevance_floor` is the only knob you should turn, and only when the teacher's
output style legitimately scores low against short references.

### Step 8 — Run, approve, download

```bash
# 1. Start the transfer (above). Poll until status = "done":
curl -s localhost:8377/api/transfers/<job_id>

# 2. Medical is high-risk, so the gate parks the packet in PENDING_HUMAN
#    (the human_approval check is hard and non-configurable). Approve by name:
curl -X POST localhost:8377/api/transfers/<job_id>/approve \
  -H 'content-type: application/json' \
  -d '{"packet_id":"<from report.pending_human>","approver":"you@example.org"}'
#    Approval re-runs the FULL gate — it satisfies exactly one check and
#    waives nothing else.

# 3. Verify the audit chain is intact:
curl -s localhost:8377/api/transfers/<job_id>/audit   # integrity.ok == true

# 4. Download the "trained model" — the approved RULE skill packet bundle:
curl -s -o medical_bundle.zip localhost:8377/api/transfers/<job_id>/export
#    Or via the CLI against the workspace:
#    PYTHONPATH=src python -m asea.cli export --workspace .studio/<job_id>
```

The bundle is a zip of the approved skill packet(s) + a dataset + manifest + the
audit chain + an honest README. **SILT trains no weights** — the "trained
model" is `<receiver> + <approved skill packet>`. At inference time the receiver
conditions on the packet via `render_skills`. See the README's "Downloading the
trained model" section.

## Escaping to a new modality

If none of TEXT / CODE / SPEECH_TTS / STRUCTURED fits your data (e.g. you are
transferring OCR post-processing skill, or audio-ASR correction), you are on the
*new modality* axis: subclass `BaseExtractor` and `BaseDistiller`, optionally a
`MetricPlugin`, and register them in `default_registry()`
(`src/asea/core/plugins.py:70`). The core stays untouched — `tests/test_conformance.py`
proves this by registering a brand-new OCR modality with no core edit. Full
instructions are in [`connector_authoring.md`](connector_authoring.md) §"Adding a
new modality".

## What you must never edit (the trust contract)

- `src/asea/promotion/gate.py` — the 14 checks, the `human_approval` hard rule,
  `strict_no_mock`, and all thresholds. These are the trust contract.
- `src/asea/filters/` policy defaults beyond the documented `relevance_floor`
  per-request override.
- `src/asea/modules/mock/` — mocks are for unit tests only and are not reachable
  through the Studio (`catalog.py:build` refuses them structurally).
- The core `if Modality` / `if domain` invariant — there intentionally isn't one.

If you find yourself wanting to relax a gate check to make a transfer "succeed",
that is the gate doing its job. The correct response is better data, a better
teacher, or a calibrated `relevance_floor` — never editing the gate.