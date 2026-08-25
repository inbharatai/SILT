# SILT — Real-model run: what actually happened

Not a demo. Real weights, real inference, strict default promotion policy, no
mock bypass anywhere. Recorded verbatim because the negative result is more
informative than the mock demo was.

## Setup

```
Sender    facebook/nllb-200-distilled-600M   bfloat16, CPU   (genuinely covers Assamese)
Receiver  Qwen/Qwen2.5-0.5B-Instruct         bfloat16, CPU
Policy    DEFAULT (strict_no_mock=True — never relaxed; nothing was mocked)
Metric    LexicalSimilarity (the bundled proxy)
Hardware  2 vCPU, 4.2 GB RAM
Runtime   201 s wall clock for the full pipeline
```

## Outcome: the packet was REJECTED

```
gap        as->en   receiver=0.513  sender=0.718  headroom=0.205
extracted  12
dropped    10  (all: sender_incorrect)
distilled  1   (2 glossary entries survived)
evaluated  baseline=0.5901  candidate=0.5795  delta=-0.0106
regression hindi 0.7701 -> 0.7522 (within tolerance)
GATE       rejected
             FAILED evaluator_threshold:   0.580 (need >= 0.60)
             FAILED benchmark_improvement: -0.0106 (need >= 0.010)
```

**The system worked.** Presented with a packet that made the receiver slightly
worse, it refused to promote, wrote the rejection reason, kept the approved store
empty, and left an intact audit chain. That is the entire point of the
architecture, demonstrated against real models rather than asserted in a README.

## Finding 1 — the lexical metric was the bottleneck, not the teacher

Ten of twelve extracted signals were discarded as `sender_incorrect`. NLLB was
**not** wrong. Spot-checking the same model directly:

| Assamese | NLLB output | Reference | Lexical score |
|---|---|---|---|
| ভাত | `The rice` | `rice` | 0.60 → dropped |
| চাহ | `The tea.` | `tea` | 0.55 → dropped |
| পানী | `Water` | `water` | 1.00 → kept |
| মই ভাত খাওঁ | `I eat rice` | `I eat rice` | 1.00 |
| মোৰ নাম ৰাম | `My name is Ram` | `My name is Ram` | 1.00 |

The translations were right. A leading definite article halves token-F1 on a
one-word target, and the 0.75 correctness floor then discards a correct entry.
The proxy metric was rejecting good data.

This is the risk documented in `risk_report.md` §5 — now empirically confirmed
rather than hypothesised. Consequence: **the glossary shipped with 2 entries
instead of 12**, so the compositional path never had the coverage to help.

## Finding 2 — a 0.5B receiver is actively harmed by reference injection

With only two entries, the injected reference block confused the model:

| Expected | Baseline | With packet | |
|---|---|---|---|
| I eat rice | *"I'm sorry, but I need more context…"* | *"I'm sorry, but I can't assist…"* | +0.06 |
| **My name is Ram** | **"My name is Ram."** ✅ | **"Ram is a name."** ❌ | **−0.23** |
| I read a book | *"I'm sorry…"* | *"A boat is a ship."* | +0.10 |
| My house | *"The translation of…"* | *"The word মোৰ means water."* | +0.01 |
| Today the rice is good | "Today I feel good." | "Today I have a good breakfast." | −0.07 |

Row 2 is the one that matters. **The baseline was already correct and the packet
broke it.** A 0.5B model handed a partial glossary starts pattern-matching on the
reference table instead of translating — note it answering *"the word মোৰ means
water"*, which is simply reading the wrong row.

Two things follow:

1. Skill injection is **not monotonically beneficial**. This is precisely why the
   per-case diff, the `receiver_competent` relevance rule and the regression
   sweep exist. An aggregate delta alone would have hidden this.
2. A 0.5B model is a plumbing test. Do not draw capability conclusions from it.

## Finding 3 — the honest numbers on the mock demos

The mock flows reported deltas of +0.13 to +0.72. The real flow reported
**−0.011**. The gap between those is the entire distance between "the pipeline
moves data correctly" and "this makes a model better". Anyone quoting the mock
numbers as evidence of efficacy is misreading them, which is why every mock
report carries a `mock_warning`.

## What to change before drawing conclusions

| Change | Why | Expected effect |
|---|---|---|
| Embedding similarity (`ASEA_SIMILARITY=embedding`) | stop discarding correct entries over articles | extraction yield 2/12 → most of 12 |
| Re-tune `sender_correctness_floor` | 0.75 means something different under cosine | fewer false drops |
| Normalise references (strip leading articles) | one-word translation targets are article-ambiguous | cheaper partial fix |
| Receiver ≥ 7B via Ollama | 0.5B cannot use a glossary | the actual open question |
| Native-speaker review of the suite | the sample data is unreviewed | trustworthy references |

The first of these was run as a follow-up; see the appendix below.

## Reproduce it

```bash
pip install torch transformers sentencepiece
cd examples
python3 flow_real_assamese.py                      # lexical, as recorded above
ASEA_SIMILARITY=embedding python3 flow_real_assamese.py
ASEA_RECEIVER=ollama ASEA_RECEIVER_MODEL=qwen2.5:7b-instruct python3 flow_real_assamese.py
```

Every run writes a hash-chained audit log; `python3 -m asea.cli audit --workspace
.work-real` replays it.

---

# Appendix — the embedding follow-up, and two bugs it found

## Bug 1: the pipeline used two different similarity backends

The first embedding run produced **byte-identical drop scores** to the lexical
run. Cause: `Pipeline` threaded an injected similarity backend into the
`BenchmarkHarness` but `RelevanceFilter` silently constructed its own
`LexicalSimilarity`. The run filtered with a proxy while scoring with embeddings
and would have been reported as semantically evaluated.

Fixed: `Pipeline` now takes a `similarity` argument and the relevance filter is
derived from `harness.similarity`, so they cannot drift.
`test_pipeline_uses_one_similarity_backend_everywhere` asserts all four
components share one object.

An inconsistency like this is exactly the kind of thing that produces a
confidently wrong experimental result, and it was only visible because two runs
were compared side by side.

## The corrected result: the metric flipped the verdict

| | Lexical proxy | Embedding (MiniLM) |
|---|---|---|
| Signals dropped | **10 / 12** (all `sender_incorrect`) | **5 / 12** (4 incorrect + 1 `receiver_competent`) |
| Glossary entries | 2 | **7** |
| Baseline → candidate | 0.5901 → 0.5795 | 0.5943 → **0.6475** |
| Delta | **−0.0106** | **+0.0532** |
| Hindi regression | 0.7701 → 0.7522 ok | 0.7757 → 0.7714 ok |
| **Gate** | **REJECTED** | **PROMOTED** |

Same models, same data, same policy. Only the similarity backend changed, and it
turned a rejected transfer into a promoted one — by no longer discarding correct
teacher output over a definite article. Finding 1 is confirmed: the lexical proxy
was the bottleneck.

Note also that `receiver_competent` fired for the first time under embeddings. It
correctly identified that the receiver already handled a case, which the lexical
metric had scored too low to notice.

## Bug 2 (design gap): an aggregate gain hid a real per-case regression

The promoted run still contains this:

```
expected : My name is Ram
baseline : My name is Ram.        <- already correct
+packet  : Ram                   <- broken by the packet
           -0.2665 WORSE
```

Five of six cases improved, one got materially worse, and because the *average*
rose by 0.053 every gate check passed. The packet was promoted while carrying a
regression on a case that already worked.

The cross-capability regression sweep could not catch this: it compares whole
suites, and this damage is inside the target suite.

Fixed by adding a fourteenth gate check, `case_regression_limit`, governed by
`PromotionPolicy.max_case_regression_ratio`:

```python
PromotionGate(PromotionPolicy(max_case_regression_ratio=0.2))
```

The default is `1.0`, preserving the previous aggregate-only behaviour so no
existing configuration changes meaning. With the real run's numbers (1 of 6 =
0.167) a limit of 0.2 permits it and a limit of 0.1 refuses it — a judgement
call that now at least *has* a dial.

## What these three findings add up to

1. The gate does its job with real models: it rejected a harmful packet
   unprompted, then promoted a helpful one once the metric was fixed.
2. The evaluation metric mattered more than either model. Swapping it changed the
   verdict on identical inputs. Anyone running this with the lexical default and
   concluding "skill transfer doesn't work" would be measuring their metric.
3. Averages hide harm. The per-case diff was the only thing that surfaced the
   broken case, which is why it is now in the audit record and the gate.

None of this was visible from the mock demos, which reported deltas of +0.13 to
+0.72 and promoted everything.

---

# Appendix 2 — cross-family receiver: NLLB teaches SmolLM2-360M

Testing whether the adapter generalises beyond the Qwen family. Gemma weights
are license-gated on HuggingFace (401 without an accepted license + token) and
Kimi has no locally runnable weights (K2 is ~1T parameters, API-only), so the
cross-family receiver is **SmolLM2-360M-Instruct** — a genuinely different
lineage: different tokenizer, different chat template, different training mix.
No connector code changed; the same `HFCausalConnector` loaded it.

## Result: REJECTED on three independent grounds

```
gap        as->en   receiver=0.405  sender=0.784  headroom=0.379
extracted  12 -> 7 glossary entries survived (embedding metric)
evaluated  baseline=0.3962  candidate=0.3919  delta=-0.0042
regression hindi_english_v1  0.5031 -> 0.4801  REGRESSED
GATE       rejected
             FAILED evaluator_threshold     0.392 < 0.60
             FAILED benchmark_improvement   -0.0042 < 0.010
             FAILED no_regression           hindi damaged
```

## What actually happened inside the model

SmolLM2-360M cannot translate Assamese **at all** — its baseline behaviour is
to echo the Assamese input back verbatim. Handed a 7-entry glossary, it did not
learn to translate; it began splicing glossary words into its echo:

```
expected : My house
baseline : মোৰ ঘৰ                      (echo)
+packet  : মোৰ ঘৰৰ চাহৰ কিতা…          (echo + "tea" + "book" — babble)

expected : My name is Ram
baseline : মোৰ নাম ৰাম                 (echo)
+packet  : মোৰ নাম চাহ                 ("my name is TEA" — wrong row again)
```

3 of 6 held-out cases got worse, and the injected reference block distracted it
enough to damage Hindi→English too — a genuine cross-capability regression that
the sweep caught exactly as designed.

## The three-receiver picture

| Receiver | Family | Baseline behaviour | With packet | Gate verdict |
|---|---|---|---|---|
| Qwen2.5-0.5B (lexical metric) | Qwen | refuses / partial | slightly worse | **REJECTED** |
| Qwen2.5-0.5B (embedding metric) | Qwen | refuses / partial | genuinely better (+0.053) | **PROMOTED** |
| SmolLM2-360M (embedding metric) | SmolLM | echoes input | babbles glossary words, hurts Hindi | **REJECTED — 3 failures** |

This is the accuracy claim SILT can honestly make: **not** "any model learns
Assamese", but "the adapter correctly discriminates between a receiver that can
use a skill and one that cannot, without any per-model configuration". The
universal layer ran identically for a new model family; the *gate*, not the
plumbing, decided the outcome — which is the design.

## Receiver capacity floor (empirical)

Two sub-1B models, two failure styles: Qwen-0.5B pattern-matched the reference
table; SmolLM2-360M ignored the task and babbled. In-context skill consumption
appears to need more capacity than either has. The open question remains ≥7B,
which requires more RAM than this sandbox (see LOCAL_SETUP.md, Tier 1).

---

# Appendix 3 — REAL medical AI→AI transfer (any-domain proof)

The "any learning from any AI" claim, exercised with two real language models
and zero mocks: **Qwen2.5-0.5B** (medically stronger) teaching **SmolLM2-360M**
(medically weaker) triage red-flag skills. Same core pipeline that moved the
Assamese glossary; only the registered modules and benchmark differ.

## Run 1 — default thresholds: everything dropped, and rightly reported

All four extracted signals fell to `sender_incorrect` at cosine 0.18–0.45
against a 0.75 floor. Inspecting the sender's actual answers showed they were
**substantively correct** ("Call emergency services immediately") but verbose —
the floor was calibrated for terse translations, not free-form advice. One
deliberate, documented calibration followed (floor 0.35 for this flow), safe
because distillation teaches the *verified reference*, never the sender's prose,
and medical still cannot auto-promote.

## Run 2 — calibrated: the full loop, including the human gate

```
gap        triage/structured/medical/en  receiver=0.564 sender=0.618
extracted  4 -> 1 dropped (sender_incorrect at 0.18) -> 3 rules distilled
evaluated  baseline=0.6419  candidate=0.9116  delta=+0.2697  (all 4 vignettes improved)
gate       PENDING_HUMAN  ("domain 'medical' is high risk; approver=none")
           -> named approval by dr.reviewer@example.org -> PROMOTED
audit      intact, 14 entries
```

**The one dropped signal was the one genuinely weak answer.** For "fever in an
infant under three months" Qwen produced "Mild fever… ensure the infant is
comfortable" — a downplaying answer that is exactly what must not be taught.
At the calibrated floor the three correct-escalation answers passed and this
one still failed at 0.18. The filter separated good expertise from bad within
the same real sender.

**The human gate fired for real.** Despite +0.27 improvement and every other
check passing, the packet parked at `PENDING_HUMAN` until a named approver
acted. No score can bypass this.

**Honest caveat.** On the infant-fever vignette (whose specific rule was
dropped), the conditioned receiver fell back to the generic emergency rule —
directionally safe escalation, but not the paediatric-specific action, and the
semantic metric scored it as an improvement anyway. Rule-matching precision
under partial coverage is a real limitation for high-stakes use; one more
reason the human reviewer sees the packet before it ships.

## The generality scoreboard (all real, no mocks)

| Transfer | Sender → Receiver | Modality | Outcome |
|---|---|---|---|
| Assamese glossary | NLLB-600M → Qwen-0.5B | text | PROMOTED (+0.053) after metric fix |
| Assamese glossary | NLLB-600M → SmolLM2-360M | text | REJECTED ×3 (receiver too weak) |
| Medical triage rules | Qwen-0.5B → SmolLM2-360M | structured | PENDING_HUMAN → human-approved (+0.270) |

Three domains of evidence, one unchanged core: the adapter moved language skill
model→model, refused a transfer that harmed, and routed a medical transfer
through mandatory human approval. That is "any learning from any AI" as far as
it can honestly be claimed at L0–L3 — with per-modality distillers and
per-domain policies doing the specialisation, exactly as designed.
