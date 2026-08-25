# SILT — Technical feasibility review

Written before and validated during implementation. The purpose of this document
is to be specific about what this project can honestly claim.

## Verdict

**Feasible** as a controlled skill-extraction, evaluation and promotion harness.
**Not feasible** as universal model-to-model capability copying.

The valuable part is the discipline, not the magic. Most "model teaches model"
pipelines are a loop that dumps a stronger model's outputs into a weaker model's
fine-tune. The contribution here is refusing to let a signal through until it has
been shown, on held-out data, to help without breaking something else.

## What is possible with current methods

| Capability | Status | Basis |
|---|---|---|
| Typed transfer protocol, registries, provenance, audit | **Built and working** | Ordinary systems engineering |
| Gap detection from measured benchmarks | **Built and working** | Comparative evaluation |
| L0 interaction / L1 context injection | **Built and working** | Prompt construction |
| L2 memory / RAG learning | **Built (retrieval is exact-match + compositional)** | Standard retrieval augmentation |
| L3 skill packets (glossary, lexicon, exemplars, rules) | **Built and working** | In-context learning, retrieval conditioning |
| L4 LoRA / PEFT | **Export only** — dataset + job spec emitted, nothing trained | PEFT/LoRA are mature, but training is not an in-request operation |
| L5 distillation dataset | **Export only** | Sequence-level KD is mature; same constraint |

L3 is the sweet spot and should stay the centre of gravity for a long time. It
produces measurable improvement with **zero weight changes**, which means
rollback is instantaneous and model collapse is structurally impossible: nothing
was overwritten.

## What is NOT possible, and must not be claimed

1. **One model becoming another.** You can shift behaviour on a narrow task
   distribution. Qwen conditioned on an Assamese glossary is not IndicTrans2 and
   never will be. Distillation recovers a fraction of a teacher's edge on the
   target slice and usually costs something elsewhere — which is why the
   regression sweep is mandatory rather than optional.

2. **A universal transfer *mechanism*.** The protocol is universal; the mechanism
   is not, and the code makes that seam explicit rather than hiding it. A
   pronunciation lexicon and a bug-fix pattern share an envelope, a gate and an
   audit trail. They cannot share a distillation algorithm or an evaluation
   metric, and any system claiming otherwise is either lying or doing something
   useless to both.

3. **Autonomous self-improvement.** A model trained on filtered outputs of models
   trained on filtered outputs degrades. Filters slow the rate; they do not
   change the direction. Every loop needs a human-verified or curated anchor.
   Enforced in code by `Provenance.synthetic_depth` and the
   `no_self_transfer` gate check.

4. **AGI.** Not related. Not on the roadmap. Not a framing this project accepts.

5. **Voice transfer between TTS systems.** Prosody and timbre are entangled with
   the acoustic model and vocoder. What transfers is the symbolic layer: G2P
   rules, lexicon entries, text normalisation, schwa-deletion and stress rules.
   The TTS packet declares this limit inside its own payload
   (`not_transferable: [voice_timbre, acoustic_prosody_embeddings]`).

## Environment constraints in this build

Verified, not assumed:

```
Python  3.9.25
pydantic 2.13.4     available
pytest   8.4.2      installed during the build
torch               NOT INSTALLED
transformers        NOT INSTALLED
model weights       NONE
API credentials     NONE
network to model APIs  not available
```

Consequences, stated plainly:

- **All model inference is mocked.** Every mock is a lookup table with documented
  fallback behaviour, lives in `src/asea/modules/mock/`, and reports
  `is_mock = True`. That flag propagates into packet provenance, and the default
  promotion policy rejects mock-derived packets outright
  (`test_full_run_under_default_strict_policy_rejects_mock_packets`).
- **Measured "improvements" in the demos prove the plumbing, not model quality.**
  A mock receiver improving when handed a glossary shows the packet reached it
  and was consumed. It says nothing about Qwen.
- **Similarity is lexical, not semantic.** No embedding model is available, so
  `LexicalSimilarity` combines normalised edit distance with token-F1. It reports
  `is_semantic = False`, and every evaluation report carries the caveat string.
- **Code fixes are compared as text, not executed.** Running model-authored code
  inside the process would be unsafe; the metric interface is there, the sandbox
  runner deliberately is not.

## Recommended first real MVP

Narrower than the full brief, and worth more:

> Assamese→English, sender = IndicTrans2 or a verified pair corpus, receiver =
> Qwen, mechanism = retrieved glossary and correction packets at inference time,
> no weight updates, held-out evaluation, promotion only on measured gain with no
> regression on Hindi and English.

If that produces a reproducible number with native-speaker validation, the
system is real and every other modality becomes a plugin against a proven
skeleton. If it does not, adding modalities will not help.
