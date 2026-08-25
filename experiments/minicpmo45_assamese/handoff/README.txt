SILT skill bundle -- the "trained model" download
=================================================

SILT trains no weights. There is no trainer in this repository and there never
will be one inside the adapter. The promoted artefact is an inspectable SKILL
PACKET (JSON: a lexicon, glossary, rule list, or exemplar set) that a receiver
model conditions on at inference time. So "the trained model" is really
<receiver model> + <approved skill packet(s) in this bundle>.

What is in this archive
-----------------------
  approved/<packet_id>.json   the raw promoted skill packet(s) -- the primary
                              artefact; this is what the receiver learns from.
  manifest.json               bundle index: per-packet capability, target,
                              learning level, provenance chain, gate verdict.
  <name>.jsonl                a supervised dataset flattened from the approved
                              packets (one row per entry/rule/example).
  <name>.manifest.json         dataset manifest (row/packet counts, skips).
  <name>.job.json              OPTIONAL -- only present if a base model was
                              supplied. A NOT_EXECUTED training-job spec for an
                              external L4/L5 trainer (LoRA / sequence KD). It is
                              a recipe, not a trained adapter.
  audit.jsonl                  OPTIONAL -- the hash-chained audit trail of the
                              run that produced these packets, if available.

How to use the skill packet (L0-L3, the primary path)
-----------------------------------------------------
At inference time the receiver injects the packet's redacted payload into its
system prompt via render_skills -- exactly the path the gate measured:

    skills = [packet.redacted_for_receiver() for packet in approved]
    answer = receiver.infer_with_skills(capability, prompt, skills)

No training, no weight surgery. The receiver is "conditioned", not retrained.
To consume the packet from a different runtime, parse approved/<id>.json and
emit its `distilled_skill` payload as the prompt prefix your model expects.

How to use the dataset + job spec (L4/L5, opt-in)
-------------------------------------------------
Take <name>.jsonl and <name>.job.json to an external trainer (HuggingFace TRL /
PEFT, an Ollama Modelfile, etc.). The job spec's `eval_gate` is the bar the
resulting adapter must clear before anyone ships it. Re-enter the trained
adapter as a NEW receiver module and re-benchmark it through SILT before use.

Honesty
--------
No "% of knowledge transferred" appears anywhere because no such measurement
exists. The only defensible claim is the one the gate made: the receiver's
held-out score rose by a measured amount after conditioning on this packet, with
no uncontrolled regression, on real (non-mock) weights.
