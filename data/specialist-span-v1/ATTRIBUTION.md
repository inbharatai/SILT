# Specialist span pilot v1: origin and permission

Source: **locally authored synthetic, AI-generated fixtures**, created specifically for this controlled source/reconstruction/recovery mechanics pilot. The prompts and labels are short original constructions, not borrowed passages, model-output labels, old case rewrites, or excerpts from copyrighted sources. Reference adjectives were fixed by the author and are literally present in each prompt before its missing span is reconstructed.

These fixture files are offered under **CC0 1.0 Universal (CC0-1.0)**: https://creativecommons.org/publicdomain/zero/1.0/ . To the extent the fixture contributor holds applicable rights, those rights are dedicated under CC0; its fallback license applies where applicable. This is a fixture-specific permission statement, not a change to the repository implementation license or any model/tokenizer license.

No human review, external annotation, independent benchmark curation, or legal clearance is claimed. Copyrightability of generated text, third-party rights, and legal sufficiency are not guaranteed; no warranty is offered. No model-generated answers were consulted or retrofitted. Historic diagnostic and calibration files were read solely for negative overlap checks, not as examples to relabel or copy.

## Deliberate limitations

- Forty rows, a shared four-adjective answer vocabulary, and four reused grammatical templates. Entity nouns are split-disjoint, but grammar and target vocabulary deliberately are not.
- Seed label `271828` identifies this fixed authoring/protocol version. Membership and order are explicit in the JSON files, not reconstructed by a random split. Future recovery uses its own documented deterministic shuffle with that seed; this is not evidence of statistical independence.
- Teacher ability to emit complete sentinel-delimited responses is **unvalidated**. Context redundancy makes the intended local copying task explicit, not guaranteed easy for a particular checkpoint.
- The authoring worker inspected final fixtures only to construct and statically validate them. Parent/operator must not read `final.json` until a later explicitly authorized final stage. This procedural quarantine is not encryption or access-control enforcement.
