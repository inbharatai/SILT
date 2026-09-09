# Specialist-v1: fresh, licensed short-function data

## Status and explicit capability contract

**Frozen data only; no model weights loaded, model generation, training, benchmark
run, reference-code execution, network acquisition, or model-output inspection.**

The primary target is **short pure Python flat-string/list operations**: transform,
filter, extract, count, test membership/conditions, or reduce a sequence. Inputs
and expected outputs are supported JSON values; the selected cohort uses flat
strings/lists and scalar strings, bounded integers, booleans or null. Preserve the
original required function name and parameter order, return the requested value,
and perform no external I/O or persistent state. Nearby controls are basic
straight-line integer arithmetic/predicates, not advanced mathematics.

A hypothetical public-source workload is a function that extracts or counts
characters, replaces a string component, or interleaves flat lists. This describes
the scope, **not a performance result**. Qwen2.5-Coder-0.5B-Instruct's capability is
**UNVALIDATED**. Short original code is a preregistered suitability hypothesis;
it does not demonstrate that the teacher can solve these tasks. Establish teacher
capability later using TRAIN/development only. There is no zero-error guarantee,
no presumption that every original reference is bug-free, and no final retuning.

## Delivered files and actual counts

Repository root: `/agent/workspace/silt-specialist-core`.
Data root: `/agent/workspace/silt-specialist-core/data/specialist-v1`.
All paths below are relative to the repository root.

| File | Purpose | Unique tasks | Target / control | Original literal cases |
|---|---|---:|---:|---:|
| `data/specialist-v1/train.json` | Canonical-reference supervised training | 42 | 40 / 2 | 128 in `train-cases.json` |
| `data/specialist-v1/calibration.json` | Calibration, deliberate TRAIN-only subset | 32 | Seeded subset | Same source cases as TRAIN |
| `data/specialist-v1/validation.json` | Development/validation references | 8 | 6 / 2 | 24 |
| `data/specialist-v1/final.json` | Locked final references; do not train/tune/read casually | 16 | 12 / 4 | 58 |
| `data/specialist-v1/validation-suite.json` | `EvaluationSuite` development input | 8 | 6 / 2 | 24 |
| `data/specialist-v1/final-suite.json` | Locked final `EvaluationSuite` | 16 | 12 / 4 | 58 |

There are **66 independent selected task families**, **210 retained original
assertions**, and **8 basic numeric controls overall**. Calibration is not an
additional held-out split. All 32 calibration records equal their TRAIN records.
Validation and final are disjoint from TRAIN/calibration and from each other by
source ID, operational family, canonical prompt/source, alpha-normalized source,
response and source-record fingerprints. No synthetic functions, repeated tasks,
near-clone augmentation, rewritten references, or synthesized cases were added.

Requested training size was 64, but the final conservative contract yielded only
67 eligible independent families: 58 target and 9 control. Reserving 6/2 validation
and 12/4 final leaves 40 target training tasks. Exactly two training controls are
retained, giving **42 TRAIN**, a **22-task shortfall**. One extra eligible numeric
control is left unused rather than changing the target/control design or padding
TRAIN. The small cohort is a real limitation, not a reason to relax exclusions.

Other artifacts:

- `selection-lock.json`: seed, source-prefixed IDs, original source IDs, split,
  group/domain, family, fingerprints, token lengths, case statistics, prior
  consumed IDs and exact tokenizer fingerprint. No reference bodies or expected
  values; appropriate for orchestration without reading final answers.
- `manifest.json`: contract, counts, exclusions by reason, source pins, checksums,
  environment, fairness limitations and governance.
- `eligibility-audit.json`: exhaustive 591-row source audit, parser omissions,
  deterministic exclusion reasons and family edges. Does not contain model data.
- `train-cases.json`, `validation-provenance.json`, `final-provenance.json`:
  retained JSON cases and exact upstream assertion strings. Final provenance is
  held-out answer material and must stay outside training/tuning inputs.
- `sources/**`: unchanged pinned source snapshots, original MIT license,
  CC-BY-4.0 legal text, publisher license card and README.
- `ATTRIBUTION.md`: retained original author/license notices and indication of
  specialist-v1 modifications.
- `SHA256SUMS`: every generated data file except the checksum list itself,
  including `manifest.json`. The manifest additionally fingerprints the builder
  and imported audited parser.

## Data API — reconstructor contract (read before wiring)

Each of train/calibration/validation/final is an object:

```json
{
  "schema": "silt.specialist.supervised.v1",
  "purpose": "train",
  "split_policy": "standalone_LOCAL_not_official_benchmark_split",
  "samples": [
    {
      "id": "source-prefixed-original-id",
      "prompt": "code-only instruction + required entrypoint + original prompt",
      "response": "complete unchanged original reference code"
    }
  ]
}
```

The example is structural, not an invented training record. Samples also carry
`source`, `source_task_id`, `license`, `entrypoint`, `function_parameters`,
`family`, `group`, `domain`, `original_prompt`, SHA256 fields, `token_lengths`,
`operation_tags` and `case_statistics`. These extras are provenance: **do not
append them, hidden assertions, or expected outputs to model input**.

- MBPP response = exact original `code` string.
- HumanEval response = exact original `prompt + canonical_solution`. The prompt
  provides imports/signature/docstring; the canonical solution supplies its body.
  This intentionally repeats the original task docstring in the full response.
- No code prettification, import removal, solution shortening, synthetic
  comments, rewritten function names, or truncated ground truth.
- `validation.json` is the development dataset; there is no separate `dev.json`
  alias that could silently drift from it.
- Evaluation suites have `schema_version: 1`, `claims: ["coding"]`,
  `policy_version: "composition-admission-v2"`, and cases with `metric:
  "function_io"`, `threshold: 1.0`, `output_format: "raw_python"`. Their
  `function_cases` have exactly `id/function/args/kwargs/expected`. Suite references
  are attribution descriptions, not replacement source code.
- Both suites validate against the existing `asea.compose.schema.EvaluationSuite`
  and the external function oracle's JSON-data validator without executing code.

## Exact offline tokenizer and no truncation

Tokenizer only: `/agent/workspace/silt-models/Qwen2.5-Coder-0.5B-Instruct`.
`AutoTokenizer.from_pretrained(..., local_files_only=True,
trust_remote_code=False)` loads **no model weights**. Its four tokenizer-file
hashes are hard-pinned in the builder and recorded in the lock/manifest.

Environment used: Python **3.9.25**, Transformers **4.51.3**, tokenizers **0.21.4**,
`Qwen2TokenizerFast`, EOS **151645**.

Every selected task must satisfy **both** complete encodings:

1. Flat: `tokenizer.encode(prompt + "\n" + response,
   add_special_tokens=False) + [tokenizer.eos_token_id]`, length <= **256**.
2. Chat: `tokenizer.apply_chat_template([{role: "user", content: prompt},
   {role: "assistant", content: response}], tokenize=True,
   add_generation_prompt=False)`, length <= **256**.
3. Complete response alone, `add_special_tokens=False`, <= **192** tokens.

The Qwen chat template may supply its default system message; the actual template
bytes are pinned by `tokenizer_config.json`. Any additional system instruction,
prompt wrapper, assistant prefill, markdown fence, or different chat template
invalidates this token-length assurance. Recheck exact encoding before wiring.

| Split | Maximum chat tokens | Maximum flat + EOS | Maximum response tokens | Tasks above 128 chat tokens |
|---|---:|---:|---:|---:|
| TRAIN | 220 | 190 | 149 | 8 |
| Validation | 169 | 139 | 98 | 3 |
| Final | 256 | 227 | 108 | 5 |

**Do not use `max_length=128` with truncation.** It would cut complete examples.
Use a 256-token training bound, reject oversize examples rather than truncate,
and pad only after checking lengths. Tokenize the complete combined text; do not
assume separately tokenized prompt and response concatenate identically. For
assistant-only label masks, account for tokens spanning the prompt/response
boundary (e.g., offset mappings) rather than assuming a separately encoded
prompt's token count is exact. Keep labels for all response content and EOS under
the selected reconstructor encoding. No training implementation is supplied here.

Generation limits are a separate model-run policy; the data response-length bound
is **not** a promise that a teacher will finish within that many generated tokens.

## Selection, exclusion and source-only audit

Seed fixed in the builder before curation/generation:

` silt-specialist-v1-pure-sequences-20260908-before-any-generation `

Ranking is ascending SHA256 of `seed + ':' + purpose + ':' + task_id`.
Selection, ordering and calibration have distinct purpose strings. Family
representatives are deterministic; validation/final reservations precede TRAIN.
No ranking uses model success, model outputs, case difficulty or achieved scores.

All **591** pinned source records (164 HumanEval, 427 sanitized MBPP) are processed.
All **32** prior pilot-v2 consumed IDs are excluded: 8 prior development and 24
prior final. The prior `selection-lock.json` is hash-pinned; the builder reads its
IDs and source metadata, **not prior generated model outputs**.

Family graph is computed over all source rows before eligibility filtering:

- Equal normalized prompt, parsed canonical source, alpha-normalized AST, or
  source-record fingerprint.
- Equal normalized function name, or SequenceMatcher >= 0.86 when both normalized
  names have at least six characters.
- Morphologically normalized description-token Jaccard >= 0.60.
- Shared conservative operation tags for obvious variants/inverses: case
  conversion, word splitting, duplicate removal/predicates, whitespace
  replacement, text-content filtering, regex repetition variants, set operations,
  numeric-string predicates, and related operations. Tags are in the builder.
- Connected components touching any consumed ID or the initial six exposed
  families are wholly excluded. Those six are addition of two values, square,
  even predicate, string reversal, larger of two, absolute value. Conservative
  expansions also remove reverse-sequence variants, power-mapping variants,
  scalar parity and two-value extrema/inverse variants.
- At most one remaining eligible task per component, across **all** splits.

These rules are deliberately conservative and auditable, but **not a proof of
perfect semantic decontamination**. Public benchmark pretraining exposure remains
unknown. Broader public relatedness may survive lexical/operation heuristics;
report operational family disjointness, not absolute semantic independence.

The narrow source-only contract rejects advanced mathematical/geometric/recursive
or combinatorial tasks, external I/O/dynamic code, unsupported imports, global
side effects, while loops and exception machinery. It permits one complete
function, at most two loops/comprehensions and 220 AST nodes, with necessary
`typing/re/collections/itertools/functools/string` imports. This is a curation
filter, **not a security sandbox guarantee**; references are inert licensed data.

The unchanged audited parser in `scripts/prepare_coding_pilot.py` decodes only
bounded original direct equality assertions with literal JSON-compatible args
and expectations. No upstream AST is compiled or executed. No loops in upstream
tests are expanded and no expected values are computed from reference code.
Unsupported assertions are recorded; accepted unique assertions remain in source
order, capped at eight. Every selected task has 2–8 cases, at least two distinct
expected values, and at least one nonempty expected value. This is a reduced
original-assertion oracle, not full upstream testing or proof of correctness.

Final exclusion counts are in `manifest.json`. Important totals: 32 consumed IDs,
175 consumed/initial related-family exclusions, 122 tasks with fewer than two
supported literal cases, 74 outside the advanced/algorithmic contract, 18
nonrepresentative family duplicates, 11 complete prompt/response length rejects,
and 7 complete-response length rejects. Categories are exclusive first-reason
counts; the detailed parser audit can contain additional unsupported constructs.

## License, benchmark and fairness limitations

Only existing pinned sources are used:

- OpenAI HumanEval MIT commit
  `6d43fb980f9fee3c892a914eda09951f772ad10d`.
- Google Research sanitized MBPP commit
  `932d4685e23f671b9e8c2abc72dd228ba5ff9252`.
- Google Research publisher dataset card declaring **CC-BY-4.0**, pinned at
  `4bb6404fdc6cacfda99d4ac4205087b89d32030c`.

Do not substitute Google Research's repository software license for MBPP's
explicit publisher dataset license. Preserve all notices, legal text, author
attribution, source links and indication of modifications with redistributed
responses/data. Original prompts containing attribution links remain unchanged.
No new datasets or undocumented third-party content were acquired.

This is a **standalone LOCAL split**, **not official HumanEval/MBPP/EvalPlus
benchmark training/development/final governance or an official score**. Public
examples may appear in original prompts; nothing has been scrubbed to imply
unseen data. Pretraining exposure is unknown. The cohort is source-imbalanced
(primarily MBPP). Controls are simpler scalar tasks and are not difficulty-matched
with targets. Target/control scores can be reported as descriptive domain
support, not causal fairness or matched-hardness effects.

## Freeze, verification and handoff

- Selection digest:
  `5c53a5411bd7031908dfcd4c51e04c3629de527c44fb30b70830b5e982d80671`
- Selection-lock file digest:
  `7ec2082da6eefcb92db204db0355c4c4b1aee64c5ae1772937d6292fd9efaa9f`
- Manifest file digest:
  `86a1cfe3426b176d98135299dc73d2954d94778af11c141201a6f155cc38f0d4`

Offline read-only reproduction, from any working directory:

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /agent/workspace/silt-venv/bin/python \
  /agent/workspace/silt-specialist-core/scripts/prepare_specialist_data.py --check

PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /agent/workspace/silt-venv/bin/python -m pytest -q -p no:cacheprovider \
  /agent/workspace/silt-specialist-core/tests/test_specialist_data.py
```

The focused test file passed **18 tests**. Tests check complete source fidelity,
all split/fingerprint/tag disjointness, every prior ID and representative semantic
regression exclusion, original literal cases, external suite schema, 256-token
reproduction, licenses, exhaustive audits, SHA inventory and byte-identical
rebuild. Source AST parsing emits upstream invalid-escape deprecation warnings;
original licensed code is intentionally not rewritten to silence those warnings.
No test executes any public reference function or runs a model.

Builder `--stats` is read-only and reports aggregate eligibility/environment.
Normal generation refuses to overwrite mismatching frozen files. `--output` must
stay inside owned `data/specialist-v1/**`; no prior-root mutation is supported.
Changes after a model consumes this lock require a separately authorized new data
version, not edits/relaxation of this cohort.

**Parent handoff:** use `train.json` + `calibration.json` and development
`validation.json`/`validation-suite.json`. Keep `final.json`, `final-suite.json`,
`final-provenance.json` and raw source snapshots out of broad training globbing.
Final quarantine is policy, not encryption. Read lock/manifest counts and hashes
without reading final answers. Validate the teacher's target capability and exact
reconstructor token alignment on TRAIN/development before considering final.
