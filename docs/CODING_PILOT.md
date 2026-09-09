# Adapted public coding pilot v2

## Scope and status

This is a **32-task adapted public benchmark pilot**, not an official/full HumanEval,
MBPP, EvalPlus, SWE-bench, or pass@k score. The builder has generated fixtures only:
**zero model runs, zero reference-solution executions, zero performance results**.
Expected values are copied from supported upstream assertion literals, never supplied
by the curator, inferred by another model, or computed by executing reference code.

Public/pretraining exposure is **unknown**. The local final split is protected from
pilot tuning, not from earlier model pretraining or public dissemination. It cannot
establish contamination-free general coding ability. Literal-case selection favors
small, data-only problems; unsupported tests and dependencies reduce coverage.

The two `EvaluationSuite` files both contain meaningful target and control tasks:

| Fixture | Tasks | Targets | Controls | HumanEval / MBPP | Adapted I/O cases |
|---|---:|---:|---:|---:|---:|
| `data/pilot-v2/development.json` | 8 | 6 | 2 | 4 / 4 | 34 |
| `data/pilot-v2/final.json` | 24 | 18 | 6 | 12 / 12 | 106 |
| Total | 32 | 24 | 8 | 16 / 16 | 140 |

Targets have text and/or collection inputs. Controls use scalar numeric inputs
(e.g. digit algorithms, primality, Bell numbers, month validity). They are genuine
coding tasks with real expected values, **not** deliberately false answers or a
source-ID counter. Each source contributes half of every split/group cell.
Some scalar-input tasks return text/lists; domain classification concerns inputs,
not the type of the result. These controls test another coding input domain, not
non-coding language competence or a matched-difficulty causal treatment.

## Sources, exact pins, and license verification

Only the two requested datasets were used. Full, small source snapshots are archived
as inert data, including original prompts and reference/test text for provenance.
The selected tasks retain their original upstream task IDs, prompts and entrypoints.
Record SHA256 means canonical JSON of the complete original task record; artifact
and source-file SHA256 mean the exact archived bytes. See `manifest.json` for every
source URL, hash, and byte count.

### HumanEval (164 source records)

- Repository: <https://github.com/openai/human-eval>
- Commit: `6d43fb980f9fee3c892a914eda09951f772ad10d`
- Data: <https://raw.githubusercontent.com/openai/human-eval/6d43fb980f9fee3c892a914eda09951f772ad10d/data/HumanEval.jsonl.gz>
- Original MIT license: <https://raw.githubusercontent.com/openai/human-eval/6d43fb980f9fee3c892a914eda09951f772ad10d/LICENSE>
- The original `Copyright (c) OpenAI (https://openai.com)` notice, MIT permission
  notice, and warranty disclaimer are preserved in `sources/HumanEval-LICENSE.txt`.

### Sanitized MBPP (427 source records)

- Repository: <https://github.com/google-research/google-research/tree/master/mbpp>
- Commit: `932d4685e23f671b9e8c2abc72dd228ba5ff9252`
- Data: <https://raw.githubusercontent.com/google-research/google-research/932d4685e23f671b9e8c2abc72dd228ba5ff9252/mbpp/sanitized-mbpp.json>
- Original README: <https://raw.githubusercontent.com/google-research/google-research/932d4685e23f671b9e8c2abc72dd228ba5ff9252/mbpp/README.md>
- Publisher dataset-card pin: `4bb6404fdc6cacfda99d4ac4205087b89d32030c`
- Explicit CC-BY-4.0 declaration: <https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/4bb6404fdc6cacfda99d4ac4205087b89d32030c/README.md>
- Legal terms: <https://creativecommons.org/licenses/by/4.0/legalcode.txt>, preserved
  verbatim as `sources/CC-BY-4.0.txt` and content-hash pinned.

The GitHub MBPP subdirectory README does **not** state a dataset license. The explicit
CC-BY-4.0 declaration is from the Google Research publisher dataset card, not an
inference from the monorepo's Apache software license. License verification is resolved
for this pilot; the manifest has an empty `license_unresolved` list. Reacquisition
must stop on any source/hash/license mismatch instead of substituting another dataset.

Redistribute `data/pilot-v2/ATTRIBUTION.md` and preserved licenses with the data.
It credits OpenAI/Chen et al. and Google Research/Austin et al., links sources and
papers, preserves supplied notices and disclaimers, and explicitly identifies
modifications. CC-BY-4.0 requires retained supplied attribution/notices, license
information and an indication of changes; no additional restrictions or endorsement
are asserted. This documentation does not replace the full legal terms.

## Adaptation grammar and security boundary

The builder uses Python's AST parser as a parser only. There is no `exec`, `eval`,
`compile`, `ast.literal_eval`, subprocess, import-from-upstream, or solution execution.
Parsing solution text for an entrypoint signature does not run its body or imports.

Accepted assertion grammar:

```text
assert entrypoint(literal_arg, ...) == literal_expected
```

For HumanEval only, a direct `candidate(...)` call in the body of the upstream
`check(candidate)` function is renamed to the original `entry_point` in JSON metadata.
For MBPP, the test call must match the single top-level reference function's name.
Parameter order comes from the original AST signature. Original tests remain archived;
accepted assertion source text also accompanies each case in `provenance.json`.

Supported JSON values: `null`, booleans, integers, strings, lists, and dictionaries
with unique string keys. Unary `+`/`-` is accepted only on an integer constant.
Floats (even finite ones), tuples, sets, bytes, complex numbers, names, expressions,
comprehensions, function/method calls inside values, imports, starred arguments,
all keyword arguments, chained/approximate comparisons, assertion messages, and
unknown AST constructs are rejected. There is no tuple-to-list or float-to-int
conversion. This intentionally matches the current outside-oracle value vocabulary.

Bounds: source file/decompressed data <= 2 MiB; parsed source <= 65,536 characters
and <= 4,096 AST nodes; values <= depth 8 and 4,096 visited nodes; individual
lists/dicts <= 128 entries; strings <= 4,096 characters; integer magnitude <=
`2**53 - 1`; positional arguments <= 16; at most 8 retained I/O cases per task.

Only direct assertion statements are adapted. HumanEval loop/conditional/assignment
statements are **not executed, unrolled, or used to derive labels**. They are audited
as ignored statements; unsupported direct assertions are recorded with reasons.
Thus an eligible task can retain only a subset of its original tests. MBPP rows with
test imports, multiple reference functions, or multi-statement test snippets are
excluded. Function bodies are not a formal purity proof; these fixtures describe
bounded deterministic input/return-value contracts. Runtime OS containment and the
outside comparator remain essential for model-generated code.

At least two unique literal cases must survive, with at least two distinct expected
JSON values among the first eight retained cases. Conflicting expected values for
identical arguments cause rejection. The expected-value diversity rule prevents a
partial adaptation from retaining only empty-result tests (for example when all
nonempty results use unsupported tuple values). No expected values are altered.

## Frozen selection, duplicate handling and exclusions

The pre-generation protocol and full ordered selection are in `selection-lock.json`.
The fixed seed is an opaque identifier, **not an acquisition-date claim**:

```text
silt-adapted-coding-pilot-v2-2026-07-13-before-generation
```

1. Consider all 164 pinned HumanEval and 427 sanitized MBPP records. Do not claim
   upstream MBPP train/prompt/validation rows are unseen; the new local split is
   explicit and does not reuse official benchmark score semantics.
2. Apply the grammar, bounds, entrypoint checks, minimum cases and expected diversity.
3. Exclude the six already-exposed diagnostic families from
   `scripts/setup_local_validation.py:37-43`: addition of two values, scalar square,
   integer-even predicate, string reversal, maximum of two values, and absolute value.
   Their IDs/descriptions and the existing script's SHA256 are recorded in the lock.
   Conservative prompt/name rules also exclude related tasks; this is intentional
   over-exclusion, not a claim that all 24 excluded public tasks are identical.
4. Form **operational task-family connected components** using identical normalized
   entrypoint names or >=0.72 Jaccard similarity of normalized description tokens.
   Transitive matches form one family. Choose one representative per detected family
   by ascending SHA256 of `seed + ':selection:' + task_id`. All 15 duplicate exclusions
   identify their retained representative and family in `eligibility-audit.json`.
   No detected family or exact selected task crosses any split/group boundary.
   This heuristic is reproducible, but is **not a proof of complete semantic
   deduplication**: distinct contracts can share algorithmic primitives, and natural
   language similarity can miss or over-merge related tasks.
5. For each source independently, take the first 12 targets and 4 controls by that
   hash rank. No model output, runtime success, or achieved score affects selection.
6. Rank selected tasks within each source/group by SHA256 of
   `seed + ':split:' + task_id`. Reserve the first 3 targets and 1 control per source
   for development (8 total); the remaining 9 targets and 3 controls per source are
   final (24 total). The script reports smaller truthful counts if a quota is short;
   no alternate-source or answer-authoring fallback exists.
7. Freeze exact source, selection, suite and builder hashes **before any model
   generation**. Existing differing fixtures are never overwritten. A selection or
   fixture change needs a new version and new lock, not silent v2 mutation.

Observed counts:

| Stage | Count |
|---|---:|
| Source records | 591 |
| Grammar/coverage/old-family exclusions | 219 |
| Eligible before family deduplication | 372 (96 HumanEval, 276 MBPP) |
| Duplicate-family exclusions | 15 |
| Eligible family representatives | 357 |
| Eligible representatives not selected | 325 |
| Selected | 32 (16 per source) |
| Retained literal I/O cases | 140 |

The 219 exclusions comprise: 166 fewer than two supported literal cases; 24 existing
diagnostic-family exclusions; 3 insufficient expected-value diversity; 12 multiple
reference functions; 13 requiring test imports; 1 unsupported entrypoint name.
The audit records each source task ID and canonical record hash, not just aggregate
counts. These categories are applied in order, so earlier exclusions can hide a
later applicable exclusion reason.

## Files and oracle contract

- `development.json`, `final.json`: schema-version-1 `EvaluationSuite`; <=64 tasks
  per file; `claims: ["coding"]`; `metric: "function_io"`; threshold `1.0`.
- `function_cases` entries contain exactly `id`, `function`, `args`, `kwargs`,
  `expected`. `reference` is descriptive attribution, **not executable reference code**.
- `provenance.json`: original prompt, original source task ID and record SHA256,
  actual entrypoint/parameter names, adapted and available case counts, exact accepted
  upstream assertions, source license, family, group and split.
- `code-function-metadata.json`: per-task Python function name and parameters for
  generated Code metadata. The runtime must use the actual entrypoint, not `solve`.
- `eligibility-audit.json`: per-task eligibility/rejection and duplicate-family log.
- `selection-lock.json`, `manifest.json`: protocol, selection, counts, source URLs,
  pins, license status and hashes.
- `sources/`: verbatim licensed source snapshots and license evidence.
- `ATTRIBUTION.md`: redistributable notices and description of modifications.

Each model input contains code-only response instructions, the required original
function name/parameter order, and the original upstream prompt verbatim (including
original prompt examples). It does not include adapted hidden cases or a reference
solution body. The **installed LLM must generate function source**. The independent
outside oracle compares returned values to the JSON expected values; source-text
comparison or execution of reference tests is not a replacement for this metric.
The preparer never supplies candidate functions.

Both generated suites validated successfully against the newly installed
`EvaluationSuite`/`function_io` schema. Builder tests passed: **49 passed**, with six
warnings from invalid escape sequences in upstream source strings parsed by AST.
No model or upstream reference code was run by those tests.

## Reproduce fixtures and run development first

From the checkout:

```sh
python3 scripts/prepare_coding_pilot.py --check
# A fresh destination can reacquire only the pinned, hash-checked sources:
python3 scripts/prepare_coding_pilot.py --output /path/to/new-pilot-copy --fetch
PYTHONPATH=src /path/to/venv/bin/python -m pytest -q tests/test_coding_pilot_builder.py
```

`--check` is offline/read-only. The builder contains no model-loading or evaluation
command. The coordinator should use SILT's public CLI, not direct model libraries:

```sh
silt-compose --workspace /path/to/pilot-runs evaluate \
  --spec /path/to/real-coder-spec.json --suite data/pilot-v2/development.json
# Freeze any model/configuration decision using development only, then:
silt-compose --workspace /path/to/pilot-runs evaluate \
  --spec /path/to/frozen-real-coder-spec.json --suite data/pilot-v2/final.json
```

The coordinator owns real model paths and budgets. Use equal predeclared prompt,
token and runtime settings for candidate/baseline, and meaningful all-coding
controls. Do not choose cases based on generated answers. Development has 8 model
prompts, final has 24 (not 140 independent generations); the 140 values are oracle
checks on those generated functions. At roughly 100 seconds per generation, one
32-task model pass approaches 53 minutes before overhead; candidate plus baseline
can exceed an hour. This is a planning illustration, not a measured runtime.
Run development first, budget final explicitly, preserve timeouts/failures, and do
not raise output/resource limits merely to hide failures. No final score is
reported until actual CLI execution produces evidence.

## Frozen hashes

```text
selection (canonical selection list):
362610ae759aeae28759600c7f5064f99f264aec1eb25cf3a1e164acf9bad085
selection-lock.json:
3f9a6326dcfc417956c7db0a378f429ef3d097110e03324027d387b0d49d802a
development.json:
2953e25720c4421c266888e61c9fd24a16802dc046379590a9d328d155dbb726
final.json:
01511d4dd5ed55cf51ad71335a7a4daa9f77ca900c183acadc9c8db08b9c002d
builder:
fada630197d618ca7b5939032f45e5112c55e2a393aa3412def513b40dcd03c8
```
