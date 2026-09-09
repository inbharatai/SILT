# Governed public composition evaluation

`function_io_pilot_v1` is a **separately versioned workflow adapter**, not a change to
`data_only_text_v1` / `exact_text`, and not a new quality gate. Old
`ValidationRegistry.register/freeze/begin/finish/history/recover/final_inputs`
APIs and closed schemas remain available. Manual `finish` still records only
`unverified_result_input_not_certificate`. The new wrapper calls the **public**
`python -m asea.compose ... evaluate` command in a subprocess; it never runs its
own model inference or evaluates a reference program in the host interpreter.

## Licensed pilot invocation

Use an existing local, real `hf_text` composition spec with local safetensors
weights and adequate process address-space allowance. No downloads or automatic
activation are added. Example (choose the numeric budgets for the actual model):

```sh
PYTHONPATH=/agent/workspace/silt-pass2/src \
/agent/workspace/silt-venv/bin/python -m asea.validation \
  governed-evaluate \
  --workspace /agent/workspace/silt-pass2-evidence/governed-pilot \
  --compose-workspace /agent/workspace/silt-pass2-evidence/compose-dev-new \
  --spec /ABSOLUTE/PATH/real-candidate-spec.json \
  --suite /agent/workspace/silt-pass2/data/pilot-v2/development.json \
  --dataset-manifest /agent/workspace/silt-pass2/data/pilot-v2/manifest.json \
  --purpose development \
  --resource-profile process_as --memory-mib 8192 \
  --case-timeout 180 --max-seconds 1800
```

The 8192 MiB/180-second values are illustrative, **not measured or recommended
capacity**. `process_as` bounds per-process virtual address space, not RSS,
aggregate descendants, or a cgroup. The public evaluation gate records actual
worker enforcement receipts; requesting a profile is not evidence of enforcement.
The overall public CLI wall budget is finite, defaults to 3600 seconds, and is
bounded to (0, 86400]. Cleanup adds at most ten seconds of wait. Preflight and
postflight inventory hashing are local I/O and outside that execution budget.
The wrapper monitors a combined 8 MiB output budget; logs can overshoot between
polls. The public evaluator additionally applies its own function-oracle limits.

For final, use **the same validation store**, change `--purpose final`, change
`--suite` to `.../final.json`, and provide a **different empty**
`--compose-workspace`. Do not run final until the candidate is fixed. Final data
cannot be reused for another candidate, another suite name, or after an
interrupted/failed attempt. Development can be repeated with new empty compose
roots. All 8 development or all 24 final pilot members are passed to the public
CLI; the wrapper never shortens a failed suite, edits thresholds, or retries.
Backend failure may stop the public evaluator early, which remains blocked with
missing cases reported, rather than being called a completed grade.

`--workspace` means the validation store for this command and is accepted before
or after `governed-evaluate`. It is disjoint from the new compose root. The
optional top-level `--registry` overrides the physical event-store directory.
For backward compatibility the store marker uses a logical sibling identity
`<workspace>.identity` (no model files are placed there). Read its ledger with:

```sh
python -m asea.validation --workspace /ABSOLUTE/PATH/governed-pilot.identity \
  --registry /ABSOLUTE/PATH/governed-pilot history
```

## Manifest mapping and immutable inputs

If `--dataset-manifest` is omitted, a sibling `manifest.json` must exist. No
license/provenance is fabricated when it is absent. This adapter consumes the
pilot's **actual existing fields**:

* `version`, `license_status`, `license_unresolved`, `source_pins`;
* `sources[].file/url/sha256/bytes`, preserving HumanEval MIT license, MBPP
  publisher dataset-card evidence, and CC-BY-4.0 legal terms;
* `artifact_sha256`, `selection_sha256`, `selection_lock_sha256`;
* selection-lock `id/source/source_task_id/source_record_sha256/family/split/group`;
* preserved `ATTRIBUTION.md`, `scope`, and `pretraining_exposure`.

It validates selected-suite bytes, the selection lock and its canonical
selection digest, license evidence bytes, and attribution bytes. It records
source IDs, immutable upstream record hashes, split/family membership hashes,
source pins, per-member content/input hashes, and the relevant rights evidence.
MIT and CC-BY-4.0 are recorded from this adapter's fixed HumanEval/MBPP mapping;
these records are **not** legal authorization, commercial eligibility, or a
privacy/consent review. It does not reinterpret the MBPP repository's software
license as the dataset's license. Raw source solutions are never executed.

The requested suite must be the manifest root's exact `development.json` or
`final.json` matching purpose; arbitrary renamed split files are refused. The
wrapper reads only the selected suite. In development it reads the shared lock
(ID/family metadata) but does **not** open `final.json` or `provenance.json`
(which contains final assertions). It does not inject `function_cases`, expected
answers, or `reference` into model prompts. Existing public compose evaluation
passes only each `case.input` to generation. That input contains the public task
prompt, including upstream public examples. Expected I/O data goes to the
existing separate function oracle after generation, outside the candidate
interpreter. The adapter binds the pinned suite unchanged rather than inventing
new prompts or answers.

A registration ID cannot overwrite its data. Source mapping cannot be rewritten
under another suite ID. Cross-split member/content/family conflicts are refused.
Final reservations check old data-only freezes as well as new adapter freezes;
legacy freeze/register also check governed reservations/splits. The store is
append-only with a lock and hash chain, **not** append-only against its OS owner.

## Lifecycle and failure semantics

1. Validate purpose, provenance, closed public suite/spec schemas, resource
   budgets, and disjoint empty compose root. Inventory actual local weight,
   tokenizer, and config files without importing an ML runtime.
2. Register the suite. Freeze binds the full normalized spec, raw spec hash,
   complete inventories, compose-compatible content graph hash, runtime version,
   installed implementation hashes and dependency/tool versions, governance code
   hashes, manifest/suite hashes, thresholds, and all requested resource limits.
3. Exclusively write normalized spec and the **complete** selected suite to the
   observation directory. Rehash candidate and source dependencies immediately
   before begin. No model runs occur before freeze.
4. `function_io_begin` consumes the reservation **before launching the public
   evaluate CLI**, hence before its first final item. This is deliberately
   conservative: executable-launch failure, backend preflight refusal, timeout,
   cancellation, crash, failed model grade, and missing outputs all consume the
   reservation once begin is recorded.
5. Capture actual exit code, stdout/stderr files and hashes, complete parsed
   public outcome, signed evaluation/candidate/artifact/run envelopes and hashes,
   case counts, gate status, and same-owner signature verification. Append
   `function_io_finish` with `trust: observed_public_cli`. This is not the manual
   unverified-finish path. Copies of signed records are stored without copying
   the compose integrity key.

**Before begin:** invalid inputs/resource requests do not consume. If failure
occurs after freeze but before begin, the reservation stays frozen and its final
families stay reserved; there is intentionally no automatic reset/retry. Hard
process death after begin may leave no finish event; begin alone permanently
records consumption. Preserve these records. The wrapper requests graceful
compose cleanup before forced termination; independent descendant closure is not
claimed. Same-owner records can still be rewritten, and no cryptographic key
protects against its owner.

## Outcome and admission

The response includes the complete `outcome.public_cli_outcome`, raw CLI receipt,
`counts` (`cases`, `measured`, `passed`, `failed`, `missing`), and status:

* `rejected` plus `completion_status: done`: model grading completed and failed;
* `blocked` plus `completion_status: backend_blocked`: infrastructure,
  missing/inconsistent evidence, timeout, or incomplete evaluation;
* `admitted` is reported as quality admission only when the public compose gate
  itself returned admitted, all cases passed, fixture flag is false, signed
  records match the frozen candidate/suite/implementation/resources, and the
  process completed with exit zero and without termination cause.

Exit zero alone, a successful wrapper invocation, or manually supplied output
never grants quality admission. The wrapper issues no certificate, promotes
nothing, and never activates a deployment. It preserves positive nonzero public
CLI exit codes; signal exits remain in the receipt and the wrapper returns 2.
Same-owner signed record verification corroborates observed CLI output; it is
not independent certification or protection against a malicious CLI/OS owner.

Observation files live under `--workspace/observed-<reservation-id>/` (or the
explicit registry root). Do not overwrite source pilot data, prior observations,
validation events, or prior composition roots. Keep the entire validation and
compose stores with pass2 evidence.

## Evidence scope and lightweight tests

The pilot is an adapted subset of public HumanEval/MBPP, not official full
HumanEval/MBPP/EvalPlus/pass@k results. Public training overlap is unknown and
cannot be disproved here. An owner can read final files directly, create a new
registry, modify code or records, or train on public answers. The wrapper guards
a local workflow only, not access control, secrecy, an independent custodian,
universal correctness, or freedom from training contamination.

`tests/test_governed_evaluation.py` uses static pilot JSON, dummy weight inventory
bytes, fake CLI seams, and one tiny sleeping Python subprocess. These are
**unit/transport evidence only**, not real model quality/admission/promotion.
No heavyweight model run is needed:

```sh
PYTHONPATH=src /agent/workspace/silt-venv/bin/python -m pytest \
  tests/test_governed_evaluation.py tests/test_validation_governance.py -q
```
