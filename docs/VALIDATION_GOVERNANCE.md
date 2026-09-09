# Validation governance and pending voice evidence (v1)

This additive subsystem implements a **local workflow ledger**, not a protected-test
service, independent custodian, evaluator, human-authentication system or certificate
issuer. It follows the separation of evidence dimensions proposed in
`research-evaluation-pass2.md`. It does not change existing composition admission,
activate anything, relax any threshold, or claim audio-output quality admission.

## Boundary and limitations

* `--workspace W` identifies the candidate workspace without opening its artifact
  store, model inventory, signing key, or secrets. Registry defaults to the sibling
  `W.validation-registry`; `--registry R` may select another disjoint directory.
  Neither directory may contain the other. Registry data must not be included in
  candidate exports or handed to candidate execution.
* Every normal registry result identifies the boundary as
  `local_owner_workflow_only; not independent custodian authorization`.
  Files contain complete inputs/references. Hashes do **not** hide answers.
  The same OS owner can read/edit/delete/rechain the entire store, or initialize a
  different store. No resistance to that owner is claimed. Direct filesystem reads,
  denied calls and copied registries are not globally monitored. Separate service/OS
  credentials and an authenticated independent custodian remain future work.
* JSON schemas are closed (`extra=forbid`, strict, versioned). Newly registered suites
  accept only `data_only_text_v1`: inert text inputs and expected text. No arbitrary
  Python trusted tests, plugins, pickle, `eval` or `exec` hooks are accepted. Text that
  happens to contain code remains data. Functional-code oracle contracts require an
  explicitly reviewed new schema, not a hidden callback here.
* `finish` accepts only an **unverified result input** with output digest/reason. A
  user-supplied `passed`, `approved`, certificate, or arbitrary result fields is
  rejected. Even `status=completed` never grants admission. This module does not run
  a model or infer observed correctness from self-reported results.
* No model downloads, ASR, heavy inference, network calls, signing credentials,
  real listener recruitment, or human-approval creation are implemented or needed.

## Public CLI

All commands work with `python -m asea.validation` in an installed environment, or
with `PYTHONPATH=src` from the repository. Registry commands require global options
**before** the subcommand:

```sh
python -m asea.validation --workspace W register --suite suite.json
python -m asea.validation --workspace W freeze --suite local-v1 \
  --candidate-hash SHA256 --policy policy.json
python -m asea.validation --workspace W begin --reservation RESERVATION_ID \
  --candidate-hash SHA256 --policy policy.json
python -m asea.validation --workspace W finish --run RUN_ID --result result.json
python -m asea.validation --workspace W history
python -m asea.validation --workspace W recover

python -m asea.validation voice-check --audio output.wav --text 'Expected speech' \
  --language en --model-manifest voice-model.json --output voice-evidence.json
python -m asea.validation export-listen-batch --manifest batch.json --output packet.json
python -m asea.validation check-review --review review.json --packet packet.json
python -m asea.validation proportion --successes 48 --cases 64
python -m asea.validation schema Suite
```

`SHA256`, `RESERVATION_ID`, `RUN_ID` are placeholders, not accepted literal values.
The candidate hash must be lowercase 64-character SHA-256, supplied by the caller's
actual candidate fingerprinting path. The registry binds the hash; it does not
recompute a model/graph fingerprint. `begin` deliberately rechecks both the exact
candidate hash and the frozen canonical policy. Policies are normalized against the
closed schema, then canonically hashed; whitespace/key order in a JSON file is not
part of semantic policy identity. `--output` is optional for `voice-check`, required
for batch export, and always refuses replacement. Outputs otherwise go to stdout.
Blocked operations exit 2 with JSON on stderr. A `pending_human` evidence packet
exits 0 because preparation succeeded, **not** because quality passed.

Machine-readable schemas are exported with `schema NAME` for `Suite`, `Policy`,
`ResultInput`, `VoiceModelManifest`, `ListeningBatch`, `ListeningReview`. Authoritative
Python definitions live in `src/asea/validation/schema.py`.

## Suite / license / policy contract

`Suite` fields:

| Field | Contract |
|---|---|
| `schema_version` | 1 (default); strict version |
| `suite_id`, `version`, `parent_suite_id` | Immutable ID/version; optional parent must already exist; changes need a new ID |
| `owner` | Declared provenance string, **not authenticated custodian approval** |
| `format` | `data_only_text_v1` only |
| `exposure` | `public_training_overlap_unknown` or `local_training_overlap_unknown`; no training-clean claim |
| `selection_method`, `selection_seed`, `sampling_frame`, `exclusions` | Predeclared subset provenance; not proof of sampling quality |
| `datasets` | Nonempty mapping from dataset ID to full `LicenseCard` |
| `members` | 1–256 closed `Member` records; unique member IDs |

Each member has `member_id`, `dataset_id`, globally meaningful `family_id`,
`split` (`development`/`final`), `role` (`target`/`control`), `language`, `input`,
`expected`, and `content_sha256 = digest({"input": input, "expected": expected})`.
`digest` is SHA-256 of sorted, compact, finite JSON as defined in `asea.artifacts`.
Registry stores exact member content, canonical suite hash and separate split
hashes over ordered member lists. Ordering is preserved in the frozen hashes.

The same family, member identity, exact input or content must not cross development
and final splits, including previously registered suites. Family IDs must span
translations/related screenshots/sentence variants rather than reset at dataset
boundaries. Declared families and exact checks cannot detect every near duplicate or
unknown model-training overlap. Control roles are explicit; an eventual evaluator
must report controls separately rather than quietly include them in accuracy.

A full license card contains:

```json
{
  "spdx": "CC-BY-NC-4.0",
  "source": "https://publisher.example/model-or-data",
  "revision": "pinned-release-or-commit",
  "license_text_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "attribution": "Actual attribution and notice obligations",
  "usage_restrictions": ["Noncommercial experimental use only"],
  "consent_privacy": "reviewed",
  "redistribution": "prohibited"
}
```

This example is illustrative, **not a verified rights grant**. Record actual pinned
source/license bytes and hashes; the ledger records supplied hashes but does not
fetch or authenticate license text. `consent_privacy` may instead be
`not_applicable`/`unknown`; `redistribution` may be `allowed`/`unknown`. Unknown license
terms or unresolved consent block final freeze. SPDX strings are preserved verbatim,
including **CC-BY-NC-4.0**. The conservative classifier recognizes MIT, Apache-2.0,
BSD-2-Clause, BSD-3-Clause, CC0-1.0, CC-BY-4.0 and CC-BY-NC-4.0; it is not a general
SPDX expression/legal parser. Unknown expressions fail closed, rather than being
relabelled permissive. NC/unknown/unreviewed restrictions block commercial use.
A recognized permissive license without other restrictions only reports
`requires_rights_review`; it never grants legal authorization. Redistribution is
recorded, not automatically exercised. The listening batch exports references and
measured metadata, not model/audio payloads, and makes no upload.

`Policy` example (also illustrative, **not a scientifically validated threshold**):

```json
{
  "schema_version": 1,
  "purpose": "noncommercial_experimental",
  "evaluator_revision": "pinned-exact-text-oracle-revision",
  "metric": "exact_text",
  "threshold": 1.0,
  "max_cases": 64,
  "max_seconds": 120,
  "stopping_rule": "one_attempt_all_cases",
  "missing_output": "failure"
}
```

No threshold is chosen or changed by statistics or after failures. The ledger records
budgets; actual timeout/resource enforcement belongs to the future evaluator, not to
this metadata store. Final must contain target and control and fit `max_cases`.

`ResultInput` example:

```json
{
  "schema_version": 1,
  "status": "infrastructure_failure",
  "evaluator_revision": "pinned-exact-text-oracle-revision",
  "output_sha256": null,
  "reason": "Worker did not complete; no result admission"
}
```

Other statuses are `completed` and `candidate_failure`. Completion is not passing.
The evaluator revision must match policy; output hash is an unverified reference.

## Lifecycle, access and crash behavior

1. `register`: append immutable suite, suite/split hashes; state `development`.
2. `freeze`: one reservation for the suite, binding exact candidate and policy;
   state `frozen_final`. Existing reserved final families/inputs/member identities
   cannot be laundered through a renamed suite. New final data are required.
3. `begin`: one atomic durable event both consumes the token and establishes the run;
   state `consumed`, even if the worker later crashes, times out, fails or never starts.
   A mismatched candidate/policy is rejected before consumption. A valid begin can
   never be silently retried. If its response is lost, recover the run ID via history.
4. `finish`: one immutable result-input event; consumed remains consumed. Invalid
   result JSON does not reset the reservation. No hidden retry, score improvement,
   certificate or inferred success follows an infrastructure failure.

Store uses a POSIX nonblocking `flock` single-writer lock; concurrent writers fail
with an explicit retry message. Non-POSIX locking is blocked rather than silently
weakened. Lock file has no secret. Kernel releases the lock on process exit.
Each event is written to a bounded temporary file, fsynced, and exclusively published
via hard link without replacing an existing filename; directory is then fsynced.
Events have sequence, prior hash and content hash. Missing/corrupt committed events
block replay; the system never repairs history by inventing success. Owner removal
of a terminal suffix cannot be detected without an external checkpoint: no such
independent checkpoint is claimed here.

`recover` validates committed events, removes uncommitted `.pending-*` files and
records recovery. It never resets a consumed token or deletes committed failure
history. Interrupted initial marker publication may discard only uncommitted marker
temporaries. Parent traversal, symlink components and nonregular files are blocked;
outputs are exclusive writes. These are cooperative filesystem safety checks, not
race-proof protection against a malicious same-owner filesystem adversary.

## Integration hooks for the parent evaluator/oracle

```python
from asea.validation import ValidationRegistry

registry = ValidationRegistry(candidate_workspace, registry_directory)
reservation = registry.freeze(suite_id, actual_candidate_hash, policy_dict)
run = registry.begin(reservation["reservation_id"], actual_candidate_hash, policy_dict)
# This read appends an access event. It returns final member records including
# expected answers: only a separate evaluator may hold these, never the candidate.
final_members = registry.final_inputs(run["run_id"])
# Parent wires actual evaluator-observed outputs, containment and decision logic.
# registry.finish(run["run_id"], result_input_dict) records a reference, not a cert.
```

`history()` logs API history access. `final_inputs()` requires a consumed active run
and logs each API read; it blocks after finish. It does not itself authorize an
independent evaluator. Observed results must be produced by the actual trusted
oracle outside candidate control; **never** use the `finish` CLI's arbitrary input as
admission evidence. This subsystem intentionally leaves that separate oracle wiring
to the parent integration, rather than faking it with a callback or pass flag.

## Scoped voice preparation: useful evidence while humans are unavailable

`prepare_voice_evidence(audio, text, language, model_manifest)` (alias not needed;
exported from `asea.validation`) reads existing local PCM WAV bytes without modifying
or normalizing them. Supports 8/16/24/32-bit integer PCM, 1–8 channels, up to 120
seconds and 32 MiB. Reports original-file SHA-256/byte count, language, reference text
hash, model-manifest hash and measurement-code hash, plus:

* channels, sample rate, frames and duration;
* normalized peak, RMS and DC offset; finite integer samples;
* clipping count/fraction, defined as samples at either integer endpoint;
* near-silence fraction at absolute normalized sample amplitude <=0.001.

The silence measure is a scalar-sample fraction, not VAD or a windowed loudness
estimate. These facts do not prove speech is present or good: noise can be valid PCM.
Unsupported/truncated/empty input blocks instead of manufacturing facts. Signal
status is `structurally_valid_pcm_not_quality`; intelligibility, naturalness and
pronunciation are independently `pending_human`. There is no automatic MOS, no ASR
run, no inference that ASR roundtrip equals quality, and no audio-output admission.
This explicit pending flow is usable now for evidence organization; it is not an
unimplemented quality promise or a workaround around existing admission blocks.

Model manifest may be the new closed `VoiceModelManifest`:

```json
{
  "schema_version": 1,
  "model_id": "publisher/model@exact-revision",
  "model_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "licenses": [{
    "spdx": "CC-BY-NC-4.0",
    "source": "https://publisher.example/model-or-data",
    "revision": "pinned-release-or-commit",
    "license_text_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "attribution": "Illustrative only; replace with verified attribution",
    "usage_restrictions": ["Noncommercial experimental use only"],
    "consent_privacy": "not_applicable",
    "redistribution": "prohibited"
  }]
}
```

These identities and hashes are illustrative placeholders, not verified evidence.
Alternatively an existing closed composition `ComponentManifest` is accepted for
preparation only. Its `model_path` is never opened. Its SPDX provenance strings are
preserved; missing complete license evidence blocks commercial eligibility. The
fixed MMS English checkpoint's CC-BY-NC-4.0 remains noncommercial experimental only,
regardless of waveform facts or future human scores.

## Listening batch and pure review checker

`ListeningBatch` local manifest:

```json
{
  "schema_version": 1,
  "protocol_id": "predeclared-english-pilot-v1",
  "clips": [{
    "clip_id": "clip-001",
    "audio": "output.wav",
    "text": "Expected speech",
    "language": "en",
    "model_manifest": "voice-model.json"
  }]
}
```

Relative paths resolve against the batch manifest's directory. Up to 64 unique clip
IDs, no traversal/symlinks. Absolute explicit local paths are allowed. Export produces
an exclusive-write **proposed coordinator review packet** with immutable audio/hash
references, measured facts, named protocol, instructions and a blank review template.
No audio/model bytes are copied; nothing is uploaded. The coordinator packet contains
scripts, so must **not** be shown to listeners before blind transcription. A future
real collection UI must separate/blind/randomize assignments and anchors.

`ListeningReview` binds `packet_sha256`, `protocol_id` and trials. Each trial binds
clip/audio hash and has initially null real-human fields: `listener_id`,
`language_proficiency`, `playback_setup`, `blind_transcript`,
`script_hidden_during_transcription`, `naturalness_rating` (1–5),
`pronunciation_notes`, `listened_at`. Empty blind transcription is a legitimate
listening response and is not silently treated as missing. The packet digest covers
all packet fields except its own digest and blank template. Real authenticated
collection, per-listener assignments/exclusions, qualified pronunciation adjudication
and personally listened approval remain unavailable.

`validate_listening_review(review, packet)` is a pure schema/completeness/hash/coverage
checker. It flags missing fields and duplicate listener/clip trials. Even populated
fictional names/ratings can only be structurally complete: it always reports
`authenticated_human_evidence=false`, `quality_admission=false`, `status=pending_human`.
The CLI cannot store/sign/create human approvals and has no approval command. It
never computes MOS from dummy data or transforms strings into proof of listening.
No full quality admission is available without a separately implemented authenticated
human process, actual evidence and a reviewed decision policy.

## Statistics and verification

`proportion_summary(successes, cases)` / `proportion` returns numerator, denominator,
proportion and two-sided 95% Wilson interval, conditional on supplied cases. Interval
assumes independent Bernoulli cases; repeated listeners/families/screenshots require
cluster-aware analysis outside this utility. Sampling uncertainty does not establish
training non-contamination or deployment representativeness. The function never
modifies thresholds or emits an admission. Example: 48/64 gives approximately
0.632–0.840; six hand-authored successes are still only six fixtures.

`tests/test_validation_governance.py` covers lifecycle/binding, one-use failures,
rename reuse, group/split integrity, closed test/results schema, NC/unknown licensing,
recovery, symlinks/traversal, exclusive writes, concurrent lock refusal, hash tampering,
all supported PCM widths, pending voice/review states, Wilson values and public CLI.
All data and ratings used in tests are **unit fixtures**, never real quality evidence.
No models or large dependencies are loaded.
