# SILT hardening pass 2 — research-led local checkpoint

This pass keeps the first checkpoint and original SILT unchanged. No commits, pushes, public deployment, paid external compute or Phase 6 work were performed. This is not a claim that every Phase 0–5 acceptance requirement has been satisfied.

## What changed and why

Three independent research tracks examined pruning, isolation/resource controls and evaluation/voice quality. An independent design challenger identified concrete defects, followed by repairs and regression tests. Primary-source reports and their citations are included with evidence.

### 1. Candidate code no longer owns the new oracle's verdict

`function_io` uses a trusted host comparator outside the candidate interpreter and chroot. Expected outputs and authoritative counters stay on the host. The candidate sees only the current bounded JSON function invocation. Setup uses a separate pipe closed before candidate import. Forged `passed` fields, exit zero, malformed/duplicate messages, oversized Unicode and attempted access to hidden references are tested.

This is stronger than the retained `functional_code` unittest mode. It establishes observable outputs on the exercised data, not algorithm identity, purity, training non-contamination or universal correctness. Existing legacy APIs are not silently rebranded as external-oracle evidence.

### 2. Resource semantics and lifecycle are explicit

Fresh trusted model workers establish kernel `RLIMIT_AS` and CPU limits before imports, with supervised deadlines, bounded I/O and Linux parent-death handling. AS means virtual address space per process, not RSS or aggregate host memory. Measured worker RSS comes from post-bootstrap VmHWM rather than a misleading inherited pre-exec watermark.

The current container does not delegate writable memory/cpu/pids controls. A requested cgroup profile fails closed before inference; no system cgroup settings were modified. Aggregate-memory and Windows execution remain unavailable rather than silently falling back. Job Objects alone would not supply hostile-code filesystem/network isolation.

A development run exposed an outer-timeout orphan. Managed SIGTERM/SIGINT handling and process-group cleanup were repaired, and the interrupted evidence was retained. Linux direct workers now establish PDEATHSIG before model imports. This does not claim kill-proof cleanup of arbitrary escaped descendant trees.

### 3. More informative, correctly scoped compiler experiments

New `diagnose` and `roundtrip` commands preserve raw token IDs, sentinel-delimited spans, EOS/repetition and intermediate numeric observations. Native versus SILT controls run in fresh workers; a full-expert identity export tests serialization without pretending to be pruning.

BF16, original FP32-router preservation and dispatched-token REAP are explicitly research configurations. They cannot silently enter v1 certification. Unknown expert coverage blocks REAP selection; an unobserved expert is not assigned zero importance.

In the completed four-debug-case controls:
- Native and wrapper generation IDs and short logits matched exactly.
- Full-retention identity/save/reload matched exactly.
- FP16 baseline showed non-finite intermediate expert outputs even when final logits were finite; BF16-plus-router-preservation avoided those observed nonfinites. These are two changed factors, so do not claim dtype alone caused the difference without a full factorial comparison.
- Keeping seven experts preserved four generated outputs relative to that BF16 source; keeping six changed outputs. Neither establishes quality retention. Source and candidates were only 2/4 on expected spans, and remain AUDIT_ONLY/UNADMITTED.
- REAP with four minimum observations refused to prune because one expert lacked coverage.

The old 0/4 literal results and references remain unchanged. Span parsing diagnoses their mismatch; it does not retrospectively replace them. Switch remains a seq2seq mechanics vehicle, not proof of an extracted coding specialist.

### 4. Governed, broader evaluation rather than six toy functions

A pinned, licensed adapted public coding pilot contains eight development and 24 final tasks with bounded literal I/O cases. HumanEval MIT and sanitized MBPP CC-BY-4.0 notices and provenance are preserved. Expected values were copied from allowed upstream assertion literals using AST parsing, never executed reference solutions or model-generated answers.

Important limitation: all 427 sanitized MBPP rows were eligible, not only the official test split. This is a disclosed local split, not an official HumanEval/MBPP/EvalPlus score or contamination-free benchmark. Selection favors data-only functions; public training exposure is unknown.

The governance wrapper freezes candidate/configuration/data/policy, consumes final reservation before generation, drives public SILT CLI, and records observed outputs and failures. It cannot protect owner-readable files from the same OS owner or impersonate an independent custodian. Failed finals cannot be renamed and reused as untouched data.

Development scored 3/8 under both tested prompt configurations. The separately frozen final pilot scored **9/24 (37.5%)**, with all 24 measured and no missing cases. Gate outcome: rejected, no certificate. Conditional Wilson 95% interval: approximately 21.2%–57.3%, subject to its stated sampling assumptions. This is more honest evidence, not an improvement claim for the underlying coding model.

### 5. Real speech data and honest listening workflow

The ASR pilot uses 60 distinct sentence IDs from the official FLEURS en_us test data, pinned to revision 4683b04af03d2d9549064c7d72060a9a94bb6046, under CC-BY-4.0. Original float WAVs were preserved and converted to PCM16 for SILT. The archive digest and per-file hashes were verified; inputs were selected before inference.

All 60 were measured: 408 word edits over 1,432 reference words, **28.49% raw corpus WER**. This metric is case/punctuation-sensitive. Fifty-seven cases met the predeclared per-case 0.5 threshold, three did not, so the whole evaluation was rejected. Do not present 57/60 as speech accuracy. Read speech in en_us is not Indian-English, conversation, accent robustness, or independent-speaker population evidence.

`voice-check` now produces measured waveform facts and an explicit pending-human state. `export-listen-batch` prepares a coordinator packet with blank real-human fields. It does not fabricate MOS, pronunciation approval or proof someone listened. Full perceptual voice admission still requires actual authenticated human evidence and a reviewed approval process. The existing MMS English model remains noncommercial-only; no weights/audio are represented as commercially authorized.

### 6. Actual UI screenshots, not only a colored rectangle

An exploratory suite captures twelve real rendered SILT UI states across two viewports. References are predeclared selected/entered form values. All images belong to one correlated app family, not twelve independent products, and are not ScreenQA/Rico data. Exact results are in the final evidence report; no score is assumed in advance here.

## Frozen implementation

The independent pre-model freeze tested **1,439 passes, nine skips and zero failures**. Skips include six real-weight opt-ins, a seed-dependent Spring case and two pending cgroup/Windows templates. Source CODE fingerprint:

`29c5a143258c776486649e867ac71cd518e9445434867bbc147a54fe0a8d3352`

Final model pilots run after that freeze. Source hashes are checked afterward. Documentation may be completed later; implementation changes require a new freeze and cannot silently inherit old final evidence.

## Usage

Use the Linux-tested environment from EXPERIMENTAL_HANDOFF.md. The existing Windows SILT core is unchanged; native Windows experimental execution remains blocked. No new heavy model downloads are required for the included pilots beyond the first checkpoint's explicitly acquired models.

```sh
python -m pip install -e '.[dev,studio]'
silt-execution probe
silt-compose --workspace /your/new/run-root evaluate \
  --spec /your/spec.json --suite /your/data-only-suite.json \
  --resource-profile process_as --memory-mib 8192 --case-timeout 180
```

The 8192 example is an address-space budget used in these experiments, NOT 8 GiB of available physical RAM or a recommended universal setting. A separate cooperative RSS budget remains in the graph. See RESOURCE_CONTROLS.md.

```sh
silt-validate governed-evaluate \
  --workspace /your/validation-registry \
  --compose-workspace /your/empty/candidate-run-root \
  --spec /your/spec.json \
  --suite data/pilot-v2/development.json \
  --dataset-manifest data/pilot-v2/manifest.json \
  --purpose development --resource-profile process_as \
  --memory-mib 8192 --case-timeout 180 --max-seconds 1800
```

Use development before freezing your own final configuration. The packaged final set has already been used and its results disclosed; it is not untouched evidence for further tuning in this development process. Never edit thresholds or references to manufacture admission.

Real speech acquisition/reproduction is documented in SPEECH_PILOT.md; audio lives outside the source ZIP and can be obtained with the bounded, pinned script. Adapt source paths to your machine without silently changing input content or licensing metadata.

Studio remains opt-in:

```sh
SILT_ENABLE_EXPERIMENTAL=1 SILT_EXPERIMENTAL_ROOT=/your/experiments \
  python -m uvicorn asea.studio.server:app --host 127.0.0.1 --port 8477
```

Open `/experimental`. New visible controls expose per-case resource profiles, compiler diagnostics/roundtrip/REAP options, resource probes and pending-human voice preparation. A completed job is not admission. Do not expose this local operator interface publicly.

## What is still not fully proven

No broad-quality coding specialist from pruning; no adequate matched-budget KD/quantization comparisons or repair training; no comprehensive multilingual/real-speaker/screenshot evaluation; no authenticated human listening approval; no delegated cgroup or native Windows/GPU/physical-device validation. Narrow hardening is implemented and tested, but these empirical and platform requirements remain open. Phase 6 remains deferred.
