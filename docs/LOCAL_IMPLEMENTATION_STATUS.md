# Local implementation status — Flint independent release audit

> HISTORICAL AUDIT: the findings below bind to the source snapshot inspected by Flint, not the final packaged tree. Subsequent repairs and evidence are summarized here; the original audit is retained rather than rewritten.
>
> F1 cancellation/recovery was repaired with crash-released flock locking, journal recovery and terminal cancellation/interruption handling; Anchor added 30 crash tests. F2 now has a separate compiler certify command with source-relative quality/resource checks and uncertainty bounds; the actual Switch candidate remains UNADMITTED. F4 now has bounded safe bundle import with path rebinding, mandatory fresh admission and a real relocated CLI inference run. F5 metadata-backed download checks and exclusive publication were repaired and independently exercised by Seal's 74 tests. These changes do not resolve the broader scientific, resource-enforcement or platform limitations in F3/F6.
>
> Subsequent real evidence includes six diagnostic Qwen coding checks passing through CLI and visible Studio controls, an actual ASR-to-text-to-TTS chain, a matched vision color test, and standalone pruned-model inference. They do not establish broad accuracy or complete all Phase 0–5 exits. Read EXPERIMENTAL_HANDOFF.md and the final evidence report for current scope and test totals.

## Verdict

**REJECT the claim that approved Phases 0–5 are complete or release-accepted.**
**ACCEPT the narrow structural-pruning mechanism evidence and the independently rerun lightweight regression results.**
**BLOCKED / NOT PROVEN:** retained coding quality after pruning, complete real voice/vision composition acceptance, a portable certified deployment, and Windows/device acceptance.

This is a static review plus tiny offline tests of `/agent/workspace/silt-local`, against the approved *SILT Code Review & Non-Breaking Implementation Plan — Compile and Compose* (document `cmtq6ca6t09o507adul1xt9t8`, implementation phases and exit gates) and its clarified Phase 0–5 execution plan (`cmtpop41h03io07adoviy93k7`). It is not a new benchmark run. The concurrent real BF16 evaluation was not disturbed or treated as completed. No heavy models were loaded, no network downloads performed, no existing workspace integrity keys inspected, no original-source edits, and no commits/pushes. Only this document was added to the source tree; reviewer scripts/logs are under `silt-evidence/flint*`.

## Independently observed evidence

- `flint-baseline.log` / `flint-baseline.xml`: **746 passed, 39 skipped**, Python 3.9 baseline venv, **Torch and Transformers absent**, 63.26 seconds. This is lightweight implementation regression evidence, not model-quality or optional-backend proof. Some skips are whole modules; do not treat 39 as necessarily 39 individual untested functions.
- `flint-independent-tests.py` / `.log`: **25 tests in 0.171 seconds**, comprising 22 mocked download-validation tests and three tiny transaction/cancellation probes. Tests named `gap` deliberately assert the observed deficient behavior; their green result is a successfully reproduced gap, NOT a security invariant passing.
- `flint-preservation.json`: all **229 files in the original source archive remain byte-identical**. Current protocol, pipeline, Gate 1, Gate 2, adapter envelope, Spring certifier, legacy CLI, and config are byte-identical to the original. No evidence of lowering Gate 1/2 was found.
- `flint-structural-headers.json`: independent safetensors **header-only** inspection confirms source **619,339,008 parameters / 1,238,805,814 weight bytes / 437 tensors**, candidate **392,809,728 / 785,734,056 / 341**. This is a genuine physical reduction, not merely masking a full checkpoint. It does not rehash the large weight payloads or rerun inference.
- `switch-prune.json:1`: public CLI reports the same counts, 226,529,280 removed parameters, four experts, `source_unchanged=true`, and `admission=UNADMITTED`. Actual implementation replaces expert modules and corresponding router rows, then safely serializes: `src/asea/compiler/core.py:299–319,327–387`.
- `switch-comparison.json:1`: a completed **four-case seq2seq diagnostic**, not coding evaluation. Candidate exact match **0.0**, intact reference exact match **0.0**, generation agreement **0.0**. Equal zero scores do not prove retention. Candidate strings such as `Paris...........` are visibly degraded relative to the intact reference's `Paris.`. `ok=true` means the evaluation command completed; it explicitly remains **UNADMITTED** and `certifies_coding_skills=false`. It cannot satisfy the Phase 4 quality exit gate.

## Phase-by-phase assessment

| Phase | Implemented / demonstrated | Exit assessment |
|---|---|---|
| 0 — Protect existing system | Path-safe storage, atomic packet writes, recoverable rollback, root-shared same-process lock, legacy regression coverage; eight frozen boundaries independently hash-checked | **ACCEPT bounded lightweight Linux repair/regression evidence.** Full Python/Windows/deep-backend release matrix and an independently exercised old-binary-on-new-output downgrade cycle are not established here. |
| 1 — Artifact/evidence contracts | Separate workspace, content inventories, immutable HMAC-checked records, safe local paths, explicit candidate/evaluation/deployment states; lazy optional imports | **PARTIAL.** Practical local foundation exists; not a portable certificate/import contract, and evidence is not fully bound to implementation/environment versions. |
| 2 — Evaluation/activation | Non-vacuous metric floors, target/control requirement, fail-closed coding sandbox interface, policy/evidence revalidation, explicit activation, journaled pointer changes | **PARTIAL / REJECT cancellation exit.** SIGTERM can leave an uncommitted new pointer and a stale lock that blocks normal recovery; CLI KeyboardInterrupt leaves a run recorded as `running`. Final-test governance/paired uncertainty and a robust adversarial correctness oracle are absent. |
| 3 — Real fixed composition | Native text/Whisper/Vits/matched SmolVLM adapters, typed linear pipelines, local loads and before/after artifact checks, opt-in Studio subprocess jobs | **BLOCKED for full acceptance.** Audio-output quality is explicitly unsupported, so the complete ASR→coder→TTS graph cannot be admitted. Required real equivalence/identifier/screenshot/injection/approved-skill-set evidence is not supplied by these unit tests. |
| 4 — One-family compiler | Native Switch structural pruning, calibration telemetry, independent serialized reduction confirmed; public CLI comparison completed | **PARTIAL / quality acceptance BLOCKED.** Standalone inference has integrity preflight, but no compiler retention/resource admission gate, executed-code retention evidence, actual compiler RAM/latency results, or required matched-budget quantization/compact/KD baselines. Repair training is explicitly unsupported, not silently simulated. |
| 5 — Builder/packaging/Studio | Suite-bound selection among supplied admitted evaluations, optional disk constraint, streamed dependency export, plan/CLI/opt-in Studio surfaces | **PARTIAL.** No portable import/rebind/revalidation/clean-machine deployment. Native Windows experimental Studio is explicitly unsupported. No end-to-end downloadable certified bundle can honestly be claimed from the supplied compiler evidence. |

Phase 6 remains deferred; its absence is not a defect in the approved Phase 0–5 scope.

## Prioritized material findings and fixes

### F1 — HIGH: real cancellation does not meet the deployment recovery exit gate

**Locations:** `src/asea/artifacts/__init__.py:157–169`; `src/asea/certification/__init__.py:263–288,291–294,322–326`; `src/asea/compose/__main__.py:92–98`; `src/asea/studio/experimental.py:409–417`; `src/asea/compose/runtime.py:383–401`.

The pointer transaction is real, not missing: it journals the old pointer and catches `BaseException`. The independent KeyboardInterrupt publication probe confirms the old pointer is restored. However, Studio cancellation sends SIGTERM, then SIGKILL, which does not run Python cleanup by default. The writer uses an exclusive-create lock file, not a process-lifetime OS lock. Normal recovery first acquires that same lock.

**Independent reproduction:** a tiny child acquires the actual `Workspace.writer` context without constructing a real workspace or reading any key, invokes the actual pointer publisher, and receives SIGTERM immediately after the new pointer is written. Child exits -15; `active.json` still points to `new`, recovery journal exists, and `.writer.lock` remains. A subsequent normal writer acquisition fails `workspace writer locked`; recovery cannot run through normal CLI `list`/activation/rollback until an operator resolves the lock. This is a reproducible interrupted-publication state, not a claim that every cancellation corrupts data.

Separately, an adapter raising KeyboardInterrupt propagates through `_run`'s `except Exception`, then `finally` persists an immutable run/audit with **status `running`**. Studio can label its outer job cancelled while the composition record is missing or nonterminal.

**Fix first:** use a crash-released single-writer lock or rigorously verified stale-lock recovery; recover journal state before accepting further work/serving active selection; translate cancellation to a terminal run/evaluation outcome. Define the commit/cancellation boundary and prove SIGTERM/SIGKILL at each pointer-transaction step preserves/reconstructs the last committed deployment. Do not simply delete arbitrary live lock files.

### F2 — HIGH: standalone compiler admission is unimplemented; Composer admission is not a substitute

**Locations:** `src/asea/compiler/__main__.py:17–38,46–51`; `src/asea/compiler/core.py:148,368–376,419–455`; `src/asea/certification/__init__.py:127–180`.

The compiler exposes inspect/prune/evaluate/infer, not admit/activate. Its evaluator only does literal text exact match; reference execution is optional; results always say UNADMITTED. There is no predeclared compression retention margin plus measured resource-benefit gate. Composer can evaluate a graph containing the checkpoint, but its target/control reference floor does not demonstrate *retention versus the intact source at comparable budget*. Merely obtaining a Composer admission would not fill this missing compiler gate.

**Fix:** implement distinct compiler admission binding source/candidate/suite/runtime, mandatory retention and control outcomes, predeclared margins and resource benefit, functional coding evaluation when coding is claimed, exact deployable artifact identity, and standalone reload verification. Keep current Switch result UNADMITTED. The required intact quantization, off-the-shelf compact, KD-only and pruned/quantized comparisons have not been implemented/demonstrated by the single four-case reference comparison.

### F3 — HIGH: end-to-end resource containment and comparison are incomplete

**Locations:** `src/asea/compose/runtime.py:325–335,338–341,366–397`; `src/asea/compiler/core.py:167–197,398–408,419–455`; `src/asea/studio/experimental.py:397–417`; `src/asea/certification/__init__.py:347–388`.

Do **not** report that resource gates are wholly missing: Composer checks wall time/RSS before and after nodes and certification enforces recorded limits. Studio also enforces a subprocess wall timeout. Nevertheless, direct CLI model execution has only cooperative checks; a model call may exceed memory or wall limits before control returns. No hard model-worker memory cap is applied by Studio's Popen. Compiler records heuristic preflight estimates, not observed peak RSS/latency. Composer's RSS is the **process lifetime high-water mark**, so repeated candidates in one process do not provide isolated per-candidate measurements. Initial `_prepare` inventory/registration happens before the run timer starts. Selector disk size covers unique model inventories, not temporary runtime/export storage.

**Fix:** isolate each measurement/run in a bounded worker, enforce model memory/time/process limits, measure cold-load and generation separately plus true end-to-end costs, and state exactly which disk/storage budget is enforced. Do not describe the current selection as an equal-deployment-budget optimality result.

### F4 — HIGH for Phase 5 completion: exported dependencies do not make a portable deployment

**Locations:** `src/asea/certification/__init__.py:391–437` (especially 407–411,415–427); `src/asea/artifacts/__init__.py:212–236`; `src/asea/compose/__main__.py:22–53`; `src/asea/studio/experimental.py:245–248`.

Export genuinely streams model dependencies and verifies hashes. Its own manifest correctly states **import/activation in another workspace is not implemented**. Model paths and reference fixture paths stay host-specific; no relocation/import command reconstructs artifact identities and reevaluates trust. Audio/image input fixtures and the executable Python/native dependency environment are not made portable merely by including the model directories. `import_directory` is a Python helper for model copying, not deployment ZIP import. Experimental Studio explicitly refuses non-POSIX platforms.

**Fix:** implement a safe import/relocation flow with complete declared transitive assets, extracted-file validation, runtime/version/license inventory and fresh local validation; prove a clean-machine run with the original model/source directories unavailable. Supply Windows instructions as unverified setup guidance until actual Windows validation passes. Preserve the current honest “integrity bundle, not portable trust credential” wording.

### F5 — MEDIUM: acquisition pins the URL/revision but does not authenticate every asset against official metadata

**Locations:** `src/asea/artifacts/__main__.py:31–55,65–85` (especially 77–80).

Positive independent tests confirm import-safe `__main__`, exact 40-hex revisions, resolved-SHA check, fixed inert filename allowlist, no pickle fallback, byte/disk budget, LFS mismatch rejection, symlink/traversal refusal, truncation/oversize cleanup, and preservation of existing destinations. No network was used.

**Reproduced gaps:** a non-LFS config with an incorrect `blobId` is accepted; weights missing LFS metadata with an incorrect Git blob ID are accepted. `remote_lfs_verified=false` accurately records the missing check, but locally computing SHA-256 is not comparison with the pinned official metadata. A weights-only repository with no config is also accepted as an acquisition; this is not proof of a runnable model. “Official” here is the Hugging Face API origin, not a curated/trusted publisher allowlist; any syntactically valid namespace/model is accepted.

**Fix:** verify Git blob IDs for non-LFS assets, require valid expected LFS hashes for LFS weights (or verify an explicitly supported alternate content hash), refuse malformed/missing expected integrity data where an integrity claim is made, and distinguish downloaded assets from a validated standalone model. Retain side-effect-free import and explicit download-only behavior.

### F6 — MEDIUM/HIGH for trustworthy scientific certification: control labels are not comparative retention or protected final tests

**Locations:** `src/asea/compose/schema.py:133–190`; `src/asea/certification/__init__.py:127–180,191–240`; `src/asea/compiler/core.py:419–455`; `src/asea/certification/sandbox.py:175–192`.

Target and control groups are required, and each case must pass. But these are caller-authored reference cases evaluated on one candidate, not source-versus-candidate regression margins. There is no managed calibration/development/final-test separation, final-test access control during repeated search, or confidence intervals. Composer evidence binds a runtime label, policy dictionary and graph, but not the actual installed dependency environment or a digest/version of every metric implementation. Keeping a runtime/policy label unchanged across implementation changes can leave old evidence apparently current.

The coding sandbox contains execution at the OS boundary but the candidate and trusted tests share one Python interpreter; candidate code can affect Python test machinery. The source explicitly disclaims adversarial correctness proof. That is an important limitation, not a basis for a universal capability certificate or a claim that tests are semantically immutable to hostile candidate code.

**Fix:** reserve final suites, predeclare comparison protocol/margins, bind metric implementation and runtime versions, record seeds/decoding/environment, add paired uncertainty where meaningful, and keep external independent test/result authority when making adversarial-code correctness claims. A tiny predeclared diagnostic is useful but must remain labeled as such.

## Integrity/reload and downgrade answers

**Does inference reload check integrity? Yes, with scope limits.** Compiler infer calls `inspect_model` before loading (`compiler/core.py:398–403`), and inspection compares current candidate file inventory to `compiler_manifest.json` (`:132–145`). Native loading refuses missing/unexpected/mismatched weights (`:200–213`) and uses local-only safetensors. Composer verifies registered artifacts both immediately before and after each adapter (`compose/runtime.py:369–374`) and again for admission/activation. Compiler evaluate also rechecks model files after generating (`compiler/core.py:432–439`). Compiler **infer** lacks that post-run recheck, and its compiler manifest is unsigned: this is lineage consistency, not protection from an owner editing both weights and manifest, nor an admission certificate. A clean-process teacher-independent *quality* acceptance remains unproven by header inspection alone.

**Were gates downgraded?** No change to legacy Gate 1/2 bytes was found. Current Composer exact/code thresholds must equal 1.0, similarity must be at least 0.8, WER at most 0.5, finite values are required, fixtures cannot be admitted, coding claims require functional targets/controls, and activation revalidates policy and measured results (`compose/schema.py:151–156`; `certification/__init__.py:23–33,91–110,183–240`). The earlier Rivet zero-threshold finding is superseded by these checks and passing hardening tests; do not repeat it as a current defect. This does not prove test adequacy or compiler quality.

**Old-binary downgrade proof?** Original preservation and current legacy round-trip tests are useful evidence, but are not the same as an executed old-binary read/operate/export cycle against representative outputs written by the new code. That broader Phase 5 exit remains unproven in this review.

## Reproduction and acceptance boundary

```
/agent/workspace/silt-review-20260907/baseline/venv/bin/python -B /agent/workspace/silt-evidence/flint-run-tests.py
/agent/workspace/silt-review-20260907/baseline/venv/bin/python -B /agent/workspace/silt-evidence/flint-independent-tests.py
/agent/workspace/silt-review-20260907/baseline/venv/bin/python -B /agent/workspace/silt-evidence/flint-preservation.py
```

Audited source SHA-256 values are printed in `flint-baseline.log`. This audit binds to those source versions and the supplied completed Switch evidence. It does not claim a browser demonstration, Windows execution, physical-device test, unseen BF16 result, or full heavy-dependency suite. Repair the concrete cancellation and integrity gaps, then rerun focused negative tests; complete the missing admission/portable-deployment work or explicitly narrow the release scope. **Ship as experimental implemented mechanisms with precise limitations, not “all Phases 0–5 complete.”**
