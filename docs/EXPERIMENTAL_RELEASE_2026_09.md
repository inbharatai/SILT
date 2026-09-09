# SILT experimental reconstruction and recovery — September 2026

**Release evidence summary, 2026-09-09. Construction verified; model quality remains experimental and uncertified.**

This report summarizes locally recorded evidence from the frozen V5 implementation. It does not announce deployment, model-weight availability, a new quality certificate or third-party authentication. It supplements rather than replaces SILT's general skill-interchange architecture: inspectable packets, held-out admission, double gating, removable LoRA, SiltStream, ZeroForge, SiltSpring, audit, rollback and skill-layer unlearning remain separate, existing mechanisms.

## What the experimental path implements

The optional source-weight pipeline performs calibration, coherent Qwen MLP reconstruction or native Switch-to-dense-T5 conversion, real teacher-guided recovery, teacher-independent export, functional validation and a once-only final comparison. It physically constructs smaller models; it is neither prompt-only packet transfer nor a masked full checkpoint. This source-derived path must not be confused with the default L3 packet path, which does not copy teacher weights.

The completed Studio build has five stages: source validation, reconstruction, reconstructed validation, recovery and recovered validation. All five completed successfully in the recorded V5 build. Its status is `BUILT_UNCERTIFIED`, with `engineering_complete=true`, `candidate_frozen=true`, `certificate=false` and `quality_pass=false`. Engine completion does not establish quality, admission or promotion.

See the implementation in [`reconstruction.py`](../src/asea/specialist/reconstruction.py), [`recovery.py`](../src/asea/specialist/recovery.py), [`standalone.py`](../src/asea/specialist/standalone.py) and [`workflow.py`](../src/asea/specialist/workflow.py). The separate public namespace is `python -m asea.specialist`; existing core transfer defaults are unchanged.

## Physical construction and real training

| Recorded quantity | Qwen-derived model |
|---|---:|
| Original pretrained source parameters | 494,032,768 |
| Reconstructed base parameters | 415,586,176 |
| Trained LoRA factor parameters | 1,413,120 |
| Complete deployed base plus factors | **416,999,296** |
| Actual optimizer steps | **64** |
| Training tasks / separate development tasks | 42 / 8 |

The parameter arithmetic is **415,586,176 + 1,413,120 = 416,999,296**, or 15.593% fewer parameters than the 494,032,768-parameter source. The source is Qwen2.5-Coder-0.5B-Instruct. Totals count unique stored base-plus-factor parameters, with tied aliases counted once, including about 1.413M trained LoRA factor parameters. They do not measure active parameters, FLOPs or memory. These are real pretrained teacher weights and actual optimizer steps, not tiny random-model fixture counts or nominal model-size labels.

Recovery used full-vocabulary teacher KL plus supervised response loss. The frozen base remained unchanged while the adapter parameter hash changed. Response-token-weighted development CE fell from **0.91756688 to 0.70981678**, and forward KL from **0.13315734 to 0.13044521**, over 411 development response tokens. These are token-loss measurements, not functional accuracy or a quality certificate.

The complete recorded deployment inventory was **853,192,780 bytes**, versus **999,604,531 bytes** for the source. These are disk-storage quantities, not RAM, VRAM, latency, throughput or energy improvements.

A second real path converted a **619,339,008-parameter Switch source** to a dense T5 base plus trained factors totaling **224,525,568 parameters**, with **32 real recovery steps** and standalone export checks. Its synthetic-span exercise is a construction/recovery probe, **not coding-quality evidence** or proof of general span quality.

## Export and serving contract

The recovered artifact includes the complete reconstructed base, trained factors and tokenizer. It requires the matching **registered SILT factor-bundle loader** and SILT/PyTorch/Transformers/PEFT runtime. It does not require the original teacher, reconstruction workspace, training data or teacher-logit bank at serving time.

It is **not an ordinary Hugging Face root checkpoint or a GGUF artifact**. Loading only its `base/` omits the trained factors and is not the registered recovered model. The trusted loader resolves the bundle's `base/` and `adapter/`, checks strict metadata and runtime identity, and verifies tensor keys, shapes, dtypes and factor values. Provenance paths are not loader inputs.

The earlier BF16 native-merge attempt failed its numerical gate. That failure is retained; factor-preserving export was selected explicitly, without relaxed tolerances or a silent merge fallback. The final native reload had exact forward and two-token greedy-generation parity on the recorded bounded fixture. This is not universal numerical equivalence.

Fresh public CLI probes loaded and generated from both recovered models without a teacher input. These are bounded loader/generation smoke successes, not scored functional passes. A separate recorded Qwen check hid the known teacher/reconstruction paths; it does not prove absence of every alternate path or resistance to a hostile program.

**Prior art and legal scope:** LoRA, low-rank adaptation and teacher-guided training are established techniques, not newly invented here. This report claims no patent novelty for these additions and no coverage of the new reconstruction/recovery mechanisms by the existing provisional. The existing [`PATENT.md`](../PATENT.md), [`LICENSE`](../LICENSE) and [`NOTICE`](../NOTICE) remain unchanged.

## All six final-comparison arms

All six models were frozen before the same 16 new local final tasks were evaluated. Every arm has 16 completed generations and Boolean task grades, with zero missing tasks, zero blocked/ungraded tasks and zero operational failures. A completed generation may still be truncated, invalid or functionally wrong and is counted as a failed task, not removed from the denominator.

| Model | Correct tasks |
|---|---:|
| Original Qwen teacher | **12/16** |
| Activation-reconstructed, unrepaired | **8/16** |
| Activation-reconstructed and recovered | **8/16** |
| Randomized MLP, retained backbone, same 64-step low-rank repair budget | **0/16** |
| Uniform-channel reconstruction, same size, unrepaired | **0/16** |
| Off-the-shelf SmolLM2-360M-Instruct | **8/16** |

Relative to the source, the recovered model has **eight both-pass, four both-fail, four source-only passes and zero recovered-only passes**. Recovery did not improve the aggregate final total over unrepaired activation reconstruction. SmolLM2 matched that total; competitive superiority is not established.

The random control tests initialization dependence with a retained pretrained backbone and a limited LoRA budget. It is not a competitive full-from-scratch distillation baseline. Uniform reconstruction received no recovery training. SmolLM2 differs in architecture and pretraining. The comparisons therefore do not isolate a universal algorithmic advantage.

This is a small local cohort, **not official HumanEval/MBPP accuracy**, a statistical noninferiority result, a broad coding/safety certificate or a percentage of universal skill retention. No 90% retention, world-first or superiority claim is supported. No post-final training or selection occurred.

### What the later read-only case analysis adds

Archived-output analysis corroborated three substantive program-semantics failures and one termination-policy failure among the four source-pass/recovered-fail tasks. The latter retained a plausible correct body but exhausted the output cap without established EOS and was rejected before execution; it remains a failure under the frozen policy.

Two tasks improved relative to unrepaired reconstruction through required function-name/API binding and executability repairs, not demonstrated acquisition of new algorithms. Neither win was a token-cap/EOS rescue. Two other previously passing tasks failed after recovery, leaving the total at 8/16. Successful completion, correct API use and correct algorithms are distinct outcomes.

This analysis read archived material without executing candidates, replaying the oracle, changing scores or modifying source/models. It does not identify a causal neuron, factor, training example or optimization step. No case-hardcoded source fix is proposed or applied. Any future improvement study must use **fresh data and a newly governed evaluation**, not train on, retune against or rescore this consumed final set.

## Frozen-source verification: what was actually recorded

The prepublication verification recorded **2,039 passed, eight expected skips, 75 warnings, zero failures and zero errors** on the verified V5 source snapshot. This is a local result, **not a promise that CI or every machine will report those counts**, and not a test of subsequent documentation edits.

The eight skips cover disabled real pretrained LoRA, NLLB/Qwen/embedding connector, streamed-LoRA and Studio transfer tests, plus pending delegated-cgroup and native Windows runner templates. Warnings included FastAPI lifecycle deprecations, inert-reference escape sequences and PEFT fixture configuration. Offline regression includes tiny random native-model training fixtures; actual pretrained training and inference evidence is reported separately above.

Observed environment: Python 3.9.25, Linux x86_64, CPU; torch 2.6.0+cpu, transformers 4.51.3, peft 0.15.2, safetensors 0.7.0, accelerate 1.10.1, pytest 8.4.2 and pydantic 2.13.5. This is not a complete transitive dependency lock or native Windows/GPU certification.

Earlier repository counts, including approximately 420/421 tests and older limited audits, are historical snapshots with their original dependency and runtime scope. They are preserved as history rather than silently promoted into current results. Consult CI for its own run and environment, and this report for the frozen prepublication evidence.

### Git-independent evidence identity

The local verification labels are **FINAL_RELEASE_SUMMARY** and **FINAL_CORE_VERIFICATION**. They are locally recorded release/verification reports, not external certifying authorities. The latter records **213/213 V5 code files unchanged** and this frozen code-manifest SHA-256:

```text
e0f482b8b302517711dbe1adb89d4fd3a05cb539b6baef359e0f6e541ca03aca
```

This is a **Git-independent manifest digest, not a commit ID**. It is computed from compact, sorted-key UTF-8 JSON of the frozen code manifest's relative-path-to-SHA-256 `files` map. Reproduction requires that exact file membership and byte inventory. It identifies the verified V5 snapshot; it is not asserted to identify the later publication tree after documentation changes.

The verification also recorded a 385-file canonical source inventory, zero added/removed/changed files during verification, all 96 final generation records checked for available evidence consistency, and 800/800 local analysis checks passed. Hash consistency is reproducible against the same inventory; it does **not** authenticate authorship, prove a complete historical access log or provide third-party attestation. This public summary intentionally omits private filesystem paths, artifact IDs, raw traces and logs.

## What remains unproven

The end-to-end construction mechanism can reconstruct, train, export, strictly reload and run smaller models. **Teacher-level quality preservation and competitive advantage remain unproven.** So do frontier/GLM-scale execution, very large teacher deployment, native Windows/GPU validation, universal capability retention and patent novelty. Successful engineering does not upgrade the experimental models to certified quality.
