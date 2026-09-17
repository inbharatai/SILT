# Historical generation-policy correction — 2026-09-12

**Generation-policy correction included in this source snapshot.** Public deployment is verified separately; publication adds no model weights or new quality result. This date identifies the correction, not a new benchmark. A fresh, governed, matched-policy protocol is required before making a quality-retention or competitive-comparison claim; the consumed final set is not a retry or tuning set.

[README](../README.md) · [Historical release](EXPERIMENTAL_RELEASE_2026_09.md) · [Integrated snapshot](INTEGRATED_RELEASE_2026_09_12.md) · [Capability catalog](CAPABILITIES.md)

## What the archived evidence establishes

The historical V5 specialist evaluator requested greedy decoding but did not reliably enforce it. Archived stderr records sampling-default overrides for **four derived arms**: activation-reconstructed, activation-recovered, random-MLP repair and unrepaired uniform-channel reconstruction. Reports nevertheless retained `greedy: true` and `do_sample: false`: these fields described the **requested configuration, not the effective native generation policy**.

The exact recorded override was:

> `generation_config` default values have been modified to match model-specific defaults: {'do_sample': True, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8, 'repetition_penalty': 1.05}. If this is not desired, please set these values explicitly.

The original source-Qwen and SmolLM2 logs do **not** record that override. This is not an “all runs sampled” finding. Nor does an absent warning retroactively certify those arms as greedy. The warning is emitted once per process; it is not a per-case or per-token measurement. Historical V5 native-loop modes and multinomial calls were not directly instrumented. Sampling in the affected arms is supported by the archived override, saved configuration and verified caller/resolver mechanism—not fabricated historical live-mode observations.

| Frozen final arm | Original correct / total | Archived policy evidence |
|---|---:|---|
| Original Qwen source | **12/16** | No override warning; historical live mode uninstrumented |
| Activation-reconstructed, unrepaired | **8/16** | Sampling-default override recorded |
| Activation-reconstructed and recovered | **8/16** | Sampling-default override recorded |
| Random MLP, retained backbone, same 64-step low-rank repair budget | **0/16** | Sampling-default override recorded |
| Uniform-channel reconstruction, unrepaired | **0/16** | Sampling-default override recorded |
| Off-the-shelf SmolLM2-360M-Instruct | **8/16** | Empty stderr; historical live mode uninstrumented |

All original counts, Boolean grades, denominators, outputs and raw receipts remain unchanged history. All six arms completed; the recovered aggregate equals unrepaired reconstruction and the compact baseline. **This mismatched-policy comparison does not establish matched-greedy retention or a controlled competitive comparison.** Retention is not established; the four source-only passes cannot be attributed solely to compression or recovery training. No corrected score can be inferred, and no final rerun or rescore is reported.

The controls retain their original limits: random MLP keeps a pretrained backbone and limited LoRA budget, not full-from-scratch distillation; uniform reconstruction had no recovery; SmolLM2 differs in architecture, tokenizer and pretraining. The small local suite is not official HumanEval/MBPP, a noninferiority study or a broad coding/safety certificate.

## Why the source and derived policies differed

The recorded runtime was Transformers **4.51.3**. This is distinct from the version metadata stored in each checkpoint's `generation_config.json`:

| Configuration class | Saved Transformers version | Saved sampling fields | Generation-config SHA-256 |
|---|---|---|---|
| Original Qwen | **4.43.1** | `do_sample=True`, temperature 0.7, top-k 20, top-p 0.8, repetition penalty 1.05 | `fdaccbcb02f3e1e7914ccb0f69ebe899071ffd27cf825166d823163e156870f2` |
| Derived Qwen students and controls | **4.51.3** | Same five sampling values | `80b8ec6e7991b64763e4ff29b59a9cd2968b390d0736e0dc586585da04bad013` |
| SmolLM2-360M | **4.42.3** | No saved sampling-policy fields; library `do_sample` default is False | `87b916edaaab66b3899b9d0dd0752727dff6666686da0504d89ae0a6e055a013` |

The old caller used `model.generate(**inputs, generation_config=cfg)` without `use_model_defaults=False` or a direct `do_sample=False` keyword. The installed resolver deep-copied that requested object. With `use_model_defaults` omitted, saved generation-config versions at least **4.50.0** enabled fallback from global-default-valued requested fields to checkpoint defaults. Direct kwargs take precedence over that merge. Derived checkpoints therefore triggered the override while the older source metadata did not. The old receipt serialized the unchanged requested object and hard-coded the greedy label.

The same warnings were retained in reconstructed/recovered DEV arms of both the first-factor and V5 studies and in random/uniform DEV controls: **six of nine audited DEV arms**, versus **four of six FINAL arms**. First-factor recovered DEV was **3/8**; the separate V5 recovered DEV was **4/8**. Source DEV was 4/8 and reconstructed DEV 2/8 in each study. These remain distinct observations, not controlled greedy improvements or proof of a causal recovery benefit. No claim is made about uninspected earlier studies or unrelated Switch runs.

## What remains real, and what is not a causal conclusion

- The archived read-only case analysis still identifies **three substantive program-semantics defects and one termination-policy failure** among the four source-pass/recovered-fail outputs. Those defects in the saved programs are real. They do not establish that compression or training caused the failures under matched greedy decoding. The termination case remains a failure under the frozen output contract.
- The two recovered-versus-unrepaired wins still concern function-name/API binding and executability, not demonstrated acquisition of new algorithms; neither was an EOS/token-cap rescue. Two other tasks changed to failures, leaving the original aggregate unchanged. These are static output comparisons, not isolated treatment effects.
- Real source weights, physical reduction, actual optimizer updates and teacher-independent factor-bundle loading remain separate engineering evidence. The Qwen counts remain **494,032,768 source**, **415,586,176 reconstructed base**, plus **1,413,120 factors**, totaling **416,999,296 stored parameters**. No format change or new weight distribution follows from this correction. Serving still requires the complete base, factors and tokenizer plus the registered SILT loader—not a plain HF-root checkpoint or GGUF artifact.
- **Teacher-forced CE/KL training and loss evaluation are unaffected by this generation-default defect:** they call forward/loss computations, not this generation resolver. They still do not certify functional quality.
- **Strict recovery export/reload forward and short generation parity are unaffected by this specific False-to-True override:** those generation probes supplied direct, higher-precedence `do_sample=False` kwargs. This does not certify every other checkpoint knob or universal numerical equivalence. The historical BF16 native-merge failure remains recorded.

## Local fix and its bounded verification

The readiness `NativeGenerator` now disables checkpoint-policy default merging with explicit `use_model_defaults=False`, validates requested versus resolved policy, and derives the greedy flag from the verified resolver result. Supported compatibility is deliberately strict: **Transformers 4.51.3**, and **non-prompt PEFT 0.15.2** for the factor-wrapper path. Other versions/APIs fail closed pending compatibility validation; this is not a promise for all versions permitted by older setup guidance. See [current evaluation source](../src/asea/specialist/evaluation.py) and [targeted regression definitions](../tests/test_specialist_generation_policy.py).

Production receipts now distinguish requested and resolved settings and identify their stage before native input-length bookkeeping and special-token device-tensor preparation. They explicitly say `live_generation_mode_observed=False`; production does not pretend a diagnostic hook ran.

Separate recorded verification supplies actual-mode instrumentation:

- Tiny native/PEFT Qwen and T5 regressions: **25 passed**. The old invocation reproduced SAMPLE with eight multinomial calls despite requested False; fixed checks observed GREEDY_SEARCH and zero multinomial calls.
- Retained recovered Qwen, a **nonfinal, unscored eight-token CPU prompt**, seeds **0 and 1** in independent processes: both usable receipts observed GREEDY_SEARCH, zero multinomial calls, unchanged RNG and loaded state, cache use, and identical tokens/per-step logit hashes. This is a mechanics/repeatability check, not a new quality evaluation or an observation of the historical final processes.
- Preserve diagnostic incidents: three retained-model generation invocations occurred, including an initial seed-0 serialization failure. The usable seed-0 receipt retains raw `pass:false` from a hook that missed a positional resolver argument; the independent review adjudicated that recorder defect using native source and the other runtime observations. Seed 1 used signature binding and passed all checks. Do not describe every raw run as successful.

This does not supply replacement benchmark scores, quality retention, EOS quality, 3B execution or CUDA certification. Existing CPU scope and **GPU_UNVERIFIED** status remain unchanged, including the six GPU-required skips in the historical hardware-branch snapshot. Historical full-suite counts refer to their own source/runtime snapshots, not this edited source. Subsequent local readiness verification rebuilt the wheel and sdist and confirmed payload equality with the previously tested installed wheel. This is packaging integrity evidence, not new model-quality or CUDA evidence; rebuild and reverify packaged resources after further source/document edits.

## Evidence identity and preservation

This correction summarizes the retained `HISTORICAL_GENERATION_POLICY_NOTICE.md` / `.json` and `PROPOSED_PUBLIC_CORRECTION_TEXT.md`, together with the separately recorded generation-policy independent review. The historical audit inspected existing log/config/report metadata and source text, not a new benchmark. Its before/after manifests record **115/115 original files unchanged**, with manifest SHA-256 `0b58d78b59381d36dd1b3c072900f625d259aa4ae7ddd12f5b0b689da06c151a`. This is integrity evidence, not independent custody or third-party authentication.

The following names identify retained frozen artifacts; they are not invented public download links. V5 names refer to the original frozen V5 study; control names refer to the original six-arm comparison. Raw generated answers, logs and private filesystem locations are not redistributed here.

| Original frozen report | SHA-256 |
|---|---|
| V5 `source-final.json` | `a617859c73b4a9d55bbd9f0282276c2ceebc9dab782cc7ab66234ec1d27cba45` |
| V5 `reconstructed-final.json` | `c658e0c3398846a11a1cc1f651a6a9bbf467d71caa82d6b395368c7fd866059e` |
| V5 `recovered-final.json` | `0ab9b6eb5501eddd89ddbbc3b09aedcd0c27ae56056639c979aa93b0cc85c3a2` |
| `random_mlp_same_backbone_same_64_step_repair.json` | `891179b409efa6cea1a4d88fcab6dc2219dcb9879cf9afa8ce859d24bf422247` |
| `uniform_channels_unrepaired_same_size.json` | `e7a284e6aaa3d4251b8d95230fe9ba8a093b767181e7c3f96ec298735aca2129` |
| `off_the_shelf_SmolLM2_360M_Instruct.json` | `e641db5eb216a23d5a63d4bc632cc5a279455397527999d4f156e27e0ce07b52` |

Frozen `specialist/evaluation.py` SHA-256: `472f57d74a93f86925e24b968f004d71ca0f3c14679fa148d02eabc5ed09505b`; frozen `specialist/recovery.py`: `88ed6a093479cba0da5c56ad8a38b3c284e6fcdefb9cb1168a8018d090c35e92`. Both match the original V5 implementation lock. These hashes identify historical source, not the readiness fix. The [original release verification digest and counts](EXPERIMENTAL_RELEASE_2026_09.md#git-independent-evidence-identity) remain preserved and separately scoped.

Future quality evidence requires fresh TRAIN/DEV/FINAL governance, a frozen runtime and effective generation policy recorded consistently for source, students and controls, reviewed data/selection bindings, and a separately authorized final evaluation. Do not tune against or relabel the consumed historical final set. No speculative corrected score is offered.
