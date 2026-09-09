# Explicit random-MLP initialization control

## Status and scope

`BASELINE_RANDOM_MLP_UNVALIDATED` is an **initialization ablation**, not skill extraction, certification, a main-model substitute, or a quality-promotion result. The helper never runs training, evaluation, generation, or a model forward. Tiny tests exercise serialization mechanics only.

This proposal is deliberately staged outside the live source tree at `/agent/workspace/silt-specialist-evidence/control-proposal/`. No existing source file, study recipe, model output, or previous result is modified. Parent review/adoption is required before the public module command exists in the core installation.

### Active implementation lock

Inspection of `src/asea/specialist/workflow.py:419-431` found an **explicit 11-file list**, not `rglob` or a directory-content hash. The active `/agent/workspace/silt-specialist-study-075-v2/implementation-lock.json` lists those same files; all 11 hashes matched at inspection. New `controls.py`, its new test, and this new document are outside that list. Adding only those three files would not invalidate this implementation lock. Editing reconstruction, recovery, workflow, `__init__.py`, or `__main__.py` would invalidate it. Recheck the active lock immediately before adoption; do not assume another study uses the same lock policy.

The proposal still remains external as a conservative response to the instruction not to write during the ongoing build. Python bytecode and pytest caches are disabled when testing against core imports.

## API and eventual standalone CLI

```python
from asea.specialist.controls import randomize_mlp_control

manifest = randomize_mlp_control(input_student, output_dir, seed=17)
```

```sh
python -m asea.specialist.controls \
  --input-student /absolute/path/to/inherited-student \
  --output /absolute/path/to/new-random-mlp-student \
  --seed 17
```

The CLI returns zero only for completed control construction, **never quality success**. It returns 2 with a JSON `BLOCKED` error on supported safety/admission failures. There is no integration with existing specialist `__main__.py`, default reconstruction behavior, or the public pipeline.

The input is the actual local student to ablate, never a hard-coded small model. The current implementation supports **native Qwen2ForCausalLM**, including the Qwen2-family checkpoint architecture used by the study. Other families (including Switch/T5, Qwen3, custom/remote-code architectures, adapters, or quantized models) fail closed. The API is model-path-independent, not a claim of universal architecture support.

## Exact initialization intervention

1. Reuse shared safe inventory, bounded JSON/safetensors-header, memory-budget, strict native-meta, tied-alias, and atomic no-replace publication helpers.
2. Construct only the native **meta** architecture. Validate every expected tensor shape and alias; missing MLP tensors fail rather than becoming a random-initialization fallback. No `from_pretrained` or teacher checkpoint is loaded.
3. Require explicit finite positive `config.initializer_range`. There is no guessed/default substitute for a missing distribution parameter.
4. Copy checkpoint shards into an independent staging directory. Preserve original shard headers, names, index, ordering, exact byte sizes, all tensor shapes and storage dtypes. Do not hardlink or symlink.
5. Replace **every** `model.layers.N.mlp.{gate_proj,up_proj,down_proj}.weight` tensor, and only those tensors, with zero-mean Gaussian draws at the configured standard deviation. All embeddings, attention weights and biases, norms, output head, tied aliases, and other inherited tensors remain byte-identical.
6. Generate bounded FP32 chunks on CPU and round once to the original F32 or BF16 storage dtype. The BF16 byte view avoids unsupported BF16 NumPy conversion. This matches the configured Gaussian distribution, not a claim of bitwise parity with Transformers' full-model initializer or direct BF16 sampling kernel.
7. Use a local CPU `torch.Generator` per storage key. Derive its seed from SHA256 of `silt-random-mlp-v1 + NUL + decimal seed + NUL + key`, first eight bytes little-endian, masked to 63 bits. It does not modify the caller's global RNG. The fixed chunk size (262144 elements) and runtime versions are recorded. Reproduction assumes the same algorithm, chunk size, and PyTorch runtime; cross-version RNG identity is not guaranteed. Shard partitioning does not change the draws.
8. Hash all stored tensor payloads before and after. Refuse publication if **any** requested MLP tensor did not change or **any** inherited tensor changed. A changed-tensor hash means the tensor changed, not that every scalar differs after BF16 rounding.
9. Verify exact count/shape/dtype/shard-size equality, byte-identical copied configuration/tokenizer assets, and unchanged source inventory. Atomically publish with Linux `renameat2(RENAME_NOREPLACE)`; any competing output wins without being replaced.

No GPU, teacher, dataset, optimizer, full model parameter allocation, or online acquisition is used by the helper. Native meta validation imports optional Torch/Transformers only when invoked. Memory and disk admission are estimates rather than guarantees; the bounded payload writer is designed to avoid allocating a full checkpoint.

## Artifacts and provenance

The new output contains a native HF checkpoint plus `random_mlp_control_manifest.json`. Original configuration and public tokenizer assets are copied byte-for-byte. Recognized assets include tokenizer JSON/model/vocabulary/merges, special tokens, generation configuration, and chat templates. Other ancillary files are listed as omitted; stale reconstruction/recovery/evaluation manifests are **not** attached to the randomized model. Input inventories still record their hashes as provenance. If a tokenizer configuration references an omitted/unrecognized asset, the output inventory safety check fails rather than publishing a broken external dependency. No private model-config fields are rewritten or hidden.

The manifest contains:

- Explicit negative-control status, input/output inventories, complete source/output configuration, seed, distribution, and implementation runtime.
- Per-stored-tensor owner, shape, dtype, parameter and byte count, source/output file and storage key, before/after SHA256, and changed flag.
- Logical-tensor-to-storage ownership for every native state-dict key, including missing-on-disk tied aliases.
- Equal input/output parameter count, payload bytes and safetensors file bytes; randomized MLP and preserved shared-tensor counts.
- Mechanical verification only; `teacher_loaded`, `checkpoint_load_performed`, `forward_or_quality_run_performed`, `quality_pass`, `certificate`, and promotion flags remain false.

Checkpoint files retain exact sizes. Total directory size is not promised identical: a new control manifest is added and unrelated old ancillary artifacts are omitted. The output inventory deliberately excludes the manifest itself to avoid recursive self-hashing.

Source immutability means this helper never writes source files and verifies content/size hashes before publication. Read access can affect filesystem access times. Published output is exclusive/no-overwrite by API contract, not a read-only filesystem permission guarantee. Concurrent hostile source mutation is not a transactional snapshot guarantee; immutable input and a trusted output parent remain requirements of the shared artifact helpers.

## Fair follow-up experiment (NOT performed here)

Use the inherited student's **same architecture** for both arms:

- Arm A: existing inherited-MLP student.
- Arm B: this student's random-MLP control with the same inherited shared backbone.

Apply the identical existing `recover(..., steps=64, ...)` protocol to both: same teacher, train and validation artifacts, complete tokenization, data order, optimizer/LoRA target modules and rank, learning rate, KD weight, teacher-cache mode, dtype, seed, length, evaluation prompts, generation settings, and resource/time budget. Record both untrained and recovered results separately, without rewriting existing artifacts. This helper neither triggers recovery nor changes its defaults.

When recovery is LoRA-only, this experiment measures whether **64-step low-rank repair under the fixed protocol** can recover randomly initialized FFNs; it is not a comparison against fully training random FFNs, and failure does not itself prove skill extraction. Compare the arms as a initialization ablation, not as evidence to promote quality. Seed 17 is the requested first control; claims about initialization variance need additional prespecified seeds and appropriate uncertainty analysis.

## Isolated test command

From any working directory:

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/agent/workspace/silt-specialist-core/src \
CONTROL_PROPOSAL_MODULE_PATH=/agent/workspace/silt-specialist-evidence/control-proposal/src/asea/specialist/controls.py \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/agent/workspace/silt-venv/bin/python -m pytest \
  -p no:cacheprovider \
  /agent/workspace/silt-specialist-evidence/control-proposal/tests/test_specialist_controls.py -q
```

These tests construct tiny random native Qwen2 fixtures from public config classes. They cover F32/BF16, tied/untied embeddings, single/sharded weights, native offline reload, finite tiny forward, full shared-backbone parity, exact configured random draws, seed and shard reproducibility, source/output safety, no actual loader call inside the helper, invalid configuration/tensor metadata, missing tensors, non-Qwen refusal, failed publication cleanup, output races, and lightweight import. They do not download or run the study model, train, evaluate task quality, or use a teacher.

## Adoption

After parent approval and lock recheck, copy only:

- `control-proposal/src/asea/specialist/controls.py` -> `silt-specialist-core/src/asea/specialist/controls.py`
- `control-proposal/tests/test_specialist_controls.py` -> `silt-specialist-core/tests/test_specialist_controls.py`
- `control-proposal/docs/SPECIALIST_CONTROLS.md` -> `silt-specialist-core/docs/SPECIALIST_CONTROLS.md`

Do not modify existing `__init__.py`, `__main__.py`, reconstruction, recovery, workflow, recipes, or old evidence. This proposal intentionally does not add itself to an ongoing study's implementation lock; a later control study should explicitly hash the control module and configuration in its own provenance.
