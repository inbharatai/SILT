# Standalone weight-derived specialist reconstruction

This module produces **physically smaller, standalone native Hugging Face weights**. It does not install a mask, monkey-patch inference, create an adapter, or require the original teacher at serving time. It is a reconstruction/initialization step, **not evidence of retained coding skill, quality parity, or promotion**.

## API

```python
from asea.specialist import reconstruct, ReconstructionBlocked

manifest = reconstruct(
    source_dir, output_dir, calibration_path,
    family="auto",          # auto | qwen2 | switch_transformers
    method="activation",    # activation | magnitude | uniform
    retention=0.75,
    dtype="bfloat16",        # bfloat16 | float32; CPU only
    max_length=256,
    max_samples=32,
    seed=17,
    memory_budget_bytes=None,  # automatic observed CPU headroom; optional lower ceiling
)
```

Exact signature:

```python
reconstruct(source_dir, output_dir, calibration_path, *, family='auto',
            method='activation', retention=0.75, dtype='bfloat16',
            max_length=256, max_samples=32, seed=17, memory_budget_bytes=None) -> dict
```

All input paths are local. `output_dir` must not exist, must have an existing parent, and must be disjoint from the source directory. A sibling staging directory is published with Linux `renameat2(RENAME_NOREPLACE)` after successful validation. Existing outputs are never replaced, including if one appears during reconstruction. Failed staging is removed. No commits, network downloads, promotion, or source writes occur.

The module imports without PyTorch/Transformers. Optional ML dependencies are imported only by `reconstruct`. Mechanical tests ran with PyTorch 2.6.0+cpu and Transformers 4.51.3. CPU threads are not changed globally by the API; callers on a two-CPU host should set their process thread limits appropriately.

### Calibration schema

```json
{"samples": [
  {"prompt": "Write a Python function that checks balanced parentheses.",
   "response": "def balanced(text): ..."},
  {"prompt": "Explain the return value of this function: ..."}
]}
```

Use permitted TRAIN calibration examples, never final evaluation data. Every row
(up to 10000 in the bounded 16 MiB file) is fully encoded and checked before source
weights load, including rows beyond the first `max_samples` used for scoring.
No padding, truncation, cropping, or response rewriting is performed. The default
complete-length bound is **256**, not a generation budget. An overlength row names
its ID and rejects the request even for magnitude/uniform baselines.

Causal calibration reuses recovery's `_encode_details` at runtime (no import-time
ML dependency): the **same native user chat plus assistant response**, jointly
encoded, EOS and prefix-stability policy as recovery. Unlabeled rows use exactly
the native user-chat generation prefix. Flat tokenizers use the explicit recovery
newline fallback, not a fabricated chat template. Switch uses independent complete
native prompt and response (prompt fallback only when unlabeled), with native EOS
and shifted decoder inputs. Per-row complete lengths, encoding/template hashes,
and actual maximum length are recorded in `calibration.encoding_audit`. The entire
calibration file is hashed; no final data is read. Baselines still require valid
calibration for the reconstructed forward check.

## A. Native Qwen2 structured MLP pruning

Supported source: native `Qwen2ForCausalLM`, including Qwen2.5-Coder-0.5B with hidden size 896, intermediate size 4864, 24 layers, 14 query heads, 2 KV heads, and vocabulary 151936. Only the native bias-free SiLU SwiGLU MLP is supported. Attention geometry, attention weights/biases, embeddings, norms, output head, and tied-weight relationships are retained.

The single global intermediate width is:

```text
kept = floor(source_intermediate_size * retention / 64) * 64
```

It must be at least 64 and strictly smaller than the source width. Every layer gets the same width, but its own selected channel indices. At retention 0.75, the named 4864-wide source becomes 3648-wide. This is **75% MLP-channel retention, not 75% total-model retention**; the full-model reduction is recorded from actual parameters.

For each layer, the exact same sorted indices slice:

- `gate_proj.weight`: rows (axis 0),
- `up_proj.weight`: rows (axis 0),
- `down_proj.weight`: columns (axis 1).

The `nn.Linear` weight tensors and dimension attributes are physically replaced in place, and native config `intermediate_size` is updated. There is no deepcopy, masked channel storage, or change to native forward code.

### Selection methods

- **Activation (default):** a read-only forward pre-hook on the native `down_proj` observes its actual input, which is `silu(gate_proj(x)) * up_proj(x)`. Importance is `mean(abs(native_down_proj_input)) * L2(down_proj.weight[:, channel])`. Accumulation uses float32, token weighting across the selected sequences, and no gradient. All layer scores are gathered on the unpruned source before any layer is changed.
- **Magnitude baseline:** `L2(gate_row) * L2(up_row) * L2(down_column)`; no activation-based claim.
- **Uniform baseline:** channel `floor(i * original_width / kept_width)` for `i = 0..kept-1`; evenly spaced, not learned.

Activation instrumentation runs the first sample without hooks and then with hooks and requires **full native-vocabulary bitwise logit parity**: shape, dtype and SHA-256 of every logit byte, with a streamed finite-value check. This uses the SHA-256 collision assumption, not approximate numeric tolerance. The first logit tensor is released **before** the hooked forward; only its small fingerprint survives. No two full `input_length * vocabulary` logit tensors or two models are held for parity. All subsequent samples stream one forward at a time. Both the explicit `use_cache=False` argument and temporarily disabled native `config.use_cache` prevent past-key/value cache allocation; config is restored even on failure. Hooks return `None` and are always removed. This validates instrumentation only; pruning is not expected to preserve logits. Stable sorting breaks score ties by original channel index. Qwen selection is deterministic and does not consume `seed`.

## B. Genuine Switch-to-dense-T5 top-1 conversion

Supported source: native ReLU `SwitchTransformersForConditionalGeneration` with at least two experts and at least one sparse layer. Output: a real `T5ForConditionalGeneration`, not a Switch wrapper. This path is fixed **one expert per sparse layer**; `retention` is recorded but does not control expert count. The manifest explicitly reports this and the actual `1 / num_experts` expert retention.

- Default activation selection observes the native router's returned capacity-filtered dispatch mask and probabilities. It selects the expert with the greatest **mean accepted router probability mass** per layer, separately for encoder and decoder.
- Magnitude chooses the greatest `L2(expert.wi) * L2(expert.wo)`.
- Uniform chooses `(seed + sparse_layer_index) % num_experts`.

Sparse FFN keys such as:

```text
encoder.block.1.layer.1.mlp.experts.expert_2.wi.weight
```

map to:

```text
encoder.block.1.layer.1.DenseReluDense.wi.weight
```

The same mapping applies to `wo` and decoder layer-2 FFNs. Nonsparse `.mlp.wi/wo` weights are renamed to `.DenseReluDense.wi/wo` without numerical changes. Shared embedding, LM head, attention projection, relative-position bias, layer-norm, and final-norm weights are retained. The compatible T5 configuration explicitly preserves vocabulary, model/head/key dimensions, encoder/decoder counts, relative attention buckets and distance, dropout, epsilon, token IDs, embedding ties, and cache configuration.

**Global FFN-width coupling:** T5 has one `d_ff`. Top-1 selection keeps every sparse and nonsparse FFN at the source `d_ff` (3072 for the named family). No expert concatenation, scaling, zero expansion, synthesized dense weights, or new random parameters are introduced. Supporting K>1 would also require a deliberate policy for every nonsparse FFN; that path is not implemented.

The T5 target is first constructed on the **meta device** (no target weight allocation). Every target key/shape must match an explicitly mapped source tensor; `load_state_dict(strict=True, assign=True)` reuses the retained source storage and native ties are restored. The source is released; this is one resident set of model weights, not two copied models. Unexpected/missing source or target keys fail. Every intentionally omitted router tensor or unselected expert tensor is listed separately.

### Important non-parity caveat

Native Switch routing uses its configured router dtype (normally float32), top-1 probabilities, token capacity masks, and an overflow path. The source dtype plan preserves header-proven original F32 router classifier tensors as F32; **before any calibration forward**, those original values are re-verified/restored directly from safetensors without cloning a second router copy. Merely upcasting rounded BF16 values is not restoration. The bounded helper reads at most 16 MiB per router and 64 MiB total, records exact tensor SHA-256/key/source-file/dtype provenance in `router_precision`, and identifies originally non-F32 routers separately. The acquired Switch checkpoint has six original F32 encoder router matrices (147456 bytes total); originally BF16 decoder routers remain identified as BF16-source values. No full teacher copy is created. Router instrumentation then preserves this loaded source computation; changing the router to a bf16 custom approximation would not be acceptable. The **output dense T5 removes routing and the per-token router-probability scale**. Selecting an expert is not mathematically equivalent to Switch forwarding. The result is labeled `structural_initialized_requires_repair_and_evaluation`; no logit parity or capability parity is claimed. Separate repair/training and fresh evaluation are required before any promotion. T5's native loader may keep `wo` in float32 for stability under some loading options; the serialized output dtype and byte counts in this manifest describe the stored reconstructed tensors, not all possible serving loader casts.

## Safety and memory admission

The implementation reuses `asea.artifacts.safe_path`, `safe_file`, `model_inventory`, `file_hash`, and `atomic_json` rather than replacing existing safety policy. Symlinks, parent traversal, unsafe/executable/pickle files, PEFT/base indirection, escaping shard/tokenizer references, model/tokenizer `auto_map` (even empty mappings), quantization/compression, unsupported architectures, malformed/duplicate JSON keys, and future or incompatible-major Transformers config versions are refused before Auto loaders. Tokenizer and generation-config loading are local only; tokenizer uses `trust_remote_code=False`. Source model construction uses the explicitly selected native class under `accelerate.init_empty_weights(include_buffers=False)` (meta parameters, small native CPU buffers), then re-establishes native ties. An explicit CPU `safe_open` loader walks sorted shards and tensors, applies exactly one per-tensor dtype conversion, and assigns each parameter and its genuine aliases to the same source-derived storage. It does not call `from_pretrained` for source weights, synthesize missing parameters, or build a full checkpoint dictionary. There is no untrusted pickle loader, random-weight repair, network model load, or second source model.

Before model construction, bounded safetensors headers supply shapes, dtypes, offsets, and element counts. Only floating source storage formats are supported. The index must account for all tensor keys and all safetensors files. Offsets, payload sizes, duplicates across shards, missing keys, extra keys, and config/shape disagreements fail. Only omitted genuine tied aliases are allowed; redundant stored copies of a tied parameter are refused as ambiguous provenance. Native source structure is inspected with meta parameters. All native keys must resolve to header-proven unique storage; missing/unexpected/mismatched keys, unexpected persistent buffers, and unmaterialized meta tensors fail. Shapes/dtypes are rechecked as each actual tensor is opened. Unique materialized source parameter and byte counts must exactly match the preflight plan before calibration. There is no second coarse logical-model estimate that discards header dtype and alias evidence.

CPU resource admission is source-aware and bounded by **observed available RAM**:

```text
safety_headroom = max(256 MiB, 10% of observed available RAM)
effective_limit = observed_available - safety_headroom
if memory_budget_bytes is not None:
    effective_limit = min(memory_budget_bytes, effective_limit)

runtime_remaining = max(0, 512 MiB - measured_post_import_current_RSS)
load_transient = sum(source bytes requiring a dtype cast)  # explicit loader only
future_workspace = all_Qwen_replacement_MLP_bytes
                 + max(native_forward_workspace, layer_cast_workspace, serialization_workspace)
                 + possible_native_router_upcast_growth
estimated_incremental_peak = unique_loaded_native_bytes + load_transient
                           + future_workspace + runtime_remaining

# After load, available memory ALREADY accounts for loaded resident weights:
post_load_required = future_workspace + runtime_remaining
# After transform, replacement weights are also already resident:
post_transform_required = post_load_required - all_Qwen_replacement_MLP_bytes
```

`memory_budget_bytes` must be a positive integer or `None`. `None` automatically
uses observed host/cgroup headroom, **not a hardcoded 4 GiB architecture ceiling**.
A larger operator number never overrides observed headroom. A sandbox operator
can explicitly pass `4 * 1024**3` (or a smaller ceiling) without modifying any host
restriction. Host `MemAvailable`, cgroup v1/v2 current usage/limits and visible
process-cgroup ancestors are considered. If a bounded available value cannot be
established, admission fails. No resource limits are raised or bypassed. Full integrity hashing is followed by
best-effort clean read-file-cache eviction advice on safetensors; unsupported
advice leaves the conservative observed-headroom guard in effect.

### Accounting evidence and conservative boundaries

- **Same-dtype mmap is one storage, not two copies.** `safe_open(..., device='cpu')` plus same-dtype CPU `.to` and `nn.Parameter` assignment retain the original tensor data pointer. Genuine tied keys share the same parameter object. The source payload and loaded model do not independently consume a full physical copy. Tests verify actual pointer identity, native ties and CPU buffers using tiny constant weights, and verify that a dtype conversion does allocate separate target storage.
- **Cast transients are explicit, not wished away.** The plan records largest source shard and largest cast-source tensor. Conversion executes one tensor at a time, but shared mmap lifetime and file-cache reclamation are not guaranteed at each shard boundary. Therefore the current conservative fallback reserves **all cast-source bytes**, not merely the largest tensor/shard. An unknown loader or missing plan gets the old full-copy-style fallback plus a conservative all-F32 target bound; it receives no mmap discount.
- **Native dtype policy is class-specific.** The meta model's actual `_keep_in_fp32_modules` applies to aliases as well as stored keys. In Transformers 4.51.3 Qwen2 and Switch declare none; native T5 declares `wo`. Thus a Switch source `.wo` does not receive a fictional blanket F32 surcharge just because T5 has a loader exception. All encoder/decoder router classifiers (including biases) load at supported native `router_dtype` before any freeze hash or forward. Original F32 bits are preserved; BF16-source values are exactly upcast when configured F32. Targets and cast-source bytes are accounted during loading, not deferred to the first forward; unexpected dtypes or lossy F32→BF16 router configurations reject. Switch-to-T5 assigns retained expert tensors without an automatic T5 serving-loader cast. Provenance reports each output tensor's **actual** dtype.
- **Pruning can keep mmap pages alive.** Even though only one layer is transformed at a time, untouched embeddings/attention can keep an entire original shard mapped. Admission reserves **all** newly materialized retained Qwen MLPs alongside the full source, not just one replacement layer. This is deliberately more conservative than assuming immediate reclamation of each removed MLP's pages. Switch's meta T5 target aliases chosen source experts; no second full dense target is charged.
- **Workspace is explicit.** For complete length `L`, vocabulary `V`, FFN width `W`, hidden size `H` and attention heads `A`, forward allowance is `L*(8*V + 32*W + 32*H) + 16*A*L²`. This includes full native logits (potentially F32), finite checking, eager attention/softmax and layer activations, without cache or gradients. Layer-cast allowance is `12*H*W`. Serialization reserves the larger of one bounded 256-MiB shard (or total loaded bytes if smaller) and the largest target tensor; a single embedding can exceed the shard setting. Allocator fragmentation, dirty output pages and concurrency are not guaranteed by these estimates.
- **Runtime is already allocated.** Current `/proc/self/statm` RSS is measured after ML imports, before source weights. Only the unfilled part of a 512-MiB runtime floor is future allocation. Unavailable RSS grants no credit. Already resident bytes are never added back to `MemAvailable`/cgroup headroom; the operator/host minimum and original headroom reserve remain unchanged. The 10%/256-MiB host reserve is still independent safety margin, not another claimed runtime copy.
- **Live stage checks are incremental.** Reports contain current RSS at pre-load, post-load/pre-observation, post-transform/pre-forward, and post-save. Post-load admission charges only future workspace, not loaded source bytes again; post-transform admission also removes already allocated replacement bytes. Materialized source counts must exactly match the header/native alias/dtype prediction. Stage RSS is **not a sampled peak**.

These remain conservative estimates, not a universal RSS bound, a 4-GiB guarantee,
or a concurrency scheduler. Use fresh supervised CPU workers with no competing
model jobs. Large models and small operator limits still fail. This does not claim
GPU support or compile-on-any-hardware capability.

### Qwen header/tokenizer-only check (no real model run)

A local read-only header/meta/tokenizer check against
`Qwen2.5-Coder-0.5B-Instruct` and the unchanged specialist-v1 calibration found:
**494,032,768 unique parameters, all BF16, 988,065,536 source bytes in one shard,
true embedding/head ties, no native keep-F32 modules, and maximum complete
calibration length 220**. No prompt was shortened. At 0.75 MLP retention the plan
reserves 470,679,552 replacement MLP bytes. Measured post-import RSS (including
the native model-class imports used by reconstruction) in the final fresh check
was **340,815,872 bytes**; after meta/tokenizer preparation it was **408,588,288
bytes**. These are live current runtime RSS, **not loaded-model RSS**. The
remaining runtime reserve was 196,055,040 bytes.

| Variant | Incremental estimate | Result at observed effective limit 2,847,010,407 bytes |
|---|---:|---|
| Explicit same-BF16 mmap/assign | 1,973,599,488 bytes | Admitted by estimate |
| Explicit BF16 source to F32 target | 4,646,149,376 bytes | Refused |
| Unproven loader, conservative fallback (no runtime credit) | 5,504,351,232 bytes | Refused |

At the previously reported effective limit **2,801,756,160 bytes**, the same-BF16
estimate also fits; the regression test covers that exact limit with an even more
conservative 232,808,448-byte runtime credit (2,081,606,912-byte estimate). Available RAM
and measured runtime vary per worker; the parent must perform the actual retry.
The known prior ~1.7-GB model RSS was not used as an admission override. No actual
Qwen weights were loaded, no generation/quality run was performed, and original
study diagnostics, model data and quality thresholds were not changed by this
check.

Only small score vectors and per-layer casts/index selections are created during Qwen reconstruction. Switch uses a weightless meta target and assigned source storage. No deepcopy or full original model copy is retained. Full source hashes and source tensor mappings are evidence; original teacher tensors are not bundled for serving.

Output uses safetensors and tokenizer/config files, with HF `max_shard_size='256MB'`. **HF cannot split one tensor across shards:** a single larger tensor (e.g. a large vocabulary embedding) may exceed 256 MB. Final output keys/shapes are checked against the native model without reloading a second full model. All original source and calibration hashes must remain unchanged before atomic publication. A native reconstructed forward must produce finite logits.

## Output and exact manifest shape

The returned dictionary equals `output_dir/reconstruction_manifest.json`. Top-level keys:

```text
schema_version: 1
artifact_kind: "standalone_native_hf_specialist"
status: "RECONSTRUCTED_UNVALIDATED"
source:
  family, architecture, files, config
output:
  architecture, model_type, config, files, standalone,
  teacher_required_at_serve, safe_serialization, max_shard_size
calibration:
  file, sha256, size, samples_available_used, max_samples, max_length, text_policy,
  encoding_audit, actual_max_length
method:
  name, requested_retention, dtype, seed,
  source_intermediate_size, output_intermediate_size, score,
  initialization, teacher_output_parity_claimed,
  [Qwen] actual_channel_retention
  [Switch] experts_per_sparse_layer, source_experts_per_sparse_layer,
           actual_expert_retention, retention_argument, router_caveat
selected_indices: {native_source_mlp_path: [original_indices...]}
weights_provenance: {output_state_key: {
  source_key, source_storage_key, source_file, operation,
  source_shape, output_shape, output_dtype,
  [index_select only] axis, selected_indices_ref
}}
removed_source_tensors: [{source_key, reason}...]
counts:
  source: {parameters, tensor_bytes, safetensors_bytes}
  output: {parameters, tensor_bytes, safetensors_bytes}
reduction:
  parameters_removed, parameter_fraction,
  tensor_bytes_removed, safetensors_bytes_removed
router_precision:
  policy, restored, source_non_f32_routers
memory_admission:
  requested_memory_budget_bytes, available_host_bytes, safety_headroom_bytes,
  limit_bytes, estimated_peak_bytes, header_numel, header_loaded_bytes, source_payload_bytes, formula,
  load_strategy, native_keep_in_fp32_modules, target_dtype_counts, dtype_exceptions, cast_tensor_plan,
  forward_workspace_bytes, layer_cast_workspace_bytes, serialization_workspace_bytes, router_upcast_growth_bytes,
  largest_shard_bytes, largest_cast_tensor_bytes, load_cast_transient_bytes, cast_policy,
  replacement_mlp_bytes, future_workspace_bytes, post_import_rss_bytes, runtime_remaining_bytes,
  materialized_source_count, stage_measurements
verification:
  native_forward_unchanged, hook_parity, samples_used,
  [activation only] observed_tokens_by_layer,
  reconstructed_forward_finite, serialized_state_shapes_and_keys_strict,
  source_files_unchanged, source_loading_strict,
  capability_evaluated: false, promotion_performed: false
runtime:
  torch, transformers, device: "cpu"
```

`files` maps each relative file name to `{sha256, size}`. Output `files` covers model/config/tokenizer payloads and excludes the manifest itself, avoiding recursive self-hashing. Config snapshots are complete native JSON configurations. `source_storage_key` resolves aliases to the actual key in a safetensors shard. Qwen selection uses original channel indices for all three coupled projections; Switch selection uses original expert indices. Qwen omitted channel slices are described by `index_select`, so `removed_source_tensors` is empty; Switch whole-tensor removal reasons are `router_omitted` or `unselected_expert`. Parameter/byte counts deduplicate native tied parameters; safetensors byte counts measure serialized weight files, not tokenizer or manifest sizes. Float-source dtype conversion is explicit in `method.dtype` and per-tensor provenance.

The manifest deliberately has no `PROMOTED` status and no quality score. Native final forward finiteness and mechanical weight mapping do not establish usefulness.

## Serving and tests

After reconstruction, move or make the source inaccessible and load the output directly:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(output_dir, local_files_only=True,
                                          trust_remote_code=False)
# Qwen output:
model = AutoModelForCausalLM.from_pretrained(output_dir, local_files_only=True,
                                            trust_remote_code=False)
# Dense T5 output instead:
# model = AutoModelForSeq2SeqLM.from_pretrained(output_dir,
#     local_files_only=True, trust_remote_code=False)
```

`tests/test_specialist_reconstruction.py` creates tiny random Qwen2/Switch native models and a tiny locally constructed tokenizer. It downloads nothing and makes **no real-weight quality claim**. Tests check float32 and bf16 reconstruction for all three methods, exact retained/sliced source values, paired channel indices, global widths, native source hook parity, source hash immutability, strict missing/unexpected/shape rejection, no random target weights, actual smaller serialized models, teacher-independent native reload/generation, memory admission, optional imports, atomic no-replace publication, and failed-stage cleanup.

```bash
/agent/workspace/silt-venv/bin/python -m pytest -q \
  /agent/workspace/silt-specialist-core/tests/test_specialist_reconstruction.py
```

Only mechanical tiny-model tests were run by the reconstruction engineer. Real local-teacher reconstruction and post-reconstruction capability evaluation belong to the integration/orchestration run.
