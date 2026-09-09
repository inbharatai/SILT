# Portable Compose import: always UNADMITTED

`asea.artifacts.bundle.import_bundle(archive, output, spec_output,
max_bytes=4 * 1024**3) -> dict` imports the ZIP emitted by
`asea.certification.export_deployment(..., include_dependencies=True)`.
It accepts only `composition-export-v1.json`, kind
`asea-composition-deployment`, integer `format_version: 1`, with
`dependencies_included: true`. It does not change or accept legacy bundle kinds.

## API / CLI wiring contract

| Parameter | Meaning |
| --- | --- |
| `archive` / `--archive` | Existing regular local export ZIP; no symlink components. |
| `output` / `--output` | **New** extraction directory; its parent must exist. |
| `spec_output` / `--spec-output` | **New** Compose JSON spec. May be inside `output`, or in an existing external parent. |
| `max_bytes` / `--max-bytes` | Positive integer upper bound on all uncompressed ZIP bytes, including the manifest; default 4 GiB. |

The function has no workspace argument and never constructs a workspace. CLI
argument parsing/dispatch is wired separately; callers should report `Blocked`
(`ValueError`) as a structured refusal. No signing keys are read or generated.

Prefer `spec_output=output / "spec.json"` for a single atomic publication:

```python
from asea.artifacts.bundle import import_bundle

result = import_bundle("export.zip", "restored", "restored/spec.json")
assert result["status"] == "UNADMITTED_REQUIRES_REEVALUATION"
assert result["admitted"] is False
```

Result fields include `status`, `admitted`, `activated`, `evidence_valid`,
absolute `output` and `spec_output`, `spec_hash`, `model_paths` (node ID to new
local directory), `dependencies_included`, `dependency_files`,
`dependency_bytes`, `reference_manifest`, and `reason`. Counts are integers and
flags are JSON booleans. The spec is an ordinary `CompositionSpec`, **not** a
wrapper containing this result.

## What is imported

- Only declared, hash-verified files under `dependencies/<original-artifact-id>/`.
- A validated original spec, with each node's `component.model_path` rebound
  through `candidate.artifacts[node_id]` to the fresh dependency directory.
  Other spec fields, including input/output types, graph edges, risk/provenance,
  and limits, are preserved. Shared artifact bindings remain shared.
- `reference/composition-export-v1.json`: the old manifest, including evaluation,
  runs and deployment, **for human reference only**. Its claims are not trusted.
- `import.json`: the explicit unadmitted import result.

Artifact directory names retain original IDs only as transport labels. New
registration computes new IDs because model paths changed. Graph/config hashes
and old execution evidence cannot establish admission in the new workspace.
No candidate, run, evaluation, deployment, active pointer, HMAC, or admission
record is installed. Provenance strings are unverified claims, not credentials.

The resulting spec can be passed to the existing real CLI:

```sh
python -m asea.compose --workspace fresh-workspace inspect --spec restored/spec.json
python -m asea.compose --workspace fresh-workspace run --spec restored/spec.json --input 'Hello'
python -m asea.compose --workspace fresh-workspace evaluate --spec restored/spec.json --suite local-suite.json
```

Use `--input-file` for audio/image graphs. Supply a locally appropriate evaluation
suite. These commands perform their usual checks; import does not guarantee that
weights are loadable, dependencies are installed, execution succeeds, or quality
passes. Even a successful import never activates anything. Tiny safetensors test
bytes validate transport only and are not production inference/admission evidence.

## Validation and resource bounds

The importer rejects duplicate JSON keys (including nested manifest/spec keys),
nonfinite numbers, unknown manifest/candidate/artifact/fingerprint fields,
non-integer versions/sizes, bad artifact manifest hashes, mismatched node bindings,
and invalid Compose schemas. Historical evaluation/run/deployment objects are
opaque untrusted JSON reference data, not validated or accepted as evidence.

ZIP names and inventory paths must be relative, canonical, portable paths. No
traversal, absolute paths, backslashes, drive/stream colons, control characters,
Windows device aliases, trailing-dot/space aliases, duplicate/case aliases,
file/directory collisions, directories, symlinks, other special files, encryption,
or unknown entries are allowed. Only stored and DEFLATE compression are accepted.
All dependency bytes are streamed with exact size and SHA-256 verification.
CRC and ZIP structure errors fail closed. The extracted tree must pass the
existing `model_inventory` validator, including its pickle/executable extension
and shard/tokenizer reference checks. Interpreted tokenizer/shard JSON also gets
duplicate-key validation. No model code, pickle, tensor loader, or shell command
runs during import, and no network access occurs.

Limits: at most 10,000 ZIP entries; central directory at most 64 MiB; export
manifest at most 16 MiB; rebuilt Compose spec at most 2 MiB; interpreted dependency
configuration at most 2 MiB; declared total uncompressed bytes at most `max_bytes`.
The importer checks disk space for extracted files, generated metadata, and a
16 MiB reserve (also checked on an external spec filesystem). No compression-ratio
heuristic is used: absolute byte limits and bounded streaming accept legitimate
highly compressible data while limiting extraction size. Free-space checks cannot
reserve capacity against concurrent writers; subsequent I/O failure rolls back.

## Publication and failure behavior

Output/spec paths must not exist, including dangling symlinks. Source archives
inside output are refused. Paths may not overlap reserved `dependencies`,
`reference`, or `import.json` destinations. Parent directories must be existing,
trusted local directories; the importer never creates or deletes outside parents.

All validation, extraction, file fsync, and directory fsync finish in a private
sibling staging directory before publication. A Linux `renameat2(RENAME_NOREPLACE)`
or macOS `renamex_np(RENAME_EXCL)` publishes the directory atomically and refuses
even an empty existing destination. Unsupported platforms/filesystems fail
closed rather than using an overwriting rename.

An external spec is staged on its own filesystem and published with an exclusive
hard link before directory publication. Exceptions roll back only this call's
owned inode(s); existing resources and racing destination creators are preserved.
There is no claim of simultaneous visibility or crash-atomicity across two
separate paths/filesystems: a process/power failure between publications can
leave an external spec pointing at absent dependencies. Use an **internal spec**
for a single atomic publication. A killed process can leave private staging
files; these are never admitted evidence. Concurrent malicious modification of
trusted parent directories by their owner is outside the filesystem threat model.
