# Experimental Composer: function I/O and per-case workers

This is the experimental pass-2 integration, not a new general certification
claim. No models are downloaded or implicitly activated. The existing public
fixture adapter remains disabled. Resource control and hostile-code isolation
are **different boundaries**.

## Public commands

```sh
PYTHONPATH=/agent/workspace/silt-pass2/src \
  /agent/workspace/silt-venv/bin/python -m asea.compose \
  --workspace /absolute/workspace evaluate \
  --spec /absolute/composition.json --suite /absolute/suite.json \
  --resource-profile process_as --memory-mib 1024 --case-timeout 60

PYTHONPATH=/agent/workspace/silt-pass2/src \
  /agent/workspace/silt-venv/bin/python -m asea.compose \
  --workspace /absolute/workspace activate --evaluation evaluation-HEX
```

The budget above is syntax, **not a model sizing recommendation**. AS reservations
can greatly exceed resident RAM; ML imports can fail with seemingly generous AS
budgets. Use operator-owned local artifacts and a separately measured budget.
High-risk provenance still requires `--approver NAME` when activating.

Evaluation flags:

| Flag | Accepted values / default |
| --- | --- |
| `--resource-profile` | `observe_only` (default), `process_as`, `cgroup_v2` |
| `--memory-mib` | integer 32–1048576, default 512 |
| `--case-timeout` | finite seconds in (0, 86400], default 60 |

`observe_only` preserves the old in-process runtime and cooperative graph
budgets. Its two new numeric flags are recorded requests **not new enforced
limits**. `process_as` launches a fresh trusted public Composer process for each
case, serially; no process reuse or fallback. AS, CPU and wall deadlines are
owned by the resource supervisor. `cgroup_v2` currently blocks **before model
generation**, because no verified delegated execution backend exists. Merely
observing a cgroup mount does not establish delegation or enforcement.

The API is:

```python
evaluate(workspace, spec, suite, allow_fixtures=False, *,
         resource_profile="observe_only", memory_mib=512, case_timeout=60.0)
```

Existing CLI JSON remains `{ok, command, result}` (or `{ok:false,error}`);
`run` success additionally includes top-level `execution_pid: integer` for
supervisor correlation. This PID alone is not trusted admission evidence.
Evaluate exits 0 only for admission; rejected/blocked evaluations exit 2.

## Exact function case schema

An `EvaluationCase` now allows:

```json
{
  "id": "target",
  "group": "target",
  "input": "Write a Python function add(a, b).",
  "reference": "Addition for these fixed examples; this is descriptive text.",
  "metric": "function_io",
  "threshold": 1.0,
  "function_cases": [
    {"id": "sum", "function": "add", "args": [2, 3], "kwargs": {}, "expected": 5}
  ]
}
```

The complete suite still requires both target and control groups, unique outer
case IDs, `name`, `reference_source` and `cases`. Use `claims:["coding"]` for a
bounded coding claim. `schema_version` remains 1; `policy_version` remains
`composition-admission-v2` for compatibility, with the separately required
`composition-execution-v1` contract below. No old record is silently upgraded.

`function_cases` defaults to null and is forbidden (except null) for other
metrics. For `function_io` it is required and nonempty. Each inner dict has
**exactly** `id,function,args,kwargs,expected`; the oracle validates original
objects before Pydantic normalization. Its canonical `_suite` validator is
reused, not duplicated. All cases are revalidated again at evaluation entry,
before any generation, including `model_copy`/`model_construct` bypass attempts.

Inner IDs are unique nonempty strings or signed 64-bit integers, not bools.
`function` is an ASCII Python identifier, not an expression/path. `args` is a
list and `kwargs` a dict. Data is limited to exact null/bool/int/string/list/dict
values: no floats, tuples, object coercions, callbacks or executable expected
strings. The oracle enforces its 128-case, byte, depth, node and integer bounds;
see [FUNCTION_ORACLE.md](FUNCTION_ORACLE.md). The descriptive `reference` is
fingerprinted but **never parsed/executed as tests** for `function_io`.

Generated text is raw Python or one `python`/`py` fence, optionally surrounded
by explanation. Multiple/malformed fences are rejected, never concatenated.
Composer calls `evaluate_functions(extract_source(output), function_cases)` in
the parent. Expected outputs, comparison counters and the definitive verdict
remain outside the candidate interpreter. A pass requires all host comparisons,
zero exit, actual isolation setup, no resource/protocol failure, and consistent
counts; stdout `passed:true` and returned verdict-shaped objects are only data.

`functional_code` is unchanged: `reference` is trusted unittest source, using the
legacy same-interpreter harness. It is recorded as
`legacy_same_interpreter_unittest`, **not** as an external correctness oracle.
Coding target/control coverage may use either metric; mixed suites retain both
mode/scope records. Neither supports universal correctness, purity, algorithm
identity or safety certification.

## Evidence additions (exact fields)

All newly written evaluations, including observe-only, contain these additions:

```text
execution_config = {
  version: "composition-execution-v1",
  resource_profile: "observe_only" | "process_as" | "cgroup_v2",
  memory_mib: integer,
  case_timeout: number,
  memory_semantics: "observed_process_lifetime_rss_no_new_hard_limit"
                  | "virtual_address_space_per_process"
                  | "cgroup_accounted_memory_requested_not_enforced",
  limits_enforced_by_request: false (observe_only) | null (other profiles)
}
execution_config_hash: SHA256 of Composer canonical execution_config JSON
implementation_fingerprint = {
  schema_version: 1,
  implementation_hashes: {relative_source_path: {sha256: string, size: integer}},
  environment: {
    python: string, executable: absolute_path, platform: string,
    dependencies: {distribution_name: version_string | null},
    runtime_tools: {tool_name: {path: absolute_path, file_hash: {sha256,size}} | null}
  }
}
oracle_modes: {metric_name: {mode: string, scope: string}}
worker_attempts: array (empty in observe_only)
```

`limits_enforced_by_request:null` is deliberately not an enforcement claim:
only successful actual worker receipts establish the requested controls.
Source hashes cover the gate, oracle, sandbox, runtime, schema, public Composer
CLI, artifact signing/registration, resource supervisor and bootstrap.
Distribution versions cover Pydantic, Torch, Transformers, Tokenizers,
Safetensors, NumPy, Pillow, SciPy, SoundFile, SentencePiece and Accelerate.
Runtime tool hashes cover Python, base Python, `unshare` and `ldd`. This is not a
full native shared-library SBOM, package-content attestation or proof that a
trusted operator has not monkeypatched a live Python process.

Protected evaluations additionally have:

```text
spec_snapshot: {path: absolute_path, file_hash: {sha256,size}}
worker_attempts[]: {
  case_id: string,
  resource_evidence: complete asea.execution result,
  resource_evidence_hash: SHA256 of that result
}
case_evidence[].worker_receipt: {
  resource_evidence: complete asea.execution result,
  workspace: absolute_path,
  arguments: ["run", "--spec", absolute_snapshot_path, ...input_flags],
  run_id: string, run_hash: SHA256 of signed run payload,
  output_hash: SHA256 of signed run output
}
case_evidence[].worker_receipt_hash: SHA256 of worker_receipt
```

For cancellation before the supervisor returns, the incomplete attempt instead
has exactly `case_id`, `status:"interrupted"`, `reason` (exception class),
`resource_evidence:null`, and `cleanup` (explicit lack of completed receipt).
It is never admitted. Timeout/failure supervisor results are retained in normal
attempt rows even when no case score could be produced.

Function I/O rows additionally have:

```text
oracle_evidence: complete evaluate_functions result
oracle_evidence_hash: SHA256 of oracle_evidence
function_cases_hash: SHA256 of Composer canonical function_cases JSON
```

Composer canonical hashes use sorted keys, compact separators and ASCII JSON
escaping. Oracle-internal `data_sha256`/`inputs_sha256` use its own Unicode-
preserved canonicalization; the two hash formats are not interchangeable.
See [FUNCTION_ORACLE.md](FUNCTION_ORACLE.md) and
[RESOURCE_CONTROLS.md](RESOURCE_CONTROLS.md) for the nested result schemas.

## Lock, integrity and activation lifecycle

1. Under the workspace writer, register immutable candidate bindings and write
   a content-addressed read-only spec snapshot in `evaluation-specs/`.
2. Release the parent writer. Run each child serially via
   `execution.run("compose", ["run","--spec",...], workspace=the_same_workspace)`.
   The child acquires its own writer and writes its own normal HMAC-signed run.
3. Treat stdout only as a run locator/envelope. Read and verify the HMAC record,
   require that the run ID did not preexist the launch, and compare the whole
   returned payload, supervisor/CLI PIDs, output digest, input and graph bindings.
   The parent does **not** manufacture/sign a supplied run or functional verdict.
4. Compute the metric in the parent. Recheck implementation, artifacts, inputs
   and snapshot. Finalize the signed evaluation under a reacquired writer.
5. Activation rechecks HMAC records, exact receipt hashes, requested/effective
   AS/CPU limits, actual zero exit, PID/run uniqueness, arguments, output/graph/
   input bindings, source/environment fingerprints, and source snapshot. It
   reruns the host functional oracle (not model generation) and compares source,
   data and oracle implementation bindings before accepting the score.

A changed implementation/environment is `stale_implementation_fingerprint`.
An older record missing required fingerprint/contract is
`stale_evidence_version_not_supported`, requiring reevaluation. No migration
silently treats old records as externally evaluated. Resource profiles are
bound to evaluation evidence, **not added to the underlying graph hash**.

Cancellation invokes supervisor kill/reap cleanup. The next parent writer
reconciles interrupted pending outputs into immutable recovery records; neither
timeouts nor abandoned outputs are dropped/promoted/retried in-process. A
concurrent living writer can still cause a deliberate lock refusal.

## Limits and regression coverage

`process_as` enforces **per-process virtual address space, not RSS**, and is not
aggregate process-tree memory, cgroup accounting, hard CPU bandwidth, process
count or thread-count enforcement. Runtime RSS remains an observed admission
metric. This supervisor runs trusted SILT/model-loader code, not hostile source;
namespace/seccomp candidate isolation remains the separate oracle boundary.
The generation deadline does not replace the oracle's separate small-source
sandbox limits. Same-UID workspace owners and installed runtime code are trusted;
HMAC records are local integrity checks, not remote attestation.

`tests/test_pass2_integration.py` covers exact data validation, pre-generation
cgroup blocking, default-path preservation, mocked serial/unlocked worker
bindings, stale-child replay, forged PID/output/limits/exit, cancellation/timeout
recovery, activation staleness, and actual public oracle CLI addition/forged-pass/
return-spoof cases. It also runs an actual limited Composer child that rejects
the public fixture adapter before generation. Mock-worker positive tests are
orchestration tests **not real kernel/model admission evidence**. Actual oracle
CLI tests fail closed and skip with a stated reason if containment is unavailable.
No heavyweight model load is required; protected real-model end-to-end admission
must be measured separately by the operator.
