# Data-only function oracle (v1)

This is an **opt-in, separate** public interface. It does not replace or strengthen the legacy same-interpreter `evaluate_code(source, trusted_tests, limits)` unittest oracle. There is no host-side execution of submitted source or test strings. This mode does not run repositories, pytest, dependency installers, callbacks, model workloads, or arbitrary object protocols.

## Public API and CLI

```python
from asea.certification.function_oracle import evaluate_functions

result = evaluate_functions(
    'def add(a, b): return a + b',
    [{'id': 'sum-1', 'function': 'add', 'args': [2, 3],
      'kwargs': {}, 'expected': 5}],
    # Optional SandboxLimits instance or exact-field dictionary.
    limits=None,
)
assert result['tests_total'] == 1
```

Every case requires **exactly** `id`, `function`, `args`, `kwargs`, `expected`. IDs are unique, nonempty strings or signed 64-bit integers (not booleans). The function is one ASCII Python identifier, never a dotted path or expression. Args are a list; kwargs an object. All values must be exact bounded builtins. A nonempty suite of at most 128 cases is required. Invalid API arguments raise `ValueError` (unknown limit fields can raise `TypeError`); runtime unavailability is a structured `BLOCKED` result.

The CLI is a separate process entry point, not merely a Python library:

```sh
PYTHONPATH=/agent/workspace/silt-pass2/src \
  /agent/workspace/silt-venv/bin/python -m asea.certification.function_oracle \
  --source candidate.py --cases cases.json --output new-result.json
```

`cases.json` contains the JSON array shown above. `--output` is optional. **The one JSON object on stdout is authoritative.** Candidate stdout/stderr are bounded strings inside that object, never independent verdict lines. Output files are created exclusively (`x` / `O_EXCL`); existing files and symlinks are not overwritten. Output publication failure adds `output_error` to stdout and returns nonzero without changing the functional verdict. Exit code is 0 only for a passing evaluation with successful requested publication; otherwise 1. CLI invalid data returns `{schema_version:1,status:"INVALID_INPUT",passed:false,supported:false,reason:...}`; argparse usage errors retain normal argparse behavior.

## Exact successful/runtime-result schema

Every valid API invocation returns these keys (types and meaning):

| Key | Type / meaning |
| --- | --- |
| `schema_version` | integer, `1` |
| `oracle_mode` | string, `host_data_only_functions_v1` |
| `status` | `BLOCKED`, `PASSED`, `FAILED`, `TIMEOUT`, or `RESOURCE_LIMIT` |
| `supported` | bool; actual-run trusted isolation receipt and EOF accepted |
| `passed` | bool; host-only final verdict |
| `reason` | string; empty on pass |
| `returncode` | integer or null; actual monitor termination, or failed preflight termination |
| `tests_total` | exact immutable suite length |
| `tests_run` | number of host-completed case comparisons; never candidate reported |
| `tests_passed` | number of matching host comparisons |
| `cases` | ordered list of `{id: string\|integer, matched: bool}` for completed comparisons only |
| `resource_flags` | list of host flags, e.g. `wall_timeout`, `output_limit`, `protocol_limit`, `resource_signal` |
| `stdout`, `stderr` | bounded, untrusted diagnostics, decoded with replacement |
| `source_sha256` | SHA-256 of exact UTF-8 source bytes |
| `inputs_sha256` | SHA-256 of canonical case array with `expected` removed (includes public case IDs) |
| `data_sha256` | SHA-256 of canonical full case array, including host-only expected values |
| `oracle_sha256` | SHA-256 of installed `function_oracle.py` source bytes |
| `sandbox_sha256` | SHA-256 of installed shared `sandbox.py` source bytes |
| `resources` | object described below |
| `claim` | explicit limited-observation statement |

Canonical hashed JSON is UTF-8, Unicode preserved, sorted object keys, compact separators; arrays preserve order. Digests establish provenance, not correctness. A digest of low-entropy expected data can itself permit guesses: **do not give the evidence/result artifact to an adversary if suite confidentiality matters.** The candidate never receives these hashes.

`resources` contains exactly `profile="single_process_as"`, `memory_enforcement="rlimit_as_single_process"`, `aggregate_memory_enforced=false`, `cpu_bandwidth_enforced=false`, `requested_limits` (all `SandboxLimits` fields), and `limits_established` (true after actual setup success). On `BLOCKED`, requested limits must not be interpreted as established controls. Defaults retain 256 MiB virtual address space, 2 CPU seconds, 5 wall seconds, 8 MiB tmpfs, 1 MiB per-file, 64 KiB combined diagnostics, and 256 KiB source. These are **RLIMIT_AS, not RSS**, not whole-host/aggregate-memory protection or hard CPU bandwidth. Existing at-most-two-CPU affinity and process/thread creation denial remain. No cgroup downgrade/upgrade is offered by this API.

A pass requires all cases completed exactly once, all typed comparisons matching, response framing complete, exit zero, and no host protocol/resource failure. A correct response followed by extra bytes, output flood, abnormal exit, or incomplete later cases does not pass. Wrong output is `FAILED`; timeout after isolation is `TIMEOUT`; budget overflow is `RESOURCE_LIMIT`; unavailable containment or setup failure is `BLOCKED` before candidate execution. Candidate import errors after setup remain execution failures.

## Two trust domains

The host owns the snapshotted expected outputs, comparator, definitive case list, completed list, counters, and final verdict. None are copied to the candidate root, arguments, environment, configuration, or descriptors. No tests.py, oracle implementation, reference output, future suite, seed, or host directory FD is present there. Public config contains only runtime path, namespace identities, resource limits and the three transport FD numbers.

Every invocation first runs the **fresh actual Linux namespace/runtime selftest** from `sandbox.py`, before even staging candidate source. It is not a cached flag or environment-variable permission. Windows/macOS and unavailable Linux namespace/filter/runtime controls fail closed, with no subprocess-only fallback.

Actual execution reuses the existing trusted bootstrap, capability/bounding/ambient drops, no-new-privileges, non-dumpability, read-only chroot, bounded private tmpfs, empty host environment, process/thread/socket and escape syscall denials, RLIMITs, affinity and close-fds allowlist. The syscall filter has one authoritative implementation: the new adapter shares the exact candidate-free prefix of the legacy runner.

**Setup transport ordering:** fixed code installs isolation and seccomp, writes a fixed receipt on a dedicated anonymous pipe, and closes that writer before any candidate import. The host requires both exact receipt and EOF before sending the first request. The adapter blocks on that request as its import gate. Setup and response pipes are distinct; no nonce accessible to candidate is treated as authentication.

Util-linux `unshare --fork` retains inherited descriptors in its waiting monitor, which would prevent setup EOF and request EOF. Therefore the new mode uses the same six namespace flags without `--fork`, followed by a fixed **candidate-free PID monitor** that forks into the pending PID namespace, closes all transport descriptors in its waiting parent, and execs the unchanged bootstrap in the child. The bootstrap still verifies every actual namespace changed and PID equals 1. The child sets parent-death SIGKILL, the host owns a fresh process group, and cleanup kills/reaps that group. Candidate process/thread/group/session creation remains denied. The legacy launcher is unchanged.

Only one request is outstanding. Each wire correlation ID is a fresh random string generated when that request is queued, not the public case ID. No future correlation ID exists in the candidate. IDs correlate; they do not authenticate. A candidate can inspect/replace adapter globals, patch unittest, write arbitrary response bytes or skip its declared function entirely. All that reaches the host is an **untrusted proposed value**. An exactly correct forged answer can pass that exercised case; a forged `passed`, `tests_run`, extra field, stale/unissued ID or setup claim cannot set a verdict.

## Bounded data protocol

Request: `{jsonrpc:"2.0",id:<opaque>,method:"call",params:{function,args,kwargs}}`.

Response: **exactly** `{jsonrpc:"2.0",id:<current opaque>,result:<bounded value>}`. Errors, batches, notifications, extra fields and callbacks are not accepted. All exceptions are execution failures; no exception-class expectations are certified.

Frames have a four-byte network-order unsigned length and UTF-8 JSON body. Lengths are checked before accepting advertised bodies. Defaults are fixed: 64 KiB/frame, 1 MiB total transport bytes in each direction including headers, depth 16, 4096 nodes/frame, 16 KiB/string, signed 64-bit integers, 128 cases. Full input suite bytes are capped at 1 MiB; each case is also bounded as a frame. JSON nesting/token budgets are checked lexically before decoding. Numeric conversion is bounded. Duplicate keys, invalid UTF-8, surrogates, NaN/Infinity, all floating point, custom Python objects and arbitrary object deserialization are rejected. Valid values are null, boolean, integer, scalar Unicode string, array, and string-key object. Exact recursive type equality rejects `True == 1`; strings are not normalized.

All setup/response/stdout/stderr drains and request writes are nonblocking and deadline-bound. Retained response buffers are bounded by a frame plus one 8 KiB read; diagnostic retention is independently capped. A nonreader cannot block the host writing forever. All exits, failures and cancellation paths close owned FDs and reap/kill the launch process. As with the existing sandbox, host crashes require external service lifecycle supervision; no whole-worker OOM or kernel-escape-proof guarantee is made.

## Scope and tests

Intended scope is small synchronous pure-function-style tasks, with one long-lived candidate process per suite. **Purity is not established**: candidate state persists. This certifies only correct observable outputs on the exercised inputs; not function identity, universal correctness, generalization, algorithm equivalence, hidden-suite secrecy after publication, or model admission. Same-host-UID tampering with trusted runtime/oracle files and kernel flaws remain outside this mechanism's assurance.

Run the focused finite suite:

```sh
PYTHONPATH=/agent/workspace/silt-pass2/src \
 /agent/workspace/silt-venv/bin/python -m pytest -q tests/test_function_oracle.py
```

Tests cover strict parsers, booleans versus integers, invalid suites blocked before launch, fresh probe gating, real OS positive/negative comparisons, config/host-file/FD/setup-pipe visibility, denied sockets/fork/root writes, monkeypatched unittest, forged counters/answers/IDs, trailing frames/flood/crash, nonreading input timeout, exact counts and source/data hashes, plus CLI stdout authority/exclusive output. Real tests skip explicitly where an actual Linux probe fails; skipped tests are not support evidence. No heavy models, fork bombs, genuine secrets or unbounded stress are used.
