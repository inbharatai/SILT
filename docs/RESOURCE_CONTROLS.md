# Resource controls: fixed, fresh SILT workers

`asea.execution` is an additive, optional-ML-free resource supervisor for the
**trusted public `asea.compose` and `asea.compiler` modules only**. It does not
execute arbitrary shell commands, Python strings, supplied module names, user
candidate source, plugins, or callbacks. The worker's application arguments are
parsed by those existing public CLIs; argument strings are never shell-expanded.

**Resource control is not a hostile-code sandbox.** Candidate code must continue
through the separate OS-isolation backend in `asea.certification`; this wrapper
must not be used as an unsandboxed candidate fallback. The trust assumptions
include the interpreter, installed site packages, SILT source, and existing
trusted local HF loaders. Models must retain the loaders' existing offline,
local-file and no-remote-code checks. Offline environment variables are not a
kernel network boundary. Filesystem permissions are unchanged for trusted code.

## Public CLI

From an installed SILT environment (or the repository with `PYTHONPATH=src`):

```sh
python -m asea.execution probe
python -m asea.execution --help

python -m asea.execution run \
  --profile process_as --memory-mib 256 --timeout 10 \
  --operation compose --workspace ./workspace -- list

python -m asea.execution run \
  --profile process_as --memory-mib 256 --timeout 10 \
  --operation compose --workspace ./workspace -- inspect --spec ./composition.json

python -m asea.execution run \
  --profile process_as --memory-mib 256 --timeout 10 \
  --operation compiler -- --help

python -m asea.execution run \
  --profile process_as --memory-mib 3072 --timeout 120 --cpu-seconds 110 \
  --operation compiler -- infer --model ./local-model --prompt 'Hello'
```

The last example is syntax, **not a measured safe model/budget recommendation**.
ML libraries often reserve much more virtual address space than their RSS; an AS
budget that looks generous in RAM terms may block import or model mapping. No
large-model load is necessary to validate the kernel control.

`--` is required before application arguments. `--workspace` is a compose-only
convenience that prepends its absolute workspace argument. Compiler paths belong
to the compiler's arguments after `--`; there is no compiler workspace rewrite.

The supervisor emits one JSON object. Captured child output is in `stdout` and
`stderr` strings, so even compiler text help remains inside valid supervisor
JSON. Exit code is `0` only for supervisor status `OK`, otherwise `2`; **the
actual child exit code or negative signal is separately preserved in
`returncode`**. Probe exits `0` for a completed read-only observation, not a
positive execution guarantee. Ordinary application errors are not automatically
resource errors or platform-unsupport errors.

## Fresh-process integration API

```python
from asea.execution import probe, run

capabilities = probe()
evidence = run(
    "compose", ["run", "--spec", "/absolute/composition.json", "--input", "hello"],
    profile="process_as", memory_mib=3072, timeout=120,
    cpu_seconds=110, workspace="/absolute/workspace",
)
if evidence["status"] != "OK":
    # Surface the failure and measured elapsed/cpu budget. Never rerun in-process.
    ...
```

Signature:

```text
run(operation, arguments=(), *, profile="process_as", memory_mib=512,
    timeout=60.0, workspace=None, cpu_seconds=None, delegated_root=None) -> dict
probe(delegated_root=None) -> dict
```

Each call starts a new interpreter. There is no user-supplied worker callable,
executable or source-code option. Wiring a runtime/Studio wrapper means invoking
this API/public command **instead of loading the model in the requesting
process**. This change alone does not retrofit existing in-process calls or bound
models already loaded in a supervisor. After failure, timeout, or denied profile,
there is no automatic weaker-profile or in-process retry.

## `process_as`: implemented Linux profile

1. Start a trusted bootstrap by absolute path using the **same interpreter** with
   `-I -S`, `close_fds=True`, a new session/process group, and no inherited stdin.
   Only output pipes and one noninheritable control pipe are passed. `-S` prevents
   site/.pth code before limits. The bootstrap does not import SILT/ML first.
2. Install **hard and soft `RLIMIT_AS=(N,N)`**, **`RLIMIT_CPU=(C,C+1)`**, and
   `RLIMIT_CORE=(0,0)`; read back exact values. Failure ends the bootstrap before
   public-module import. Default `C=ceil(timeout)`; explicit C is integral seconds.
3. Only then initialize trusted installation site directories and import one of
   the two fixed public modules. Installation directories come from the parent
   interpreter's `site.getsitepackages()`, not its arbitrary `sys.path`, user
   site, cwd, environment, model files, or a caller-supplied directory. This also
   supports virtual environments on Python versions where `-S` changes prefixes.
4. Use a host monotonic deadline covering **validation, setup, startup, imports,
   model load, and execution**. Drain output concurrently, retaining at most
   1 MiB combined stdout/stderr and 16 KiB control data. Kill the process group on
   timeout/output overflow and also on normal leader exit; reap the worker.
   Cancellation/exception cleanup also attempts group kill and reaping.
5. Collect `wait4` CPU usage for **this particular child**. Do **not** use its
   `ru_maxrss` as model RSS: native testing found that it retains the requesting
   parent's pre-exec fork watermark, even across isolated startup. Instead,
   observe this worker image's `/proc/PID/status` **VmHWM**, only after the
   bootstrap limit receipt; also collect a final worker observation on orderly
   exit. KiB are converted to bytes. Raw wait4 RSS is kept separately and labeled
   `wait4_peak_rss_bytes_including_preexec`, never used as model peak.

### Guarantees and non-guarantees

| Property | Actual semantics |
|---|---|
| Memory limit | Hard **virtual address space per process**. Not hard RSS; not aggregate model/child/host memory. Inherited AS limits still apply separately to descendants. |
| CPU limit | Kernel per-process CPU-time limit, not CPU bandwidth, core count, wall time, or combined descendant CPU. Soft expiry normally emits `SIGXCPU`; hard expiry may emit `SIGKILL`, which alone cannot identify cause. |
| Threads | Known BLAS/OpenMP pools are **requested** to use one thread through environment settings. This is not a hard process/thread count ceiling. |
| Process count | Not hard-bounded by this profile. `RLIMIT_NPROC=1` is deliberately **not** set: it is real-UID scoped, includes threads on Linux, has root/capability exceptions, and breaks ordinary Torch initialization. |
| Deadline cleanup | Kill the worker's POSIX process group and reap its leader, including on cancellation. Trusted descendants in the group are killed. It is not escape-proof containment of hostile children calling `setsid`, nor proof all adopted grandchildren have been reaped by init. |
| RSS telemetry | Worker-image kernel VmHWM observations, excluding inherited parent pre-exec watermark. Linux `/proc` RSS counters are estimates with asynchronous accounting, not byte-exact instrumentation. Fatal exits may prevent a final observation, so the value is then the highest observed lower-bound estimate. Not summed descendant RSS or enforcement. |
| Whole-host OOM | Not prevented. Parent preparation, other jobs/Studio roots, file/kernel allocations and independent processes are outside this per-process bound. |

A caller requiring aggregate memory, hard process/thread counts or CPU bandwidth
must **not accept `process_as` as equivalent**. Request `cgroup_v2` and handle its
current blocked outcome; no downgrade is performed. Concurrent jobs need a
separate operator/service-wide policy. Cleanup and scheduler latency can make
returned `elapsed_seconds` exceed the requested wall budget slightly; kernel
uninterruptible I/O can delay reaping despite SIGKILL. There is no promise of a
hard real-time return deadline or cleanup after supervisor SIGKILL/host crash.

### Environment and evidence

The environment is rebuilt, not copied: no tokens, credentials, proxy settings,
`PYTHONPATH`, `PYTHONHOME`, `LD_*`, arbitrary user PATH, or inherited HOME. It has
an ephemeral private HOME/TMPDIR, OS-default PATH, fixed locale, offline HF flags,
disabled tokenizer parallelism/CUDA visibility, and single-thread BLAS/OpenMP
hints. Existing absolute directory values for `HF_HOME`, `HF_HUB_CACHE`,
`HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, and `XDG_CACHE_HOME` are the only
inherited cache configuration. An existing conventional `~/.cache/huggingface`
can be preserved explicitly as HF_HOME; these are paths, never Python imports.

Evidence distinguishes:

- `launched`: bootstrap started; not itself proof the module started.
- `supported` / `enforcement_established`: this run's exact kernel-limit receipt
  was received, not model success or hostile-code isolation support.
- `enforcement_tested`: false on probe/ordinary runs. The private allocation
  fixture sets it true only after observing its actual kernel-bounded failure.
- `requested_limits` versus `effective_limits`: exact AS bytes, CPU seconds, wall
  budget and output allowance; a setup failure cannot be reported as enforced.
- `elapsed_seconds`: host monotonic total through result collection;
  `setup_elapsed_seconds`: time to successful limit receipt;
  `cpu_consumed_seconds`: this child's wait4 user+system CPU.
- `observed_peak_rss_bytes` and `peak_rss_source`: worker-image VmHWM observation,
  not a quota. `peak_rss_complete=true` means a final orderly-exit observation was
  received, not byte-exact RAM accounting. False means a final observation was
  unavailable (e.g. SIGKILL/SIGXCPU); null means none could be obtained. Polling
  these kernel high-water counters is telemetry only, never a hard-RSS limit.
- `status`: `OK`, `FAILED`, `BLOCKED`, `TIMEOUT`, `RESOURCE_LIMIT`, `OUTPUT_LIMIT`.
  An intercepted `MemoryError` under RLIMIT_AS is labeled
  `worker_MemoryError_under_kernel_RLIMIT_AS_not_RSS_OOM`, **not a cgroup/RSS OOM**.
  The small allocation fixture exits `70`; wall timeout preserves `-SIGKILL`;
  `SIGXCPU` is identified as CPU exhaustion. Other SIGKILLs are not guessed to be
  OOM. Application code may catch its own MemoryError and return a normal error
  envelope; that remains an application failure rather than fabricated OOM proof.
- `group_kill_succeeded`, `pipes_drained`, `descendant_closure`: limited lifecycle
  evidence, with process-group rather than cgroup/namespace semantics.

The worker's control channel is trusted application telemetry, not a candidate
protocol/authentication boundary. It is noninheritable across exec. Existing
candidate sandbox FD isolation must remain intact; do not grant candidate access
to this worker or its control channel.

## Cgroup v2: visible is not delegated; execution pending

`probe` reads platform/native control information only. It distinguishes visible
controllers, proposed delegation, actual delegation verification, write testing,
and enforcement testing. It **never writes or creates a cgroup**, migrates a PID,
requests privileges, asks systemd for delegation, or changes a system root.

On this Linux worker, `memory.max`, `pids.max`, `cpu.max` and
`cgroup.subtree_control` were readable but `os.access(..., os.W_OK)` was false for
all four. No authorized delegated root was available. Neither root identity,
a writable mount nor a readable `cgroup.controllers` proves delegation.

This release deliberately does **not** ship an unverified cgroup execution
backend. These both fail closed **before worker launch**:

```sh
python -m asea.execution run --profile cgroup_v2 --memory-mib 256 --timeout 5 \
  --operation compose -- --help

python -m asea.execution run --profile cgroup_v2 --memory-mib 256 --timeout 5 \
  --delegated-root /operator/authorized/leaf-parent --operation compose -- --help
```

`--delegated-root` records operator intent, not proof of authority or support.
Even a supplied writable ordinary directory or `/sys/fs/cgroup` remains blocked.
A future backend must validate an operator-authorized delegated hierarchy,
create only its own leaf, effectively write/read back mandatory memory/swap/pids/
CPU controls, attach and verify a gated worker **before model import/children**,
collect correctly attributed events, and kill/verify empty population/clean up
only owned cgroups. It must retain existing candidate OS isolation when used for
candidates. Until native tests pass, no aggregate or cgroup enforcement is claimed.

## Windows and other native platforms

On real Windows, probe may discover the native Job Object API via ctypes symbol
lookup; it does not create/modify a job or process. **Execution remains BLOCKED**.
Job Objects alone are resource/lifecycle controls, not sufficient hostile-code
filesystem/network security. A future implementation requires real native
suspended assignment/readback/commit-limit/CPU/process/kill-on-close tests plus a
separately validated security boundary. A mock platform flag is not evidence.
Other non-Linux platforms also remain explicitly unsupported by this profile.

## Reproducible small tests and pending target tests

```sh
python -m pytest tests/test_execution_limits.py -q
```

The Linux tests use the actual kernel and fresh interpreter: one finite
AS-over-limit allocation fails before touching that oversized memory; a one-CPU-
second bounded loop gets SIGXCPU; finite sleepers and one same-group descendant
are killed on a short deadline; RSS is checked per new child; inherited FD/env
leakage is checked; public compose help/list exercise the fixed CLI. No models,
root/controller writes, privilege requests, user code execution, unbounded
allocator loops or fork bombs are involved. Test fixtures are private, a fixed
enumeration, and inaccessible through the public operation allowlist.

The suite contains **explicitly skipped target-test templates** for delegated
cgroup and Windows enforcement. Their skips are pending work, not passing
support evidence. The native platform contract test uses the actual host OS;
Linux test success cannot substantiate Windows support.

Reference: Linux `getrlimit(2)` and the kernel cgroup v2 documentation distinguish
AS, CPU time, pids and hierarchical-accounting semantics:

- https://man7.org/linux/man-pages/man2/getrlimit.2.html
- https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html
- https://systemd.io/CGROUP_DELEGATION/
- https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
