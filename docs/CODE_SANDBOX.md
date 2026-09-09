# Optional Linux functional-code sandbox

`asea.certification.sandbox` is an opt-in extension. It does not change existing
certification defaults, automatically admit a model, or certify adversarial
correctness. No package re-export is required.

```python
from asea.certification.sandbox import evaluate_code, probe_code_sandbox, SandboxLimits

support = probe_code_sandbox()
result = evaluate_code(
    'def add(a, b): return a + b',
    '''import unittest
import candidate
class Check(unittest.TestCase):
    def test_add(self):
        self.assertEqual(candidate.add(2, 3), 5)
''',
    SandboxLimits(wall_seconds=5, cpu_seconds=2),
)
print(result.to_dict())
```

The two arguments are **separate Python source strings**. Trusted tests import
`candidate`, use stdlib `unittest`, and must contain at least one executed test.
The harness counts failures/errors/skips/unexpected successes as non-passes.
There is no shell interpolation, model-print parsing, AST allowlist, or
unsafe host-side `exec` of candidate code. Math, regex and unittest are available;
third-party packages, shell commands, network and multiprocessing are not.

## Callable contract

- `probe_code_sandbox(limits=None) -> SandboxResult`: real containment selftest.
- `evaluate_code(source, trusted_tests, limits=None) -> SandboxResult`: performs
  a fresh complete selftest before every candidate, then runs in a second fresh
  namespace. Failed selftests return **BLOCKED before candidate execution**.
- `SandboxResult`: `status`, `supported`, `passed`, bounded `stdout`/`stderr`,
  `reason`, `returncode`, `tests_run`, `resource_flags`; `.to_dict()` is available.
- Status: `BLOCKED`, `PASSED`, `FAILED`, `TIMEOUT`, `RESOURCE_LIMIT`.
  `supported` means the execution reached the isolated runner, not that the
  answer is correct. BLOCKED must never count as a functional pass.
- Invalid source types/sizes or limit values raise `ValueError` before execution.
- Resource flags identify observed wall timeout, output overflow, or resource
  signals. A caught `MemoryError` or file-limit exception can instead yield FAILED
  or even PASSED if that is what the trusted tests expect. The runner cannot
  reliably infer every resource-limit encounter from arbitrary code.

## Kernel boundary (actual OS isolation)

Requirements: Linux x86-64 or aarch64, Python 3.9+, util-linux `unshare`, `ldd`,
permission for unprivileged user/mount/PID/network/IPC/UTS namespaces, bind and
tmpfs mounts, capability manipulation, `no_new_privs`, and seccomp BPF. This is
stdlib-only Python; no privileged helper is installed. Unsupported configurations
(including Windows/macOS) fail closed. The implementation presently requires a
trusted system interpreter and stdlib under `/usr`, not a user-managed runtime.

The bootstrap verifies changed namespace IDs and PID 1, makes mount propagation
private, constructs a new chroot from a copied Python executable, stdlib and its
resolved shared-library dependencies, and bind-remounts that root read-only,
nodev, nosuid. It mounts only `/tmp` writable on a size/inode-bounded tmpfs with
nodev/nosuid/noexec. No `/proc`, `/dev`, `/etc`, home, project directory, credential
file, host network interface, GPU device, or original filesystem root is exposed.
The root contains only runtime files, runner/config, candidate and trusted tests.
`/work` is the test root and is read-only; write temporary outputs to `/tmp`.
Python runtime files are intentionally readable outside `/work`.

All effective/permitted/inheritable, bounding and ambient capabilities are dropped
before candidate execution. `no_new_privs` is set; bootstrap memory is discarded
by exec of an isolated `-I -B` Python with a tiny fixed environment. Inherited
host descriptors are closed except explicit output/protocol pipes and `/dev/null`
stdin. Candidate access to these channels does not grant a host file descriptor
or host filesystem/network access.

A seccomp filter, with architecture checking and x32 rejection, denies process
creation (fork/vfork/clone/clone3), exec, socket/socketpair, namespace/mount-changing
operations and selected dangerous kernel APIs including ptrace, process_vm,
keyring, BPF, userfaultfd, io_uring, SysV IPC and handle-based file opens.
`setsid` and `setpgid` are denied too, so even disabling PDEATHSIG cannot let
candidate code leave the parent's kill group. This is a
**deny filter**, not a complete syscall allowlist. The process policy is stricter
than a process-count limit: **no child processes or threads**. This avoids the
namespace-root exemption and per-real-UID complications of RLIMIT_NPROC.
An RLIMIT_NPROC of 1 is still installed, but is not relied on alone.

Default hard limits: 256 MiB address space, 2 seconds CPU (3-second hard kill),
5 seconds wall time, 8 MiB tmpfs with 256 inodes, 1 MiB per file, 64 KiB combined
stdout/stderr, 16 descriptors, no core dumps and affinity of at most two existing
CPUs. Configurable values have hard ceilings. The protocol channel is separately
capped at 8 KiB. The parent drains output with bounded buffers and kills on
flooding. Output may contain terminal escape characters: display it as plain text,
not executable HTML or terminal control output.

On timeout/overflow the parent SIGKILLs the new process group. Killing namespace
PID 1 destroys its descendants; process creation and process-group/session
changes are additionally denied, so a candidate cannot escape the kill group. CPU hard-limit kill
remains important because PID 1 treats default signal dispositions specially.
No process-per-input host stress test is used. Preparation is trusted host work,
not included in the candidate wall-time budget. Concurrent evaluations multiply
resource use; callers must limit concurrency for a 4 GiB / 2 CPU worker.

The preflight runs the same bootstrap/root/filter and checks: capabilities zero,
no_new_privs/seccomp enabled, namespace PID and CPU affinity, NPROC, missing proc
and devices, read-only test root, tmpfs bound/writeability, inability to read a
non-sensitive scratch canary outside the root, and EPERM for a single socket and
fork attempt. It never reads real secrets or runs an actual fork bomb. Namespace
creation/mount success alone is not considered adequate support.

## Correctness honesty — NOT an independent adversarial oracle

Trusted test source and candidate functions execute in the **same Python process**.
The parent independently requires successful exit **and** a well-formed,
per-run nonce-bearing readiness/result protocol from the harness, with a positive
test count. Printing `pass`, bare `os._exit(0)`, or raising `SystemExit(0)` does not
produce a successful test result. The harness catches BaseException and captures
some runner entry points before candidate import.

However, arbitrary Python can inspect interpreter state, read test/config source,
locate the nonce/output descriptor, monkeypatch unittest, mutate result objects,
or forge the completion protocol. **The nonce is correlation/framing, not a
secret authentication barrier against code in that interpreter.** Captured
entry points, class restrictions or AST scanning cannot fix this. Even a normal
function can inspect test inputs and hardcode answers. PASSED therefore means
ordinary functional test observation, NOT proof against a deliberately malicious
candidate spoofing the oracle. Consumers requiring that guarantee must use an
independent oracle process with a constrained serialization/RPC function API,
external hidden inputs/expected outputs, and no test/correctness state inside the
candidate interpreter. This general Python-callable runner does not implement
such an oracle and must not be promoted as an adversarial correctness gate.

## Threat model and residual risks

This reduces the host impact of untrusted generated Python, including unrestricted
imports/object introspection: Python syntax is not the security boundary. It is
not bulletproof, a VM, a multi-tenant hostile-user service, or a defense against
kernel vulnerabilities. A malicious local user with the same host identity could
alter staging/runtime/helper files or interfere with the parent; that adversary
is explicitly out of scope. The trusted system runtime/kernel/admin must be
trusted. No GPU access exists. tmpfs/RLIMIT_AS are not a cgroup aggregate quota;
process/thread creation denial is essential to keeping the single-process bound.
Runtime preparation temporarily copies stdlib files to a private host temp dir;
only the namespace's bounded tmpfs is candidate-writable. Parent/kernel crashes
may leave staging data behind, requiring ordinary temp-dir cleanup policy.

## Verification

Run with the project interpreter:

```text
/agent/workspace/silt-venv/bin/python -m pytest tests/test_code_sandbox.py -q
```

Live isolation tests skip with the explicit probe reason where the platform cannot
provide the boundary. Fail-closed tests still run. Tests use scratch canaries,
a single denied fork/socket attempt, bounded disk writes, memory allocation,
infinite loops, output floods, real passing/failing unittest suites and early-exit
spoof attempts. Passing these tests does not resolve the same-interpreter oracle
limitation described above.
