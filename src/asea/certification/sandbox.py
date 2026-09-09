"""Opt-in Linux namespace code evaluation. Never an automatic admission decision.

See docs/CODE_SANDBOX.md for the threat model and correctness limitations.
The default policy forbids creating processes and sockets, not merely limits them.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import platform
import secrets
import selectors
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time
from typing import Optional


@dataclasses.dataclass(frozen=True)
class SandboxLimits:
    wall_seconds: float = 5.0
    cpu_seconds: int = 2
    memory_bytes: int = 256 * 1024 * 1024
    tmp_bytes: int = 8 * 1024 * 1024
    file_bytes: int = 1024 * 1024
    output_bytes: int = 64 * 1024
    source_bytes: int = 256 * 1024

    def validate(self):
        # Hard ceilings keep callers from accidentally removing containment.
        bounds = {"wall_seconds": (0.1, 30), "cpu_seconds": (1, 10),
                  "memory_bytes": (64 * 1024**2, 1024**3),
                  "tmp_bytes": (4096, 32 * 1024**2),
                  "file_bytes": (1024, 8 * 1024**2),
                  "output_bytes": (1024, 1024**2),
                  "source_bytes": (1024, 1024**2)}
        for key, (low, high) in bounds.items():
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
                raise ValueError("invalid sandbox limit: " + key)
            if key != "wall_seconds" and not isinstance(value, int):
                raise ValueError("integer sandbox limit required: " + key)


@dataclasses.dataclass(frozen=True)
class SandboxResult:
    status: str  # BLOCKED, PASSED, FAILED, TIMEOUT, RESOURCE_LIMIT
    supported: bool
    passed: bool = False
    stdout: str = ""
    stderr: str = ""
    reason: str = ""
    returncode: Optional[int] = None
    tests_run: int = 0
    resource_flags: tuple = ()

    def to_dict(self):
        return dataclasses.asdict(self)


# Trusted bootstrap runs only inside unshare; it never evaluates candidate code.
_BOOTSTRAP = r'''
import ctypes, json, os, resource, sys
root, config_path = sys.argv[1:]
with open(config_path) as f: c = json.load(f)
lib = ctypes.CDLL(None, use_errno=True)
def check(value, what):
    if value != 0: raise OSError(ctypes.get_errno(), what)
def mount(source, target, kind, flags, data=None):
    check(lib.mount(source, target.encode(), kind, ctypes.c_ulong(flags), data), 'mount')
# Verify all namespaces actually changed, including PID (not only pid_for_children).
for kind, old in c['namespaces'].items():
    if os.readlink('/proc/self/ns/' + kind) == old: raise RuntimeError('namespace unchanged: ' + kind)
if os.getpid() != 1: raise RuntimeError('not PID namespace init')
mount(None, '/', None, (1 << 18) | (1 << 14))  # MS_PRIVATE | MS_REC
mount(root.encode(), root, None, 4096)  # bind root
mount(None, root, None, 4096 | 32 | 1 | 2 | 4)  # remount bind RO,nosuid,nodev
mount(b'tmpfs', root + '/tmp', b'tmpfs', 2 | 4 | 8,
      ('size=%d,nr_inodes=256,mode=1777' % c['limits']['tmp_bytes']).encode())
os.chroot(root)
os.chdir('/work')
# Remove every namespace capability, including bounding and ambient sets.
for cap in range(64):
    if lib.prctl(24, cap, 0, 0, 0) != 0 and ctypes.get_errno() != 22:
        raise OSError(ctypes.get_errno(), 'cap bounding drop')
check(lib.prctl(47, 4, 0, 0, 0), 'ambient clear')
class Header(ctypes.Structure): _fields_ = [('version', ctypes.c_uint32), ('pid', ctypes.c_int)]
class Data(ctypes.Structure): _fields_ = [('effective', ctypes.c_uint32), ('permitted', ctypes.c_uint32), ('inheritable', ctypes.c_uint32)]
h = Header(0x20080522, 0); d = (Data * 2)()
check(lib.capset(ctypes.byref(h), ctypes.byref(d)), 'capset')
check(lib.prctl(38, 1, 0, 0, 0), 'no_new_privs')
check(lib.prctl(4, 0, 0, 0, 0), 'dumpable')
l = c['limits']
for which, pair in [(resource.RLIMIT_AS, (l['memory_bytes'],)*2),
                    (resource.RLIMIT_CPU, (l['cpu_seconds'], l['cpu_seconds'] + 1)),
                    (resource.RLIMIT_NPROC, (1, 1)),
                    (resource.RLIMIT_FSIZE, (l['file_bytes'],)*2),
                    (resource.RLIMIT_NOFILE, (16,16)), (resource.RLIMIT_CORE, (0,0))]:
    resource.setrlimit(which, pair)
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
# exec discards bootstrap's host-loaded modules/memory; no inherited host env.
os.execve(c['python'], [c['python'], '-I', '-B', '/work/runner.py'],
          {'PATH': '/nonexistent', 'HOME': '/tmp', 'TMPDIR': '/tmp', 'LANG': 'C.UTF-8'})
'''

_RUNNER = r'''
import ctypes, errno, importlib.util, io, json, os, platform, resource, socket, sys, unittest
with open('/work/config.json') as f: config = json.load(f)
lib = ctypes.CDLL(None, use_errno=True)
# Architecture checked before enabling a deny filter; x32 ABI is killed too.
arch = platform.machine()
if arch == 'x86_64':
    audit = 0xc000003e
    denied = [29,30,31,41,53,56,57,58,59,62,64,65,66,67,68,69,70,71,
              101,109,112,155,161,165,166,169,175,176,200,234,
              246,248,249,250,272,298,303,304,308,310,311,313,317,321,322,323,
              425,426,427,428,429,430,431,432,435,442]
elif arch == 'aarch64':
    audit = 0xc00000b7
    denied = [39,40,41,51,97,104,105,106,117,129,130,131,142,154,157,
              186,187,188,189,190,191,192,193,194,195,196,197,198,199,220,
              221,224,241,264,265,268,270,271,273,277,280,281,282,
              425,426,427,428,429,430,431,432,435,442]
else: raise RuntimeError('unsupported seccomp architecture')
class Filter(ctypes.Structure):
    _fields_ = [('code', ctypes.c_ushort), ('jt', ctypes.c_ubyte), ('jf', ctypes.c_ubyte), ('k', ctypes.c_uint32)]
class Program(ctypes.Structure):
    _fields_ = [('len', ctypes.c_ushort), ('filter', ctypes.POINTER(Filter))]
f = [(0x20,0,0,4), (0x15,1,0,audit), (0x06,0,0,0x80000000),
     (0x20,0,0,0), (0x35,0,1,0x40000000), (0x06,0,0,0x80000000)]
for nr in denied: f.extend([(0x15,0,1,nr), (0x06,0,0,0x00050000 | errno.EPERM)])
f.append((0x06,0,0,0x7fff0000))
a = (Filter * len(f))(*(Filter(*x) for x in f)); p = Program(len(f), a)
if lib.prctl(38,1,0,0,0) != 0 or lib.prctl(22,2,ctypes.byref(p),0,0) != 0:
    raise RuntimeError('seccomp unavailable')
# Report readiness only AFTER kernel isolation and seccomp have been installed.
write = os.write
protocol_fd = config['protocol_fd']
nonce = config['nonce']
def report(event, **kw):
    write(protocol_fd, (json.dumps(dict(nonce=nonce, event=event, **kw)) + '\n').encode())
if config['probe']:
    class H(ctypes.Structure): _fields_ = [('v', ctypes.c_uint32), ('pid', ctypes.c_int)]
    class D(ctypes.Structure): _fields_ = [('e',ctypes.c_uint32),('p',ctypes.c_uint32),('i',ctypes.c_uint32)]
    h = H(0x20080522,0); d = (D*2)()
    assert lib.capget(ctypes.byref(h),ctypes.byref(d)) == 0
    assert not any(x.e or x.p or x.i for x in d)
    assert lib.prctl(39,0,0,0,0) == 1
    assert lib.prctl(21,0,0,0,0) == 2
    assert os.getpid() == 1 and len(os.sched_getaffinity(0)) <= 2
    assert resource.getrlimit(resource.RLIMIT_NPROC) == (1,1)
    assert not os.path.exists('/proc') and not os.path.exists('/dev')
    assert not os.path.exists(config['canary'])
    assert os.statvfs('/tmp').f_blocks * os.statvfs('/tmp').f_frsize <= config['limits']['tmp_bytes'] + 4096
    try:
        open('/work/forbidden','w')
        raise AssertionError('writable root')
    except OSError as e: assert e.errno == errno.EROFS
    for operation in [lambda: socket.socket(), lambda: os.fork(),
                      lambda: os.setsid(), lambda: os.setpgid(0, 0)]:
        try:
            operation()
            raise AssertionError('forbidden syscall worked')
        except OSError as e: assert e.errno == errno.EPERM
    with open('/tmp/probe','w') as f: f.write('ok')
    report('ready')
    report('result', passed=True, tests_run=1)
else:
    report('ready')
    # Capture the trusted runner entry points before candidate execution.
    suite_loader = unittest.TestLoader().loadTestsFromModule
    run_suite = unittest.TestSuite.run
    result = unittest.TestResult()
    try:
        spec = importlib.util.spec_from_file_location('candidate','/work/candidate.py')
        candidate = importlib.util.module_from_spec(spec)
        sys.modules['candidate'] = candidate
        spec.loader.exec_module(candidate)
        spec = importlib.util.spec_from_file_location('trusted_tests','/work/tests.py')
        tests = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tests)
        suite = suite_loader(tests)
        run_suite(suite, result)
        count = result.testsRun
        passed = count > 0 and not result.errors and not result.failures and not result.skipped and not result.unexpectedSuccesses
        report('result', passed=bool(passed), tests_run=count)
    except BaseException as exc:
        report('result', passed=False, tests_run=0)
        # Do not call candidate-controlled exception __str__ or __repr__.
        write(2, b'candidate/test runner raised BaseException\n')
'''


# Shared candidate-free prelude for the separate data-only function adapter.
# Keep one authoritative syscall filter; legacy runner and API are unchanged.
_ISOLATION_PREFIX = _RUNNER.split('# Report readiness only AFTER', 1)[0]


def _copy_file(source: Path, root: Path):
    target = root / str(source).lstrip('/')
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(str(source), str(target))
        target.chmod(0o555 if os.access(str(source), os.X_OK) else 0o444)


def _runtime(root: Path):
    """Copy only trusted system Python + stdlib, not venv/site packages or /etc."""
    python = Path(sys._base_executable).resolve()
    stdlib = Path(sysconfig.get_path('stdlib')).resolve()
    # A user-managed/custom runtime can contain arbitrary startup hooks/data.
    if not str(python).startswith('/usr/') or not str(stdlib).startswith('/usr/'):
        raise RuntimeError('only a trusted /usr system Python runtime is supported')
    _copy_file(python, root)
    target = root / str(stdlib).lstrip('/')
    shutil.copytree(stdlib, target, ignore=shutil.ignore_patterns(
        'site-packages', 'dist-packages', '__pycache__', 'test', 'tests', 'idlelib', 'tkinter', 'ensurepip'),
        symlinks=False)
    # ldd is applied ONLY to trusted runtime files, never submitted artifacts.
    binaries = [python] + list((stdlib / 'lib-dynload').glob('*.so'))
    ldd = shutil.which('ldd')
    if not ldd:
        raise RuntimeError('ldd not found')
    libs = set()
    for binary in binaries:
        proc = subprocess.run([ldd, str(binary)], capture_output=True, text=True, timeout=5,
                              env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'}, close_fds=True)
        if proc.returncode:
            raise RuntimeError('cannot resolve runtime libraries')
        for line in proc.stdout.splitlines():
            if 'not found' in line:
                raise RuntimeError('missing runtime shared library')
            for part in line.split():
                if part.startswith('/'):
                    libs.add(Path(part))
    for lib in libs:
        _copy_file(lib, root)
    return str(python)


def _kill(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=5)


def _launch(root, config, limits):
    read_fd, write_fd = os.pipe()
    config['protocol_fd'] = write_fd
    (root / 'work/config.json').write_text(json.dumps(config), encoding='utf-8')
    selector = selectors.DefaultSelector()
    proc = None
    streams = {'stdout': bytearray(), 'stderr': bytearray(), 'protocol': bytearray()}
    flags = []
    try:
        proc = subprocess.Popen([shutil.which('unshare'), '--user', '--map-root-user',
            '--mount', '--pid', '--net', '--ipc', '--uts', '--fork', '--kill-child=SIGKILL',
            sys._base_executable, '-I', '-B', '-c', _BOOTSTRAP, str(root), str(root / 'work/config.json')],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            pass_fds=(write_fd,), close_fds=True, start_new_session=True,
            env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'})
        os.close(write_fd); write_fd = -1
        for fd, name in [(proc.stdout.fileno(),'stdout'),(proc.stderr.fileno(),'stderr'),(read_fd,'protocol')]:
            os.set_blocking(fd, False)
            selector.register(fd, selectors.EVENT_READ, name)
        deadline = time.monotonic() + limits.wall_seconds
        total = 0
        while selector.get_map():
            if time.monotonic() > deadline:
                flags.append('wall_timeout'); _kill(proc); break
            for key, _ in selector.select(min(0.05, max(0, deadline-time.monotonic()))):
                data = os.read(key.fd, 8192)
                if not data:
                    selector.unregister(key.fd); continue
                name = key.data
                cap = 8192 if name == 'protocol' else limits.output_bytes
                remaining = max(0, cap - (len(streams[name]) if name == 'protocol' else total))
                streams[name].extend(data[:remaining])
                total += len(data) if name != 'protocol' else 0
                if len(data) > remaining or total > limits.output_bytes:
                    flags.append('output_limit'); _kill(proc); break
            if flags: break
        if not flags:
            try: proc.wait(timeout=max(0.01, deadline-time.monotonic()))
            except subprocess.TimeoutExpired: flags.append('wall_timeout'); _kill(proc)
        return proc.returncode, streams, tuple(flags)
    finally:
        if proc is not None and proc.poll() is None: _kill(proc)
        if proc is not None:
            proc.stdout.close(); proc.stderr.close()
        selector.close()
        os.close(read_fd)
        if write_fd >= 0: os.close(write_fd)


def _evaluate(source, trusted_tests, limits, probe=False):
    limits.validate()
    if platform.system() != 'Linux':
        return SandboxResult('BLOCKED', False, reason='Linux required; Windows/macOS explicitly unsupported')
    if platform.machine() not in ('x86_64', 'aarch64') or not shutil.which('unshare'):
        return SandboxResult('BLOCKED', False, reason='supported architecture and unshare required')
    for name, value in [('source',source),('trusted_tests',trusted_tests)]:
        if not isinstance(value, str) or len(value.encode('utf-8')) > limits.source_bytes:
            raise ValueError(name + ' must be a bounded Python source string')
    try:
        with tempfile.TemporaryDirectory(prefix='silt-code-') as staging:
            base = Path(staging)
            root = base / 'root'; root.mkdir()
            (root / 'work').mkdir(); (root / 'tmp').mkdir()
            canary = base / 'outside-canary'; canary.write_text('non-sensitive scratch marker')
            python = _runtime(root)
            (root / 'work/runner.py').write_text(_RUNNER, encoding='utf-8')
            namespaces = {k: os.readlink('/proc/self/ns/' + k) for k in ('user','mnt','pid','net','ipc','uts')}
            config = dict(python=python, namespaces=namespaces, nonce=secrets.token_hex(32),
                          limits=dataclasses.asdict(limits), probe=True, canary=str(canary))
            # Run full kernel/runtime selftest in a fresh namespace BEFORE even
            # placing the candidate into a root. Never cache successful probes.
            rc, streams, flags = _launch(root, config, limits)
            records = _records(streams['protocol'], config['nonce'])
            if rc != 0 or flags or records != [('ready',None,None), ('result',True,1)]:
                return SandboxResult('BLOCKED', False, stderr=streams['stderr'].decode('utf-8','replace'),
                                     reason='namespace/runtime selftest failed; candidate was not executed',
                                     returncode=rc, resource_flags=flags)
            if probe:
                return SandboxResult('PASSED', True, True, reason='namespace/runtime selftest passed', tests_run=1)
            (root / 'work/candidate.py').write_text(source, encoding='utf-8')
            (root / 'work/tests.py').write_text(trusted_tests, encoding='utf-8')
            config.update(probe=False, nonce=secrets.token_hex(32))
            rc, streams, flags = _launch(root, config, limits)
            records = _records(streams['protocol'], config['nonce'])
            # Malformed later protocol bytes must not erase the observation that
            # setup completed before candidate execution.
            first_line = bytes(streams['protocol']).split(b'\n', 1)[0]
            ready = _records(first_line, config['nonce']) == [('ready',None,None)]
            valid = (len(records) == 2 and ready and records[1][0] == 'result'
                     and type(records[1][1]) is bool and type(records[1][2]) is int
                     and records[1][2] >= 0)
            passed = valid and records[1][1] and records[1][2] > 0 and rc == 0 and not flags
            status = 'PASSED' if passed else 'FAILED'
            if not ready: status = 'BLOCKED'
            elif 'wall_timeout' in flags: status = 'TIMEOUT'
            elif flags or rc in (-signal.SIGKILL, -signal.SIGXCPU, -signal.SIGXFSZ):
                status = 'RESOURCE_LIMIT'
                if not flags: flags = ('resource_signal',)
            return SandboxResult(status, ready, bool(passed),
                streams['stdout'].decode('utf-8','replace'), streams['stderr'].decode('utf-8','replace'),
                '' if passed else 'tests failed, execution incomplete, or resource/isolation limit',
                rc, records[1][2] if valid else 0, flags)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return SandboxResult('BLOCKED', False, reason='sandbox setup failed: ' + str(exc))


def _records(data, nonce):
    try:
        values = [json.loads(line) for line in data.decode('utf-8').splitlines()]
        if not all(isinstance(v, dict) and v.get('nonce') == nonce for v in values): return []
        return [(v.get('event'), v.get('passed'), v.get('tests_run')) for v in values]
    except (ValueError, UnicodeError):
        return []


def probe_code_sandbox(limits: Optional[SandboxLimits] = None) -> SandboxResult:
    """Exercise namespaces, RO root, tmpfs, capability drop and syscall denials."""
    return _evaluate('', '', limits or SandboxLimits(), probe=True)


def evaluate_code(source: str, trusted_tests: str,
                  limits: Optional[SandboxLimits] = None) -> SandboxResult:
    """Execute candidate.py and independent unittest source importing candidate.

    BLOCKED means no permission to count a functional score. PASSED is a test
    observation, not an adversarial-correctness proof or model admission.
    """
    return _evaluate(source, trusted_tests, limits or SandboxLimits())
