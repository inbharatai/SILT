"""Real OS containment tests; never read actual host secrets or run fork bombs."""
import errno

import pytest

from asea.certification import sandbox
from asea.certification.sandbox import SandboxLimits, evaluate_code, probe_code_sandbox


@pytest.fixture(scope='module')
def enabled():
    result = probe_code_sandbox()
    if not result.supported:
        pytest.skip('kernel sandbox unavailable: ' + result.reason + ' ' + result.stderr)
    assert result.passed


def _tests_for(body):
    return 'import unittest\nimport candidate\nclass Check(unittest.TestCase):\n    def test_behavior(self):\n' + ''.join(
        '        ' + line + '\n' for line in body.splitlines())


def test_probe(enabled):
    assert probe_code_sandbox().passed


def test_functional_pass_math_re(enabled):
    result = evaluate_code('import math, re\ndef solve(s): return math.isqrt(int(re.sub(r"\\D", "", s)))',
                           _tests_for('self.assertEqual(candidate.solve("x81"), 9)'))
    assert result.status == 'PASSED', result.to_dict()
    assert result.passed and result.supported and result.tests_run == 1


def test_functional_failure(enabled):
    result = evaluate_code('def solve(): return 3', _tests_for('self.assertEqual(candidate.solve(), 4)'))
    assert result.status == 'FAILED' and not result.passed and result.tests_run == 1


@pytest.mark.parametrize('source', ['print("pass")', 'import os; os._exit(0)',
                                  'raise SystemExit(0)', 'raise KeyboardInterrupt()'])
def test_prints_or_early_exit_not_pass(enabled, source):
    result = evaluate_code(source, _tests_for('self.fail("must not pass")'))
    assert not result.passed and result.status == 'FAILED', result.to_dict()


def test_empty_and_skipped_suite_not_pass(enabled):
    assert not evaluate_code('', '').passed
    result = evaluate_code('', 'import unittest\n@unittest.skip("skip")\nclass Check(unittest.TestCase):\n def test_x(self): pass\n')
    assert not result.passed


def test_outside_canary_and_environment_inaccessible(enabled, tmp_path, monkeypatch):
    marker = tmp_path / 'outside-marker'
    marker.write_text('scratch-only-canary')
    monkeypatch.setenv('SILT_SCRATCH_SECRET', 'canary-not-real-secret')
    source = '''import os
def read(path):
    try:
        with open(path) as f: return f.read()
    except OSError: return 'denied'
def env(): return os.environ.get('SILT_SCRATCH_SECRET')
'''
    result = evaluate_code(source, _tests_for(
        'self.assertEqual(candidate.read(' + repr(str(marker)) + '), "denied")\n'
        'self.assertEqual(candidate.read("/proc/1/root' + str(marker) + '"), "denied")\n'
        'self.assertIsNone(candidate.env())'))
    assert result.passed, result.to_dict()
    assert marker.read_text() == 'scratch-only-canary'


def test_network_and_single_spawn_denied(enabled):
    # Exactly one fork attempt; never launch a bomb, connect to a real service,
    # or depend on a network response to decide whether isolation works.
    source = '''import os, socket, errno
def network():
    try:
        s = socket.socket(); s.connect(('127.0.0.1', 9))
    except OSError as e: return e.errno
def spawn():
    try: return os.fork()
    except OSError as e: return -e.errno
'''
    result = evaluate_code(source, _tests_for(
        'self.assertEqual(candidate.network(), ' + str(errno.EPERM) + ')\n'
        'self.assertEqual(candidate.spawn(), -' + str(errno.EPERM) + ')'))
    assert result.passed, result.to_dict()


def test_infinite_loop_timeout(enabled):
    limits = SandboxLimits(wall_seconds=0.7, cpu_seconds=2)
    result = evaluate_code('while True: pass', _tests_for('self.assertTrue(True)'), limits)
    assert result.status == 'TIMEOUT', result.to_dict()
    assert not result.passed and 'wall_timeout' in result.resource_flags


def test_disk_file_and_total_tmp_bounded(enabled):
    source = '''import errno, os
def fill():
    count = 0
    for i in range(128):
        try:
            with open('/tmp/x%d' % i, 'wb') as f: f.write(b'x' * 65536)
            count += 1
        except OSError as e: return count, e.errno
    return count, None
def large():
    try:
        with open('/tmp/large', 'wb') as f: f.write(b'x' * 524288)
    except OSError as e: return e.errno
'''
    limits = SandboxLimits(tmp_bytes=1024*1024, file_bytes=128*1024)
    result = evaluate_code(source, _tests_for(
        'self.assertEqual(candidate.large(), ' + str(errno.EFBIG) + ')\n'
        'count, error = candidate.fill()\n'
        'self.assertLess(count, 20)\n'
        'self.assertEqual(error, ' + str(errno.ENOSPC) + ')'), limits)
    assert result.passed, result.to_dict()


def test_stdout_flood_is_bounded(enabled):
    limits = SandboxLimits(output_bytes=4096)
    result = evaluate_code('import os\nwhile True: os.write(1, b"x"*4096)',
                           _tests_for('self.assertTrue(True)'), limits)
    assert result.status == 'RESOURCE_LIMIT', result.to_dict()
    assert 'output_limit' in result.resource_flags
    assert len(result.stdout.encode()) <= limits.output_bytes


def test_memory_limit(enabled):
    result = evaluate_code('x = bytearray(512 * 1024 * 1024)', _tests_for('self.assertTrue(True)'),
                           SandboxLimits(memory_bytes=128*1024*1024))
    assert not result.passed, result.to_dict()


def test_root_readonly_and_no_host_fds(enabled):
    source = '''import os, errno
def root_write():
    try: open('/work/tests.py', 'w')
    except OSError as e: return e.errno
def dirs(): return os.path.exists('/proc'), os.path.exists('/dev'), os.path.exists('/home')
'''
    result = evaluate_code(source, _tests_for(
        'self.assertEqual(candidate.root_write(), ' + str(errno.EROFS) + ')\n'
        'self.assertEqual(candidate.dirs(), (False,False,False))'))
    assert result.passed, result.to_dict()


def test_windows_explicitly_blocked(monkeypatch):
    monkeypatch.setattr(sandbox.platform, 'system', lambda: 'Windows')
    result = evaluate_code('raise AssertionError("never execute")', '')
    assert result.status == 'BLOCKED' and not result.supported
    assert 'Windows' in result.reason


def test_failed_selftest_blocks_before_candidate(monkeypatch):
    calls = []
    def failed_launch(root, config, limits):
        calls.append(config['probe'])
        assert not (root / 'work/candidate.py').exists()
        return 1, {'stdout': bytearray(), 'stderr': bytearray(b'probe refused'), 'protocol': bytearray()}, ()
    monkeypatch.setattr(sandbox, '_launch', failed_launch)
    result = evaluate_code('raise AssertionError("never execute")', '')
    assert result.status == 'BLOCKED' and calls == [True]


def test_invalid_limits_and_input():
    with pytest.raises(ValueError): evaluate_code('', '', SandboxLimits(wall_seconds=float('nan')))
    with pytest.raises(ValueError): evaluate_code('', '', SandboxLimits(memory_bytes=4*1024**3))
    with pytest.raises(ValueError): evaluate_code('x' * (1024**2+1), '')


def test_inheritable_host_descriptor_closed(enabled, tmp_path):
    import fcntl
    import os
    marker = tmp_path / 'fd-canary'
    marker.write_text('scratch-only-descriptor-canary')
    with marker.open('rb') as handle:
        fd = fcntl.fcntl(handle.fileno(), fcntl.F_DUPFD, 80)
        os.set_inheritable(fd, True)
        try:
            source = 'import os\ndef read_fd():\n try: return os.read(' + str(fd) + ', 64)\n except OSError as e: return e.errno'
            result = evaluate_code(source, _tests_for('self.assertEqual(candidate.read_fd(), ' + str(errno.EBADF) + ')'))
            assert result.passed, result.to_dict()
        finally:
            os.close(fd)


def test_cannot_escape_timeout_group(enabled):
    source = '''import os, errno, ctypes
# Removing PDEATHSIG must not let code evade the parent's process-group kill.
ctypes.CDLL(None).prctl(1, 0, 0, 0, 0)
for change_group in [os.setsid, lambda: os.setpgid(0, 0)]:
    try: change_group()
    except OSError as e:
        assert e.errno == errno.EPERM
    else: raise AssertionError('escaped process group')
while True: pass
'''
    result = evaluate_code(source, _tests_for('self.assertTrue(True)'),
                           SandboxLimits(wall_seconds=0.7, cpu_seconds=2))
    assert result.status == 'TIMEOUT', result.to_dict()


def test_bad_protocol_rejected():
    assert sandbox._records(b'{"nonce":"wrong","event":"result","passed":true}', 'secret') == []
    assert sandbox._records(b'pass', 'secret') == []
