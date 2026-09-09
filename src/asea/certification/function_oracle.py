"""Bounded data-only function oracle; the definitive comparator stays on the host.

Not a proof of purity, function identity, algorithm correctness, or model admission.
See docs/FUNCTION_ORACLE.md. No candidate source is evaluated in this interpreter.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import keyword
import os
from pathlib import Path
import secrets
import selectors
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time

from . import sandbox
from .sandbox import SandboxLimits

FRAME_BYTES = 64 * 1024
TOTAL_BYTES = 1024 * 1024
MAX_CASES = 128
MAX_DEPTH = 16
MAX_NODES = 4096
STRING_BYTES = 16 * 1024
SETUP_RECEIPT = b'SILT-FUNCTION-ISOLATED-v1\n'


class ProtocolError(ValueError):
    """Invalid or over-budget data; never a candidate verdict."""


def _validate_value(value, *, byte_cap=None):
    """Validate exact builtins and count compact UTF-8 JSON before serialization.

    The count includes JSON escapes, punctuation and encoded bytes, not Unicode
    character count. Only bounded individual strings are encoded while counting;
    no aggregate JSON string/body is allocated until the frame fits its budget.
    """
    count = size = 0

    def add(n):
        nonlocal size
        size += n
        if byte_cap is not None and size > byte_cap:
            raise ProtocolError('JSON frame budget exceeded')

    def walk(v, depth):
        nonlocal count
        count += 1
        if count > MAX_NODES or depth > MAX_DEPTH:
            raise ProtocolError('JSON node/depth budget exceeded')
        t = type(v)
        if t is str:
            if len(v) > STRING_BYTES:
                raise ProtocolError('JSON string budget exceeded')
            encoded_bytes = len(v.encode('utf-8', 'strict'))
            if encoded_bytes > STRING_BYTES:
                raise ProtocolError('JSON string budget exceeded')
            add(2 + encoded_bytes + sum(
                1 if ord(c) in (34, 92, 8, 12, 10, 13, 9)
                else 5 if ord(c) < 32 else 0 for c in v))
        elif t is int:
            if not -(2**63) <= v < 2**63:
                raise ProtocolError('JSON integer outside signed 64-bit range')
            add(len(str(v)))
        elif v is None or t is bool:
            add(4 if v is None or v is True else 5)
        elif t is list:
            if len(v) > MAX_NODES:
                raise ProtocolError('JSON node budget exceeded')
            add(2 + max(0, len(v) - 1))
            for item in v:
                walk(item, depth + 1)
        elif t is dict:
            if len(v) > MAX_NODES:
                raise ProtocolError('JSON node budget exceeded')
            add(2 + max(0, len(v) - 1) + len(v))
            for k, item in v.items():
                if type(k) is not str:
                    raise ProtocolError('JSON object keys must be exact strings')
                walk(k, depth + 1)
                walk(item, depth + 1)
        else:
            raise ProtocolError('only null, bool, int, string, list and dict are supported')

    try:
        walk(value, 0)
    except UnicodeError as exc:
        raise ProtocolError('JSON strings must be Unicode scalars') from exc
    return size


def _canonical(value):
    size = _validate_value(value, byte_cap=FRAME_BYTES)
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True,
                     allow_nan=False).encode('utf-8', 'strict')
    if len(raw) != size or len(raw) > FRAME_BYTES:
        raise ProtocolError('JSON frame byte accounting mismatch')
    return raw


def _decode(raw, *, byte_cap=FRAME_BYTES, depth_cap=MAX_DEPTH, node_cap=MAX_NODES):
    if len(raw) > byte_cap:
        raise ProtocolError('JSON byte budget exceeded')
    # Lexical bound BEFORE json.loads: depth, token/node count, and numeric token
    # bounds prevent the decoder doing unbounded work. Strings respect escaping.
    try:
        text = raw.decode('utf-8', 'strict')
        depth = 0
        quoted = escaped = False
        tokens = 0
        in_token = False
        for c in text:
            if quoted:
                if escaped:
                    escaped = False
                elif c == '\\':
                    escaped = True
                elif c == '"':
                    quoted = False
                continue
            if c == '"':
                quoted = True
                tokens += 1
                in_token = False
            elif c in '[{':
                depth += 1
                tokens += 1
                in_token = False
                if depth > depth_cap:
                    raise ProtocolError('JSON lexical depth exceeded')
            elif c in ']}':
                depth -= 1
                in_token = False
            elif c in ',: \t\r\n':
                in_token = False
            elif not in_token:
                tokens += 1
                in_token = True
            if tokens > node_cap:
                raise ProtocolError('JSON lexical node budget exceeded')

        def integer(s):
            if len(s) > 20:
                raise ProtocolError('JSON integer token too long')
            value = int(s)
            if not -(2**63) <= value < 2**63:
                raise ProtocolError('JSON integer outside signed 64-bit range')
            return value

        def reject_number(_):
            raise ProtocolError('floats and nonfinite numbers are not supported')

        def pairs(items):
            obj = {}
            for key, value in items:
                if key in obj:
                    raise ProtocolError('duplicate JSON key')
                obj[key] = value
            return obj

        value = json.loads(text, parse_int=integer, parse_float=reject_number,
                           parse_constant=reject_number, object_pairs_hook=pairs)
        if byte_cap == FRAME_BYTES:
            _validate_value(value)
        return value
    except (UnicodeError, RecursionError, ValueError) as exc:
        raise ProtocolError('invalid bounded JSON: ' + str(exc)) from exc


def _equal(left, right):
    """Exact type equality (True never equals 1), only after builtin validation."""
    if type(left) is not type(right):
        return False
    if type(left) is list:
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    if type(left) is dict:
        return left.keys() == right.keys() and all(_equal(left[k], right[k]) for k in left)
    return left == right


def _suite(source, cases, limits):
    limits.validate()
    if (type(source) is not str or len(source) > limits.source_bytes
            or len(source.encode('utf-8', 'strict')) > limits.source_bytes):
        raise ValueError('source must be a bounded exact string')
    if type(cases) is not list or not 1 <= len(cases) <= MAX_CASES:
        raise ValueError('cases must be a nonempty list of at most 128 cases')
    seen, copied, inputs = set(), [], []
    suite_bytes = 1
    request_bytes = response_bytes = 0
    for case in cases:
        if type(case) is not dict:
            raise ValueError('each case must be an exact dict')
        _validate_value(case)
        if set(case) != {'id', 'function', 'args', 'kwargs', 'expected'}:
            raise ValueError('each case requires exactly id,function,args,kwargs,expected')
        ident = case['id']
        if type(ident) not in (str, int) or (type(ident) is str and not ident):
            raise ValueError('case id must be a nonempty string or signed 64-bit integer')
        key = (type(ident), ident)
        if key in seen:
            raise ValueError('duplicate case id')
        seen.add(key)
        name = case['function']
        if type(name) is not str or not name.isascii() or not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError('function must be one ASCII identifier, not an expression or path')
        if type(case['args']) is not list or type(case['kwargs']) is not dict:
            raise ValueError('args and kwargs must be exact list and dict')
        # Snapshot all data, so concurrent caller mutation cannot change comparisons.
        raw = _canonical(case)
        suite_bytes += len(raw) + 1
        if suite_bytes > TOTAL_BYTES:
            raise ValueError('suite data exceeds byte budget')
        frozen = _decode(raw)
        public = {k: v for k, v in frozen.items() if k != 'expected'}
        # Validate every eventual request envelope before any runtime launch.
        request_bytes += len(_frame({'jsonrpc': '2.0', 'id': '0' * 32, 'method': 'call',
                                    'params': {k: frozen[k] for k in ('function', 'args', 'kwargs')}}))
        if request_bytes > TOTAL_BYTES:
            raise ValueError('total request byte budget exceeded')
        # Host-only feasibility check, including the actual ID width and framing.
        # Neither this envelope nor any expected value is staged or sent.
        response_bytes += len(_frame({'jsonrpc': '2.0', 'id': '0' * 32,
                                      'result': frozen['expected']}))
        if response_bytes > TOTAL_BYTES:
            raise ValueError('total response byte budget exceeded')
        copied.append(frozen)
        inputs.append(public)
    return copied, inputs


# Before candidate import this is fixed trusted code. Afterwards all its output,
# globals and entry points may be forged: host correctness never relies on them.
_ADAPTER = r'''
import struct
setup_fd = config['setup_fd']
request_fd = config['request_fd']
response_fd = config['response_fd']
os.write(setup_fd, b'SILT-FUNCTION-ISOLATED-v1\n')
os.close(setup_fd)
# Receipt writer has no duplicates; there is no /proc, /dev or host directory FD.
# The first host request is also the execution gate, sent after receipt AND EOF.
def read_exact(n):
    out = bytearray()
    while len(out) < n:
        data = os.read(request_fd, n-len(out))
        if not data:
            if not out: return None
            raise RuntimeError('truncated request')
        out.extend(data)
    return bytes(out)
def read_request():
    header = read_exact(4)
    if header is None: return None
    size = struct.unpack('!I', header)[0]
    if size > 65536: raise RuntimeError('request too large')
    return json.loads(read_exact(size))
def answer(value):
    raw = _canonical(value)
    data = struct.pack('!I', len(raw)) + raw
    while data:
        n = os.write(response_fd, data)
        data = data[n:]
# Exact identity only: never inspect arbitrary exception attributes or call str/repr.
_error_types = ((ValueError, 'ValueError'), (TypeError, 'TypeError'),
    (RuntimeError, 'RuntimeError'), (KeyError, 'KeyError'),
    (IndexError, 'IndexError'), (ZeroDivisionError, 'ZeroDivisionError'),
    (ImportError, 'ImportError'), (ModuleNotFoundError, 'ModuleNotFoundError'),
    (AttributeError, 'AttributeError'), (NameError, 'NameError'),
    (SyntaxError, 'SyntaxError'), (MemoryError, 'MemoryError'),
    (RecursionError, 'RecursionError'), (AssertionError, 'AssertionError'),
    (OSError, 'OSError'), (SystemExit, 'SystemExit'))
_exact_type = type
request = read_request()
phase = 'import'
if request is not None:
    try:
        spec = importlib.util.spec_from_file_location('candidate', '/work/candidate.py')
        candidate = importlib.util.module_from_spec(spec)
        sys.modules['candidate'] = candidate
        spec.loader.exec_module(candidate)
        while request is not None:
            phase = 'call'
            params = request['params']
            result = getattr(candidate, params['function'])(*params['args'], **params['kwargs'])
            phase = 'serialize'
            answer({'jsonrpc': '2.0', 'id': request['id'], 'result': result})
            request = read_request()
    except BaseException as error:
        # Optional diagnostic, not a verdict. Mutated adapter/forged reports remain
        # untrusted; the host accepts only this closed metadata schema.
        code = 'opaque'
        for cls, label in _error_types:
            if _exact_type(error) is cls:
                code = label
                break
        try:
            answer({'jsonrpc': '2.0', 'id': request['id'], 'error': {
                'code': -32000, 'message': 'candidate error', 'data': {
                    'schema_version': 1, 'phase': phase, 'type_code': code}}})
        except BaseException:
            pass
        os._exit(1)
'''


# util-linux unshare --fork keeps inherited pipe writers open in its monitor.
# That prevents setup EOF and request EOF. Use its exact namespace flags without
# --fork, then this candidate-free monitor forks into the pending PID namespace
# and closes ALL transport descriptors in the parent before waiting.
_PID_MONITOR = r'''
import ctypes, json, os, signal, sys
bootstrap, root, config_path = sys.argv[1:]
with open(config_path) as handle: c = json.load(handle)
pid = os.fork()
if pid == 0:
    lib = ctypes.CDLL(None, use_errno=True)
    if lib.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        os._exit(125)
    os.execv(sys.executable, [sys.executable, '-I', '-B', '-c', bootstrap, root, config_path])
for key in ('setup_fd', 'request_fd', 'response_fd'):
    os.close(c[key])
_, status = os.waitpid(pid, 0)
if os.WIFSIGNALED(status):
    sig = os.WTERMSIG(status)
    if sig not in (signal.SIGKILL, signal.SIGSTOP):
        signal.signal(sig, signal.SIG_DFL)
    os.kill(os.getpid(), sig)
os._exit(os.WEXITSTATUS(status))
'''


def _adapter_source():
    # Share ONLY the public data codec, not the host module, suite, comparator or
    # expected values. These helpers become untrusted after candidate import;
    # the host independently parses, validates and compares every response.
    constants = '\n'.join('%s = %r' % (name, value) for name, value in (
        ('FRAME_BYTES', FRAME_BYTES), ('MAX_DEPTH', MAX_DEPTH),
        ('MAX_NODES', MAX_NODES), ('STRING_BYTES', STRING_BYTES)))
    codec = '\n'.join(inspect.getsource(fn) for fn in
                      (ProtocolError, _validate_value, _canonical))
    return constants + '\n' + codec + '\n' + _ADAPTER


def _frame(request):
    raw = _canonical(request)
    return struct.pack('!I', len(raw)) + raw


ERROR_TYPES = frozenset(('opaque', 'ValueError', 'TypeError', 'RuntimeError',
    'KeyError', 'IndexError', 'ZeroDivisionError', 'ImportError', 'ModuleNotFoundError',
    'AttributeError', 'NameError', 'SyntaxError', 'MemoryError', 'RecursionError',
    'AssertionError', 'OSError', 'SystemExit'))


class CandidateError(ProtocolError):
    """Host-validated shape, UNTRUSTED candidate claims; always fails execution."""
    def __init__(self, metadata):
        super().__init__('candidate reported an error (untrusted metadata)')
        self.metadata = {'trust': 'untrusted_candidate', **metadata}


def _response(raw, outstanding):
    obj = _decode(raw)
    if (type(obj) is not dict or set(obj) not in (
            {'jsonrpc', 'id', 'result'}, {'jsonrpc', 'id', 'error'})
            or obj['jsonrpc'] != '2.0' or type(obj['id']) is not str
            or outstanding is None or obj['id'] != outstanding):
        raise ProtocolError('invalid, duplicate, stale or unissued response envelope')
    if 'error' in obj:
        error = obj['error']
        if (type(error) is not dict or set(error) != {'code', 'message', 'data'}
                or type(error['code']) is not int or error['code'] != -32000
                or error['message'] != 'candidate error'):
            raise ProtocolError('invalid candidate error envelope')
        data = error['data']
        if (type(data) is not dict or set(data) != {'schema_version', 'phase', 'type_code'}
                or type(data['schema_version']) is not int or data['schema_version'] != 1
                or type(data['phase']) is not str or data['phase'] not in ('import', 'call', 'serialize')
                or type(data['type_code']) is not str or data['type_code'] not in ERROR_TYPES):
            raise ProtocolError('invalid candidate error metadata')
        raise CandidateError(data)
    return obj['result']


def _trace_policy(policy):
    if policy is None:
        policy = {'schema_version': 1, 'return_retention': 'none'}
    if (type(policy) is not dict or set(policy) != {'schema_version', 'return_retention'}
            or type(policy['schema_version']) is not int or policy['schema_version'] != 1
            or type(policy['return_retention']) is not str
            or policy['return_retention'] not in ('none', 'digest', 'value')):
        raise ValueError('trace_policy requires schema_version=1 and return_retention=none|digest|value')
    return dict(policy)


def _return_trace(case_id, answer, policy):
    # Only called AFTER full host frame validation. No truncation or candidate
    # counters. This does not serialize any expected answer, args or kwargs.
    retention = policy['return_retention']
    entry = {'id': case_id, 'phase': 'response', 'retention': retention,
             'reason': 'policy_none' if retention == 'none' else 'retained'}
    if retention != 'none':
        entry.update(value_type={type(None): 'null', bool: 'bool', int: 'int',
            str: 'string', list: 'list', dict: 'dict'}[type(answer)],
            sha256=_digest(_canonical(answer)))
        if retention == 'value':
            entry['value'] = answer
    return entry


def replay_returns(trace, cases):
    """Recompare retained host-validated values; NOT an execution/admission verdict.

    Caller supplies expectations independently. Stored hashes detect accidental
    edits, not authenticity: the caller must authenticate its trusted local store.
    """
    if (type(trace) is not dict or set(trace) != {'schema_version', 'policy', 'returns'}
            or type(trace['schema_version']) is not int or trace['schema_version'] != 1
            or type(trace['policy']) is not dict
            or type(trace['returns']) is not list or len(trace['returns']) > MAX_CASES):
        raise ValueError('invalid return trace')
    policy = _trace_policy(trace['policy'])
    frozen, _ = _suite('', cases, SandboxLimits())
    if policy['return_retention'] != 'value':
        return {'schema_version': 1, 'available': False, 'reason': 'values_not_retained', 'cases': []}
    if len(trace['returns']) > len(frozen):
        raise ValueError('too many retained returns')
    completed, total = [], 0
    for entry, case in zip(trace['returns'], frozen):
        if (type(entry) is not dict or set(entry) != {
                'id', 'phase', 'retention', 'reason', 'value_type', 'sha256', 'value'}):
            raise ValueError('invalid retained value entry')
        _validate_value({k: v for k, v in entry.items() if k != 'value'})
        raw = _frame({'jsonrpc': '2.0', 'id': '0' * 32, 'result': entry['value']})
        total += len(raw)
        if total > TOTAL_BYTES:
            raise ValueError('total retained response byte budget exceeded')
        validated = _response(raw[4:], '0' * 32)
        expected_entry = _return_trace(case['id'], validated, policy)
        if not _equal(entry, expected_entry):
            raise ValueError('trace identity, type or digest mismatch')
        completed.append({'id': case['id'], 'matched': _equal(validated, case['expected'])})
    return {'schema_version': 1, 'available': len(completed) == len(frozen),
            'reason': 'replayed' if len(completed) == len(frozen) else 'incomplete_trace',
            'cases': completed}


def _termination(proc, deadline, reason):
    """Observe before cleanup; grace/reap never extends the evaluation deadline."""
    observed = proc.poll() if proc is not None else None
    evidence = {'schema_version': 1, 'observed_exit': observed,
        'observed_exit_state': 'known' if observed is not None else 'unknown',
        'exit_before_signal': observed, 'cleanup_action': 'none', 'cleanup_signal': None,
        'cleanup_reason': reason or 'execution_finished', 'grace_seconds': 0,
        'final_exit': observed, 'final_exit_state': 'known' if observed is not None else 'unknown',
        'oom_proven': False}
    if proc is not None and observed is None:
        grace = min(.05, max(0, deadline - time.monotonic()))
        evidence['grace_seconds'] = grace
        if grace:
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
        before_signal = proc.poll()
        evidence['exit_before_signal'] = before_signal
        if before_signal is None:
            evidence.update(cleanup_action='kill_process_group', cleanup_signal='SIGKILL')
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                evidence['cleanup_action'] = 'process_group_already_gone'
            try:
                proc.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        evidence['final_exit'] = proc.poll()
        evidence['final_exit_state'] = 'known' if evidence['final_exit'] is not None else 'unknown'
    return evidence


def _exchange(root, config, cases, limits, trace_policy=None):
    """One bounded, nonblocking host state machine; never trusts a verdict field."""
    setup_r, setup_w = os.pipe()
    req_r, req_w = os.pipe()
    resp_r, resp_w = os.pipe()
    owned = {setup_r, setup_w, req_r, req_w, resp_r, resp_w}
    selector = selectors.DefaultSelector()
    proc = None
    streams = {'stdout': bytearray(), 'stderr': bytearray()}
    receipt, buffer = bytearray(), bytearray()
    completed, flags = [], []
    ready = False
    pending = b''
    outstanding = None
    sent_total = response_total = output_total = 0
    reason = ''
    policy = _trace_policy(trace_policy)
    returns = []
    diagnostic = {'schema_version': 1, 'host_phase': 'setup', 'event': 'not_started',
                  'requested_case_id': None, 'candidate_error': None}
    deadline = time.monotonic() + limits.wall_seconds

    def close(fd):
        if fd in owned:
            try:
                selector.unregister(fd)
            except KeyError:
                pass
            os.close(fd)
            owned.remove(fd)

    def queue():
        nonlocal pending, outstanding, sent_total
        case = cases[len(completed)]
        diagnostic.update(host_phase='request', event='request_queued', requested_case_id=case['id'])
        # Generated ONLY now. No future ID or hidden suite state enters the root.
        outstanding = secrets.token_hex(16)
        pending = _frame({'jsonrpc': '2.0', 'id': outstanding, 'method': 'call',
                          'params': {'function': case['function'], 'args': case['args'],
                                     'kwargs': case['kwargs']}})
        sent_total += len(pending)
        if sent_total > TOTAL_BYTES:
            raise ProtocolError('total request byte budget exceeded')
        selector.register(req_w, selectors.EVENT_WRITE, 'request')

    try:
        config.update(setup_fd=setup_w, request_fd=req_r, response_fd=resp_w)
        (root / 'work/config.json').write_text(json.dumps(config), encoding='utf-8')
        proc = subprocess.Popen([shutil.which('unshare'), '--user', '--map-root-user',
            '--mount', '--pid', '--net', '--ipc', '--uts',
            sys._base_executable, '-I', '-B', '-c', _PID_MONITOR, sandbox._BOOTSTRAP,
            str(root), str(root / 'work/config.json')],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            pass_fds=(setup_w, req_r, resp_w), close_fds=True, start_new_session=True,
            env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'})
        for fd in (setup_w, req_r, resp_w):
            close(fd)
        for fd, name in ((setup_r, 'setup'), (resp_r, 'response'),
                         (proc.stdout.fileno(), 'stdout'), (proc.stderr.fileno(), 'stderr')):
            os.set_blocking(fd, False)
            selector.register(fd, selectors.EVENT_READ, name)
        os.set_blocking(req_w, False)
        deadline = time.monotonic() + limits.wall_seconds
        while selector.get_map():
            if time.monotonic() >= deadline:
                flags.append('wall_timeout')
                reason = 'wall deadline exceeded'
                break
            for key, _ in selector.select(min(.05, max(0, deadline - time.monotonic()))):
                name = key.data
                if name == 'request':
                    try:
                        n = os.write(key.fd, pending)
                    except BlockingIOError:
                        continue
                    pending = pending[n:]
                    if not pending:
                        selector.unregister(key.fd)
                        diagnostic.update(host_phase='response', event='awaiting_response')
                    continue
                try:
                    data = os.read(key.fd, 8192)
                except BlockingIOError:
                    continue
                if not data:
                    selector.unregister(key.fd)
                    if name == 'setup':
                        diagnostic.update(host_phase='setup', event='setup_eof')
                        if bytes(receipt) != SETUP_RECEIPT:
                            raise ProtocolError('isolation setup receipt missing or invalid')
                        ready = True
                        queue()
                    elif name == 'response' and (buffer or len(completed) != len(cases)):
                        diagnostic.update(event='response_eof' if ready else 'response_eof_before_setup')
                        raise ProtocolError('incomplete response stream')
                    continue
                if name == 'setup':
                    receipt.extend(data[:len(SETUP_RECEIPT) + 1])
                    if len(receipt) > len(SETUP_RECEIPT):
                        raise ProtocolError('oversized setup receipt')
                elif name in streams:
                    remaining = max(0, limits.output_bytes - output_total)
                    streams[name].extend(data[:remaining])
                    output_total += len(data)
                    if output_total > limits.output_bytes:
                        flags.append('output_limit')
                        raise ProtocolError('combined diagnostic output limit exceeded')
                else:
                    response_total += len(data)
                    if response_total > TOTAL_BYTES:
                        flags.append('protocol_limit')
                        raise ProtocolError('total response byte budget exceeded')
                    if not ready or outstanding is None or pending:
                        raise ProtocolError('response before a complete issued request')
                    buffer.extend(data)
                    if len(buffer) >= 4:
                        size = struct.unpack('!I', buffer[:4])[0]
                        if size > FRAME_BYTES:
                            flags.append('protocol_limit')
                            raise ProtocolError('advertised response frame too large')
                        if len(buffer) >= size + 4:
                            answer = _response(bytes(buffer[4:4 + size]), outstanding)
                            if len(buffer) != size + 4:
                                raise ProtocolError('unsolicited trailing response bytes')
                            buffer.clear()
                            case = cases[len(completed)]
                            completed.append({'id': case['id'], 'matched': _equal(answer, case['expected'])})
                            returns.append(_return_trace(case['id'], answer, policy))
                            diagnostic.update(host_phase='response', event='return_validated')
                            outstanding = None
                            if len(completed) == len(cases):
                                close(req_w)
                            else:
                                queue()
        if not flags:
            try:
                proc.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                flags.append('wall_timeout')
                reason = 'wall deadline exceeded'
    except (ProtocolError, OSError, subprocess.SubprocessError) as exc:
        reason = str(exc)
        if isinstance(exc, CandidateError):
            diagnostic.update(event='candidate_error', candidate_error=exc.metadata)
        elif diagnostic['event'] not in ('setup_eof', 'response_eof', 'response_eof_before_setup'):
            diagnostic['event'] = 'transport_error'
    finally:
        if 'wall_timeout' in flags:
            diagnostic['event'] = 'wall_timeout'
        termination = _termination(proc, deadline, reason)
        if proc is not None:
            proc.stdout.close()
            proc.stderr.close()
        selector.close()
        for fd in owned:
            os.close(fd)
    return {'ready': ready, 'completed': completed, 'flags': flags, 'reason': reason,
            'returncode': termination['final_exit'], 'termination': termination,
            'diagnostic': diagnostic,
            'return_trace': {'schema_version': 1, 'policy': policy, 'returns': returns},
            **{name: bytes(value).decode('utf-8', 'replace') for name, value in streams.items()}}


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _base_result(source, cases, inputs, limits):
    suite_raw = json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    input_raw = json.dumps(inputs, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return {'schema_version': 1, 'oracle_mode': 'host_data_only_functions_v1',
            'status': 'BLOCKED', 'supported': False, 'passed': False,
            'reason': '', 'returncode': None, 'tests_total': len(cases),
            'tests_run': 0, 'tests_passed': 0, 'cases': [], 'resource_flags': [],
            'stdout': '', 'stderr': '',
            'source_sha256': _digest(source.encode('utf-8')),
            'inputs_sha256': _digest(input_raw), 'data_sha256': _digest(suite_raw),
            'oracle_sha256': _digest(Path(__file__).read_bytes()),
            'sandbox_sha256': _digest(Path(sandbox.__file__).read_bytes()),
            'resources': {'profile': 'single_process_as', 'memory_enforcement': 'rlimit_as_single_process',
                          'aggregate_memory_enforced': False, 'cpu_bandwidth_enforced': False,
                          'requested_limits': dataclasses.asdict(limits),
                          'limits_established': False},
            'claim': 'Correct observable outputs on these cases only; not purity, algorithm identity or universal correctness.'}


def evaluate_functions(source: str, cases: list[dict], limits=None, trace_policy=None) -> dict:
    """Evaluate a nonempty data-only suite; invalid arguments raise ValueError.

    limits is SandboxLimits or its exact-field dict. Runtime unsupported/setup
    failure returns BLOCKED; no weaker execution path exists.
    """
    if type(limits) is dict:
        limits = SandboxLimits(**limits)
    elif limits is None:
        limits = SandboxLimits()
    if type(limits) is not SandboxLimits:
        raise ValueError('limits must be SandboxLimits or an exact-field dict')
    policy = _trace_policy(trace_policy)
    cases, inputs = _suite(source, cases, limits)
    result = _base_result(source, cases, inputs, limits)
    result.update(return_trace={'schema_version': 1, 'policy': policy, 'returns': []},
        diagnostic={'schema_version': 1, 'host_phase': 'preflight', 'event': 'not_executed',
                    'requested_case_id': None, 'candidate_error': None},
        termination=_termination(None, time.monotonic(), 'not_executed'),
        evidence_flags=['sensitive_actual_value_trace'] if policy['return_retention'] == 'value' else [])
    # Mandatory FRESH ACTUAL Linux containment test, before staging candidate.
    preflight = sandbox.probe_code_sandbox(limits)
    if not preflight.passed or not preflight.supported:
        result.update(reason='fresh namespace/runtime preflight failed; candidate was not executed',
                      stderr=preflight.stderr, resource_flags=list(preflight.resource_flags),
                      returncode=preflight.returncode)
        return result
    result['diagnostic'].update(host_phase='setup', event='not_started')
    try:
        with tempfile.TemporaryDirectory(prefix='silt-functions-') as staging:
            root = Path(staging) / 'root'
            (root / 'work').mkdir(parents=True)
            (root / 'tmp').mkdir()
            python = sandbox._runtime(root)
            # This root contains source and PUBLIC execution configuration only.
            (root / 'work/runner.py').write_text(sandbox._ISOLATION_PREFIX + _adapter_source(), encoding='utf-8')
            (root / 'work/candidate.py').write_text(source, encoding='utf-8')
            config = {'python': python, 'limits': dataclasses.asdict(limits),
                      'namespaces': {k: os.readlink('/proc/self/ns/' + k)
                                     for k in ('user', 'mnt', 'pid', 'net', 'ipc', 'uts')}}
            run = _exchange(root, config, cases, limits, policy)
        result.update(return_trace=run['return_trace'], diagnostic=run['diagnostic'],
                      termination=run['termination'])
        result.update(supported=run['ready'], returncode=run['returncode'],
                      tests_run=len(run['completed']),
                      tests_passed=sum(c['matched'] for c in run['completed']),
                      cases=run['completed'], resource_flags=run['flags'],
                      stdout=run['stdout'], stderr=run['stderr'], reason=run['reason'])
        result['resources']['limits_established'] = run['ready']
        passed = (run['ready'] and not run['reason'] and not run['flags']
                  and run['returncode'] == 0 and len(run['completed']) == len(cases)
                  and all(c['matched'] for c in run['completed']))
        status = 'PASSED' if passed else ('FAILED' if run['ready'] else 'BLOCKED')
        if run['ready'] and 'wall_timeout' in run['flags']:
            status = 'TIMEOUT'
        elif run['ready'] and (run['flags'] or (not run['reason'] and run['returncode'] in (-signal.SIGKILL, -signal.SIGXCPU, -signal.SIGXFSZ))):
            status = 'RESOURCE_LIMIT'
            if not run['flags']:
                result['resource_flags'] = ['resource_signal']
        result.update(status=status, passed=passed)
        if not passed and not result['reason']:
            result['reason'] = 'wrong output, incomplete execution or abnormal termination'
        return result
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        result['reason'] = 'function sandbox setup failed: ' + str(exc)
        result['diagnostic'].update(host_phase='setup', event='transport_error')
        result['termination']['cleanup_reason'] = 'setup_failed_before_execution'
        return result


def _read_bounded(path, cap):
    with open(path, 'rb') as handle:
        raw = handle.read(cap + 1)
    if len(raw) > cap:
        raise ValueError('input file exceeds byte budget')
    return raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--output', help='optional new output file; existing files are never replaced')
    parser.add_argument('--trace-policy', choices=('none', 'digest', 'value'), default='none',
                        help='return retention; value exposes sensitive actual values in stdout/output; trusted local review only')
    args = parser.parse_args(argv)
    try:
        source = _read_bounded(args.source, SandboxLimits().source_bytes).decode('utf-8', 'strict')
        cases = _decode(_read_bounded(args.cases, TOTAL_BYTES), byte_cap=TOTAL_BYTES,
                        depth_cap=MAX_DEPTH + 1, node_cap=MAX_NODES * MAX_CASES)
        result = evaluate_functions(source, cases, trace_policy={
            'schema_version': 1, 'return_retention': args.trace_policy})
    except (ValueError, TypeError, OSError) as exc:
        result = {'schema_version': 1, 'status': 'INVALID_INPUT', 'passed': False,
                  'supported': False, 'reason': str(exc)}
    encoded = json.dumps(result, ensure_ascii=True, sort_keys=True)
    if args.output:
        try:
            # O_EXCL via mode x: no following an existing symlink or clobbering.
            def private_opener(path, flags):
                return os.open(path, flags, 0o600)
            with open(args.output, 'x', encoding='utf-8',
                      opener=private_opener if args.trace_policy == 'value' else None) as handle:
                handle.write(encoded + '\n')
        except OSError as exc:
            # stdout remains the authoritative evaluation; publication is separate.
            result['output_error'] = str(exc)
            encoded = json.dumps(result, ensure_ascii=True, sort_keys=True)
    print(encoded)
    return 0 if result['passed'] and 'output_error' not in result else 1


if __name__ == '__main__':
    raise SystemExit(main())
