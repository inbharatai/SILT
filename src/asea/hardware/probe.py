"""Read-only, stdlib-first resource discovery; no host identifiers or model loads."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading

PROBE_TIMEOUT_SECONDS = 20
PROBE_OUTPUT_BYTES = 65536


def _text(path):
    with Path(path).open() as handle:
        return handle.read(1024 * 1024)


def _cpuset(value):
    cpus = set()
    for item in value.strip().split(','):
        if not item:
            continue
        ends = item.split('-')
        start, stop = int(ends[0]), int(ends[-1])
        if len(ends) > 2 or not 0 <= start <= stop <= 1048576:
            raise ValueError('invalid cpuset')
        cpus.update(range(start, stop + 1))
    return cpus


def _cgroup_locations(read=_text):
    """Resolve namespace mount roots, not just /sys/fs/cgroup; visit ancestors."""
    result = {(Path('/sys/fs/cgroup'), Path('/sys/fs/cgroup'), 'v2'),
              (Path('/sys/fs/cgroup/memory'), Path('/sys/fs/cgroup/memory'), 'memory'),
              (Path('/sys/fs/cgroup/cpu'), Path('/sys/fs/cgroup/cpu'), 'cpu'),
              (Path('/sys/fs/cgroup/cpuset'), Path('/sys/fs/cgroup/cpuset'), 'cpuset')}
    def unescape(s):
        for code, char in [('040', ' '), ('011', '\t'), ('012', '\n'), ('134', '\\')]:
            s = s.replace('\\' + code, char)
        return s
    try:
        groups = [s.split(':', 2) for s in read('/proc/self/cgroup').splitlines()]
        for line in read('/proc/self/mountinfo').splitlines():
            left, right = line.split(' - ', 1)
            fields, fs = left.split(), right.split()
            if fs[0] not in ('cgroup', 'cgroup2'):
                continue
            mount, root = Path(unescape(fields[4])), Path(unescape(fields[3]))
            for _, controllers, group in groups:
                kinds = ['v2'] if fs[0] == 'cgroup2' and not controllers else [
                    k for k in ('memory', 'cpu', 'cpuset')
                    if k in controllers.split(',') and k in fs[-1].split(',')]
                group = Path(group)
                if group == root or root in group.parents:
                    for kind in kinds:
                        result.add((mount / group.relative_to(root), mount, kind))
    except (OSError, ValueError, IndexError):
        pass
    return result


def _linux_resources(read=_text, cpu_count=None, affinity=None):
    errors, total, available = [], None, None
    try:
        mem = {line.split(':')[0]: int(line.split()[1]) * 1024
               for line in read('/proc/meminfo').splitlines() if ':' in line}
        total, available = mem.get('MemTotal'), mem.get('MemAvailable')
    except (OSError, ValueError, IndexError):
        errors.append('host_memory_unavailable')
    remaining, limits, quotas, cpu_sets = [], [], [], []
    if affinity is not None:
        cpu_sets.append(set(affinity))
    seen = set()
    for location, mount, kind in _cgroup_locations(read):
        while location == mount or mount in location.parents:
            if (location, kind) not in seen:
                seen.add((location, kind))
                if kind in ('v2', 'memory'):
                    a, b = ('memory.max', 'memory.current') if kind == 'v2' else ('memory.limit_in_bytes', 'memory.usage_in_bytes')
                    try:
                        value = read(location / a).strip()
                        if value != 'max':
                            limit, used = int(value), int(read(location / b))
                            if limit < 0 or used < 0:
                                raise ValueError('negative memory constraint')
                            limits.append(limit)
                            remaining.append(max(0, limit - used))
                    except FileNotFoundError:
                        pass
                    except (OSError, ValueError):
                        errors.append('cgroup_memory_constraint_unreadable')
                if kind in ('v2', 'cpu'):
                    try:
                        if kind == 'v2':
                            quota, period = read(location / 'cpu.max').split()
                            quota, period = (-1 if quota == 'max' else int(quota)), int(period)
                        else:
                            quota, period = int(read(location / 'cpu.cfs_quota_us')), int(read(location / 'cpu.cfs_period_us'))
                        if quota >= 0 and period > 0:
                            quotas.append(quota / period)
                    except FileNotFoundError:
                        pass
                    except (OSError, ValueError):
                        errors.append('cgroup_cpu_constraint_unreadable')
                if kind in ('v2', 'cpuset'):
                    try:
                        cpus = _cpuset(read(location / ('cpuset.cpus.effective' if kind == 'v2' else 'cpuset.cpus')))
                        if cpus:
                            cpu_sets.append(cpus)
                    except FileNotFoundError:
                        pass
                    except (OSError, ValueError):
                        errors.append('cgroup_cpuset_constraint_unreadable')
            if location == mount:
                break
            location = location.parent
    host = cpu_count if cpu_count is not None else os.cpu_count()
    intersection = set.intersection(*cpu_sets) if cpu_sets else None
    counts = ([host] if host else []) + quotas + ([len(intersection)] if intersection is not None else [])
    values = ([available] if available is not None else []) + remaining
    return {'cpu': {'logical_count': host, 'affinity_count': len(affinity) if affinity is not None else None,
                    'cpuset_intersection_count': len(intersection) if intersection is not None else None,
                    'quota_cores': min(quotas) if quotas else None,
                    'effective_cores': min(counts) if counts and not any('cpu' in e for e in errors) else None},
            'memory': {'host_total_bytes': total, 'host_available_bytes': available,
                       'cgroup_limit_bytes': min(limits) if limits else None,
                       'cgroup_remaining_bytes': min(remaining) if remaining else None,
                       'available_bytes': min(values) if values and not any('memory_constraint' in e for e in errors) else None},
            'errors': sorted(set(errors))}


def _windows_memory():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ('total_phys', 'avail_phys', 'total_page', 'avail_page', 'total_virtual', 'avail_virtual', 'avail_extended')]
    value = MemoryStatus()
    value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise OSError('GlobalMemoryStatusEx failed')
    return {'host_total_bytes': value.total_phys, 'host_available_bytes': value.avail_phys,
            'available_bytes': value.avail_phys, 'cgroup_limit_bytes': None, 'cgroup_remaining_bytes': None}


def _mac_memory():
    # sysctlbyname is read-only; no vm tuning commands. vm_stat free+inactive is
    # only a conservative approximation and macOS execution is not admitted.
    libc = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
    value, size = ctypes.c_uint64(), ctypes.c_size_t(8)
    if libc.sysctlbyname(b'hw.memsize', ctypes.byref(value), ctypes.byref(size), None, 0):
        raise OSError('sysctl hw.memsize failed')
    return {'host_total_bytes': value.value, 'host_available_bytes': None, 'available_bytes': None,
            'cgroup_limit_bytes': None, 'cgroup_remaining_bytes': None}


def disk_snapshot(paths=()):
    """One entry per requested path on its actual existing ancestor/volume."""
    result = []
    for raw in paths:
        path = Path(raw).expanduser().absolute()
        original = path
        while not path.exists() and path != path.parent:
            path = path.parent
        if path.is_file():
            path = path.parent
        try:
            usage = shutil.disk_usage(path)
            result.append({'path': str(original), 'existing_parent': str(path),
                           'total_bytes': usage.total, 'free_bytes': usage.free,
                           'used_bytes': usage.used, 'error': None})
        except OSError:
            result.append({'path': str(original), 'existing_parent': str(path),
                           'total_bytes': None, 'free_bytes': None, 'used_bytes': None,
                           'error': 'disk_usage_unavailable'})
    return result


# Trusted fixed script: selected interpreter, no model/config/tokenizer loaders.
_TORCH_SCRIPT = r'''
import importlib.metadata, json
result = {"status":"unavailable", "torch_version":None, "cuda_version":None, "hip_version":None,
          "mps_available":False, "cpu":{"float32":False,"bfloat16":False}, "devices":[], "packages":{}}
for name in ("torch", "transformers", "peft", "accelerate", "safetensors"):
    try: result["packages"][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: result["packages"][name] = None
try:
    import torch
    torch.set_num_threads(1)
    result.update(status="ok", torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                  hip_version=torch.version.hip, mps_available=bool(hasattr(torch.backends,"mps") and torch.backends.mps.is_available()))
    for name in ("float32", "bfloat16"):
        try:
            x = torch.ones((2,2), dtype=getattr(torch,name)); y=x@x
            result["cpu"][name] = bool(torch.isfinite(y).all())
            del x,y
        except Exception: pass
    if torch.cuda.is_available():
        for index in range(min(torch.cuda.device_count(), 64)):
            device={"index":index,"name":None,"vendor":"AMD" if torch.version.hip else "NVIDIA",
                    "total_bytes":None,"free_bytes":None,"allocated_bytes":None,"reserved_bytes":None,"bf16_supported":False,
                    "primitives":{"float32":False,"bfloat16":False}}
            try:
                with torch.cuda.device(index):
                    props=torch.cuda.get_device_properties(index)
                    device.update(name=str(props.name), total_bytes=int(props.total_memory),
                                  compute_capability=[int(props.major), int(props.minor)])
                    free,total=torch.cuda.mem_get_info(index)
                    device.update(free_bytes=int(free),total_bytes=int(total))
                    try: device["bf16_supported"]=bool(torch.cuda.is_bf16_supported(including_emulation=False))
                    except TypeError: device["bf16_supported"]=bool(props.major>=8 and torch.cuda.is_bf16_supported())
                    for name in ("float32", "bfloat16"):
                        if name=="bfloat16" and not device["bf16_supported"]: continue
                        try:
                            x=torch.ones((2,2),device="cuda:%d"%index,dtype=getattr(torch,name)); y=x@x
                            torch.cuda.synchronize(index)
                            device["primitives"][name]=bool(torch.isfinite(y).all().item())
                            del x,y
                        except Exception: pass
                    # Observe AFTER primitives, on this exact selected Torch device.
                    # Allocator caches remain reserved; never assume empty_cache.
                    free,total=torch.cuda.mem_get_info(index)
                    device.update(free_bytes=int(free),total_bytes=int(total),
                                  allocated_bytes=int(torch.cuda.memory_allocated(index)),
                                  reserved_bytes=int(torch.cuda.memory_reserved(index)))
                device["probe_valid"]=True
            except Exception: device["probe_valid"]=False
            result["devices"].append(device)
except Exception: pass
print("SILT_HARDWARE_JSON="+json.dumps(result,sort_keys=True))
'''


def _torch_probe(python_executable=None):
    executable = os.fspath(python_executable) if python_executable is not None else sys.executable
    failure = {'status': 'unavailable', 'torch_version': None, 'cuda_version': None,
               'hip_version': None, 'mps_available': False, 'cpu': {}, 'devices': [], 'packages': {}}
    try:
        process = subprocess.Popen([executable, '-I', '-c', _TORCH_SCRIPT], stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        captured = []
        def drain():
            captured.append(process.stdout.read(PROBE_OUTPUT_BYTES + 1))
            if len(captured[0]) > PROBE_OUTPUT_BYTES:
                process.kill()
        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        try:
            process.wait(timeout=PROBE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            return {**failure, 'status': 'timeout'}
        finally:
            reader.join(timeout=5)
        if not captured or len(captured[0]) > PROBE_OUTPUT_BYTES or process.returncode:
            return {**failure, 'status': 'invalid_or_oversized_output'}
        lines = captured[0].decode('utf-8').splitlines()
        values = [line.removeprefix('SILT_HARDWARE_JSON=') for line in lines if line.startswith('SILT_HARDWARE_JSON=')]
        if len(values) != 1:
            return failure
        return json.loads(values[0])
    except (OSError, ValueError, UnicodeError):
        return failure
    finally:
        if 'process' in locals() and process.stdout:
            process.stdout.close()


def probe_hardware(python_executable=None, paths=()):
    """Return JSON-compatible snapshot. No serials, UUIDs, hostnames or credentials.

    A subprocess imports Torch for bounded tiny arithmetic/capability probes, not
    model loading. Unknown readings stay null; free VRAM never substitutes RAM.
    """
    system = platform.system()
    cpu = {'logical_count': os.cpu_count(), 'effective_cores': os.cpu_count(),
           'quota_cores': None, 'affinity_count': None, 'cpuset_intersection_count': None}
    resource = {'cpu': cpu, 'memory': {'available_bytes': None, 'host_total_bytes': None,
                'host_available_bytes': None, 'cgroup_limit_bytes': None, 'cgroup_remaining_bytes': None}, 'errors': []}
    wsl = False
    if system == 'Linux':
        try:
            affinity = os.sched_getaffinity(0)
        except (OSError, AttributeError):
            affinity = None
        resource = _linux_resources(affinity=affinity)
        try:
            wsl = 'microsoft' in _text('/proc/sys/kernel/osrelease').lower()
        except OSError:
            pass
    elif system in ('Windows', 'Darwin'):
        try:
            resource['memory'] = _windows_memory() if system == 'Windows' else _mac_memory()
        except (OSError, AttributeError):
            resource['errors'].append('platform_memory_unavailable')
    torch = _torch_probe(python_executable)
    return {'schema_version': 1, 'profile_kind': 'detected',
            'python_executable': os.fspath(python_executable) if python_executable is not None else sys.executable,
            'platform': {'system': system, 'wsl': wsl,
            'execution_supported': system == 'Linux',
            'wsl_launcher_available': shutil.which('wsl.exe') is not None if system=='Windows' else None,
            'wsl_note': 'kernel/launcher presence is descriptive only; no distributions enumerated and no Windows execution validation'},
            **resource, 'disks': disk_snapshot(paths), 'torch': torch,
            'observation_policy': 'read-only host/cgroup intersections and bounded selected-Python Torch primitive probe; no model loads',
            'limitations': ['Snapshot, not reservation; runtime must recheck.',
                            'Invisible enclosing cgroups and concurrent consumers cannot be predicted.',
                            'Windows, macOS/MPS, ROCm, multi-GPU and offload execution are unsupported.']}
