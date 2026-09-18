"""Authoring self-check for the GLM-5.3-Flash pilot case set (2026-09-18).

Verifies, for every authored case:
  1. the SHIPPED buggy function fails at least one oracle check (the seeded
     bug is real, not decorative), and
  2. a reference implementation that meets the written specification passes
     EVERY oracle check exactly (the checks are satisfiable as written).

This is AUTHOR DILIGENCE ONLY -- it is not the host oracle and not any model
measurement. The official judgments run through the Linux code sandbox via
`silt-capability evaluate` in WSL.
"""
import inspect
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
cases = [json.loads(line) for line in
         (HERE / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

# Reference implementations that meet each written specification exactly.
def flatten(items):
    out = []
    for item in items:
        if isinstance(item, list):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out

def median_odd(numbers):
    s = sorted(numbers)
    return s[len(s) // 2]

def slugify(text):
    import re
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

def chunk(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]

def dedupe(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def transpose(matrix):
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]

def count_words(text):
    return len(text.split())

def rotate(items, k):
    k = k % len(items)
    return (items[-k:] + items[:-k]) if k else list(items)

def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged

def capitalize_words(text):
    return " ".join(w[:1].upper() + w[1:] for w in text.split(" "))

def binary_search(items, target):
    lo, hi = 0, len(items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

def running_max(items):
    out = []
    biggest = None
    for x in items:
        biggest = x if biggest is None or x > biggest else biggest
        out.append(biggest)
    return out

def pad_center(text, width):
    if len(text) >= width:
        return text
    pad = width - len(text)
    left = pad // 2
    return " " * left + text + " " * (pad - left)

def is_balanced(text):
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0

def running_sum(items):
    out = []
    total = 0
    for x in items:
        total += x
        out.append(total)
    return out

def greeting(name, punct):
    return "Hello, " + name + punct

def total_cents(prices, quantity):
    return sum(prices) * quantity

def initials(full_name):
    return ".".join(w[0].upper() for w in full_name.split()) + "."

def date_label(year, month, day):
    return "%04d-%02d-%02d" % (year, month, day)

BUGGY = {
    "flatten_recursive_v1": lambda items: (lambda result: result)(None) or _buggy_flatten(items),
}
def _buggy_flatten(items):
    result = []
    for item in items:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result

def _buggy_median_odd(numbers):
    n = len(numbers)
    return numbers[n // 2]

def _buggy_slugify(text):
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append('-')
    return ''.join(out)

def _buggy_chunk(items, size):
    end = len(items) // size * size
    return [items[i:i + size] for i in range(0, end, size)]

def _buggy_dedupe(items):
    return list(set(items))

def _buggy_transpose(matrix):
    return [list(col) for col in matrix]

def _buggy_count_words(text):
    return len(text.split(' '))

def _buggy_rotate(items, k):
    n = len(items)
    k = k % n
    return items[k:] + items[:k]

def _buggy_merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged

def _buggy_capitalize_words(text):
    words = text.split(' ')
    out = []
    for w in words:
        out.append(w.upper() if w else w)
    return ' '.join(out)

def _buggy_binary_search(items, target):
    lo, hi = 0, len(items) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo

def _buggy_running_max(items):
    out = []
    biggest = None
    for x in items:
        if biggest is None or x > biggest:
            biggest = x
        out.append(x)
    return out

def _buggy_pad_center(text, width):
    pad = width - len(text)
    left = pad // 2
    return ' ' * left + text

def _buggy_is_balanced(text):
    return text.count('(') == text.count(')')

def _buggy_running_sum(items):
    total = 0
    out = []
    for x in items:
        out.append(total)
        total = total + x
    return out

BUGGY_IMPLS = {
    "flatten_recursive_v1": _buggy_flatten,
    "median_odd_v1": _buggy_median_odd,
    "slugify_v1": _buggy_slugify,
    "chunk_v1": _buggy_chunk,
    "dedupe_stable_v1": _buggy_dedupe,
    "transpose_rows_v1": _buggy_transpose,
    "count_words_v1": _buggy_count_words,
    "rotate_list_v1": _buggy_rotate,
    "merge_intervals_v1": _buggy_merge_intervals,
    "capitalize_words_v1": _buggy_capitalize_words,
    "binary_search_v1": _buggy_binary_search,
    "running_max_v1": _buggy_running_max,
    "pad_center_v1": _buggy_pad_center,
    "is_balanced_v1": _buggy_is_balanced,
    "running_sum_v1": _buggy_running_sum,
}

REF_IMPLS = {
    "flatten_recursive_v1": flatten,
    "median_odd_v1": median_odd,
    "slugify_v1": slugify,
    "chunk_v1": chunk,
    "dedupe_stable_v1": dedupe,
    "transpose_rows_v1": transpose,
    "count_words_v1": count_words,
    "rotate_list_v1": rotate,
    "merge_intervals_v1": merge_intervals,
    "capitalize_words_v1": capitalize_words,
    "binary_search_v1": binary_search,
    "running_max_v1": running_max,
    "pad_center_v1": pad_center,
    "is_balanced_v1": is_balanced,
    "running_sum_v1": running_sum,
    "greeting_format_v1": greeting,
    "price_total_v1": total_cents,
    "initials_v1": initials,
    "date_label_v1": date_label,
}

failures = 0
for case in cases:
    sid = case["sample_id"]
    checks = case["oracle"]
    assert len(checks) >= 2, sid
    ids = [c["id"] for c in checks]
    assert len(set(ids)) == len(ids), "%s duplicate check ids" % sid
    # 0. args arity must exactly match the reference signature -- an extra
    #    nesting level turns a list argument into a wrapped list and the
    #    check silently measures the wrong thing
    ref = REF_IMPLS[sid]
    arity = len(inspect.signature(ref).parameters)
    for check in checks:
        assert len(check["args"]) == arity, (
            "%s/%s: %d args for a %d-parameter function"
            % (sid, check["id"], len(check["args"]), arity))
    # 2. reference implementation passes every check exactly
    for check in checks:
        got = ref(*check["args"], **check.get("kwargs", {}))
        assert got == check["expected"] and type(got) is type(check["expected"]), \
            "%s/%s: reference returned %r (%s), expected %r (%s)" % (
                sid, check["id"], got, type(got).__name__,
                check["expected"], type(check["expected"]).__name__)
    # 1. shipped buggy function fails at least one check (targets only;
    #    control cases ship no buggy code)
    if sid in BUGGY_IMPLS:
        buggy = BUGGY_IMPLS[sid]
        passing = sum(
            1 for check in checks
            if buggy(*check["args"], **check.get("kwargs", {})) == check["expected"]
        )
        assert passing < len(checks), (
            "%s: buggy function passes EVERY check -- the seeded bug is "
            "decorative" % sid)

print("authoring self-check OK: %d cases, %d total checks; every seeded bug "
      "is real and every check set is satisfiable by its written spec"
      % (len(cases), sum(len(c["oracle"]) for c in cases)))
sys.exit(0)