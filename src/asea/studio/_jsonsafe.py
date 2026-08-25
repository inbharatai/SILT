"""JSON-safety sanitizer for the Studio job surfaces.

Why this exists (binding honesty rule: "no half measures, no loopholes"):

  A quantization or training run can legitimately produce a NON-FINITE float --
  e.g. an int2-quantized state of a tiny model collapses to NaN loss, or a
  parity ratio divides by zero. Python's ``float`` happily carries ``nan`` /
  ``inf``, but JSON cannot encode them: ``json.dumps`` with the settings the
  web framework uses raises ``ValueError: Out of range float values are not
  JSON compliant``, which turns a single NaN in one job's report into an HTTP
  500 that also kills the job *listing* endpoint (it serializes every job).
  That breaks the "Past runs" card silently and hides the real, honest outcome
  (the state produced no usable loss) behind a server crash.

  A second loophole: a NaN degradation is *neither* ``<= tolerance`` (certified)
  *nor* ``> tolerance`` (revoked) -- NaN comparisons are always False -- so a
  state with an unmeasurable skill used to slip through as "certified" because
  nothing was revoked. That is a false promise. The sanitizer replaces
  non-finite floats with ``None`` (the UI renders "—", honest) and the spring
  state-builder now treats any non-finite degradation as "unmeasured" so the
  state is NOT marked overall_certified.

This module is the single chokepoint: both ``spring_jobs`` and ``deepapply_jobs``
run every served artifact (the report dict and every telemetry event) through
``json_safe`` at the boundary, so a non-finite float can never reach the HTTP
response or the SSE stream. Source values stay untouched in the runner; only
the served copy is sanitized.
"""

from __future__ import annotations

import math
from typing import Any

try:  # numpy is a [deep]-extra dependency, not a hard one for the Studio core.
    import numpy as _np
except ImportError:  # pragma: no cover - numpy absent in the minimal env
    _np = None  # type: ignore[assignment]

# Numpy scalar types are NOT subclasses of the matching Python scalars
# (np.int64 is not an int, np.bool_ is not a bool on every version), so the
# isinstance ladder below would let them fall through to ``return obj`` -- and
# a numpy int reaching FastAPI's encoder would raise ``TypeError: Object of
# type int64 is not JSON serializable`` -> HTTP 500, while one reaching a
# telemetry ``json.dumps(..., default=str)`` would be stringized ("5" instead
# of 5), silently corrupting the type the client renders. Coerce explicitly.
if _np is not None:
    _NP_FLOAT = (_np.floating,)
    _NP_INT = (_np.integer,)
    _NP_BOOL = (_np.bool_,)
else:  # pragma: no cover
    _NP_FLOAT = _NP_INT = _NP_BOOL = ()


def _finite(v: Any) -> Any:
    """Round a float to 6 dp, returning None for non-finite values (nan/inf).

    ints and non-numeric values pass through unchanged. A None stays None.
    Numpy scalars are coerced to native Python scalars here so the walker's
    isinstance ladder never has to handle them.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return v  # bool is an int subclass; do not numericize
    if _NP_BOOL and isinstance(v, _NP_BOOL):
        return bool(v)
    if _NP_INT and isinstance(v, _NP_INT):
        return int(v)
    if _NP_FLOAT and isinstance(v, _NP_FLOAT):
        f = float(v)
        return None if not math.isfinite(f) else round(f, 6)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if not math.isfinite(v):
            return None
        return round(v, 6)
    # Anything else (str, etc.) is left for the recursive walker to handle.
    return v


def json_safe(obj: Any) -> Any:
    """Recursively copy ``obj`` replacing every non-finite float with None and
    rounding finite floats to 6 dp. Numpy scalars are coerced to native Python
    scalars. Dicts/lists are walked; other scalars pass through. Returns a NEW
    structure (never mutates the runner's report)."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if _NP_BOOL and isinstance(obj, _NP_BOOL):
        return bool(obj)
    if _NP_INT and isinstance(obj, _NP_INT):
        return int(obj)
    if _NP_FLOAT and isinstance(obj, _NP_FLOAT):
        f = float(obj)
        return None if not math.isfinite(f) else round(f, 6)
    if isinstance(obj, float):
        return _finite(obj)
    if isinstance(obj, int):
        return obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj