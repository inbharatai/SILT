"""Capability spec loading and validation.

A spec file is a closed JSON document (bounded, no symlinks, strict schema)
following :data:`asea.capability_build.schema.CAPABILITY_SPEC_SCHEMA`. Every
capability-build command begins by validating a spec; nothing runs without
one. Thresholds inside the spec are operator configuration, never
guarantees.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from asea.artifacts import digest, safe_file

from .errors import SpecInvalid
from .schema import CAPABILITY_SPEC_SCHEMA, CapabilitySpec, spec_fingerprint

_MAX_SPEC_BYTES = 256 * 1024


def load_spec(path) -> Dict[str, Any]:
    """Load and validate a capability spec from a bounded JSON file.

    Returns ``{"spec": CapabilitySpec, "spec_sha256": str}``. Raises
    :class:`SpecInvalid` on any malformed input -- never a guessed default.
    """
    try:
        resolved = safe_file(path)
    except Exception as exc:  # asea.artifacts.Blocked and friends
        raise SpecInvalid("spec path rejected: %s" % exc) from exc
    if not resolved.is_file():
        raise SpecInvalid("spec path is not a file: %s" % resolved)
    if resolved.stat().st_size > _MAX_SPEC_BYTES:
        raise SpecInvalid("spec file exceeds %d bytes" % _MAX_SPEC_BYTES)
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise SpecInvalid("spec is not valid JSON: %s" % exc) from exc
    if not isinstance(raw, dict) or raw.get("schema") != CAPABILITY_SPEC_SCHEMA:
        raise SpecInvalid(
            "spec must carry schema '%s'" % CAPABILITY_SPEC_SCHEMA
        )
    try:
        spec = CapabilitySpec.model_validate(raw)
    except ValueError as exc:
        raise SpecInvalid("spec validation failed: %s" % exc) from exc
    return {
        "spec": spec,
        "spec_sha256": digest(spec.model_dump(mode="json", by_alias=True)),
        "spec_fingerprint": spec_fingerprint(spec),
    }


def validate_spec_payload(raw: Any) -> CapabilitySpec:
    """Validate an already-parsed spec payload (used by tests and the
    in-memory path). Raises :class:`SpecInvalid` on failure."""
    if not isinstance(raw, dict) or raw.get("schema") != CAPABILITY_SPEC_SCHEMA:
        raise SpecInvalid("spec must carry schema '%s'" % CAPABILITY_SPEC_SCHEMA)
    try:
        return CapabilitySpec.model_validate(raw)
    except ValueError as exc:
        raise SpecInvalid("spec validation failed: %s" % exc) from exc