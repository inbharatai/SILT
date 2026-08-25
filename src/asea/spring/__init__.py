"""SiltSpring compression certification -- the THIRD SILT surface.

Beside transfer (instant inference-time skills) and deep-apply (gated LoRA
training), SILT can certify which quantized "spring states" of a model preserve
each gated skill. See :mod:`.certifier` for the trust discipline (per-state
capability certificates, honest refusal, staleness-bound certificates).

Heavy deps (``torch``) are imported lazily inside the certifier, so this
package imports cleanly without the ``[deep]`` extra.
"""

from __future__ import annotations

from .certifier import (
    CompressionCertifier,
    suites_from_benchmark,
)

__all__ = [
    "CompressionCertifier",
    "suites_from_benchmark",
]