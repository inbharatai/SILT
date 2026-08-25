"""Trainer backends for deep-apply, behind the :class:`TrainerBackend` ABC.

This package groups the real training implementations so ``trainer.py`` stays
a thin registry. Three backends live here:

* :class:`StandardTrainerBackend` (in ``trainer.py``) -- model resident on
  device; the default, CPU-graceful for small models.
* :class:`SiltStreamBackend` (``streamed.py``) -- low-VRAM layer-streamed
  LoRA training backed by the vendored ``siltstream`` package. Parity is the
  admission bar: a configuration whose parity is unverified is recorded as
  ``parity_verified=false``; a parity FAILURE aborts the run with
  :class:`DeepApplyBlocked` (never a warning, never a fallback).
* :class:`ZeroForgeBackend` (``zeroforge.py``) -- forward-only zeroth-order
  LoRA (no backward passes); sits behind Gate 2 like every other backend.

The vendored siltstream package is in :mod:`.siltstream_vendor` (first-party,
Apache-2.0, verbatim copy -- see its module docstring for version + test
status). Gate 2 contains zero backend-conditional branches: a streamed or
zeroforge adapter is judged identically to a standard one.
"""

from __future__ import annotations

from .streamed import SiltStreamArtifact, SiltStreamBackend
from .zeroforge import ZeroForgeBackend

__all__ = [
    "SiltStreamArtifact",
    "SiltStreamBackend",
    "ZeroForgeBackend",
]