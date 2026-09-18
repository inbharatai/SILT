"""Script collection hygiene (audit 2026-09-18, S-series).

Sibling scripts in ``scripts/`` used to have import-time side effects or
machine-specific hard-coded defaults:

  * S1 ``prepare_specialist_data.py``: DEFAULT_TOKENIZER is a hard-coded
    ``/agent/workspace/silt-models/...`` path that exists on exactly one
    machine. NOT FIXED, deliberately: the frozen
    ``data/specialist-v1/manifest.json`` pins ``builder_sha256`` -- the
    byte-exact identity of the script that built the frozen dataset --
    and ``test_sha_inventory_covers_all_frozen_data_and_builder`` enforces
    it. Any edit to the builder (including this fix) breaks frozen-data
    integrity verification, and the manifest's own governance requires a
    NEW dataset version and lock for any builder revision. The fix is
    recorded here as pending specialist-v2 work, not silently shipped.
  * S2 ``setup_local_validation.py``: the whole body used to run at
    import -- it parsed ``sys.argv`` and wrote fixture files. Collection
    (pytest, help tooling) must never have filesystem side effects. FIXED.
  * S3 ``probe_7b_4bit_footprint.py``: the GPU probe body (CUDA reset,
    4-bit config, model load) used to run at import. Importing it must
    import torch and NOTHING more. FIXED.

These are MECHANISM tests of script collection discipline, not model or
GPU results of any kind.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_specialist_builder_bytes_are_pinned_by_the_frozen_manifest():
    """S1 status check: the specialist builder must stay byte-identical to
    the ``builder_sha256`` pin in the FROZEN specialist-v1 manifest until a
    v2 dataset replaces it. This is what keeps the hardcoded tokenizer-path
    finding unfixed -- fixing it requires a new dataset version and lock,
    never an in-place builder edit."""
    import json

    manifest = json.loads((ROOT / "data/specialist-v1/manifest.json").read_bytes())
    import hashlib

    current = hashlib.sha256(
        (ROOT / "scripts/prepare_specialist_data.py").read_bytes()
    ).hexdigest()
    assert current == manifest["builder_sha256"]


# ---------------------------------------------------------------------------
# S2: fixture script import writes nothing and parses nothing
# ---------------------------------------------------------------------------

def test_validation_fixture_script_import_is_side_effect_free(tmp_path, monkeypatch):
    """Old body parsed sys.argv at import (SystemExit under pytest) and then
    wrote fixture directories. Collection must import clean and write
    nothing."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["setup_local_validation.py"])
    module = _load("validation_fixtures_s2", "scripts/setup_local_validation.py")
    assert callable(module.main)
    assert list(tmp_path.iterdir()) == []  # no fixture file was written


# ---------------------------------------------------------------------------
# S3: the GPU probe imports torch and nothing more
# ---------------------------------------------------------------------------

def test_gpu_probe_imports_without_running_gpu_work():
    """Old body called ``torch.cuda.reset_peak_memory_stats()`` and loaded a
    4-bit model at import. On this repo's CPU-only torch an import CRASHED;
    on a GPU host it silently allocated the card during test collection.
    Importing must define ``main`` and do no GPU work."""
    pytest.importorskip("torch")
    module = _load("gpu_probe_s3", "scripts/probe_7b_4bit_footprint.py")
    assert callable(module.main)