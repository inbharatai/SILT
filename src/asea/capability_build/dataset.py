"""Fresh capability-build dataset builder (brief §N, Phase 6).

Builds ``data/capability_v1/`` for a capability spec: five splits
(training/development/heldout/final/controls), each case carrying a unique
ID, content hash, family ID, provenance and license.

Discipline inherited verbatim from the specialist path (recovery.py:693-749,
workflow.py data_preflight) plus one NEW guard:

  * cross-split sample-id disjointness
  * cross-split exact-content disjointness (sha256 of prompt+expected)
  * cross-split family disjointness -- a family NEVER crosses splits
  * within-split id/content uniqueness
  * NEAR-DUPLICATE token-shape guard (new: the existing repo only guards
    exact duplicates) -- lowercased alphanumeric/whitespace shape must be
    unique within a split AND across splits
  * the frozen September sets are quarantined as INPUTS: no case may be
    sourced from data/specialist-v1, data/pilot-v2, data/speech-pilot-v2
    or data/specialist-span-v1 -- the consumed 16-task final set is never
    reused as training material
  * the manifest records ``frozen_before_model_generation: True`` and
    ``model_outputs_consulted: False`` at BUILD time, before any teacher,
    student or reduction sees a single case (selection-lock pattern)

The builder writes the dataset; it never grades anything, never runs a
model, and never touches the frozen September result sets.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .errors import CapabilityBuildError
from .schema import spec_fingerprint

DATASET_SCHEMA = "silt.capability_cases.v1"
MANIFEST_SCHEMA = "silt.capability_dataset_manifest.v1"
SELECTION_LOCK_SCHEMA = "silt.capability_selection_lock.v1"

#: The five splits, exactly as the spec's evaluation section names them.
SPLITS = ("training", "development", "heldout", "final", "controls")

#: Directory names of the frozen September 2026 sets. Any INPUT whose path
#: contains one of these parts is refused: those sets are consumed evidence,
#: never training material.
FROZEN_SOURCE_MARKERS = (
    "specialist-v1", "pilot-v2", "speech-pilot-v2", "specialist-span-v1",
)

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_GROUP_RE = ("target", "control")


class DatasetInvalid(CapabilityBuildError):
    """A dataset (or a single case) violates the build discipline."""


def _content_hash(case: Dict[str, Any]) -> str:
    payload = case["prompt"] + "\x00" + (case.get("expected") or "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _token_shape(text: str) -> str:
    """Coarse near-duplicate shape (same helper contract as distillation)."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _quarantine_source(path: Path) -> None:
    parts = {p.lower() for p in path.parts}
    for marker in FROZEN_SOURCE_MARKERS:
        if marker in parts:
            raise DatasetInvalid(
                "input %s is inside the frozen September set %r; consumed "
                "sets are never reused as capability-build material"
                % (path, marker)
            )


def validate_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """Validate one case record. Every case needs: unique-format sample_id,
    split, group, prompt, family_id, provenance and license. ``expected``
    is optional (cases judged by the host oracle do not need it)."""
    if not isinstance(case, dict):
        raise DatasetInvalid("case %r is not an object" % (case,))
    for field in ("sample_id", "split", "group", "prompt", "family_id",
                  "provenance", "license"):
        if field not in case:
            raise DatasetInvalid("case is missing required field %r" % field)
    if not isinstance(case["sample_id"], str) or not _ID_RE.match(case["sample_id"]):
        raise DatasetInvalid("bad sample_id %r" % (case.get("sample_id"),))
    if case["split"] not in SPLITS:
        raise DatasetInvalid(
            "sample %s: split %r not in %s" % (case["sample_id"], case["split"], SPLITS)
        )
    if case["group"] not in _GROUP_RE:
        raise DatasetInvalid(
            "sample %s: group %r not in %s" % (case["sample_id"], case["group"], _GROUP_RE)
        )
    if not isinstance(case["prompt"], str) or not case["prompt"].strip():
        raise DatasetInvalid("sample %s: empty prompt" % case["sample_id"])
    for field in ("family_id", "provenance", "license"):
        if not isinstance(case[field], str) or not case[field].strip():
            raise DatasetInvalid("sample %s: empty %s" % (case["sample_id"], field))
    if "expected" in case and case["expected"] is not None:
        if not isinstance(case["expected"], str):
            raise DatasetInvalid("sample %s: expected must be a string" % case["sample_id"])
    return case


def read_cases(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL case file with the frozen-set quarantine guard."""
    path = Path(path)
    _quarantine_source(path)
    if not path.is_file():
        raise DatasetInvalid("case file %s does not exist" % path)
    cases: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetInvalid("%s line %d is not JSON: %s" % (path, lineno, exc))
            raw.setdefault("source_file", path.name)
            cases.append(validate_case(raw))
    return cases


def _check_disjointness(by_split: Dict[str, List[Dict[str, Any]]]) -> None:
    seen_ids: Dict[str, str] = {}
    seen_content: Dict[str, str] = {}
    seen_shapes: Dict[str, str] = {}
    seen_families: Dict[str, str] = {}
    for split in SPLITS:
        for case in by_split[split]:
            sid = case["sample_id"]
            if sid in seen_ids:
                raise DatasetInvalid(
                    "sample_id %r appears in both %r and %r"
                    % (sid, seen_ids[sid], split)
                )
            seen_ids[sid] = split
            content = _content_hash(case)
            if content in seen_content:
                raise DatasetInvalid(
                    "exact content duplicate: sample %r collides with a case "
                    "in %r" % (sid, seen_content[content])
                )
            seen_content[content] = split
            shape = _token_shape(case["prompt"])
            if shape and shape in seen_shapes:
                raise DatasetInvalid(
                    "near-duplicate prompt (token-shape collision): sample %r "
                    "collides with a case in %r" % (sid, seen_shapes[shape])
                )
            seen_shapes[shape] = split
            family = case["family_id"]
            if family in seen_families:
                raise DatasetInvalid(
                    "family %r crosses splits (%r and %r); a family never "
                    "crosses splits" % (family, seen_families[family], split)
                )
            seen_families[family] = split


def build_dataset(
    cases: Sequence[Dict[str, Any]], *, output_dir: Path, spec=None
) -> Dict[str, Any]:
    """Validate and write the five split files + manifest + selection lock.

    ``output_dir`` must not exist (new exclusive path, recovery.py pattern);
    nothing is ever written in place. ``spec`` is the capability spec the
    dataset is built for and is REQUIRED in practice (the CLI enforces it):
    the manifest records the capability id, the teacher pin and the spec
    fingerprint so every downstream consumer (distillation, search,
    certification) can verify dataset identity instead of trusting a path.
    A dataset without an identity binding cannot back any training pairs.
    """
    if spec is None:
        raise DatasetInvalid(
            "dataset build requires the capability spec (--spec); without "
            "it the manifest cannot bind capability/teacher identity and "
            "no downstream consumer may trust the splits"
        )
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise DatasetInvalid(
            "output %s already exists; a dataset build is a new exclusive path"
            % output_dir
        )
    validated = [validate_case(dict(c)) for c in cases]
    by_split: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SPLITS}
    for case in validated:
        by_split[case["split"]].append(case)
    missing = [s for s in SPLITS if not by_split[s]]
    if missing:
        raise DatasetInvalid("splits with no cases: %s" % ", ".join(missing))
    _check_disjointness(by_split)
    output_dir.mkdir(parents=True)
    files: Dict[str, str] = {}
    for split in SPLITS:
        path = output_dir / ("%s.jsonl" % split)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for case in by_split[split]:
                record = dict(case)
                record["content_sha256"] = _content_hash(case)
                record["schema"] = DATASET_SCHEMA
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        files["%s.jsonl" % split] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": 1,
        "capability_id": spec.capability_id,
        "teacher": {
            "provider": spec.teacher.provider,
            "model": spec.teacher.model,
            "revision": spec.teacher.revision,
            "access": spec.teacher.access,
        },
        "spec_fingerprint": spec_fingerprint(spec),
        "counts": {s: len(by_split[s]) for s in SPLITS},
        "groups": {
            s: {
                g: sum(1 for c in by_split[s] if c["group"] == g)
                for g in _GROUP_RE
            }
            for s in SPLITS
        },
        "families": {
            s: sorted({c["family_id"] for c in by_split[s]}) for s in SPLITS
        },
        "licenses": sorted({c["license"] for c in validated}),
        "files": files,
        "leakage_checks": [
            "cross-split sample_id disjointness",
            "cross-split exact content hash disjointness",
            "cross-split family disjointness",
            "within-split id/content uniqueness",
            "near-duplicate token-shape guard (within and across splits)",
            "frozen September sets quarantined as inputs",
        ],
        "frozen_before_model_generation": True,
        "model_outputs_consulted": False,
        "final_opened": False,
        "note": (
            "cases are data only; building this dataset involved no model "
            "runs and no grading, and confers no capability on anything"
        ),
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    lock = {
        "schema": SELECTION_LOCK_SCHEMA,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "frozen_before_model_generation": True,
        "model_outputs_consulted": False,
        "final_opened": False,
    }
    with (output_dir / "selection-lock.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return manifest


def validate_dataset(dataset_dir: Path) -> Dict[str, Any]:
    """Re-read a built dataset and re-check every invariant and file hash.

    The selection lock is REQUIRED, not optional: a dataset directory whose
    ``selection-lock.json`` is missing was never frozen by a build (or the
    lock was deleted), and its manifest cannot be trusted -- the lock's
    ``manifest_sha256`` must match the manifest byte for byte, and the
    frozen flags it pins must hold. A manifest that changed after the
    lock is a tampered dataset, refused here."""
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise DatasetInvalid("no manifest.json in %s" % dataset_dir)
    lock_path = dataset_dir / "selection-lock.json"
    if not lock_path.is_file():
        raise DatasetInvalid(
            "no selection-lock.json in %s; a dataset without its selection "
            "lock was never frozen by a build (or the lock was removed), "
            "and may not back any training pairs" % dataset_dir
        )
    manifest_bytes = manifest_path.read_bytes()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != SELECTION_LOCK_SCHEMA:
        raise DatasetInvalid("selection lock is not %s" % SELECTION_LOCK_SCHEMA)
    if lock.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
        raise DatasetInvalid(
            "selection-lock manifest_sha256 does not match manifest.json; "
            "the manifest changed after the dataset was frozen"
        )
    if not lock.get("frozen_before_model_generation"):
        raise DatasetInvalid(
            "selection lock does not record frozen_before_model_generation"
        )
    if lock.get("model_outputs_consulted"):
        raise DatasetInvalid(
            "selection lock records model_outputs_consulted: true; cases "
            "selected with model outputs consulted may never be teaching "
            "material"
        )
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise DatasetInvalid("manifest is not %s" % MANIFEST_SCHEMA)
    for identity_field in ("capability_id", "teacher", "spec_fingerprint"):
        if not manifest.get(identity_field):
            raise DatasetInvalid(
                "manifest is missing the identity field %r; a dataset that "
                "cannot prove which capability and teacher it was built for "
                "may not back any training pairs" % identity_field
            )
    by_split: Dict[str, List[Dict[str, Any]]] = {}
    for split in SPLITS:
        path = dataset_dir / ("%s.jsonl" % split)
        if not path.is_file():
            raise DatasetInvalid("split file %s missing" % path.name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if manifest["files"].get(path.name) != digest:
            raise DatasetInvalid(
                "manifest hash mismatch for %s; the dataset changed after "
                "the build" % path.name
            )
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        if len(rows) != manifest["counts"][split]:
            raise DatasetInvalid(
                "count mismatch in %s (manifest %d, file %d)"
                % (split, manifest["counts"][split], len(rows))
            )
        by_split[split] = rows
    _check_disjointness(by_split)
    for split in SPLITS:
        for row in by_split[split]:
            if row.get("content_sha256") != _content_hash(row):
                raise DatasetInvalid(
                    "content hash mismatch at sample %s" % row.get("sample_id")
                )
    return {
        "ok": True,
        "capability_id": manifest["capability_id"],
        "teacher": manifest["teacher"],
        "spec_fingerprint": manifest["spec_fingerprint"],
        "counts": manifest["counts"],
        "leakage_checks": manifest["leakage_checks"],
        "final_opened": manifest["final_opened"],
        "selection_lock_verified": True,
    }