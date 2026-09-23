"""Assemble the E2 dataset input file and verify the split-pinning rules.

Builds ``experiments/glm53_flash_e2/cases.jsonl`` from:

  * the six v1 TRAINING rows (byte-for-byte from data/capability_v1/
    training.jsonl -- their receipted teacher traces remain valid teaching
    material and their prompts must not change),
  * the sixteen NEW training rows (authored by authoring_e2.py),
  * the v1 development / heldout / final / controls rows, byte-for-byte,
    in original order.

After ``silt-capability dataset build`` runs, this script's ``verify``
mode re-reads data/capability_v2 and checks, per evaluation split, that
the file sha256 equals the hash recorded in the v1 manifest -- proof that
the frozen evaluation material is bit-identical and the v1 transfer
measurement is directly comparable.

Usage:
  python experiments/glm53_flash_e2/assemble_e2_dataset.py assemble
  python experiments/glm53_flash_e2/assemble_e2_dataset.py verify
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
V1 = REPO / "data" / "capability_v1"
V2 = REPO / "data" / "capability_v2"
OUT = HERE / "cases.jsonl"

SPLITS = ("training", "development", "heldout", "final", "controls")


def _rows(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    v1_manifest = json.loads((V1 / "manifest.json").read_text("utf-8"))
    if mode == "assemble":
        # v1 rows verbatim, preserving within-split order
        rows = {"training": _rows(V1 / "training.jsonl"),
                "development": _rows(V1 / "development.jsonl"),
                "heldout": _rows(V1 / "heldout.jsonl"),
                "final": _rows(V1 / "final.jsonl"),
                "controls": _rows(V1 / "controls.jsonl")}
        v1_training_ids = [r["sample_id"] for r in rows["training"]]
        assert len(v1_training_ids) == len(set(v1_training_ids))
        new_rows = _rows(HERE / "cases-new-training.jsonl")
        assert all(r["split"] == "training" for r in new_rows)
        assert not ({r["sample_id"] for r in new_rows}
                    & set(v1_training_ids)), "new id collides with v1"
        rows["training"] = rows["training"] + new_rows
        with OUT.open("w", encoding="utf-8", newline="\n") as handle:
            for split in SPLITS:
                for row in rows[split]:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        print("WROTE %s" % OUT)
        print("  training   %d (6 v1 + %d new)"
              % (len(rows["training"]), len(new_rows)))
        for split in SPLITS[1:]:
            print("  %-10s %d" % (split, len(rows[split])))
        return 0
    if mode == "verify":
        v2_manifest = json.loads((V2 / "manifest.json").read_text("utf-8"))
        problems = []
        for split in SPLITS[1:]:
            digest = hashlib.sha256(
                (V2 / ("%s.jsonl" % split)).read_bytes()).hexdigest()
            recorded_v2 = v2_manifest["files"]["%s.jsonl" % split]
            recorded_v1 = v1_manifest["files"]["%s.jsonl" % split]
            if digest != recorded_v2:
                problems.append("%s: v2 file does not match its own manifest"
                                % split)
            if digest != recorded_v1:
                problems.append(
                    "%s: NOT a byte-copy of v1 (hash differs from the frozen "
                    "v1 manifest)" % split)
        # training split: v1's six rows must appear unmodified inside it
        v2_training = {r["sample_id"]: r
                       for r in _rows(V2 / "training.jsonl")}
        v1_training = _rows(V1 / "training.jsonl")
        for row in v1_training:
            other = v2_training.get(row["sample_id"])
            if other != row:
                problems.append(
                    "v1 training row %s changed inside the v2 training split"
                    % row["sample_id"])
        counts = v2_manifest["counts"]
        print("v2 counts:", counts)
        if problems:
            print("VERIFY FAILED:")
            for p in problems:
                print("  -", p)
            return 1
        print("VERIFIED: all four evaluation splits are byte-identical to "
              "the frozen v1 dataset; the six v1 training rows are "
              "unmodified inside the expanded training split")
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())