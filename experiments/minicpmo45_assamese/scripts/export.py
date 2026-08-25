"""Export a job's approved skill packets as the SILT "trained model" bundle.

Thin wrapper around src/asea/distill/export.py:export_artifact_bundle. Reads the
approved packets from .studio/<job_id>/memory/approved, writes a downloadable
zip (approved packets + manifest + dataset.jsonl + dataset manifest) and — when
--base_model is given — a NOT_EXECUTED L4 LoRA job spec a human runs on a GPU
box. SILT trains no weights; this is the recipe + data, not a trained adapter.

Runs on CPU (it only reads JSON packets). Reuses the production export path —
no parallel infrastructure.

  PYTHONPATH=src python experiments/minicpmo45_assamese/scripts/export.py \
      --job_id e0ebb0553635 \
      --out experiments/minicpmo45_assamese/handoff/C_g2p \
      --base_model openbmb/MiniCPM-o-2_6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from asea.distill.export import export_artifact_bundle  # noqa: E402
from asea.memory.store import MemoryStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_id", required=True)
    ap.add_argument("--out", required=True, help="output directory for the bundle")
    ap.add_argument("--base_model", default=None,
                    help="if set, also emit a NOT_EXECUTED L4 job spec targeting this model")
    ap.add_argument("--name", default="silt_skill_bundle")
    args = ap.parse_args()

    store = MemoryStore(REPO / ".studio" / args.job_id / "memory")
    packets = store.list("approved")
    if not packets:
        print("no approved packets for job {}".format(args.job_id))
        sys.exit(1)

    out_dir = Path(args.out)
    zip_path = export_artifact_bundle(
        packets, out_dir, name=args.name, base_model=args.base_model,
    )
    print("wrote {} ({} approved packets, base_model={})".format(
        zip_path, len(packets), args.base_model or "none (L0-L3 bundle only)"))


if __name__ == "__main__":
    main()