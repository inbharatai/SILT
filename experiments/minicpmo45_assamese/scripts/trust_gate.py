"""Read-only trust-gate summarizer for a SILT job.

Replays a job's hash-chained audit log + packet buckets and prints the gate
verdicts without mutating anything. Reuses TransferJob.events / the job's
pipeline store — no parallel infrastructure. CPU-feasible.

  PYTHONPATH=src python experiments/minicpmo45_assamese/scripts/trust_gate.py \
      --job_id e0ebb0553635
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from asea.studio.jobs import JobManager  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_id", required=True)
    args = ap.parse_args()
    manager = JobManager(REPO / ".studio")
    try:
        job = manager.get(args.job_id)
    except KeyError as exc:
        # The job may be from a prior session; fall back to reading its audit log
        # + approved packets directly from disk.
        print("live job not found ({}); reading on-disk artefacts...".format(exc))
        audit = REPO / ".studio" / args.job_id / "audit" / "audit.jsonl"
        if audit.exists():
            for line in audit.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    print(line)
        else:
            print("no audit log at {}".format(audit))
        return

    for ev in job.events():
        print("[{}] {}".format(ev.get("index"), ev.get("event") or ev.get("kind")))
    print("status:", job.status, "error:", job.error)


if __name__ == "__main__":
    main()