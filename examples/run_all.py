"""Run all four demonstration flows in sequence.

    python3 examples/run_all.py

Every module involved is a MOCK. The numbers describe pipeline behaviour, not
model capability.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import flow_a_assamese
import flow_b_tts
import flow_c_coding
import flow_d_medical

FLOWS = [
    ("A  Assamese language transfer", flow_a_assamese.main),
    ("B  TTS pronunciation transfer", flow_b_tts.main),
    ("C  Coding skill transfer", flow_c_coding.main),
    ("D  Medical transfer (human approval)", flow_d_medical.main),
]


def main() -> None:
    for label, fn in FLOWS:
        print("\n\n########## FLOW {} ##########".format(label))
        with tempfile.TemporaryDirectory() as tmp:
            fn(Path(tmp))
    print("\n\nAll four flows completed. Reminder: every module was a MOCK.")


if __name__ == "__main__":
    main()
