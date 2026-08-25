"""Flow C -- coding skill transfer.

Sender:   MOCK stronger coding source (solved bug-fix traces)
Receiver: MOCK Qwen with a weaker fix rate
Goal:     transfer reusable buggy->fixed fragment patterns that apply to lines
          never seen during extraction.

Honest note on scoring: the metric compares text, it does not run tests. One
held-out case (hd04) is intentionally not covered by any extracted pattern, so
the candidate score should land clearly below 1.0. A demo that scored perfectly
would mean the benchmark was rigged.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _common import demo_pipeline, knowledge_from, show, suite

from asea.modules.mock.zoo import code_cap, make_generic_sender, make_qwen


def main(workspace: Path) -> None:
    bugs = suite("coding_bugfix")

    sender = make_generic_sender(
        module_id="strong-coder-mock",
        display_name="Stronger coding source (MOCK)",
        capabilities=[code_cap()],
        knowledge=knowledge_from([bugs], splits=("extraction",)),
    )
    receiver = make_qwen(knowledge=None, fallback="echo")

    pipeline = demo_pipeline(workspace)
    pipeline.register_module(sender)
    pipeline.register_module(receiver)
    pipeline.bind_adapter(
        "coder-to-qwen", sender.module_id, receiver.module_id,
        description="Bug-fix pattern transfer",
    )

    report = pipeline.run("coder-to-qwen", suites=[bugs])
    show(report, "FLOW C -- coding skill transfer (all modules MOCK)")

    print("\nextracted fix patterns:")
    for packet in report.distilled:
        for ex in (packet.distilled_skill or {}).get("examples", []):
            print("   pattern  {!r} -> {!r}  (test-verified: {})".format(
                ex["buggy"], ex["fixed"], ex["verified_by_tests"]))

    print("\nmemory store: {}".format(pipeline.store.stats()))
    print("audit chain : {}".format(pipeline.audit.verify()))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        main(Path(tmp))
