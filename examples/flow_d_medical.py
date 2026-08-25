"""Flow D -- medical domain transfer, SAFETY RESTRICTED.

Sender:   MOCK verified triage corpus
Receiver: MOCK small medical assistant
Goal:     transfer red-flag escalation rules -- and demonstrate that the system
          REFUSES to promote them autonomously.

What this flow is really testing is the refusal. Even with every score passing,
a medical packet lands in PENDING_HUMAN and stays there until a named approver
acts. The second half of the script shows the human approval step and the audit
record it leaves.

NOT MEDICAL ADVICE. The rules in the sample suite have not been reviewed by any
clinician. See risk_report.md.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _common import demo_pipeline, knowledge_from, show, suite

from asea.core.protocol import Domain, PromotionStatus
from asea.modules.mock.zoo import make_generic_receiver, make_generic_sender, rule_cap


def main(workspace: Path) -> None:
    triage = suite("medical_triage")
    cap = [rule_cap(Domain.MEDICAL, "triage")]

    sender = make_generic_sender(
        module_id="triage-corpus-mock",
        display_name="Verified triage corpus (MOCK)",
        capabilities=cap,
        knowledge=knowledge_from([triage], splits=("extraction",)),
    )
    receiver = make_generic_receiver(
        module_id="small-medical-assistant-mock",
        display_name="Small medical assistant (MOCK)",
        capabilities=cap,
        fallback="english",
    )

    pipeline = demo_pipeline(workspace)
    pipeline.register_module(sender)
    pipeline.register_module(receiver)
    pipeline.bind_adapter(
        "triage-to-assistant", sender.module_id, receiver.module_id,
        description="Red-flag triage rules -- human approval mandatory",
    )

    report = pipeline.run("triage-to-assistant", suites=[triage])
    show(report, "FLOW D -- medical transfer, human approval required (MOCK)")

    if not report.pending_human:
        print("\nUNEXPECTED: no packet was held for human approval.")
        return

    print("\n{} packet(s) parked awaiting a named approver. Nothing is live yet.".format(
        len(report.pending_human)))
    print("approved skills visible to receiver right now: {}".format(
        len(pipeline.store.approved_skills(receiver.module_id))))

    packet_id = report.pending_human[0]
    decision = pipeline.approve_pending(packet_id, approver="dr.reviewer@example.org")
    print("\nafter human approval by dr.reviewer@example.org:")
    print("   status: {}".format(decision["status"]))
    print("   approved skills now visible: {}".format(
        len(pipeline.store.approved_skills(receiver.module_id))))

    trail = pipeline.audit.for_packet(packet_id)
    print("\naudit trail for this packet:")
    for entry in trail:
        print("   {:<24} actor={}".format(entry["event"], entry["actor"]))
    print("audit chain : {}".format(pipeline.audit.verify()))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        main(Path(tmp))
