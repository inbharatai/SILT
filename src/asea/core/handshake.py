"""Connection handshake.

Two modules do not simply get wired together. They exchange manifests, the
adapter validates compatibility, and only then is a :class:`Session` issued.
The session id is what ties every subsequent packet, audit entry and rollback
snapshot to one negotiated connection.

Compatibility checks performed here (all of which have bitten real pipelines):
  * the pair must overlap on at least one modality, else there is nothing to say
  * the receiver must be able to consume skills at the requested level
  * a mock on either side degrades the session to non-strict, which the
    promotion gate then refuses to auto-promote from
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .errors import HandshakeError
from .interfaces import ModuleAdapter
from .protocol import CapabilityManifest, LearningLevel, Modality


class Session:
    """An established sender -> receiver connection."""

    def __init__(
        self,
        session_id: str,
        adapter_id: str,
        sender: ModuleAdapter,
        receiver: ModuleAdapter,
        sender_manifest: CapabilityManifest,
        receiver_manifest: CapabilityManifest,
        shared_modalities: List[Modality],
        negotiated_level: LearningLevel,
    ) -> None:
        self.session_id = session_id
        self.adapter_id = adapter_id
        self.sender = sender
        self.receiver = receiver
        self.sender_manifest = sender_manifest
        self.receiver_manifest = receiver_manifest
        self.shared_modalities = shared_modalities
        self.negotiated_level = negotiated_level
        self.opened_at = datetime.now(timezone.utc).isoformat()

    @property
    def involves_mock(self) -> bool:
        return bool(self.sender.is_mock or self.receiver.is_mock)

    def to_dict(self) -> Dict[str, object]:
        return {
            "session_id": self.session_id,
            "adapter_id": self.adapter_id,
            "sender": self.sender.module_id,
            "receiver": self.receiver.module_id,
            "shared_modalities": [m.value for m in self.shared_modalities],
            "negotiated_level": int(self.negotiated_level),
            "involves_mock": self.involves_mock,
            "opened_at": self.opened_at,
        }

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<Session {} {} -> {}>".format(
            self.session_id[:8], self.sender.module_id, self.receiver.module_id
        )


class Handshake:
    def open(
        self,
        adapter_id: str,
        sender: ModuleAdapter,
        receiver: ModuleAdapter,
        requested_level: LearningLevel = LearningLevel.L3_SKILL_PACKET,
    ) -> Session:
        if sender.module_id == receiver.module_id:
            raise HandshakeError(
                "refusing self-transfer for '{}': training a module on its own "
                "output is the canonical model-collapse loop".format(sender.module_id)
            )

        s_manifest = sender.manifest()
        r_manifest = receiver.manifest()

        if "sender" not in s_manifest.roles:
            raise HandshakeError(
                "module '{}' does not declare the sender role".format(sender.module_id)
            )
        if "receiver" not in r_manifest.roles:
            raise HandshakeError(
                "module '{}' does not declare the receiver role".format(receiver.module_id)
            )

        s_modalities = {c.modality for c in s_manifest.capabilities}
        r_modalities = {c.modality for c in r_manifest.capabilities}
        shared = sorted(s_modalities & r_modalities, key=lambda m: m.value)
        if not shared:
            raise HandshakeError(
                "no shared modality between '{}' ({}) and '{}' ({}); there is "
                "nothing transferable".format(
                    sender.module_id,
                    sorted(m.value for m in s_modalities),
                    receiver.module_id,
                    sorted(m.value for m in r_modalities),
                )
            )

        negotiated = LearningLevel(
            min(
                int(requested_level),
                int(r_manifest.max_learning_level),
                int(s_manifest.max_learning_level),
            )
        )

        return Session(
            session_id=str(uuid.uuid4()),
            adapter_id=adapter_id,
            sender=sender,
            receiver=receiver,
            sender_manifest=s_manifest,
            receiver_manifest=r_manifest,
            shared_modalities=list(shared),
            negotiated_level=negotiated,
        )
