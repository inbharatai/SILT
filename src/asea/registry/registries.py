"""Module / sender / receiver / adapter registries.

A module registers once. Its declared roles determine which of the sender and
receiver views it appears in; a module may legitimately be both (today's
receiver can be tomorrow's sender once it has been improved and verified).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..core.errors import RegistrationError
from ..core.interfaces import ModuleAdapter
from .base import BaseRegistry


class ModuleRegistry(BaseRegistry):
    kind = "module"

    def register_module(self, module: ModuleAdapter, replace: bool = False) -> ModuleAdapter:
        manifest = module.manifest()
        if manifest.module_id != module.module_id:
            raise RegistrationError(
                "manifest module_id '{}' does not match adapter id '{}'".format(
                    manifest.module_id, module.module_id
                )
            )
        if manifest.is_mock != module.is_mock:
            raise RegistrationError(
                "module '{}' misreports mock status; manifest says {}, adapter says {}".format(
                    module.module_id, manifest.is_mock, module.is_mock
                )
            )
        if not manifest.roles:
            raise RegistrationError(
                "module '{}' declares no roles; expected 'sender' and/or 'receiver'".format(
                    module.module_id
                )
            )
        unknown = set(manifest.roles) - {"sender", "receiver"}
        if unknown:
            raise RegistrationError(
                "module '{}' declares unknown roles: {}".format(
                    module.module_id, sorted(unknown)
                )
            )
        return self.register(module.module_id, module, replace=replace)

    def by_role(self, role: str) -> List[ModuleAdapter]:
        return [m for m in self.all() if role in m.manifest().roles]


class _RoleView:
    """Read-only projection of the module registry for one role."""

    role = ""

    def __init__(self, modules: ModuleRegistry) -> None:
        self._modules = modules

    def get(self, module_id: str) -> ModuleAdapter:
        module = self._modules.get(module_id)
        if self.role not in module.manifest().roles:
            raise RegistrationError(
                "module '{}' is not registered as a {}".format(module_id, self.role)
            )
        return module

    def ids(self) -> List[str]:
        return [m.module_id for m in self._modules.by_role(self.role)]

    def all(self) -> List[ModuleAdapter]:
        return self._modules.by_role(self.role)

    def __len__(self) -> int:
        return len(self.ids())


class SenderRegistry(_RoleView):
    role = "sender"


class ReceiverRegistry(_RoleView):
    role = "receiver"


class AdapterBinding:
    """A configured sender->receiver channel.

    Holding this as a first-class object (rather than passing two module ids
    around) is what lets policy, audit and rollback be scoped to a pair.
    """

    def __init__(
        self,
        adapter_id: str,
        sender_id: str,
        receiver_id: str,
        description: str = "",
        policy_overrides: Optional[Dict[str, object]] = None,
    ) -> None:
        self.adapter_id = adapter_id
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.description = description
        self.policy_overrides = policy_overrides or {}

    def to_dict(self) -> Dict[str, object]:
        return {
            "adapter_id": self.adapter_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "description": self.description,
            "policy_overrides": self.policy_overrides,
        }

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<AdapterBinding {}: {} -> {}>".format(
            self.adapter_id, self.sender_id, self.receiver_id
        )


class AdapterRegistry(BaseRegistry):
    kind = "adapter"

    def __init__(self, modules: ModuleRegistry) -> None:
        super().__init__()
        self._modules = modules

    def bind(
        self,
        adapter_id: str,
        sender_id: str,
        receiver_id: str,
        description: str = "",
        policy_overrides: Optional[Dict[str, object]] = None,
        replace: bool = False,
    ) -> AdapterBinding:
        if sender_id == receiver_id:
            # Self-transfer is the canonical model-collapse loop.
            raise RegistrationError(
                "sender and receiver must differ; self-transfer degrades the receiver"
            )
        sender = self._modules.get(sender_id)
        receiver = self._modules.get(receiver_id)
        if "sender" not in sender.manifest().roles:
            raise RegistrationError("module '{}' cannot act as sender".format(sender_id))
        if "receiver" not in receiver.manifest().roles:
            raise RegistrationError(
                "module '{}' cannot act as receiver".format(receiver_id)
            )
        binding = AdapterBinding(
            adapter_id, sender_id, receiver_id, description, policy_overrides
        )
        return self.register(adapter_id, binding, replace=replace)
