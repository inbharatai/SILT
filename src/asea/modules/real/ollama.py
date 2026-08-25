"""Real connector for a local Ollama server.

REAL: ``is_mock = False``.

This is the connector to reach for on a laptop. Ollama holds the weights, does
its own quantisation and memory management, and exposes an HTTP API -- so you can
run a 7B or 14B receiver without this process ever loading a tensor. It needs no
Python ML dependencies at all: only ``urllib`` from the standard library.

Setup::

    ollama serve
    ollama pull qwen2.5:7b-instruct
    ollama pull gemma2:9b-instruct

Determinism: ``temperature=0`` and a fixed ``seed`` are sent on every request.
Ollama is not bit-for-bit reproducible across versions or GPU backends, so treat
repeated runs as very-low-variance rather than identical, and do not read
meaning into improvements smaller than a couple of points.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from ...core.interfaces import ModuleAdapter
from ...core.protocol import CapabilityKey, CapabilityManifest, LearningLevel
from .prompting import build_messages


class OllamaConnectionError(RuntimeError):
    pass


class OllamaConnector(ModuleAdapter):
    is_mock = False

    def __init__(
        self,
        model: str,
        capabilities: List[CapabilityKey],
        roles: Optional[List[str]] = None,
        module_id: Optional[str] = None,
        display_name: Optional[str] = None,
        host: str = "http://localhost:11434",
        max_new_tokens: int = 128,
        seed: int = 0,
        timeout: int = 300,
        max_learning_level: LearningLevel = LearningLevel.L3_SKILL_PACKET,
        think: Optional[bool] = None,
    ) -> None:
        super().__init__(
            module_id or "ollama-" + model.replace(":", "-"),
            display_name or "Ollama {}".format(model),
        )
        self.model = model
        self._capabilities = list(capabilities)
        self._roles = list(roles or ["sender", "receiver"])
        self.host = host.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.timeout = timeout
        self.max_learning_level = max_learning_level
        # Reasoning models (Qwen3, GLM-thinking, Kimi, DeepSeek-R1, ...) emit
        # their chain-of-thought in a separate ``thinking`` field and only the
        # final answer in ``content``. With a small ``num_predict`` the model
        # spends every token thinking, ``done_reason`` becomes ``length``, and
        # ``content`` is empty -- so the connector silently returned ``""`` and
        # the learner looked "0 on everything" even when it could answer.
        # ``think=False`` asks the server to answer directly into ``content``.
        # ``None`` (default) sends no key, preserving the historical behaviour
        # for every existing connector (non-reasoning models ignore it anyway).
        self.think = think

    # -- identity ---------------------------------------------------------

    def manifest(self) -> CapabilityManifest:
        return CapabilityManifest(
            module_id=self.module_id,
            display_name=self.display_name,
            roles=self._roles,
            capabilities=self._capabilities,
            max_learning_level=self.max_learning_level,
            is_mock=False,
            version="ollama:{}".format(self.model),
        )

    # -- transport --------------------------------------------------------

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(
            "{}{}".format(self.host, path),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaConnectionError(
                "cannot reach Ollama at {}: {}. Is `ollama serve` running?".format(
                    self.host, exc
                )
            ) from exc

    def health(self) -> Dict[str, Any]:
        """Check the server is up and the model is present. Call before a run."""
        try:
            request = urllib.request.Request("{}/api/tags".format(self.host))
            with urllib.request.urlopen(request, timeout=10) as response:
                tags = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaConnectionError(
                "cannot reach Ollama at {}: {}".format(self.host, exc)
            ) from exc

        available = [m["name"] for m in tags.get("models", [])]
        # Exact match only. ``startswith`` would make ``qwen2.5:7b`` match a
        # pulled ``qwen2.5:7b-instruct`` -> a false-positive model_present, then
        # the subsequent /api/chat fails mid-sweep with "model not found".
        present = any(name == self.model for name in available)
        return {
            "host": self.host,
            "model": self.model,
            "model_present": present,
            "available_models": available,
            "hint": None if present else "run: ollama pull {}".format(self.model),
        }

    # -- inference --------------------------------------------------------

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0,
                "top_p": 1,
                "seed": self.seed,
                "num_predict": self.max_new_tokens,
            },
        }
        if self.think is not None:
            payload["think"] = self.think
        result = self._post("/api/chat", payload)
        msg = result.get("message") or {}
        content = msg.get("content", "")
        # Belt-and-braces: a reasoning model that ignored ``think=False`` (or a
        # connector built with ``think=None``) can leave ``content`` empty and
        # put the answer in ``thinking``. In that case surface the thinking
        # text rather than an empty string, so the evaluator sees *something*
        # instead of silently scoring a phantom zero.
        if not content and msg.get("thinking"):
            content = msg["thinking"]
        return self._clean(content)

    @staticmethod
    def _clean(text: str) -> str:
        text = (text or "").strip()
        if text.startswith("```"):
            text = "\n".join(
                l for l in text.splitlines() if not l.startswith("```")
            ).strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
            text = text[1:-1].strip()
        return text

    def infer(self, capability: CapabilityKey, prompt: Any) -> Any:
        return self._chat(build_messages(capability, prompt))

    def infer_with_skills(
        self, capability: CapabilityKey, prompt: Any, skills: List[Dict[str, Any]]
    ) -> Any:
        return self._chat(build_messages(capability, prompt, skills))
