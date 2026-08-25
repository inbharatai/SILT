"""LayerBank -- where the frozen base layers live while they are not computing.

Two tiers:
  ram  : per-layer dicts of CPU tensors (the classic host-RAM tier)
  disk : per-layer files, loaded with torch.load(mmap=True) so the OS pages
         weights in lazily -- for machines whose RAM cannot hold the base
         either.

Fetching returns tensors ON THE COMPUTE DEVICE. On CUDA, fetch uses pinned
staging buffers and non_blocking copies when prefetch is enabled (that path
is written for correctness but is NOT exercised by the CPU-only test suite;
the parity harness must be re-run on a CUDA host before trusting it there).
"""

from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional

import torch

from .errors import StorageError


class LayerBank:
    def __init__(
        self,
        layer_states: List[Dict[str, torch.Tensor]],
        storage_tier: str = "ram",
        disk_dir: str = "",
    ) -> None:
        self.n_layers = len(layer_states)
        self.storage_tier = storage_tier
        self.disk_dir = disk_dir
        self._ram: List[Optional[Dict[str, torch.Tensor]]] = [None] * self.n_layers
        self._pinned: Dict[int, Dict[str, torch.Tensor]] = {}

        if storage_tier == "ram":
            # .clone() so the bank NEVER shares memory with caller tensors --
            # otherwise an in-place mutation outside corrupts the bank silently.
            for i, state in enumerate(layer_states):
                self._ram[i] = {
                    k: v.detach().to("cpu").clone() for k, v in state.items()
                }
        elif storage_tier == "disk":
            os.makedirs(disk_dir, exist_ok=True)
            for i, state in enumerate(layer_states):
                path = self._path(i)
                try:
                    torch.save({k: v.detach().to("cpu") for k, v in state.items()}, path)
                except OSError as exc:  # pragma: no cover - disk-full etc.
                    raise StorageError(f"could not write layer {i} to {path}: {exc}") from exc
        else:
            raise StorageError(f"unknown storage tier {storage_tier!r}")

    def _path(self, i: int) -> str:
        return os.path.join(self.disk_dir, f"layer_{i:04d}.pt")

    def fetch(self, i: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Return layer i's frozen weights on the compute device."""
        if not 0 <= i < self.n_layers:
            raise StorageError(f"layer index {i} out of range 0..{self.n_layers - 1}")
        if self.storage_tier == "ram":
            stored = self._ram[i]
            assert stored is not None
            # Return CLONES: callers must never receive handles into the
            # bank's own storage (isolation is what makes the parity harness
            # able to detect corruption -- verified by adversarial tests).
            state = {k: v.clone() for k, v in stored.items()}
        else:
            path = self._path(i)
            if not os.path.exists(path):
                raise StorageError(f"layer file missing: {path}")
            state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if device.type == "cpu":
            return state
        # CUDA/MPS path: move via pinned staging when possible.
        out: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            src = v
            if device.type == "cuda":
                key = i
                if key not in self._pinned:
                    self._pinned[key] = {}
                if k not in self._pinned[key] or self._pinned[key][k].shape != v.shape:
                    self._pinned[key][k] = torch.empty_like(v, pin_memory=True)
                self._pinned[key][k].copy_(v)
                src = self._pinned[key][k]
            out[k] = src.to(device, non_blocking=(device.type == "cuda"))
        return out

    def layers(self, device: torch.device) -> Iterator[Dict[str, torch.Tensor]]:
        for i in range(self.n_layers):
            yield self.fetch(i, device)

    def approx_bytes_resident(self) -> int:
        """Bytes the bank itself keeps in process RAM (0 for disk tier)."""
        if self.storage_tier != "ram":
            return 0
        total = 0
        for state in self._ram:
            assert state is not None
            total += sum(v.numel() * v.element_size() for v in state.values())
        return total
