"""SiltStream -- low-VRAM layer-streamed LoRA training with a built-in
bit-exact parity harness.

VENDORED INTO SILT 2026-08-16 (Uni Guru Technologies LLP, first-party code,
Apache-2.0). This directory is a verbatim copy of the standalone
``siltstream`` package (version 0.1.0) into ``src/asea/deepapply/backends/``
so the SILT trust product has no external runtime dependency for its streamed
trainer -- the code is auditable line-by-line in-tree. The standalone package
remains at ``../siltstream`` for independent development and its own test
suite; this copy is the one SILT's ``streamed`` backend imports.

Test status at vendor time (run from ``../siltstream``):
  python -m pytest tests -q  ->  63 passed, 1 skipped
(58 original + 5 adversarial tests added during SILT integration review; the
1 skip is the CUDA parity test, skipped on a CPU host). Re-run
``python -m pytest ../siltstream/tests -q`` from the SILT root to re-verify
the vendor stays green; this copy is byte-identical, so it shares that status.

Do NOT edit these files in-place as a SILT integration convenience: fixes
belong in the standalone package first, then are re-vendored verbatim, so the
two trees never silently diverge. The version and test status above are the
provenance of this copy.
---

SiltStream -- low-VRAM layer-streamed LoRA training with a built-in
bit-exact parity harness.

Original implementation (no third-party code). The general idea of training
with a frozen base held off-device and layers streamed to the accelerator
one at a time exists in public literature (e.g. layer-streaming fine-tuning
preprints, 2026); this package implements its own architecture for it:

  - pure-functional decoder blocks (identical code path resident/streamed)
  - streaming in BACKWARD as well, via per-layer recompute checkpointing
  - a parity harness (forward and backward verified as separate claims,
    bitwise on CPU fp32) intended as an admission check for trust layers
  - ram and disk storage tiers
  - deterministic seeds + config fingerprints for audit metadata

Honest limits (v1): embeddings/LM-head stay resident; no dropout (parity
first); the CUDA pinned-copy path is written but only correct-by-inspection
until the parity suite is re-run on a CUDA host; toy GPT-style architecture
-- porting the block contract to a production architecture (Llama/Qwen) is
the integration step, and parity must be re-verified there.
"""

from .bank import LayerBank
from .config import ModelConfig, StreamConfig, config_fingerprint
from .errors import (
    BackendUnavailableError,
    ParityError,
    SiltStreamError,
    StorageError,
    UnsupportedModelError,
)
from .model import StreamedCausalLM
from .parity import ParityReport, verify_parity
from .trainer import TrainReport, train_lora
from .zeroforge import ZeroForgeReport, train_zeroforge
from .quant import quantize_state, dequantize_state
from .spring import (
    BudgetError,
    SpringModel,
    StateCertificate,
    StateNotCertifiedError,
)

__version__ = "0.1.0"

__all__ = [
    "LayerBank",
    "ModelConfig",
    "StreamConfig",
    "config_fingerprint",
    "StreamedCausalLM",
    "verify_parity",
    "ParityReport",
    "train_lora",
    "train_zeroforge",
    "ZeroForgeReport",
    "SpringModel",
    "StateCertificate",
    "BudgetError",
    "StateNotCertifiedError",
    "quantize_state",
    "dequantize_state",
    "TrainReport",
    "SiltStreamError",
    "UnsupportedModelError",
    "StorageError",
    "ParityError",
    "BackendUnavailableError",
]
