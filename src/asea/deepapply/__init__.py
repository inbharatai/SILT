"""DEEP-APPLY -- native gated LoRA trainer.

SILT trains a LoRA adapter on the receiver using ONLY packets that already
passed Gate 1, then gates the trained adapter AGAIN (Gate 2) with the same
14-check discipline before admission. The double-gate is the invention; it is
never collapsed.

Public surface (all importable without torch/transformers/peft installed -- the
heavy deps are imported lazily inside the real trainer backends)::

    DeepApplyRunner          orchestrator: build -> train -> evaluate -> Gate 2 -> admit
    DeepApplyConfig          hyper-parameters + policy overrides (no gate-weakening knobs)
    DeepApplyReport          the result of one run
    DeepApplyGate / DeepApplyPolicy      Gate 2 (reuses Check/GateDecision machinery)
    DeepApplyEvaluator       held-out A/B + regression sweep (reuses BenchmarkHarness)
    AdapterPacket            the typed record of a trained adapter
    AdapterStore             separated candidate/approved/rejected + rollback
    build_training_dataset   promoted packets -> traceable training set (intake guards)
    TrainerBackend / get_backend / StandardTrainerBackend / StreamedTrainerBackend
    deep_apply(...)          one-shot convenience
    from_pipeline(pipeline)  share a pipeline's store/audit/harness

Errors (typed, named -- nothing returns a fake success)::

    DeepApplyError           base
    DeepApplyIntakeError     a packet was refused at intake (not PROMOTED, or mock)
    DeepApplyBlocked         cannot run here: missing [deep] extra, no CUDA for a big
                             model, or an unsupported architecture. Names the remedy.
    AdapterNotPromoted       attempted to admit an adapter that did not pass Gate 2
"""

from __future__ import annotations

from .adapter_packet import AdapterPacket, max_risk_tier
from .dataset import TrainingDataset, build_training_dataset
from .errors import (
    AdapterNotPromoted,
    DeepApplyBlocked,
    DeepApplyError,
    DeepApplyIntakeError,
)
from .evaluator import DeepApplyEvaluationReport, DeepApplyEvaluator
from .gate2 import DeepApplyGate, DeepApplyPolicy
from .runner import DeepApplyConfig, DeepApplyReport, DeepApplyRunner, deep_apply, from_pipeline
from .store import AdapterRollbackLayer, AdapterStore
from .trainer import (
    AdapterArtifact,
    CPU_PARAM_CEILING,
    STREAMING_CREDIT,
    StandardTrainerBackend,
    StreamedTrainerBackend,
    TrainerBackend,
    get_backend,
)

__all__ = [
    "AdapterArtifact",
    "AdapterNotPromoted",
    "AdapterPacket",
    "AdapterRollbackLayer",
    "AdapterStore",
    "CPU_PARAM_CEILING",
    "DeepApplyBlocked",
    "DeepApplyConfig",
    "DeepApplyError",
    "DeepApplyEvaluationReport",
    "DeepApplyEvaluator",
    "DeepApplyGate",
    "DeepApplyIntakeError",
    "DeepApplyPolicy",
    "DeepApplyReport",
    "DeepApplyRunner",
    "STREAMING_CREDIT",
    "StandardTrainerBackend",
    "StreamedTrainerBackend",
    "TrainerBackend",
    "TrainingDataset",
    "build_training_dataset",
    "deep_apply",
    "from_pipeline",
    "get_backend",
    "max_risk_tier",
]