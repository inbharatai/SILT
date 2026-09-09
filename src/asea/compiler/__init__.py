"""Local, safetensors-only Switch structural expert-pruning compiler.

Heavy ML dependencies are imported only when running a model. Nothing here
implements capability transfer, repair training, or candidate admission.
"""
from .core import CompilerError, inspect_model, prune_model, evaluate_model, infer_model

__all__ = ["CompilerError", "inspect_model", "prune_model", "evaluate_model", "infer_model"]
