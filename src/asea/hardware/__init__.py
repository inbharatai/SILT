"""Local hardware discovery and metadata-only specialist feasibility planning.

Importing this module does not import Torch, Transformers, PEFT or CUDA.
"""
from .probe import probe_hardware
from .planning import plan_specialist

__all__ = ['probe_hardware', 'plan_specialist']
