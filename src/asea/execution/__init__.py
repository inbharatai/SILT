"""Hard, explicitly scoped resource controls for fixed trusted SILT workers.

This is not a candidate-code sandbox. No ML dependency is imported here.
"""
from .controls import probe, run

__all__ = ["probe", "run"]
