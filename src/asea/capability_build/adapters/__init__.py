"""Adapters for MoE teacher instrumentation.

``switch`` wraps the existing compiler telemetry contracts; ``glm53_flash``
runs only inside the isolated worker (Transformers 5.x). Import discipline:
this package's ``__init__`` imports nothing -- adapters are imported lazily
so the bare ``import asea`` sanity gate never touches torch.
"""