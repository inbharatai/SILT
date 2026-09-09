"""Additive local composition v1. Importing this module never imports torch/transformers."""
from .schema import ComponentManifest, CompositionSpec, EvaluationSuite, load_json
from .runtime import inspect_spec, run_spec

__all__ = ["ComponentManifest", "CompositionSpec", "EvaluationSuite", "load_json", "inspect_spec", "run_spec", "main"]


def main(argv=None):
    from .__main__ import main as cli_main
    return cli_main(argv)
