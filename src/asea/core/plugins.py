"""Plugin resolution.

This is what keeps the core universal. The pipeline never writes
``if modality == Modality.CODE``; it asks this registry for the extractor,
distiller and metric bound to whatever modality the packet declares. Adding a
new modality is a registration call, not a core edit.
"""

from __future__ import annotations

from typing import Dict, Optional, Type

from .errors import PluginNotFound
from .interfaces import Distiller, Extractor, MetricPlugin
from .protocol import Modality


class PluginRegistry:
    def __init__(self) -> None:
        self._extractors: Dict[Modality, Extractor] = {}
        self._distillers: Dict[Modality, Distiller] = {}
        self._metrics: Dict[Modality, MetricPlugin] = {}

    # -- registration -----------------------------------------------------

    def register_extractor(self, extractor: Extractor) -> None:
        self._extractors[extractor.modality] = extractor

    def register_distiller(self, distiller: Distiller) -> None:
        self._distillers[distiller.modality] = distiller

    def register_metric(self, metric: MetricPlugin) -> None:
        self._metrics[metric.modality] = metric

    # -- resolution -------------------------------------------------------

    def extractor(self, modality: Modality) -> Extractor:
        try:
            return self._extractors[modality]
        except KeyError:
            raise PluginNotFound(
                "no extractor registered for modality '{}'".format(modality.value)
            )

    def distiller(self, modality: Modality) -> Distiller:
        try:
            return self._distillers[modality]
        except KeyError:
            raise PluginNotFound(
                "no distiller registered for modality '{}'".format(modality.value)
            )

    def metric(self, modality: Modality) -> Optional[MetricPlugin]:
        """Metrics are optional; the universal evaluator has a fallback."""
        return self._metrics.get(modality)

    def modalities(self) -> Dict[str, Dict[str, bool]]:
        """Introspection used by the CLI and the conformance test."""
        keys = set(self._extractors) | set(self._distillers) | set(self._metrics)
        return {
            m.value: {
                "extractor": m in self._extractors,
                "distiller": m in self._distillers,
                "metric": m in self._metrics,
            }
            for m in sorted(keys, key=lambda x: x.value)
        }


def default_registry() -> PluginRegistry:
    """Registry with the bundled plugins wired in.

    Imported lazily to avoid a circular import at module load.
    """
    from ..distill.strategies import (
        CodeDistiller,
        StructuredDistiller,
        TextDistiller,
        TTSDistiller,
    )
    from ..evaluator.metrics_plugins import CodeMetric, TextMetric, TTSMetric
    from ..extraction.extractors import (
        CodeExtractor,
        StructuredExtractor,
        TextExtractor,
        TTSExtractor,
    )

    reg = PluginRegistry()
    for ex in (TextExtractor(), TTSExtractor(), CodeExtractor(), StructuredExtractor()):
        reg.register_extractor(ex)
    for di in (TextDistiller(), TTSDistiller(), CodeDistiller(), StructuredDistiller()):
        reg.register_distiller(di)
    for me in (TextMetric(), TTSMetric(), CodeMetric()):
        reg.register_metric(me)
    return reg
