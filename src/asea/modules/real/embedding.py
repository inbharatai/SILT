"""Real semantic similarity backends.

Replacing the bundled lexical proxy is the single highest-value upgrade in the
system. The lexical backend scores "I rice eat" against "I eat rice" at 1.0
because token-F1 ignores order; a sentence embedding does not make that mistake,
and it also recognises legitimate paraphrase, which the lexical backend punishes.

Two implementations:

* :class:`HFEmbeddingSimilarity` -- uses ``transformers`` directly with mean
  pooling. No ``sentence-transformers`` dependency needed.
* :class:`SentenceTransformerSimilarity` -- uses ``sentence-transformers`` when
  available, which applies each model's own trained pooling and normalisation.
  Slightly more faithful; one more dependency.

Both report ``is_semantic = True``, which flows into every evaluation report.

**Recalibrate after switching.** Thresholds tuned against edit distance are
meaningless against cosine similarity: embedding scores for unrelated text sit
around 0.3-0.5 rather than near 0, so a 0.75 correctness floor that was strict
becomes permissive. Re-tune ``RelevancePolicy`` and ``PromotionPolicy``.

Model suggestions for the Indic targets:
  ``sentence-transformers/LaBSE``                        strong, 1.8 GB
  ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``  fast, ~470 MB
  ``l3cube-pune/indic-sentence-similarity-sbert``        Indic-tuned
"""

from __future__ import annotations

from typing import Optional

from ...core.interfaces import SimilarityBackend
from ...evaluator.similarity import LexicalSimilarity


class HFEmbeddingSimilarity(SimilarityBackend):
    """Mean-pooled transformer embeddings, cosine similarity."""

    def __init__(
        self,
        model_id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "auto",
        max_length: int = 256,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._torch = None

    def load(self) -> "HFEmbeddingSimilarity":
        if self._model is not None:
            return self
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "HFEmbeddingSimilarity needs torch and transformers:\n"
                "  pip install torch transformers"
            ) from exc

        self._torch = torch
        device = (
            self.device
            if self.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id).to(device)
        self._model.eval()
        self.resolved_device = device
        return self

    def _embed(self, texts):
        torch = self._torch
        batch = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self._model.device)
        with torch.no_grad():
            output = self._model(**batch)
        hidden = output.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

    def similarity(self, a: str, b: str) -> float:
        a, b = str(a or ""), str(b or "")
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        self.load()
        vectors = self._embed([a, b])
        score = float((vectors[0] * vectors[1]).sum().item())
        # Cosine is in [-1, 1]; clamp negatives to 0 so downstream 0..1 contracts hold.
        return max(0.0, min(1.0, score))

    @property
    def is_semantic(self) -> bool:
        return True


class SentenceTransformerSimilarity(SimilarityBackend):
    """Uses each model's own trained pooling via ``sentence-transformers``."""

    def __init__(self, model_id: str = "sentence-transformers/LaBSE") -> None:
        self.model_id = model_id
        self._model = None
        self._util = None

    def load(self) -> "SentenceTransformerSimilarity":
        if self._model is not None:
            return self
        try:
            from sentence_transformers import SentenceTransformer, util
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SentenceTransformerSimilarity needs sentence-transformers:\n"
                "  pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(self.model_id)
        self._util = util
        return self

    def similarity(self, a: str, b: str) -> float:
        a, b = str(a or ""), str(b or "")
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        self.load()
        vectors = self._model.encode([a, b], convert_to_tensor=True)
        score = float(self._util.cos_sim(vectors[0], vectors[1]).item())
        return max(0.0, min(1.0, score))

    @property
    def is_semantic(self) -> bool:
        return True


def best_available_similarity(
    prefer: Optional[str] = None, quiet: bool = False
) -> SimilarityBackend:
    """Return the strongest similarity backend this machine can actually run.

    Degrades explicitly rather than silently: if it falls back to the lexical
    proxy it says so, because a run scored with a proxy must not be mistaken for
    a semantically evaluated one.
    """
    try:
        backend = SentenceTransformerSimilarity(
            prefer or "sentence-transformers/LaBSE"
        )
        backend.load()
        return backend
    except Exception:
        pass
    try:
        backend = HFEmbeddingSimilarity(
            prefer or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        backend.load()
        return backend
    except Exception as exc:
        if not quiet:
            print(
                "[asea] WARNING: no embedding backend available ({}). "
                "Falling back to LEXICAL similarity -- scores are a proxy, not "
                "semantic.".format(type(exc).__name__)
            )
        return LexicalSimilarity()
