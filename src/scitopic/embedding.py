"""Text embedding for scientific papers.

Each paper is encoded from three views — title, abstract, and a metadata
sentence — producing a stacked embedding used by downstream clustering and
sampling.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np


def _metadata_sentence(meta: dict) -> str:
    """Render a single metadata record as a natural-language sentence."""
    return (
        f"Author: {meta.get('authors', '')}, "
        f"Year: {meta.get('year', '')}, "
        f"Venue: {meta.get('conference', '')}"
    )


class TextEmbedding:
    """Encode papers with a BGE-M3 or SentenceTransformer backend.

    Args:
        backend: ``"bge"`` selects ``FlagEmbedding.BGEM3FlagModel``; any other
            value selects a ``sentence_transformers.SentenceTransformer``.
        model_path: Path to (or HuggingFace id of) the embedding model.
        device: Torch device string, e.g. ``"cuda"`` or ``"cpu"``.
        use_fp16: Whether to load the BGE backend in half precision.
    """

    def __init__(
        self,
        backend: str = "bge",
        model_path: Optional[str] = None,
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        if model_path is None:
            raise ValueError("model_path must be provided")

        self.backend = backend
        if backend == "bge":
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(model_path, use_fp16=use_fp16, device=device)
        else:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_path, device=device)

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into a dense matrix."""
        if self.backend == "bge":
            return self._model.encode(texts)["dense_vecs"]
        return np.asarray(self._model.encode(texts))

    def encode(
        self,
        titles: list[str],
        abstracts: list[str],
        metadata: list[dict],
        save_dir: Optional[str] = None,
    ) -> tuple[np.ndarray, list[dict]]:
        """Encode papers into stacked (title, abstract, metadata) embeddings.

        Args:
            titles: Paper titles.
            abstracts: Paper abstracts, aligned with ``titles``.
            metadata: Per-paper metadata dicts with ``authors``/``year``/
                ``conference`` keys, aligned with ``titles``.
            save_dir: If given, ``embedding.npy`` and ``metadata.json`` are
                written there.

        Returns:
            A tuple of the embedding array with shape ``(n_papers, 3, dim)`` and
            the list of per-paper info dicts.
        """
        if not (len(titles) == len(abstracts) == len(metadata)):
            raise ValueError("titles, abstracts and metadata must have equal length")

        embeddings: list[np.ndarray] = []
        info: list[dict] = []
        for idx, title in enumerate(titles):
            meta_sentence = _metadata_sentence(metadata[idx])
            info.append(
                {"title": title, "abstract": abstracts[idx], "metadata": meta_sentence}
            )
            embeddings.append(self._encode([title, abstracts[idx], meta_sentence]))

        stacked = np.array(embeddings)
        if save_dir is not None:
            self.save(stacked, info, save_dir)
        return stacked, info

    @staticmethod
    def save(embedding: np.ndarray, info: list[dict], save_dir: str) -> None:
        """Persist an embedding array and its info sidecar to ``save_dir``."""
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "embedding.npy"), embedding)
        with open(os.path.join(save_dir, "info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4)
