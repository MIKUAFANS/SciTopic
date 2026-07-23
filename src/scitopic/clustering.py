"""Clustering of paper embeddings.

Wraps scikit-learn K-means and HDBSCAN behind a small, uniform interface so the
pipeline can swap algorithms via configuration.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans


class TextCluster:
    """Cluster embeddings with K-means or HDBSCAN.

    Args:
        method: ``"k_means"`` or ``"hdbscan"``.
        n_clusters: Number of clusters (required for K-means).
        random_state: Seed for K-means reproducibility.
        **kwargs: Extra keyword arguments forwarded to the HDBSCAN constructor.
    """

    def __init__(
        self,
        method: str = "k_means",
        n_clusters: Optional[int] = None,
        random_state: int = 2024,
        **kwargs,
    ):
        self.method = method
        self.n_clusters = n_clusters

        if method == "k_means":
            if n_clusters is None:
                raise ValueError("n_clusters must be provided for k_means")
            self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        elif method == "hdbscan":
            self.model = HDBSCAN(**kwargs)
        else:
            raise ValueError(f"Unknown clustering method: {method!r}")

    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit the model and return cluster labels."""
        self.model.fit(embeddings)
        return self.model.labels_

    @property
    def cluster_centers(self) -> Optional[np.ndarray]:
        """Cluster centroids, or ``None`` for HDBSCAN (density-based)."""
        if self.method == "hdbscan":
            return None
        return self.model.cluster_centers_

    def save(self, save_dir: str) -> None:
        """Persist labels (and centers for K-means) to ``save_dir``."""
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"labels_{self.method}_{self.n_clusters}.npy"), self.model.labels_)
        if self.method != "hdbscan":
            np.save(
                os.path.join(save_dir, f"centers_{self.method}_{self.n_clusters}.npy"),
                self.model.cluster_centers_,
            )

    def __call__(
        self, embeddings: np.ndarray, save_dir: Optional[str] = None
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit, optionally save, and return ``(labels, centers)``."""
        self.fit(embeddings)
        if save_dir is not None:
            self.save(save_dir)
        return self.model.labels_, self.cluster_centers
