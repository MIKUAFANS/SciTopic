"""Entropy-based sampling of representative paper triplets.

For each paper we estimate how ambiguous its cluster assignment is, using an
entropy over a soft distribution across the nearest cluster centers. Papers in a
chosen entropy band are turned into ``(query, choice_1, choice_2)`` triplets that
are later posed to an LLM to obtain topic-similarity preferences.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from tqdm import tqdm


def compute_entropy(
    cluster_centers: np.ndarray,
    embedding: np.ndarray,
    alpha: float = 1.0,
    epsilon: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-point entropy over the nearest cluster centers.

    A Student-t style kernel converts center distances to a probability
    distribution; the entropy of the top-``K`` closest centers measures how
    ambiguous a point's assignment is.

    Args:
        cluster_centers: Array of shape ``(n_clusters, dim)``.
        embedding: Array of shape ``(n_points, dim)``.
        alpha: Kernel degrees-of-freedom parameter.
        epsilon: Fraction of centers to keep as "closest" (at least 2).

    Returns:
        A tuple ``(entropy, closest)`` where ``entropy`` has shape
        ``(n_points,)`` and ``closest`` has shape ``(n_points, K)`` holding the
        indices of the K closest centers per point.
    """
    entropies: list[np.ndarray] = []
    closest: list[np.ndarray] = []
    k_closest = max(int(epsilon * len(cluster_centers)), 2)

    for point in tqdm(embedding, desc="entropy"):
        distance = np.linalg.norm(cluster_centers - point, axis=1)
        weight = (1 + distance / alpha) ** (-(alpha + 1) / 2)
        prob = weight / np.sum(weight)
        top = np.argsort(prob)[::-1][:k_closest]
        top_prob = prob[top]
        entropy = -np.sum(top_prob * np.log(top_prob))
        entropies.append(np.array(entropy))
        closest.append(np.array(top))

    return np.array(entropies), np.array(closest)


def build_query_prompt(data: dict) -> str:
    """Render a single topic-similarity comparison prompt."""
    return (
        "Query:\n"
        f"Title: {data['title']}\n"
        f"Abstract: {data['abstract']}\n\n"
        "Choose the option most similar in topic to the Query:\n\n"
        "Choice 1:\n"
        f"Title: {data['choice1_title']}\n"
        f"Abstract: {data['choice1_abstract']}\n\n"
        "Choice 2:\n"
        f"Title: {data['choice2_title']}\n"
        f"Abstract: {data['choice2_abstract']}\n\n"
        "Respond only with 'Choice 1' or 'Choice 2' based on the closest topic "
        "similarity. Do not provide any explanation. You must choose "
        "'Choice 1', 'Choice 2' or 'Neither'."
    )


class EntropySample:
    """Select ambiguous papers and turn them into LLM comparison queries.

    Args:
        cluster_centers: Centroids of shape ``(n_clusters, dim)``.
        cluster_labels: Per-paper cluster label, shape ``(n_papers,)``.
        embedding: Paper embeddings of shape ``(n_papers, dim)``.
        cluster_text: Mapping from ``str(label)`` to the list of paper indices
            in that cluster.
        gamma_low: Lower bound of the entropy percentile band to sample from.
        gamma_high: Upper bound of the entropy percentile band.
        alpha: Entropy kernel parameter (see :func:`compute_entropy`).
        epsilon: Closest-center fraction (see :func:`compute_entropy`).
        random_state: Seed for shuffling and triplet choice.
    """

    def __init__(
        self,
        cluster_centers: np.ndarray,
        cluster_labels: np.ndarray,
        embedding: np.ndarray,
        cluster_text: dict[str, list[int]],
        gamma_low: float = 0.0,
        gamma_high: float = 0.2,
        alpha: float = 1.0,
        epsilon: float = 0.5,
        random_state: Optional[int] = None,
    ):
        self.cluster_centers = cluster_centers
        self.cluster_labels = cluster_labels
        self.embedding = embedding
        self.cluster_text = cluster_text
        self._rng = np.random.default_rng(random_state)

        self.entropy, self.closest_center = compute_entropy(
            cluster_centers, embedding, alpha=alpha, epsilon=epsilon
        )
        order = np.argsort(self.entropy)[::-1]
        n = self.entropy.shape[0]
        self.selected = order[int(n * gamma_low) : int(n * gamma_high)]
        self._rng.shuffle(self.selected)

    def _generate_triplets(self) -> list[tuple[int, int, int]]:
        """Form unique ``(anchor, choice_1, choice_2)`` index triplets."""
        triplets: list[tuple[int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        for idx in tqdm(self.selected, desc="triplets"):
            while True:
                c1_center, c2_center = self._rng.choice(
                    self.closest_center[idx], 2, replace=False
                )
                choice1 = self._rng.choice(
                    self.cluster_text[str(self.cluster_labels[c1_center])]
                )
                choice2 = self._rng.choice(
                    self.cluster_text[str(self.cluster_labels[c2_center])]
                )
                triplet = (int(idx), int(choice1), int(choice2))
                if triplet not in seen and idx != choice1 and idx != choice2:
                    seen.add(triplet)
                    triplets.append(triplet)
                    break
        return triplets

    def generate_queries(
        self, titles: list[str], abstracts: list[str]
    ) -> list[dict]:
        """Build LLM query dicts for each sampled triplet.

        The order of the two choices is randomized so the LLM cannot exploit
        position bias; ``output`` records the label implied by the cluster
        assignment when it is unambiguous (``1``/``2``), ``0`` when both choices
        share the anchor's cluster, and ``-1`` when neither does.
        """
        triplets = self._generate_triplets()
        queries: list[dict] = []
        for idx, choice1, choice2 in triplets:
            swap = self._rng.random() > 0.5
            first, second = (choice1, choice2) if swap else (choice2, choice1)

            data = {
                "title": titles[idx],
                "abstract": abstracts[idx],
                "choice1_title": titles[first],
                "choice1_abstract": abstracts[first],
                "choice2_title": titles[second],
                "choice2_abstract": abstracts[second],
            }

            anchor_label = self.cluster_labels[idx]
            same1 = anchor_label == self.cluster_labels[choice1]
            same2 = anchor_label == self.cluster_labels[choice2]
            if same1 and same2:
                output = 0
            elif same1 and not same2:
                output = 1 if swap else 2
            elif not same1 and same2:
                output = 2 if swap else 1
            else:
                output = -1

            queries.append(
                {
                    "prompt": build_query_prompt(data),
                    "idx": int(idx),
                    "choice1_idx": int(choice1),
                    "choice2_idx": int(choice2),
                    "output": int(output),
                }
            )
        return queries

    def __call__(
        self, titles: list[str], abstracts: list[str]
    ) -> list[dict]:
        if len(titles) != len(abstracts):
            raise ValueError("titles and abstracts must have equal length")
        if len(titles) < len(self.cluster_labels):
            raise ValueError(
                f"need at least {len(self.cluster_labels)} titles/abstracts, "
                f"got {len(titles)}"
            )
        return self.generate_queries(titles, abstracts)
