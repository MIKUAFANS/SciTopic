"""Tests for entropy computation and query construction."""

from __future__ import annotations

import numpy as np

from scitopic.sampling import EntropySample, build_query_prompt, compute_entropy


def test_compute_entropy_shapes():
    centers = np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    points = np.array([[0.0, 0.0], [5.0, 5.0], [9.0, 9.0]])

    entropy, closest = compute_entropy(centers, points, alpha=1.0, epsilon=0.5)
    assert entropy.shape == (3,)
    # epsilon * 3 -> 1, floored to a minimum of 2 closest centers.
    assert closest.shape == (3, 2)


def test_compute_entropy_ambiguous_point_has_higher_entropy():
    centers = np.array([[0.0, 0.0], [10.0, 0.0]])
    # One point sits on a center (confident), one is equidistant (ambiguous).
    points = np.array([[0.0, 0.0], [5.0, 0.0]])

    entropy, _ = compute_entropy(centers, points)
    assert entropy[1] > entropy[0]


def test_build_query_prompt_contains_fields():
    prompt = build_query_prompt(
        {
            "title": "T",
            "abstract": "A",
            "choice1_title": "C1T",
            "choice1_abstract": "C1A",
            "choice2_title": "C2T",
            "choice2_abstract": "C2A",
        }
    )
    for token in ("C1T", "C2T", "Choice 1", "Choice 2", "Neither"):
        assert token in prompt


def test_entropy_sample_generates_queries():
    rng = np.random.default_rng(0)
    embedding = rng.normal(size=(20, 4))
    centers = rng.normal(size=(4, 4))
    labels = rng.integers(0, 4, size=20)
    cluster_text = {str(c): np.where(labels == c)[0].tolist() for c in range(4)}
    # Ensure every referenced cluster is non-empty.
    for c in range(4):
        if not cluster_text[str(c)]:
            cluster_text[str(c)] = [0]

    sampler = EntropySample(
        cluster_centers=centers,
        cluster_labels=labels,
        embedding=embedding,
        cluster_text=cluster_text,
        gamma_low=0.0,
        gamma_high=1.0,
        random_state=0,
    )
    titles = [f"title {i}" for i in range(20)]
    abstracts = [f"abstract {i}" for i in range(20)]
    queries = sampler(titles, abstracts)

    assert len(queries) > 0
    for q in queries:
        assert set(q) == {"prompt", "idx", "choice1_idx", "choice2_idx", "output"}
        assert q["output"] in (-1, 0, 1, 2)
