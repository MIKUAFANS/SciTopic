"""High-level orchestration of the SciTopic stages.

Two entry points are provided:

* :func:`run_train` — embed papers, cluster them, sample high-entropy triplets,
  query an LLM, and write FlagEmbedding fine-tuning data.
* :func:`run_evaluate` — re-embed with the fine-tuned model, compute topic
  metrics, and render per-topic word clouds.

Both consume a :class:`~scitopic.config.Config` and are intentionally thin: the
real work lives in the single-purpose modules they call.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from .clustering import TextCluster
from .config import Config
from .embedding import TextEmbedding
from .finetune import FineTuneDataBuilder
from .llm import LLMQueryOnline
from .sampling import EntropySample

_REQUIRED_COLUMNS = ["title", "authors", "conference", "year", "abstract"]


def _load_dataset(config: Config) -> pd.DataFrame:
    """Load the input CSV and validate its schema."""
    data = pd.read_csv(config.data.input_csv)
    missing = [c for c in _REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(
            f"{config.data.input_csv} is missing required column(s): {missing}"
        )
    return data


def _build_metadata(data: pd.DataFrame) -> list[dict]:
    return [
        {"authors": row["authors"], "year": row["year"], "conference": row["conference"]}
        for _, row in data.iterrows()
    ]


def _embed_or_load(config: Config, data: pd.DataFrame, embedding_dir: str, model_path: str) -> np.ndarray:
    """Return embeddings from ``embedding_dir`` if cached, else compute and save."""
    embedding_file = os.path.join(embedding_dir, "embedding.npy")
    if os.path.exists(embedding_file):
        return np.load(embedding_file)

    embedder = TextEmbedding(
        backend=config.embedding.backend,
        model_path=model_path,
        device=config.embedding.device,
        use_fp16=config.embedding.use_fp16,
    )
    embedding, _ = embedder.encode(
        data["title"].tolist(),
        data["abstract"].tolist(),
        _build_metadata(data),
        save_dir=embedding_dir,
    )
    return embedding


def run_train(config: Config) -> None:
    """Run stage 1: embedding, clustering, sampling, LLM query, fine-tune data.

    Requires ``SCITOPIC_LLM_API_KEY`` / ``SCITOPIC_LLM_BASE_URL`` in the
    environment (see ``.env.example``).
    """
    if not config.llm.base_url:
        raise ValueError(
            "LLM base URL is not set. Provide SCITOPIC_LLM_BASE_URL in your .env "
            "(and SCITOPIC_LLM_API_KEY if the endpoint requires it)."
        )

    data = _load_dataset(config)
    os.makedirs(config.paths.embedding_dir, exist_ok=True)

    embedding = _embed_or_load(
        config, data, config.paths.embedding_dir, config.embedding.model_path
    )
    print(f"Embedding shape: {embedding.shape}")
    embedding = embedding.reshape(embedding.shape[0], -1)

    cluster = TextCluster(
        method=config.clustering.method,
        n_clusters=config.clustering.n_clusters,
        random_state=config.clustering.random_state,
    )
    labels, centers = cluster(embedding)

    cluster_text: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_text[str(label)].append(idx)

    sampler = EntropySample(
        cluster_centers=centers,
        cluster_labels=labels,
        embedding=embedding,
        cluster_text=cluster_text,
        gamma_low=config.sampling.gamma_low,
        gamma_high=config.sampling.gamma_high,
        alpha=config.sampling.alpha,
        epsilon=config.sampling.epsilon,
        random_state=config.clustering.random_state,
    )
    queries = sampler(data["title"].tolist(), data["abstract"].tolist())

    llm = LLMQueryOnline(
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        model=config.llm.model,
        n_jobs=config.llm.n_jobs,
        prompt=config.llm.prompt,
    )
    query_results = llm(queries)

    results_path = os.path.join(config.paths.embedding_dir, "llm_query_result.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(query_results, f)

    builder = FineTuneDataBuilder(
        query_results, data["title"].tolist(), data["abstract"].tolist()
    )
    examples, max_query_len, max_passage_len = builder.build()
    out_dir = config.paths.finetune_data_dir
    path = builder.save(examples, out_dir)
    print(
        f"Wrote {len(examples)} fine-tuning examples to {path}\n"
        f"Suggested --query_max_len {max_query_len} --passage_max_len {max_passage_len}"
    )


def run_evaluate(config: Config) -> None:
    """Run stage 3: re-embed with the fine-tuned model, score topics, visualize."""
    # Imported lazily so the heavy evaluation dependencies are only required
    # when evaluation is actually run.
    from .evaluation import evaluation_scitopic
    from .visualization import render_wordclouds

    data = _load_dataset(config)
    os.makedirs(config.paths.finetune_embedding_dir, exist_ok=True)

    embedding = _embed_or_load(
        config,
        data,
        config.paths.finetune_embedding_dir,
        config.embedding.finetuned_model_path,
    )
    print(f"Embedding shape: {embedding.shape}")

    documents = [
        f"{title}, {abstract}"
        for title, abstract in zip(data["title"].tolist(), data["abstract"].tolist())
    ]

    td, tc, dbi, silhouette_avg, chi_score, topic_data = evaluation_scitopic(
        documents,
        embedding,
        num_topic=config.evaluation.num_topic,
        topic_words=config.evaluation.topic_words,
    )

    print(f"Topic coherence:          {tc}")
    print(f"Topic diversity:          {td}")
    print(f"Davies-Bouldin Index:     {dbi}")
    print(f"Silhouette Score:         {silhouette_avg}")
    print(f"Calinski-Harabasz Index:  {chi_score}")

    render_wordclouds(topic_data, config.paths.result_dir)
