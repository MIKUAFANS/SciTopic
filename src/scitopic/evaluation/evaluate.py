"""Topic-model evaluation entry point.

Runs the SciTopic evaluation over a set of documents and their embeddings,
returning standard topic-quality metrics alongside the extracted topics.
"""

from __future__ import annotations

import topmost
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from topmost.data import RawDataset
from topmost.preprocessing import Preprocessing

from .metrics import calculate_topic_diversity
from .preprocessing import preprocessing_dataset
from .topic_model import EvaluationModel


def evaluation_scitopic(documents, embedding, num_topic=10, topic_words=10):
    """Evaluate topic quality for a document set and its embeddings.

    Args:
        documents: Raw document strings (e.g. ``"title, abstract"``).
        embedding: Embedding matrix aligned with ``documents``; reshaped to 2-D.
        num_topic: Number of topics to extract / cluster into.
        topic_words: Number of top words retained per topic.

    Returns:
        A tuple ``(td, tc, dbi, silhouette_avg, chi_score, topic_data)``:
        topic diversity, topic coherence, Davies-Bouldin index, silhouette
        score, Calinski-Harabasz index, and the per-topic word/score lists.
    """
    model = EvaluationModel(nr_topics=num_topic, top_n_words=topic_words)
    output_embedding = embedding.reshape(embedding.shape[0], -1)

    new_documents = preprocessing_dataset(documents)
    new_documents = [" ".join(document) for document in new_documents]
    model.fit_transform(new_documents, embeddings=output_embedding)

    topic_data = model.get_topics().values()

    preprocessing = Preprocessing(vocab_size=10000, stopwords="English")
    dataset = RawDataset(documents, preprocessing, device="cuda")

    top_words = []
    top_words_cal = []
    for item in topic_data:
        top_words.append(" ".join([x[0] for x in item]))
        top_words_cal.append([x[0] for x in item])

    td = calculate_topic_diversity(top_words_cal, top_n=topic_words)
    tc = topmost.evaluations.compute_topic_coherence(
        dataset.train_texts, dataset.vocab, top_words
    )

    kmeans = KMeans(n_clusters=num_topic)
    kmeans.fit(output_embedding)
    labels = kmeans.labels_

    dbi = davies_bouldin_score(output_embedding, labels)
    silhouette_avg = silhouette_score(output_embedding, labels)
    chi_score = calinski_harabasz_score(output_embedding, labels)

    return td, tc, dbi, silhouette_avg, chi_score, topic_data
