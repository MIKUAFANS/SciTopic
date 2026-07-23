"""Topic visualization.

Renders each topic's top words as a word cloud (one PDF per topic) and writes the
underlying word/score table as CSV, mirroring the figures reported in the paper.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def render_wordclouds(topic_data, result_dir: str, colormap: str = "viridis") -> None:
    """Render per-topic word clouds and word-score CSVs.

    Args:
        topic_data: Iterable of topics, each a list of ``(word, score)`` pairs
            (the value returned by ``EvaluationModel.get_topics().values()``).
        result_dir: Base directory; outputs go under ``topic_word_score/scitopic``
            and ``wordcloud/scitopic``.
        colormap: Matplotlib colormap name used to color the words.
    """
    # Imported lazily so plotting dependencies are optional until visualization runs.
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import rgb2hex
    from wordcloud import WordCloud

    score_dir = os.path.join(result_dir, "topic_word_score", "scitopic")
    cloud_dir = os.path.join(result_dir, "wordcloud", "scitopic")
    os.makedirs(score_dir, exist_ok=True)
    os.makedirs(cloud_dir, exist_ok=True)

    cmap = cm.get_cmap(colormap)
    rng = np.random.default_rng(42)

    for idx, topic_words in enumerate(topic_data):
        word_scores = dict(
            sorted(
                {word: score for word, score in topic_words}.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
        )
        words = list(word_scores.keys())
        colors = {word: rgb2hex(cmap(rng.uniform(0, 1))) for word in words}

        pd.DataFrame(word_scores.items(), columns=["word", "score"]).to_csv(
            os.path.join(score_dir, f"{idx}.csv"), index=False
        )

        wordcloud = WordCloud(
            width=800, height=400, background_color="white", colormap=colormap
        ).generate_from_frequencies(word_scores)
        wordcloud.recolor(color_func=lambda word, **_: colors[word])

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(
            os.path.join(cloud_dir, f"wordcloud_topic_{idx}.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        plt.close()
