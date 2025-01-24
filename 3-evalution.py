
import json
import os
from matplotlib import cm, pyplot as plt
from matplotlib.colors import rgb2hex
import numpy as np
import pandas as pd
from wordcloud import WordCloud

from models import TextEmbedding, evaluation_scitopic


if __name__ == '__main__':
    data = pd.read_csv("dataset/paper_info.csv")

    path = os.path.join(os.getcwd(), "output/embedding")
    if not os.path.exists(os.path.join(path, "metadata.json")):
        metadata = []
        for idx, row in data.iterrows():
            metadata.append({
                "authors": row["authors"],
                "year": row["year"],
                "conference": row["conference"]
            })
    else:
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)

    # Embedding
    finetune_path = os.path.join(os.getcwd(), "output/finetune")
    if not os.path.exists(os.path.join(finetune_path, "embedding.npy")):
        embedder = TextEmbedding(model_path="./finetune_result")
        output_embedding, output_info = embedder(data["title"].tolist(), data["abstract"].tolist(), metadata)

        with open(os.path.join(finetune_path, "info.json"), "w") as f:
            json.dump(output_info, f, indent=4)
    else:
        output_embedding = np.load(os.path.join(finetune_path, "embedding.npy"))
        with open(os.path.join(finetune_path, "info.json"), "r") as f:
            output_info = json.load(f)

    print("Embedding shape:", output_embedding.shape)

    documents = [title + ", " + abstract for title, abstract in zip(data["title"].tolist(), data["abstract"].tolist())]

    td, tc, dbi, silhouette_avg, chi_score, topic_data = evaluation_scitopic(documents, output_embedding, num_topic=10, topic_words=10)

    print("Topic coherence:", tc)
    print("Topic diversity:", td)
    print("Davies-Bouldin Index:", dbi)
    print("Silhouette Score:", silhouette_avg)
    print("Calinski-Harabasz Index:", chi_score)

    viridis = cm.get_cmap("viridis")
    word_list = []

    np.random.seed(42)
    colors = [viridis(np.random.uniform(0, 1)) for _ in range(50)]
    def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        idx = word_list.index(word)
        hex_color = rgb2hex(colors[idx])
        return hex_color

    for idx, topic_words in enumerate(topic_data):
        word_scores = {word: score for word, score in topic_words}
        word_scores_sorted = dict(sorted(word_scores.items(), key=lambda x: x[1], reverse=True))
        word_list = list(word_scores_sorted.keys())
        os.makedirs("result/topic_word_score/scitopic", exist_ok=True)
        os.makedirs("result/wordcloud/scitopic", exist_ok=True)

        df = pd.DataFrame(word_scores_sorted.items(), columns=["word", "score"])
        df.to_csv(f"result/topic_word_score/scitopic/{idx}.csv", index=False)

        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color="white",
            colormap="viridis"
        ).generate_from_frequencies(word_scores)

        wordcloud.recolor(color_func=custom_color_func)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(f"result/wordcloud/scitopic/wordcloud_topic_{idx}.pdf", format="pdf", bbox_inches="tight")
        plt.close()

    