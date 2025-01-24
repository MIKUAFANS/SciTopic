from collections import defaultdict
import json
import os
import numpy as np
import pandas as pd
from models import TextCluster, EntropySample, LLMQueryOnline, ModelFineTune, TextEmbedding


if __name__ == '__main__':
    data = pd.read_csv("dataset/paper_info.csv")
    path = os.path.join(os.getcwd(), "output/embedding")
    os.makedirs(path, exist_ok=True)
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
    if not os.path.exists(os.path.join(path, "embedding.npy")):
        embedder = TextEmbedding(model_path="./pretrained_model/bge-m3")
        output_embedding, output_info = embedder(data["title"].tolist(), data["abstract"].tolist(), metadata)

        with open(os.path.join(path, "info.json"), "w") as f:
            json.dump(output_info, f, indent=4)
    else:
        output_embedding = np.load(os.path.join(path, "embedding.npy"))
        with open(os.path.join(path, "info.json"), "r") as f:
            output_info = json.load(f)

    print("Embedding shape:", output_embedding.shape)

    output_embedding = output_embedding.reshape(output_embedding.shape[0], -1)
    cluster = TextCluster(model_name="k_means", n_clusters=100)
    labels, centers = cluster(output_embedding)
    text_cluster = defaultdict(list)
    for idx, label in enumerate(labels):
        text_cluster[str(label)].append(idx)

    # Entropy Sample
    entropy_sample_model = EntropySample(centers, labels, output_embedding, text_cluster)
    llm_query, max_len = entropy_sample_model(data["title"].tolist(), data["abstract"].tolist())
    llm_query_model = LLMQueryOnline(api_key="api-key", model="model_name")
    llm_query_result = llm_query_model(llm_query)

    with open(os.path.join(path, "llm_query_result.json"), "w") as f:
        json.dump(llm_query_result, f)

    with open(s.path.join(path, "llm_query_result.json"), "r") as f:
        llm_query_result = json.load(f)

    finetune = ModelFineTune(llm_query_result, data["title"].tolist(), data["abstract"].tolist())

    fine_tune_abstract_result, max_query_len_result, max_passage_len_result = finetune.generate_finetune_data()

    max_query_len_result, max_passage_len_result = max(max_query_len_result), max(max_passage_len_result)

    output_path = os.path.join(os.getcwd(), "output/fine_tune/AI-DM")

    finetune.save(fine_tune_abstract_result, 100, query_max_len=max_query_len_result, passage_max_len=max_passage_len_result, save_path=output_path)