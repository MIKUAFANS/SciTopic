from collections import defaultdict
import json
import os
import re
import subprocess
from typing import Any
from FlagEmbedding import BGEM3FlagModel
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class MultiheadAttentionFusion(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiheadAttentionFusion, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)
        self.linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        # seq_len, batch_size, embedding_dim]
        x = x.permute(1, 0, 2)  # [3, batch_size, embedding_dim]

        # Multihead Attention
        attention_output, _ = self.multihead_attention(x, x, x)  # [3, batch_size, embedding_dim]

        fused_embedding = attention_output.mean(dim=0)  # [batch_size, embedding_dim]

        fused_embedding = self.linear(fused_embedding)  # [batch_size, embedding_dim]
        return fused_embedding

class TextEmbedding:
    def __init__(self, embedder : str = "bge", device : str="cuda", model_path : str = None):
        assert embedder is not None, "embedder must be provided"
        if embedder == "bge":
            self.embedder = BGEM3FlagModel(model_path, use_fp16=True, device=device)
        else:
            self.embedder = SentenceTransformer(model_path, device=device)
        
        self.output_embedding = []
        self.output_metadata = []
        self.attention_embedding = []

    def forward(self, title : list, abstract : list, metadata : list[dict], save_path : str = None):
        for idx, row in enumerate(title):
            metadata_sentences = f"Author: {metadata[idx]['authors']}, Year: {metadata[idx]['year']}, Venue: {metadata[idx]['conference']}"
            self.output_metadata.append({
                "title": row,
                "abstract": abstract[idx],
                "metadata": metadata_sentences
            })

            self.output_embedding.append(self.embedder.encode([row, abstract[idx], metadata_sentences])["dense_vecs"])

        self.output_embedding = np.array(self.output_embedding)
        self.save(save_path)

        return self.output_embedding, self.output_metadata

    def save(self, save_path : str):
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, "embedding.npy"), self.output_embedding)
            
            with open(os.path.join(save_path, "metadata.json"), "w") as f:
                json.dump(self.output_metadata, f, indent=4)
        else:
            save_path = os.path.join(os.getcwd(), "output", "embedding")
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, "embedding.npy"), self.output_embedding)
            
            with open(os.path.join(save_path, "metadata.json"), "w") as f:
                json.dump(self.output_metadata, f, indent=4)

    def __call__(self, title : str, abstract : str, metadata : dict, save_path : str = None, **kwds: Any) -> Any:
        return self.forward(title, abstract, metadata, save_path=save_path, **kwds)


class ModelFineTune:
    def __init__(self, query_result : list, title : list, abstract : list):
        self.query_result = query_result
        if title is None:
            self.title = ["" for _ in range(len(abstract))]
        else:
            self.title = title

        if abstract is None:
            self.abstract = ["" for _ in range(len(title))]
        else:
            self.abstract = abstract
        self.number_mapping = {
            "1": 1,
            "one": 1,
            "2": 2,
            "two": 2,
        }

    def choice_recognition(self, query_result):
        query_result = query_result.replace(".", "")
        choice = query_result.split(" ")[-1]

        return self.number_mapping.get(choice.lower())

    def generate_finetune_data(self):
        self.fine_tune_data = []
        max_query_len = []
        max_passage_len = []
        for item in self.query_result:
            response = item["llm_response"]
            label = self.choice_recognition(response)
            if label == 1:
                data = {
                    "query": f"Title: {self.title[item['idx']]}, Abstract: {self.abstract[item['idx']]}",
                    "pos": [f"Title: {self.title[item['choice1_idx']]}, Abstract: {self.abstract[item['choice1_idx']]}"],
                    "neg": [f"Title: {self.title[item['choice2_idx']]}, Abstract: {self.abstract[item['choice2_idx']]}"]
                }
            elif label == 2:
                data = {
                    "query": f"Title: {self.title[item['idx']]}, Abstract: {self.abstract[item['idx']]}",
                    "pos": [f"Title: {self.title[item['choice2_idx']]}, Abstract: {self.abstract[item['choice2_idx']]}"],
                    "neg": [f"Title: {self.title[item['choice1_idx']]}, Abstract: {self.abstract[item['choice1_idx']]}"]
                }
            else:
                print(f"Error: unrecognized choice {label}")
                continue

            max_query_len.append(len(data["query"].split(" ")))
            max_passage_len.append(max(len(data["pos"][0].split(" ")), len(data["neg"][0].split(" ")), len(data["query"].split(" "))))

            self.fine_tune_data.append(data)

        return self.fine_tune_data, max_query_len, max_passage_len
    
    def save(self, fine_tune_data, n_cluster : int, query_max_len : int, passage_max_len : int, save_path : str = None):
        if save_path is None:
            save_path = os.path.join(os.getcwd(), "output", "fine_tune_data")
        else:
            os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, f"fine_tune_data_{n_cluster}_{query_max_len}_{passage_max_len}.jsonl"), "w", encoding="utf-8") as f:
            for data in fine_tune_data:
                f.write(json.dumps(data) + "\n")

        return save_path

    def fine_tune(self, n_cluster : int, query_max_len : int, passage_max_len : int, file : str):
        args = f"""
            CUDA_VISIBLE_DEVICES="6,7" torchrun --nproc_per_node 2 \
            -m FlagEmbedding.baai_general_embedding.finetune.run \
            --output_dir finetune_result/bge-m3-finetune-{n_cluster} \
            --model_name_or_path bge-m3 \
            --train_data {file} \
            --learning_rate 1e-5 \
            --fp16 \
            --num_train_epochs 5 \
            --per_device_train_batch_size 4 \
            --dataloader_drop_last True \
            --normlized True \
            --temperature 0.02 \
            --query_max_len {query_max_len} \
            --passage_max_len {passage_max_len} \
            --train_group_size 2 \
            --negatives_cross_device \
            --logging_steps 10 \
            --save_steps 1000 \
            --query_instruction_for_retrieval "" 
            """
        
        subprocess.run(args, shell=True, check=True)

        return args

    def __call__(self, n_cluster : int, save_path : str = None, **kwds: Any) -> Any:
        fine_tune_data, max_query_len, max_passage_len = self.generate_finetune_data()
        save_path = self.save(fine_tune_data, query_max_len=max(max_query_len), passage_max_len=max(max_passage_len), save_path=save_path)
        self.fine_tune(n_cluster=n_cluster, query_max_len=max(max_query_len), passage_max_len=max(max_passage_len), file=os.path.join(save_path, f"fine_tune_data_{n_cluster}.jsonl"))

        return fine_tune_data, max_query_len, max_passage_len

class TripletDataset(Dataset):
    def __init__(self, embedding, llm_query_result):
        self.pattern = re.compile(r"neither", re.I)
        self.number_mapping = {
            "1": 1,
            "one": 1,
            "2": 2,
            "two": 2,
        }
        self.embedding = torch.tensor(embedding, dtype=torch.float32)
        self.llm_query_result = self._filter(llm_query_result)

    def __len__(self):
        return len(self.llm_query_result)

    def _choice_recognition(self, query_result):
        if self.pattern.search(query_result):
            return 0
        query_result = query_result.replace(".", "")
        choice = query_result.split(" ")[-1]

        return self.number_mapping.get(choice.lower())

    def _filter(self, query_result):
        valid_llm_response = []

        for item in query_result:
            response = item["llm_response"]
            label = self._choice_recognition(response)
            if label == 1 or label == 2:
                valid_llm_response.append(item)

        return valid_llm_response

    def __getitem__(self, idx):
        item = self.llm_query_result[idx]

        response = item["llm_response"]
        label = self._choice_recognition(response)
        anchor = self.embedding[item["idx"]]

        if label == 1:
            positive = self.embedding[item["choice1_idx"]]
            negative = self.embedding[item["choice2_idx"]]
        elif label == 2:
            positive = self.embedding[item["choice2_idx"]]
            negative = self.embedding[item["choice1_idx"]]
        else:
            print(f"Error: unrecognized choice {label}")
            return None

        return anchor, positive, negative

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = torch.norm(anchor - positive, p=2, dim=1)  # ||a - p||_2
        negative_distance = torch.norm(anchor - negative, p=2, dim=1)  # ||a - n||_2
        loss = torch.clamp(positive_distance - negative_distance + self.margin, min=0.0)  # max(0, d+ - d- + margin)
        return loss.mean()