# SciTopic

[![arXiv](https://img.shields.io/badge/arXiv-2508.20514-b31b1b.svg?style=plastic)](https://arxiv.org/pdf/2508.20514.pdf)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-GPL%3.0-green.svg)](LICENSE)

> **Enhancing Topic Discovery in Scientific Literature through Advanced LLM**

SciTopic is a novel framework that leverages Large Language Models (LLMs) to enhance topic discovery in scientific literature. By combining text embeddings, entropy-based sampling, and LLM-guided fine-tuning, SciTopic achieves superior topic coherence and diversity compared to traditional topic modeling methods.

---

## Overview

Traditional topic models often struggle with the complexity and nuanced vocabulary of scientific texts. SciTopic addresses this by:

- **Semantic Embedding**: Using BGE-M3 for high-quality text representations
- **Intelligent Clustering**: K-means clustering on embedded representations
- **Entropy-based Sampling**: Selecting representative samples for LLM queries
- **LLM Enhancement**: Leveraging LLMs to generate topic-aware training data
- **Fine-tuning**: Refining embeddings with knowledge distillation

## Features

- Multi-modal text embedding with BGE-M3
- Entropy-based representative sample selection
- LLM-guided topic label generation
- Knowledge distillation for embedding fine-tuning
- Comprehensive evaluation metrics (Topic Coherence, Topic Diversity, DBI, Silhouette Score, CHI)
- Word cloud visualization for topic interpretation

## Project Structure

```
SciTopic/
├── 1-train.py           # Training pipeline: embedding, clustering, LLM query
├── 2-finetune.sh        # Fine-tuning script using FlagEmbedding
├── 3-evalution.py       # Evaluation and visualization
├── dataset/
│   └── paper_info.csv   # Input dataset (title, authors, conference, year, abstract)
├── models/
│   ├── scitopic/        # Core modules
│   │   ├── _embedding.py    # Text embedding module
│   │   ├── _cluster.py      # Clustering module
│   │   └── _llm_query.py    # LLM query and fine-tune data generation
│   └── evalution/       # Evaluation metrics
└── output/              # Generated outputs (embeddings, fine-tune data, results)
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA (recommended for GPU acceleration)

### Dependencies

```bash
pip install numpy pandas torch transformers FlagEmbedding wordcloud matplotlib scikit-learn openai
```

### Pre-trained Model

Download the BGE-M3 model and place it in the `pretrained_model/` directory:

```bash
mkdir -p pretrained_model
# Download BGE-M3 from HuggingFace
```

## Usage

SciTopic follows a three-stage pipeline:

### Stage 1: Training Data Generation

Generate embeddings, perform clustering, and query LLM for topic-aware fine-tuning data:

```bash
python 1-train.py
```

This script will:
1. Load paper data from `dataset/paper_info.csv`
2. Generate text embeddings using BGE-M3
3. Perform K-means clustering (default: 100 clusters)
4. Apply entropy-based sampling to select representative papers
5. Query LLM to generate topic labels
6. Create fine-tuning dataset

### Stage 2: Fine-tuning

Fine-tune the embedding model with the generated data:

```bash
bash 2-finetune.sh
```

Key parameters:
- `--model_name_or_path`: Path to pre-trained BGE-M3
- `--train_data`: Path to generated fine-tune data
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--learning_rate`: Learning rate (default: 1e-5)

### Stage 3: Evaluation

Evaluate the fine-tuned model and generate visualizations:

```bash
python 3-evalution.py
```

Evaluation metrics include:
- **Topic Coherence (TC)**: Measures semantic similarity within topics
- **Topic Diversity (TD)**: Measures uniqueness across topics
- **Davies-Bouldin Index (DBI)**: Measures cluster separation
- **Silhouette Score**: Measures cluster cohesion and separation
- **Calinski-Harabasz Index (CHI)**: Measures cluster density

Results will be saved to `result/` including:
- Topic word scores (`result/topic_word_score/`)
- Word cloud visualizations (`result/wordcloud/`)

## Dataset Format

Input CSV file should contain the following columns:

| Column | Description |
|--------|-------------|
| `title` | Paper title |
| `authors` | Author names |
| `conference` | Publication venue |
| `year` | Publication year |
| `abstract` | Paper abstract |

## Configuration

### LLM API Configuration

In `1-train.py`, configure your LLM API:

```python
llm_query_model = LLMQueryOnline(api_key="your-api-key", model="model_name")
```

### Clustering Parameters

Adjust clustering parameters in `1-train.py`:

```python
cluster = TextCluster(model_name="k_means", n_clusters=100)  # Number of clusters
```

### Evaluation Parameters

Adjust evaluation parameters in `3-evalution.py`:

```python
evaluation_scitopic(documents, output_embedding, num_topic=10, topic_words=10)
```

## Results

SciTopic demonstrates significant improvements over baseline methods:

- Higher topic coherence scores
- Better topic diversity
- More interpretable topic representations
- Improved clustering quality metrics

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{li2025scitopic,
  title={Scitopic: Enhancing topic discovery in scientific literature through advanced llm},
  author={Li, Pengjiang and Wang, Zaitian and Zhang, Xinhao and Zhang, Ran and Jiang, Lu and Wang, Pengfei and Zhou, Yuanchun},
  journal={arXiv preprint arXiv:2508.20514},
  year={2025}
}
```

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) for the BGE-M3 model and fine-tuning framework
- The authors of the papers in our dataset for their valuable research contributions
