# SciTopic

[![arXiv](https://img.shields.io/badge/arXiv-2508.20514-b31b1b.svg?style=plastic)](https://arxiv.org/pdf/2508.20514.pdf)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-GPL%203.0-green.svg)](LICENSE)

> **Enhancing Topic Discovery in Scientific Literature through Advanced LLM**

SciTopic is a framework that leverages Large Language Models (LLMs) to enhance
topic discovery in scientific literature. It combines text embeddings,
entropy-based sampling, and LLM-guided fine-tuning to achieve higher topic
coherence and diversity than traditional topic-modeling methods.

---

## Overview

Traditional topic models often struggle with the complex, nuanced vocabulary of
scientific texts. SciTopic addresses this with a three-stage pipeline:

1. **Semantic embedding** — encode each paper's title, abstract, and metadata
   with BGE-M3.
2. **Clustering** — group papers with K-means (or HDBSCAN) over the embeddings.
3. **Entropy-based sampling** — select the papers whose cluster assignment is
   most ambiguous and turn them into pairwise comparison queries.
4. **LLM enhancement** — ask an LLM which candidate is topically closest, then
   turn its judgements into fine-tuning triplets.
5. **Fine-tuning & evaluation** — refine the embedding model with knowledge
   distillation and score the resulting topics.

## Project Structure

```
SciTopic/
├── pyproject.toml              # Packaging, dependencies, console script
├── configs/
│   └── default.yaml            # All tunable parameters
├── .env.example                # Template for LLM credentials (secrets)
├── scripts/
│   └── finetune.sh             # FlagEmbedding fine-tuning wrapper (stage 2)
├── src/scitopic/
│   ├── cli.py                  # `scitopic train|evaluate`
│   ├── config.py               # Layered YAML + .env configuration
│   ├── embedding.py            # BGE-M3 / SentenceTransformer embedding
│   ├── clustering.py           # K-means / HDBSCAN
│   ├── sampling.py             # Entropy-based triplet sampling
│   ├── llm.py                  # Online (HTTP) and local (vLLM) LLM backends
│   ├── finetune.py             # Build FlagEmbedding fine-tune data
│   ├── pipeline.py             # Stage orchestration (train / evaluate)
│   ├── visualization.py        # Per-topic word clouds
│   └── evaluation/             # Topic-quality metrics (TC, TD, DBI, ...)
├── examples/
│   └── run_pipeline.py         # Programmatic usage
├── tests/                      # Unit tests for the pure-logic modules
└── dataset/
    └── paper_info.csv          # Input data (title, authors, conference, year, abstract)
```

## Installation

Requires Python 3.8+. A CUDA-capable GPU is recommended for embedding and
fine-tuning.

```bash
# Core install (embedding, clustering, sampling, LLM query, fine-tune data)
pip install -e .

# Add the evaluation/visualization stack (topmost, bertopic, spacy, ...)
pip install -e ".[evaluation]"

# Development tools (pytest, ruff, black)
pip install -e ".[dev]"
```

### Pre-trained model

Download BGE-M3 and point `embedding.model_path` (in `configs/default.yaml`) at
it — the default is `pretrained_model/bge-m3`.

## Configuration

All parameters live in [`configs/default.yaml`](configs/default.yaml). Override
them with your own YAML:

```bash
scitopic train --config my_config.yaml
```

**Secrets are never stored in YAML.** Copy `.env.example` to `.env` and fill in
your LLM endpoint:

```bash
cp .env.example .env
# .env
SCITOPIC_LLM_API_KEY=your-api-key
SCITOPIC_LLM_BASE_URL=https://your-endpoint/v1
```

`.env` is gitignored so credentials stay out of version control.

## Usage

### Stage 1 — Training-data generation

```bash
scitopic train
```

Loads the dataset, embeds papers, clusters them, applies entropy-based sampling,
queries the LLM, and writes FlagEmbedding fine-tuning data. The command prints
the suggested `--query_max_len` / `--passage_max_len` for the next stage.

### Stage 2 — Fine-tuning

```bash
bash scripts/finetune.sh
```

A thin wrapper over `FlagEmbedding.finetune.embedder.encoder_only.m3`. Edit the
paths and GPU selection at the top of the script to match your environment.

### Stage 3 — Evaluation

```bash
scitopic evaluate
```

Re-embeds with the fine-tuned model, computes topic-quality metrics, and renders
per-topic word clouds under `result/`.

| Metric | Meaning |
|--------|---------|
| Topic Coherence (TC) | Semantic similarity within topics |
| Topic Diversity (TD) | Uniqueness across topics |
| Davies-Bouldin Index (DBI) | Cluster separation (lower is better) |
| Silhouette Score | Cluster cohesion and separation |
| Calinski-Harabasz Index (CHI) | Cluster density |

### Programmatic use

```python
from scitopic import load_config, run_train, run_evaluate

config = load_config()          # or load_config("my_config.yaml")
run_train(config)
run_evaluate(config)
```

See [`examples/run_pipeline.py`](examples/run_pipeline.py) for a complete script.

## Dataset Format

The input CSV must contain these columns:

| Column | Description |
|--------|-------------|
| `title` | Paper title |
| `authors` | Author names |
| `conference` | Publication venue |
| `year` | Publication year |
| `abstract` | Paper abstract |

## Development

```bash
pip install -e ".[dev]"
pytest            # run the test suite
ruff check .      # lint
black .           # format
```

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

This project is licensed under the GPL-3.0 License — see the [LICENSE](LICENSE)
file for details.

## Acknowledgments

- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) for the BGE-M3 model
  and fine-tuning framework.
- The authors of the papers in our dataset for their valuable research.
