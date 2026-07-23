"""Configuration loading for SciTopic.

Configuration is layered: values from ``configs/default.yaml`` are the base,
an optional user-supplied YAML overrides them, and secrets (LLM credentials)
are read from the environment (loaded from a ``.env`` file if present).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

# Default config shipped with the package repository.
_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass
class DataConfig:
    input_csv: str = "dataset/paper_info.csv"


@dataclass
class PathsConfig:
    output_dir: str = "output"
    embedding_dir: str = "output/embedding"
    finetune_data_dir: str = "output/fine_tune/AI-DM"
    finetune_embedding_dir: str = "output/finetune"
    result_dir: str = "result"
    cache_dir: str = "cache"


@dataclass
class EmbeddingConfig:
    backend: str = "bge"
    model_path: str = "pretrained_model/bge-m3"
    finetuned_model_path: str = "finetune_result"
    device: str = "cuda"
    use_fp16: bool = True


@dataclass
class ClusteringConfig:
    method: str = "k_means"
    n_clusters: int = 100
    random_state: int = 2024


@dataclass
class SamplingConfig:
    gamma_low: float = 0.0
    gamma_high: float = 0.2
    alpha: float = 1.0
    epsilon: float = 0.5


@dataclass
class LLMConfig:
    model: str = "Meta-Llama-3.1-70B-Instruct"
    n_jobs: int = 8
    prompt: Optional[str] = None
    # Secrets — populated from the environment, never from YAML.
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class EvaluationConfig:
    num_topic: int = 10
    topic_words: int = 10


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: Optional[str] = None) -> Config:
    """Build a :class:`Config` from the default YAML, an optional override, and env.

    Args:
        config_path: Optional path to a YAML file whose values override the
            packaged defaults.

    Returns:
        A fully populated :class:`Config`. LLM credentials are pulled from the
        environment variables ``SCITOPIC_LLM_API_KEY`` and
        ``SCITOPIC_LLM_BASE_URL`` (loaded from ``.env`` if present).
    """
    load_dotenv()

    merged = _load_yaml(_DEFAULT_CONFIG)
    if config_path:
        merged = _deep_merge(merged, _load_yaml(Path(config_path)))

    config = Config(
        data=DataConfig(**merged.get("data", {})),
        paths=PathsConfig(**merged.get("paths", {})),
        embedding=EmbeddingConfig(**merged.get("embedding", {})),
        clustering=ClusteringConfig(**merged.get("clustering", {})),
        sampling=SamplingConfig(**merged.get("sampling", {})),
        llm=LLMConfig(**merged.get("llm", {})),
        evaluation=EvaluationConfig(**merged.get("evaluation", {})),
    )

    # Secrets are environment-only.
    config.llm.api_key = os.getenv("SCITOPIC_LLM_API_KEY")
    config.llm.base_url = os.getenv("SCITOPIC_LLM_BASE_URL", config.llm.base_url)

    return config
