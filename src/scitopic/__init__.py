"""SciTopic — enhancing topic discovery in scientific literature through LLMs.

The public API mirrors the pipeline stages: text embedding, clustering,
entropy-based sampling, LLM querying, and fine-tuning-data construction. Heavy
optional dependencies (evaluation, visualization) are imported lazily by the
functions that need them and are not re-exported here.
"""

from __future__ import annotations

from .clustering import TextCluster
from .config import Config, load_config
from .embedding import TextEmbedding
from .finetune import FineTuneDataBuilder
from .llm import LLMQueryLocal, LLMQueryOnline
from .pipeline import run_evaluate, run_train
from .sampling import EntropySample, compute_entropy

__version__ = "0.1.0"

__all__ = [
    "Config",
    "load_config",
    "TextEmbedding",
    "TextCluster",
    "EntropySample",
    "compute_entropy",
    "LLMQueryOnline",
    "LLMQueryLocal",
    "FineTuneDataBuilder",
    "run_train",
    "run_evaluate",
    "__version__",
]
