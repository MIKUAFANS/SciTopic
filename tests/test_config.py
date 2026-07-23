"""Tests for layered configuration loading."""

from __future__ import annotations

import textwrap

from scitopic.config import load_config


def test_defaults_load():
    config = load_config()
    assert config.clustering.n_clusters == 100
    assert config.embedding.backend == "bge"
    assert config.evaluation.num_topic == 10


def test_override_yaml_merges(tmp_path):
    override = tmp_path / "custom.yaml"
    override.write_text(
        textwrap.dedent(
            """
            clustering:
              n_clusters: 42
            evaluation:
              num_topic: 5
            """
        ),
        encoding="utf-8",
    )

    config = load_config(str(override))
    # Overridden values take effect...
    assert config.clustering.n_clusters == 42
    assert config.evaluation.num_topic == 5
    # ...while untouched values keep their defaults.
    assert config.embedding.backend == "bge"
    assert config.clustering.method == "k_means"


def test_secrets_come_from_environment(monkeypatch):
    monkeypatch.setenv("SCITOPIC_LLM_API_KEY", "secret-key")
    monkeypatch.setenv("SCITOPIC_LLM_BASE_URL", "http://example.test/v1")

    config = load_config()
    assert config.llm.api_key == "secret-key"
    assert config.llm.base_url == "http://example.test/v1"
