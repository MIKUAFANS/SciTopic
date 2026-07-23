"""Tests for fine-tuning-data construction from LLM judgements."""

from __future__ import annotations

import json

from scitopic.finetune import FineTuneDataBuilder, parse_choice


def test_parse_choice_variants():
    assert parse_choice("Choice 1") == 1
    assert parse_choice("Choice 2.") == 2
    assert parse_choice("The answer is one") == 1
    assert parse_choice("two") == 2


def test_parse_choice_neither_and_unknown():
    assert parse_choice("Neither of them") is None
    assert parse_choice("I am not sure") is None
    assert parse_choice("") is None


def _query_result(response: str) -> dict:
    return {
        "idx": 0,
        "choice1_idx": 1,
        "choice2_idx": 2,
        "llm_response": response,
    }


def test_build_orders_pos_neg_by_choice():
    titles = ["anchor", "cand1", "cand2"]
    abstracts = ["a0", "a1", "a2"]

    builder = FineTuneDataBuilder([_query_result("Choice 1")], titles, abstracts)
    examples, _, _ = builder.build()
    assert len(examples) == 1
    assert "cand1" in examples[0]["pos"][0]
    assert "cand2" in examples[0]["neg"][0]

    builder = FineTuneDataBuilder([_query_result("Choice 2")], titles, abstracts)
    examples, _, _ = builder.build()
    assert "cand2" in examples[0]["pos"][0]
    assert "cand1" in examples[0]["neg"][0]


def test_build_skips_neither():
    titles = ["anchor", "cand1", "cand2"]
    abstracts = ["a0", "a1", "a2"]
    builder = FineTuneDataBuilder([_query_result("Neither")], titles, abstracts)
    examples, max_q, max_p = builder.build()
    assert examples == []
    assert max_q == 0
    assert max_p == 0


def test_save_writes_jsonl(tmp_path):
    titles = ["anchor", "cand1", "cand2"]
    abstracts = ["a0", "a1", "a2"]
    builder = FineTuneDataBuilder([_query_result("Choice 1")], titles, abstracts)
    examples, _, _ = builder.build()

    path = builder.save(examples, str(tmp_path))
    lines = [json.loads(line) for line in open(path, encoding="utf-8")]
    assert len(lines) == 1
    assert set(lines[0]) == {"query", "pos", "neg"}
