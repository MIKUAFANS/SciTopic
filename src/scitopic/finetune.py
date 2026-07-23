"""Build fine-tuning data from LLM topic-similarity judgements.

Each LLM answer ("Choice 1" / "Choice 2") turns a triplet into a
``(query, positive, negative)`` example in the JSONL format expected by
FlagEmbedding's fine-tuning entry points.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

_NUMBER_MAPPING = {"1": 1, "one": 1, "2": 2, "two": 2}
_NEITHER = re.compile(r"neither", re.IGNORECASE)


def parse_choice(response: str) -> Optional[int]:
    """Parse an LLM response into ``1``, ``2``, or ``None``.

    ``None`` is returned for a "Neither" answer or anything unrecognized, so the
    caller can skip degenerate triplets.
    """
    if _NEITHER.search(response):
        return None
    token = response.replace(".", "").split(" ")[-1].lower()
    return _NUMBER_MAPPING.get(token)


def _passage(title: str, abstract: str) -> str:
    return f"Title: {title}, Abstract: {abstract}"


class FineTuneDataBuilder:
    """Turn LLM query results into FlagEmbedding fine-tuning examples.

    Args:
        query_results: The list returned by an LLM query backend; each item
            carries ``idx``/``choice1_idx``/``choice2_idx`` and ``llm_response``.
        titles: Paper titles indexed by paper id.
        abstracts: Paper abstracts indexed by paper id.
    """

    def __init__(
        self,
        query_results: list[dict],
        titles: list[str],
        abstracts: list[str],
    ):
        self.query_results = query_results
        self.titles = titles
        self.abstracts = abstracts

    def build(self) -> tuple[list[dict], int, int]:
        """Build ``(examples, max_query_len, max_passage_len)``.

        ``max_query_len`` / ``max_passage_len`` are word counts useful for
        choosing ``--query_max_len`` / ``--passage_max_len`` when fine-tuning.
        """
        examples: list[dict] = []
        max_query_len = 0
        max_passage_len = 0

        for item in self.query_results:
            choice = parse_choice(item.get("llm_response", ""))
            if choice not in (1, 2):
                continue

            anchor = _passage(self.titles[item["idx"]], self.abstracts[item["idx"]])
            c1 = _passage(
                self.titles[item["choice1_idx"]], self.abstracts[item["choice1_idx"]]
            )
            c2 = _passage(
                self.titles[item["choice2_idx"]], self.abstracts[item["choice2_idx"]]
            )
            pos, neg = (c1, c2) if choice == 1 else (c2, c1)

            examples.append({"query": anchor, "pos": [pos], "neg": [neg]})
            max_query_len = max(max_query_len, len(anchor.split(" ")))
            max_passage_len = max(
                max_passage_len,
                len(pos.split(" ")),
                len(neg.split(" ")),
                len(anchor.split(" ")),
            )

        return examples, max_query_len, max_passage_len

    @staticmethod
    def save(examples: list[dict], save_dir: str) -> str:
        """Write examples as ``finetune_data.jsonl`` in ``save_dir``.

        Returns:
            The full path to the written JSONL file.
        """
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "finetune_data.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
        return path
