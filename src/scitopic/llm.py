"""LLM querying for topic-similarity preferences.

Given the triplet queries produced by :mod:`scitopic.sampling`, ask an LLM which
of two candidate papers is topically closest to the anchor. Two backends are
provided: an OpenAI-compatible HTTP endpoint (``LLMQueryOnline``) and a local
vLLM engine (``LLMQueryLocal``).
"""

from __future__ import annotations

from typing import Optional

import joblib
from tqdm import tqdm

_DEFAULT_PROMPT = (
    "Select the title and abstract of the paper that better corresponds with "
    "the Query in terms of scientific topic. You must choose 'Choice 1', "
    "'Choice 2' or 'Neither'. Only answer 'Choice 1', 'Choice 2' or 'Neither'."
)


def _query_one(item: dict, model: str, client_info: dict, prompt: str) -> dict:
    """Send a single query to an OpenAI-compatible endpoint.

    A fresh client is created per call so the function is safe to run under
    ``joblib`` process-based parallelism.
    """
    from openai import OpenAI

    client = OpenAI(**client_info)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": item["prompt"]},
            ],
        )
        item["llm_response"] = completion.choices[0].message.content
    finally:
        client.close()
    return item


class LLMQueryOnline:
    """Query an OpenAI-compatible HTTP endpoint in parallel.

    Args:
        api_key: API key for the endpoint. Read from configuration/environment;
            never hard-code it.
        base_url: Base URL of the OpenAI-compatible server.
        model: Model name to request.
        n_jobs: Number of parallel workers.
        prompt: System prompt; falls back to a built-in default when ``None``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "Meta-Llama-3.1-70B-Instruct",
        n_jobs: int = 8,
        prompt: Optional[str] = None,
    ):
        self.model = model
        self.n_jobs = n_jobs
        self.prompt = prompt or _DEFAULT_PROMPT
        self.client_info = {"api_key": api_key, "base_url": base_url}

    def __call__(self, queries: list[dict]) -> list[dict]:
        """Answer every query, attaching an ``llm_response`` field to each."""
        return joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(_query_one)(item, self.model, self.client_info, self.prompt)
            for item in tqdm(queries, desc="llm query")
        )


class LLMQueryLocal:
    """Query a locally hosted model through vLLM.

    Args:
        model_path: Path to (or HuggingFace id of) the model.
        max_model_len: Maximum context length.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        prompt: System prompt; falls back to a built-in default when ``None``.
    """

    def __init__(
        self,
        model_path: str,
        max_model_len: int = 9216,
        tensor_parallel_size: int = 1,
        prompt: Optional[str] = None,
    ):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            max_seq_len_to_capture=max_model_len,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0, stop=["</s>"], max_tokens=1024
        )
        self.prompt = prompt or _DEFAULT_PROMPT

    def __call__(self, queries: list[dict]) -> list[dict]:
        """Answer every query, attaching an ``llm_response`` field to each."""
        results = []
        for item in tqdm(queries, desc="llm query"):
            response = self.llm.chat(
                [
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": item["prompt"]},
                ],
                self.sampling_params,
                use_tqdm=False,
            )
            item["llm_response"] = response[0].outputs[0].text
            results.append(item)
        return results
