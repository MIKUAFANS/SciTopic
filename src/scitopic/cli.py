"""Command-line interface for SciTopic.

Exposes the pipeline stages as subcommands::

    scitopic train    [--config PATH]
    scitopic evaluate [--config PATH]

Fine-tuning between the two stages is driven by ``scripts/finetune.sh`` (a thin
wrapper over FlagEmbedding) rather than this CLI, since it typically runs on a
separate multi-GPU node.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .config import load_config
from .pipeline import run_evaluate, run_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scitopic",
        description="Enhancing topic discovery in scientific literature through LLMs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("train", "Embed, cluster, sample, query the LLM, and write fine-tune data."),
        ("evaluate", "Re-embed with the fine-tuned model, score topics, and visualize."),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument(
            "--config",
            default=None,
            help="Path to a YAML config overriding the packaged defaults.",
        )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``scitopic`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)

    if args.command == "train":
        run_train(config)
    elif args.command == "evaluate":
        run_evaluate(config)
    else:  # pragma: no cover - argparse enforces a valid command
        parser.error(f"unknown command: {args.command}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
