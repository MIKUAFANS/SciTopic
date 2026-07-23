"""Minimal end-to-end example driving the SciTopic pipeline from Python.

This mirrors the `scitopic train` / `scitopic evaluate` CLI commands but shows
how to call the pipeline programmatically, e.g. from a notebook or a larger
experiment harness.

Prerequisites:
  * `pip install -e ".[evaluation]"`
  * A `.env` file with SCITOPIC_LLM_BASE_URL (and SCITOPIC_LLM_API_KEY if needed)
  * A BGE-M3 checkpoint at the path configured in configs/default.yaml
"""

from scitopic import load_config, run_evaluate, run_train

if __name__ == "__main__":
    # Pass a path here to override the packaged defaults, e.g.
    # load_config("configs/my_experiment.yaml")
    config = load_config()

    # Stage 1: embedding -> clustering -> sampling -> LLM query -> fine-tune data
    run_train(config)

    # Stage 2 (fine-tuning) runs separately via scripts/finetune.sh.

    # Stage 3: re-embed with the fine-tuned model, score topics, render word clouds.
    run_evaluate(config)
