# AI Order Desk — dev tasks. Thin wrappers over uv so the commands are
# discoverable and CI/README can reference one canonical entrypoint.

.PHONY: setup setup-agent test lint eval eval-langsmith ui

# Core dev env: domain + deterministic core + test tooling. Fast, no API key.
setup:
	uv sync

# Full agent runtime (LLM, RAG, UI). Needs API keys — used by later slices.
setup-agent:
	uv sync --extra agent

# Unit + integration tests. Evals are skipped by default (see pyproject addopts).
test:
	uv run pytest

lint:
	uv run ruff check .

# Launch the Gradio order desk (needs the agent extra + GOOGLE_API_KEY).
ui:
	uv run python -m src.interfaces.gradio_app

# Local eval runner — scores the dataset and prints the metrics. Needs
# GOOGLE_API_KEY (agent) and OPENAI_API_KEY (the cross-model faithfulness judge).
eval:
	uv run python -m evals.run_eval

# LangSmith-native eval — upserts the dataset and runs a versioned experiment in
# the UI over the same metrics. Also needs LANGSMITH_API_KEY + LANGSMITH_TRACING=true.
eval-langsmith:
	uv run python -m evals.langsmith_eval