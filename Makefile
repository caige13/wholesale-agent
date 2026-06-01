# AI Order Desk — dev tasks. Thin wrappers over uv so the commands are
# discoverable and CI/README can reference one canonical entrypoint.

.PHONY: setup setup-agent test lint eval ui

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

# LangSmith eval harness — deferred to the eval slice.
eval:
	@echo "eval harness deferred — see docs/spec.md §9 and evals/datasets/order_desk.jsonl"