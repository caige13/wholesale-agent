# AI Order Desk — dev tasks. Thin wrappers over uv so the commands are
# discoverable and CI/README can reference one canonical entrypoint.
#
# No make or uv? Use the stdlib task runner instead — same commands, only a
# Python interpreter required:  python run.py setup  &&  python run.py ui

.PHONY: setup setup-agent test lint eval eval-langsmith ui requirements

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

# Regenerate the pinned requirements*.txt from uv.lock so the pip path (run.py)
# never drifts from the lockfile. Run this after changing dependencies.
#
# The CPU torch wheels (torch==…+cpu on Linux) live on PyTorch's CPU index, not
# PyPI, and `uv export` can't emit a package-scoped index into pip's flat format —
# so prepend the index URL to both files, leaving them installable with a plain
# `pip install -r requirements.txt`.
PYTORCH_CPU_INDEX = https://download.pytorch.org/whl/cpu
requirements:
	uv export --no-hashes --no-emit-project --no-dev --extra agent -o requirements.txt
	uv export --no-hashes --no-emit-project --extra agent -o requirements-dev.txt
	@for f in requirements.txt requirements-dev.txt; do \
		printf -- '--extra-index-url $(PYTORCH_CPU_INDEX)\n' | cat - $$f > $$f.tmp && mv $$f.tmp $$f; \
	done