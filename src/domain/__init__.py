"""Domain layer — pure Python + Pydantic, zero LLM/LangChain imports.

Domain types are the lingua franca of the agent: nodes accept and return these,
adapters translate provider-specific shapes into them, and tests construct them
directly. The deterministic business logic (redaction, rules, cart, resolution)
lives here and is unit-tested test-first.

Consumers import from the concrete submodule (e.g. ``from src.domain.models
import LineItem``) so the package root stays free of import-order coupling as
modules land across the test-first build.
"""