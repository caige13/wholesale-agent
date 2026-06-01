# CLAUDE.md

Guidance for working in this repository.

## Git commits

Do **not** attribute commits to Claude. Commit messages must not include a
`Co-Authored-By: Claude ...` trailer, "Generated with Claude Code" lines, or any
other reference mapping the commit back to Claude. Write the message as the
human author's own.

## GitHub

Do **not** attribute pull requests (or any GitHub content) to Claude. PR titles,
descriptions, and comments must not include "Generated with Claude Code",
a Claude co-author trailer, or any other reference mapping the work back to
Claude. Write them as the human author's own.

## Testing

### No real LLM in unit/contract/integration tests

Only `eval`-marked tests may hit a real model. Everywhere else, inject a scripted
fake chat model via `build_graph(model, …)` (see `ScriptedModel` in
`test/integration/test_order_graph.py`) — the deterministic nodes already run for
free; the fake covers the LLM nodes, so the suite stays keyless and costs nothing.

### Regression evals for fixed bugs

Every time we fix a **behavioral/quality bug the agent got wrong in conversation**
(SKU resolution, clarification timing, parsing, a multi-turn follow-up), add a row
to `evals/datasets/order_desk.jsonl` that reproduces the scenario, so the eval set
guards against regression. Keep the keyless unit/contract/integration test too —
the eval row is the end-to-end, real-model guard, not a replacement for it.

- Use a behavioral `id` (e.g. `size_not_in_family`, `companion_confirm_yes`).
- For a **multi-turn** bug, add a `history` field (the runner passes
  `row.get("history")` to `agent.run`) and seed prior state via `cart_before`.
- `expected.skus`/`suppliers`/`quantities` describe the **full** resulting cart
  (Jaccard-scored); `expects_clarification` asserts whether it should ask.

### Test naming convention

Test names should describe the **behavior** under test, not just repeat the
name of the function being called. Use `test/unit/test_validate_rules.py` as the
reference example.

Pick whichever semantic pattern fits the test:

- **Behavioral** — `test_[entity]_[behavior]`
  e.g. `test_user_profile_returns_404_when_user_not_found`
- **Outcome-oriented** — `test_should_[expected_outcome]_when_[condition]`
  e.g. `test_should_reject_invalid_email_format`
- **Contextual** — `test_[method_under_test]_[condition]`
  e.g. `test_calculate_tax_applies_correct_rate_for_state`

Rules:

- **Drop the entity name when it's the test file's own entity.** In
  `test_cart.py` the cart is implied, so prefer
  `test_appends_a_new_line_when_adding_an_unseen_sku` over
  `test_cart_appends_...`. Only name an entity when it's a *different* one
  (e.g. the `CatalogRepository` port in `test_catalog_loader.py`, or
  `TurnResult` in `test_turn.py`).
- **Keep the method-under-test name** when it's the contextual subject
  (e.g. `test_set_quantity_replaces_the_line_rather_than_appending`,
  `test_get_returns_none_for_an_unknown_sku`).
- Write verb-first, readable names that state the condition and the expected
  outcome — a reader should understand the case without reading the body.
