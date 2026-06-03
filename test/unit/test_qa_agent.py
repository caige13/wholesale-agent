"""QA subgraph — the question-path tool-calling loop, run in isolation (fake model).

Exercises the real compiled subgraph (assistant <-> tools) deterministically: a scripted
model stands in for Gemini and emits tool calls / a final answer, while the read-only
tools run against in-memory fakes. Skipped when langgraph isn't installed.
"""

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import ToolMessage  # noqa: E402

from src.app.graph.subgraphs.qa_agent import build_qa_agent  # noqa: E402
from src.app.graph.subgraphs.tools import build_order_desk_tools  # noqa: E402
from src.domain.models import Intent  # noqa: E402
from test.fakes import FakeCatalog, FakeSupplier, ScriptedModel  # noqa: E402


def _qa(model, catalog=None, supplier=None):
    tools = build_order_desk_tools(catalog or FakeCatalog(), supplier or FakeSupplier())
    return build_qa_agent(model, tools)


def test_answers_directly_when_the_model_calls_no_tools():
    qa = _qa(ScriptedModel(Intent.QUESTION, answer="Each case has 500 units."))
    final = qa.invoke({"clean_message": "how many per case?"})
    assert final["answer"] == "Each case has 500 units."


def test_calls_a_tool_then_answers_from_its_result():
    # The model asks check_inventory, the ToolNode runs it, then the model answers.
    model = ScriptedModel(
        Intent.QUESTION,
        tool_steps=[
            [{"name": "check_inventory", "args": {"sku": "DELI-16"}}],
            "Yes, the 16oz deli container is in stock.",
        ],
    )
    final = _qa(model).invoke({"clean_message": "is the 16oz deli in stock?"})
    assert final["answer"] == "Yes, the 16oz deli container is in stock."
    tool_msgs = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs and "in stock" in tool_msgs[0].content.lower()  # the tool actually ran


def test_flattens_list_content_blocks_into_a_string_answer():
    # Gemini returns content as a list of blocks; the answer must still be a string.
    model = ScriptedModel(Intent.QUESTION, answer=[{"type": "text", "text": "500 per case."}])
    final = _qa(model).invoke({"clean_message": "how many per case?"})
    assert final["answer"] == "500 per case."