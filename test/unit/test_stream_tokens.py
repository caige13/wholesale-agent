"""Streamed-token filtering — only freshly generated tokens reach the chat bubble.

The QA subgraph's ``messages`` channel carries the seeded human message, the seeded
prior turns (full AIMessages), and tool results — none of which are this turn's answer.
Only the live LLM's ``AIMessageChunk`` tokens may stream (the regression where a turn
echoed the user's text, tool results, or the whole prior conversation into the bubble).
"""

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from src.app.graph.agent import _token_text


def test_streams_freshly_generated_token_chunks():
    assert _token_text((AIMessageChunk(content="Hello "), {})) == "Hello "


def test_skips_the_seeded_human_message():
    assert _token_text((HumanMessage(content="where can i pick it up?"), {})) == ""


def test_skips_seeded_history_ai_messages():
    # Prior turns are seeded as full AIMessages, not generated chunks — they must not
    # stream (the bug where earlier assistant replies echoed into the new bubble).
    assert _token_text((AIMessage(content="Hello! I am your assistant."), {})) == ""


def test_skips_tool_results():
    tool_msg = ToolMessage(content="Escalated to a specialist", tool_call_id="x")
    assert _token_text((tool_msg, {})) == ""


def test_skips_an_empty_tool_call_chunk():
    # The assistant's tool-deciding chunk carries no text — nothing to stream.
    assert _token_text((AIMessageChunk(content=""), {})) == ""


def test_flattens_gemini_content_blocks():
    assert _token_text((AIMessageChunk(content=[{"type": "text", "text": "Hi"}]), {})) == "Hi"