"""QA tool wrappers — the read-only StructuredTools the model calls on the question path.

These assert each tool renders its port's data for the model and exposes the schema the
model binds against; the agentic loop that *calls* them is covered in test_qa_agent.
"""

from src.app.graph.subgraphs.tools import build_escalation_tool, build_order_desk_tools
from src.domain.models import ResolutionCandidate
from test.fakes import FakeCatalog, FakeEscalation, FakeSupplier
from test.fakes import catalog_item as _item


def _tools(catalog=None, supplier=None):
    tools = build_order_desk_tools(catalog or FakeCatalog(), supplier or FakeSupplier())
    return {t.name: t for t in tools}


def test_exposes_the_three_read_only_tools_with_names_and_schemas():
    tools = _tools()
    assert set(tools) == {"search_catalog", "check_inventory", "get_price"}
    for tool in tools.values():
        assert tool.description  # the description is what the model sees
        assert tool.args_schema is not None


def test_search_catalog_renders_matching_candidates_with_their_skus():
    deli = _item("DELI-16", "16oz Deli Container", min_order=2)  # helper fixes case_pack=100
    catalog = FakeCatalog(
        candidates_by_phrase={"16oz deli": [ResolutionCandidate(item=deli, score=0.9)]}
    )
    out = _tools(catalog=catalog)["search_catalog"].invoke({"query": "16oz deli"})
    assert "DELI-16" in out and "16oz Deli Container" in out and "100 per case" in out


def test_search_catalog_reports_when_nothing_matches():
    out = _tools()["search_catalog"].invoke({"query": "unicorn meat"})
    assert "no matching" in out.lower()


def test_check_inventory_reports_in_stock_and_out_of_stock():
    in_stock = _tools()["check_inventory"].invoke({"sku": "DELI-16"})
    assert "in stock" in in_stock.lower()
    out = _tools(supplier=FakeSupplier(out_of_stock={"LIME-FRESH"}))["check_inventory"]
    assert "out of stock" in out.invoke({"sku": "LIME-FRESH"}).lower()


def test_get_price_renders_the_price_or_says_unpriced():
    priced = _tools(supplier=FakeSupplier(prices={"STRAW-WRAP": 16.60}))["get_price"]
    assert "16.60" in priced.invoke({"sku": "STRAW-WRAP"})
    assert "not priced" in _tools()["get_price"].invoke({"sku": "STRAW-WRAP"}).lower()


def test_escalate_to_human_opens_a_handoff_ticket_and_renders_it():
    # The escalation tool is built separately from the three read-only data tools
    # (it's a write), so the read-only set above stays exactly three.
    tool = build_escalation_tool(FakeEscalation())[0]
    assert tool.name == "escalate_to_human"
    assert tool.args_schema is not None
    out = tool.invoke({"reason": "wants to return a case of limes"})
    assert "TEST-HANDOFF" in out and "specialist" in out.lower()