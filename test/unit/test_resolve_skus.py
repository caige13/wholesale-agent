"""resolve_skus — pick a SKU from retriever candidates (spec §4).

Mostly pure: given a line's phrase and scored candidates, choose one SKU and set
confidence (which the gate later uses). Precedence: item-memory > exact alias >
a clearly-best candidate; two close (or weak) candidates stay unresolved at low
confidence so the gate asks. Candidates are passed in, so no embeddings are needed.
"""

from src.domain.models import LineItem
from src.domain.resolution import resolve_skus


def test_resolves_a_single_strong_match_with_high_confidence(make_candidate):
    item = LineItem(raw_text="16oz deli")
    candidates = [make_candidate(score=0.92, sku="DELI-16", product_name="16oz Deli Container")]
    out = resolve_skus(item, candidates)
    assert out.sku == "DELI-16"
    assert out.confidence >= 0.6


def test_sets_product_name_and_supplier_from_the_resolved_item(make_candidate):
    item = LineItem(raw_text="16oz deli")
    candidates = [make_candidate(score=0.92, sku="DELI-16", product_name="16oz Deli Container")]
    out = resolve_skus(item, candidates)
    assert out.product_name == "16oz Deli Container"
    assert out.supplier == "acme-foodservice"


def test_resolves_via_an_exact_alias_match(make_candidate):
    item = LineItem(raw_text="salsa cups")
    candidates = [
        make_candidate(
            score=0.55, sku="PCUP-2", product_name="2oz Portion Cup", aliases=["salsa cups"]
        ),
        make_candidate(score=0.5, sku="PCUP-4", product_name="4oz Portion Cup"),
    ]
    out = resolve_skus(item, candidates)
    assert out.sku == "PCUP-2"
    assert out.confidence >= 0.6


def test_resolves_a_size_qualified_phrase_when_an_alias_is_contained_in_it(make_candidate):
    # "16oz deli containers" must land on DELI-16 even though the deli sizes cluster
    # close together — the size qualifier disambiguates via alias containment.
    item = LineItem(raw_text="16oz deli containers")
    candidates = [
        make_candidate(
            score=0.85, sku="DELI-08", product_name="8oz Deli Container",
            aliases=["8oz deli container"],
        ),
        make_candidate(
            score=0.84, sku="DELI-16", product_name="16oz Deli Container",
            aliases=["16oz deli container"],
        ),
    ]
    assert resolve_skus(item, candidates).sku == "DELI-16"


def test_prefers_item_memory_over_the_catalog_candidates(make_candidate):
    item = LineItem(raw_text="deli")
    candidates = [  # genuinely ambiguous on their own
        make_candidate(score=0.81, sku="DELI-08", product_name="8oz Deli Container"),
        make_candidate(score=0.80, sku="DELI-16", product_name="16oz Deli Container"),
    ]
    out = resolve_skus(item, candidates, item_memory={"deli": "DELI-16"})
    assert out.sku == "DELI-16"
    assert out.confidence >= 0.9


def test_returns_low_confidence_when_two_candidates_are_too_close(make_candidate):
    item = LineItem(raw_text="deli containers")
    candidates = [
        make_candidate(score=0.84, sku="DELI-08", product_name="8oz Deli Container"),
        make_candidate(score=0.82, sku="DELI-16", product_name="16oz Deli Container"),
    ]
    out = resolve_skus(item, candidates)
    assert out.confidence < 0.6


def test_leaves_the_sku_unresolved_when_candidates_are_ambiguous(make_candidate):
    item = LineItem(raw_text="deli containers")
    candidates = [
        make_candidate(score=0.84, sku="DELI-08", product_name="8oz Deli Container"),
        make_candidate(score=0.82, sku="DELI-16", product_name="16oz Deli Container"),
    ]
    assert resolve_skus(item, candidates).sku is None


def test_resolves_by_size_token_when_candidates_share_a_family(make_candidate):
    # "16oz deli" (no "container") doesn't alias-match and the sizes cluster — but
    # the 16oz token matches exactly one candidate's unit_size, which disambiguates.
    item = LineItem(raw_text="16oz deli")
    candidates = [
        make_candidate(score=0.84, sku="DELI-08", product_name="8oz Deli", unit_size="8oz"),
        make_candidate(score=0.83, sku="DELI-16", product_name="16oz Deli", unit_size="16oz"),
    ]
    out = resolve_skus(item, candidates)
    assert out.sku == "DELI-16"
    assert out.confidence >= 0.6


def test_attaches_the_candidate_options_when_ambiguous(make_candidate):
    item = LineItem(raw_text="deli containers")
    candidates = [
        make_candidate(score=0.84, sku="DELI-08", product_name="8oz Deli Container"),
        make_candidate(score=0.82, sku="DELI-16", product_name="16oz Deli Container"),
    ]
    options = resolve_skus(item, candidates).options
    assert "8oz Deli Container" in options
    assert "16oz Deli Container" in options


def test_returns_low_confidence_when_the_best_score_is_weak(make_candidate):
    item = LineItem(raw_text="thingamajig")
    out = resolve_skus(item, [make_candidate(score=0.2, sku="DELI-16")])
    assert out.confidence < 0.6


def test_returns_zero_confidence_when_there_are_no_candidates():
    out = resolve_skus(LineItem(raw_text="unknown item"), [])
    assert out.sku is None
    assert out.confidence == 0.0


def test_returns_a_new_item_without_mutating_the_input(make_candidate):
    item = LineItem(raw_text="16oz deli")
    out = resolve_skus(item, [make_candidate(score=0.92, sku="DELI-16")])
    assert out is not item
    assert item.sku is None
    assert item.confidence == 0.0