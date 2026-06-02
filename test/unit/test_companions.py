"""companion_case_count — the deterministic add-on sizing math.

The LLM does no companion math: this pure function sizes the add-on order to cover
the parent without going under. *Which* companions are still under-covered (a
catalog-dependent question) is tested via the graph-node helpers
(``_undercovered_companions`` / ``pending_companions``) in ``test_graph_nodes.py``.
"""

from src.domain.companions import companion_case_count


def test_sizes_a_companion_to_cover_the_parent_without_going_under():
    # 3 cases of 32oz deli = 1440 containers; lids packed 500/case → 3 (1500 ≥ 1440).
    assert companion_case_count(1440, 500) == 3


def test_rounds_up_to_the_next_whole_case_rather_than_under():
    assert companion_case_count(1001, 500) == 3  # 2 cases (1000) would be short


def test_returns_at_least_one_case_for_an_exact_or_unknown_parent():
    assert companion_case_count(1000, 500) == 2
    assert companion_case_count(0, 500) == 1  # unknown parent count → still offer a case