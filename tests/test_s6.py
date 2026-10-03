# tests/test_s6.py
import pytest

from camdigest.pipeline.s6_recognize import hybrid_score


@pytest.mark.parametrize("category,model,expect", [
    ("family", 95, 95),      # 95 >= 70+20=90 → 采信
    ("family", 89, 70),      # 89 < 90        → 规则兜底
    ("empty", 90, 90),
    ("empty", 5, 10),
    ("stranger", 40, 50),
])
def test_hybrid_score(category, model, expect):
    assert hybrid_score(category, model) == expect


def test_hybrid_score_unknown_category_falls_back():
    from camdigest.pipeline.s6_recognize import RULE_SCORE
    assert set(RULE_SCORE) == {"family", "stranger", "visitor", "animal", "vehicle", "empty"}
