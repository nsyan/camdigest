# tests/test_contracts.py
import pytest

from camdigest.llm.contracts import EventDraft, RecognitionError


def test_draft_from_json():
    d = EventDraft.from_json("""{"category": "family", "people": ["妈妈"],
        "score": 85, "title": "妈妈带小宝回家", "description": "..."}""")
    assert d.category == "family" and d.score == 85


def test_draft_rejects_bad_category():
    with pytest.raises(ValueError):
        EventDraft.from_json('{"category": "ufo", "people": [], "score": 1, "title": "t", "description": "d"}')


def test_draft_strips_code_fence():
    d = EventDraft.from_json('```json\n{"category":"empty","people":[],"score":10,"title":"t","description":"d"}\n```')
    assert d.category == "empty"


def test_draft_clamps_score():
    hi = EventDraft(category="animal", people=[], score=150, title="t", description="d")
    lo = EventDraft(category="animal", people=[], score=-5, title="t", description="d")
    assert hi.score == 100 and lo.score == 0


def test_recognition_error_is_exception():
    assert issubclass(RecognitionError, Exception)
