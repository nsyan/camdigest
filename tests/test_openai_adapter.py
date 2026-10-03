# tests/test_openai_adapter.py
import json

import httpx
import pytest

from camdigest.config import ModelCfg
from camdigest.llm.contracts import RecognitionError, SegmentHint
from camdigest.llm.openai_adapter import OpenAIRecognition

CFG = ModelCfg(base_url="http://fake/v1", model="m", api_key="k")

GOOD = {"choices": [{"message": {"content":
        '{"category":"family","people":["妈妈"],"score":85,"title":"t","description":"d"}'}}]}


def _client(handler) -> OpenAIRecognition:
    transport = httpx.MockTransport(handler)
    r = OpenAIRecognition(CFG)
    r._http = httpx.Client(transport=transport)   # 测试注入口
    return r


def test_analyze_ok(video_file, tmp_path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=GOOD)

    draft = _client(handler).analyze(__import__("pathlib").Path(video_file),
                                     SegmentHint(camera="大厅", lens="single"))
    assert draft.category == "family"
    body = calls[0]
    parts = body["messages"][1]["content"]
    kinds = {p.get("type") for p in parts}
    assert "video_url" in kinds and "input_audio" in kinds and "text" in kinds
    assert body["response_format"] == {"type": "json_object"}


def test_analyze_retries_on_bad_json(video_file, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)  # 重试退避不真睡，保持套件快速
    n = {"n": 0}

    def handler(request):
        n["n"] += 1
        content = "not json" if n["n"] <= 2 else GOOD["choices"][0]["message"]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    draft = _client(handler).analyze(__import__("pathlib").Path(video_file),
                                     SegmentHint(camera="x", lens="ptz"))
    assert n["n"] == 3 and draft.score == 85


def test_analyze_gives_up_after_2_retries(video_file, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)  # 重试退避不真睡，保持套件快速

    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "junk"}}]})

    with pytest.raises(RecognitionError):
        _client(handler).analyze(__import__("pathlib").Path(video_file),
                                 SegmentHint(camera="x", lens="single"))


def test_analyze_retries_on_429(video_file, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)  # 5xx/429 指数退避不真睡，保持套件快速
    n = {"n": 0}

    def handler(request):
        n["n"] += 1
        if n["n"] == 1:
            return httpx.Response(429, json={"error": "rate"})
        return httpx.Response(200, json=GOOD)

    draft = _client(handler).analyze(__import__("pathlib").Path(video_file),
                                     SegmentHint(camera="x", lens="single"))
    assert n["n"] == 2 and draft.score == 85


def test_analyze_4xx_raises_immediately(video_file):
    n = {"n": 0}

    def handler(request):
        n["n"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    with pytest.raises(httpx.HTTPStatusError):
        _client(handler).analyze(__import__("pathlib").Path(video_file),
                                 SegmentHint(camera="x", lens="single"))
    assert n["n"] == 1  # 4xx 不重试，即抛
