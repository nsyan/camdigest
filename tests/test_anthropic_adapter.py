# tests/test_anthropic_adapter.py
import httpx

from camdigest.config import ModelCfg
from camdigest.llm.anthropic_adapter import AnthropicReport

CFG = ModelCfg(protocol="anthropic", base_url="http://fake", model="m", api_key="k")


def test_write_messages_api_shape():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "# 日报"}]})

    r = AnthropicReport(CFG)
    r._http = httpx.Client(transport=httpx.MockTransport(handler))
    out = r.write("2026-09-29", {"events": []})
    assert out == "# 日报"
    req = calls[0]
    assert req.url.path.endswith("/v1/messages")
    assert req.headers["x-api-key"] == "k"
    assert req.headers["anthropic-version"] == "2023-06-01"
    import json
    body = json.loads(req.content)
    assert body["model"] == "m" and body["max_tokens"] == 4096
    assert body["system"]  # REPORT_SYSTEM 注入 system 字段
    assert body["messages"][0]["role"] == "user" and "2026-09-29" in body["messages"][0]["content"]


def test_write_joins_text_blocks():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"content": [{"type": "text", "text": "a"},
                                                     {"type": "tool_use", "id": "x"},
                                                     {"type": "text", "text": "b"}]})

    r = AnthropicReport(CFG)
    r._http = httpx.Client(transport=httpx.MockTransport(handler))
    assert r.write("2026-09-29", {}) == "ab"
