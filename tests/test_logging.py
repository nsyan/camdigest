# tests/test_logging.py —— 统一日志初始化 + 编排器阶段日志
import logging

import pytest

from camdigest.logging_setup import setup_logging


@pytest.fixture(autouse=True)
def _fresh_camdigest_logger():
    """setup_logging 写的是进程级全局 logger：每个用例前清空，避免跨文件污染。"""
    log = logging.getLogger("camdigest")
    log.handlers.clear()
    yield
    log.handlers.clear()


def test_setup_idempotent(tmp_path):
    setup_logging(tmp_path)
    setup_logging(tmp_path)                      # 重复调用不叠加
    log = logging.getLogger("camdigest")
    assert len(log.handlers) == 2                # stdout + 文件
    assert (tmp_path / "logs" / "camdigest.log").exists()


def test_run_day_stage_logs(caplog, tmp_path):
    from camdigest import db
    from camdigest.config import Settings
    from camdigest.pipeline import orchestrator as orch

    setup_logging(tmp_path)
    settings = Settings(
        cameras=[{"id": "gate", "name": "大门", "device": "gate",
                  "lens": "single", "dir": "/a"}],
        models={"recognition": {"base_url": "http://x", "model": "m", "api_key": "k"},
                "report": {"base_url": "http://x", "model": "m", "api_key": "k"}},
        storage={"data_dir": str(tmp_path)})
    url = f"sqlite:///{tmp_path}/t.db"
    db.init_db(url)
    calls = []

    def fake(name):
        return lambda date, session, s: calls.append(name) or 1

    stages = [(n, fake(n)) for n in ["s1_index", "s2_prefilter"]]
    with db.session_scope(url) as s, caplog.at_level(logging.INFO, logger="camdigest"):
        orch.run_day("2026-09-29", settings, session=s, stages=stages)
    msgs = [r.message for r in caplog.records]
    assert any("run_day 2026-09-29 开始" in m for m in msgs)
    assert any("s1_index done" in m for m in msgs)


def test_llm_retry_logged(caplog, video_file, tmp_path):
    import httpx

    from camdigest.config import ModelCfg
    from camdigest.llm.contracts import SegmentHint
    from camdigest.llm.openai_adapter import OpenAIRecognition

    n = {"n": 0}

    def handler(request):
        n["n"] += 1
        content = "junk" if n["n"] < 3 else (
            '{"category":"empty","people":[],"score":10,"title":"t","description":"d"}')
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    r = OpenAIRecognition(ModelCfg(base_url="http://fake/v1", model="m", api_key="k"))
    r._http = httpx.Client(transport=httpx.MockTransport(handler))
    with caplog.at_level(logging.WARNING, logger="camdigest"):
        r.analyze(__import__("pathlib").Path(video_file), SegmentHint(camera="x", lens="single"))
    assert any("LLM 重试" in rec.message for rec in caplog.records)
