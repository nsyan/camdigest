# tests/test_s8.py —— S8 日报：build_payload 确定性 + run_report 落盘（模型假体）
from datetime import UTC, datetime, timedelta

from camdigest.config import Settings
from camdigest.pipeline.s8_report import build_payload, build_report_model


def _settings():
    return Settings(
        cameras=[{"id": "gate", "name": "大门", "device": "gate",
                  "lens": "single", "dir": "/a"}],
        models={"recognition": {"base_url": "http://x", "model": "m", "api_key": "k"},
                "report": {"protocol": "anthropic", "base_url": "http://y",
                           "model": "m2", "api_key": "k2"}},
        storage={"data_dir": "/data"})


def test_build_payload_fields_and_times():
    t0 = datetime(2026, 9, 29, 10, 5, tzinfo=UTC)
    ev = {"id": 7, "start_ts": t0, "end_ts": t0 + timedelta(minutes=2),
          "category": "family", "score": 85, "title": "妈妈带小宝回家",
          "description": "从大门进入", "people": ["妈妈"], "is_anomaly": False,
          "keyframe_path": "/data/keyframes/2026-09-29/ev7.jpg"}
    payload = build_payload([ev], _settings())
    assert payload["highlight_paths"] == [
        "/data/highlights/2026-09-29/精华_5min.mp4",
        "/data/highlights/2026-09-29/精华_10min.mp4",
        "/data/highlights/2026-09-29/精华_30min.mp4",
        "/data/highlights/2026-09-29/精华_60min.mp4",
    ]
    e = payload["events"][0]
    assert e["id"] == 7 and e["time"] == "10:05"      # HH:MM
    assert e["title"] == "妈妈带小宝回家" and e["category"] == "family"
    assert e["people"] == ["妈妈"] and e["is_anomaly"] is False
    assert e["has_keyframe"] is True and e["keyframe"] == ev["keyframe_path"]


def test_build_report_model_anthropic():
    from camdigest.llm.anthropic_adapter import AnthropicReport
    assert isinstance(build_report_model(_settings()), AnthropicReport)


def test_build_report_model_openai_default():
    from camdigest.llm.openai_adapter import OpenAIReport
    s = _settings()
    s.models.report.protocol = "openai"
    assert isinstance(build_report_model(s), OpenAIReport)


def test_run_report_writes_md_and_row(tmp_path, monkeypatch):
    from camdigest import db
    from camdigest.pipeline import s8_report

    class FakeReport:
        def write(self, date, payload):
            assert payload["events"] and payload["highlight_paths"]
            return "## 今日总览\n\n安好。\n\n## 时间线\n\n## 人物出没\n\n## 异常事件\n\n## 精华清单\n"

    monkeypatch.setattr(s8_report, "build_report_model", lambda s: FakeReport())
    settings = _settings()
    settings.storage.data_dir = tmp_path
    url = f"sqlite:///{tmp_path}/t.db"
    db.init_db(url)
    t0 = datetime(2026, 9, 29, 10, 5, tzinfo=UTC)
    with db.session_scope(url) as s:
        s.add(db.Event(date="2026-09-29", start_ts=t0, end_ts=t0 + timedelta(minutes=2),
                       category="family", score=85, title="妈妈带小宝回家",
                       description="从大门进入", camera_ids=["gate"], segment_ids=[1],
                       is_anomaly=False))
    with db.session_scope(url) as s:
        out = s8_report.run_report("2026-09-29", s, settings)
    assert out.exists() and "## 时间线" in out.read_text(encoding="utf-8")
    with db.session_scope(url) as s:
        row = s.query(db.Report).one()
        assert row.date == "2026-09-29" and str(out) == row.md_path
