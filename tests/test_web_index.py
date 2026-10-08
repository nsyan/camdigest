# tests/test_web_index.py —— 应用骨架 + 首页
import shutil
from datetime import UTC, datetime

import pytest
from starlette.testclient import TestClient

from camdigest import db
from camdigest.config import Settings
from camdigest.web.app import create_app


def _settings(tmp_path):
    return Settings(
        cameras=[{"id": "gate", "name": "大门", "device": "gate",
                  "lens": "single", "dir": "/a"}],
        models={"recognition": {"base_url": "http://x", "model": "m", "api_key": "k"},
                "report": {"base_url": "http://x", "model": "m", "api_key": "k"}},
        storage={"data_dir": str(tmp_path / "data")})


@pytest.fixture
def client(tmp_path):
    settings = _settings(tmp_path)
    app = create_app(settings)
    seed_url = f"sqlite:///{tmp_path}/t.db"          # 种子库（区别于 app 默认库路径）
    app.state.db_url = seed_url
    db.init_db(seed_url)
    with db.session_scope(seed_url) as s:
        s.merge(db.Report(date="2026-09-29", md_path=str(tmp_path / "r.md")))
        s.add(db.Event(date="2026-09-29", start_ts=datetime(2026, 9, 29, 10, tzinfo=UTC),
                       end_ts=datetime(2026, 9, 29, 10, 5, tzinfo=UTC),
                       category="family", score=85, title="回家", description="d",
                       camera_ids=["gate"], segment_ids=[1], is_anomaly=False))
    shutil.copy(__file__, tmp_path / "r.md")
    return TestClient(app), settings, tmp_path


def test_index_lists_days(client):
    c, _, _ = client
    r = c.get("/")
    assert r.status_code == 200 and "2026-09-29" in r.text


def test_index_badge_counts(client):
    c, _, _ = client
    r = c.get("/")
    assert "回家" not in r.text          # 首页不展开事件明细
    assert "1" in r.text                 # 事件数徽章


def test_index_includes_days_without_report(tmp_path):
    """补跑到半程（有精华无日报）的日期也在首页可见（M2.1-1）。"""
    from camdigest import db
    from camdigest.web.app import create_app

    settings = _settings(tmp_path)
    app = create_app(settings)
    seed_url = f"sqlite:///{tmp_path}/t.db"
    app.state.db_url = seed_url
    db.init_db(seed_url)
    with db.session_scope(seed_url) as s:
        s.add(db.Highlight(date="2026-09-28", camera_id=None, tier_minutes=5,
                           file_path="/x.mp4", duration=1.0, bytes=1, event_ids=[1]))
    c = TestClient(app)
    r = c.get("/")
    assert "2026-09-28" in r.text


def test_index_report_badge(client):
    c, _, _ = client
    assert "报告" in c.get("/").text
