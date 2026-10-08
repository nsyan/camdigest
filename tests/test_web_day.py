# tests/test_web_day.py —— /day 点播页：播放器、关键帧、404
from datetime import UTC, datetime
from urllib.parse import unquote

import pytest

from tests.test_web_index import _settings


@pytest.fixture
def day_client(tmp_path):
    from camdigest import db
    from camdigest.web.app import create_app

    settings = _settings(tmp_path)
    app = create_app(settings)
    seed_url = f"sqlite:///{tmp_path}/t.db"
    app.state.db_url = seed_url
    db.init_db(seed_url)
    hl_dir = settings.storage.data_dir / "highlights" / "2026-09-29"
    kf_dir = settings.storage.data_dir / "keyframes" / "2026-09-29"
    hl_dir.mkdir(parents=True)
    kf_dir.mkdir(parents=True)
    from tests.conftest import make_video
    make_video(hl_dir / "精华_5min.mp4", seconds=2)
    (kf_dir / "ev1.jpg").write_bytes(b"\xff\xd8fake")
    with db.session_scope(seed_url) as s:
        s.add(db.Highlight(date="2026-09-29", camera_id=None, tier_minutes=5,
                           file_path=str(hl_dir / "精华_5min.mp4"), duration=2.0,
                           bytes=1, event_ids=[1]))
        s.add(db.Event(date="2026-09-29", start_ts=datetime(2026, 9, 29, 10, tzinfo=UTC),
                       end_ts=datetime(2026, 9, 29, 10, 5, tzinfo=UTC),
                       category="family", score=85, title="妈妈带小宝回家",
                       description="从大门进入", camera_ids=["gate"], segment_ids=[1],
                       keyframe_path=str(kf_dir / "ev1.jpg"), is_anomaly=False))
    from starlette.testclient import TestClient
    return TestClient(app), settings


def test_day_page_video_and_keyframe(day_client):
    c, _ = day_client
    r = c.get("/day/2026-09-29")
    assert r.status_code == 200
    assert "<video" in r.text
    assert "/highlights/2026-09-29/精华_5min.mp4" in r.text
    assert "/keyframes/2026-09-29/ev1.jpg" in r.text
    assert "妈妈带小宝回家" in r.text


def test_day_video_static_served(day_client):
    c, _ = day_client
    r = c.get("/highlights/2026-09-29/精华_5min.mp4")
    assert r.status_code == 200


def test_day_unknown_date_404(day_client):
    c, _ = day_client
    assert c.get("/day/1999-01-01").status_code == 404
    assert c.get("/day/not-a-date").status_code == 404


def test_render_report_html_links_and_headings():
    from camdigest.web.views import render_report_html

    md = ("## 精华清单\n\n见 /data/highlights/2026-09-29/精华_5min.mp4。"
          "\n\n本地 NAS 文件：/data/highlights/2026-09-29/精华_60min.mp4")
    html = unquote(render_report_html(md))       # mistune 会百分号编码非 ASCII href（账本 Task 8 裁决）
    assert "<h2>" in html
    assert 'href="/highlights/2026-09-29/精华_5min.mp4"' in html
    assert 'href="/highlights/2026-09-29/精华_60min.mp4"' in html
    assert "/data/highlights/" not in html      # 绝对路径不再出现


def test_day_page_renders_report(day_client, tmp_path):
    from camdigest import db

    c, settings = day_client
    md = ("## 今日总览\n\n安好。\n\n## 精华清单\n\n"
          "/data/highlights/2026-09-29/精华_5min.mp4")
    rp = settings.storage.data_dir / "reports" / "2026-09-29.md"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(md, encoding="utf-8")
    with db.session_scope(f"sqlite:///{tmp_path}/t.db") as s:
        s.merge(db.Report(date="2026-09-29", md_path=str(rp)))
    r = c.get("/day/2026-09-29")
    assert "<h2>今日总览</h2>" in r.text
    assert 'href="/highlights/2026-09-29/精华_5min.mp4"' in unquote(r.text)
