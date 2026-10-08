# tests/test_web_e2e.py —— M2 全链路：点播 → 补跑 → 人脸归档（模型全假体）
from datetime import UTC, datetime, timedelta

import pytest
from starlette.testclient import TestClient

from camdigest import db
from camdigest.web.app import create_app
from tests.conftest import make_video
from tests.test_web_index import _settings


@pytest.fixture
def e2e(tmp_path, monkeypatch):
    import camdigest.pipeline.orchestrator as orch

    settings = _settings(tmp_path)
    settings.faces.registry_dir = tmp_path / "faces"
    (settings.faces.registry_dir).mkdir(parents=True)

    # 真实产物文件 + 种子库
    hl = settings.storage.data_dir / "highlights" / "2026-09-29"
    hl.mkdir(parents=True)
    make_video(hl / "精华_5min.mp4", seconds=2)
    kf = settings.storage.data_dir / "keyframes" / "2026-09-29"
    kf.mkdir(parents=True)
    (kf / "ev1.jpg").write_bytes(b"\xff\xd8kf")
    app = create_app(settings)
    seed_url = f"sqlite:///{tmp_path}/t.db"
    app.state.db_url = seed_url
    db.init_db(seed_url)
    t0 = datetime(2026, 9, 29, 10, 0, tzinfo=UTC)
    with db.session_scope(seed_url) as s:
        s.add(db.Highlight(date="2026-09-29", camera_id=None, tier_minutes=5,
                           file_path=str(hl / "精华_5min.mp4"), duration=2.0,
                           bytes=1, event_ids=[1]))
        m = db.MediaFile(camera_id="gate", path=str(tmp_path / "gate.mp4"),
                         start_ts=t0, end_ts=t0 + timedelta(seconds=6), duration=6.0)
        s.add(m)
        s.flush()
        s.add(db.Segment(media_file_id=m.id, start_s=0, end_s=6, person_count=1))
        s.add(db.Event(date="2026-09-29", start_ts=t0,
                       end_ts=t0 + timedelta(minutes=1), category="family",
                       score=85, title="妈妈回家", description="d",
                       camera_ids=["gate"], segment_ids=[1],
                       keyframe_path=str(kf / "ev1.jpg"), is_anomaly=False))
        s.add(db.Job(date="2026-09-28", stage="s1_index", status="done"))
        s.merge(db.Report(date="2026-09-29", md_path=str(tmp_path / "r.md")))
        (tmp_path / "r.md").write_text("## 今日总览\n\n安好。", encoding="utf-8")
    make_video(tmp_path / "gate.mp4", seconds=6)

    # 补跑假体：记录日期
    ran = []
    monkeypatch.setattr(orch, "run_day",
                        lambda date, s, until=None: ran.append(date) or {})

    # 人脸扫描/归档假体
    from camdigest.pipeline.s4_face import FaceDet

    class FakeClient:
        def __init__(self, base_url=None, timeout=30.0):
            pass

        def extract(self, image_bytes):
            return [FaceDet(bbox=[0, 0, 1, 1], det_score=0.9, norm=[1.0, 0.0])]

    monkeypatch.setattr("camdigest.web.faces.FaceClient", FakeClient)
    monkeypatch.setattr("camdigest.pipeline.s4_face.FaceClient", FakeClient)
    return TestClient(app), app, settings, ran, seed_url, tmp_path


def test_web_end_to_end(e2e):
    c, app, settings, ran, seed_url, _tmp_path = e2e

    # 1. 首页列出日期
    r = c.get("/")
    assert r.status_code == 200 and "2026-09-29" in r.text

    # 2. 单日页：播放器 + 静态视频可访问
    r = c.get("/day/2026-09-29")
    assert r.status_code == 200 and "<video" in r.text
    assert c.get("/highlights/2026-09-29/精华_5min.mp4").status_code == 200

    # 3. 进度页：历史 jobs 可见
    r = c.get("/jobs")
    assert "2026-09-28" in r.text and "done" in r.text

    # 4. 手动补跑：提交 → 后台执行 → 完成后 jobs 页可见
    r = c.post("/run", data={"date": "2026-09-27"}, follow_redirects=False)
    assert r.status_code == 303
    import time
    for _ in range(100):
        if not app.state.runs.busy:
            break
        time.sleep(0.05)
    assert ran == ["2026-09-27"]

    # 5. 人脸：扫描入簇 → 归档 → 事件时间线身份可见
    from camdigest import db
    from camdigest.web.faces import scan_unknown_faces
    with db.session_scope(seed_url) as s:
        assert scan_unknown_faces("2026-09-29", s, settings) >= 1
    r = c.get("/faces")
    assert "未知脸" in r.text
    r = c.post("/faces/archive", data={"cluster_id": "1", "name": "妈妈"},
               follow_redirects=False)
    assert r.status_code == 303
    r = c.get("/faces")
    assert "妈妈" in r.text
