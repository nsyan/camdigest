# tests/test_query.py —— R1：共享查询层的最小直测（日期过滤 + pending_only）
from datetime import UTC, datetime
from pathlib import Path

from camdigest import db
from camdigest.config import CameraCfg, ModelCfg, ModelsCfg, Settings
from camdigest.pipeline.query import cams_by_id, medias_for_date, segments_for_date


def _settings():
    return Settings(
        cameras=[
            CameraCfg(id="gate", name="大门", device="gate", dir=Path("/data/gate")),
            CameraCfg(id="yard_fixed", name="后院定焦", device="yard", lens="fixed",
                      dir=Path("/data/yard")),
            CameraCfg(id="yard_ptz", name="后院云台", device="yard", lens="ptz",
                      dir=Path("/data/yard")),
        ],
        models=ModelsCfg(
            recognition=ModelCfg(base_url="http://x", model="m", api_key="k"),
            report=ModelCfg(base_url="http://x", model="m", api_key="k"),
        ),
    )


def test_cams_medias_segments_for_date(tmp_path):
    url = f"sqlite:///{tmp_path}/t.db"  # 内存库多连接各开新库，必须用文件库
    settings = _settings()
    assert set(cams_by_id(settings)) == {"gate", "yard_fixed", "yard_ptz"}
    db.init_db(url)
    with db.session_scope(url) as s:
        for cam in settings.cameras:
            s.merge(db.Camera(id=cam.id, name=cam.name, device=cam.device, dir=str(cam.dir)))
        s.add(db.MediaFile(camera_id="gate", path="/a.mp4",
                           start_ts=datetime(2026, 9, 29, 10, 0, tzinfo=UTC),
                           end_ts=datetime(2026, 9, 29, 10, 1, tzinfo=UTC),
                           duration=60.0))
        s.add(db.MediaFile(camera_id="yard_fixed", path="/b.mp4",
                           start_ts=datetime(2026, 9, 30, 10, 0, tzinfo=UTC),
                           end_ts=datetime(2026, 9, 30, 10, 1, tzinfo=UTC),
                           duration=60.0))
    with db.session_scope(url) as s:
        a = medias_for_date("2026-09-29", s)
        b = medias_for_date("2026-09-30", s)
        assert [m.path for m in a] == ["/a.mp4"]            # 日期过滤：各命中一条
        assert [m.path for m in b] == ["/b.mp4"]
        s.add(db.Segment(media_file_id=a[0].id, start_s=0.0, end_s=8.0))  # 默认 pending
        s.add(db.Segment(media_file_id=a[0].id, start_s=8.0, end_s=16.0,
                         recognition_status="ok"))
        s.add(db.Segment(media_file_id=b[0].id, start_s=0.0, end_s=8.0,
                         recognition_status="ok"))
    with db.session_scope(url) as s:
        segs29 = segments_for_date("2026-09-29", s)
        assert {(seg.start_s, seg.recognition_status) for seg, _ in segs29} == {
            (0.0, "pending"), (8.0, "ok")}
        pending = segments_for_date("2026-09-29", s, pending_only=True)
        assert len(pending) == 1 and pending[0][0].recognition_status == "pending"
        assert len(segments_for_date("2026-09-30", s)) == 1
        assert segments_for_date("2026-09-30", s, pending_only=True) == []  # ok 段被过滤
