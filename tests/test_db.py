# tests/test_db.py
from datetime import UTC, datetime

from camdigest import db


def _ts() -> datetime:
    return datetime(2026, 9, 29, 10, 0, tzinfo=UTC)


def test_tables_and_roundtrip(tmp_path):
    url = f"sqlite:///{tmp_path}/t.db"
    db.init_db(url)
    with db.session_scope(url) as s:
        cam = db.Camera(id="gate", name="大门", device="gate", lens="single",
                        dir="/media/nvr/gate", pattern="*.mp4", timezone="Asia/Shanghai")
        s.add(cam)
        mf = db.MediaFile(camera_id="gate", path="/x/a.mp4",
                          start_ts=_ts(), end_ts=_ts(), duration=60.0)
        s.add(mf)
        s.flush()
        s.add(db.Segment(media_file_id=mf.id, start_s=1.0, end_s=9.0, motion_score=0.3,
                         person_count=1, face_labels=[{"identity": "妈妈", "conf": 0.8, "ts": 2.0}],
                         audio_active=True, recognition_status="pending",
                         draft={"category": "family", "score": 80}))
    with db.session_scope(url) as s:
        seg = s.query(db.Segment).one()
        assert seg.draft["category"] == "family"
        assert seg.face_labels[0]["identity"] == "妈妈"
        job = db.Job(date="2026-09-29", stage="s1_index", status="done")
        s.add(job)
    with db.session_scope(url) as s:  # 唯一约束生效
        import pytest
        from sqlalchemy.exc import IntegrityError
        s.add(db.Job(date="2026-09-29", stage="s1_index", status="done"))
        with pytest.raises(IntegrityError):
            s.commit()
