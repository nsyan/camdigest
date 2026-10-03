# tests/test_s7.py
from datetime import UTC, datetime, timedelta

from camdigest.pipeline.s7_merge import (
    SegLite,
    cross_device_merge,
    detect_anomaly,
    same_device_merge,
)


def _seg(i, device, start, *, lens="fixed", cats=("family",), faces=("妈妈",), camera=None):
    t0 = datetime(2026, 9, 29, tzinfo=UTC)
    return SegLite(seg_id=i, device=device, camera_id=camera or f"{device}-{lens}", lens=lens,
                   start_ts=t0 + timedelta(minutes=start),
                   end_ts=t0 + timedelta(minutes=start + 2),
                   media_path="/x.mp4", start_s=0, end_s=120,
                   face_labels=[{"identity": f, "conf": 0.9, "ts": 1} for f in faces],
                   draft={"category": cats[0], "score": 80, "title": "t", "description": "d",
                          "people": list(faces)}, score=80)


def test_same_device_overlap_merges():
    a = _seg(1, "living", 10)              # 10:00-10:02
    b = _seg(2, "living", 11, lens="ptz")  # 10:01-10:03 重叠 → 必并
    groups = same_device_merge([a, b])
    assert {a.seg_id, b.seg_id} in [set(g) for g in groups]


def test_cross_device_identity_merges():
    a = _seg(1, "living", 10, camera="living-fixed")
    c = _seg(3, "gate", 12, camera="gate")     # 间隔 0，共享"妈妈"
    groups = cross_device_merge([{a.seg_id}], [a, c])
    assert {1, 3} in [set(g) for g in groups]


def test_cross_device_noface_same_category():
    a = _seg(1, "yard", 10, cats=("animal",), faces=(), camera="yard")
    c = _seg(2, "gate", 13, cats=("animal",), faces=(), camera="gate")
    groups = cross_device_merge([{a.seg_id}], [a, c])  # 间隔 1min ≤2min，同 category
    assert {1, 2} in [set(g) for g in groups]


def test_cross_device_far_apart_not_merged():
    a = _seg(1, "yard", 10, faces=(), cats=("animal",), camera="yard")
    c = _seg(2, "gate", 30, faces=(), cats=("animal",), camera="gate")
    groups = cross_device_merge([{a.seg_id}, {c.seg_id}], [a, c])
    assert sorted(map(set, groups)) == [{1}, {2}]   # 间隔 18min >2min → 不合并（T13-b 裁决：全量分组参与）


def test_anomaly_stranger_while_family_absent():
    from camdigest.config import AnomalyCfg
    fam = _seg(1, "living", 0, camera="a")
    stranger = _seg(2, "gate", 200, cats=("stranger",), faces=("未知",), camera="b")
    evs = [{"ids": {1}, "category": "family", "start": fam.start_ts, "end": fam.end_ts, "is_anomaly": False},
           {"ids": {2}, "category": "stranger", "start": stranger.start_ts, "end": stranger.end_ts, "is_anomaly": False}]
    detect_anomaly(evs, AnomalyCfg(), family_names={"妈妈"})
    assert evs[1]["is_anomaly"] is True and evs[0]["is_anomaly"] is False


def test_run_merge_integration(tmp_path):
    """真 DB 胶水：同源归并→事件落库→异常判定回写→关键帧容错（媒体缺失留 NULL）。"""

    from camdigest import db
    from camdigest.config import Settings
    from camdigest.pipeline.s7_merge import run_merge

    url = f"sqlite:///{tmp_path}/t.db"
    db.init_db(url)
    t0 = datetime(2026, 9, 29, 10, 0, tzinfo=UTC)
    with db.session_scope(url) as s:
        s.add(db.Camera(id="living", name="客厅", device="living", lens="single",
                        dir="/a", pattern="*.mp4", timezone="Asia/Shanghai"))
        m = db.MediaFile(camera_id="living", path=str(tmp_path / "missing.mp4"),
                         start_ts=t0, end_ts=t0 + timedelta(seconds=120), duration=120.0)
        s.add(m)
        s.flush()
        s.add(db.Segment(media_file_id=m.id, start_s=0, end_s=60, person_count=1,
                         face_labels=[{"identity": "未知", "conf": 0.9, "ts": 30}],
                         draft={"category": "stranger", "score": 50, "title": "陌生人",
                                "description": "d", "people": ["未知"]},
                         recognition_status="ok"))
        s.add(db.Segment(media_file_id=m.id, start_s=30, end_s=90, person_count=1,
                         face_labels=[{"identity": "未知", "conf": 0.9, "ts": 40}],
                         draft={"category": "stranger", "score": 60, "title": "陌生人2",
                                "description": "d", "people": ["未知"]},
                         recognition_status="ok"))
    settings = Settings(
        cameras=[{"id": "living", "name": "客厅", "device": "living",
                  "lens": "single", "dir": "/a"}],
        models={"recognition": {"base_url": "http://x", "model": "m", "api_key": "k"},
                "report": {"base_url": "http://x", "model": "m", "api_key": "k"}},
        anomaly={"stranger_while_family_absent":
                 {"enabled": True, "family_absent_minutes": 30}},
        storage={"data_dir": str(tmp_path)})
    with db.session_scope(url) as s:
        assert run_merge("2026-09-29", s, settings) == 1  # 重叠同源 → 1 个事件
        ev = s.query(db.Event).one()
        assert sorted(ev.segment_ids) == [1, 2]           # 两个候选片段归并
        assert ev.category == "stranger" and ev.score == 60
        assert ev.is_anomaly is True                      # 无 family 事件 → 异常
        assert ev.keyframe_path is None                   # 媒体文件缺失 → 容错留 NULL
        assert ev.camera_ids == ["living"]
    with db.session_scope(url) as s:
        assert run_merge("2026-09-29", s, settings) == 0  # 幂等：已有事件跳过
        assert s.query(db.Event).count() == 1
