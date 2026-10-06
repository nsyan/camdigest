# tests/test_s10_clip.py —— 切片/拼接真 ffmpeg；导出四档+highlights 行
from datetime import UTC, datetime

from camdigest.config import Settings
from camdigest.media import has_audio_stream, probe
from camdigest.pipeline.s10_clip import concat_ts, cut_ts, export_highlights
from camdigest.pipeline.s10_selection import ClipPlan


def test_cut_and_concat_audio_video(video_file, tmp_path):
    """两段各切 2s → concat → 时长 ≈4s、含音轨、可被 ffprobe 解析。"""
    a = cut_ts(video_file, 0.0, 2.0, tmp_path / "a.ts")
    b = cut_ts(video_file, 2.0, 4.0, tmp_path / "b.ts")
    assert a.exists() and b.exists()
    out = concat_ts([a, b], tmp_path / "out.mp4")
    meta = probe(out)
    assert 3.5 < meta.duration < 4.7          # -c copy 有关键帧对齐误差
    assert has_audio_stream(out)              # aac 重编码保原声


def test_concat_silent_video_has_no_audio(silent_video_file, tmp_path):
    """无声视频切片拼接：不因 -c:a aac 失败，产物无音轨。"""
    a = cut_ts(silent_video_file, 0.0, 2.0, tmp_path / "a.ts")
    b = cut_ts(silent_video_file, 2.0, 4.0, tmp_path / "b.ts")
    out = concat_ts([a, b], tmp_path / "silent.mp4")
    assert 3.5 < probe(out).duration < 4.7
    assert not has_audio_stream(out)


def _plan(eid, cam, start_min, score=80, media_path="/x/gate.mp4"):
    t0 = datetime(2026, 9, 29, start_min, 0, tzinfo=UTC)
    return ClipPlan(event_id=eid, media_path=media_path, start_ts=t0,
                    start_s=0.0, end_s=2.0, score=score, camera_id=cam)


def test_export_highlights_tiers_and_rows(tmp_path, video_file):
    """两段 → tiers=[5] 单档导出：文件存在、时长>0、highlights 行 camera_id NULL。"""
    import shutil

    from camdigest import db

    shutil.copy(video_file, tmp_path / "gate.mp4")
    shutil.copy(video_file, tmp_path / "yard.mp4")
    pool = [_plan(1, "gate", 10, media_path=str(tmp_path / "gate.mp4")),
            _plan(2, "yard", 15, score=70, media_path=str(tmp_path / "yard.mp4"))]
    settings = Settings(
        cameras=[{"id": "gate", "name": "大门", "device": "gate",
                  "lens": "single", "dir": "/x"},
                 {"id": "yard", "name": "院子", "device": "yard",
                  "lens": "single", "dir": "/x"}],
        models={"recognition": {"base_url": "http://x", "model": "m", "api_key": "k"},
                "report": {"base_url": "http://x", "model": "m", "api_key": "k"}},
        highlight={"tiers": [5]},
        storage={"data_dir": str(tmp_path)})
    url = f"sqlite:///{tmp_path}/t.db"
    db.init_db(url)
    with db.session_scope(url) as s:
        outs = export_highlights("2026-09-29", pool, settings, session=s)
    assert len(outs) == 1
    p = tmp_path / "highlights" / "2026-09-29" / "精华_5min.mp4"
    assert p == outs[0] and p.exists() and probe(p).duration > 0
    with db.session_scope(url) as s:
        row = s.query(db.Highlight).one()
        assert row.date == "2026-09-29" and row.tier_minutes == 5
        assert row.camera_id is None and row.event_ids == [1, 2]
        assert row.duration > 0 and row.bytes > 0


def test_export_highlights_per_camera(tmp_path, video_file):
    """per_camera=true：合并版之外每机位各出一份（不占档位时长）。"""
    import shutil

    from camdigest import db

    shutil.copy(video_file, tmp_path / "gate.mp4")
    settings = Settings(
        cameras=[{"id": "gate", "name": "大门", "device": "gate",
                  "lens": "single", "dir": "/x"}],
        models={"recognition": {"base_url": "http://x", "model": "m", "api_key": "k"},
                "report": {"base_url": "http://x", "model": "m", "api_key": "k"}},
        highlight={"tiers": [5], "per_camera": True},
        storage={"data_dir": str(tmp_path)})
    url = f"sqlite:///{tmp_path}/t.db"
    db.init_db(url)
    pool = [_plan(1, "gate", 10, media_path=str(tmp_path / "gate.mp4"))]
    with db.session_scope(url) as s:
        outs = export_highlights("2026-09-29", pool, settings, session=s)
    assert len(outs) == 2   # 合并版 + gate 单机位版
    assert (tmp_path / "highlights" / "2026-09-29" / "精华_5min_gate.mp4").exists()
    with db.session_scope(url) as s:
        cams = {r.camera_id for r in s.query(db.Highlight).all()}
        assert cams == {None, "gate"}
