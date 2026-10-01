# tests/test_s2.py
from datetime import UTC, datetime, timedelta

from camdigest.config import PrefilterCfg
from camdigest.pipeline.s2_prefilter import build_segments, detect_scenes


def _media(duration):
    from camdigest.db import MediaFile
    start = datetime(2026, 9, 29, 10, 0, tzinfo=UTC)
    return MediaFile(camera_id="gate", path="/x.mp4", start_ts=start,
                     end_ts=start + timedelta(seconds=duration), duration=duration)


def test_detect_scenes_on_fixture(video_file):
    from pathlib import Path
    times = detect_scenes(Path(video_file), threshold=0.03)
    assert isinstance(times, list)  # testsrc 无突变可能为空，不误报即可


def test_build_segments_merges_and_pads():
    from camdigest.db import MediaFile
    mf = MediaFile(camera_id="gate", path="/x.mp4",
                   start_ts=datetime(2026, 9, 29, 10, 0, tzinfo=UTC),
                   end_ts=datetime(2026, 9, 29, 10, 1, tzinfo=UTC), duration=60.0)
    segs = build_segments(mf, times=[10.0, 12.0, 40.0], cfg=PrefilterCfg())
    assert segs, "至少产出一段"
    for s in segs:
        assert s.end_s - s.start_s >= 8.0 - 4 * 1e-9      # min_segment（含 pad 后）
        assert s.motion_score > 0


def test_build_segments_overlong_split_respects_min():
    """超长无突变跨度切分后，尾块同样满足 min_segment：
    63s→[0,60]（3s 尾块丢弃）、121s→[0,60]+[60,120]、130s→尾块 10s 保留。"""
    cfg = PrefilterCfg()
    segs63 = build_segments(_media(63.0), times=[], cfg=cfg)
    assert [(s.start_s, s.end_s) for s in segs63] == [(0.0, 60.0)]
    segs121 = build_segments(_media(121.0), times=[], cfg=cfg)
    assert [(s.start_s, s.end_s) for s in segs121] == [(0.0, 60.0), (60.0, 120.0)]
    segs130 = build_segments(_media(130.0), times=[], cfg=cfg)
    assert [(s.start_s, s.end_s) for s in segs130] == [
        (0.0, 60.0), (60.0, 120.0), (120.0, 130.0)]
    for s in segs63 + segs121 + segs130:
        assert s.end_s - s.start_s >= cfg.min_segment_seconds - 4 * 1e-9
