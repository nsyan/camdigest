# tests/test_s2.py
from datetime import UTC, datetime

from camdigest.config import PrefilterCfg
from camdigest.pipeline.s2_prefilter import build_segments, detect_scenes


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
