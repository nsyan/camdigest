# tests/test_media.py
from datetime import datetime
from pathlib import Path

from camdigest.media import parse_start_ts, probe


def test_probe(video_file):
    meta = probe(Path(video_file))
    assert 5.5 < meta.duration < 6.5
    assert meta.size_bytes > 0


def test_parse_start_ts_mijia_layout():
    # 米家 SD 卡常见命名：14 位时间戳（CAL-1 真机校准点，解析策略可配置化）
    ts = parse_start_ts(Path("/nvr/gate/20260929100000.mp4"))
    # 文件名时间约定即 naive 本地时间
    assert ts == datetime(2026, 9, 29, 10, 0, 0)  # noqa: DTZ001
