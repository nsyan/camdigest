# src/camdigest/media.py
"""ffprobe/ffmpeg 子进程公共封装。所有视频操作集中在此。"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_TS_PATTERNS = [  # 优先文件名（米家转存按时间命名），失败退回 mtime
    re.compile(r"(20\d{12})"),            # 20260929100000
    re.compile(r"(20\d{6})[_\- ]?(\d{6})"),  # 20260929_100000
]


@dataclass
class MediaMeta:
    path: Path
    start_ts: datetime
    duration: float
    size_bytes: int


def parse_start_ts(path: Path) -> datetime | None:
    for pat in _TS_PATTERNS:
        m = pat.search(path.name)
        if m:
            digits = "".join(m.groups())
            # 文件名时间是机位本地墙钟（无时区），按约定以 naive datetime 入库
            return datetime.strptime(digits[:14], "%Y%m%d%H%M%S")  # noqa: DTZ007
    return None


def probe(path: Path) -> MediaMeta:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", str(path)],
        check=True, capture_output=True, text=True).stdout
    fmt = json.loads(out)["format"]
    duration = float(fmt["duration"])
    ts = parse_start_ts(path) or datetime.fromtimestamp(path.stat().st_mtime)  # noqa: DTZ006
    return MediaMeta(path=path, start_ts=ts, duration=duration, size_bytes=int(fmt["size"]))
