# src/camdigest/media.py
"""ffprobe/ffmpeg 子进程公共封装。所有视频操作集中在此。"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

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


# —— Task 6 追加：抽帧统一入口，s3/s4 共用（评审：消除 cv2 抽帧重复）——
def sample_frames(path: Path, times: list[float]) -> list[tuple[float, np.ndarray]]:
    """按时刻抽帧（cv extra，惰性 import）。返回实际读到的 (t, frame) 列表。"""
    import cv2
    cap = cv2.VideoCapture(str(path))
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if ok:
            out.append((t, frame))
    cap.release()
    return out


def frame_jpg(frame) -> bytes:
    """单帧编码 JPEG（cv extra）；编码失败返回空 bytes。"""
    import cv2
    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else b""


# —— Task 7 追加：ffmpeg 单帧抽取，S6 关键帧/S4 共用 ——
def grab_frame(path: Path, t: float, out: Path) -> Path:
    """ffmpeg 在 t 秒处抽单帧写 out（JPEG）；成功返回 out 路径。"""
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(t), "-i", str(path), "-frames:v", "1", str(out)],
        check=True, capture_output=True)
    return out


def extract_audio(path: Path, out_wav: Path) -> Path:
    subprocess.run(["ffmpeg", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", str(out_wav)], check=True, capture_output=True)
    return out_wav


def cut_clip(path: Path, start_s: float, end_s: float, out: Path) -> Path:
    subprocess.run(["ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}",
                    "-i", str(path), "-c", "copy", str(out)], check=True, capture_output=True)
    return out
