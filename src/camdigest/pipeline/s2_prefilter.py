# src/camdigest/pipeline/s2_prefilter.py
"""S2 预筛：FFmpeg scene 检测 → 候选片段（spec §4）。

云台路防误报：阈值由 CameraCfg.effective_scene_threshold() 提供（默认 0.08），
且云台路候选片段还需 S3 人形计数>0 才最终入库（Task 6 落实）。
"""
from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from sqlalchemy.orm import Session

from camdigest.config import PrefilterCfg, Settings
from camdigest.db import MediaFile, Segment
from camdigest.pipeline.query import cams_by_id, medias_for_date

log = logging.getLogger(__name__)

_PTS_RE = re.compile(r"pts_time:([\d.]+)")


def detect_scenes(path: Path, threshold: float) -> list[float]:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-an", "-vf",
         f"select='gt(scene\\,{threshold})',showinfo", "-f", "null", "-"],
        check=True, capture_output=True, text=True).stderr
    return sorted(float(t) for t in _PTS_RE.findall(out))


@dataclass
class SegmentDraft:
    start_s: float
    end_s: float
    motion_score: float


def build_segments(media: MediaFile, times: list[float], cfg: PrefilterCfg) -> list[SegmentDraft]:
    """以 scene 突变点为锚，切 [t-pad, next_gap] 区间；无突变则整文件一段。"""
    anchors = [t for t in times if 0 <= t <= media.duration]
    bounds = [0.0]
    for a, nxt in pairwise(anchors):  # 相邻锚中点切分
        bounds.append((a + nxt) / 2)
    bounds.append(media.duration)  # 尾界=文件末：尾段并入最后锚区间（无突变则整文件一段）
    out: list[SegmentDraft] = []
    for i in range(len(bounds) - 1):
        start = max(0.0, bounds[i] - cfg.pad_seconds)
        end = min(media.duration, bounds[i + 1] + cfg.pad_seconds)
        if end - start < cfg.min_segment_seconds:
            continue
        while end - start > cfg.max_segment_seconds:  # 超长均分为多段
            mid = start + cfg.max_segment_seconds
            n = len([a for a in anchors if start <= a < mid])
            out.append(SegmentDraft(start, mid, motion_score=n / (mid - start)))
            start = mid
        if end - start >= cfg.min_segment_seconds:  # 尾块同样受 min_segment 约束
            n = len([a for a in anchors if start <= a < end])
            out.append(SegmentDraft(start, end, motion_score=n / (end - start)))
    return out


def run_prefilter(date: str, session: Session, settings: Settings) -> int:
    added = 0
    cams = cams_by_id(settings)
    for media in medias_for_date(date, session):
        if session.query(Segment).filter(Segment.media_file_id == media.id).count():
            continue  # 幂等
        cam = cams.get(media.camera_id)
        thr = cam.effective_scene_threshold(settings.prefilter.scene_threshold)
        for d in build_segments(media, detect_scenes(Path(media.path), thr), settings.prefilter):
            session.add(Segment(media_file_id=media.id, start_s=d.start_s, end_s=d.end_s,
                                motion_score=d.motion_score))
            added += 1
    log.info("%s 预筛 %d 段", date, added)
    return added
