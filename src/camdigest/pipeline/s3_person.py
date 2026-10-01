# src/camdigest/pipeline/s3_person.py
"""S3 人形：区间内 1fps 采样帧 → YOLO11n 人形计数（spec §4）。

模型惰性加载（cv extra），单测注入 sampler/infer 假实现。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from camdigest.config import Settings
from camdigest.pipeline.query import segments_for_date


def sample_times(start_s: float, end_s: float, fps: float = 1.0) -> list[float]:
    """[start_s, end_s) 内按 fps 取采样时刻：int((end-start)*fps) 个点。"""
    n = int((end_s - start_s) * fps)
    return [round(start_s + i / fps, 3) for i in range(n)]


@dataclass
class PersonResult:
    max_count: int
    frames: list[tuple[float, int]] = field(default_factory=list)


def count_people(video_path: Path, start_s: float, end_s: float,
                 sampler, infer) -> PersonResult:
    times = sample_times(start_s, end_s)
    frames = sampler.sample(video_path, times)
    counts = infer(frames)                       # [(count_at_t)] 与 frames 对齐
    pairs = [(t, c) for (t, _), c in zip(frames, counts)]
    return PersonResult(max_count=max((c for _, c in pairs), default=0), frames=pairs)


class YoloPersonSampler:
    """真实现：ultralytics YOLO11n，person=class 0，conf>0.35。heavy。"""

    def __init__(self, weights: str = "yolo11n.pt"):
        from ultralytics import YOLO  # cv extra
        self.model = YOLO(weights)

    def sample(self, path: Path, times: list[float]):
        from camdigest.media import sample_frames  # 抽帧统一入口，s3/s4 共用
        return sample_frames(path, times)

    def infer(self, frames) -> list[int]:
        out = []
        for _, frame in frames:
            res = self.model.predict(frame, verbose=False, classes=[0], conf=0.35)
            out.append(len(res[0].boxes) if res else 0)
        return out


def run_person(date: str, session: Session, settings: Settings) -> int:
    sampler = YoloPersonSampler()
    updated = 0
    for seg, media in segments_for_date(date, session):
        if seg.person_count >= 0:      # -1=未处理（Task 3 默认值），重跑不重复推断
            continue
        r = count_people(Path(media.path), seg.start_s, seg.end_s,
                         sampler=sampler, infer=sampler.infer)
        seg.person_count = r.max_count
        updated += 1
    # 云台路门控：ptz 机位 0 人候选片段删除（spec §4 云台防误报）
    ptz_cams = {c.id for c in settings.cameras if c.lens == "ptz"}
    for seg, media in list(segments_for_date(date, session)):
        if media.camera_id in ptz_cams and seg.person_count == 0:
            session.delete(seg)
    return updated
