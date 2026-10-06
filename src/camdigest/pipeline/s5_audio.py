# src/camdigest/pipeline/s5_audio.py
"""S5 音频：silencedetect 标记有声段 → faster-whisper 转写（spec §4）。

同设备双机位仅转固定路（CAL-2），云台路复用（dual_lens_source）。
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from sqlalchemy.orm import Session

from camdigest.config import Settings
from camdigest.db import MediaFile, Segment
from camdigest.media import has_audio_stream, probe
from camdigest.pipeline.query import cams_by_id, medias_for_date, segments_for_date

_SIL_RE = re.compile(r"silence_start: ([\d.]+)")
_SIL_END_RE = re.compile(r"silence_end: ([\d.]+)")


def audio_active_regions(path: Path, noise: float = -35.0) -> list[tuple[float, float]]:
    if not has_audio_stream(path):  # 无音轨 → 无任何有声段
        return []
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af",
         f"silencedetect=noise={noise}dB:d=0.5", "-f", "null", "-"],
        check=True, capture_output=True, text=True).stderr
    total = probe(path).duration
    starts = [float(t) for t in _SIL_RE.findall(out)]
    ends = [float(t) for t in _SIL_END_RE.findall(out)]
    regions, cur = [], 0.0
    for s, e in zip(starts, ends):
        if s - cur > 0.5:
            regions.append((cur, s))
        cur = max(cur, e)
    if total - cur > 0.5:
        regions.append((cur, total))
    return regions


class Transcriber:
    def __init__(self, model_name: str = "small-int8"):
        from faster_whisper import WhisperModel  # cv extra
        self.model = WhisperModel(model_name, compute_type="int8")

    def transcribe(self, path: Path, start_s: float, end_s: float) -> str:
        segs, _ = self.model.transcribe(str(path), language="zh",
                                        clip_timestamps=f"{start_s:.2f},{end_s:.2f}",
                                        vad_filter=True)
        return " ".join(s.text.strip() for s in segs).strip()


def pick_transcribe_sources(medias: list[MediaFile], cams_by_id: dict, source: str) -> set[int]:
    if source == "all":
        return {m.id for m in medias}
    keep: set[int] = set()
    by_device: dict[str, list[MediaFile]] = {}
    for m in medias:
        by_device.setdefault(cams_by_id[m.camera_id].device, []).append(m)
    for group in by_device.values():
        wanted = [m for m in group if cams_by_id[m.camera_id].lens == source]
        keep.update(m.id for m in (wanted or group))   # 无目标 lens 则全转
    return keep


def run_audio(date: str, session: Session, settings: Settings) -> int:
    medias = medias_for_date(date, session)
    cams = cams_by_id(settings)
    wcfg = settings.prefilter.whisper
    sources = pick_transcribe_sources(medias, cams, wcfg.dual_lens_source)
    src_by_id = {m.id: m for m in medias if m.id in sources}
    tr = Transcriber(wcfg.model) if wcfg.enabled else None
    updated = 0
    for seg, media in segments_for_date(date, session):
        if media.id in sources:
            regions = audio_active_regions(Path(media.path))
            seg.audio_active = any(s < seg.end_s and e > seg.start_s for s, e in regions)
            # selected_only 语义（S6 后执行时 draft 已落，编排器负责调序，见 Task 18）
            selected = (seg.draft or {}).get("score", 0) >= 60
            if tr and seg.audio_active and (not wcfg.selected_only or selected):
                seg.transcript = tr.transcribe(Path(media.path), seg.start_s, seg.end_s) or seg.transcript
        else:
            # 双机位另一路：按时间重叠复用同设备源路结果
            donor = next((m for m in src_by_id.values()
                          if cams[m.camera_id].device == cams[media.camera_id].device
                          and m.start_ts <= media.end_ts and m.end_ts >= media.start_ts), None)
            if donor:
                for dseg in session.query(Segment).filter(Segment.media_file_id == donor.id):
                    if dseg.start_s < seg.end_s and dseg.end_s > seg.start_s:
                        seg.audio_active = seg.audio_active or dseg.audio_active
                        seg.transcript = seg.transcript or dseg.transcript
        updated += 1
    return updated
