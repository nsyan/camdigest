# src/camdigest/pipeline/query.py —— 跨阶段共享查询（s2-s6 复用）
from __future__ import annotations

from sqlalchemy.orm import Session

from camdigest.config import CameraCfg, Settings
from camdigest.db import MediaFile, Segment


def cams_by_id(settings: Settings) -> dict[str, CameraCfg]:
    return {c.id: c for c in settings.cameras}


def medias_for_date(date: str, session: Session) -> list[MediaFile]:
    return [m for m in session.query(MediaFile).all()
            if m.start_ts.strftime("%Y-%m-%d") == date]


def segments_for_date(date: str, session: Session, *,
                      pending_only: bool = False) -> list[tuple[Segment, MediaFile]]:
    out = []
    for seg, media in (session.query(Segment, MediaFile)
                       .join(MediaFile, Segment.media_file_id == MediaFile.id).all()):
        if media.start_ts.strftime("%Y-%m-%d") != date:
            continue
        if pending_only and seg.recognition_status != "pending":
            continue
        out.append((seg, media))
    return out
