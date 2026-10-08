# src/camdigest/pipeline/s1_index.py
"""S1 索引：扫描机位目录 → ffprobe 元数据入库（spec §4）。幂等：按 path 去重。"""
from __future__ import annotations

import logging
from datetime import timedelta

from sqlalchemy.orm import Session

from camdigest.config import CameraCfg
from camdigest.db import Camera, MediaFile
from camdigest.media import probe

log = logging.getLogger(__name__)


def index_camera(camera: CameraCfg, session: Session) -> int:
    session.merge(Camera(id=camera.id, name=camera.name, device=camera.device,
                         lens=camera.lens, dir=str(camera.dir), pattern=camera.pattern,
                         timezone=camera.timezone))
    known = {m.path for m in session.query(MediaFile.path).filter(
        MediaFile.camera_id == camera.id)}
    added = 0
    for f in sorted(camera.dir.glob(camera.pattern)):
        p = str(f.resolve())
        if p in known:
            continue
        meta = probe(f)
        session.add(MediaFile(camera_id=camera.id, path=p, start_ts=meta.start_ts,
                              end_ts=meta.start_ts + timedelta(seconds=meta.duration),
                              duration=meta.duration))
        added += 1
    log.info("%s 新增 %d 个文件", camera.id, added)
    return added
