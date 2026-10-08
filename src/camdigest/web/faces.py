# src/camdigest/web/faces.py
"""未知脸聚类扫描与归档（设计 §5）。"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from sqlalchemy.orm import Session

from camdigest.config import Settings
from camdigest.pipeline.s4_face import FaceClient

log = logging.getLogger(__name__)

CLUSTER_THRESHOLD = 0.45   # 高于识别阈值 0.4，避免错并（设计 §5）


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(x * x for x in b) ** 0.5
    return num / (da * db + 1e-9)


def assign_cluster(vec: list[float], clusters: dict[int, list[list[float]]],
                   threshold: float = CLUSTER_THRESHOLD) -> int:
    """贪心近邻：与簇内任一向量余弦 ≥ threshold 归簇，否则新簇。"""
    best_id, best_cos = None, threshold
    for cid, vecs in clusters.items():
        c = max(cosine(vec, v) for v in vecs)
        if c >= best_cos:
            best_id, best_cos = cid, c
    if best_id is None:
        best_id = max(clusters, default=0) + 1
        clusters[best_id] = []
    clusters[best_id].append(vec)
    return best_id


def scan_unknown_faces(date: str, session: Session, settings: Settings,
                       client=None) -> int:
    """收集该日期「未知」脸 embedding 并贪心归簇，返回新增脸数（设计 §5）。"""
    from camdigest.db import UnknownFace
    from camdigest.media import grab_frame
    from camdigest.pipeline.query import segments_for_date
    from camdigest.pipeline.s4_face import label_faces, load_registry

    client = client or FaceClient(settings.faces.rest_url)
    registry = load_registry(settings.faces.registry_dir, client)
    clusters: dict[int, list[list[float]]] = {}
    for row in session.query(UnknownFace).all():
        clusters.setdefault(row.cluster_id, []).append(list(row.embedding))
    cluster_dir = Path(settings.storage.data_dir) / "keyframes" / ".clusters"
    added = 0
    for seg, media in segments_for_date(date, session):
        if seg.person_count <= 0:
            continue
        times = [seg.start_s + (seg.end_s - seg.start_s) * k / 4 for k in range(1, 4)]
        with tempfile.TemporaryDirectory() as td:
            for k, t in enumerate(times):
                jpg = Path(td) / f"f{k}.jpg"
                try:
                    grab_frame(Path(media.path), t, jpg)
                    dets = client.extract(jpg.read_bytes())
                except Exception as e:  # noqa: BLE001 —— 单帧失败跳过（媒体可能损坏）
                    log.warning("扫描抽帧失败 seg=%s t=%.2f：%s", seg.id, t, e)
                    continue
                labels = label_faces(dets, registry, settings.faces.threshold,
                                     ts=round(t, 3))
                for det, label in zip(dets, labels):
                    if label["identity"] != "未知":
                        continue
                    vec = [float(x) for x in det.norm]
                    cid = assign_cluster(vec, clusters)
                    session.add(UnknownFace(
                        cluster_id=cid, embedding=vec, segment_id=seg.id,
                        media_path=media.path, ts_in_seg=round(t, 3),
                        det_score=det.det_score))
                    added += 1
                    rep = cluster_dir / f"{cid}.jpg"
                    if not rep.exists():       # 代表照先到先得；簇内高分替换仅限本次扫描
                        rep.parent.mkdir(parents=True, exist_ok=True)
                        import shutil
                        shutil.copy(jpg, rep)
    log.info("扫描 %s：新增 %d 张未知脸（%d 簇）", date, added, len(clusters))
    return added


def _scan_day(date: str, settings: Settings) -> None:
    """日批 post_run 钩子：session 包装 + 异常只记日志。"""
    from camdigest import db
    try:
        with db.session_scope(f"sqlite:///{Path(settings.storage.data_dir) / 'camdigest.db'}") as s:
            scan_unknown_faces(date, s, settings)
    except Exception:
        log.exception("日批后扫描未知脸失败（%s）", date)
