# src/camdigest/pipeline/s4_face.py
"""S4 人脸：有人帧 → InsightFace-REST → 身份标签（ADR-0003，spec §4）。

registry 自管：/data/faces/{身份名}/*.jpg 建档时批量 embed 存 .npz。
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
from sqlalchemy.orm import Session

from camdigest.config import Settings
from camdigest.db import Identity
from camdigest.pipeline.query import segments_for_date

log = logging.getLogger(__name__)


@dataclass
class FaceDet:
    bbox: list[float]
    det_score: float
    norm: list[float]


class FaceClient:
    def __init__(self, base_url: str, timeout: float = 30.0,
                 transport: httpx.BaseTransport | None = None):
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def extract(self, image_bytes: bytes) -> list[FaceDet]:
        r = self._client.post("/extract", json={
            "data": base64.b64encode(image_bytes).decode(), "_threshold": 0.6})
        r.raise_for_status()
        data = r.json().get("data", [])
        return [FaceDet(bbox=d["bbox"], det_score=d["det_score"], norm=d["norm"])
                for d in data]


def load_registry(data_dir: Path, client: FaceClient) -> dict[str, list[np.ndarray]]:
    """enroll_faces 落盘的 .npz 优先；无则现场扫目录 embed（首次）。"""
    npz = data_dir / "registry.npz"
    if npz.exists():
        with np.load(npz) as z:
            return {k: list(z[k]) for k in z.files}
    registry: dict[str, list[np.ndarray]] = {}
    if not data_dir.exists():
        return registry  # 尚未建档：空 registry，全部落「未知」
    for id_dir in sorted(data_dir.iterdir()):
        if not id_dir.is_dir():
            continue
        for img in sorted(id_dir.glob("*.jpg")):
            dets = client.extract(img.read_bytes())
            if dets:
                registry.setdefault(id_dir.name, []).append(
                    np.asarray(dets[0].norm, dtype=np.float32))
    return registry


def label_faces(dets: list[FaceDet], registry: dict[str, list[np.ndarray]],
                threshold: float, ts: float) -> list[dict]:
    out = []
    for d in dets:
        v = np.asarray(d.norm, dtype=np.float32)
        best_name, best_cos = "未知", threshold
        for name, vecs in registry.items():
            cos = float(np.max(
                [float(np.dot(v, u) / (np.linalg.norm(v) * np.linalg.norm(u) + 1e-9))
                 for u in vecs])) if vecs else 0.0
            if cos >= best_cos:
                best_name, best_cos = name, cos
        out.append({"identity": best_name, "conf": round(best_cos, 4), "ts": ts})
    return out


def run_face(date: str, session: Session, settings: Settings) -> int:
    """对 person_count>0 的候选片段，3 个采样帧抽帧打标。

    抽帧走 media.grab_frame（ffmpeg 单帧，无 cv2 依赖——保持 [dev] 基线可跑 E2E）。
    """
    import tempfile

    from camdigest.media import grab_frame
    client = FaceClient(settings.faces.rest_url)
    registry = load_registry(settings.faces.registry_dir, client)
    log.info("S4 registry：%d 身份 %d 向量",
             len(registry), sum(len(v) for v in registry.values()))
    updated = 0
    for seg, media in segments_for_date(date, session):
        if seg.person_count <= 0 or seg.face_labels:
            continue
        times = [seg.start_s + (seg.end_s - seg.start_s) * k / 4 for k in range(1, 4)]
        labels = []
        with tempfile.TemporaryDirectory() as td:
            for k, t in enumerate(times):
                jpg = Path(td) / f"frame{k}.jpg"
                try:
                    grab_frame(Path(media.path), t, jpg)
                    dets = client.extract(jpg.read_bytes())
                except Exception as e:  # noqa: BLE001 —— 单帧失败跳过（媒体可能损坏）
                    log.warning("S4 frame extract failed seg=%s t=%.2f: %s",
                                    seg.id, t, e)
                    continue
                labels += label_faces(dets, registry, settings.faces.threshold,
                                      ts=round(t, 3))
        seg.face_labels = labels
        updated += 1
    return updated


def enroll_faces(data_dir: Path, rest_url: str, session=None,
                 client: FaceClient | None = None) -> int:
    """CLI enroll-faces：扫 /faces/{身份名}/*.jpg → embed → registry.npz + identities 行。

    npz 键即身份名（多张照片 stack 成 (n,512) 矩阵），load_registry 读回时直接还原。
    """
    client = client or FaceClient(rest_url)
    registry: dict[str, list[np.ndarray]] = {}
    for id_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        vecs = []
        for img in sorted(id_dir.glob("*.jpg")):
            dets = client.extract(img.read_bytes())
            if dets:
                vecs.append(np.asarray(dets[0].norm, dtype=np.float32))
        if vecs:
            registry[id_dir.name] = vecs
        if session is not None:  # 同步登记 identities 表（spec §5）；重建档更新而非重复插入
            row = session.query(Identity).filter(
                Identity.name == id_dir.name).one_or_none()
            if row:
                row.dir = str(id_dir)
            else:
                session.add(Identity(name=id_dir.name, dir=str(id_dir)))
    if not registry:
        return 0
    np.savez(data_dir / "registry.npz",
             **{name: np.stack(vecs) for name, vecs in registry.items()})
    return sum(len(v) for v in registry.values())
