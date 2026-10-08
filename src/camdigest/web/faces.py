# src/camdigest/web/faces.py
"""未知脸聚类扫描与归档（设计 §5）。"""
from __future__ import annotations

import logging

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
