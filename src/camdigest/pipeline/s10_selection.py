# src/camdigest/pipeline/s10_selection.py
"""S10 选段（纯函数）：单池四步 → 档位 Top-N 子集（spec §7，ADR-0002）。

四步：①每小时保底 1（含夜间）→ ②每小时补到 max_per_hour（间隔 <event_gap_minutes
视为同一事件跳过）→ ③按分补齐到 60min 目标 → ④放开去重兜底。
单事件取窗 ≤max_segment_seconds：人脸时间戳密度优先，scene 突变密度次之，均无取居中。
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import datetime

from camdigest.config import HighlightCfg


@dataclass
class SelEvent:
    """选段视角的事件（与 Task 13 SegLite 相近是有意的读写两侧视图，M1 不强行合并）。"""
    event_id: int
    camera_id: str
    device: str
    lens: str
    media_path: str
    start_s: float
    end_s: float
    score: int
    start_ts: datetime
    face_ts: list[float] = field(default_factory=list)
    scene_ts: list[float] = field(default_factory=list)
    face_confs: list[float] = field(default_factory=list)  # 与 face_ts 对齐，供 best_of_event

    @staticmethod
    def best_of_event(candidates: list[SelEvent]) -> SelEvent:
        """同事件多路择一：人脸平均置信度高者胜；平局取 fixed 路（spec §7 云台特写通常占优）。"""
        if len(candidates) == 1:
            return candidates[0]

        def key(e: SelEvent):
            avg = sum(e.face_confs) / len(e.face_confs) if e.face_confs else 0.0
            return (avg, e.lens == "fixed")

        return max(candidates, key=key)

    @classmethod
    def from_rows(cls, events, segments, medias_by_id, cams_by_id) -> list[SelEvent]:
        """Event 行 + 候选片段行 → SelEvent 清单（同事件多路先 best_of_event 择一）。

        scene_ts：S2 的场景锚时刻未持久化（segments 只有 motion_score），M1 置空——
        densest_window 退化为人脸密度优先（计划「声明偏离」条目的一致延伸）。
        """
        by_id = {s.id: s for s in segments}
        out: list[SelEvent] = []
        for ev in events:
            cands: list[SelEvent] = []
            for seg_id in ev.segment_ids or []:
                seg = by_id.get(seg_id)
                if seg is None:
                    continue
                media = medias_by_id[seg.media_file_id]
                cam = cams_by_id[media.camera_id]
                labels = seg.face_labels or []
                cands.append(cls(
                    event_id=ev.id, camera_id=cam.id, device=cam.device, lens=cam.lens,
                    media_path=media.path, start_s=seg.start_s, end_s=seg.end_s,
                    score=(seg.draft or {}).get("score", 0), start_ts=ev.start_ts,
                    face_ts=[f.get("ts", 0.0) for f in labels],
                    face_confs=[f.get("conf", 0.0) for f in labels],
                    scene_ts=[]))
            if cands:
                out.append(cls.best_of_event(cands))
        return out


@dataclass
class ClipPlan:
    event_id: int
    media_path: str
    start_ts: datetime
    start_s: float
    end_s: float
    score: int
    camera_id: str


def densest_window(face_ts, scene_ts, seg_start, seg_end, max_seconds):
    """≤max_seconds 内人脸时间戳最多的窗口；并列时 scene 突变数多者胜；均无取居中。"""
    span = min(max_seconds, seg_end - seg_start)
    mid = (seg_start + seg_end) / 2
    if not face_ts and not scene_ts:
        return (max(seg_start, mid - span / 2), min(seg_end, mid + span / 2))
    faces = sorted(t for t in face_ts if seg_start <= t <= seg_end)
    scenes = sorted(t for t in scene_ts if seg_start <= t <= seg_end)

    def _count(ts, s, e):
        return bisect.bisect_right(ts, e) - bisect.bisect_left(ts, s)

    anchors = sorted(set(faces) | {seg_start})
    best, best_key = (seg_start, seg_start + span), (-1, -1)
    for t in anchors:
        e = min(seg_end, t + span)
        key = (_count(faces, t, e), _count(scenes, t, e))     # 人脸优先，scene 次之
        if key > best_key:
            best_key = key
            best = (t, min(seg_end, max(e, t + 1.0)))  # 钳制不越过段尾
    return best


def _clipped(e: SelEvent, cfg: HighlightCfg) -> float:
    s, t = densest_window(e.face_ts, e.scene_ts, e.start_s, e.end_s, cfg.max_segment_seconds)
    return t - s


def select_pool(events: list[SelEvent], cfg: HighlightCfg) -> list[ClipPlan]:
    target = max(cfg.tiers) * 60
    selected: list[SelEvent] = []

    def gap_ok(e):
        return all(abs((e.start_ts - s.start_ts).total_seconds())
                   >= cfg.event_gap_minutes * 60 for s in selected)

    by_hour: dict[int, list[SelEvent]] = {}
    for e in sorted(events, key=lambda e: e.score, reverse=True):
        by_hour.setdefault(e.start_ts.hour, []).append(e)
    for hour in sorted(by_hour):                      # 1. 每小时保底 1（含夜间）
        selected.append(by_hour[hour][0])
    for hour in sorted(by_hour):                      # 2. 每小时补到 N，间隔去重
        for e in by_hour[hour][1:]:
            if sum(1 for s in selected if s.start_ts.hour == hour) >= cfg.max_per_hour:
                break
            if e not in selected and gap_ok(e):
                selected.append(e)
    for e in sorted(events, key=lambda e: e.score, reverse=True):   # 3. 按分补齐
        if sum(_clipped(x, cfg) for x in selected) >= target:
            break
        if e not in selected and gap_ok(e):
            selected.append(e)
    for e in sorted(events, key=lambda e: e.score, reverse=True):   # 4. 放开去重
        if sum(_clipped(x, cfg) for x in selected) >= target:
            break
        if e not in selected:
            selected.append(e)
    plans = []
    for e in sorted(selected, key=lambda e: e.start_ts):
        s, t = densest_window(e.face_ts, e.scene_ts, e.start_s, e.end_s, cfg.max_segment_seconds)
        plans.append(ClipPlan(event_id=e.event_id, media_path=e.media_path,
                              start_ts=e.start_ts, start_s=s, end_s=t,
                              score=e.score, camera_id=e.camera_id))
    return plans


def tier_subset(pool: list[ClipPlan], minutes: int) -> list[ClipPlan]:
    """段池内按 score 降序取到时长满 minutes，输出按时间排序（ADR-0002 子集承诺）。"""
    target = minutes * 60
    chosen, total = [], 0.0
    for p in sorted(pool, key=lambda p: p.score, reverse=True):
        if total >= target:
            break
        chosen.append(p)
        total += p.end_s - p.start_s
    return sorted(chosen, key=lambda p: p.start_ts)
