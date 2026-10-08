# src/camdigest/pipeline/s7_merge.py
"""S7 归并：同源必并/跨设备规则归并事件、异常判定、关键帧提取（spec §4）。

归并规则成文（spec §4）：①同设备双机位时间重叠必并（同源）；②跨设备时间相邻 ≤120s
且共享已知身份、或双方均无已知身份且 category 相同；③无人脸事件只走规则②。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from camdigest.config import AnomalyCfg, Settings
from camdigest.db import Event, Identity
from camdigest.pipeline.query import cams_by_id, segments_for_date

log = logging.getLogger(__name__)


@dataclass
class SegLite:
    """事件归并用的轻量段视图（event 之外的读写两侧视图之一，与 S10 SelEvent 相近是有意的）。"""
    seg_id: int
    device: str
    camera_id: str
    lens: str
    start_ts: datetime
    end_ts: datetime
    media_path: str
    start_s: float
    end_s: float
    face_labels: list
    draft: dict
    score: int


def _union_find(ids):
    parent = {i: i for i in ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b): parent[find(a)] = find(b)
    return find, union


def same_device_merge(segs: list[SegLite]) -> list[set[int]]:
    find, union = _union_find([s.seg_id for s in segs])
    by_dev: dict[str, list[SegLite]] = {}
    for s in segs:
        by_dev.setdefault(s.device, []).append(s)
    for group in by_dev.values():
        group.sort(key=lambda s: s.start_ts)
        for i, s in enumerate(group):          # 时间重叠 → 同源必并
            for t in group[i + 1:]:
                if t.start_ts >= s.end_ts:
                    break
                union(s.seg_id, t.seg_id)
    out: dict[int, set[int]] = {}
    for s in segs:
        out.setdefault(find(s.seg_id), set()).add(s.seg_id)
    return list(out.values())


def cross_device_merge(groups: list[set[int]], segs: list[SegLite]) -> list[set[int]]:
    by_id = {s.seg_id: s for s in segs}
    ids = [i for g in groups for i in g]
    ids += [s.seg_id for s in segs if s.seg_id not in set(ids)]  # 未分组段以单例参与
    find, union = _union_find(ids)
    for g in groups:                     # 保留同设备归并结果（规则①不可被规则②重评丢掉）
        g_list = [i for i in g if i in by_id]
        for other in g_list[1:]:
            union(g_list[0], other)

    def known_ids(g):   # 组内已知身份（非"未知"）
        return {f["identity"] for i in g for f in by_id[i].face_labels
                if f["identity"] != "未知"}

    def span(g):
        return (min(by_id[i].start_ts for i in g), max(by_id[i].end_ts for i in g))

    def top_cat(g):     # 组内最高分段的 category
        best = max(g, key=lambda i: by_id[i].score)
        return by_id[best].draft["category"]

    changed = True
    while changed:      # 合并传递闭包，扫到不动点
        changed = False
        reps = [find(i) for i in set(ids)]
        gs = [ {i for i in ids if find(i) == r} for r in reps ]
        for a in range(len(gs)):
            for b in range(a + 1, len(gs)):
                if find(next(iter(gs[a]))) == find(next(iter(gs[b]))):
                    continue
                (a0, a1), (b0, b1) = span(gs[a]), span(gs[b])
                gap = max((b0 - a1).total_seconds(), (a0 - b1).total_seconds())
                if gap > 120:                      # 跨设备容差 ≤2 分钟
                    continue
                ka, kb = known_ids(gs[a]), known_ids(gs[b])
                if (ka & kb) or (not ka and not kb and top_cat(gs[a]) == top_cat(gs[b])):
                    union(next(iter(gs[a])), next(iter(gs[b])))
                    changed = True
    out: dict[int, set[int]] = {}
    for i in ids:
        out.setdefault(find(i), set()).add(i)
    return list(out.values())


def detect_anomaly(evs, cfg: AnomalyCfg, family_names: set[str]) -> None:
    """规则 stranger_while_family_absent：stranger 事件前后 family_absent_minutes
    内无任何 family 事件（含命中家人身份）→ is_anomaly=True。"""
    rule = cfg.stranger_while_family_absent
    if not rule.enabled:
        return
    fam_times = [(e["start"], e["end"]) for e in evs if e["category"] == "family"]
    for e in evs:
        if e["category"] != "stranger":
            continue
        window = timedelta(minutes=rule.family_absent_minutes)
        near = any(fs - window <= e["end"] and fe + window >= e["start"] for fs, fe in fam_times)
        e["is_anomaly"] = not near


def pick_keyframe_ts(seg: SegLite) -> float:
    """人脸时间戳密度峰（众数）；无脸取中点。"""
    ts = [f["ts"] for f in seg.face_labels
          if seg.start_s <= f["ts"] <= seg.end_s]
    if not ts:
        return (seg.start_s + seg.end_s) / 2
    return max(set(ts), key=ts.count)


def run_merge(date: str, session: Session, settings: Settings) -> int:
    if session.query(Event).filter(Event.date == date).count():
        return 0  # 幂等：该日期已归并
    cams = cams_by_id(settings)
    lites: list[SegLite] = []
    for seg, media in segments_for_date(date, session):
        if not seg.draft or "category" not in seg.draft:
            continue  # 仅识别完成的候选片段参与归并
        cam = cams[media.camera_id]
        lites.append(SegLite(
            seg_id=seg.id, device=cam.device, camera_id=cam.id, lens=cam.lens,
            start_ts=media.start_ts + timedelta(seconds=seg.start_s),
            end_ts=media.start_ts + timedelta(seconds=seg.end_s),
            media_path=media.path, start_s=seg.start_s, end_s=seg.end_s,
            face_labels=seg.face_labels or [], draft=seg.draft,
            score=seg.draft.get("score", 0)))
    if not lites:
        return 0
    groups = cross_device_merge(same_device_merge(lites), lites)
    by_id = {s.seg_id: s for s in lites}
    family_names = {i.name for i in session.query(Identity).all()}
    created = 0
    ev_top: list[tuple[Event, SegLite]] = []
    for group in sorted(groups, key=lambda g: min(by_id[i].start_ts for i in g)):
        members = [by_id[i] for i in group]
        top = max(members, key=lambda s: s.score)
        ev = Event(
            date=date,
            start_ts=min(s.start_ts for s in members),
            end_ts=max(s.end_ts for s in members),
            category=top.draft["category"], score=max(s.score for s in members),
            title=top.draft.get("title", ""), description=top.draft.get("description", ""),
            camera_ids=sorted({s.camera_id for s in members}),
            segment_ids=sorted(group), is_anomaly=False)
        session.add(ev)
        session.flush()  # 取 id 供关键帧落盘
        ev_top.append((ev, top))
        created += 1
    # 异常判定需全日事件相互参照：整日一起判（stranger 邻近 family 才豁免）
    evs = session.query(Event).filter(Event.date == date).all()
    day_dicts = [{"ids": {e.id}, "category": e.category, "start": e.start_ts,
                  "end": e.end_ts, "is_anomaly": e.is_anomaly, "_row": e} for e in evs]
    detect_anomaly(day_dicts, settings.anomaly, family_names)
    for d in day_dicts:
        d["_row"].is_anomaly = d["is_anomaly"]
    # 关键帧：每组最高分段的密度峰处抓帧（失败留 NULL 不阻断）
    from pathlib import Path

    from camdigest.media import grab_frame
    kf_dir = settings.storage.data_dir / "keyframes" / date
    for ev, top in ev_top:
        out = kf_dir / f"ev{ev.id}.jpg"
        try:
            kf_dir.mkdir(parents=True, exist_ok=True)
            grab_frame(Path(top.media_path), pick_keyframe_ts(top), out)
            ev.keyframe_path = str(out) if out.exists() else None
        except Exception as e:  # noqa: BLE001 —— 关键帧失败不阻断归并（M1 容错）
            log.warning("S7 关键帧失败 ev=%d：%s", ev.id, e)
            ev.keyframe_path = None
    anomalies = sum(1 for e in evs if e.is_anomaly)
    log.info("S7 %s：%d 片段 → %d 事件（异常 %d）", date, len(lites), created, anomalies)
    return created
