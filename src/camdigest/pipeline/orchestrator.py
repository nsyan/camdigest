# src/camdigest/pipeline/orchestrator.py
"""编排器：STAGES 顺序执行 + jobs(date, stage) 断点续跑（spec §4）。

每阶段幂等靠 jobs 表：done 跳过、failed 重跑、异常置 failed 并上抛。
whisper.selected_only=true 时 S5 挪到 S6 之后（只转写已选中片段，spec §4 兜底）。
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from sqlalchemy.orm import Session

from camdigest.config import Settings
from camdigest.db import Job

log = logging.getLogger(__name__)


def _run_s1(date: str, session: Session, settings: Settings) -> int:
    from camdigest.pipeline.s1_index import index_camera
    return sum(index_camera(c, session) for c in settings.cameras)


def _run_s2(date: str, session: Session, settings: Settings) -> int:
    from camdigest.pipeline.s2_prefilter import run_prefilter
    return run_prefilter(date, session, settings)


def _run_s3(date: str, session: Session, settings: Settings) -> int:
    from camdigest.pipeline.s3_person import run_person
    return run_person(date, session, settings)


def _run_s4(date: str, session: Session, settings: Settings) -> int:
    from camdigest.pipeline.s4_face import run_face
    return run_face(date, session, settings)


def _run_s5(date: str, session: Session, settings: Settings) -> int:
    from camdigest.pipeline.s5_audio import run_audio
    return run_audio(date, session, settings)


def _run_s6(date: str, session: Session, settings: Settings) -> int:
    from camdigest.pipeline.s6_recognize import run_recognition
    return run_recognition(date, session, settings)


def _run_s7(date: str, session: Session, settings: Settings) -> int:
    from camdigest.pipeline.s7_merge import run_merge
    return run_merge(date, session, settings)


def _run_s8(date: str, session: Session, settings: Settings) -> int:
    from camdigest.pipeline.s8_report import run_report
    return 1 if run_report(date, session, settings) else 0   # Path → 计数


def _run_s9(date: str, session: Session, settings: Settings) -> int:
    from camdigest.publish.feishu import publish_feishu
    return 1 if publish_feishu(date, session, settings) else 0


def _run_s10(date: str, session: Session, settings: Settings) -> int:
    from camdigest.db import Event
    from camdigest.pipeline.query import cams_by_id, medias_for_date, segments_for_date
    from camdigest.pipeline.s10_clip import export_highlights
    from camdigest.pipeline.s10_selection import SelEvent, select_pool
    events = session.query(Event).filter(Event.date == date).all()
    segs = [seg for seg, _ in segments_for_date(date, session)]
    medias = {m.id: m for m in medias_for_date(date, session)}
    sels = SelEvent.from_rows(events, segs, medias, cams_by_id(settings))
    pool = select_pool(sels, settings.highlight)
    export_highlights(date, pool, settings, session=session)
    return len(pool)


STAGES: list[tuple[str, Callable]] = [
    ("s1_index", _run_s1), ("s2_prefilter", _run_s2), ("s3_person", _run_s3),
    ("s4_face", _run_s4), ("s5_audio", _run_s5), ("s6_recognize", _run_s6),
    ("s7_merge", _run_s7), ("s8_report", _run_s8), ("s9_publish", _run_s9),
    ("s10_clip", _run_s10),
]


def _order(stages: list[tuple[str, Callable]], selected_only: bool) -> list:
    """selected_only 时把 s5_audio 挪到 s6_recognize 之后（对注入列表同样生效——
    阶段顺序语义归 run_day，列表只是候选项）。"""
    if not selected_only or not any(n == "s5_audio" for n, _ in stages):
        return list(stages)
    out = [s for s in stages if s[0] != "s5_audio"]
    pos = next(i for i, s in enumerate(out) if s[0] == "s6_recognize") + 1
    s5 = next(s for s in stages if s[0] == "s5_audio")
    return out[:pos] + [s5] + out[pos:]


def run_day(date: str, settings: Settings, *, until: str | None = None,
            session: Session | None = None,
            stages: list[tuple[str, Callable]] | None = None) -> dict[str, int]:
    """跑完整日批；jobs(date,stage) done 跳过 / failed 重跑 / 异常置 failed 并上抛。

    stages/until/session 为可注入参数（测试与 CLI --until 用）。
    """
    candidates = stages if stages is not None else STAGES
    order = _order(candidates, settings.prefilter.whisper.selected_only)
    log.info("run_day %s 开始，%d 个阶段", date, len(order))
    if until:
        names = [n for n, _ in order]
        order = order[:names.index(until) + 1]
    own_session = session is None
    url = f"sqlite:///{Path(settings.storage.data_dir) / 'camdigest.db'}"
    if own_session:
        from camdigest.db import init_db, session_scope
        init_db(url)
        ctx = session_scope(url)
    else:
        ctx = None
    results: dict[str, int] = {}
    try:
        if own_session:
            session = ctx.__enter__()   # type: ignore[union-attr]
        for name, fn in order:
            job = (session.query(Job)
                   .filter(Job.date == date, Job.stage == name).one_or_none())
            if job and job.status == "done":
                continue
            if job is None:
                job = Job(date=date, stage=name, status="running")
                session.add(job)
            else:
                job.status = "running"
                job.error = None
            session.flush()
            t0 = time.monotonic()
            try:
                results[name] = fn(date, session, settings)
                job.status = "done"
                session.flush()
                log.info("run_day %s %s done（%.1fs，%s）",
                         date, name, time.monotonic() - t0, results[name])
            except Exception as e:
                job.status = "failed"
                job.error = str(e)[:500]
                session.flush()
                log.error("run_day %s %s failed（%.1fs）：%s",
                          date, name, time.monotonic() - t0, e)
                raise
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    return results
