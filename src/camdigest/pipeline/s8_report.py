# src/camdigest/pipeline/s8_report.py
"""S8 日报：事件清单 → 报告模型 → 五板块 Markdown（spec §4/§8）。

板块⑤的数据源（精华路径）在 S10 之前即可确定性给出（spec §8）。
"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from camdigest.config import Settings
from camdigest.db import Event, Report, Segment
from camdigest.llm.anthropic_adapter import AnthropicReport
from camdigest.llm.openai_adapter import OpenAIReport


def build_report_model(settings: Settings):
    """按 models.report.protocol 选报告角色适配器（ADR-0001：识别角色不走此处）。"""
    cfg = settings.models.report
    if cfg.protocol == "anthropic":
        return AnthropicReport(cfg)
    return OpenAIReport(cfg)


def highlight_paths(date: str, settings: Settings) -> list[str]:
    """四档精华的确定性落盘路径（S10 未跑即可预知，spec §8 板块⑤）。"""
    base = Path(settings.storage.data_dir) / "highlights" / date
    return [str(base / f"精华_{t}min.mp4") for t in settings.highlight.tiers]


def build_payload(events: list[dict], settings: Settings) -> dict:
    """事件 dict 清单 → 报告模型输入 payload。时刻 HH:MM；关键帧有值为真。"""
    date = (events[0].get("date") if events else None) \
        or (events[0]["start_ts"].strftime("%Y-%m-%d") if events else "")
    return {
        "events": [{
            "id": e["id"],
            "time": e["start_ts"].strftime("%H:%M"),
            "title": e["title"],
            "description": e["description"],
            "category": e["category"],
            "people": e.get("people", []),
            "is_anomaly": e["is_anomaly"],
            "has_keyframe": bool(e.get("keyframe_path")),
            "keyframe": e.get("keyframe_path"),
        } for e in events],
        "highlight_paths": highlight_paths(date, settings),
    }


def _events_as_dicts(date: str, session: Session) -> list[dict]:
    """Event 行 → payload dict；people 从事件所属候选片段的 draft 聚合（去重保序）。"""
    out = []
    for ev in session.query(Event).filter(Event.date == date).order_by(Event.start_ts):
        people: list[str] = []
        for seg_id in ev.segment_ids or []:
            seg = session.get(Segment, seg_id)
            for p in (seg.draft or {}).get("people", []) if seg else []:
                if p not in people:
                    people.append(p)
        out.append({"id": ev.id, "date": ev.date, "start_ts": ev.start_ts,
                    "end_ts": ev.end_ts, "category": ev.category, "score": ev.score,
                    "title": ev.title, "description": ev.description, "people": people,
                    "is_anomaly": ev.is_anomaly, "keyframe_path": ev.keyframe_path})
    return out


def run_report(date: str, session: Session, settings: Settings) -> Path:
    events = _events_as_dicts(date, session)
    payload = build_payload(events, settings)
    md = build_report_model(settings).write(date, payload)
    out = Path(settings.storage.data_dir) / "reports" / f"{date}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    session.merge(Report(date=date, md_path=str(out)))
    return out
