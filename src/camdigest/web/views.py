# src/camdigest/web/views.py
"""Web 路由：全部只读自 SQLite + /data（设计 §4）。"""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/")
def index(request: Request):
    from camdigest import db
    with db.session_scope(request.app.state.db_url) as s:
        dates = {r.date for r in s.query(db.Report).all()}
        days = []
        for d in sorted(dates, reverse=True):
            hl = s.query(db.Highlight).filter(db.Highlight.date == d).count()
            evs = s.query(db.Event).filter(db.Event.date == d).all()
            days.append({"date": d, "tiers": hl, "events": len(evs),
                         "anomalies": sum(1 for e in evs if e.is_anomaly)})
    return request.app.state.templates.TemplateResponse(
        request, "index.html", {"days": days})
