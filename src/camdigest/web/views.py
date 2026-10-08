# src/camdigest/web/views.py
"""Web 路由：全部只读自 SQLite + /data（设计 §4）。"""
from __future__ import annotations

import re
from pathlib import Path

import mistune
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import RedirectResponse

router = APIRouter()

# 日报板块⑤的绝对路径 → 点播链接（设计 §4：渲染时替换，md 文件不动）
_HL_PATH_RE = re.compile(r"/data/highlights/([\d-]+)/([^\s。，；）]+?\.mp4)")


def render_report_html(md: str) -> str:
    md = _HL_PATH_RE.sub(r"[\1/\2](/highlights/\1/\2)", md)
    return mistune.html(md)


def _mount_url(settings, file_path: str, mount: str) -> str | None:
    """/data/{mount}/ 下的文件 → 静态挂载 URL；不在该目录下返回 None。"""
    try:
        rel = Path(file_path).resolve().relative_to(
            (Path(settings.storage.data_dir) / mount).resolve())
        return f"/{mount}/{rel}"
    except ValueError:
        return None


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


@router.get("/day/{date}")
def day(date: str, request: Request):
    import datetime as _dt

    from camdigest import db
    try:
        _dt.date.fromisoformat(date)
    except ValueError:
        raise HTTPException(404)           # 非法日期与不存在同等对待（设计 §4）
    settings = request.app.state.settings
    with db.session_scope(request.app.state.db_url) as s:
        highlights = s.query(db.Highlight).filter(db.Highlight.date == date).all()
        events = (s.query(db.Event).filter(db.Event.date == date)
                  .order_by(db.Event.start_ts).all())
        if not highlights and not events:
            raise HTTPException(404)
        report = s.query(db.Report).filter(db.Report.date == date).one_or_none()
        report_html = None
        if report and Path(report.md_path).exists():
            report_html = render_report_html(
                Path(report.md_path).read_text(encoding="utf-8"))
        tiers = sorted({h.tier_minutes for h in highlights})
        hls = [{"tier": h.tier_minutes,
                "url": _mount_url(settings, h.file_path, "highlights")}
               for h in highlights]
        evs = [{"time": e.start_ts.strftime("%H:%M"), "title": e.title,
                "description": e.description, "anomaly": e.is_anomaly,
                "keyframe": _mount_url(settings, e.keyframe_path, "keyframes")
                if e.keyframe_path else None}
               for e in events]
    return request.app.state.templates.TemplateResponse(
        request, "day.html", {"date": date, "tiers": tiers, "highlights": hls,
                              "events": evs, "report_html": report_html})


@router.get("/jobs")
def jobs_page(request: Request):
    from camdigest import db
    from camdigest.pipeline.orchestrator import STAGES
    stage_names = [n for n, _ in STAGES]
    with db.session_scope(request.app.state.db_url) as s:
        jobs = s.query(db.Job).all()
    by_date: dict[str, dict[str, tuple[str, str | None]]] = {}
    for j in jobs:
        by_date.setdefault(j.date, {})[j.stage] = (j.status, j.error)
    dates = sorted(by_date, reverse=True)
    rows = [{"date": d,
             "stages": [by_date.get(d, {}).get(n) for n in stage_names]}
            for d in dates]
    return request.app.state.templates.TemplateResponse(
        request, "jobs.html", {"rows": rows, "stage_names": stage_names})


@router.get("/run")
def run_form(request: Request):
    from camdigest.pipeline.orchestrator import STAGES
    return request.app.state.templates.TemplateResponse(
        request, "run.html", {"stages": [n for n, _ in STAGES]})


@router.post("/run")
def run_submit(request: Request, date: str = Form(...),
               until: str | None = Form(None)):
    import datetime as _dt

    from camdigest.pipeline.orchestrator import STAGES
    try:
        _dt.date.fromisoformat(date)
    except ValueError:
        raise HTTPException(422)           # 非法日期（设计 §4 参数白名单）
    if until is not None and until not in {n for n, _ in STAGES}:
        raise HTTPException(422)           # 未知阶段名
    ok = request.app.state.runs.submit(date, until)
    return RedirectResponse("/jobs?msg=busy" if not ok else "/jobs", status_code=303)


@router.get("/faces")
def faces_page(request: Request):
    from camdigest import db
    settings = request.app.state.settings
    registry_dir = Path(settings.faces.registry_dir)
    with db.session_scope(request.app.state.db_url) as s:
        identities = [{"name": i.name,
                       "archived": i.unknown_cluster}
                      for i in s.query(db.Identity).order_by(db.Identity.name)]
        clusters: dict[int, int] = {}
        for uf in s.query(db.UnknownFace).all():
            clusters[uf.cluster_id] = clusters.get(uf.cluster_id, 0) + 1
    known = [{"name": i["name"],
              "photo": f"/identity-photo/{i['name']}"}
             for i in identities if (registry_dir / i["name"]).is_dir()]
    unknown = [{"cluster": cid, "count": n,
                "photo": f"/keyframes/.clusters/{cid}.jpg"}
               for cid, n in sorted(clusters.items())]
    return request.app.state.templates.TemplateResponse(
        request, "faces.html", {"known": known, "unknown": unknown})


@router.get("/identity-photo/{name}")
def identity_photo(name: str, request: Request):
    from fastapi.responses import FileResponse
    if "/" in name or ".." in name or not name:
        raise HTTPException(422)           # 路径穿越白名单（设计 §7）
    path = Path(request.app.state.settings.faces.registry_dir) / name
    photos = sorted(path.glob("*.jpg")) if path.is_dir() else []
    if not photos:
        raise HTTPException(404)
    return FileResponse(photos[0])


@router.post("/faces/archive")
def faces_archive(request: Request, cluster_id: int = Form(...),
                  name: str = Form(...)):
    if not name or "/" in name or ".." in name:
        raise HTTPException(422)           # 身份名白名单
    from camdigest import db
    from camdigest.web.faces import archive_cluster
    with db.session_scope(request.app.state.db_url) as s:
        archive_cluster(cluster_id, name, s, request.app.state.settings)
    return RedirectResponse("/faces", status_code=303)