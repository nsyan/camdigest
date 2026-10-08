# src/camdigest/web/app.py
"""FastAPI 应用工厂：三只读静态挂载 + 模板 + 路由（设计 §3）。"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from camdigest.config import Settings

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app(settings: Settings, *, with_scheduler: bool = False) -> FastAPI:
    app = FastAPI(title="CamDigest", docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.db_url = f"sqlite:///{Path(settings.storage.data_dir) / 'camdigest.db'}"
    app.state.templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    data = Path(settings.storage.data_dir)
    for sub, mount in (("highlights", "/highlights"), ("keyframes", "/keyframes"),
                       ("reports", "/reports")):
        (data / sub).mkdir(parents=True, exist_ok=True)
        app.mount(mount, StaticFiles(directory=data / sub), name=sub)
    from camdigest.web.views import router
    app.include_router(router)
    from camdigest.web.runner import RunManager, start_scheduler
    app.state.runs = RunManager(settings)
    if with_scheduler:
        start_scheduler(app)
    return app
