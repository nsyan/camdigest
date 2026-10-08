# src/camdigest/web/runner.py
"""后台补跑 + 与日批调度互斥（设计 §3）。"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from fastapi import FastAPI

from camdigest.config import Settings

log = logging.getLogger(__name__)


class RunManager:
    """进程级单实例：手动补跑与 APScheduler 日批竞争同一把锁。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._busy = threading.Event()
        self.post_run: list[Callable[[str], None]] = []   # run_day 成功后的钩子（日批扫脸等）

    def submit(self, date: str, until: str | None = None) -> bool:
        if self._busy.is_set():
            return False
        self._busy.set()

        def work():
            try:
                with self._lock:
                    from camdigest.pipeline.orchestrator import run_day
                    run_day(date, self.settings, until=until)
                for hook in self.post_run:
                    try:
                        hook(date)
                    except Exception:
                        log.exception("post_run 钩子失败（%s）", date)
            except Exception:
                log.exception("run_day %s 失败", date)
            finally:
                self._busy.clear()

        threading.Thread(target=work, name=f"run-{date}", daemon=True).start()
        return True

    @property
    def busy(self) -> bool:
        return self._busy.is_set()


def start_scheduler(app: FastAPI) -> None:
    """随 web 进程启动日批调度（与手动补跑共用 RunManager）。"""
    import datetime as _dt
    from zoneinfo import ZoneInfo

    from apscheduler.schedulers.background import BackgroundScheduler

    from camdigest.scheduler import _job_date, build_trigger
    settings = app.state.settings
    manager: RunManager = app.state.runs

    def job():
        now = _dt.datetime.now(tz=ZoneInfo(settings.schedule.timezone))
        day = _job_date(now)
        if not manager.submit(day):
            log.warning("日批 %s 跳过：已有任务在跑", day)

    sched = BackgroundScheduler(timezone=settings.schedule.timezone)
    sched.add_job(job, build_trigger(settings), misfire_grace_time=6 * 3600)
    sched.start()
    app.state.scheduler = sched
