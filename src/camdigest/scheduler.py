# src/camdigest/scheduler.py
"""APScheduler 日批入口：daily_at 跑前一天（spec §4，默认 03:00）。"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from camdigest.config import Settings

log = logging.getLogger(__name__)


def _job_date(now: datetime) -> str:
    """批处理目标日期 = 调度时区下的昨天（跨月/跨年安全）。"""
    return (now.date() - timedelta(days=1)).isoformat()


def build_trigger(settings: Settings) -> CronTrigger:
    hour, minute = (int(x) for x in settings.schedule.daily_at.split(":"))
    return CronTrigger(hour=hour, minute=minute, timezone=settings.schedule.timezone)


def run_scheduler(settings: Settings) -> None:
    sched = BlockingScheduler(timezone=settings.schedule.timezone)
    trigger = build_trigger(settings)

    def job():
        from camdigest.pipeline.orchestrator import run_day
        now = datetime.now(tz=ZoneInfo(settings.schedule.timezone))
        run_day(_job_date(now), settings)

    sched.add_job(job, trigger, misfire_grace_time=6 * 3600)  # NAS 休眠醒来后仍补跑
    nxt = sched.get_jobs()
    log.info("camdigest scheduler started; next run at %s",
             nxt[0].next_run_time if nxt else "(unknown)")
    sched.start()
