# tests/test_scheduler.py —— _job_date 与 trigger 构造（不真跑 BlockingScheduler）
from datetime import UTC, datetime

from camdigest.config import Settings
from camdigest.scheduler import _job_date, build_trigger


def _settings():
    return Settings(
        cameras=[{"id": "gate", "name": "大门", "device": "gate",
                  "lens": "single", "dir": "/a"}],
        models={"recognition": {"base_url": "http://x", "model": "m", "api_key": "k"},
                "report": {"base_url": "http://x", "model": "m", "api_key": "k"}},
        schedule={"daily_at": "03:00", "timezone": "Asia/Shanghai"})


def test_job_date_returns_yesterday():
    assert _job_date(datetime(2026, 10, 1, 3, 0, tzinfo=UTC)) == "2026-09-30"
    assert _job_date(datetime(2026, 1, 1, 0, 30, tzinfo=UTC)) == "2025-12-31"   # 跨年
    assert _job_date(datetime(2026, 3, 1, 23, 59, tzinfo=UTC)) == "2026-02-28"  # 平年 2 月


def test_build_trigger_hour_minute_timezone():
    trig = build_trigger(_settings())
    text = str(trig)
    assert "minute='0'" in text and "hour='3'" in text
    assert "Asia/Shanghai" in str(trig.timezone) or "CST" in str(trig.timezone) \
        or "UTC+08:00" in str(trig.timezone)


def test_run_scheduler_registers_cron_job():
    from apscheduler.triggers.cron import CronTrigger

    from camdigest import scheduler as sch
    registered = {}

    class FakeScheduler:
        def __init__(self, *a, **kw):
            pass

        def add_job(self, fn, trigger, **kw):
            registered["fn"] = fn
            registered["trigger"] = trigger

        def start(self):
            registered["started"] = True

        def get_jobs(self):
            return []

    monkey_orig = sch.BlockingScheduler
    sch.BlockingScheduler = FakeScheduler
    try:
        sch.run_scheduler(_settings())
    finally:
        sch.BlockingScheduler = monkey_orig
    assert registered["started"] is True
    assert isinstance(registered["trigger"], CronTrigger)
    assert callable(registered["fn"])
