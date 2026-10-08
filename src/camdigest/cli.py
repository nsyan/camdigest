# src/camdigest/cli.py
"""camdigest 命令行：run / enroll-faces / backfill / schedule（spec §10）。"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

from camdigest.config import Settings

DEFAULT_CONFIG = "config/config.yaml"


def _load_settings(path: Path) -> Settings:
    return Settings.load(path)


def _run_day_for_cli(date: str, settings: Settings, until: str | None = None) -> dict:
    from camdigest.pipeline.orchestrator import run_day
    return run_day(date, settings, until=until)


def _enroll_for_cli(data_dir: Path, rest_url: str, session=None) -> int:
    from camdigest.pipeline.s4_face import enroll_faces
    return enroll_faces(data_dir, rest_url, session=session)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="camdigest",
                                description="家庭监控录像 → 每日精华 + 飞书日报")
    p.add_argument("--config", default=None, help="配置文件路径（默认 config/config.yaml，"
                                                  "CAMDIGEST_CONFIG 优先）")
    sub = p.add_subparsers(dest="command", required=True)

    def _add(sub_p):  # --config 在子命令后也可用（计划 §19 规格形态）
        sub_p.add_argument("--config", default=None, help="配置文件路径")

    run = sub.add_parser("run", help="跑某一天的完整流水线")
    run.add_argument("--date", required=True, help="YYYY-MM-DD")
    run.add_argument("--until", default=None, help="跑到指定阶段为止（如 s6_recognize）")
    _add(run)

    enroll = sub.add_parser("enroll-faces", help="人脸建档：/faces/{身份名}/*.jpg → registry")
    enroll.add_argument("--data-dir", default=None, help="人脸目录（默认 settings.faces.registry_dir）")
    _add(enroll)

    bf = sub.add_parser("backfill", help="逐日补跑历史日期")
    bf.add_argument("--from", dest="from_date", required=True, help="起始 YYYY-MM-DD")
    bf.add_argument("--to", dest="to_date", required=True, help="结束 YYYY-MM-DD")
    bf.add_argument("--force", action="store_true", help="已有日报的日期也重跑")
    _add(bf)

    sched = sub.add_parser("schedule", help="按 schedule.daily_at 每日批处理（跑前一天）")
    _add(sched)
    return p


def _resolve_config(args) -> Path:
    return Path(args.config or os.environ.get("CAMDIGEST_CONFIG") or DEFAULT_CONFIG)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = _load_settings(_resolve_config(args))
    from camdigest.logging_setup import setup_logging
    setup_logging(settings.storage.data_dir)

    if args.command == "run":
        results = _run_day_for_cli(args.date, settings, until=args.until)
        for stage, n in results.items():
            print(f"{stage}: {n}")
        return 0
    if args.command == "enroll-faces":
        data_dir = Path(args.data_dir) if args.data_dir else settings.faces.registry_dir
        from camdigest.db import init_db, session_scope
        url = f"sqlite:///{Path(settings.storage.data_dir) / 'camdigest.db'}"
        init_db(url)
        with session_scope(url) as s:   # spec §5：同步 identities 行
            n = _enroll_for_cli(data_dir, settings.faces.rest_url, session=s)
        print(f"enrolled {n} face photos")
        return 0
    if args.command == "backfill":
        from camdigest.db import Report, init_db, session_scope
        d = date.fromisoformat(args.from_date)
        end = date.fromisoformat(args.to_date)
        url = f"sqlite:///{Path(settings.storage.data_dir) / 'camdigest.db'}"
        init_db(url)
        while d <= end:
            day = d.isoformat()
            with session_scope(url) as s:
                has_report = s.query(Report).filter(Report.date == day).one_or_none()
            if has_report and not args.force:
                print(f"skipped {day} (report exists; --force to rerun)")
            else:
                _run_day_for_cli(day, settings)
                print(f"backfilled {day}")
            d += timedelta(days=1)
        return 0
    if args.command == "schedule":
        from camdigest.scheduler import run_scheduler
        run_scheduler(settings)
        return 0
    parser.error(f"未知子命令 {args.command}")   # unreachable（required=True 已兜底）
    return 2


if __name__ == "__main__":
    sys.exit(main())
