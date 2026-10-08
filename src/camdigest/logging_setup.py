# src/camdigest/logging_setup.py
"""统一日志初始化：stdout（docker logs 可见）+ /data/logs 滚动文件。"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_FMT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def setup_logging(data_dir: Path, level: int = logging.INFO) -> None:
    root = logging.getLogger("camdigest")
    if root.handlers:                      # 幂等：重复调用不叠加 handler
        return
    root.setLevel(level)
    fmt = logging.Formatter(_FMT)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    try:
        log_dir = Path(data_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_dir / "camdigest.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:                        # 目录不可写：退化为仅 stdout，日志永不阻断主流程
        root.warning("日志文件不可写，退化为仅 stdout：%s/logs", data_dir)
