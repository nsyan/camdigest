# src/camdigest/pipeline/s10_clip.py
"""S10 剪辑导出：TS 切片 → concat 统一时间轴 → 四档精华（spec §7，ADR-0002）。

先切 TS（-c copy）再 concat（-c:v copy + -c:a aac 重编码保原声）；无声输入不因
音频参数失败。per_camera=true 时合并版之外每机位再各导出一份（不占档位时长）。
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from camdigest.config import Settings
from camdigest.pipeline.s10_selection import ClipPlan, tier_subset

log = logging.getLogger(__name__)


def cut_ts(media_path: str | Path, start_s: float, end_s: float, out: Path) -> Path:
    """按秒切 TS 片段：输入 seek + 输出 -t 时长（-c copy；切点对齐最近前关键帧，
    GOP 1-2s 时误差可接受。计划原命令 -ss/-to 均前置会因关键帧回退产出超长段，
    见账本 T17-a）。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-i", str(media_path),
         "-t", f"{max(0.0, end_s - start_s):.3f}",
         "-c", "copy", "-f", "mpegts", str(out)],
        check=True, capture_output=True)
    return out


def concat_ts(ts_files: list[Path], out_mp4: Path) -> Path:
    """concat demuxer 拼接：视频流直拷，音频 aac 重编码统一时间轴（无声输入安全）。"""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    listing = out_mp4.parent / f"{out_mp4.stem}.concat.txt"
    listing.write_text("".join(f"file '{f.resolve()}'\n" for f in ts_files), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c:v", "copy", "-c:a", "aac", "-movflags", "+faststart", str(out_mp4)],
        check=True, capture_output=True)
    listing.unlink(missing_ok=True)
    return out_mp4


def _export_one(date: str, plans: list[ClipPlan], tier: int,
                settings: Settings, out: Path) -> Path:
    tmp_dir = Path(settings.storage.data_dir) / "tmp" / f"h{tier}-{date}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ts_files = [cut_ts(p.media_path, p.start_s, p.end_s,
                       tmp_dir / f"clip{i}.ts") for i, p in enumerate(plans)]
    try:
        concat_ts(ts_files, out)
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return out


def export_highlights(date: str, pool: list[ClipPlan], settings: Settings,
                      session=None) -> list[Path]:
    """四档导出：60min 池 → tier_subset Top-N → 文件 + highlights 行。

    per_camera=true：合并版完成后每机位再各导一份（取该机位子集，camera_id 落行）。
    """
    import time

    from camdigest.db import Highlight
    from camdigest.media import probe

    outs: list[Path] = []
    base = Path(settings.storage.data_dir) / "highlights" / date
    for tier in settings.highlight.tiers:
        subset = tier_subset(pool, tier)
        if not subset:
            continue
        t0 = time.monotonic()
        out = _export_one(date, subset, tier, settings, base / f"精华_{tier}min.mp4")
        outs.append(out)
        log.info("S10 导出 tier=%d → %s（%.1fs，%.1fs 素材）",
                 tier, out, time.monotonic() - t0, probe(out).duration)
        if session is not None:
            session.add(Highlight(date=date, camera_id=None, tier_minutes=tier,
                                  file_path=str(out), duration=probe(out).duration,
                                  bytes=out.stat().st_size,
                                  event_ids=[p.event_id for p in subset]))
    if settings.highlight.per_camera:
        cams = sorted({p.camera_id for p in pool})
        for cam in cams:
            cam_pool = [p for p in pool if p.camera_id == cam]
            for tier in settings.highlight.tiers:
                subset = tier_subset(cam_pool, tier)
                if not subset:
                    continue
                out = _export_one(date, subset, tier, settings,
                                  base / f"精华_{tier}min_{cam}.mp4")
                outs.append(out)
                if session is not None:
                    session.add(Highlight(date=date, camera_id=cam, tier_minutes=tier,
                                          file_path=str(out), duration=probe(out).duration,
                                          bytes=out.stat().st_size,
                                          event_ids=[p.event_id for p in subset]))
    if session is not None:
        session.flush()
    return outs
