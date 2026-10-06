# tests/test_s10_selection.py —— 单池四步选段纯函数重单测（ADR-0002）
from datetime import UTC, datetime, timedelta

from camdigest.config import HighlightCfg
from camdigest.pipeline.s10_selection import (
    SelEvent,
    densest_window,
    select_pool,
    tier_subset,
)

_T0 = datetime(2026, 9, 29, 0, 0, tzinfo=UTC)


def _ev(eid, hour, minute=0, score=80, cam="gate", lens="single",
        face_ts=None, scene_ts=None, seg=(0.0, 120.0), confs=None):
    return SelEvent(
        event_id=eid, camera_id=cam, device=cam.rsplit("-", 1)[0], lens=lens,
        media_path=f"/{cam}.mp4", start_s=seg[0], end_s=seg[1], score=score,
        start_ts=_T0 + timedelta(hours=hour, minutes=minute),
        face_ts=face_ts or [], scene_ts=scene_ts or [],
        face_confs=confs or [])


def test_hourly_guarantee_includes_night():
    """夜间 3 点的 1 个事件也必须进池（每小时保底含 0-6 点）。"""
    night = _ev(1, 3, score=10)
    day = _ev(2, 10, score=90)
    pool = select_pool([day, night], HighlightCfg())
    hours = {e.start_ts.hour for e in pool}
    assert 3 in hours and 10 in hours


def test_gap_dedup_30min_within_hour():
    """间隔 <30 分钟视为同一事件跳过——仅在目标提前达成时（步骤④不触发）可观察。

    tiers=[1] → 目标 60s = 3×20s 剪辑；a+d+c 恰好达标，b（距 a 20min）被间隔去重排除。
    （默认 60min 目标在 20s/段上限下实际不可达，步骤④放开兜底几乎必触发——见
    test_relaxed_fallback_under_60min 与 CONTEXT.md「当天事件不足则同步缩水」。）
    """
    a = _ev(1, 10, 0, score=90)
    b = _ev(2, 10, 20, score=89)   # 距 a 20 分钟 <30 → 步骤②③都跳过
    c = _ev(3, 10, 45, score=88)   # 距 a 45 分钟 → 可选
    d = _ev(4, 14, 0, score=50)    # 另一小时保底
    pool = select_pool([a, b, c, d], HighlightCfg(tiers=[1]))
    ids = {e.event_id for e in pool}
    assert ids == {1, 3, 4} and 2 not in ids


def test_relaxed_fallback_under_60min():
    """总分时长不足目标时，第四步放开去重兜底全收（含被间隔去重跳过的）。"""
    a = _ev(1, 9, score=90, seg=(0.0, 30.0))
    b = _ev(2, 9, 10, score=85, seg=(0.0, 30.0))   # 与 a 间隔 <30min，正常会跳过
    pool = select_pool([a, b], HighlightCfg())
    assert {e.event_id for e in pool} == {1, 2}    # 放开去重兜底


def test_tier_subset_5min_subsets_60min():
    """ADR-0002：短档 ⊆ 长档——5min 档事件必在 60min 档中。"""
    evs = [_ev(i, h, score=90 - i * 5, seg=(0.0, 120.0))
           for i, h in enumerate([8, 9, 10, 11, 12, 13, 14, 15], start=1)]
    pool = select_pool(evs, HighlightCfg())
    t60 = {p.event_id for p in tier_subset(pool, 60)}
    t5 = {p.event_id for p in tier_subset(pool, 5)}
    assert t5 and t5 <= t60


def test_densest_window_centered_when_no_signals():
    w = densest_window([], [], 0.0, 100.0, 20.0)
    s, e = w
    assert abs((s + e) / 2 - 50.0) < 1e-6 and e - s <= 20.0 + 1e-6


def test_densest_window_faces_beat_scene_tiebreak():
    # 脸在 60s 处，scene 突变在 10s 处 → 人脸密度优先
    w = densest_window([60.0], [10.0], 0.0, 120.0, 20.0)
    assert 50.0 <= w[0] <= 61.0


def test_best_of_event_prefers_conf_then_fixed():
    """同事件多路：人脸平均置信度高者胜；平局取 fixed 路。"""
    a = _ev(1, 10, cam="living-fixed", lens="fixed", confs=[0.6], face_ts=[10.0])
    b = _ev(1, 10, cam="living-ptz", lens="ptz", confs=[0.9], face_ts=[10.0])
    best = SelEvent.best_of_event([a, b])
    assert best.camera_id == "living-ptz"
    c = _ev(2, 11, cam="living-fixed", lens="fixed", confs=[0.8], face_ts=[5.0])
    d = _ev(2, 11, cam="living-ptz", lens="ptz", confs=[0.8], face_ts=[5.0])
    assert SelEvent.best_of_event([c, d]).lens == "fixed"   # 平局取 fixed


def test_select_pool_sorted_by_real_time():
    evs = [_ev(1, 15, score=90), _ev(2, 8, score=80), _ev(3, 12, score=70)]
    pool = select_pool(evs, HighlightCfg())
    ts = [e.start_ts for e in pool]
    assert ts == sorted(ts)
