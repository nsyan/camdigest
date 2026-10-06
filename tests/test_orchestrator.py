# tests/test_orchestrator.py —— jobs 断点、阶段顺序、selected_only 调序、until
import pytest

from camdigest import db
from camdigest.config import Settings
from camdigest.pipeline import orchestrator as orch


def _settings(tmp_path, selected_only=False):
    return Settings(
        cameras=[{"id": "gate", "name": "大门", "device": "gate",
                  "lens": "single", "dir": str(tmp_path / "gate")}],
        models={"recognition": {"base_url": "http://x", "model": "m", "api_key": "k"},
                "report": {"base_url": "http://x", "model": "m", "api_key": "k"}},
        prefilter={"whisper": {"selected_only": selected_only}},
        storage={"data_dir": str(tmp_path)})


def _fake_stages(calls, fail_at=None):
    def make(name):
        def fn(date, session, settings):
            calls.append(name)
            if fail_at == name:
                raise RuntimeError(f"boom-{name}")
            return 1
        return fn
    return [(n, make(n)) for n in ["s1_index", "s2_prefilter", "s3_person",
        "s4_face", "s5_audio", "s6_recognize", "s7_merge", "s8_report",
        "s9_publish", "s10_clip"]]


def test_run_day_order_and_completion(tmp_path):
    calls = []
    settings = _settings(tmp_path)
    url = f"sqlite:///{tmp_path}/t.db"
    db.init_db(url)
    with db.session_scope(url) as s:
        result = orch.run_day("2026-09-29", settings, session=s, stages=_fake_stages(calls))
    assert calls == ["s1_index", "s2_prefilter", "s3_person", "s4_face", "s5_audio",
                     "s6_recognize", "s7_merge", "s8_report", "s9_publish", "s10_clip"]
    assert result == {"s1_index": 1, "s2_prefilter": 1, "s3_person": 1,
                      "s4_face": 1, "s5_audio": 1, "s6_recognize": 1,
                      "s7_merge": 1, "s8_report": 1, "s9_publish": 1,
                      "s10_clip": 1}
    with db.session_scope(url) as s:
        jobs = {j.stage: j.status for j in s.query(db.Job).all()}
        assert set(jobs.values()) == {"done"}


def test_run_day_skips_done_stages(tmp_path):
    calls = []
    settings = _settings(tmp_path)
    url = f"sqlite:///{tmp_path}/t.db"
    db.init_db(url)
    with db.session_scope(url) as s:
        orch.run_day("2026-09-29", settings, session=s, stages=_fake_stages(calls))
    calls.clear()
    with db.session_scope(url) as s:
        orch.run_day("2026-09-29", settings, session=s, stages=_fake_stages(calls))
    assert calls == []            # 全部 done → 跳过


def test_run_day_failed_stage_interrupts_and_reruns_from_there(tmp_path):
    calls = []
    settings = _settings(tmp_path)
    url = f"sqlite:///{tmp_path}/t.db"
    db.init_db(url)
    with db.session_scope(url) as s, pytest.raises(RuntimeError, match="boom-s3"):
        orch.run_day("2026-09-29", settings, session=s,
                     stages=_fake_stages(calls, fail_at="s3_person"))
    with db.session_scope(url) as s:
        jobs = {j.stage: j.status for j in s.query(db.Job).all()}
        assert jobs["s1_index"] == "done"
        assert jobs["s2_prefilter"] == "done"
        assert jobs["s3_person"] == "failed"
    calls.clear()
    with db.session_scope(url) as s:   # 重跑：s1/s2 done 跳过，从 s3 继续
        orch.run_day("2026-09-29", settings, session=s, stages=_fake_stages(calls))
    assert calls == ["s3_person", "s4_face", "s5_audio", "s6_recognize",
                     "s7_merge", "s8_report", "s9_publish", "s10_clip"]


def test_run_day_selected_only_moves_s5_after_s6(tmp_path):
    calls = []
    settings = _settings(tmp_path, selected_only=True)
    url = f"sqlite:///{tmp_path}/t.db"
    db.init_db(url)
    with db.session_scope(url) as s:
        orch.run_day("2026-09-29", settings, session=s, stages=_fake_stages(calls))
    assert calls.index("s6_recognize") < calls.index("s5_audio")


def test_run_day_until_stops_after_stage(tmp_path):
    calls = []
    settings = _settings(tmp_path)
    url = f"sqlite:///{tmp_path}/t.db"
    db.init_db(url)
    with db.session_scope(url) as s:
        orch.run_day("2026-09-29", settings, session=s, stages=_fake_stages(calls),
                     until="s3_person")
    assert calls == ["s1_index", "s2_prefilter", "s3_person"]


def test_stages_constant_has_ten_named_stages():
    names = [n for n, _ in orch.STAGES]
    assert names == ["s1_index", "s2_prefilter", "s3_person", "s4_face", "s5_audio",
                     "s6_recognize", "s7_merge", "s8_report", "s9_publish", "s10_clip"]
