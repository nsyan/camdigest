# tests/test_logging.py —— 统一日志初始化 + 编排器阶段日志
import logging

import pytest

from camdigest.logging_setup import setup_logging


@pytest.fixture(autouse=True)
def _fresh_camdigest_logger():
    """setup_logging 写的是进程级全局 logger：每个用例前清空，避免跨文件污染。"""
    log = logging.getLogger("camdigest")
    log.handlers.clear()
    yield
    log.handlers.clear()


def test_setup_idempotent(tmp_path):
    setup_logging(tmp_path)
    setup_logging(tmp_path)                      # 重复调用不叠加
    log = logging.getLogger("camdigest")
    assert len(log.handlers) == 2                # stdout + 文件
    assert (tmp_path / "logs" / "camdigest.log").exists()


def test_run_day_stage_logs(caplog, tmp_path):
    from camdigest import db
    from camdigest.config import Settings
    from camdigest.pipeline import orchestrator as orch

    setup_logging(tmp_path)
    settings = Settings(
        cameras=[{"id": "gate", "name": "大门", "device": "gate",
                  "lens": "single", "dir": "/a"}],
        models={"recognition": {"base_url": "http://x", "model": "m", "api_key": "k"},
                "report": {"base_url": "http://x", "model": "m", "api_key": "k"}},
        storage={"data_dir": str(tmp_path)})
    url = f"sqlite:///{tmp_path}/t.db"
    db.init_db(url)
    calls = []

    def fake(name):
        return lambda date, session, s: calls.append(name) or 1

    stages = [(n, fake(n)) for n in ["s1_index", "s2_prefilter"]]
    with db.session_scope(url) as s, caplog.at_level(logging.INFO, logger="camdigest"):
        orch.run_day("2026-09-29", settings, session=s, stages=stages)
    msgs = [r.message for r in caplog.records]
    assert any("run_day 2026-09-29 开始" in m for m in msgs)
    assert any("s1_index done" in m for m in msgs)


def test_llm_retry_logged(caplog, video_file, tmp_path):
    import httpx

    from camdigest.config import ModelCfg
    from camdigest.llm.contracts import SegmentHint
    from camdigest.llm.openai_adapter import OpenAIRecognition

    n = {"n": 0}

    def handler(request):
        n["n"] += 1
        content = "junk" if n["n"] < 3 else (
            '{"category":"empty","people":[],"score":10,"title":"t","description":"d"}')
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    r = OpenAIRecognition(ModelCfg(base_url="http://fake/v1", model="m", api_key="k"))
    r._http = httpx.Client(transport=httpx.MockTransport(handler))
    with caplog.at_level(logging.WARNING, logger="camdigest"):
        r.analyze(__import__("pathlib").Path(video_file), SegmentHint(camera="x", lens="single"))
    assert any("LLM 重试" in rec.message for rec in caplog.records)


def test_s7_s10_stage_logs(caplog, tmp_path, video_file):
    import shutil
    from datetime import UTC, datetime, timedelta

    from camdigest import db
    from camdigest.config import Settings
    from camdigest.pipeline.s7_merge import run_merge
    from camdigest.pipeline.s10_clip import export_highlights
    from camdigest.pipeline.s10_selection import ClipPlan

    setup_logging(tmp_path / "data")
    settings = Settings(
        cameras=[{"id": "gate", "name": "大门", "device": "gate",
                  "lens": "single", "dir": "/x"}],
        models={"recognition": {"base_url": "http://x", "model": "m", "api_key": "k"},
                "report": {"base_url": "http://x", "model": "m", "api_key": "k"}},
        highlight={"tiers": [5]},
        storage={"data_dir": str(tmp_path / "data")})
    url = f"sqlite:///{tmp_path}/t.db"
    db.init_db(url)
    t0 = datetime(2026, 9, 29, 10, 0, tzinfo=UTC)
    with db.session_scope(url) as s:
        s.add(db.Camera(id="gate", name="大门", device="gate", lens="single",
                        dir="/x", pattern="*.mp4", timezone="Asia/Shanghai"))
        m = db.MediaFile(camera_id="gate", path="/x.mp4", start_ts=t0,
                         end_ts=t0 + timedelta(seconds=120), duration=120.0)
        s.add(m)
        s.flush()
        s.add(db.Segment(media_file_id=m.id, start_s=0, end_s=60, person_count=1,
                         draft={"category": "family", "score": 85, "title": "回家",
                                "description": "d", "people": ["妈妈"]},
                         recognition_status="ok"))
    with db.session_scope(url) as s, caplog.at_level(logging.INFO, logger="camdigest"):
        run_merge("2026-09-29", s, settings)
    assert any("S7 2026-09-29：" in r.message for r in caplog.records)

    shutil.copy(video_file, tmp_path / "gate.mp4")
    pool = [ClipPlan(event_id=1, media_path=str(tmp_path / "gate.mp4"),
                     start_ts=t0, start_s=0.0, end_s=2.0, score=85, camera_id="gate")]
    with db.session_scope(url) as s, caplog.at_level(logging.INFO, logger="camdigest"):
        export_highlights("2026-09-29", pool, settings, session=s)
    assert any("S10 导出 tier=5" in r.message for r in caplog.records)



def test_s1_s3_log_lines(caplog, tmp_path, monkeypatch, video_file):
    """Phase 0 计划契约：S1 新增文件行与 S3 门控行真实输出（终审应修 5）。"""
    import shutil as _sh

    from camdigest import db
    from camdigest.config import CameraCfg, Settings
    from camdigest.pipeline import s3_person
    from camdigest.pipeline.s1_index import index_camera

    setup_logging(tmp_path)
    settings = Settings(
        cameras=[CameraCfg(id="gate", name="大门", device="gate", lens="single",
                           dir=tmp_path / "footage")],
        models={"recognition": {"base_url": "http://x", "model": "m", "api_key": "k"},
                "report": {"base_url": "http://x", "model": "m", "api_key": "k"}},
        storage={"data_dir": str(tmp_path)})
    cam_dir = tmp_path / "footage"
    cam_dir.mkdir()
    _sh.copy(video_file, cam_dir / "20260929100000.mp4")
    url = f"sqlite:///{tmp_path}/t.db"
    db.init_db(url)

    with db.session_scope(url) as s, caplog.at_level(logging.INFO, logger="camdigest"):
        assert index_camera(settings.cameras[0], s) == 1
    assert any("gate 新增 1 个文件" in r.message for r in caplog.records)

    class FakeSampler:
        def __init__(self, weights="yolo11n.pt"):
            pass

        def sample(self, path, times):
            return [(t, None) for t in times]

        def infer(self, frames):
            return [1] * len(frames)

    monkeypatch.setattr(s3_person, "YoloPersonSampler", FakeSampler)
    with db.session_scope(url) as s:
        m = s.query(db.MediaFile).one()
        s.add(db.Segment(media_file_id=m.id, start_s=0, end_s=6))
    with db.session_scope(url) as s, caplog.at_level(logging.INFO, logger="camdigest"):
        s3_person.run_person("2026-09-29", s, settings)
    assert any("S3 2026-09-29" in r.message for r in caplog.records)
