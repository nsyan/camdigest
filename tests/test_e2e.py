# tests/test_e2e.py —— M1 全链路：模型全部假体，ffmpeg 真跑（不标 heavy）

from pathlib import Path

import numpy as np
import pytest

from camdigest import db
from camdigest.config import Settings
from camdigest.llm.contracts import EventDraft, SegmentHint
from camdigest.pipeline.orchestrator import run_day
from camdigest.pipeline.s4_face import FaceDet
from tests.conftest import make_video

FIVE_SECTIONS = ["## 今日总览", "## 时间线", "## 人物出没", "## 异常事件", "## 精华清单"]


def _settings(tmp_path):
    return Settings(
        cameras=[
            {"id": "gate", "name": "大门", "device": "gate", "lens": "single",
             "dir": str(tmp_path / "gate")},
            {"id": "yard", "name": "院子", "device": "yard", "lens": "single",
             "dir": str(tmp_path / "yard")},
        ],
        models={"recognition": {"base_url": "http://x", "model": "m", "api_key": "k"},
                "report": {"base_url": "http://x", "model": "m", "api_key": "k"}},
        prefilter={"whisper": {"enabled": False}},
        publish={"feishu": {"app_id": "a", "app_secret": "s", "chat_id": "c"}},
        storage={"data_dir": str(tmp_path / "data")})


def _seed_footage(tmp_path):
    """2 机位 × 2 文件微型视频（12s > min_segment_seconds=8，无 scene 锚 → 整文件一段）。"""
    for cam, hours in (("gate", (10, 15)), ("yard", (11, 16))):
        d = tmp_path / cam
        d.mkdir(parents=True)
        for h in hours:
            make_video(d / f"20260929{h}0000.mp4", seconds=12)


def _install_fakes(monkeypatch, settings):
    """S3/S4/S6/S8/S9 的模型假体（注入各阶段自身的符号绑定）。"""
    from camdigest.pipeline import s3_person, s4_face, s6_recognize, s8_report
    from camdigest.publish import feishu

    # S3：恒 1 人
    class FakeSampler:
        def __init__(self, weights="yolo11n.pt"):
            pass

        def sample(self, path, times):
            return [(t, None) for t in times]

        def infer(self, frames):
            return [1] * len(frames)

    monkeypatch.setattr(s3_person, "YoloPersonSampler", FakeSampler)

    # S4：固定向量 512 维（与建档向量一致 → 命中"妈妈"）
    vec = np.zeros(512, dtype=np.float32)
    vec[0] = 1.0

    class FakeFaceClient:
        def __init__(self, base_url="http://fake", timeout=30.0):
            pass

        def extract(self, image_bytes):
            return [FaceDet(bbox=[0, 0, 1, 1], det_score=0.9, norm=vec.tolist())]

    monkeypatch.setattr(s4_face, "FaceClient", FakeFaceClient)

    # 建档：/faces/妈妈/fake.jpg（extract 已被假体接管，jpg 内容不解析）
    faces_dir = settings.storage.data_dir / "faces"
    (faces_dir / "妈妈").mkdir(parents=True, exist_ok=True)
    (faces_dir / "妈妈" / "fake.jpg").write_bytes(b"\xff\xd8fake")
    n = s4_face.enroll_faces(faces_dir, "http://fake", client=FakeFaceClient())
    assert n == 1

    # S6：固定 draft（family/85 → hybrid 85）
    class FakeRecognition:
        def __init__(self, cfg):
            self.last_usage = {"total_tokens": 0}

        def analyze(self, video_path, hint: SegmentHint) -> EventDraft:
            return EventDraft(category="family", people=["妈妈"], score=85,
                              title="妈妈带小宝回家", description="从大门进入")

    monkeypatch.setattr(s6_recognize, "OpenAIRecognition", FakeRecognition)

    # S8：固定五板块日报
    class FakeReport:
        def __init__(self, cfg):
            pass

        def write(self, date, payload):
            return "\n".join(FIVE_SECTIONS) + "\n"

    monkeypatch.setattr(s8_report, "build_report_model", lambda s: FakeReport(s))

    # S9：假体复刻真实契约——回填 reports 行后返回 URL
    from camdigest.db import Report

    def fake_publish(date, session, settings, client=None):
        row = session.query(Report).filter(Report.date == date).one()
        row.feishu_doc_url = "https://feishu.example/doc/e2e"
        row.feishu_msg_id = "omsg-e2e"
        session.merge(row)
        return row.feishu_doc_url

    monkeypatch.setattr(feishu, "publish_feishu", fake_publish)


def test_run_day_end_to_end(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    _seed_footage(tmp_path)
    _install_fakes(monkeypatch, settings)

    results = run_day("2026-09-29", settings)

    from camdigest.pipeline.orchestrator import STAGES

    url = f"sqlite:///{settings.storage.data_dir}/camdigest.db"
    # 1. jobs 全 done
    with db.session_scope(url) as s:
        jobs = {j.stage: j.status for j in s.query(db.Job).all()}
    assert set(jobs) == {name for name, _ in STAGES} and set(jobs.values()) == {"done"}
    assert set(results) == set(jobs)

    # 2. events ≥ 1
    with db.session_scope(url) as s:
        events = s.query(db.Event).filter(db.Event.date == "2026-09-29").all()
        assert len(events) >= 1
        report = s.query(db.Report).one()
        md_path = report.md_path
        doc_url = report.feishu_doc_url
    assert doc_url == "https://feishu.example/doc/e2e"

    # 3. 日报 md 含五板块标题
    md = Path(md_path).read_text(encoding="utf-8")
    for h in FIVE_SECTIONS:
        assert h in md

    # 4. 四档精华文件存在且 ffprobe 时长 > 0
    from camdigest.media import probe
    hl_dir = settings.storage.data_dir / "highlights" / "2026-09-29"
    for tier in (5, 10, 30, 60):
        p = hl_dir / f"精华_{tier}min.mp4"
        assert p.exists(), f"missing {p}"
        assert probe(p).duration > 0
    with db.session_scope(url) as s:
        assert s.query(db.Highlight).count() == 4


def test_run_day_resume_after_failure(tmp_path, video_file, monkeypatch):
    """断点：首跑在 s4 失败 → 重跑跳过 s1-s3，从 s4 继续。"""
    settings = _settings(tmp_path)
    _seed_footage(tmp_path)
    _install_fakes(monkeypatch, settings)
    from camdigest.pipeline import s4_face

    # 注入点选 FaceClient 构造（run_face 逐帧容错只吞单帧异常——最终评审阻断项修复后的预期行为）
    class DownSidecar:
        def __init__(self, *a, **kw):
            raise RuntimeError("sidecar down")

    monkeypatch.setattr(s4_face, "FaceClient", DownSidecar)
    with pytest.raises(RuntimeError, match="sidecar down"):
        run_day("2026-09-29", settings)

    # 修复：恢复假 client（与 _install_fakes 同款）后重跑
    vec = np.zeros(512, dtype=np.float32)
    vec[0] = 1.0

    class RecoveredClient:
        def __init__(self, base_url="http://fake", timeout=30.0):
            pass

        def extract(self, image_bytes):
            return [FaceDet(bbox=[0, 0, 1, 1], det_score=0.9, norm=vec.tolist())]

    monkeypatch.setattr(s4_face, "FaceClient", RecoveredClient)
    results = run_day("2026-09-29", settings)
    url = f"sqlite:///{settings.storage.data_dir}/camdigest.db"
    with db.session_scope(url) as s:
        jobs = {j.stage: j.status for j in s.query(db.Job).all()}
    assert set(jobs.values()) == {"done"}
    assert "s4_face" in results   # s4 在重跑中执行
    # s1/s2/s3 不在 results（done 跳过）
    assert "s1_index" not in results and "s2_prefilter" not in results
