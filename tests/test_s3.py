# tests/test_s3.py
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from camdigest.pipeline.s3_person import PersonResult, count_people

if TYPE_CHECKING:
    import numpy as np


class FakeSampler:
    """非 heavy 替身：sample 不读文件，infer 返回预置计数（zip 截断语义与真实现一致）。"""

    def __init__(self, counts=None, count_for=None):
        self.counts = counts          # infer 直接返回该列表（可被 zip 截断）
        self.count_for = count_for    # callable(path) -> int：按文件给出单一计数
        self.infer_calls = 0

    def sample(self, path: Path, times: list[float]) -> list[tuple[float, "np.ndarray"]]:
        self._path = path
        return [(t, None) for t in times]

    def infer(self, frames) -> list[int]:
        self.infer_calls += 1
        if self.count_for is not None:
            return [self.count_for(self._path)]
        return self.counts if self.counts is not None else [0]


def test_count_people_max(video_file):
    r = count_people(Path(video_file), 0, 6,
                     sampler=FakeSampler(None), infer=lambda frames: [0, 2, 1])
    assert isinstance(r, PersonResult)
    assert r.max_count == 2


def test_sampler_times():
    from camdigest.pipeline.s3_person import sample_times
    assert sample_times(0, 10) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


# —— run_person：计数写回 + 云台路门控 + 幂等（真 sampler 由 heavy 用例覆盖）——

_DAY = datetime(2026, 9, 29, 10, 0, tzinfo=UTC)


def _settings():
    from camdigest.config import CameraCfg, ModelCfg, ModelsCfg, Settings
    return Settings(
        cameras=[
            CameraCfg(id="gate", name="大门", device="gate", dir=Path("/data/gate")),
            CameraCfg(id="yard_fixed", name="后院定焦", device="yard", lens="fixed",
                      dir=Path("/data/yard")),
            CameraCfg(id="yard_ptz", name="后院云台", device="yard", lens="ptz",
                      dir=Path("/data/yard")),
        ],
        models=ModelsCfg(
            recognition=ModelCfg(base_url="http://x", model="m", api_key="k"),
            report=ModelCfg(base_url="http://x", model="m", api_key="k"),
        ),
    )


def _seed_db(tmp_path):
    """四段候选：gate/fixed/ptz 待计数（-1），ptz_b 已计数（person_count=2）。"""
    from camdigest import db
    url = f"sqlite:///{tmp_path}/s3.db"
    db.init_db(url)
    with db.session_scope(url) as s:
        for cam in _settings().cameras:
            s.merge(db.Camera(id=cam.id, name=cam.name, device=cam.device,
                              lens=cam.lens, dir=str(cam.dir)))
        for path, cam in [("/gate.mp4", "gate"), ("/yard_fixed.mp4", "yard_fixed"),
                          ("/yard_ptz.mp4", "yard_ptz"), ("/yard_ptz_b.mp4", "yard_ptz")]:
            mf = db.MediaFile(camera_id=cam, path=path, start_ts=_DAY,
                              end_ts=_DAY + timedelta(seconds=60), duration=60.0)
            s.add(mf)
            s.flush()
            s.add(db.Segment(media_file_id=mf.id, start_s=0.0, end_s=8.0,
                             person_count=2 if path.endswith("_b.mp4") else -1))
    return url


def _install_fake(monkeypatch):
    """替换 s3_person.YoloPersonSampler（惰性导入，不触 ultralytics）；返回每次构造的实例。"""
    from camdigest.pipeline import s3_person
    made = []

    def factory():
        fake = FakeSampler(count_for=lambda p: 2 if "gate" in str(p) else 0)
        made.append(fake)
        return fake

    monkeypatch.setattr(s3_person, "YoloPersonSampler", factory)
    return made


def test_run_person_counts_and_ptz_gate(tmp_path, monkeypatch):
    """计数写回；云台路门控：ptz 计 0 删除、fixed/single 计 0 保留、ptz>0 保留。"""
    from camdigest import db
    from camdigest.pipeline.query import segments_for_date
    from camdigest.pipeline.s3_person import run_person

    url = _seed_db(tmp_path)
    made = _install_fake(monkeypatch)
    with db.session_scope(url) as s:
        assert run_person("2026-09-29", s, _settings()) == 3  # 3 段待计数
    with db.session_scope(url) as s:  # 新会话验证持久化
        rows = {(m.path, seg.start_s): seg.person_count
                for seg, m in segments_for_date("2026-09-29", s)}
        assert rows[("/gate.mp4", 0.0)] == 2           # single 计数写回
        assert rows[("/yard_fixed.mp4", 0.0)] == 0     # 非 ptz 0 人保留
        assert ("/yard_ptz.mp4", 0.0) not in rows      # 云台路 0 人候选删除
        assert rows[("/yard_ptz_b.mp4", 0.0)] == 2     # 云台路 >0 保留
    assert made[0].infer_calls == 3                    # 已计数段未被推断


def test_run_person_idempotent(tmp_path, monkeypatch):
    """重跑：person_count>=0 的段跳过推断，计数与段集合不变。"""
    from camdigest import db
    from camdigest.pipeline.query import segments_for_date
    from camdigest.pipeline.s3_person import run_person

    url = _seed_db(tmp_path)
    made = _install_fake(monkeypatch)
    with db.session_scope(url) as s:
        assert run_person("2026-09-29", s, _settings()) == 3
    with db.session_scope(url) as s:
        assert run_person("2026-09-29", s, _settings()) == 0
    assert made[1].infer_calls == 0                    # 第二次零推断
    with db.session_scope(url) as s:
        rows = {(m.path, seg.start_s): seg.person_count
                for seg, m in segments_for_date("2026-09-29", s)}
        assert rows[("/gate.mp4", 0.0)] == 2
        assert rows[("/yard_ptz_b.mp4", 0.0)] == 2
        assert ("/yard_ptz.mp4", 0.0) not in rows      # 删除不复活


@pytest.mark.heavy
def test_real_yolo_counts_testsrc(video_file):
    from camdigest.pipeline.s3_person import YoloPersonSampler
    s = YoloPersonSampler()
    r = count_people(Path(video_file), 0, 3, sampler=s, infer=s.infer)
    assert r.max_count >= 0
