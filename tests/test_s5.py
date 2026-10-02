# tests/test_s5.py —— S5 音频：silencedetect 用真 fixture；whisper mock/惰性
import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ClassVar

import pytest

from camdigest.config import (
    CameraCfg,
    ModelCfg,
    ModelsCfg,
    PrefilterCfg,
    Settings,
    WhisperCfg,
)
from camdigest.pipeline.query import segments_for_date


# —— 纯函数测试辅助：未入库实例（pick_transcribe_sources 只读属性）——
def _cams_by_id():
    cams = [
        CameraCfg(id="gate", name="大门", device="gate", dir=Path("/data/gate")),
        CameraCfg(id="yard_fixed", name="后院定焦", device="yard", lens="fixed",
                  dir=Path("/data/yard")),
        CameraCfg(id="yard_ptz", name="后院云台", device="yard", lens="ptz",
                  dir=Path("/data/yard")),
    ]
    return {c.id: c for c in cams}


def _media(mid, camera_id, dur=60.0):
    from camdigest.db import MediaFile
    start = datetime(2026, 9, 29, 10, 0, tzinfo=UTC)
    return MediaFile(id=mid, camera_id=camera_id, path=f"/{camera_id}.mp4",
                     start_ts=start, end_ts=start + timedelta(seconds=dur), duration=dur)


def test_sine_is_active(video_file):
    from camdigest.pipeline.s5_audio import audio_active_regions
    regions = audio_active_regions(Path(video_file))
    assert regions and regions[0][0] <= 0.5   # 440Hz 正弦全程有声
    total = sum(e - s for s, e in regions)
    assert total >= 5.0


def test_silent_video_no_regions(silent_video_file):
    from camdigest.pipeline.s5_audio import audio_active_regions
    assert audio_active_regions(Path(silent_video_file)) == []


def test_pick_transcribe_sources_fixed_with_fallback():
    """CAL-2：source=fixed 时各设备只留定焦路；无定焦路的设备（single）兜底取自身。"""
    from camdigest.pipeline.s5_audio import pick_transcribe_sources
    cams = _cams_by_id()
    medias = [_media(1, "yard_fixed"), _media(2, "yard_ptz"), _media(3, "gate")]
    assert pick_transcribe_sources(medias, cams, "fixed") == {1, 3}


def test_pick_transcribe_sources_all_keeps_everything():
    from camdigest.pipeline.s5_audio import pick_transcribe_sources
    cams = _cams_by_id()
    medias = [_media(1, "yard_fixed"), _media(2, "yard_ptz"), _media(3, "gate")]
    assert pick_transcribe_sources(medias, cams, "all") == {1, 2, 3}


def test_transcriber_uses_clip_timestamps_and_joins(monkeypatch):
    """cv extra 惰性：向 sys.modules 注入假 faster_whisper 验证转写胶水逻辑。"""
    from camdigest.pipeline.s5_audio import Transcriber
    seen = {}

    class FakeModel:
        def __init__(self, name, compute_type=None):
            seen["model"] = (name, compute_type)

        def transcribe(self, path, language=None, clip_timestamps=None, vad_filter=None):
            seen["call"] = (path, language, clip_timestamps, vad_filter)
            return iter([types.SimpleNamespace(text=" 你好 "),
                         types.SimpleNamespace(text="世界")]), None

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    tr = Transcriber("tiny")
    out = tr.transcribe(Path("/x.mp4"), 1.0, 3.5)
    assert out == "你好 世界"
    assert seen["model"] == ("tiny", "int8")
    assert seen["call"] == ("/x.mp4", "zh", "1.00,3.50", True)


@pytest.mark.heavy
def test_transcriber_smoke_sine(video_file):
    """重测：真实 faster-whisper 冒烟（截 2s 正弦，不崩溃、返回字符串）。

    延后验证：需安装 cv extra（faster-whisper）+ 下载模型，本环境未装，默认跳过。
    """
    from camdigest.pipeline.s5_audio import Transcriber
    tr = Transcriber()
    text = tr.transcribe(Path(video_file), 0.0, 2.0)
    assert isinstance(text, str)


# —— run_audio：真 DB + 真 ffmpeg（whisper 关闭或 mock）——
def _settings(whisper: WhisperCfg):
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
        prefilter=PrefilterCfg(whisper=whisper),
    )


def _seed_db(url, specs):
    """specs: [(camera_id, path, [(start_s, end_s, draft), ...]), ...]，按顺序入库。"""
    from camdigest import db
    db.init_db(url)
    with db.session_scope(url) as s:
        for cid, path, segs in specs:
            start = datetime(2026, 9, 29, 10, 0, tzinfo=UTC)
            m = db.MediaFile(camera_id=cid, path=path, start_ts=start,
                             end_ts=start + timedelta(seconds=6), duration=6.0)
            s.add(m)
            s.flush()
            for row in segs:
                start_s, end_s = row[0], row[1]
                draft = row[2] if len(row) > 2 else None
                s.add(db.Segment(media_file_id=m.id, start_s=start_s, end_s=end_s,
                                 draft=draft))


class FakeTranscriber:
    calls: ClassVar[list[str]] = []

    def __init__(self, model_name):
        FakeTranscriber.calls.append(model_name)

    def transcribe(self, path, start_s, end_s):
        return f"voice@{start_s:.1f}-{end_s:.1f}"


def test_run_audio_marks_active_and_copies_donor(tmp_path, video_file, silent_video_file):
    """无声路源（single 兜底）判静默；定焦路全程有声；云台路（非源）按时间重叠复制 donor。"""
    from camdigest import db
    from camdigest.pipeline.s5_audio import run_audio
    url = f"sqlite:///{tmp_path}/t.db"
    _seed_db(url, [
        ("gate", silent_video_file, [(0.0, 8.0)]),
        ("yard_fixed", video_file, [(0.0, 8.0)]),
        ("yard_ptz", str(tmp_path / "ptz.mp4"), [(0.0, 8.0)]),  # 非源路不读文件，占位即可
    ])
    settings = _settings(WhisperCfg(enabled=False, dual_lens_source="fixed"))
    with db.session_scope(url) as s:
        n = run_audio("2026-09-29", s, settings)
        rows = {m.camera_id: seg for seg, m in segments_for_date("2026-09-29", s)}
    assert n == 3
    assert rows["gate"].audio_active is False        # 无音轨 → 无有声段
    assert rows["yard_fixed"].audio_active is True   # 440Hz 全程有声
    assert rows["yard_ptz"].audio_active is True     # donor 复制自同设备 fixed 路
    assert rows["yard_ptz"].transcript is None


def test_run_audio_transcribes_sources_and_shares_transcript(tmp_path, video_file,
                                                             monkeypatch):
    from camdigest import db
    from camdigest.pipeline import s5_audio
    from camdigest.pipeline.s5_audio import run_audio
    url = f"sqlite:///{tmp_path}/t.db"
    _seed_db(url, [
        ("yard_fixed", video_file, [(0.0, 8.0)]),
        ("yard_ptz", str(tmp_path / "ptz.mp4"), [(0.0, 8.0)]),
    ])
    FakeTranscriber.calls = []
    monkeypatch.setattr(s5_audio, "Transcriber", FakeTranscriber)
    settings = _settings(WhisperCfg(enabled=True, dual_lens_source="fixed"))
    with db.session_scope(url) as s:
        n = run_audio("2026-09-29", s, settings)
        rows = {m.camera_id: seg for seg, m in segments_for_date("2026-09-29", s)}
    assert n == 2
    assert FakeTranscriber.calls == ["small-int8"]
    assert rows["yard_fixed"].transcript == "voice@0.0-8.0"
    assert rows["yard_ptz"].transcript == "voice@0.0-8.0"   # donor 复制 transcript


def test_run_audio_selected_only_gates_transcription(tmp_path, video_file, monkeypatch):
    """selected_only：仅 draft.score>=60 的片段转写（S6 后执行时 draft 已落）。"""
    from camdigest import db
    from camdigest.pipeline import s5_audio
    from camdigest.pipeline.s5_audio import run_audio
    url = f"sqlite:///{tmp_path}/t.db"
    _seed_db(url, [("yard_fixed", video_file, [
        (0.0, 2.0, {"score": 70}),
        (2.0, 4.0, {"score": 40}),
        (4.0, 6.0),                                   # draft 未落（S5 先于 S6 的默认态）
    ])])
    FakeTranscriber.calls = []
    monkeypatch.setattr(s5_audio, "Transcriber", FakeTranscriber)
    settings = _settings(WhisperCfg(enabled=True, selected_only=True,
                                    dual_lens_source="fixed"))
    with db.session_scope(url) as s:
        n = run_audio("2026-09-29", s, settings)
        segs = sorted((seg for seg, _ in segments_for_date("2026-09-29", s)),
                      key=lambda x: x.start_s)
    assert n == 3
    assert all(seg.audio_active for seg in segs)      # 三段均与有声区重叠
    assert [seg.transcript for seg in segs] == ["voice@0.0-2.0", None, None]
