# CamDigest M1（CLI 全链路）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 M1 —— `camdigest run --date` 一键跑完 S1-S10 流水线，产出四档精华视频 + 飞书日报；支持人脸建档与历史补跑。

**Architecture:** 纯 CLI 应用（FastAPI 留到 M2）。每阶段一个 `pipeline/sN_*.py` 模块，靠 SQLite 表 + `jobs` 断点表串起来；LLM 走 `llm/` 双协议适配层；飞书发布独立 `publish/feishu.py`。测试全部跑在 ffmpeg 生成的微型 fixture 视频上，重模型/真实网络用例标记 `heavy` 默认跳过。

**Tech Stack:** Python 3.12 · SQLAlchemy 2.0 · pydantic v2 · httpx · APScheduler · FFmpeg/ffprobe 子进程 · ultralytics(YOLO11n) · faster-whisper · pytest

**Spec:** `docs/architecture.md`（术语见 `CONTEXT.md`；决策见 `docs/adr/`）。计划与 spec 冲突时以 spec 为准并回来改计划。

## Global Constraints

- Python `>=3.12`；src 布局，包名 `camdigest`
- 依赖分档：核心依赖不含 CV 重库；`ultralytics`/`faster-whisper`/`numpy` 放 `[cv]` extra；CI 无模型也能跑 `pytest`
- `ffmpeg`/`ffprobe` 为硬前置（本机已有）；所有视频操作走 subprocess，不引入 moviepy
- 测试默认 `addopts = -m "not heavy"`；`heavy` = 需下载模型或真实凭据的用例
- 术语与命名严格跟随 `CONTEXT.md`：机位/候选片段/事件/段池/精华/档位
- 模块命名固定：`s1_index.py` `s2_prefilter.py` `s3_person.py` `s4_face.py` `s5_audio.py` `s6_recognize.py` `s7_merge.py` `s8_report.py` `s9_publish.py` `s10_clip.py`（选段纯函数在 `s10_selection.py`）
- 每阶段幂等：重跑只处理未完成数据；`jobs(date, stage)` 记录状态
- 提交风格：`feat:`/`test:`/`chore:` + 中文一句话（沿用仓库 `docs:` 惯例）

## 前置校准点（真机确认后只改 config.yaml，不改代码）

- **CAL-1** 双摄转存目录形态：本计划假设双摄两路**分目录**（`living/` 固定路、`living_pt/` 云台路）。真机若为同目录可区分文件名，改 `cameras[].dir + pattern` 即可
- **CAL-2** 双机位音轨：假设固定路带音轨、云台路复用（`whisper.dual_lens_source: fixed`）。真机 ffprobe 确认后改配置

## File Structure（全图）

```
src/camdigest/
├── __init__.py
├── cli.py              # argparse: run / enroll-faces / backfill
├── config.py           # Settings + ${ENV} 展开 + cameras 校验
├── db.py               # SQLAlchemy 模型 + init_db + session
├── scheduler.py        # APScheduler 日批入口
├── media.py            # ffprobe/ffmpeg 公共封装（probe、切片段、抽帧）
├── pipeline/
│   ├── __init__.py
│   ├── orchestrator.py # STAGES 表 + run_day + jobs 断点
│   ├── s1_index.py … s10_clip.py
│   └── workers.py      # S3-S5 进程池并行执行器
├── llm/
│   ├── __init__.py
│   ├── contracts.py    # SegmentHint/EventDraft/两个 Protocol
│   ├── openai_adapter.py   # 识别+报告（DashScope 兼容协议）
│   └── anthropic_adapter.py# 仅报告
└── publish/
    ├── __init__.py
    └── feishu.py       # token→docx→图→blocks→卡片
tests/
├── conftest.py         # ffmpeg 生成微型视频 fixture + 内存库
├── test_config.py … test_s10_selection.py（每任务对应）
docker/Dockerfile
docker-compose.yml      # 对齐 architecture.md §3
```

**对 spec §5 的一处细化**：`segments` 表增加 `draft JSON NULL` 列存放 S6 识别草稿（断点续跑需要 S6 结果在 S7 之前落库；spec 的表是草图级别，此列不违背其结构）。

---

# Phase 1 —— 骨架 + S1 索引

## Task 1: 项目脚手架

**Files:**
- Create: `pyproject.toml`、`src/camdigest/__init__.py`、`tests/__init__.py`、`tests/conftest.py`（先放空 fixture，Task 4 充实）、`.gitignore`（追加）
- Test: `tests/test_smoke.py`

**Interfaces:**
- Produces: 可安装包 `camdigest`；`tests/conftest.py::video_file`（Task 4 完整实现，本任务先占位返回 None）

- [ ] **Step 1: 写 pyproject.toml**

```toml
[project]
name = "camdigest"
version = "0.1.0"
description = "3+2 机位家庭监控录像 → 每日精华 + 飞书日报"
requires-python = ">=3.12"
dependencies = [
  "pydantic>=2.7",
  "sqlalchemy>=2.0.30",
  "httpx>=0.27",
  "apscheduler>=3.10,<4",
  "pyyaml>=6.0",
]

[project.optional-dependencies]
cv = ["ultralytics>=8.2", "faster-whisper>=1.0", "numpy>=1.26"]
dev = ["pytest>=8", "ruff>=0.4"]

[project.scripts]
camdigest = "camdigest.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/camdigest"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not heavy'"
markers = ["heavy: 需要下载模型或真实凭据，默认跳过"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]
```

- [ ] **Step 2: 建包目录与空文件**

```bash
mkdir -p src/camdigest/pipeline src/camdigest/llm src/camdigest/publish tests
touch src/camdigest/__init__.py src/camdigest/pipeline/__init__.py \
      src/camdigest/llm/__init__.py src/camdigest/publish/__init__.py tests/__init__.py
```

- [ ] **Step 3: 写冒烟测试**

```python
# tests/test_smoke.py
def test_import():
    import camdigest  # noqa: F401
```

- [ ] **Step 4: 安装并验证**

Run: `pip install -e ".[dev]" && pytest -v`
Expected: `test_import` PASS（not heavy 默认收集）

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src tests
git commit -m "chore: 项目脚手架——src布局、依赖分档、pytest基线"
```

## Task 2: 配置层 config.py

**Files:**
- Create: `src/camdigest/config.py`
- Test: `tests/test_config.py`、`tests/fixtures/config.example.yaml`

**Interfaces:**
- Produces: `Settings`、`CameraCfg`、`Settings.load(path: Path) -> Settings`；后续所有任务从 `settings.cameras / settings.prefilter / settings.anomaly / settings.highlight / settings.models` 取值
- 字段名与 `docs/architecture.md §9` 逐一对应

- [ ] **Step 1: 写失败测试**

```python
# tests/test_config.py
import textwrap
from pathlib import Path

import pytest

from camdigest.config import Settings


def _yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_load_minimal(tmp_path):
    p = _yaml(tmp_path, """\
        schedule: { daily_at: "03:00", timezone: Asia/Shanghai }
        cameras:
          - { id: gate, name: 大门, device: gate, lens: single, dir: /media/nvr/gate }
        models:
          recognition: { protocol: openai, base_url: http://x/v1, model: m1, api_key: k1 }
          report: { protocol: openai, base_url: http://x/v1, model: m2, api_key: k2 }
        storage: { data_dir: /data }
        """)
    s = Settings.load(p)
    assert s.cameras[0].id == "gate"
    assert s.highlight.tiers == [5, 10, 30, 60]      # 架构 §9 默认值
    assert s.prefilter.scene_threshold == 0.03
    assert s.anomaly.stranger_while_family_absent.enabled is True


def test_env_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-xyz")
    p = _yaml(tmp_path, """\
        schedule: { daily_at: "03:00", timezone: Asia/Shanghai }
        cameras:
          - { id: gate, name: 大门, device: gate, dir: /media/nvr/gate }
        models:
          recognition: { protocol: openai, base_url: http://x/v1, model: m1, api_key: ${TEST_KEY} }
          report: { protocol: anthropic, base_url: http://y, model: m2, api_key: ${TEST_KEY} }
        storage: { data_dir: /data }
        """)
    s = Settings.load(p)
    assert s.models.recognition.api_key == "sk-xyz"
    assert s.models.report.protocol == "anthropic"


def test_dual_lens_validation(tmp_path):
    # 双摄设备必须 fixed+ptz 成对；ptz 无覆盖时自动给 0.08
    p = _yaml(tmp_path, """\
        schedule: { daily_at: "03:00", timezone: Asia/Shanghai }
        cameras:
          - { id: living,    name: 客厅固定, device: living, lens: fixed, dir: /a }
          - { id: living_pt, name: 客厅云台, device: living, lens: ptz,   dir: /b }
        models:
          recognition: { protocol: openai, base_url: http://x/v1, model: m1, api_key: k1 }
          report: { protocol: openai, base_url: http://x/v1, model: m2, api_key: k2 }
        storage: { data_dir: /data }
        """)
    s = Settings.load(p)
    assert s.cameras[1].effective_scene_threshold() == 0.08


def test_broken_pair_rejected(tmp_path):
    p = _yaml(tmp_path, """\
        schedule: { daily_at: "03:00", timezone: Asia/Shanghai }
        cameras:
          - { id: living, name: 客厅固定, device: living, lens: fixed, dir: /a }
        models:
          recognition: { protocol: openai, base_url: http://x/v1, model: m1, api_key: k1 }
          report: { protocol: openai, base_url: http://x/v1, model: m2, api_key: k2 }
        storage: { data_dir: /data }
        """)
    with pytest.raises(ValueError, match="fixed.*ptz"):
        Settings.load(p)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_config.py -v`
Expected: FAIL `ModuleNotFoundError: camdigest.config`

- [ ] **Step 3: 实现 config.py**

```python
# src/camdigest/config.py
"""配置装载：yaml + ${ENV} 展开 + 机位校验。字段对应 docs/architecture.md §9。"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

_ENV_RE = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
PTZ_DEFAULT_SCENE_THRESHOLD = 0.08  # 云台路默认阈值（架构 §4 云台防误报）


def _expand_env(value: str) -> str:
    return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)


def _deep_expand(node: object) -> object:
    if isinstance(node, str):
        return _expand_env(node)
    if isinstance(node, dict):
        return {k: _deep_expand(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_deep_expand(v) for v in node]
    return node


class PrefilterOverride(BaseModel):
    scene_threshold: float | None = None


class CameraCfg(BaseModel):
    id: str
    name: str
    device: str
    lens: Literal["single", "fixed", "ptz"] = "single"
    dir: Path
    pattern: str = "*.mp4"
    timezone: str = "Asia/Shanghai"
    prefilter: PrefilterOverride | None = None

    def effective_scene_threshold(self, base: float = 0.03) -> float:
        if self.prefilter and self.prefilter.scene_threshold is not None:
            return self.prefilter.scene_threshold
        if self.lens == "ptz":
            return PTZ_DEFAULT_SCENE_THRESHOLD
        return base


class WhisperCfg(BaseModel):
    enabled: bool = True
    model: str = "small-int8"
    selected_only: bool = False
    dual_lens_source: Literal["fixed", "ptz", "all"] = "fixed"


class PrefilterCfg(BaseModel):
    scene_threshold: float = 0.03
    min_segment_seconds: float = 8.0
    max_segment_seconds: float = 60.0
    pad_seconds: float = 2.0
    whisper: WhisperCfg = WhisperCfg()


class AnomalyRuleCfg(BaseModel):
    enabled: bool = True
    family_absent_minutes: int = 30


class AnomalyCfg(BaseModel):
    stranger_while_family_absent: AnomalyRuleCfg = AnomalyRuleCfg()


class ModelCfg(BaseModel):
    protocol: Literal["openai", "anthropic"] = "openai"
    base_url: str
    model: str
    api_key: str
    timeout_seconds: float = 120.0


class ModelsCfg(BaseModel):
    recognition: ModelCfg
    report: ModelCfg

    @model_validator(mode="after")
    def _recognition_openai_only(self):  # ADR-0001 协议边界
        if self.recognition.protocol != "openai":
            raise ValueError("识别角色仅支持 openai 协议（ADR-0001）")
        return self


class HighlightCfg(BaseModel):
    tiers: list[int] = [5, 10, 30, 60]
    max_segment_seconds: float = 20.0
    min_per_hour: int = 1
    max_per_hour: int = 3
    event_gap_minutes: int = 30
    per_camera: bool = False


class FeishuCfg(BaseModel):
    app_id: str = ""
    app_secret: str = ""
    folder_token: str = ""
    chat_id: str = ""
    image_max_bytes: int = 200 * 1024


class PublishCfg(BaseModel):
    feishu: FeishuCfg = FeishuCfg()


class StorageCfg(BaseModel):
    data_dir: Path = Path("/data")
    retention: Literal["permanent"] = "permanent"


class ScheduleCfg(BaseModel):
    daily_at: str = "03:00"
    timezone: str = "Asia/Shanghai"


class FacesCfg(BaseModel):
    rest_url: str = "http://insightface:18080"
    threshold: float = 0.4
    registry_dir: Path = Path("/data/faces")


class Settings(BaseModel):
    schedule: ScheduleCfg = ScheduleCfg()
    cameras: list[CameraCfg]
    models: ModelsCfg
    prefilter: PrefilterCfg = PrefilterCfg()
    anomaly: AnomalyCfg = AnomalyCfg()
    faces: FacesCfg = FacesCfg()
    highlight: HighlightCfg = HighlightCfg()
    publish: PublishCfg = PublishCfg()
    storage: StorageCfg = StorageCfg()

    @model_validator(mode="after")
    def _check_cameras(self):
        ids = [c.id for c in self.cameras]
        if len(ids) != len(set(ids)):
            raise ValueError(f"机位 id 重复: {ids}")
        by_device: dict[str, list[CameraCfg]] = {}
        for c in self.cameras:
            by_device.setdefault(c.device, []).append(c)
        for device, cams in by_device.items():
            lenses = sorted(c.lens for c in cams if c.lens != "single")
            if cams and cams[0].lens == "single":
                if len(cams) > 1:
                    raise ValueError(f"设备 {device}: single 不能与其他机位共存")
            elif lenses not in ([], ["fixed", "ptz"]):
                raise ValueError(f"设备 {device}: 双摄必须 fixed+ptz 成对，当前 {lenses}")
        return self

    @classmethod
    def load(cls, path: Path) -> "Settings":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(_deep_expand(raw))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/camdigest/config.py tests/test_config.py
git commit -m "feat: 配置层——yaml+env展开、双摄机位校验、云台默认阈值"
```

## Task 3: 数据库层 db.py

**Files:**
- Create: `src/camdigest/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Produces: ORM 模型 `Camera/MediaFile/Segment/Event/Highlight/Report/Job/Identity`；`init_db(url: str) -> None`；`session_scope(url)` 上下文管理器。列名与 spec §5 一致，另有 `segments.draft`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_db.py
from datetime import datetime, timezone

from camdigest import db


def _ts() -> datetime:
    return datetime(2026, 9, 29, 10, 0, tzinfo=timezone.utc)


def test_tables_and_roundtrip(tmp_path):
    url = f"sqlite:///{tmp_path}/t.db"
    db.init_db(url)
    with db.session_scope(url) as s:
        cam = db.Camera(id="gate", name="大门", device="gate", lens="single",
                        dir="/media/nvr/gate", pattern="*.mp4", timezone="Asia/Shanghai")
        s.add(cam)
        mf = db.MediaFile(camera_id="gate", path="/x/a.mp4",
                          start_ts=_ts(), end_ts=_ts(), duration=60.0)
        s.add(mf)
        s.flush()
        s.add(db.Segment(media_file_id=mf.id, start_s=1.0, end_s=9.0, motion_score=0.3,
                         person_count=1, face_labels=[{"identity": "妈妈", "conf": 0.8, "ts": 2.0}],
                         audio_active=True, recognition_status="pending",
                         draft={"category": "family", "score": 80}))
    with db.session_scope(url) as s:
        seg = s.query(db.Segment).one()
        assert seg.draft["category"] == "family"
        assert seg.face_labels[0]["identity"] == "妈妈"
        job = db.Job(date="2026-09-29", stage="s1_index", status="done")
        s.add(job)
    with db.session_scope(url) as s:  # 唯一约束生效
        import pytest
        from sqlalchemy.exc import IntegrityError
        s.add(db.Job(date="2026-09-29", stage="s1_index", status="done"))
        with pytest.raises(IntegrityError):
            s.commit()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_db.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: 实现 db.py**

```python
# src/camdigest/db.py
"""SQLAlchemy 2.0 模型。表结构 = docs/architecture.md §5（segments 增加 draft 列）。"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Integer, String, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Camera(Base):
    __tablename__ = "cameras"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    device: Mapped[str] = mapped_column(String)
    lens: Mapped[str] = mapped_column(String, default="single")
    dir: Mapped[str] = mapped_column(String)
    pattern: Mapped[str] = mapped_column(String, default="*.mp4")
    timezone: Mapped[str] = mapped_column(String, default="Asia/Shanghai")


class MediaFile(Base):
    __tablename__ = "media_files"
    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"))
    path: Mapped[str] = mapped_column(String, unique=True)
    start_ts: Mapped[datetime]
    end_ts: Mapped[datetime]
    duration: Mapped[float]


class Segment(Base):
    __tablename__ = "segments"
    id: Mapped[int] = mapped_column(primary_key=True)
    media_file_id: Mapped[int] = mapped_column(ForeignKey("media_files.id"))
    start_s: Mapped[float]
    end_s: Mapped[float]
    motion_score: Mapped[float] = 0.0
    person_count: Mapped[int] = 0
    face_labels: Mapped[list | None] = mapped_column(JSON, default=list)  # [{identity,conf,ts}]
    audio_active: Mapped[bool] = False
    transcript: Mapped[str | None] = mapped_column(String)
    recognition_status: Mapped[str] = mapped_column(String, default="pending")  # pending|ok|failed
    draft: Mapped[dict | None] = mapped_column(JSON)  # S6 EventDraft 快照
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[str] = mapped_column(String, index=True)
    start_ts: Mapped[datetime]
    end_ts: Mapped[datetime]
    category: Mapped[str] = mapped_column(String)
    score: Mapped[int] = mapped_column(Integer, default=0)
    title: Mapped[str] = mapped_column(String, default="")
    description: Mapped[str] = mapped_column(String, default="")
    camera_ids: Mapped[list] = mapped_column(JSON, default=list)
    segment_ids: Mapped[list] = mapped_column(JSON, default=list)
    keyframe_path: Mapped[str | None] = mapped_column(String)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class Highlight(Base):
    __tablename__ = "highlights"
    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[str] = mapped_column(String, index=True)
    camera_id: Mapped[str | None] = mapped_column(String)  # NULL=跨机位合并版
    tier_minutes: Mapped[int]
    file_path: Mapped[str] = mapped_column(String)
    duration: Mapped[float]
    bytes: Mapped[int]
    event_ids: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class Report(Base):
    __tablename__ = "reports"
    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[str] = mapped_column(String, unique=True)
    md_path: Mapped[str]
    feishu_doc_url: Mapped[str | None] = mapped_column(String)
    feishu_msg_id: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (UniqueConstraint("date", "stage", name="uq_jobs_date_stage"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[str] = mapped_column(String)
    stage: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="running")  # running|done|failed
    error: Mapped[str | None] = mapped_column(String)
    started_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTimeNullable())


def DateTimeNullable():
    from sqlalchemy import DateTime
    return DateTime()


class Identity(Base):
    __tablename__ = "identities"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    dir: Mapped[str] = mapped_column(String)
    unknown_cluster: Mapped[str | None] = mapped_column(String)
    enrolled_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


_engines: dict[str, object] = {}


def init_db(url: str) -> None:
    eng = create_engine(url, json_serializer=_json_dumps)
    Base.metadata.create_all(eng)
    _engines[url] = eng


def session_scope(url: str):
    """用法: with session_scope(url) as s: ...  成功即 commit，异常即 rollback。"""
    from contextlib import contextmanager

    if url not in _engines:
        init_db(url)
    factory = sessionmaker(bind=_engines[url], expire_on_commit=False)

    @contextmanager
    def _scope():
        with factory.begin() as s:
            yield s

    return _scope()


def _json_dumps(obj, **kw):
    import json
    return json.dumps(obj, ensure_ascii=False, **kw)
```

（实现时把 `DateTimeNullable()` 这个丑陋占位改成顶部直接 `from sqlalchemy import DateTime`，`finished_at: Mapped[datetime | None] = mapped_column(DateTime)`——写进文件时即用干净版本。）

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_db.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/camdigest/db.py tests/test_db.py
git commit -m "feat: 数据层——10张表ORM、jobs唯一约束、draft列"
```

## Task 4: media 公共封装 + S1 索引

**Files:**
- Create: `src/camdigest/media.py`、`src/camdigest/pipeline/s1_index.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_media.py`、`tests/test_s1.py`

**Interfaces:**
- Produces: `probe(path) -> MediaMeta(path,start_ts,duration,size_bytes)`；`parse_start_ts(name) -> datetime|None`；`index_camera(camera: CameraCfg, session) -> int`（返回新增文件数，幂等）
- `conftest.video_file(tmp_path_factory)`：ffmpeg 生成 6 秒带音轨测试视频（后续所有阶段测试复用）

- [ ] **Step 1: conftest 写视频 fixture（真实 ffmpeg，不用 mock）**

```python
# tests/conftest.py
import subprocess

import pytest


def make_video(path, seconds=6, with_audio=True) -> str:
    cmd = ["ffmpeg", "-y", "-f", "lavfi",
           "-i", f"testsrc=duration={seconds}:size=320x240:rate=10"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if with_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd.append(str(path))
    subprocess.run(cmd, check=True, capture_output=True)
    return str(path)


@pytest.fixture(scope="session")
def video_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("media") / "20260929100000.mp4"
    return make_video(p)


@pytest.fixture(scope="session")
def silent_video_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("media") / "20260929110000.mp4"
    return make_video(p, with_audio=False)
```

- [ ] **Step 2: 写失败测试**

```python
# tests/test_media.py
from datetime import datetime

from camdigest.media import parse_start_ts, probe


def test_probe(video_file):
    meta = probe(__import__("pathlib").Path(video_file))
    assert 5.5 < meta.duration < 6.5
    assert meta.size_bytes > 0


def test_parse_start_ts_mijia_layout():
    # 米家 SD 卡常见命名：14 位时间戳（CAL-1 真机校准点，解析策略可配置化）
    from pathlib import Path
    ts = parse_start_ts(Path("/nvr/gate/20260929100000.mp4"))
    assert ts == datetime(2026, 9, 29, 10, 0, 0)
```

```python
# tests/test_s1.py
from camdigest.config import CameraCfg
from camdigest.pipeline.s1_index import index_camera


def test_index_idempotent(tmp_path, video_file):
    cam_dir = tmp_path / "gate"
    cam_dir.mkdir()
    (cam_dir / "20260929100000.mp4").write_bytes(__import__("pathlib").Path(video_file).read_bytes())
    cam = CameraCfg(id="gate", name="大门", device="gate", dir=cam_dir, pattern="20260929*.mp4")
    from camdigest import db
    db.init_db("sqlite:///:memory:")
    with db.session_scope("sqlite:///:memory:") as s:
        assert index_camera(cam, s) == 1
    with db.session_scope("sqlite:///:memory:") as s:
        assert index_camera(cam, s) == 0            # 幂等：已入库跳过
        assert s.query(db.MediaFile).count() == 1
        mf = s.query(db.MediaFile).one()
        assert mf.start_ts.hour == 10               # 文件名解析成功
```

注意：内存库多连接会各开新库，Task 实现时把测试改为文件库 `tmp_path/t.db`（写计划时已知，执行者照做，勿用 `:memory:`）。

- [ ] **Step 3: 跑测试确认失败**

Run: `pytest tests/test_media.py tests/test_s1.py -v`
Expected: FAIL `ModuleNotFoundError: camdigest.media`

- [ ] **Step 4: 实现 media.py 与 s1_index.py**

```python
# src/camdigest/media.py
"""ffprobe/ffmpeg 子进程公共封装。所有视频操作集中在此。"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_TS_PATTERNS = [  # 优先文件名（米家转存按时间命名），失败退回 mtime
    re.compile(r"(20\d{12})"),            # 20260929100000
    re.compile(r"(20\d{6})[_\- ]?(\d{6})"),  # 20260929_100000
]


@dataclass
class MediaMeta:
    path: Path
    start_ts: datetime
    duration: float
    size_bytes: int


def parse_start_ts(path: Path) -> datetime | None:
    for pat in _TS_PATTERNS:
        m = pat.search(path.name)
        if m:
            digits = "".join(m.groups())
            return datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
    return None


def probe(path: Path, tz=None) -> MediaMeta:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", str(path)],
        check=True, capture_output=True, text=True).stdout
    fmt = json.loads(out)["format"]
    duration = float(fmt["duration"])
    ts = parse_start_ts(path) or datetime.fromtimestamp(path.stat().st_mtime)
    return MediaMeta(path=path, start_ts=ts, duration=duration, size_bytes=int(fmt["size"]))
```

```python
# src/camdigest/pipeline/s1_index.py
"""S1 索引：扫描机位目录 → ffprobe 元数据入库（spec §4）。幂等：按 path 去重。"""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy.orm import Session

from camdigest.config import CameraCfg
from camdigest.db import Camera, MediaFile
from camdigest.media import probe


def index_camera(camera: CameraCfg, session: Session) -> int:
    session.merge(Camera(id=camera.id, name=camera.name, device=camera.device,
                         lens=camera.lens, dir=str(camera.dir), pattern=camera.pattern,
                         timezone=camera.timezone))
    known = {m.path for m in session.query(MediaFile.path).filter(
        MediaFile.camera_id == camera.id)}
    added = 0
    for f in sorted(camera.dir.glob(camera.pattern)):
        p = str(f.resolve())
        if p in known:
            continue
        meta = probe(f)
        session.add(MediaFile(camera_id=camera.id, path=p, start_ts=meta.start_ts,
                              end_ts=meta.start_ts + timedelta(seconds=meta.duration),
                              duration=meta.duration))
        added += 1
    return added
```

（实现时测试用文件库：`url = f"sqlite:///{tmp_path}/t.db"`，两处 session_scope 用同一 url。）

- [ ] **Step 5: 跑测试确认通过**

Run: `pytest tests/test_media.py tests/test_s1.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/camdigest/media.py src/camdigest/pipeline/s1_index.py tests/conftest.py tests/test_media.py tests/test_s1.py
git commit -m "feat: media封装+S1索引——ffprobe入库、时间戳解析、幂等扫描"
```

---

# Phase 2 —— 本地检测层（S2-S5）

## Task 5: S2 预筛 scene 检测

**Files:**
- Create: `src/camdigest/pipeline/s2_prefilter.py`
- Test: `tests/test_s2.py`

**Interfaces:**
- Consumes: `MediaFile`、`CameraCfg.effective_scene_threshold()`
- Produces: `detect_scenes(path: Path, threshold: float) -> list[float]`（场景突变时刻，秒）；`build_segments(media: MediaFile, times: list[float], cfg: PrefilterCfg) -> list[SegmentDraft]`，`SegmentDraft(start_s, end_s, motion_score)`；`run_prefilter(date: str, session, settings) -> int`（新增 segments 数）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_s2.py
from datetime import datetime, timezone

from camdigest.config import PrefilterCfg
from camdigest.pipeline.s2_prefilter import build_segments, detect_scenes


def test_detect_scenes_on_fixture(video_file):
    from pathlib import Path
    times = detect_scenes(Path(video_file), threshold=0.03)
    assert isinstance(times, list)  # testsrc 无突变可能为空，不误报即可


def test_build_segments_merges_and_pads():
    from camdigest.db import MediaFile
    mf = MediaFile(camera_id="gate", path="/x.mp4",
                   start_ts=datetime(2026, 9, 29, 10, 0, tzinfo=timezone.utc),
                   end_ts=datetime(2026, 9, 29, 10, 1, tzinfo=timezone.utc), duration=60.0)
    segs = build_segments(mf, times=[10.0, 12.0, 40.0], cfg=PrefilterCfg())
    assert segs, "至少产出一段"
    for s in segs:
        assert s.end_s - s.start_s >= 8.0 - 4 * 1e-9      # min_segment（含 pad 后）
        assert s.motion_score > 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_s2.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: 实现**

```python
# src/camdigest/pipeline/s2_prefilter.py
"""S2 预筛：FFmpeg scene 检测 → 候选片段（spec §4）。

云台路防误报：阈值由 CameraCfg.effective_scene_threshold() 提供（默认 0.08），
且云台路候选片段还需 S3 人形计数>0 才最终入库（Task 6 落实）。
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from camdigest.config import PrefilterCfg, Settings
from camdigest.db import MediaFile, Segment

_PTS_RE = re.compile(r"pts_time:([\d.]+)")


def detect_scenes(path: Path, threshold: float) -> list[float]:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-an", "-vf",
         f"select='gt(scene\\,{threshold})',showinfo", "-f", "null", "-"],
        check=True, capture_output=True, text=True).stderr
    return sorted(float(t) for t in _PTS_RE.findall(out))


@dataclass
class SegmentDraft:
    start_s: float
    end_s: float
    motion_score: float


def build_segments(media: MediaFile, times: list[float], cfg: PrefilterCfg) -> list[SegmentDraft]:
    """以 scene 突变点为锚，切 [t-pad, next_gap] 区间；无突变则整文件一段。"""
    anchors = [t for t in times if 0 <= t <= media.duration]
    bounds = [0.0]
    for i, t in enumerate(anchors):
        nxt = anchors[i + 1] if i + 1 < len(anchors) else media.duration
        bounds.append((t + nxt) / 2)  # 中点切分
    bounds.append(media.duration)
    out: list[SegmentDraft] = []
    for i in range(len(bounds) - 1):
        start = max(0.0, bounds[i] - cfg.pad_seconds)
        end = min(media.duration, bounds[i + 1] + cfg.pad_seconds)
        if end - start < cfg.min_segment_seconds:
            continue
        while end - start > cfg.max_segment_seconds:  # 超长均分为多段
            mid = start + cfg.max_segment_seconds
            n = len([a for a in anchors if start <= a < mid])
            out.append(SegmentDraft(start, mid, motion_score=n / (mid - start)))
            start = mid
        n = len([a for a in anchors if start <= a < end])
        out.append(SegmentDraft(start, end, motion_score=n / (end - start)))
    return out


def run_prefilter(date: str, session: Session, settings: Settings) -> int:
    added = 0
    cams = {c.id: c for c in settings.cameras}
    for media in session.query(MediaFile).filter(MediaFile.path.like("%/%")):
        if media.start_ts.strftime("%Y-%m-%d") != date:
            continue
        if session.query(Segment).filter(Segment.media_file_id == media.id).count():
            continue  # 幂等
        cam = cams.get(media.camera_id)
        thr = cam.effective_scene_threshold(settings.prefilter.scene_threshold)
        for d in build_segments(media, detect_scenes(Path(media.path), thr), settings.prefilter):
            session.add(Segment(media_file_id=media.id, start_s=d.start_s, end_s=d.end_s,
                                motion_score=d.motion_score))
            added += 1
    return added
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_s2.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/camdigest/pipeline/s2_prefilter.py tests/test_s2.py
git commit -m "feat: S2预筛——scene检测、中点切分、云台路独立阈值接入"
```

## Task 6: S3 人形计数（YOLO11n）

**Files:**
- Create: `src/camdigest/pipeline/s3_person.py`
- Test: `tests/test_s3.py`（单测 mock YOLO；真模型用例标 heavy）

**Interfaces:**
- Produces: `count_people(video_path: Path, start_s: float, end_s: float, sampler) -> PersonResult`，`PersonResult(max_count: int, frames: list[tuple[float, int]])`；`run_person(date, session, settings) -> int`；`YoloPersonSampler`（cv extra 内，惰性加载 ultralytics）
- 本任务同时落实**云台路门控**：`run_person` 后，`lens=ptz` 机位上 `person_count==0` 的候选片段直接删除（不进段池），对应架构 §4「云台机位防误报」

- [ ] **Step 1: 写失败测试**

```python
# tests/test_s3.py
from pathlib import Path

from camdigest.pipeline.s3_person import PersonResult, count_people


class FakeSampler:
    def __init__(self, counts): self.counts = counts

    def sample(self, path: Path, times: list[float]) -> list[tuple[float, "np.ndarray"]]:
        return [(t, None) for t in times]


def test_count_people_max(video_file):
    r = count_people(Path(video_file), 0, 6,
                     sampler=FakeSampler(None), infer=lambda frames: [0, 2, 1])
    assert isinstance(r, PersonResult)
    assert r.max_count == 2


def test_sampler_times():
    from camdigest.pipeline.s3_person import sample_times
    assert sample_times(0, 10) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_s3.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: 实现**

```python
# src/camdigest/pipeline/s3_person.py
"""S3 人形：区间内 1fps 采样帧 → YOLO11n 人形计数（spec §4）。

模型惰性加载（cv extra），单测注入 sampler/infer 假实现。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from camdigest.config import Settings
from camdigest.db import MediaFile, Segment


def sample_times(start_s: float, end_s: float, fps: float = 1.0) -> list[float]:
    n = int((end_s - start_s) * fps) + 1
    return [round(start_s + i / fps, 3) for i in range(n)]


@dataclass
class PersonResult:
    max_count: int
    frames: list[tuple[float, int]] = field(default_factory=list)


def count_people(video_path: Path, start_s: float, end_s: float,
                 sampler, infer) -> PersonResult:
    times = sample_times(start_s, end_s)
    frames = sampler.sample(video_path, times)
    counts = infer(frames)                       # [(count_at_t)] 与 frames 对齐
    pairs = [(t, c) for (t, _), c in zip(frames, counts)]
    return PersonResult(max_count=max((c for _, c in pairs), default=0), frames=pairs)


class YoloPersonSampler:
    """真实现：ultralytics YOLO11n，person=class 0，conf>0.35。heavy。"""

    def __init__(self, weights: str = "yolo11n.pt"):
        from ultralytics import YOLO  # cv extra
        self.model = YOLO(weights)

    def sample(self, path: Path, times: list[float]):
        import cv2
        cap = cv2.VideoCapture(str(path))
        out = []
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if ok:
                out.append((t, frame))
        cap.release()
        return out

    def infer(self, frames) -> list[int]:
        out = []
        for _, frame in frames:
            res = self.model.predict(frame, verbose=False, classes=[0], conf=0.35)
            out.append(len(res[0].boxes) if res else 0)
        return out


def run_person(date: str, session: Session, settings: Settings) -> int:
    sampler = YoloPersonSampler()
    cams = {c.id: c for c in settings.cameras}
    updated = 0
    for seg, media in (session.query(Segment, MediaFile)
                       .join(MediaFile, Segment.media_file_id == MediaFile.id)
                       .filter(Segment.person_count == 0).all()):
        if media.start_ts.strftime("%Y-%m-%d") != date or seg.motion_score == 0 and seg.person_count == 0 and False:
            continue
        r = count_people(Path(media.path), seg.start_s, seg.end_s,
                         sampler=sampler, infer=sampler.infer)
        seg.person_count = r.max_count
        updated += 1
    # 云台路门控：ptz 机位 0 人片段删除（spec §4 云台防误报）
    ptz_cams = {c.id for c in settings.cameras if c.lens == "ptz"}
    for seg, media in (session.query(Segment, MediaFile)
                       .join(MediaFile, Segment.media_file_id == MediaFile.id).all()):
        if media.camera_id in ptz_cams and seg.person_count == 0:
            session.delete(seg)
    return updated
```

（`run_person` 中那行防御性判断写成干净版本：只按日期过滤 `person_count` 已为默认 0 的段——即"未处理"。执行者直接写 `if media.start_ts.strftime(...) != date: continue`，删除那串 `and False` 败笔。）

- [ ] **Step 4: 跑测试确认通过 + heavy 用例**

```python
# tests/test_s3.py 追加
@pytest.mark.heavy
def test_real_yolo_counts_testsrc(video_file):
    s = YoloPersonSampler()
    r = count_people(Path(video_file), 0, 3, sampler=s, infer=s.infer)
    assert r.max_count >= 0
```

Run: `pytest tests/test_s3.py -v && pytest tests/test_s3.py -v -m heavy`（后者本机跑通即可）
Expected: 非 heavy 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/camdigest/pipeline/s3_person.py tests/test_s3.py
git commit -m "feat: S3人形——1fps采样YOLO计数、云台路0人门控"
```

## Task 7: S4 人脸标注（InsightFace-REST）

**Files:**
- Create: `src/camdigest/pipeline/s4_face.py`
- Test: `tests/test_s4.py`（httpx MockTransport mock REST；enroll 用随机向量）

**Interfaces:**
- Produces: `FaceClient(base_url)`：`extract(image_bytes) -> list[FaceDet(bbox, det_score, norm: list[float])]`；`load_registry(data_dir) -> dict[str, list[vec]]`（`/faces/{身份名}/*.jpg` → 向量，REST 批量 embed）；`label_faces(dets, registry, threshold) -> list[dict]`（`{identity, conf, ts}`，未命中为 `未知-NN` 聚类暂用 `未知`）；`run_face(date, session, settings) -> int`；CLI 建档命令 `enroll_faces(data_dir, rest_url)`
- 关键帧抽取复用 Task 4 的 ffmpeg 单帧命令（在 `media.py::grab_frame(path, t, out)`，本任务补上并测试）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_s4.py
import numpy as np

from camdigest.pipeline.s4_face import FaceDet, label_faces


def test_label_faces_matches_registry():
    base = np.zeros(512, dtype=np.float32); base[0] = 1.0
    other = np.zeros(512, dtype=np.float32); other[1] = 1.0
    registry = {"妈妈": [base], "爸爸": [other]}
    dets = [FaceDet(bbox=[1, 2, 3, 4], det_score=0.9, norm=base),
            FaceDet(bbox=[5, 6, 7, 8], det_score=0.8, norm=other * 0.5 + base * 0.5)]
    labels = label_faces(dets, registry, threshold=0.4, ts=12.0)
    assert labels[0] == {"identity": "妈妈", "conf": pytest.approx(1.0, abs=1e-3), "ts": 12.0}
    assert labels[1]["identity"] in {"妈妈", "爸爸"}   # 0.707 边界，两可都接受


def test_label_faces_unknown():
    stranger = np.zeros(512, dtype=np.float32); stranger[2] = 1.0
    base = np.zeros(512, dtype=np.float32); base[0] = 1.0
    labels = label_faces([FaceDet(bbox=[0, 0, 1, 1], det_score=0.9, norm=stranger)],
                         {"妈妈": [base]}, threshold=0.4, ts=1.0)
    assert labels[0]["identity"] == "未知"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_s4.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: 实现（REST 走 httpx，可 mock）**

```python
# src/camdigest/pipeline/s4_face.py
"""S4 人脸：有人帧 → InsightFace-REST → 身份标签（ADR-0003，spec §4）。

registry 自管：/data/faces/{身份名}/*.jpg 建档时批量 embed 存 .npz。
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
from sqlalchemy.orm import Session

from camdigest.config import Settings
from camdigest.db import MediaFile, Segment


@dataclass
class FaceDet:
    bbox: list[float]
    det_score: float
    norm: list[float]


class FaceClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def extract(self, image_bytes: bytes) -> list[FaceDet]:
        r = self._client.post("/extract", json={
            "data": base64.b64encode(image_bytes).decode(), "_threshold": 0.6})
        r.raise_for_status()
        data = r.json().get("data", [])
        return [FaceDet(bbox=d["bbox"], det_score=d["det_score"], norm=d["norm"])
                for d in data]


def load_registry(data_dir: Path, client: FaceClient) -> dict[str, list[np.ndarray]]:
    """enroll_faces 落盘的 .npz 优先；无则现场扫目录 embed（首次）。"""
    npz = data_dir / "registry.npz"
    if npz.exists():
        z = np.load(npz)
        return {k: [z[k]] for k in z.files}
    registry: dict[str, list[np.ndarray]] = {}
    for id_dir in sorted(data_dir.iterdir()):
        if not id_dir.is_dir():
            continue
        for img in sorted(id_dir.glob("*.jpg")):
            dets = client.extract(img.read_bytes())
            if dets:
                registry.setdefault(id_dir.name, []).append(np.asarray(dets[0].norm, dtype=np.float32))
    return registry


def label_faces(dets: list[FaceDet], registry: dict[str, list[np.ndarray]],
                threshold: float, ts: float) -> list[dict]:
    out = []
    for d in dets:
        v = np.asarray(d.norm, dtype=np.float32)
        best_name, best_cos = "未知", threshold
        for name, vecs in registry.items():
            cos = float(np.max([float(np.dot(v, u) / (np.linalg.norm(v) * np.linalg.norm(u) + 1e-9))
                                for u in vecs])) if vecs else 0.0
            if cos >= best_cos:
                best_name, best_cos = name, cos
        out.append({"identity": best_name, "conf": round(best_cos, 4), "ts": ts})
    return out


def run_face(date: str, session: Session, settings: Settings) -> int:
    """对 person_count>0 的候选片段，在人脸密度最高的 3 个采样帧抽帧打标。"""
    import cv2
    client = FaceClient(settings.faces.rest_url)
    registry = load_registry(settings.storage.data_dir / "faces", client)
    cams = {c.id: c for c in settings.cameras}
    updated = 0
    for seg, media in (session.query(Segment, MediaFile)
                       .join(MediaFile, Segment.media_file_id == MediaFile.id)
                       .filter(Segment.person_count > 0).all()):
        if media.start_ts.strftime("%Y-%m-%d") != date or seg.face_labels:
            continue
        cap = cv2.VideoCapture(media.path)
        times = [seg.start_s + (seg.end_s - seg.start_s) * k / 4 for k in range(1, 4)]
        labels = []
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            ok2, buf = cv2.imencode(".jpg", frame)
            dets = client.extract(buf.tobytes()) if ok2 else []
            labels += label_faces(dets, registry, settings.faces.threshold, ts=round(t, 3))
        cap.release()
        seg.face_labels = labels
        updated += 1
    return updated


def enroll_faces(data_dir: Path, rest_url: str, session=None) -> int:
    """CLI enroll-faces：扫 /faces/{身份名}/*.jpg → registry.npz + identities 行。"""
    client = FaceClient(rest_url)
    registry = load_registry(data_dir, client)
    for name in registry:
        if session is not None:  # 同步登记 identities 表（spec §5）
            session.merge(Identity(name=name, dir=str(data_dir / name)))
    # …npz 落盘同前
    client = FaceClient(rest_url)
    registry = load_registry(data_dir, client)
    arrs, names = [], []
    for name, vecs in registry.items():
        for v in vecs:
            arrs.append(v); names.append(name)
    if not arrs:
        return 0
    np.savez(data_dir / "registry.npz", **{f"{n}_{i}": v for i, (n, v) in enumerate(zip(names, arrs))})
    return len(arrs)
```

注意：`load_registry` 的 npz 键含身份名，执行时保证 `label_faces` 读回的结构是 `{身份: [vec,...]}`（把 `f"{n}_{i}"` 还原成按前缀分组；或直接存一个 JSON 索引 + npz 两个文件，取实现更直白者，但接口签名不变）。

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_s4.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/camdigest/pipeline/s4_face.py tests/test_s4.py
git commit -m "feat: S4人脸——REST抽特征、registry余弦匹配、未知兜底"
```

## Task 8: S5 音频（silencedetect + faster-whisper）

**Files:**
- Create: `src/camdigest/pipeline/s5_audio.py`
- Test: `tests/test_s5.py`（silencedetect 用真 fixture；whisper mock）

**Interfaces:**
- Produces: `audio_active_regions(path) -> list[tuple[float, float]]`；`Transcriber`（cv extra 惰性）`transcribe(path, start_s, end_s) -> str`；`run_audio(date, session, settings) -> int`
- **CAL-2 落点**：`run_audio` 按 `device` 分组，`dual_lens_source != "all"` 时只转写该 lens 的 media，另一路 segment 按**时间重叠**复制 transcript；`selected_only=true` 时 S5 只做 `audio_active` 标记不转写（转写延后人工触发，架构 §4 兜底路径）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_s5.py
from pathlib import Path

from camdigest.pipeline.s5_audio import audio_active_regions


def test_sine_is_active(video_file):
    regions = audio_active_regions(Path(video_file))
    assert regions and regions[0][0] <= 0.5   # 440Hz 正弦全程有声
    total = sum(e - s for s, e in regions)
    assert total >= 5.0


def test_silent_video_no_regions(silent_video_file):
    assert audio_active_regions(Path(silent_video_file)) == []
```

（再加一个 `dual_lens_source` 复制逻辑的纯函数测试：`pick_transcribe_sources(medias, source="fixed") -> set[media_id]`——同 device 取 fixed 路文件、无固定路取自身。）

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_s5.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: 实现**

```python
# src/camdigest/pipeline/s5_audio.py
"""S5 音频：silencedetect 标记有声段 → faster-whisper 转写（spec §4）。

同设备双机位仅转固定路（CAL-2），云台路复用（dual_lens_source）。
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from sqlalchemy.orm import Session

from camdigest.config import Settings
from camdigest.db import MediaFile, Segment

_SIL_RE = re.compile(r"silence_start: ([\d.]+)")
_SIL_END_RE = re.compile(r"silence_end: ([\d.]+)")


def audio_active_regions(path: Path, noise: float = -35.0) -> list[tuple[float, float]]:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af",
         f"silencedetect=noise={noise}dB:d=0.5", "-f", "null", "-"],
        check=True, capture_output=True, text=True).stderr
    dur = float(re.search(r"Duration: (\d+):(\d+):([\d.]+)", out).group(0).replace("Duration:", "").strip().split(":")[0])  # 简化：直接再 probe
    from camdigest.media import probe
    total = probe(path).duration
    starts = [float(t) for t in _SIL_RE.findall(out)]
    ends = [float(t) for t in _SIL_END_RE.findall(out)]
    regions, cur = [], 0.0
    for s, e in zip(starts, ends):
        if s - cur > 0.5:
            regions.append((cur, s))
        cur = max(cur, e)
    if total - cur > 0.5:
        regions.append((cur, total))
    return regions


class Transcriber:
    def __init__(self, model_name: str = "small-int8"):
        from faster_whisper import WhisperModel  # cv extra
        self.model = WhisperModel(model_name, compute_type="int8")

    def transcribe(self, path: Path, start_s: float, end_s: float) -> str:
        segs, _ = self.model.transcribe(str(path), language="zh",
                                        clip_timestamps=f"{start_s:.2f},{end_s:.2f}",
                                        vad_filter=True)
        return " ".join(s.text.strip() for s in segs).strip()


def pick_transcribe_sources(medias: list[MediaFile], cams_by_id: dict, source: str) -> set[int]:
    if source == "all":
        return {m.id for m in medias}
    keep: set[int] = set()
    by_device: dict[str, list[MediaFile]] = {}
    for m in medias:
        by_device.setdefault(cams_by_id[m.camera_id].device, []).append(m)
    for device, group in by_device.items():
        wanted = [m for m in group if cams_by_id[m.camera_id].lens == source]
        keep.update(m.id for m in (wanted or group))   # 无目标 lens 则全转
    return keep


def run_audio(date: str, session: Session, settings: Settings) -> int:
    medias = [m for m in session.query(MediaFile).all()
              if m.start_ts.strftime("%Y-%m-%d") == date]
    cams = {c.id: c for c in settings.cameras}
    wcfg = settings.prefilter.whisper
    sources = pick_transcribe_sources(medias, cams, wcfg.dual_lens_source)
    src_by_id = {m.id: m for m in medias if m.id in sources}
    tr = Transcriber(wcfg.model) if wcfg.enabled else None
    updated = 0
    for seg, media in (session.query(Segment, MediaFile)
                       .join(MediaFile, Segment.media_file_id == MediaFile.id).all()):
        if media.start_ts.strftime("%Y-%m-%d") != date:
            continue
        if media.id in sources:
            regions = audio_active_regions(Path(media.path))
            seg.audio_active = any(s < seg.end_s and e > seg.start_s for s, e in regions)
            if tr and seg.audio_active:
                seg.transcript = tr.transcribe(Path(media.path), seg.start_s, seg.end_s) or seg.transcript
        else:
            # 双机位另一路：按时间重叠复用同设备源路结果
            donor = next((m for m in src_by_id.values()
                          if cams[m.camera_id].device == cams[media.camera_id].device
                          and m.start_ts <= media.end_ts and m.end_ts >= media.start_ts), None)
            if donor:
                for dseg in session.query(Segment).filter(Segment.media_file_id == donor.id):
                    if dseg.start_s < seg.end_s and dseg.end_s > seg.start_s:
                        seg.audio_active = seg.audio_active or dseg.audio_active
                        seg.transcript = seg.transcript or dseg.transcript
        updated += 1
    return updated
```

（`audio_active_regions` 里那行坏掉的 Duration 解析删掉，直接用 `probe(path).duration`——执行者清理。）

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_s5.py -v`
Expected: 2~3 passed

- [ ] **Step 5: Commit**

```bash
git add src/camdigest/pipeline/s5_audio.py tests/test_s5.py
git commit -m "feat: S5音频——有声段检测、whisper转写、双机位源路复用"
```

---

# Phase 3 —— API 层（S6）

## Task 9: LLM 契约 contracts.py

**Files:**
- Create: `src/camdigest/llm/contracts.py`
- Test: `tests/test_contracts.py`

**Interfaces:**
- Produces: `FaceHint`、`SegmentHint(camera, lens, face_labels, transcript)`、`EventDraft`（pydantic，含 `from_json`）、`RecognitionModel` / `ReportModel` 两个 `Protocol`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_contracts.py
import pytest

from camdigest.llm.contracts import EventDraft


def test_draft_from_json():
    d = EventDraft.from_json("""{"category": "family", "people": ["妈妈"],
        "score": 85, "title": "妈妈带小宝回家", "description": "..."}""")
    assert d.category == "family" and d.score == 85


def test_draft_rejects_bad_category():
    with pytest.raises(ValueError):
        EventDraft.from_json('{"category": "ufo", "people": [], "score": 1, "title": "t", "description": "d"}')


def test_draft_strips_code_fence():
    d = EventDraft.from_json('```json\n{"category":"empty","people":[],"score":10,"title":"t","description":"d"}\n```')
    assert d.category == "empty"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_contracts.py -v` → FAIL `ModuleNotFoundError`

- [ ] **Step 3: 实现**

```python
# src/camdigest/llm/contracts.py
"""模型双角色契约（ADR-0001）。识别=全模态仅 openai 协议；报告=文本可选协议。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, field_validator

CATEGORIES = ("family", "stranger", "visitor", "animal", "vehicle", "empty")


@dataclass
class FaceHint:
    identity: str
    conf: float
    ts: float


@dataclass
class SegmentHint:
    camera: str            # 机位显示名
    lens: str              # single|fixed|ptz
    face_labels: list[FaceHint] = field(default_factory=list)
    transcript: str | None = None


class EventDraft(BaseModel):
    category: Literal["family", "stranger", "visitor", "animal", "vehicle", "empty"]
    people: list[str] = []
    score: int = 0
    title: str = ""
    description: str = ""

    @field_validator("score")
    @classmethod
    def _clamp(cls, v: int) -> int:
        return max(0, min(100, v))

    @classmethod
    def from_json(cls, text: str) -> "EventDraft":
        cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
        return cls.model_validate(json.loads(cleaned))


@runtime_checkable
class RecognitionModel(Protocol):
    def analyze(self, video_path: Path, hint: SegmentHint) -> EventDraft: ...


@runtime_checkable
class ReportModel(Protocol):
    def write(self, date: str, events: list[dict]) -> str: ...
```

- [ ] **Step 4: 跑测试确认通过** → `pytest tests/test_contracts.py -v`，3 passed
- [ ] **Step 5: Commit**

```bash
git add src/camdigest/llm/contracts.py tests/test_contracts.py
git commit -m "feat: LLM契约——SegmentHint/EventDraft/双角色Protocol"
```

## Task 10: OpenAI 适配器（识别 + 报告）

**Files:**
- Create: `src/camdigest/llm/openai_adapter.py`
- Modify: `src/camdigest/media.py`（追加 `extract_audio(path, out_wav)`、`cut_clip(path, start_s, end_s, out)`）
- Test: `tests/test_openai_adapter.py`（httpx `MockTransport` 假服务端）

**Interfaces:**
- Consumes: `ModelCfg`、Task 9 契约
- Produces: `OpenAIRecognition(cfg: ModelCfg).analyze(video_path, hint) -> EventDraft`；`OpenAIReport(cfg: ModelCfg).write(date, events) -> str`；`media.cut_clip / extract_audio`
- 行为：视频 base64 进 `video_url`（data URI）、抽音频 16k mono wav 进 `input_audio`（DashScope 百炼兼容格式）；JSON 解析失败**重试 2 次**，每次把校验错误附回 messages；仍失败抛 `RecognitionError`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_openai_adapter.py
import json

import httpx
import pytest

from camdigest.config import ModelCfg
from camdigest.llm.contracts import RecognitionError, SegmentHint
from camdigest.llm.openai_adapter import OpenAIRecognition, OpenAIReport

CFG = ModelCfg(base_url="http://fake/v1", model="m", api_key="k")

GOOD = {"choices": [{"message": {"content":
        '{"category":"family","people":["妈妈"],"score":85,"title":"t","description":"d"}'}}]}


def _client(handler) -> OpenAIRecognition:
    transport = httpx.MockTransport(handler)
    r = OpenAIRecognition(CFG)
    r._http = httpx.Client(transport=transport)   # 测试注入口
    return r


def test_analyze_ok(video_file, tmp_path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=GOOD)

    draft = _client(handler).analyze(__import__("pathlib").Path(video_file),
                                     SegmentHint(camera="大厅", lens="single"))
    assert draft.category == "family"
    body = calls[0]
    parts = body["messages"][1]["content"]
    kinds = {p.get("type") for p in parts}
    assert "video_url" in kinds and "input_audio" in kinds and "text" in kinds
    assert body["response_format"] == {"type": "json_object"}


def test_analyze_retries_on_bad_json(video_file):
    n = {"n": 0}

    def handler(request):
        n["n"] += 1
        content = "not json" if n["n"] <= 2 else GOOD["choices"][0]["message"]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    draft = _client(handler).analyze(__import__("pathlib").Path(video_file),
                                     SegmentHint(camera="x", lens="ptz"))
    assert n["n"] == 3 and draft.score == 85


def test_analyze_gives_up_after_2_retries(video_file):
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "junk"}}]})

    with pytest.raises(RecognitionError):
        _client(handler).analyze(__import__("pathlib").Path(video_file),
                                 SegmentHint(camera="x", lens="single"))
```

（`RecognitionError` 放 `contracts.py`：`class RecognitionError(Exception)`——执行时补进 Task 9 文件并加一行 import 测试。）

- [ ] **Step 2: 跑测试确认失败** → `pytest tests/test_openai_adapter.py -v` FAIL
- [ ] **Step 3: 实现**

```python
# src/camdigest/llm/openai_adapter.py
"""OpenAI 兼容协议适配器。识别角色用百炼私有 content-part（video_url/input_audio）。"""
from __future__ import annotations

import base64
from pathlib import Path

import httpx

from camdigest.config import ModelCfg
from camdigest.llm.contracts import EventDraft, FaceHint, RecognitionError, SegmentHint
from camdigest.media import extract_audio

SYSTEM = (
    "你是家庭监控视频分析器。画面人物身份标签由人脸识别给出，直接采信，不要重新判断身份。"
    "音频转写文本是上下文参考。输出严格 JSON："
    '{"category":"family|stranger|visitor|animal|vehicle|empty",'
    '"people":[出现的身份名],"score":0-100,"title":"≤12字标题","description":"≤60字描述"}'
)


class OpenAIRecognition:
    def __init__(self, cfg: ModelCfg):
        self.cfg = cfg
        self._http = httpx.Client(timeout=cfg.timeout_seconds)

    def analyze(self, video_path: Path, hint: SegmentHint) -> EventDraft:
        audio_wav = video_path.with_suffix(".hint.wav")
        extract_audio(video_path, audio_wav)
        content = [
            {"type": "video_url", "video_url": {
                "url": "data:video/mp4;base64," + base64.b64encode(video_path.read_bytes()).decode()}},
            {"type": "input_audio", "input_audio": {
                "data": base64.b64encode(audio_wav.read_bytes()).decode(), "format": "wav"}},
            {"type": "text", "text": self._hint_text(hint)},
        ]
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": content}]
        last_err = "unknown"
        for _ in range(3):  # 首次 + 2 次重试（附错误反馈）
            r = self._http.post(f"{self.cfg.base_url}/chat/completions",
                                headers={"Authorization": f"Bearer {self.cfg.api_key}"},
                                json={"model": self.cfg.model, "messages": messages,
                                      "response_format": {"type": "json_object"}})
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            try:
                return EventDraft.from_json(text)
            except Exception as e:  # 解析/校验失败 → 带反馈重试
                last_err = str(e)
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": f"输出不是合法 EventDraft JSON：{last_err}，请只输出修正后的 JSON"})
        raise RecognitionError(last_err)

    @staticmethod
    def _hint_text(hint: SegmentHint) -> str:
        who = "、".join({f.identity for f in hint.face_labels}) or "未检出人脸"
        lines = [f"机位：{hint.camera}（{hint.lens}）", f"画面中为：{who}"]
        if hint.transcript:
            lines.append(f"音频转写：{hint.transcript[:500]}")
        return "\n".join(lines)


class OpenAIReport:
    def __init__(self, cfg: ModelCfg):
        self.cfg = cfg
        self._http = httpx.Client(timeout=cfg.timeout_seconds)

    def write(self, date: str, events: list[dict]) -> str:
        import json as _json
        r = self._http.post(f"{self.cfg.base_url}/chat/completions",
                            headers={"Authorization": f"Bearer {self.cfg.api_key}"},
                            json={"model": self.cfg.model, "messages": [
                                {"role": "system", "content": REPORT_SYSTEM},
                                {"role": "user", "content": f"日期：{date}\n事件清单：\n"
                                 + _json.dumps(events, ensure_ascii=False, indent=1)}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"


REPORT_SYSTEM = (
    "你是家庭日报撰稿人。根据事件清单写 Markdown 日报，固定五板块，标题用二级标题："
    "## 今日总览、## 时间线、## 人物出没、## 异常事件、## 精华清单。"
    "时间线按时间排序，每条含时刻、标题、一句话描述；有 keyframe 的事件在该条目后单独一行输出占位符 {{img:<event_id>}}。"
    "人物出没按人统计出现时段。语言温暖简洁，面向家庭成员。"
)
```

`media.py` 追加：

```python
def extract_audio(path: Path, out_wav: Path) -> Path:
    subprocess.run(["ffmpeg", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", str(out_wav)], check=True, capture_output=True)
    return out_wav


def cut_clip(path: Path, start_s: float, end_s: float, out: Path) -> Path:
    subprocess.run(["ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}",
                    "-i", str(path), "-c", "copy", str(out)], check=True, capture_output=True)
    return out


def grab_frame(path: Path, t: float, out: Path) -> Path:
    subprocess.run(["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(path),
                    "-frames:v", "1", "-q:v", "3", str(out)], check=True, capture_output=True)
    return out
```

- [ ] **Step 4: 跑测试确认通过** → `pytest tests/test_openai_adapter.py tests/test_media.py -v` 全 PASS
- [ ] **Step 5: Commit**

```bash
git add src/camdigest/llm/openai_adapter.py src/camdigest/media.py tests/test_openai_adapter.py
git commit -m "feat: OpenAI适配器——video_url/input_audio组装、JSON重试反馈"
```

## Task 11: Anthropic 适配器（仅报告）

**Files:**
- Create: `src/camdigest/llm/anthropic_adapter.py`
- Test: `tests/test_anthropic_adapter.py`

**Interfaces:**
- Produces: `AnthropicReport(cfg).write(date, events) -> str`（Messages API，`x-api-key` + `anthropic-version: 2023-06-01`，复用 Task 10 的 `REPORT_SYSTEM`）

- [ ] **Step 1: 写失败测试**（MockTransport 校验 header、body 结构与返回取值，模式同 Task 10）
- [ ] **Step 2: 确认失败**
- [ ] **Step 3: 实现**

```python
# src/camdigest/llm/anthropic_adapter.py
"""Anthropic Messages API 适配器——仅报告角色（ADR-0001 协议边界）。"""
from __future__ import annotations

import json

import httpx

from camdigest.config import ModelCfg
from camdigest.llm.openai_adapter import REPORT_SYSTEM


class AnthropicReport:
    def __init__(self, cfg: ModelCfg):
        self.cfg = cfg
        self._http = httpx.Client(timeout=cfg.timeout_seconds)

    def write(self, date: str, events: list[dict]) -> str:
        r = self._http.post(f"{self.cfg.base_url}/v1/messages",
                            headers={"x-api-key": self.cfg.api_key,
                                     "anthropic-version": "2023-06-01"},
                            json={"model": self.cfg.model, "max_tokens": 4096,
                                  "system": REPORT_SYSTEM,
                                  "messages": [{"role": "user", "content":
                                      f"日期：{date}\n事件清单：\n"
                                      + json.dumps(events, ensure_ascii=False, indent=1)}]})
        r.raise_for_status()
        return "".join(b["text"] for b in r.json()["content"] if b.get("type") == "text")
```

- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: Commit** `git commit -m "feat: Anthropic报告适配器"`

## Task 12: S6 识别（混合打分 + 逐段调用）

**Files:**
- Create: `src/camdigest/pipeline/s6_recognize.py`
- Test: `tests/test_s6.py`

**Interfaces:**
- Consumes: `OpenAIRecognition`、`media.cut_clip`
- Produces: `hybrid_score(category: str, model_score: int) -> int`；`build_recognition(settings) -> RecognitionModel`（工厂，按 protocol 选适配器）；`run_recognition(date, session, settings) -> int`
- 规则（spec §6）：`family 70 / stranger 50 / visitor 55 / animal 40 / vehicle 35 / empty 10`；模型分 ≥ 规则分+20 才采信模型分

- [ ] **Step 1: 写失败测试**

```python
# tests/test_s6.py
import pytest

from camdigest.pipeline.s6_recognize import hybrid_score


@pytest.mark.parametrize("category,model,expect", [
    ("family", 85, 85),      # 85 >= 70+20 → 采信
    ("family", 80, 70),      # 80 < 90     → 规则兜底
    ("empty", 90, 90),
    ("empty", 5, 10),
    ("stranger", 40, 50),
])
def test_hybrid_score(category, model, expect):
    assert hybrid_score(category, model) == expect
```

- [ ] **Step 2: 确认失败** → FAIL
- [ ] **Step 3: 实现**

```python
# src/camdigest/pipeline/s6_recognize.py
"""S6 识别：候选片段逐段送识别模型 → EventDraft 落 segments.draft（spec §6）。"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from camdigest.config import Settings
from camdigest.db import MediaFile, Segment
from camdigest.llm.contracts import FaceHint, RecognitionModel, SegmentHint
from camdigest.media import cut_clip

RULE_SCORE = {"family": 70, "stranger": 50, "visitor": 55,
              "animal": 40, "vehicle": 35, "empty": 10}


def hybrid_score(category: str, model_score: int) -> int:
    rule = RULE_SCORE.get(category, 10)
    return model_score if model_score >= rule + 20 else rule


def build_recognition(settings: Settings) -> RecognitionModel:
    from camdigest.llm.openai_adapter import OpenAIRecognition
    return OpenAIRecognition(settings.models.recognition)


def run_recognition(date: str, session: Session, settings: Settings) -> int:
    model = build_recognition(settings)
    cams = {c.id: c for c in settings.cameras}
    done = 0
    for seg, media in (session.query(Segment, MediaFile)
                       .join(MediaFile, Segment.media_file_id == MediaFile.id)
                       .filter(Segment.recognition_status == "pending").all()):
        if media.start_ts.strftime("%Y-%m-%d") != date:
            continue
        cam = cams[media.camera_id]
        hint = SegmentHint(camera=cam.name, lens=cam.lens,
                           face_labels=[FaceHint(**f) for f in (seg.face_labels or [])],
                           transcript=seg.transcript)
        clip = cut_clip(Path(media.path), seg.start_s, seg.end_s,
                        Path(media.path).with_suffix(f".seg{seg.id}.mp4"))
        try:
            draft = model.analyze(clip, hint)
            seg.draft = {"category": draft.category, "people": draft.people,
                         "score": hybrid_score(draft.category, draft.score),
                         "title": draft.title, "description": draft.description}
            seg.recognition_status = "ok"
        except Exception as e:  # 重试已耗尽 → 标记失败，人工补跑（ADR-0001）
            seg.recognition_status = "failed"
            seg.draft = {"error": str(e)[:500]}
        finally:
            clip.unlink(missing_ok=True)
            clip.with_suffix(".hint.wav").unlink(missing_ok=True)
            Path(media.path).with_suffix(".hint.wav").unlink(missing_ok=True)
        done += 1
    return done
```

- [ ] **Step 4: 确认通过** → `pytest tests/test_s6.py -v`
- [ ] **Step 5: Commit** `git commit -m "feat: S6识别——混合打分、draft落库、失败标记"`

---

# Phase 4 —— 产物层（S7-S10）

## Task 13: S7 归并（规则成文）+ 异常 + 关键帧

**Files:**
- Create: `src/camdigest/pipeline/s7_merge.py`
- Test: `tests/test_s7.py`

**Interfaces:**
- Consumes: `Segment.draft`、`anomaly` 配置、`media.grab_frame`
- Produces:
  - `@dataclass SegLite`（`event 之外的轻量段视图`：`seg_id, device, camera_id, lens, start_ts, end_ts, media_path, start_s, end_s, face_labels, draft, score`）
  - `same_device_merge(segs) -> list[set[int]]`（同设备时间重叠 → 同一事件，并查集）
  - `cross_device_merge(groups, segs) -> list[set[int]]`（跨设备：时间相邻 ≤120s 且「共享已知身份」或「双方均无已知身份且 category 相同」）
  - `detect_anomaly(events, cfg, family_names) -> None`（规则②`stranger_while_family_absent`：事件为 stranger 且前后 `family_absent_minutes` 内无任何 family 事件 → `is_anomaly=True`）
  - `pick_keyframe_ts(seg) -> float`（人脸时间戳密度峰，无脸取中点）
  - `run_merge(date, session, settings) -> int`（产出 events 行 + `/data/keyframes/{date}/ev{id}.jpg`）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_s7.py
from datetime import datetime, timedelta, timezone

from camdigest.pipeline.s7_merge import SegLite, cross_device_merge, detect_anomaly, same_device_merge


def _seg(i, device, start, *, lens="fixed", cats=("family",), faces=("妈妈",), camera=None):
    t0 = datetime(2026, 9, 29, tzinfo=timezone.utc)
    return SegLite(seg_id=i, device=device, camera_id=camera or f"{device}-{lens}", lens=lens,
                   start_ts=t0 + timedelta(minutes=start),
                   end_ts=t0 + timedelta(minutes=start + 2),
                   media_path="/x.mp4", start_s=0, end_s=120,
                   face_labels=[{"identity": f, "conf": 0.9, "ts": 1} for f in faces],
                   draft={"category": cats[0], "score": 80, "title": "t", "description": "d",
                          "people": list(faces)}, score=80)


def test_same_device_overlap_merges():
    a = _seg(1, "living", 10)              # 10:00-10:02
    b = _seg(2, "living", 11, lens="ptz")  # 10:01-10:03 重叠 → 必并
    groups = same_device_merge([a, b])
    assert {a.seg_id, b.seg_id} in [set(g) for g in groups]


def test_cross_device_identity_merges():
    a = _seg(1, "living", 10, camera="living-fixed")
    c = _seg(3, "gate", 12, camera="gate")     # 间隔 0，共享"妈妈"
    groups = cross_device_merge([{a.seg_id}], [a, c])
    assert {1, 3} in [set(g) for g in groups]


def test_cross_device_noface_same_category():
    a = _seg(1, "yard", 10, cats=("animal",), faces=(), camera="yard")
    c = _seg(2, "gate", 13, cats=("animal",), faces=(), camera="gate")
    groups = cross_device_merge([{a.seg_id}], [a, c])  # 间隔 1min ≤2min，同 category
    assert {1, 2} in [set(g) for g in groups]


def test_cross_device_far_apart_not_merged():
    a = _seg(1, "yard", 10, faces=(), cats=("animal",), camera="yard")
    c = _seg(2, "gate", 30, faces=(), cats=("animal",), camera="gate")
    groups = cross_device_merge([{a.seg_id}], [a, c])
    assert [set(g) for g in groups] == [{1}]


def test_anomaly_stranger_while_family_absent():
    from camdigest.config import AnomalyCfg
    fam = _seg(1, "living", 0, camera="a")
    stranger = _seg(2, "gate", 200, cats=("stranger",), faces=("未知",), camera="b")
    evs = [{"ids": {1}, "category": "family", "start": fam.start_ts, "end": fam.end_ts, "is_anomaly": False},
           {"ids": {2}, "category": "stranger", "start": stranger.start_ts, "end": stranger.end_ts, "is_anomaly": False}]
    detect_anomaly(evs, AnomalyCfg(), family_names={"妈妈"})
    assert evs[1]["is_anomaly"] is True and evs[0]["is_anomaly"] is False
```

- [ ] **Step 2: 确认失败** → FAIL
- [ ] **Step 3: 实现核心（并查集两轮）**

```python
# src/camdigest/pipeline/s7_merge.py 核心

def _union_find(ids):
    parent = {i: i for i in ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b): parent[find(a)] = find(b)
    return find, union


def same_device_merge(segs: list[SegLite]) -> list[set[int]]:
    find, union = _union_find([s.seg_id for s in segs])
    by_dev: dict[str, list[SegLite]] = {}
    for s in segs:
        by_dev.setdefault(s.device, []).append(s)
    for group in by_dev.values():
        group.sort(key=lambda s: s.start_ts)
        for i, s in enumerate(group):          # 时间重叠 → 同源必并
            for t in group[i + 1:]:
                if t.start_ts >= s.end_ts:
                    break
                union(s.seg_id, t.seg_id)
    out: dict[int, set[int]] = {}
    for s in segs:
        out.setdefault(find(s.seg_id), set()).add(s.seg_id)
    return list(out.values())


def cross_device_merge(groups: list[set[int]], segs: list[SegLite]) -> list[set[int]]:
    by_id = {s.seg_id: s for s in segs}
    ids = [i for g in groups for i in g]
    find, union = _union_find(ids)

    def known_ids(g):   # 组内已知身份（非"未知"）
        return {f["identity"] for i in g for f in by_id[i].face_labels
                if f["identity"] != "未知"}

    def span(g):
        return (min(by_id[i].start_ts for i in g), max(by_id[i].end_ts for i in g))

    def top_cat(g):     # 组内最高分段的 category
        best = max(g, key=lambda i: by_id[i].score)
        return by_id[best].draft["category"]

    changed = True
    while changed:      # 合并传递闭包，扫到不动点
        changed = False
        reps = [find(i) for i in set(ids)]
        gs = [ {i for i in ids if find(i) == r} for r in reps ]
        for a in range(len(gs)):
            for b in range(a + 1, len(gs)):
                if find(next(iter(gs[a]))) == find(next(iter(gs[b]))):
                    continue
                (a0, a1), (b0, b1) = span(gs[a]), span(gs[b])
                gap = max((b0 - a1).total_seconds(), (a0 - b1).total_seconds())
                if gap > 120:                      # 跨设备容差 ≤2 分钟
                    continue
                ka, kb = known_ids(gs[a]), known_ids(gs[b])
                if (ka & kb) or (not ka and not kb and top_cat(gs[a]) == top_cat(gs[b])):
                    union(next(iter(gs[a])), next(iter(gs[b])))
                    changed = True
    out: dict[int, set[int]] = {}
    for i in ids:
        out.setdefault(find(i), set()).add(i)
    return list(out.values())


def detect_anomaly(evs, cfg: AnomalyCfg, family_names: set[str]) -> None:
    """规则 stranger_while_family_absent：stranger 事件前后 family_absent_minutes
    内无任何 family 事件（含命中家人身份）→ is_anomaly=True。"""
    rule = cfg.stranger_while_family_absent
    if not rule.enabled:
        return
    fam_times = [(e["start"], e["end"]) for e in evs if e["category"] == "family"]
    for e in evs:
        if e["category"] != "stranger":
            continue
        window = timedelta(minutes=rule.family_absent_minutes)
        near = any(fs - window <= e["end"] and fe + window >= e["start"] for fs, fe in fam_times)
        e["is_anomaly"] = not near
```

`run_merge` 按概述落库：组→Event 行（score=组内 max、title/description 取最高分段 draft、camera_ids/segment_ids 去重、时间 min/max）→ `detect_anomaly` → `grab_frame(pick_keyframe_ts(最高分段))` 落 `keyframes/{date}/ev{id}.jpg` 回填。
- [ ] **Step 4: 确认通过** → `pytest tests/test_s7.py -v`，5 passed
- [ ] **Step 5: Commit** `git commit -m "feat: S7归并——同源必并/跨设备规则/异常判定/关键帧"`

## Task 14: S8 日报

**Files:**
- Create: `src/camdigest/pipeline/s8_report.py`
- Test: `tests/test_s8.py`

**Interfaces:**
- Consumes: `events` 行 + `reports` 表
- Produces: `build_report_model(settings) -> ReportModel`（openai/anthropic 按 `models.report.protocol`）；`events_payload(events) -> list[dict]`（事件 → 喂模型的 JSON：id/时刻/标题/描述/category/人物/是否异常/有无关键帧）；`run_report(date, session, settings) -> Path`（写 `/data/reports/{date}.md` + `reports` 行）

- [ ] **Step 1: 写失败测试**：`events_payload` 字段齐全且时间为 `HH:MM`；`run_report` 用假 `ReportModel`（monkeypatch `build_report_model`）产出 md 文件与 reports 行
- [ ] **Step 2: 确认失败**
- [ ] **Step 3: 实现**（`events_payload` 把 Event 行转 dict，时刻 `start_ts.strftime("%H:%M")`；`run_report` 调 `model.write(date, payload)`，落盘 `reports/{date}.md`，`session.merge(Report(date=..., md_path=...))`）
- [ ] **Step 4: 确认通过**
- [ ] **Step 5: Commit** `git commit -m "feat: S8日报——事件清单→五板块Markdown"`

## Task 15: S9 飞书发布

**Files:**
- Create: `src/camdigest/publish/feishu.py`
- Test: `tests/test_feishu.py`（md→blocks 纯函数单测；真实 API 用例标 heavy）

**Interfaces:**
- Produces: `md_to_blocks(md: str, images: dict[str, bytes]) -> list[dict]`（支持 `#/##` 标题、段落、`- ` 列表、`{{img:<event_id>}}` 图片占位）；`FeishuClient(cfg)`：`_token()` 缓存 tenant_access_token、`create_doc(title) -> (doc_id, url)`、`upload_image(doc_id, jpg_bytes) -> file_token`、`append_blocks(doc_id, blocks)`、`send_card(title, url)`；`publish_feishu(date, session, settings) -> str`（返回 doc_url，回填 `reports.feishu_doc_url/feishu_msg_id`）
- 细节：图片压缩到 `image_max_bytes`（ffmpeg `-vf scale` 或 PIL，取简）；批量上传间 `time.sleep(0.5)`（spec §12 频控）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_feishu.py
from camdigest.publish.feishu import md_to_blocks


def test_md_to_blocks_structure():
    md = "## 今日总览\n\n一切安好。\n\n## 时间线\n\n- 10:00 妈妈回家\n\n{{img:7}}\n"
    blocks = md_to_blocks(md, images={"7": b"\xff\xd8fake"})
    kinds = [b["block_type"] for b in blocks]
    assert 3 in kinds            # heading
    assert 2 in kinds            # text
    assert 27 in kinds           # image (docx image block type)
```

（block_type 数值以飞书 docx API 为准：heading=3、text=2、image=27；执行者以官方 blocks 文档核对，不对就改常量，测试随之改。）
- [ ] **Step 2: 确认失败**
- [ ] **Step 3: 实现**（HTTP 走 httpx，端点与 spec §8 一致：`POST /open-apis/docx/v1/documents`、`POST /open-apis/drive/v1/medias/upload_all`（`parent_type=docx_image`）、`PATCH /open-apis/docx/v1/documents/{id}/blocks/{block_id}/children`、`POST /open-apis/im/v1/messages?receive_id_type=chat_id`）
- [ ] **Step 4: 确认通过 + heavy 真实凭据用例（可选跑）**
- [ ] **Step 5: Commit** `git commit -m "feat: S9飞书——md转blocks、建文档传图发卡片"`

## Task 16: S10 选段（纯函数，重单测）

**Files:**
- Create: `src/camdigest/pipeline/s10_selection.py`
- Test: `tests/test_s10_selection.py`

**Interfaces:**
- Produces:
  - `@dataclass SelEvent`（`event_id, camera_id, device, lens, media_path, start_s, end_s, score, start_ts, face_ts: list[float], scene_density: float`——由 `from_rows(events, segments)` 构造，**同事件多路段先选一路**：`best_of_event()` 取人脸平均置信度高的一路，平局取固定路）
  - `densest_window(face_ts, seg_start, seg_end, max_seconds) -> tuple[float, float]`（≤max_seconds 内人脸时间戳最多窗口，无脸取居中）
  - `select_pool(events: list[SelEvent], cfg: HighlightCfg) -> list[ClipPlan]`（四步：每小时保底 1 → 每小时补到 `max_per_hour`（与已选间隔 <`event_gap_minutes` 跳过）→ 按分补到 60min → 放开去重兜底；输出按真实时间排序）
  - `@dataclass ClipPlan(event_id, media_path, start_s, end_s, score, camera_id, tier_rank)`
  - `tier_subset(pool, minutes, cfg) -> list[ClipPlan]`（**池内**按 score 降序取到时长满 `minutes`，再按 start_ts 排序输出——保证短档 ⊆ 长档，ADR-0002）

- [ ] **Step 1: 写失败测试**（覆盖：每小时保底含夜间 0-6 点、30 分钟间隔去重、不足 60min 放开兜底、5min 档 ⊆ 60min 档、densest_window 边界）
- [ ] **Step 2: 确认失败**
- [ ] **Step 3: 实现核心**

```python
# src/camdigest/pipeline/s10_selection.py 核心

def densest_window(face_ts, seg_start, seg_end, max_seconds):
    """≤max_seconds 内人脸时间戳最多的窗口；无脸取居中窗口。"""
    span = min(max_seconds, seg_end - seg_start)
    if not face_ts:
        mid = (seg_start + seg_end) / 2
        return (max(seg_start, mid - span / 2), min(seg_end, mid + span / 2))
    ts = sorted(t for t in face_ts if seg_start <= t <= seg_end)
    best, best_n = (seg_start, seg_start + span), 0
    for i, t in enumerate(ts):
        j = i
        while j + 1 < len(ts) and ts[j + 1] - t <= span:
            j += 1
        n = j - i + 1
        if n > best_n:
            best_n = n
            best = (t, min(seg_end, t + max(span, 1.0)))
    return best


def _total(plans):
    return sum(p.end_s - p.start_s for p in plans)


def select_pool(events: list[SelEvent], cfg: HighlightCfg) -> list[ClipPlan]:
    target = max(cfg.tiers) * 60
    selected: list[SelEvent] = []

    def gap_ok(e):
        return all(abs((e.start_ts - s.start_ts).total_seconds())
                   >= cfg.event_gap_minutes * 60 for s in selected)

    by_hour: dict[int, list[SelEvent]] = {}
    for e in sorted(events, key=lambda e: e.score, reverse=True):
        by_hour.setdefault(e.start_ts.hour, []).append(e)
    for hour in sorted(by_hour):                      # 1. 每小时保底 1（含夜间）
        selected.append(by_hour[hour][0])
    for hour in sorted(by_hour):                      # 2. 每小时补到 N，间隔去重
        for e in by_hour[hour][1:]:
            if sum(1 for s in selected if s.start_ts.hour == hour) >= cfg.max_per_hour:
                break
            if e not in selected and gap_ok(e):
                selected.append(e)
    for e in sorted(events, key=lambda e: e.score, reverse=True):   # 3. 按分补齐
        if _total_plans(selected, cfg) >= target:
            break
        if e not in selected and gap_ok(e):
            selected.append(e)
    for e in sorted(events, key=lambda e: e.score, reverse=True):   # 4. 放开去重
        if _total_plans(selected, cfg) >= target:
            break
        if e not in selected:
            selected.append(e)
    plans = []
    for e in sorted(selected, key=lambda e: e.start_ts):
        s, t = densest_window(e.face_ts, e.start_s, e.end_s, cfg.max_segment_seconds)
        plans.append(ClipPlan(event_id=e.event_id, media_path=e.media_path,
                              start_s=s, end_s=t, score=e.score,
                              camera_id=e.camera_id, tier_rank=0))
    return plans


def tier_subset(pool: list[ClipPlan], minutes: int, cfg) -> list[ClipPlan]:
    """段池内按 score 降序取到时长满 minutes，输出按时间排序（ADR-0002 子集承诺）。"""
    target = minutes * 60
    chosen, total = [], 0.0
    for p in sorted(pool, key=lambda p: p.score, reverse=True):
        if total >= target:
            break
        chosen.append(p)
        total += p.end_s - p.start_s
    return sorted(chosen, key=lambda p: p.start_ts)
```

（`_total_plans(selected, cfg)` = 对已选事件逐个 `densest_window` 后时长求和；为免重复计算可缓存，M1 直接算即可。）`SelEvent.from_rows` 同事件多路段先 `best_of_event()` 择优（人脸平均置信度高者胜，平局取 fixed 路）。
- [ ] **Step 4: 确认通过** → 至少 6 个用例
- [ ] **Step 5: Commit** `git commit -m "feat: S10选段——单池四步、同源择优、档位Top-N"`

## Task 17: S10 剪辑导出

**Files:**
- Create: `src/camdigest/pipeline/s10_clip.py`
- Test: `tests/test_s10_clip.py`

**Interfaces:**
- Consumes: `ClipPlan`、`HighlightCfg.tiers`
- Produces: `cut_ts(media_path, start_s, end_s, out) -> Path`（`-c copy -f mpegts`）；`concat_ts(ts_files, out_mp4) -> Path`（concat demuxer，`-c:v copy -c:a aac` 重编码保原声，spec §7）；`export_highlights(date, pool, settings) -> list[Path]`（60min 合并版 + 三档子集 → `/data/highlights/{date}/精华_{tier}min.mp4`；`per_camera` 时每机位追加独立文件；写 `highlights` 行）
- 空音频流处理：`-c:a aac` 前探测，无声视频加 `-an` 或静音轨，避免 concat 失败

- [ ] **Step 1: 写失败测试**：两个 fixture 视频各切 2s → concat → probe 时长 ≈4s 且可播放（ffprobe 校验 duration/stream）
- [ ] **Step 2: 确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 确认通过**
- [ ] **Step 5: Commit** `git commit -m "feat: S10剪辑——TS切片concat、四档导出、per_camera"`

---

# Phase 5 —— 编排、CLI、部署、验收

## Task 18: 编排器 orchestrator + jobs 断点

**Files:**
- Create: `src/camdigest/pipeline/orchestrator.py`
- Modify: `src/camdigest/config.py`（`Settings` 加 `workers: int = 1`）
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Produces: `STAGES: list[tuple[str, Callable]]`（10 项，函数签名统一 `(date, session, settings) -> int`）；`run_day(date, settings, until: str | None = None) -> dict[str, int]`
- 行为：每阶段查/建 `jobs(date, stage)`：`done` 跳过，否则跑完置 `done`（异常置 `failed` 并 re-raise）；S2 前 seed `Camera` 行（复用 s1 的 merge）；`until` 支持跑到指定阶段（`--until s6` 调试用）

- [ ] **Step 1: 写失败测试**：monkeypatch 各 stage 函数（记录调用序），断言：顺序执行、done 跳过（第二次 run_day 只跑未完成阶段）、failed 阶段中断
- [ ] **Step 2: 确认失败**
- [ ] **Step 3: 实现**（约 50 行；M1 默认串行——`workers>1` 的进程池分片留 TODO 注释指向 M3 watch 优化，不在 M1 实现）
- [ ] **Step 4: 确认通过**
- [ ] **Step 5: Commit** `git commit -m "feat: 编排器——jobs断点续跑、阶段顺序执行"`

## Task 19: CLI

**Files:**
- Create: `src/camdigest/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `main(argv) -> int`；子命令：
  - `camdigest run --date 2026-09-29 [--config config.yaml] [--until s6]`
  - `camdigest enroll-faces [--config]`（调 `s4_face.enroll_faces`）
  - `camdigest backfill --from 2026-09-01 --to 2026-09-28`（逐日 `run_day`，跳过已有 `reports` 的日期可 `--force`）
  - `camdigest schedule [--config]`（APScheduler 按 `schedule.daily_at` 跑前一天，Task 20）
- `--config` 缺省 `config/config.yaml`，环境变量 `CAMDIGEST_CONFIG` 优先

- [ ] **Step 1: 写失败测试**：argparse 解析（`run --date` 正确传参 monkeypatch `run_day`）、未知子命令退出码 2
- [ ] **Step 2/3/4**：实现→失败→通过（纯 argparse + 薄封装）
- [ ] **Step 5: Commit** `git commit -m "feat: CLI——run/enroll-faces/backfill/schedule"`

## Task 20: 调度器 scheduler.py

**Files:**
- Create: `src/camdigest/scheduler.py`
- Test: `tests/test_scheduler.py`

**Interfaces:**
- Produces: `run_scheduler(settings)`：BlockingScheduler + CronTrigger(hour/min, timezone)，job=昨日日期 `run_day`；启动即打日志下一触发时间
- 测试：不真跑 BlockingScheduler——测 `_job_date(now) -> "YYYY-MM-DD"` 与 trigger 构造参数

- [ ] **Step 1-5**：TDD 三步 + Commit `git commit -m "feat: APScheduler日批调度"`

## Task 21: Docker 化

**Files:**
- Create: `docker/Dockerfile`、`docker-compose.yml`、`config.example.yaml`（仓库根，含 CAL-1/CAL-2 假设值标注）
- Test: 手动验证（无自动化）

- [ ] **Step 1: Dockerfile**

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir ".[cv]"
ENTRYPOINT ["camdigest"]
CMD ["schedule", "--config", "/app/config/config.yaml"]
```

- [ ] **Step 2: docker-compose.yml**（对齐 architecture.md §3：`/vol1/nvr:/media/nvr:ro`、`./data:/data`、`./config:/app/config`、insightface sidecar `USE_ONNX=1`、`./models/insightface:/models`）
- [ ] **Step 3: `docker compose build` 通过；`camdigest --help` 容器内可用**
- [ ] **Step 4: README 增补「运行」小节**（ enroll-faces → run --date 首跑流程）
- [ ] **Step 5: Commit** `git commit -m "chore: Docker镜像与compose——双容器拓扑"`

## Task 22: 端到端集成测试 + M1 验收

**Files:**
- Test: `tests/test_e2e.py`（**不标 heavy**：模型全部 monkeypatch，ffmpeg 真跑）

**Interfaces:**
- Consumes: 全部阶段
- Produces: `test_run_day_end_to_end`——fixture：2 机位×2 文件微型视频 + 假 `YoloPersonSampler`（恒 1 人）+ 假 `FaceClient.extract`（返回固定向量）+ 假 `Transcriber`（固定文本）+ 假 `OpenAIRecognition/Report`（固定 draft/固定 md）+ 飞书 monkeypatch（`publish_feishu` 直接返回 URL）。断言：`jobs` 全 done、`events≥1`、`reports` 行存在且 md 文件含五板块标题、`highlights` 四档文件存在且 ffprobe 时长>0

- [ ] **Step 1: 写测试（跑红：orchestrator 缺口全暴露）**
- [ ] **Step 2: 补齐让测试通过（允许改此前任务的接线 bug，禁止改接口）**
- [ ] **Step 3: 全量回归** → `pytest -v` 全绿（heavy 除外）
- [ ] **Step 4: 真机验收（人工）**：`camdigest run --date <昨天>` 在 NAS 或本机对真实转存目录跑通，产物四档 + 飞书日报可打开；顺带落实 CAL-1/CAL-2 实测值
- [ ] **Step 5: Commit** `git commit -m "test: E2E全链路——mock模型+真ffmpeg"`

---

## 验收对照（M1 定义 ↔ 任务）

| spec §11 M1 验收 | 覆盖任务 |
|---|---|
| S1-S10 全链路 | Task 4-17 |
| 人脸文件夹建档 | Task 7 + Task 19 `enroll-faces` |
| 任意历史日期补跑 | Task 18 断点 + Task 19 `backfill` |
| 一键产出四档精华 + 飞书日报 | Task 22 E2E + 真机验收 |

## 执行注意

- **禁止跳步**：每个任务的测试先行；接口签名以本计划 Interfaces 块为准，改动需同步更新计划
- 计划中标注「执行者清理/以官方文档核对」的段落（db 的 DateTimeNullable、s2 的坏行、飞书 block_type 数值）是**已知待清理项**，实现时直接写干净版本，不要照抄
- CAL-1/CAL-2 真机确认后只改 `config.yaml`，代码与计划不动
