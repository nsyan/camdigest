# src/camdigest/db.py
"""SQLAlchemy 2.0 模型。表结构 = docs/architecture.md §5（segments 增加 draft 列）。"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
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
    person_count: Mapped[int] = -1  # -1=未处理（S3 幂等标记）
    face_labels: Mapped[list | None] = mapped_column(JSON, default=list)  # [{identity,conf,ts}]
    audio_active: Mapped[bool] = False
    transcript: Mapped[str | None] = mapped_column(String)
    recognition_status: Mapped[str] = mapped_column(String, default="pending")  # pending|ok|failed
    draft: Mapped[dict | None] = mapped_column(JSON)  # S6 EventDraft 快照
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))


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
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))


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
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))


class Report(Base):
    __tablename__ = "reports"
    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[str] = mapped_column(String, unique=True)
    md_path: Mapped[str]
    feishu_doc_url: Mapped[str | None] = mapped_column(String)
    feishu_msg_id: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (UniqueConstraint("date", "stage", name="uq_jobs_date_stage"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[str] = mapped_column(String)
    stage: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="running")  # running|done|failed
    error: Mapped[str | None] = mapped_column(String)
    started_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)


class Identity(Base):
    __tablename__ = "identities"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    dir: Mapped[str] = mapped_column(String)
    unknown_cluster: Mapped[str | None] = mapped_column(String)
    enrolled_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))


_engines: dict[str, object] = {}


def init_db(url: str) -> None:
    eng = create_engine(url, json_serializer=_json_dumps)
    Base.metadata.create_all(eng)
    _engines[url] = eng


@contextmanager
def session_scope(url: str):
    """用法: with session_scope(url) as s: ...  成功即 commit，异常即 rollback。"""
    if url not in _engines:
        init_db(url)
    factory = sessionmaker(bind=_engines[url], expire_on_commit=False)
    with factory.begin() as s:
        yield s


def _json_dumps(obj, **kw):
    import json
    return json.dumps(obj, ensure_ascii=False, **kw)
