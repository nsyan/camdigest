# src/camdigest/llm/contracts.py
"""模型双角色契约（ADR-0001）。识别=全模态仅 openai 协议；报告=文本可选协议。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, field_validator


class RecognitionError(Exception):
    """识别模型调用最终失败（重试耗尽或 4xx 即抛）——调用方捕获后降级处理。"""


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
    """S6 对单个候选片段的识别结果（spec §4 称"事件 JSON"；语义上是候选片段级草稿，
    归并后才成为 CONTEXT 意义上的事件）。"""
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
    def from_json(cls, text: str) -> EventDraft:
        cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        return cls.model_validate(json.loads(cleaned))


@runtime_checkable
class RecognitionModel(Protocol):
    def analyze(self, video_path: Path, hint: SegmentHint) -> EventDraft: ...


@runtime_checkable
class ReportModel(Protocol):
    def write(self, date: str, payload: dict) -> str: ...
