# src/camdigest/config.py
"""配置装载：yaml + ${ENV} 展开 + 机位校验。字段对应 docs/architecture.md §9。"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

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
    def load(cls, path: Path) -> Settings:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(_deep_expand(raw))
