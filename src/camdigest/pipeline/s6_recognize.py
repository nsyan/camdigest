# src/camdigest/pipeline/s6_recognize.py
"""S6 识别：候选片段逐段送识别模型 → EventDraft 落 segments.draft（spec §6）。"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from sqlalchemy.orm import Session

from camdigest.config import Settings
from camdigest.llm.contracts import FaceHint, SegmentHint
from camdigest.llm.openai_adapter import OpenAIRecognition
from camdigest.media import cut_clip
from camdigest.pipeline.query import cams_by_id, segments_for_date

# family 70 / empty 10 来自 spec §6；stranger/visitor/animal/vehicle 为计划补全值（见计划「声明偏离」）
RULE_SCORE = {"family": 70, "stranger": 50, "visitor": 55,
              "animal": 40, "vehicle": 35, "empty": 10}


def hybrid_score(category: str, model_score: int) -> int:
    rule = RULE_SCORE.get(category, 10)
    return model_score if model_score >= rule + 20 else rule


def run_recognition(date: str, session: Session, settings: Settings) -> int:
    model = OpenAIRecognition(settings.models.recognition)
    cams = cams_by_id(settings)
    done = 0
    totals: dict = {}   # spec §12：全日 token 用量汇总（适配器 last_usage 为单次值）
    for seg, media in segments_for_date(date, session, pending_only=True):
        cam = cams[media.camera_id]
        hint = SegmentHint(camera=cam.name, lens=cam.lens,
                           face_labels=[FaceHint(**f) for f in (seg.face_labels or [])],
                           transcript=seg.transcript)
        tmp = settings.storage.data_dir / "tmp" / f"seg{seg.id}"   # 临时文件只进 /data/tmp，
        tmp.mkdir(parents=True, exist_ok=True)                     # 绝不写只读源目录（spec §3 :ro）
        clip = cut_clip(Path(media.path), seg.start_s, seg.end_s, tmp / "clip.mp4")
        try:
            draft = model.analyze(clip, hint)
            for k, v in getattr(model, "last_usage", {}).items():
                if isinstance(v, (int, float)):
                    totals[k] = totals.get(k, 0) + v
            seg.draft = {"category": draft.category, "people": draft.people,
                         "score": hybrid_score(draft.category, draft.score),
                         "title": draft.title, "description": draft.description}
            seg.recognition_status = "ok"
        except Exception as e:  # noqa: BLE001 —— 重试已耗尽 → 标记失败，人工补跑（ADR-0001）
            seg.recognition_status = "failed"
            seg.draft = {"error": str(e)[:500]}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        done += 1
    usage = settings.storage.data_dir / "usage" / f"{date}.json"   # spec §12 token 校准
    usage.parent.mkdir(parents=True, exist_ok=True)
    usage.write_text(json.dumps({"segments": done, "usage": totals},
                                ensure_ascii=False), encoding="utf-8")
    return done
