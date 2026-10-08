# src/camdigest/llm/openai_adapter.py
"""OpenAI 兼容协议适配器。识别角色用百炼私有 content-part（video_url/input_audio）。"""
from __future__ import annotations

import base64
import logging
import time
from pathlib import Path

import httpx

from camdigest.config import ModelCfg
from camdigest.llm.contracts import EventDraft, RecognitionError, SegmentHint
from camdigest.media import extract_audio

log = logging.getLogger(__name__)

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
        self.last_usage: dict = {}

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
        last_err, text = "unknown", ""
        self.last_usage = {}
        for attempt in range(3):  # 首次 + 2 次重试；JSON 失败附反馈，5xx/429 退避（spec §4）
            try:
                r = self._http.post(f"{self.cfg.base_url}/chat/completions",
                                    headers={"Authorization": f"Bearer {self.cfg.api_key}"},
                                    json={"model": self.cfg.model, "messages": messages,
                                          "response_format": {"type": "json_object"}})
                r.raise_for_status()
                body = r.json()
                self.last_usage = body.get("usage", {})
                text = body["choices"][0]["message"]["content"]
                return EventDraft.from_json(text)
            except httpx.HTTPStatusError as e:
                last_err = f"HTTP {e.response.status_code}"
                if e.response.status_code < 500 and e.response.status_code != 429:
                    raise  # 4xx（鉴权/参数错）重试无意义
                log.warning("LLM 重试（第 %d 次）：HTTP %d",
                            attempt + 1, e.response.status_code)
                time.sleep(2 ** attempt)
            except Exception as e:  # noqa: BLE001  解析/校验失败 → 带反馈重试
                last_err = str(e)
                log.warning("LLM 重试（第 %d 次）：%s", attempt + 1, last_err)
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": f"输出不是合法 EventDraft JSON：{last_err}，请只输出修正后的 JSON"}]
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
        self.last_usage: dict = {}

    def write(self, date: str, payload: dict) -> str:
        import json as _json
        r = self._http.post(f"{self.cfg.base_url}/chat/completions",
                            headers={"Authorization": f"Bearer {self.cfg.api_key}"},
                            json={"model": self.cfg.model, "messages": [
                                {"role": "system", "content": REPORT_SYSTEM},
                                {"role": "user", "content": f"日期：{date}\n"
                                 + _json.dumps(payload, ensure_ascii=False, indent=1)}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


REPORT_SYSTEM = (
    "你是家庭日报撰稿人。根据输入 JSON（events=事件清单、highlight_paths=四档精华文件路径）写 Markdown 日报，"
    "固定五板块，标题用二级标题：## 今日总览、## 时间线、## 人物出没、## 异常事件、## 精华清单。"
    "时间线按时间排序，每条含时刻、标题、一句话描述；有 keyframe 的事件在该条目后单独一行输出占位符 {{img:<event_id>}}。"
    "人物出没按人统计出现时段。精华清单逐条列出 highlight_paths 的文件（标注'本地 NAS 文件，局域网内可访问'），不要编造。"
    "语言温暖简洁，面向家庭成员。"
)
