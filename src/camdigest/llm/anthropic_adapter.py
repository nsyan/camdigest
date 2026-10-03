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

    def write(self, date: str, payload: dict) -> str:
        r = self._http.post(f"{self.cfg.base_url}/v1/messages",
                            headers={"x-api-key": self.cfg.api_key,
                                     "anthropic-version": "2023-06-01"},
                            json={"model": self.cfg.model, "max_tokens": 4096,
                                  "system": REPORT_SYSTEM,
                                  "messages": [{"role": "user", "content":
                                      f"日期：{date}\n"
                                      + json.dumps(payload, ensure_ascii=False, indent=1)}]})
        r.raise_for_status()
        return "".join(b["text"] for b in r.json()["content"] if b.get("type") == "text")
