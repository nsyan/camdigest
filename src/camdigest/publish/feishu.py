# src/camdigest/publish/feishu.py
"""S9 飞书发布：建 docx → 传关键帧图 → md 转 blocks 写入 → 推群卡片（spec §8）。

端点：POST /open-apis/docx/v1/documents、/open-apis/drive/v1/medias/upload_all
（parent_type=docx_image）、PATCH /open-apis/docx/v1/documents/{id}/blocks/...、
POST /open-apis/im/v1/messages?receive_id_type=chat_id。
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import httpx

from camdigest.config import FeishuCfg, Settings

_IMG_RE = re.compile(r"\{\{img:(\w+)\}\}")

FEISHU_OPEN_BASE = "https://open.feishu.cn"  # 飞书开放平台固定基址，不可配置

# 飞书 docx block_type（官方文档核对值；heading2 与 heading1 同为 3，level 在块字段区分）
BT_TEXT = 2
BT_HEADING = 3
BT_IMAGE = 27


def md_to_blocks(md: str, images: dict[str, bytes],
                 token_of: dict[str, str] | None = None) -> list[dict]:
    """md → docx blocks：#/## 标题、段落、- 列表、{{img:<event_id>}} 图块。

    images: event_id → jpg 字节（无字节不生成图块，占位文本保留）。
    token_of: event_id → file_token（未提供则图块 token 留空，由发布流程回填）。
    """
    blocks: list[dict] = []
    for raw in md.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        m = _IMG_RE.fullmatch(line.strip())
        if m and m.group(1) in images:
            blocks.append({"block_type": BT_IMAGE,
                           "image": {"token": (token_of or {}).get(m.group(1), "")},
                           "_event_id": m.group(1)})
            continue
        if line.startswith("## "):
            blocks.append({"block_type": BT_HEADING,
                           "heading2": {"elements": [{"text_run": {"content": line[3:]}}]}})
        elif line.startswith("# "):
            blocks.append({"block_type": BT_HEADING,
                           "heading1": {"elements": [{"text_run": {"content": line[2:]}}]}})
        elif line.startswith("- "):
            blocks.append({"block_type": BT_TEXT,
                           "bullet": {"elements": [{"text_run": {"content": line[2:]}}]}})
        else:
            blocks.append({"block_type": BT_TEXT,
                           "text": {"elements": [{"text_run": {"content": line}}]}})
    return blocks


def _compress_jpg(data: bytes, max_bytes: int) -> bytes:
    """超限时用 ffmpeg 降采样压缩（取简：三档宽度尝试），失败返回原字节。"""
    if len(data) <= max_bytes:
        return data
    for width in (1280, 800, 480):
        try:
            with tempfile.TemporaryDirectory() as td:
                src, dst = Path(td) / "in.jpg", Path(td) / "out.jpg"
                src.write_bytes(data)
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(src), "-vf", f"scale={width}:-1",
                     "-q:v", "7", str(dst)], check=True, capture_output=True)
                if dst.exists() and dst.stat().st_size <= max_bytes:
                    return dst.read_bytes()
        except Exception:  # noqa: BLE001 —— 压缩失败退回原字节，由频控兜底
            break
    return data


class FeishuClient:
    def __init__(self, cfg: FeishuCfg):
        self.cfg = cfg
        self._http = httpx.Client(timeout=30.0)
        self._token_cache: str | None = None

    def _token(self) -> str:
        """tenant_access_token，进程内缓存。"""
        if not self._token_cache:
            r = self._http.post(
                f"{FEISHU_OPEN_BASE}/open-apis/auth/v3/tenant_access_token/internal",
                json={"app_id": self.cfg.app_id, "app_secret": self.cfg.app_secret})
            r.raise_for_status()
            self._token_cache = r.json()["tenant_access_token"]
        return self._token_cache

    def _auth(self) -> dict:
        return {"Authorization": f"Bearer {self._token()}"}

    def create_doc(self, title: str) -> tuple[str, str]:
        r = self._http.post(
            f"{FEISHU_OPEN_BASE}/open-apis/docx/v1/documents",
            headers=self._auth(),
            json={"folder_token": self.cfg.folder_token, "title": title})
        r.raise_for_status()
        data = r.json()["data"]
        return data["document"]["document_id"], data["url"]

    def upload_image(self, doc_id: str, jpg_bytes: bytes) -> str:
        r = self._http.post(
            f"{FEISHU_OPEN_BASE}/open-apis/drive/v1/medias/upload_all",
            headers=self._auth(),
            files={"file": ("ev.jpg", jpg_bytes, "image/jpeg")},
            data={"parent_type": "docx_image", "parent_node": doc_id, "size": str(len(jpg_bytes))})
        r.raise_for_status()
        return r.json()["data"]["file_token"]

    def append_blocks(self, doc_id: str, blocks: list[dict]) -> int:
        payload = [{k: v for k, v in b.items() if not k.startswith("_")} for b in blocks]
        r = self._http.post(
            f"{FEISHU_OPEN_BASE}/open-apis/docx/v1/documents/{doc_id}/blocks/"
            f"{doc_id}/children", headers=self._auth(), json={"children": payload, "index": -1})
        r.raise_for_status()
        return len(payload)

    def send_card(self, title: str, url: str) -> str:
        card = {"config": {"wide_screen_mode": True},
                "header": {"title": {"tag": "plain_text", "content": title},
                           "template": "blue"},
                "elements": [{"tag": "div", "text": {"tag": "lark_md",
                                                     "content": f"[点击查看日报]({url})"}},
                             {"tag": "action", "actions": [
                                 {"tag": "button", "text": {"tag": "plain_text", "content": "打开日报"},
                                  "type": "primary", "url": url}]}]}
        r = self._http.post(
            f"{FEISHU_OPEN_BASE}/open-apis/im/v1/messages?receive_id_type=chat_id",
            headers=self._auth(),
            json={"receive_id": self.cfg.chat_id, "msg_type": "interactive",
                  "content": json.dumps(card, ensure_ascii=False)})
        r.raise_for_status()
        return r.json()["data"]["message_id"]


def publish_feishu(date: str, session, settings: Settings,
                   client: FeishuClient | None = None) -> str:
    """日报 md → 飞书文档 → 群卡片；回填 reports.feishu_doc_url/feishu_msg_id，返回 doc_url。"""
    from camdigest.db import Event, Report

    cfg = settings.publish.feishu
    fc = client or FeishuClient(cfg)
    report = session.query(Report).filter(Report.date == date).one()
    md = Path(report.md_path).read_text(encoding="utf-8")

    # 关键帧图：event_id → 压缩后字节
    images: dict[str, bytes] = {}
    for ev in session.query(Event).filter(Event.date == date).all():
        if ev.keyframe_path and Path(ev.keyframe_path).exists():
            data = Path(ev.keyframe_path).read_bytes()
            images[str(ev.id)] = _compress_jpg(data, cfg.image_max_bytes)

    doc_id, doc_url = fc.create_doc(f"{date} 家庭日报")
    token_of: dict[str, str] = {}
    for event_id in _IMG_RE.findall(md):
        if event_id in images:
            token_of[event_id] = fc.upload_image(doc_id, images[event_id])
            time.sleep(0.5)  # spec §12 频控：批量上传间 sleep
    blocks = md_to_blocks(md, images, token_of)
    fc.append_blocks(doc_id, blocks)
    msg_id = fc.send_card(f"{date} 家庭日报", doc_url)

    report.feishu_doc_url = doc_url
    report.feishu_msg_id = msg_id
    session.merge(report)
    return doc_url
