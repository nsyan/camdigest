# tests/test_feishu.py —— md→blocks 纯函数单测；HTTP 走 MockTransport；真实 API 用例标 heavy
import httpx
import pytest

from camdigest.publish.feishu import md_to_blocks


def test_md_to_blocks_structure():
    md = "## 今日总览\n\n一切安好。\n\n## 时间线\n\n- 10:00 妈妈回家\n\n{{img:7}}\n"
    blocks = md_to_blocks(md, images={"7": b"\xff\xd8fake"})
    kinds = [b["block_type"] for b in blocks]
    assert 3 in kinds            # heading
    assert 2 in kinds            # text
    assert 27 in kinds           # image (docx image block type)


def test_md_to_blocks_h1_and_paragraph():
    blocks = md_to_blocks("# 日报\n\n正文段落。\n", images={})
    kinds = [b["block_type"] for b in blocks]
    assert 3 in kinds and 2 in kinds
    # 一级标题块与二级标题块同用 block_type=3（level 在块内字段区分）


def test_md_to_blocks_image_without_bytes_keeps_placeholder_text():
    blocks = md_to_blocks("前文\n\n{{img:99}}\n", images={})
    assert all(b["block_type"] != 27 for b in blocks)  # 无图可传 → 占位不生成 image 块
    assert any("{{img:99}}" in str(b) for b in blocks)


def test_feishu_client_token_caches_and_creates_doc():
    from camdigest.config import FeishuCfg
    from camdigest.publish.feishu import FeishuClient
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path.endswith("/open-apis/auth/v3/tenant_access_token/internal"):
            return httpx.Response(200, json={"tenant_access_token": "tok-1", "code": 0})
        if request.url.path == "/open-apis/docx/v1/documents":
            return httpx.Response(200, json={"data": {"document": {
                "document_id": "doc123"}, "url": "https://feishu/doc/doc123"}})
        return httpx.Response(404, json={})

    import httpx as _h
    c = FeishuClient(FeishuCfg(app_id="a", app_secret="s"))
    c._http = _h.Client(transport=_h.MockTransport(handler))
    assert c._token() == "tok-1"
    assert c._token() == "tok-1"          # 缓存：只请求一次
    assert len(calls) == 1
    doc_id, url = c.create_doc("2026-09-29 家庭日报")
    assert doc_id == "doc123" and url.endswith("doc123")


def test_publish_feishu_end_to_end_mock(tmp_path, monkeypatch):
    """假客户端注入：建文档→传图→写块→发卡片→回填 reports 行。"""
    from camdigest import db
    from camdigest.config import Settings
    from camdigest.publish import feishu as fe

    class FakeClient:
        def __init__(self, cfg):
            self.cfg = cfg

        def _token(self):
            return "tok"

        def create_doc(self, title):
            return "docX", "https://feishu/doc/docX"

        def upload_image(self, doc_id, jpg_bytes):
            return "filetok1"

        def append_blocks(self, doc_id, blocks):
            assert doc_id == "docX" and any(b["block_type"] == 27 for b in blocks)
            return len(blocks)

        def send_card(self, title, url):
            return "omsg1"

    monkeypatch.setattr(fe, "FeishuClient", FakeClient)
    settings = Settings(
        cameras=[{"id": "gate", "name": "大门", "device": "gate",
                  "lens": "single", "dir": "/a"}],
        models={"recognition": {"base_url": "http://x", "model": "m", "api_key": "k"},
                "report": {"base_url": "http://x", "model": "m", "api_key": "k"}},
        publish={"feishu": {"app_id": "a", "app_secret": "s", "chat_id": "c1"}},
        storage={"data_dir": str(tmp_path)})
    url = f"sqlite:///{tmp_path}/t.db"
    db.init_db(url)
    md = "## 今日总览\n\n好。\n\n## 时间线\n\n- 10:00 回家\n\n{{img:1}}\n"
    rp = tmp_path / "reports" / "2026-09-29.md"
    rp.parent.mkdir(parents=True)
    rp.write_text(md, encoding="utf-8")
    kf = tmp_path / "keyframes" / "2026-09-29" / "ev1.jpg"
    kf.parent.mkdir(parents=True)
    kf.write_bytes(b"\xff\xd8fake")
    with db.session_scope(url) as s:
        from datetime import UTC, datetime, timedelta
        t0 = datetime(2026, 9, 29, 10, 0, tzinfo=UTC)
        s.add(db.Event(date="2026-09-29", start_ts=t0, end_ts=t0 + timedelta(minutes=1),
                       category="family", score=85, title="回家", description="d",
                       camera_ids=["gate"], segment_ids=[1],
                       keyframe_path=str(kf), is_anomaly=False))
        s.merge(db.Report(date="2026-09-29", md_path=str(rp)))
    with db.session_scope(url) as s:
        doc_url = fe.publish_feishu("2026-09-29", s, settings)
    assert doc_url == "https://feishu/doc/docX"
    with db.session_scope(url) as s:
        row = s.query(db.Report).one()
        assert row.feishu_doc_url == doc_url and row.feishu_msg_id == "omsg1"


@pytest.mark.heavy
def test_publish_real_credentials():
    """真实凭据冒烟（需环境变量+网络），默认跳过。"""
    pytest.skip("需真实飞书凭据（Task 22 真机验收覆盖）")
