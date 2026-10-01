# tests/test_s4.py
import base64
import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from camdigest.media import grab_frame
from camdigest.pipeline.s4_face import (
    FaceClient,
    FaceDet,
    enroll_faces,
    label_faces,
    load_registry,
)


def test_label_faces_matches_registry():
    base = np.zeros(512, dtype=np.float32)
    base[0] = 1.0
    other = np.zeros(512, dtype=np.float32)
    other[1] = 1.0
    registry = {"妈妈": [base], "爸爸": [other]}
    dets = [FaceDet(bbox=[1, 2, 3, 4], det_score=0.9, norm=base),
            FaceDet(bbox=[5, 6, 7, 8], det_score=0.8, norm=other * 0.5 + base * 0.5)]
    labels = label_faces(dets, registry, threshold=0.4, ts=12.0)
    assert labels[0] == {"identity": "妈妈", "conf": pytest.approx(1.0, abs=1e-3), "ts": 12.0}
    assert labels[1]["identity"] in {"妈妈", "爸爸"}   # 0.707 边界，两可都接受


def test_label_faces_unknown():
    stranger = np.zeros(512, dtype=np.float32)
    stranger[2] = 1.0
    base = np.zeros(512, dtype=np.float32)
    base[0] = 1.0
    labels = label_faces([FaceDet(bbox=[0, 0, 1, 1], det_score=0.9, norm=stranger)],
                         {"妈妈": [base]}, threshold=0.4, ts=1.0)
    assert labels[0]["identity"] == "未知"


def test_label_faces_threshold_band():
    """阈值带两侧：cos≈0.5 命中、cos≈0.3 落入未知（threshold=0.4）。"""
    base = np.zeros(512, dtype=np.float32)
    base[0] = 1.0
    perp = np.zeros(512, dtype=np.float32)
    perp[1] = 1.0
    above = (base * 0.5 + perp * np.sqrt(0.75)).astype(np.float32)   # cos≈0.5
    below = (base * 0.3 + perp * np.sqrt(0.91)).astype(np.float32)   # cos≈0.3
    labels = label_faces([FaceDet(bbox=[0, 0, 1, 1], det_score=0.9, norm=above),
                          FaceDet(bbox=[2, 2, 3, 3], det_score=0.9, norm=below)],
                         {"妈妈": [base]}, threshold=0.4, ts=3.0)
    assert labels[0]["identity"] == "妈妈"
    assert labels[0]["conf"] == pytest.approx(0.5, abs=1e-3)
    assert labels[1]["identity"] == "未知"


def test_face_client_extract_roundtrip():
    """POST /extract：base64 data 字段出、bbox/det_score/norm 列表入。"""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"data": [
            {"bbox": [1.0, 2.0, 3.0, 4.0], "det_score": 0.99, "norm": [0.1, 0.2]}]})

    client = FaceClient("http://insightface:18080", transport=httpx.MockTransport(handler))
    dets = client.extract(b"fake-jpeg")
    assert len(dets) == 1
    assert dets[0].bbox == [1.0, 2.0, 3.0, 4.0]
    assert dets[0].det_score == 0.99
    assert dets[0].norm == [0.1, 0.2]
    assert seen[0]["data"] == base64.b64encode(b"fake-jpeg").decode()
    assert seen[0]["_threshold"] == 0.6


def _rest_handler(norm_for: dict[bytes, list[float]]):
    """MockTransport 处理器：按图内容返回单 det；不在表内视为未检出人脸。"""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        norm = norm_for.get(base64.b64decode(body["data"]))
        if norm is None:
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, json={"data": [
            {"bbox": [1.0, 2.0, 3.0, 4.0], "det_score": 0.99, "norm": norm}]})

    return handler


def test_load_registry_reads_npz(tmp_path):
    base = np.zeros(4, dtype=np.float32)
    base[0] = 1.0
    other = np.zeros(4, dtype=np.float32)
    other[1] = 1.0
    np.savez(tmp_path / "registry.npz", 妈妈=np.stack([base]), 爸爸=np.stack([other]))

    def boom(request: httpx.Request) -> httpx.Response:
        raise AssertionError("npz 命中时不应发起 HTTP")

    client = FaceClient("http://insightface:18080", transport=httpx.MockTransport(boom))
    reg = load_registry(tmp_path, client)
    assert set(reg) == {"妈妈", "爸爸"}
    assert np.allclose(reg["妈妈"][0], base)
    assert np.allclose(reg["爸爸"][0], other)


def test_load_registry_scans_dir_when_no_npz(tmp_path):
    (tmp_path / "妈妈").mkdir()
    (tmp_path / "妈妈" / "a.jpg").write_bytes(b"m1")
    base = np.zeros(4, dtype=np.float32)
    base[0] = 1.0
    client = FaceClient("http://insightface:18080",
                        transport=httpx.MockTransport(_rest_handler({b"m1": base.tolist()})))
    reg = load_registry(tmp_path, client)
    assert set(reg) == {"妈妈"}
    assert np.allclose(reg["妈妈"][0], base)


def test_enroll_faces_writes_npz(tmp_path):
    (tmp_path / "妈妈").mkdir()
    (tmp_path / "妈妈" / "a.jpg").write_bytes(b"m1")
    (tmp_path / "妈妈" / "b.jpg").write_bytes(b"noface")   # 未检出人脸的照片跳过
    (tmp_path / "爸爸").mkdir()
    (tmp_path / "爸爸" / "a.jpg").write_bytes(b"d1")
    base = np.zeros(4, dtype=np.float32)
    base[0] = 1.0
    other = np.zeros(4, dtype=np.float32)
    other[1] = 1.0
    client = FaceClient("http://insightface:18080", transport=httpx.MockTransport(
        _rest_handler({b"m1": base.tolist(), b"d1": other.tolist()})))

    n = enroll_faces(tmp_path, "http://insightface:18080", client=client)

    assert n == 2                                          # 妈妈 1 张 + 爸爸 1 张
    z = np.load(tmp_path / "registry.npz")
    assert set(z.files) == {"妈妈", "爸爸"}
    assert np.allclose(z["妈妈"][0], base)
    assert np.allclose(z["爸爸"][0], other)


def test_enroll_faces_syncs_identities(tmp_path):
    from camdigest import db
    for name, content in (("妈妈", b"m1"), ("爸爸", b"d1")):
        d = tmp_path / name
        d.mkdir()
        (d / "a.jpg").write_bytes(content)
    base = np.zeros(4, dtype=np.float32)
    base[0] = 1.0
    other = np.zeros(4, dtype=np.float32)
    other[1] = 1.0
    client = FaceClient("http://insightface:18080", transport=httpx.MockTransport(
        _rest_handler({b"m1": base.tolist(), b"d1": other.tolist()})))

    url = f"sqlite:///{tmp_path}/enroll.db"
    db.init_db(url)
    with db.session_scope(url) as s:
        enroll_faces(tmp_path, "http://insightface:18080", session=s, client=client)
    with db.session_scope(url) as s:                       # 新会话验证持久化
        rows = {i.name: i.dir for i in s.query(db.Identity).all()}
    assert rows == {"妈妈": str(tmp_path / "妈妈"), "爸爸": str(tmp_path / "爸爸")}


def test_grab_frame_real_ffmpeg(video_file, tmp_path):
    out = tmp_path / "key.jpg"
    assert grab_frame(Path(video_file), 2.0, out) == out
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:2] == b"\xff\xd8"             # JPEG 魔数
