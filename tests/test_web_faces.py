# tests/test_web_faces.py —— 聚类纯函数 / 扫描 / 归档 / 路由
import pytest

from camdigest.web.faces import assign_cluster, cosine


def test_cosine_unit():
    # +1e-9 防护使余弦不可能精确 1.0（账本 Task 9 裁决 a）
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0, abs=1e-6)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-6)


def test_assign_cluster_merges_and_splits():
    a = [1.0, 0.0]
    b = [0.999, 0.04]                    # 与 a 余弦≈0.99996 ≥0.45 → 同簇
    c = [0.0, 1.0]                       # 正交 → 新簇
    clusters: dict[int, list] = {}
    c1 = assign_cluster(a, clusters)
    assert assign_cluster(b, clusters) == c1
    c2 = assign_cluster(c, clusters)
    assert c2 != c1 and set(clusters) == {c1, c2}


def test_assign_cluster_threshold_boundary():
    # [0.44, 0.898] 是单位向量，与 [1,0] 夹角余弦≈0.44 < 0.45 → 新簇
    #（原计划把点积 0.44 误当余弦——平行向量余弦恒 1.0，账本 Task 9 裁决 b）
    clusters = {1: [[1.0, 0.0]]}
    assert assign_cluster([0.44, 0.898], clusters, threshold=0.45) == 2


def test_scan_unknown_faces_clusters_and_reps(tmp_path, monkeypatch):
    from datetime import UTC, datetime, timedelta

    from camdigest import db
    from camdigest.config import Settings
    from camdigest.pipeline.s4_face import FaceDet
    from camdigest.web.faces import scan_unknown_faces
    from tests.conftest import make_video

    settings = Settings(
        cameras=[{"id": "gate", "name": "大门", "device": "gate",
                  "lens": "single", "dir": "/x"}],
        models={"recognition": {"base_url": "http://x", "model": "m", "api_key": "k"},
                "report": {"base_url": "http://x", "model": "m", "api_key": "k"}},
        storage={"data_dir": str(tmp_path / "data")})
    media = tmp_path / "gate.mp4"
    make_video(media, seconds=6)
    url = f"sqlite:///{tmp_path}/t.db"
    db.init_db(url)
    t0 = datetime(2026, 9, 29, 10, 0, tzinfo=UTC)
    with db.session_scope(url) as s:
        m = db.MediaFile(camera_id="gate", path=str(media), start_ts=t0,
                         end_ts=t0 + timedelta(seconds=6), duration=6.0)
        s.add(m)
        s.flush()
        s.add(db.Segment(media_file_id=m.id, start_s=0, end_s=6, person_count=1))

    vec_a = [1.0, 0.0]
    vec_b = [0.0, 1.0]
    calls = {"n": 0}

    class FakeClient:
        def __init__(self, base_url, timeout=30.0):
            pass

        def extract(self, image_bytes):
            calls["n"] += 1
            vec = vec_a if calls["n"] % 2 == 1 else vec_b
            score = 0.9 if calls["n"] % 2 == 1 else 0.8
            return [FaceDet(bbox=[0, 0, 1, 1], det_score=score, norm=vec)]

    monkeypatch.setattr("camdigest.web.faces.FaceClient", FakeClient)
    with db.session_scope(url) as s:
        n = scan_unknown_faces("2026-09-29", s, settings)
    assert n == 3                                   # 3 采样帧全部未知入簇
    with db.session_scope(url) as s:
        faces = s.query(db.UnknownFace).all()
        assert {f.cluster_id for f in faces} == {1, 2}
    rep1 = settings.storage.data_dir / "keyframes" / ".clusters" / "1.jpg"
    rep2 = settings.storage.data_dir / "keyframes" / ".clusters" / "2.jpg"
    assert rep1.exists() and rep2.exists()

    # 二次扫描：向量与既有簇合并，不新开簇
    calls["n"] = 0
    with db.session_scope(url) as s:
        scan_unknown_faces("2026-09-29", s, settings)
    with db.session_scope(url) as s:
        assert {f.cluster_id for f in s.query(db.UnknownFace).all()} == {1, 2}
        assert s.query(db.UnknownFace).count() == 6
