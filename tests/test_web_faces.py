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


def test_archive_cluster_backfills_and_reenrolls(tmp_path, monkeypatch):
    from datetime import UTC, datetime, timedelta

    from camdigest import db
    from camdigest.config import Settings
    from camdigest.web.faces import archive_cluster

    settings = Settings(
        cameras=[{"id": "gate", "name": "大门", "device": "gate",
                  "lens": "single", "dir": "/x"}],
        models={"recognition": {"base_url": "http://x", "model": "m", "api_key": "k"},
                "report": {"base_url": "http://x", "model": "m", "api_key": "k"}},
        storage={"data_dir": str(tmp_path / "data")})
    settings.faces.registry_dir = tmp_path / "faces"
    (settings.faces.registry_dir).mkdir(parents=True)
    cluster_dir = settings.storage.data_dir / "keyframes" / ".clusters"
    cluster_dir.mkdir(parents=True)
    rep = cluster_dir / "1.jpg"
    rep.write_bytes(b"\xff\xd8fake-rep")

    url = f"sqlite:///{tmp_path}/t.db"
    db.init_db(url)
    t0 = datetime(2026, 9, 29, 10, 0, tzinfo=UTC)
    with db.session_scope(url) as s:
        m = db.MediaFile(camera_id="gate", path="/x.mp4", start_ts=t0,
                         end_ts=t0 + timedelta(seconds=120), duration=120.0)
        s.add(m)
        s.flush()
        s.add(db.Segment(media_file_id=m.id, start_s=0, end_s=60, person_count=1,
                         face_labels=[{"identity": "未知", "conf": 0.9, "ts": 30.0}]))
        s.flush()
        s.add(db.UnknownFace(cluster_id=1, embedding=[1.0, 0.0], segment_id=1,
                             media_path="/x.mp4", ts_in_seg=30.0, det_score=0.9))
        s.add(db.UnknownFace(cluster_id=1, embedding=[0.99, 0.01], segment_id=1,
                             media_path="/x.mp4", ts_in_seg=40.0, det_score=0.8))

    class FakeClient:
        def __init__(self, base_url, timeout=30.0):
            pass

        def extract(self, image_bytes):
            return [__import__("camdigest.pipeline.s4_face", fromlist=["FaceDet"]).FaceDet(
                bbox=[0, 0, 1, 1], det_score=0.9, norm=[1.0, 0.0])]

    monkeypatch.setattr("camdigest.pipeline.s4_face.FaceClient", FakeClient)
    with db.session_scope(url) as s:
        target = archive_cluster(1, "妈妈", s, settings)
    assert (target / "c1.jpg").exists()
    assert (settings.faces.registry_dir / "registry.npz").exists()
    with db.session_scope(url) as s:
        ident = s.query(db.Identity).filter(db.Identity.name == "妈妈").one()
        assert ident.unknown_cluster == "1"
        seg = s.query(db.Segment).one()
        assert seg.face_labels[0]["identity"] == "妈妈"      # 历史回填

    # 重新归档同簇为另一身份：旧身份解绑（幂等语义）
    with db.session_scope(url) as s:
        archive_cluster(1, "爸爸", s, settings)
    with db.session_scope(url) as s:
        mama = s.query(db.Identity).filter(db.Identity.name == "妈妈").one()
        baba = s.query(db.Identity).filter(db.Identity.name == "爸爸").one()
        assert mama.unknown_cluster is None and baba.unknown_cluster == "1"
