# tests/test_web_faces.py —— 聚类纯函数 / 扫描 / 归档 / 路由
from camdigest.web.faces import assign_cluster, cosine


def test_cosine_unit():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


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
    clusters = {1: [[1.0, 0.0]]}
    v = [0.44, 0.0]                      # 余弦 0.44 < 0.45 → 新簇
    assert assign_cluster(v, clusters, threshold=0.45) == 2
