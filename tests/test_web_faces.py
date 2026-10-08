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
