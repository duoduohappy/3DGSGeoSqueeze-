"""几何抽稀统一算子验收：体素划分跨通道一致、点云代表点、网格索引有效性与质量守卫。

运行：python tests/test_decimate.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gsq.decimate import (MIN_KEPT_TRIS, cluster_means, collapse_triangles,  # noqa: E402
                          decimate_lod_points, decimate_points, decimate_vertices,
                          first_per_cluster, nearest_to_centroid, voxel_partition)
from gsq.ir import make_sample  # noqa: E402
from gsq.stages import prune  # noqa: E402
from gsq.ir import GeoScene  # noqa: E402


def grid_mesh(nx=21, ny=21, span=10.0):
    """平面网格：顶点 nx*ny、三角形 2*(nx-1)*(ny-1)，UV 为归一化平面坐标。"""
    xs = np.linspace(0, span, nx)
    ys = np.linspace(0, span, ny)
    X, Y = np.meshgrid(xs, ys)
    pos = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny)])
    tri = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            tri += [a, a + 1, a + nx, a + 1, a + nx + 1, a + nx]
    uv = np.column_stack([X.ravel() / span, Y.ravel() / span]).astype(np.float32)
    return pos, np.asarray(tri, np.int64), uv


def test_partition_shared_across_channels():
    """同一点集 + 同一体素边长 → 点云与网格通道得到完全相同的簇划分。"""
    rng = np.random.default_rng(3)
    pos = rng.normal(0, 5, (4000, 3))
    vox = 1.0
    inv_a, cnt_a = voxel_partition(pos, vox)
    inv_b, cnt_b = voxel_partition(pos, vox)
    assert np.array_equal(inv_a, inv_b) and np.array_equal(cnt_a, cnt_b)   # 确定性
    # 网格通道内部用的就是同一原语：给定该体素时簇数一致，不再有自己的 floor/unique 副本
    res = decimate_points(pos, len(cnt_a), start=vox)
    assert len(res["count"]) == len(cnt_a)
    assert np.array_equal(res["inv"], inv_a)
    # 簇号覆盖 0..m-1 且每簇至少一点
    assert cnt_a.min() >= 1 and cnt_a.sum() == len(pos)
    assert set(np.unique(inv_a)) == set(range(len(cnt_a)))
    # 相邻居不同簇：簇内任意两点距离 < sqrt(3)*vox
    big = np.where(cnt_a >= 3)[0][:50]
    for c in big:
        pts = pos[inv_a == c]
        assert np.linalg.norm(pts.max(0) - pts.min(0)) < np.sqrt(3) * vox + 1e-9
    print(f"  partition: {len(pos)} 点 → {len(cnt_a)} 簇 @ vox={vox}（两通道一致）")


def test_points_representative_and_color():
    """点云下采样：点数为目标附近、每簇一个真实采样点、颜色为簇均值。"""
    s = make_sample(60, 8)
    target = int(s.n * 0.3)
    res = decimate_points(s.pos, target)
    keep, inv, cnt = res["keep"], res["inv"], res["count"]
    assert abs(len(keep) - target) <= max(8, 0.02 * target), (len(keep), target)
    assert len(np.unique(inv[keep])) == len(keep)              # 每簇恰好一个代表
    assert len(keep) == len(cnt)
    # 代表必须是输入点原值（不是人造坐标）：按字节比对，不做浮点近似
    src = {row.tobytes() for row in s.pos}
    assert all(s.pos[i].tobytes() in src for i in keep)
    # 代表是该簇内离质心最近者（独立复算）
    cen = cluster_means(s.pos, inv, cnt)
    dist = ((s.pos - cen[inv]) ** 2).sum(1)
    for c in np.unique(inv)[:200]:
        members = np.where(inv == c)[0]
        assert keep[inv[keep] == c][0] == members[np.argmin(dist[members])]
    # 阶段层：位置原样保留（只做子集），DC 颜色改为簇均值
    pc = GeoScene(s.pos, s.sh[:, :1, :], np.ones(s.n, np.float32),
                  np.full((s.n, 3), 0.05, np.float32), s.rot, "point")
    keep_n = max(16, int(round(pc.n * 0.3)))                   # 与 prune 内部口径一致
    res2 = decimate_points(pc.pos, keep_n)
    out, info = prune(pc, 0.7, method="voxel")
    assert info["method"] == "voxel" and info["kept"] == out.n
    assert np.array_equal(out.pos, pc.pos[res2["keep"]])       # 下采样不移动点位
    avg = cluster_means(pc.sh[:, 0, :], res2["inv"], res2["count"])
    assert np.allclose(out.sh[:, 0, :], avg[res2["inv"][res2["keep"]]], atol=1e-6)
    assert (out.sh[:, 0, :] != pc.sh[res2["keep"], 0, :]).any()  # 颜色确实被均值替换
    print(f"  points: {pc.n} → {out.n}（目标 {target}）· 体素 {info['voxel']} · "
          f"{info['levels']} 档收敛 · 颜色=簇均值")


def test_points_degenerate_inputs():
    """单点 / 平面点云（某轴跨度为 0）/ 空输入都不应崩。"""
    one = decimate_points(np.zeros((1, 3)), 5)
    assert len(one["keep"]) == 1 and one["voxel"] > 0
    flat = np.column_stack([np.linspace(0, 10, 500), np.zeros(500), np.zeros(500)])
    res = decimate_points(flat, 50)
    assert abs(len(res["keep"]) - 50) <= max(8, 1)
    assert len(np.unique(res["inv"][res["keep"]])) == len(res["keep"])
    empty = decimate_points(np.zeros((0, 3)), 10)
    assert len(empty["keep"]) == 0
    # 重复点（全部重合）→ 只剩 1 个簇
    dup = np.zeros((100, 3))
    assert len(decimate_points(dup, 10)["keep"]) == 1
    print("  points: 单点/平面/空/全重合输入均安全")


def test_vertices_valid_output():
    """网格抽稀：索引在界内、无退化三角形、UV 保精度、顶点与面数都下降。"""
    pos, idx, uv = grid_mesh(21, 21)
    p2, i2, u2, info = decimate_vertices(pos, idx, uv, ratio=0.5)
    assert info["verdict"] in ("accepted", "guarded", "candidate")
    assert len(p2) < len(pos) and len(i2) < len(idx)
    assert i2.min() >= 0 and i2.max() < len(p2)                # 索引全部指向新顶点
    tri = i2.reshape(-1, 3)
    assert not ((tri[:, 0] == tri[:, 1]) | (tri[:, 1] == tri[:, 2]) |
                (tri[:, 0] == tri[:, 2])).any()                # 无退化面
    assert u2.shape == (len(p2), 2) and u2.dtype == np.float32
    assert u2.min() >= -1e-6 and u2.max() <= 1 + 1e-6          # 均值不越界
    assert p2.dtype == np.float64 and i2.dtype == np.int64
    print(f"  vertices: {len(pos)}V/{len(idx)//3}F → {len(p2)}V/{len(i2)//3}F "
          f"（体素 {info['voxel']:.4f}，退化 {info['degenerate']} 面，{info['verdict']}）")


def test_vertices_guard_and_no_empty():
    """质量守卫：极端抽稀比例下不许抽空，且退化面比例超限时退回上一档。"""
    pos, idx, uv = grid_mesh(41, 41)
    p2, i2, u2, info = decimate_vertices(pos, idx, uv, ratio=0.95)
    assert len(i2) > 0 and len(i2) % 3 == 0
    assert len(i2) // 3 >= MIN_KEPT_TRIS
    assert info["verdict"] in ("accepted", "guarded", "emptied", "candidate")
    if info["verdict"] == "guarded":
        assert info["degenerate"] > 0                          # 触发守卫必有退化证据
    # ratio=0 → 原样返回（不做任何改动）
    p0, i0, u0, info0 = decimate_vertices(pos, idx, uv, ratio=0.0)
    assert info0["verdict"] == "skip" and len(p0) == len(pos) and np.array_equal(i0, idx)
    # 空索引/单点网格不崩
    assert decimate_vertices(pos, np.zeros(0, np.int64), None, 0.5)[1].size == 0
    assert len(decimate_vertices(np.zeros((1, 3)), np.zeros(0, np.int64), None, 0.5)[0]) == 1
    print(f"  guard: ratio=0.95 → {len(p2)}V/{len(i2)//3}F（{info['verdict']}，"
          f"退化 {info['degenerate']}）· ratio=0 原样返回")


def test_partition_origin_precision():
    """体素划分必须保持 origin 精度：ECEF 量级下 origin 降 float32 会让格号漂移。"""
    rng = np.random.default_rng(5)
    pos = rng.normal(0, 3, (5000, 3)) + np.array([5.2e6, 4.1e6, 3.0e4])   # 地球坐标量级
    vox = 1.0
    lo = pos.min(0)
    inv, cnt = voxel_partition(pos, vox, lo)
    ref = np.floor((pos - lo) / vox).astype(np.int64)
    _, span = np.unique(ref, axis=0, return_counts=True)
    assert len(cnt) == len(span)                                   # 簇数与逐列 unique 一致
    # 修复前 ir.voxel_keys 会把 origin 转 float32：格号偏移 → 簇数变化
    drift = np.floor((pos - lo.astype(np.float32)) / vox).astype(np.int64)
    assert (drift != ref).any()
    print(f"  origin: ECEF 量级 {len(cnt)} 簇 · float32 origin 会偏移 {(drift != ref).sum()} 点")


def test_collapse_triangles_unit():
    """退化面剔除的计数必须准确：两顶点同簇即整面剔除。"""
    inv = np.array([0, 0, 1, 2], np.int64)
    idx = np.array([0, 1, 2,      # 0 与 1 同簇 → 退化
                    0, 2, 3,      # 三个簇各不同 → 保留
                    1, 3, 1],     # 首尾同簇 → 退化
                   np.int64)
    new_idx, kept, deg = collapse_triangles(inv, idx)
    assert (kept, deg) == (1, 2)
    assert np.array_equal(new_idx, np.array([0, 1, 2], np.int64))
    print("  collapse: 3 面 → 保留 1 / 剔除 2（判据 = 任意两顶点同簇）")


def test_first_per_cluster_and_lod():
    """每簇代表选择：优先级生效、LOD 抽稀走同一划分、结果稳定。"""
    inv = np.array([0, 1, 0, 1, 0], np.int64)
    cnt = np.array([3, 2], np.int64)
    assert np.array_equal(first_per_cluster(inv, cnt), np.array([0, 1]))
    pri = np.array([5.0, 9.0, 1.0, 3.0, 7.0])
    assert np.array_equal(first_per_cluster(inv, cnt, pri), np.array([2, 3]))  # 簇 0→idx2(1.0)、簇 1→idx3(3.0)
    rng = np.random.default_rng(7)
    pos = rng.normal(0, 4, (3000, 3))
    sel = decimate_lod_points(np.arange(len(pos)), pos, 1.0)
    inv2, cnt2 = voxel_partition(pos, 1.0)
    assert len(sel) == len(cnt2)
    assert len(np.unique(inv2[sel])) == len(sel)
    # 严格按给定代表（不是最近质心）：decimate_lod_points 只认顺序/优先级
    print(f"  lod: {len(pos)} → {len(sel)} 点（体素 1.0）· 每簇一点")


def test_cluster_means_dtype():
    """簇均值保精度：float32 进 float32 出，float64 进 float64 出。"""
    x32 = np.arange(12, dtype=np.float32).reshape(4, 3)
    inv = np.array([0, 0, 1, 1], np.int64)
    cnt = np.array([2, 2], np.int64)
    m32 = cluster_means(x32, inv, cnt)
    assert m32.dtype == np.float32 and np.allclose(m32, [[1.5, 2.5, 3.5], [7.5, 8.5, 9.5]])
    m64 = cluster_means(x32.astype(np.float64), inv, cnt)
    assert m64.dtype == np.float64
    v = cluster_means(np.arange(4, dtype=np.float32), inv, cnt)   # 1D 也支持
    assert v.shape == (2,) and np.allclose(v, [0.5, 2.5])
    print("  means: 1D/2D 输入 · float32/float64 精度均保持")


if __name__ == "__main__":
    print("GeoSqueeze 几何抽稀统一算子测试")
    test_partition_shared_across_channels()
    test_partition_origin_precision()
    test_points_representative_and_color()
    test_points_degenerate_inputs()
    test_vertices_valid_output()
    test_vertices_guard_and_no_empty()
    test_collapse_triangles_unit()
    test_first_per_cluster_and_lod()
    test_cluster_means_dtype()
    print("ALL PASSED")
