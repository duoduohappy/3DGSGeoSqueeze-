"""几何抽稀统一算子：点云体素下采样与网格顶点聚类抽稀共用同一套体素划分与簇聚合口径。

两条通道的差异**只在「每簇留下什么」**：

| 通道 | 每簇保留 | 后续处理 |
|---|---|---|
| 点云（`decimate_points`） | 离簇质心最近的**真实采样点**（不引入人造坐标） | DC 颜色取簇均值降噪 |
| 网格（`decimate_vertices`） | 簇**质心**（新顶点，位置/UV 同步取均值） | 三角形索引重映射 + 剔除退化面 |

共享部分收在本模块：体素划分（`voxel_partition`，复用 `ir.voxel_keys` 的空间索引）、
按目标数量逐档反推体素边长、簇内均值（`cluster_means`）、每簇代表点（`first_per_cluster`
/ `nearest_to_centroid`）、索引重映射与退化剔除（`collapse_triangles`）。

两套策略的差异是显式的：点云按簇数容差收敛（`tol`），网格由细到粗逐档试探并带
「不许抽空 / 退化面不超过 60%」的质量守卫——守卫参数见 `decimate_vertices`。
"""
from __future__ import annotations

import numpy as np

from .ir import voxel_keys

MAX_STEPS = 24          # 体素边长最多试探档数（防止病态数据下无界迭代）
MIN_KEPT_TRIS = 3       # 抽稀后至少保留的三角形数（低于此视为抽空，回退上一档）


# ---------------- 共享原语 ----------------
def voxel_partition(pos: np.ndarray, voxel: float, origin=None) -> tuple[np.ndarray, np.ndarray]:
    """体素划分 → ``(inv, count)``。

    ``inv`` 为每点所属簇号（按体素坐标字典序编号，确定性）；``count`` 为各簇点数。
    ``origin`` 缺省取数据最小角——同一批点在相同 ``voxel`` 下划分唯一，
    因此点云与网格通道对同一坐标系得出**完全一致**的簇，跨通道指标可比。
    """
    pos = np.asarray(pos)
    key, _ = voxel_keys(pos, voxel, origin)
    _, inv, count = np.unique(key, return_inverse=True, return_counts=True)
    return np.asarray(inv).ravel(), count


def cluster_means(feat: np.ndarray, inv: np.ndarray, count: np.ndarray) -> np.ndarray:
    """簇内均值：支持 ``(N,)`` 与 ``(N,k)``，按 float64 累加后还原输入浮点精度。

    ``pos`` 传 float64 得 float64（网格写出用），``uv``/颜色传 float32 得 float32
    （避免把精度变化带进产物）。
    """
    feat = np.asarray(feat)
    flat = feat.reshape(len(feat), -1).astype(np.float64)
    acc = np.zeros((len(count), flat.shape[1]), np.float64)
    np.add.at(acc, inv, flat)
    acc /= np.maximum(count, 1)[:, None]
    out = acc.reshape((len(count),) + feat.shape[1:])
    return out.astype(feat.dtype) if feat.dtype.kind == "f" else out


def first_per_cluster(inv: np.ndarray, count: np.ndarray, priority=None) -> np.ndarray:
    """每簇取一个代表下标（升序返回）。

    ``priority`` 缺省时取原始顺序最早者；给定优先级时取该值最小者
    （如「离质心最近」）。稳定排序保证同分时仍是原始顺序，结果确定性可复现。
    """
    inv = np.asarray(inv).ravel()
    if len(count) == 0:
        return np.zeros(0, np.int64)
    if priority is None:
        order = np.argsort(inv, kind="stable")
    else:
        order = np.lexsort((np.asarray(priority).ravel(), inv))
    starts = np.concatenate(([0], np.cumsum(count)[:-1])).astype(np.int64)
    return np.sort(order[starts])


def nearest_to_centroid(pos: np.ndarray, inv: np.ndarray, count: np.ndarray,
                        centers: np.ndarray | None = None) -> np.ndarray:
    """每簇取离质心最近的点（点云代表点：位置保持真实采样值）。"""
    pos = np.asarray(pos)
    cen = cluster_means(pos, inv, count) if centers is None else centers
    dist = ((pos - cen[inv]) ** 2).sum(1)
    return first_per_cluster(inv, count, dist)


def collapse_triangles(inv: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, int, int]:
    """三角形索引重映射到簇号，剔除退化面（任意两顶点落入同一簇）。

    返回 ``(新索引, 保留面数, 剔除面数)``——剔除数是抽稀是否过猛的直接证据。
    """
    tri = np.asarray(inv)[np.asarray(idx, np.int64).reshape(-1, 3)]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    keep = (a != b) & (b != c) & (a != c)
    return tri[keep].ravel(), int(keep.sum()), int((~keep).sum())


# ---------------- 通道 1：点云体素下采样 ----------------
def decimate_points(pos: np.ndarray, target: int, *, origin=None, start=None,
                    max_steps: int = MAX_STEPS, tol: float = 0.02) -> dict:
    """点云体素下采样：按目标点数反推体素边长，每簇留最接近质心的真实采样点。

    ``tol`` 为允许的相对簇数偏差（绝对下限 8 个簇）。返回
    ``voxel / inv / count / centers / keep / levels``：``keep`` 是保留点索引（升序），
    ``inv`` 与 ``count`` 供调用方做逐簇属性聚合（如 DC 颜色取均值降噪）。
    """
    pos = np.asarray(pos)
    if len(pos) == 0:
        return {"voxel": 0.0, "inv": np.zeros(0, np.int64), "count": np.zeros(0, np.int64),
                "centers": np.zeros((0, pos.shape[1] if pos.ndim > 1 else 3)), "keep": np.zeros(0, np.int64),
                "levels": 0}
    target = max(1, int(target))
    lo = pos.min(0) if origin is None else np.asarray(origin, np.float64)
    ext = float(np.max(pos.max(0) - lo)) or 1.0
    vox = float(start) if start else ext / 64.0
    band = max(8.0, tol * target)
    inv = count = np.zeros(len(pos), np.int64)
    levels = 0
    for levels in range(1, max_steps + 1):
        inv, count = voxel_partition(pos, vox, lo)
        m = len(count)
        if abs(m - target) <= band:                    # 簇数进入容差带即收工
            break
        vox *= (m / max(target, 1)) ** (1.0 / 3.0)     # 立方根外推：簇数 ∝ 边长³
    centers = cluster_means(pos, inv, count)
    return {"voxel": float(vox), "inv": inv, "count": count, "centers": centers,
            "keep": nearest_to_centroid(pos, inv, count, centers), "levels": int(levels)}


# ---------------- 通道 2：网格顶点聚类抽稀 ----------------
def decimate_vertices(pos: np.ndarray, idx: np.ndarray, uv: np.ndarray | None = None,
                      ratio: float = 0.0, *, min_kept_ratio: float = 0.4,
                      grow: float = 1.6, max_steps: int = MAX_STEPS,
                      start_div: float = 128.0) -> tuple:
    """网格顶点体素聚类抽稀（保 UV，重建索引，剔除退化面）。

    由细到粗逐档试探：簇数达标（``≤ 目标顶点数``）且保留三角形比例不低于
    ``min_kept_ratio`` 才采纳；更粗一档会让几何崩坏时**停在上一档**（宁少抽不毁几何），
    任何情况下不会抽空。返回 ``(pos, idx, uv, info)``。
    """
    pos = np.asarray(pos, np.float64)
    idx = np.asarray(idx, np.int64)
    if ratio <= 0 or len(pos) == 0 or len(idx) == 0:
        return pos, idx, uv, {"voxel": 0.0, "clusters": len(pos), "degenerate": 0,
                              "verdict": "skip", "levels": 0}
    n = len(pos)
    target = max(8, int(n * (1.0 - min(ratio, 0.98))))
    lo = pos.min(0)
    ext = float(np.max(pos.max(0) - lo)) or 1.0
    tris0 = len(idx) // 3
    best = (pos, idx, uv, {"voxel": 0.0, "clusters": n, "degenerate": 0,
                           "verdict": "identity", "levels": 0})
    vox = ext / start_div
    for step in range(1, max_steps + 1):
        inv, count = voxel_partition(pos, vox, lo)
        m = len(count)
        if m >= n:                                     # 该档无合并空间 → 加粗重试
            vox *= grow
            continue
        new_pos = cluster_means(pos, inv, count)
        new_uv = cluster_means(uv, inv, count) if uv is not None else None
        new_idx, kept, deg = collapse_triangles(inv, idx)
        info = {"voxel": float(vox), "clusters": int(m), "degenerate": int(deg), "levels": step}
        if kept < MIN_KEPT_TRIS:                       # 兜底：不许抽空
            best = (best[0], best[1], best[2], {**best[3], "verdict": "emptied"})
            break
        if m <= target:
            if kept >= tris0 * min_kept_ratio:          # 达标且退化可控 → 采纳
                best = (new_pos, new_idx, new_uv, {**info, "verdict": "accepted"})
            else:                                       # 否则停在上一档
                best = (best[0], best[1], best[2], {**best[3], "verdict": "guarded"})
            break
        best = (new_pos, new_idx, new_uv, {**info, "verdict": "candidate"})
        vox *= grow
    return best


# ---------------- 通道 3：LOD 点代表选择（保留原顺序/优先级的抽稀） ----------------
def decimate_lod_points(idx: np.ndarray, pos: np.ndarray, voxel: float,
                        priority=None) -> np.ndarray:
    """对给定点集做体素抽稀，每簇按 ``priority`` 取代表（缺省取顺序最早者）。

    与 ``decimate_points`` 的区别：不追求目标点数，体素边长由调用方（LOD 层级）给定。
    """
    idx = np.asarray(idx, np.int64)
    if len(idx) == 0:
        return idx
    p = np.asarray(pos)[idx]
    inv, count = voxel_partition(p, voxel, p.min(0))
    pri = None if priority is None else np.asarray(priority)[idx]
    return idx[first_per_cluster(inv, count, pri)]
