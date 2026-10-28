"""自动分析：属性画像 / 空间分析 / 重要性打分 / 配方规划。"""
from __future__ import annotations
import json
import numpy as np
from .ir import GeoScene, voxel_counts

BYTES = {"pos": 12, "sh_coef": 12, "opacity": 4, "scale": 12, "rot": 16}


def profile(s: GeoScene) -> dict:
    lo, hi = s.bounds()
    extent = float(np.max(hi - lo)) or 1.0
    k = s.sh.shape[1]
    attr = {"position": BYTES["pos"], "sh": k * BYTES["sh_coef"],
            "opacity": BYTES["opacity"], "scale": BYTES["scale"], "rot": BYTES["rot"]}
    total = sum(attr.values()) * s.n
    share = {a: round(b * s.n / total, 4) for a, b in attr.items()}
    counts = {f"v{lv}": int(len(np.unique(np.floor((s.pos - lo) / (extent / 2 ** lv)).astype(np.int64), axis=0)))
              for lv in (5, 8, 10)}
    op = s.opacity
    return {
        "n_primitives": int(s.n), "kind": s.kind,
        "sh_degree": s.sh_degree, "bytes_estimate": int(total),
        "attr_bytes_per_prim": attr, "attr_share": share,
        "extent": round(extent, 3),
        "bbox_min": [round(float(v), 3) for v in lo],
        "bbox_max": [round(float(v), 3) for v in hi],
        "opacity_mean": round(float(op.mean()), 4),
        "opacity_low_ratio": round(float((op < 0.1).mean()), 4),
        "scale_mean": round(float(s.scale.mean()), 4),
        "voxel_occupancy": counts,
        "bottleneck": max(share, key=share.get),
    }


def significance(s: GeoScene, voxel: float | None = None) -> np.ndarray:
    """渲染贡献代理：不透明度 × 投影面积(尺度) × 颜色能量 × 密度修正。"""
    lo, hi = s.bounds()
    if voxel is None:
        voxel = float(np.max(hi - lo)) / 2 ** 9
    cnt = voxel_counts(s.pos, voxel).astype(np.float32)
    geo = np.cbrt(np.maximum(np.prod(s.scale, axis=1), 1e-12))
    energy = np.linalg.norm(s.sh[:, 0, :], axis=1) + 1e-6
    dens = np.where(cnt <= 1, 0.15, np.where(cnt > 8, 0.8, 1.0))   # 漂浮点降权，过密冗余降权
    score = s.opacity * geo * energy * dens
    return score.astype(np.float32)


PLATFORMS = {
    "web":     {"prune": 0.70, "sh_deg": 1, "pos_bits": 16, "sh_bits": 8, "vq": True,  "lowrank": False, "lod": [15, 18, 21]},
    "mobile":  {"prune": 0.85, "sh_deg": 1, "pos_bits": 12, "sh_bits": 8, "vq": True,  "lowrank": True,  "lod": [15, 18]},
    "desktop": {"prune": 0.30, "sh_deg": 2, "pos_bits": 16, "sh_bits": 8, "vq": False, "lowrank": False, "lod": [15, 18, 21]},
    "edge":    {"prune": 0.60, "sh_deg": 1, "pos_bits": 12, "sh_bits": 8, "vq": True,  "lowrank": True,  "lod": [15, 18]},
}


def plan(s: GeoScene, platform="web", size_budget_mb=120.0, psnr_floor=28.0, prof=None) -> dict:
    p = dict(PLATFORMS.get(platform, PLATFORMS["web"]))
    prof = prof or profile(s)
    # 预算反推：按压缩后每基元字节数估算可容纳基元数
    per = (p["pos_bits"] + p["sh_bits"] * s.sh.shape[1] * 3 / 8 * 1 + 8 + 8 + 8) / 8 + 4
    budget_n = size_budget_mb * 1024 ** 2 / max(per, 1)
    if budget_n < s.n:
        p["prune"] = float(min(0.95, max(p["prune"], 1 - budget_n / s.n)))
    is_pc = s.kind == "point"                       # 点云：体素下采样，无 SH 蒸馏/渲染恢复
    method = "voxel" if is_pc else ("render" if s.n <= 80000 else "significance")
    stages = [{"stage": "prune", "ratio": round(p["prune"], 3), "method": method}]
    if not is_pc:
        stages += [{"stage": "distill", "target_degree": p["sh_deg"], "preserve_mean": True},
                   {"stage": "recover", "iters": 3, "lr": 1.5, "n_views": 2, "grow": p.get("grow", 1.06)}]
    # 编码方式由代价模型决定：VQ 每基元 1B，低秩每基元 rank×1B —— 仅高维 SH 时低秩才更省
    n_after = int(s.n * (1 - p["prune"]))
    k = s.sh.shape[1]
    vq_cost = n_after * (1 if 256 >= 256 else 2) + 256 * k * 3 * 2
    lr_rank = _est_rank(s, energy=0.995)
    lr_cost = n_after * lr_rank + lr_rank * k * 3 + lr_rank * 2 + k * 3 * 2
    use_lowrank = bool(p["lowrank"]) and lr_cost < 0.9 * vq_cost
    if use_lowrank:                       # 低秩必须先于量化（因子要被量化，且 sh 由因子重建）
        stages.append({"stage": "lowrank", "energy": 0.995, "bits": 8})
    stages.append({"stage": "quantize", "pos_bits": p["pos_bits"], "sh_bits": p["sh_bits"],
                   "vq": bool(p["vq"]) and not use_lowrank, "codebook": 256})
    p["_enc"] = {"sh": "lowrank" if use_lowrank else ("vq" if p["vq"] else "scalar"),
                 "vq_cost": int(vq_cost), "lowrank_cost": int(lr_cost), "rank": lr_rank}
    return {"target": {"platform": platform, "size_budget_mb": size_budget_mb, "psnr_floor": psnr_floor},
            "stages": stages, "export": {"format": "3dtiles_1.1", "lod_levels": p["lod"],
                                        "geometric_error_scale": 2.0, "quantize_geometry": True,
                                        "sh_in_tiles": "lod"},
            "analysis": prof, "encoding": {"sh": p["_enc"]["sh"], "cost_model": p["_enc"]}}


def relax(recipe: dict, rnd: int) -> dict:
    """质量门禁回调：放宽剪枝、提升 SH 阶数与位宽。"""
    r = json.loads(json.dumps(recipe))
    for st in r["stages"]:
        if st["stage"] == "prune":
            st["ratio"] = round(max(0.1, st["ratio"] * 0.8), 3)
        if st["stage"] == "distill":
            st["target_degree"] = min(3, st["target_degree"] + 1)
        if st["stage"] == "quantize":
            st["pos_bits"] = min(16, st["pos_bits"] + 2)
    r["target"]["relaxed_round"] = rnd
    return r


def _est_rank(s: GeoScene, energy=0.995, cap=4096) -> int:
    """在子样本上估计达到目标能量所需秩，供编码代价模型使用。"""
    x = s.sh.reshape(s.n, -1).astype(np.float32)
    if len(x) > cap:
        x = x[np.random.default_rng(0).choice(len(x), cap, replace=False)]
    x = x - x.mean(0, keepdims=True)
    sv = np.linalg.svd(x, compute_uv=False)
    cum = np.cumsum(sv ** 2) / max(float(np.sum(sv ** 2)), 1e-12)
    return int(max(2, np.searchsorted(cum, energy) + 1))


def significance_render(s: GeoScene, n_views=2, W=320, H=240) -> np.ndarray:
    """渲染贡献显著性（LightGaussian global significance 的 CPU 等价）：累计每基元的前向合成权重。"""
    from .raster import rasterize, orbit_cameras
    tot = np.zeros(s.n, np.float32)
    for cam in orbit_cameras(s, n_views, W, H):
        _, _, cb = rasterize(s, cam, collect=True)
        if cb is not None:
            tot += cb
    mx = float(tot.max())
    return tot / mx if mx > 0 else tot
