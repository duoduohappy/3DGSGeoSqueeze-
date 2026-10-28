"""压缩阶段：剪枝 / 蒸馏 / 量化 / 低秩分解。阶段统一接口 run(scene, cfg) -> (scene, info)。"""
from __future__ import annotations
import numpy as np
from .ir import GeoScene, SH_C0, sh_basis
from .analyze import significance
from .decimate import cluster_means, decimate_points, voxel_partition
from .quant import affine_decode, affine_encode, max_abs_error

def _basis(d: np.ndarray, deg: int) -> np.ndarray:
    return sh_basis(d, min(deg, 3))


def _rand_dirs(n: int, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------- 1. 剪枝 ----------------
def prune(s: GeoScene, ratio=0.7, method="significance", **_) -> tuple[GeoScene, dict]:
    keep_n = max(16, int(round(s.n * (1 - ratio))))
    if keep_n >= s.n:
        return s, {"stage": "prune", "ratio": 0.0, "kept": s.n}
    if method == "voxel" or (s.kind == "point" and method not in ("render", "significance", "merge")):
        return _voxel_downsample(s, keep_n)
    if method == "merge":
        return _merge_prune(s, keep_n)
    if method == "render":
        from .analyze import significance_render
        score = significance_render(s)
        if float(score.max()) <= 0:
            score = significance(s)
    else:
        score = significance(s)
    idx = np.argpartition(-score, keep_n - 1)[:keep_n]
    return s.subset(np.sort(idx)), {"stage": "prune", "ratio": ratio, "kept": int(keep_n),
                                    "method": method, "score_cut": float(score[idx].min())}


def _merge_prune(s: GeoScene, keep_n: int) -> tuple[GeoScene, dict]:
    """合并吸收剪枝（设计 §5.1 merge）：同体素内旋转相近、中心邻近的高斯并为一个更大高斯。

    数量下降但覆盖不丢：pos/DC 颜色按不透明度加权平均，scale 取加权均值 + 群体离散度，
    opacity 取 1-∏(1-o)（联合覆盖概率），rot 保留显著性最高的成员。
    """
    from .analyze import significance
    score = significance(s)
    lo, hi = s.bounds()
    vox = max(float(np.max(hi - lo)) / 128.0, 1e-6)
    inv, _ = voxel_partition(s.pos, vox, lo)              # 与网格抽稀共用体素划分（decimate.py）
    order = np.argsort(-score)                            # 显著性高者作合并种子
    keep_mask = np.ones(s.n, bool)
    merged = 0
    q = s.rot / np.maximum(np.linalg.norm(s.rot, axis=1, keepdims=True), 1e-9)
    rad = np.cbrt(np.maximum(np.prod(s.scale, axis=1), 1e-12))
    # 合并在副本上进行：本阶段不可原地改写入参场景（否则质量门禁下一轮从"已改动"的原始场景重跑）
    pos, sh0 = s.pos.copy(), s.sh[:, 0, :].copy()
    scale, opacity = s.scale.copy(), s.opacity.copy()
    for i in order:
        if not keep_mask[i]:
            continue
        cand = np.where((inv == inv[i]) & keep_mask & (np.arange(s.n) != i))[0]
        if not len(cand):
            continue
        dq = np.abs(q[cand] @ q[i])                           # 四元数点积（|dot|≈1 即同向）
        near = cand[(dq > 0.95) & (np.linalg.norm(pos[cand] - pos[i], axis=1) < 2.0 * rad[i])]
        if not len(near):
            continue
        grp = np.concatenate([[i], near])
        o = opacity[grp][:, None]
        w = o / o.sum()
        pos[i] = (pos[grp] * w).sum(0)
        sh0[i] = (sh0[grp] * w).sum(0)
        # 覆盖扩大：群体越大尺度越大；不透明度取联合覆盖概率
        scale[i] = np.maximum(scale[i] * (1 + 0.25 * len(grp)), scale[grp].max(0) * 0.8)
        opacity[i] = float(1.0 - np.prod(1.0 - opacity[grp]))
        keep_mask[near] = False
        merged += len(near)
        if keep_mask.sum() <= keep_n:
            break
    idx = np.sort(np.where(keep_mask)[0])
    out = GeoScene(pos[idx], s.sh[idx], opacity[idx], scale[idx], s.rot[idx],
                   s.kind, dict(s.meta))
    out.sh[:, 0, :] = sh0[idx]                            # subset 已拷贝，直接写回合并后的 DC
    return out, {"stage": "prune", "method": "merge", "kept": int(out.n),
                 "absorbed": int(merged), "target": int(keep_n)}


def _voxel_downsample(s: GeoScene, keep_n: int) -> tuple[GeoScene, dict]:
    """点云体素下采样（体素划分与网格抽稀共用 `decimate` 算子）。

    每体素保留最接近质心的**真实采样点**（不引入人造坐标），DC 颜色取体素均值降噪。
    """
    res = decimate_points(s.pos, keep_n)
    inv, count, keep = res["inv"], res["count"], res["keep"]
    if len(keep) == 0:
        return s, {"stage": "prune", "method": "voxel", "kept": s.n, "voxel": 0.0,
                   "occupied_voxels": 0, "levels": 0}
    sub = s.subset(keep)
    sub.sh = sub.sh.copy()
    sub.sh[:, 0, :] = cluster_means(s.sh[:, 0, :], inv, count)[inv[keep]]   # 体素平均色，抑制噪声
    return sub, {"stage": "prune", "method": "voxel", "kept": int(len(keep)),
                 "voxel": round(float(res["voxel"]), 5), "occupied_voxels": int(len(count)),
                 "levels": int(res["levels"])}


# ---------------- 2. 蒸馏（SH 截断 + 能量补偿） ----------------
def distill(s: GeoScene, target_degree=1, preserve_mean=True, samples=32, **_) -> tuple[GeoScene, dict]:
    deg = min(target_degree, s.sh_degree)
    if deg >= s.sh_degree:
        return s, {"stage": "distill", "skipped": True, "degree": s.sh_degree}
    k = (deg + 1) ** 2
    full, low = s.sh, s.sh[:, :k, :].copy()
    if preserve_mean:
        d = _rand_dirs(samples)
        bf, bl = _basis(d, s.sh_degree), _basis(d, deg)
        cf = np.einsum("dk,nkc->ndc", bf, full)               # (N,D,3) 各方向颜色偏移
        cl = np.einsum("dk,nkc->ndc", bl, low)
        denom = np.einsum("ndc,ndc->n", cl, cl) + 1e-8
        a = np.clip(np.einsum("ndc,ndc->n", cf, cl) / denom, 0.5, 2.0)
        low = low * a[:, None, None]
    out = GeoScene(s.pos, low, s.opacity, s.scale, s.rot, s.kind, dict(s.meta))
    return out, {"stage": "distill", "degree_from": s.sh_degree, "degree_to": deg,
                 "coef_kept": int(k * 3), "energy_compensated": bool(preserve_mean)}


# ---------------- 3. 量化 ----------------
# 编解码统一走 quant（见 gsq/quant.py）：affine 码型，反解 x = q·step + lo
def _kmeans(x: np.ndarray, c: int, iters=8, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(x) <= c:
        return x.copy()
    cb = x[rng.choice(len(x), c, replace=False)].copy()
    for _ in range(iters):
        d = ((x[:, None, :] - cb[None]) ** 2).sum(-1) if len(x) < 3000 else None
        if d is None:
            lab = np.empty(len(x), np.int64)
            for i in range(0, len(x), 4096):
                blk = x[i:i + 4096]
                lab[i:i + 4096] = ((blk[:, None, :] - cb[None]) ** 2).sum(-1).argmin(1)
        else:
            lab = d.argmin(1)
        for j in range(c):
            m = lab == j
            if m.any():
                cb[j] = x[m].mean(0)
    return cb


def _vq_encode(x: np.ndarray, c: int, seed=0):
    rng = np.random.default_rng(seed)
    samp = x if len(x) <= 4096 else x[rng.choice(len(x), 4096, replace=False)]
    cb = _kmeans(samp, c, seed=seed)
    codes = np.empty(len(x), np.uint8 if c <= 256 else np.uint16)
    for i in range(0, len(x), 4096):
        blk = x[i:i + 4096]
        codes[i:i + 4096] = ((blk[:, None, :] - cb[None]) ** 2).sum(-1).argmin(1)
    return codes, cb


def _rvq(x: np.ndarray, c: int, stages: int = 2, seed: int = 0):
    """残差级联 VQ（VecTree 思想的 CPU 等价）：逐级对残差建码本，重建 = Σ cb_i[code_i]。

    高维 SH 下比单级 VQ 每字节误差更低；每级固定 c 个码字，代价 = stages 字节/基元。
    """
    codes, cbs = [], []
    resid = x.astype(np.float32)
    for lv in range(stages):
        rng = np.random.default_rng(seed + lv)
        samp = resid if len(resid) <= 4096 else resid[rng.choice(len(resid), 4096, replace=False)]
        cb = _kmeans(samp, c, seed=seed + lv)
        lab = np.empty(len(resid), np.uint8 if c <= 256 else np.uint16)
        for i in range(0, len(resid), 4096):
            blk = resid[i:i + 4096]
            lab[i:i + 4096] = ((blk[:, None, :] - cb[None]) ** 2).sum(-1).argmin(1)
        codes.append(lab)
        cbs.append(cb.astype(np.float32))
        resid = resid - cb[lab]
    recon = np.sum([cb[lab] for cb, lab in zip(cbs, codes)], axis=0)
    return codes, cbs, recon


def quantize(s: GeoScene, pos_bits=16, sh_bits=8, vq=True, codebook=256, method=None,
             **_) -> tuple[GeoScene, dict]:
    """真实量化：产出 codec（整型张量 + 反量化参数），产物按 codec 落盘，不再是 float32 转储。

    method: None（自动：vq 布尔决定）/ "vq" / "vectree"（残差级联码本）/ "scalar"。
    """
    n, k = s.n, s.sh.shape[1]
    tensors, errs = [], {}

    def affine(name, x, bits):
        q, lo, st = affine_encode(x, bits)
        tensors.append({"name": name, "arr": q, "lo": float(lo), "step": st})
        rec = affine_decode(q, lo, st)
        errs[name] = round(max_abs_error(x, rec), 6)      # 统一误差口径（见 quant.py）
        return rec

    pos = affine("pos", s.pos, pos_bits)
    opacity = affine("opacity", s.opacity, 8)
    scale = affine("scale", s.scale, 8)
    rot = affine("rot", s.rot, 8)
    sh_codec = s.codec.get("sh") if isinstance(s.codec, dict) else None
    info = {"stage": "quantize", "pos_bits": pos_bits, "pos_mse": float(((pos - s.pos) ** 2).mean()),
            "pos_max_err": errs["pos"]}

    if sh_codec and sh_codec.get("mode") == "lowrank":
        # 低秩因子已存在（由 lowrank 阶段产出），此处只做 fp16 存储，不再二次量化
        tensors.append({"name": "sh.u", "arr": sh_codec["u"],
                        "lo": sh_codec["u_lo"], "step": sh_codec["u_step"]})
        tensors.append({"name": "sh.vt", "arr": sh_codec["vt"],
                        "lo": sh_codec["vt_lo"], "step": sh_codec["vt_step"]})
        tensors.append({"name": "sh.s", "arr": sh_codec["s"]})
        tensors.append({"name": "sh.mu", "arr": sh_codec["mu"]})
        sh = _reconstruct_lowrank(sh_codec, n, k)
        info.update({"sh_mode": "lowrank", "rank": int(sh_codec["rank"])})
    elif method == "vectree":
        x = s.sh.reshape(n, -1)
        stages = 2 if x.shape[1] <= 24 else 3               # 维度越高残差级数越多
        codes, cbs, recon = _rvq(x, min(codebook, max(16, n // 4)), stages=stages)
        sh = recon.reshape(n, k, 3).astype(np.float32)
        tensors.append({"name": "sh.codes", "arr": np.stack(codes, 1)})
        for lv, cb in enumerate(cbs):
            tensors.append({"name": f"sh.cb{lv}", "arr": cb.astype(np.float16)})
        info.update({"sh_mode": "vectree", "codebook": int(len(cbs[0])), "rvq_stages": stages,
                     "sh_mse": float(((sh - s.sh) ** 2).mean())})
    elif vq:
        x = s.sh.reshape(n, -1)
        rng = np.random.default_rng(0)
        samp = x if n <= 4096 else x[rng.choice(n, 4096, replace=False)]
        cb = _kmeans(samp, min(codebook, max(16, n // 4)), seed=1)
        codes = np.empty(n, np.uint16 if len(cb) > 256 else np.uint8)
        for i in range(0, n, 4096):
            codes[i:i + 4096] = ((x[i:i + 4096, None, :] - cb[None]) ** 2).sum(-1).argmin(1)
        sh = cb[codes].reshape(n, k, 3).astype(np.float32)
        tensors.append({"name": "sh.codes", "arr": codes})
        tensors.append({"name": "sh.cb", "arr": cb.astype(np.float16)})
        info.update({"sh_mode": "vq", "codebook": int(len(cb)),
                     "sh_mse": float(((sh - s.sh) ** 2).mean())})
    else:
        flat = s.sh.reshape(n, -1)
        q, lo, st = affine_encode(flat, sh_bits)
        tensors.append({"name": "sh.q", "arr": q, "lo": lo, "step": st})
        sh = affine_decode(q, lo, st).reshape(n, k, 3)
        info.update({"sh_mode": "scalar", "sh_bits": sh_bits,
                     "sh_mse": float(((sh - s.sh) ** 2).mean())})

    payload = sum(int(np.asarray(t["arr"]).nbytes) for t in tensors)
    out = GeoScene(pos, sh, opacity, scale, rot, s.kind, dict(s.meta),
                   {"tensors": tensors, "sh_mode": info["sh_mode"], "sh_shape": [k, 3]})
    info["payload_bytes"] = payload
    info["max_err"] = errs                                 # 逐属性最大绝对误差（m / 无量纲）
    return out, info


def _reconstruct_lowrank(c: dict, n: int, k: int) -> np.ndarray:
    u = affine_decode(c["u"], c["u_lo"], c["u_step"])
    vt = affine_decode(c["vt"], c["vt_lo"], c["vt_step"])
    sv = c["s"].astype(np.float32)
    mu = c["mu"].astype(np.float32)
    return ((u * sv) @ vt + mu).reshape(n, k, 3).astype(np.float32)


# ---------------- 4. 低秩分解 ----------------
def lowrank(s: GeoScene, energy=0.995, min_rank=2, bits=8, **_) -> tuple[GeoScene, dict]:
    """低秩分解：SH 系数矩阵 X(n×k·3) ≈ U·diag(S)·Vᵀ + μ，因子按 bits 量化后存储。

    仅当 SH 维度很高（神经特征 / 高阶表示）时才比向量量化更省字节，由 planner 用代价模型决策。
    """
    n, k = s.n, s.sh.shape[1]
    x = s.sh.reshape(n, -1).astype(np.float32)
    mu = x.mean(0, keepdims=True)
    u, sv, vt = np.linalg.svd(x - mu, full_matrices=False)
    cum = np.cumsum(sv ** 2) / max(float(np.sum(sv ** 2)), 1e-12)
    r = int(max(min_rank, min(len(sv), np.searchsorted(cum, energy) + 1)))
    u, sv, vt = u[:, :r], sv[:r], vt[:r]

    uq, u_lo, u_st = affine_encode(u, bits)
    tq, t_lo, t_st = affine_encode(vt, bits)
    uf, tf = affine_decode(uq, u_lo, u_st), affine_decode(tq, t_lo, t_st)
    rec = ((uf * sv) @ tf + mu).astype(np.float32)
    codec = {"mode": "lowrank", "rank": r, "bits": bits,
             "u": uq, "u_lo": u_lo, "u_step": u_st,
             "vt": tq, "vt_lo": t_lo, "vt_step": t_st,
             "s": sv.astype(np.float16), "mu": mu.reshape(-1).astype(np.float16)}
    out = GeoScene(s.pos, rec.reshape(n, k, 3), s.opacity, s.scale, s.rot, s.kind, dict(s.meta),
                   {"tensors": [], "sh": codec})
    full_b = n * x.shape[1] * 4
    fact_b = uq.nbytes + tq.nbytes + sv.size * 2 + mu.size * 2
    return out, {"stage": "lowrank", "rank": r, "full_rank": int(min(n, x.shape[1])),
                 "factor_ratio": round(fact_b / max(full_b, 1), 4), "factor_bytes": int(fact_b),
                 "sh_mse": float(((rec - x) ** 2).mean()),
                 "rel_err": float(np.linalg.norm(rec - x) / (np.linalg.norm(x) + 1e-12))}


def recover(orig: GeoScene, comp: GeoScene, iters=2, lr=1.0, n_views=2, W=320, H=240, grow=1.0, **_) -> tuple[GeoScene, dict]:
    """恢复微调：以原始场景渲染图为监督，按前向合成权重把像素残差回传给各基元的 SH DC。

    等价于对颜色做若干轮加权 Jacobi 迭代（渲染对颜色是线性的），无需反向传播框架；
    可选 grow 统一放大尺度以填补剪枝造成的覆盖空洞。
    """
    from .raster import rasterize, orbit_cameras, render_eval
    out = comp.subset(np.arange(comp.n))
    if grow and abs(grow - 1.0) > 1e-6:
        out.scale = (out.scale * grow).astype(np.float32)
    cams = orbit_cameras(orig, n_views, W, H)
    tgts = [rasterize(orig, c)[0] for c in cams]                 # 监督图只渲一次
    ev = min(n_views, 2)
    before = render_eval(orig, comp, ev)          # 与管线评测同分辨率，便于对齐
    resid = []
    for _ in range(iters):
        gw = np.zeros((out.n, 3), np.float32)
        ws = np.zeros(out.n, np.float32)
        for cam, tgt in zip(cams, tgts):
            _, _, _, g, w = rasterize(out, cam, target=tgt)
            gw += g
            ws += w
        d = gw / (ws[:, None] + 1e-6)                            # 加权平均残差（目标 − 当前）
        pos = ws[ws > 0]                                         # 覆盖过弱的基元不更新，
        if len(pos):                                             # 否则其残差被背景像素主导 → 白噪
            d[ws < 0.35 * float(np.median(pos))] = 0
        resid.append(round(float(np.abs(d).mean()), 5))
        lim = 0.5 / SH_C0
        out.sh[:, 0, :] = np.clip(out.sh[:, 0, :] + lr * d / SH_C0, -lim, lim)
    after = render_eval(orig, out, ev)
    return out, {"stage": "recover", "iters": iters, "lr": lr, "grow": round(float(grow), 3),
                 "psnr_before": before["psnr_render"], "psnr_after": after["psnr_render"],
                 "ssim_before": before["ssim_render"], "ssim_after": after["ssim_render"],
                 "delta_db": round(after["psnr_render"] - before["psnr_render"], 2),
                 "resid": resid}


REGISTRY = {"prune": prune, "distill": distill, "quantize": quantize, "lowrank": lowrank}


def run_stage(name: str, s: GeoScene, cfg: dict):
    fn = REGISTRY[name]
    kw = {k: v for k, v in cfg.items() if k != "stage"}
    return fn(s, **kw)
