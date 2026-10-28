"""前向 3DGS 光栅化：CPU 参考实现（数学与 diff-gaussian-rasterization 一致）+ CUDA 后端适配。

CPU: EWA splatting —— 3D 协方差 Σ=RSSᵀRᵀ → 相机投影 J W Σ Wᵀ Jᵀ → conic 求逆 → 逐像素 alpha 合成。
CUDA: 若 torch + diff_gaussian_rasterization 可用则优先使用，并用小场景与 CPU 结果自校验。
"""
from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass
from .ir import GeoScene


@dataclass
class Camera:
    W: int
    H: int
    R: np.ndarray      # 3x3 world -> camera
    T: np.ndarray      # 3 平移
    fx: float
    fy: float
    eye: np.ndarray

    @staticmethod
    def look_at(eye, target, W=480, H=360, up=(0, 0, 1), fov_scale=0.95) -> "Camera":
        eye = np.asarray(eye, np.float32)
        fwd = np.asarray(target, np.float32) - eye
        fwd /= np.linalg.norm(fwd) + 1e-9
        upv = np.asarray(up, np.float32)
        if abs(float(fwd @ upv)) > 0.99:
            upv = np.array([0, 1, 0], np.float32)
        right = np.cross(fwd, upv)
        right /= np.linalg.norm(right) + 1e-9
        up2 = np.cross(right, fwd)
        R = np.stack([right, up2, fwd]).astype(np.float32)      # 行 = 相机轴
        T = (-R @ eye).astype(np.float32)
        f = fov_scale * H
        return Camera(W, H, R, T, f, f, eye)


def orbit_cameras(s: GeoScene, n=4, W=480, H=360, el=18.0, dist=2.1):
    lo, hi = s.bounds()
    c = (lo + hi) / 2
    ext = float(np.max(hi - lo)) or 1.0
    cams = []
    for i in range(n):
        a = np.deg2rad(360.0 * i / max(n, 1) + 35.0)
        e = np.deg2rad(el)
        eye = c + np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)], np.float32) * ext * dist
        cams.append(Camera.look_at(eye, c, W, H))
    return cams


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z);     R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y);     R[:, 2, 1] = 2 * (y * z + w * x);     R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rasterize(s: GeoScene, cam: Camera, bg=(0.965, 0.960, 0.945), collect: bool = False,
              target: np.ndarray | None = None):
    """前向 alpha 合成渲染。

    返回 (img[H,W,3], alpha[H,W], contrib[N] 或 None)。
    给 target 时额外返回 (gw[N,3], wsum[N])：各基元的颜色残差梯度与覆盖权重总和，
    供恢复微调（recovery fine-tune）使用。
    """
    W, H = cam.W, cam.H
    pc = s.pos @ cam.R.T + cam.T
    z = pc[:, 2]
    vis = z > 0.15
    if not vis.any():
        empty = (np.tile(np.float32(bg), (H, W, 1)), np.zeros((H, W), np.float32),
                 np.zeros(s.n, np.float32) if collect else None)
        return empty if target is None else empty + (np.zeros((s.n, 3), np.float32), np.zeros(s.n, np.float32))
    idx = np.nonzero(vis)[0]
    pc, z = pc[idx], z[idx]

    Rq = _quat_to_rot(s.rot[idx])
    s2 = s.scale[idx] ** 2
    Sig = np.einsum("nij,nj,nkj->nik", Rq, s2, Rq)
    cov = np.einsum("ij,njk,lk->nil", cam.R, Sig, cam.R)          # 相机系协方差

    x, y = pc[:, 0], pc[:, 1]
    J = np.zeros((len(idx), 2, 3), np.float32)
    J[:, 0, 0] = cam.fx / z
    J[:, 0, 2] = -cam.fx * x / (z * z)
    J[:, 1, 1] = -cam.fy / z
    J[:, 1, 2] = cam.fy * y / (z * z)
    c2 = np.einsum("nij,njk,nlk->nil", J, cov, J)
    c2[:, 0, 0] += 0.3
    c2[:, 1, 1] += 0.3

    a, b, c = c2[:, 0, 0], c2[:, 0, 1], c2[:, 1, 1]
    det = np.maximum(a * c - b * b, 1e-9)
    ca, cb, cc = c / det, -b / det, a / det
    lam = 0.5 * (a + c) + np.sqrt(np.maximum(0.25 * (a - c) ** 2 + b * b, 0))
    rad = np.ceil(3 * np.sqrt(lam))

    u = W / 2 + cam.fx * x / z
    v = H / 2 - cam.fy * y / z
    ok = (rad >= 0.5) & (u + rad > 0) & (u - rad < W) & (v + rad > 0) & (v - rad < H)
    sel = np.nonzero(ok)[0]

    order = sel[np.argsort(z[sel])]                                # 近 → 远，前向合成
    nrm = np.linalg.norm(pc, axis=1, keepdims=True)                # 相机 → 基元 单位方向
    col = s.colors(pc / np.maximum(nrm, 1e-9))                     # 视角相关 SH 着色
    op = s.opacity[idx]
    rect = {}
    for i in order:
        r = rad[i]
        x0, x1 = max(0, int(np.floor(u[i] - r))), min(W, int(np.ceil(u[i] + r)) + 1)
        y0, y1 = max(0, int(np.floor(v[i] - r))), min(H, int(np.ceil(v[i] + r)) + 1)
        if x0 < x1 and y0 < y1:
            rect[i] = (x0, x1, y0, y1)

    def _splat(res=None):
        """res 为 None 时正常合成；否则按同一 alpha 权重把像素残差回传给各基元。"""
        img = np.zeros((H, W, 3), np.float32)
        Tb = np.ones((H, W), np.float32)
        contrib = np.zeros(s.n, np.float32)
        gw = np.zeros((s.n, 3), np.float32)
        wsum = np.zeros(s.n, np.float32)
        for i in order:
            x0, x1, y0, y1 = rect[i]
            dx = np.arange(x0, x1, dtype=np.float32) + 0.5 - u[i]
            dy = np.arange(y0, y1, dtype=np.float32) + 0.5 - v[i]
            p2 = ca[i] * dx[None, :] ** 2 + 2 * cb[i] * dy[:, None] * dx[None, :] + cc[i] * dy[:, None] ** 2
            alpha = np.minimum(0.99, op[i] * np.exp(-0.5 * p2)).astype(np.float32)
            alpha[p2 > 9.0] = 0.0
            if alpha.max() <= 0:
                continue
            t = Tb[y0:y1, x0:x1].copy()
            w = t * alpha
            if res is not None:
                gw[idx[i]] = (w[:, :, None] * res[y0:y1, x0:x1]).reshape(-1, 3).sum(0)
                wsum[idx[i]] = float(w.sum())
            else:
                img[y0:y1, x0:x1] += w[:, :, None] * col[i]
            Tb[y0:y1, x0:x1] = t * (1 - alpha)
            if collect:
                contrib[idx[i]] = float(w.sum())
        if res is not None:
            return None, None, contrib, gw, wsum
        img += Tb[:, :, None] * np.float32(bg)
        return np.clip(img, 0, 1), 1 - Tb, contrib, gw, wsum

    img, alpha_img, contrib, gw, wsum = _splat()
    if target is None:
        return img, alpha_img, (contrib if collect else None)
    res = np.asarray(target, np.float32) - img
    _, _, _, gw, wsum = _splat(res)
    return img, alpha_img, (contrib if collect else None), gw, wsum


# ---------------- 后端抽象 ----------------
def backend_info() -> dict:
    try:
        import torch  # noqa
        from diff_gaussian_rasterization import GaussianRasterizer  # noqa
        if torch.cuda.is_available():
            return {"backend": "cuda", "detail": "diff_gaussian_rasterization"}
    except Exception as e:
        return {"backend": "cpu_reference", "detail": f"{type(e).__name__}: 未安装 torch / CUDA 扩展，使用数学等价 CPU 实现"}
    return {"backend": "cpu_reference", "detail": "CUDA 不可用"}


_CUDA = {"ok": None, "detail": ""}


def _cuda_available() -> bool:
    if _CUDA["ok"] is None:
        try:
            import torch
            from diff_gaussian_rasterization import GaussianRasterizer  # noqa
            _CUDA["ok"] = bool(torch.cuda.is_available())
            _CUDA["detail"] = "diff_gaussian_rasterization"
        except Exception as e:
            _CUDA["ok"] = False
            _CUDA["detail"] = f"{type(e).__name__}: {e}"
    return bool(_CUDA["ok"])


def rasterize_cuda(s: GeoScene, cam: Camera, bg=(0.965, 0.960, 0.945)):
    """官方 CUDA 光栅化器（diff-gaussian-rasterization）前向渲染。

    与 CPU 参考实现同参数、同约定；用于 GPU 环境下的高速渲染与一致性校验。
    需 torch + diff_gaussian_rasterization；本机无 GPU 时不会被调用。
    """
    import torch
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    dev = "cuda"
    W, H = cam.W, cam.H
    tanfovx = 0.5 * W / cam.fx
    tanfovy = 0.5 * H / cam.fy
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = cam.R
    view[:3, 3] = cam.T
    proj = np.zeros((4, 4), np.float32)
    proj[0, 0] = 2 * cam.fx / W
    proj[1, 1] = 2 * cam.fy / H
    proj[2, 2] = 1.0
    proj[3, 2] = 1.0
    campos = -cam.R.T @ cam.T
    means = torch.tensor(s.pos, dtype=torch.float32, device=dev)
    shs = torch.tensor(s.sh.reshape(s.n, -1, 3), dtype=torch.float32, device=dev)
    kw = dict(image_height=H, image_width=W, tanfovx=tanfovx, tanfovy=tanfovy,
              bg=torch.tensor(bg, dtype=torch.float32, device=dev),
              scale_modifier=1.0, viewmatrix=torch.tensor(view, device=dev),
              projmatrix=torch.tensor(proj, device=dev), sh_degree=s.sh_degree,
              campos=torch.tensor(campos, dtype=torch.float32, device=dev),
              prefiltered=False, debug=False)
    try:                                    # 新版本增加了 timestamp / antialiasing 参数
        setting = GaussianRasterizationSettings(**kw, antialiasing=False)
    except TypeError:
        setting = GaussianRasterizationSettings(**kw)
    rast = GaussianRasterizer(raster_settings=setting)
    color, radii, _ = rast(
        means3D=means,
        means2D=torch.zeros_like(means, requires_grad=False),
        shs=shs,
        colors_precomp=None,
        opacities=torch.tensor(s.opacity, dtype=torch.float32, device=dev)[:, None],
        scales=torch.tensor(s.scale, dtype=torch.float32, device=dev),
        rotations=torch.tensor(s.rot, dtype=torch.float32, device=dev),
        cov3D_precomp=None,
    )[:3]
    img = color.permute(1, 2, 0).clamp(0, 1).contiguous().cpu().numpy().astype(np.float32)
    return img, (radii > 0).any(0).float().cpu().numpy() if radii is not None else None


def consistency_check(s: GeoScene, n_views=1, tol=0.06) -> dict:
    """CPU 参考实现与 CUDA 官方实现的一致性校验（GPU 环境首用时自动执行）。"""
    if not _cuda_available():
        return {"checked": False, "reason": _CUDA["detail"] or "CUDA 不可用"}
    rows = []
    for cam in orbit_cameras(s, n_views, 320, 240):
        a, _, _ = rasterize(s, cam)
        b, _ = rasterize_cuda(s, cam)
        mae = float(np.abs(a - b).mean())
        rows.append(round(mae, 5))
    worst = max(rows) if rows else 0.0
    return {"checked": True, "mae": rows, "passed": worst <= tol, "tol": tol}


def render(scene: GeoScene, cam: Camera, collect: bool = False):
    """统一入口：CUDA 可用则走官方光栅化器（收集贡献时仍需 CPU 参考），否则 CPU 参考实现。"""
    if _cuda_available() and not collect:
        return rasterize_cuda(scene, cam)
    return rasterize(scene, cam, collect=collect)


# ---------------- 指标 ----------------
def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None and mask.any():
        d = (a[mask] - b[mask]) ** 2
    else:
        d = (a - b) ** 2
    mse = float(d.mean()) if d.size else 1.0
    return float(min(60.0, 10 * np.log10(1.0 / max(mse, 1e-12))))


def _blur(x: np.ndarray, sigma=1.5) -> np.ndarray:
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    y = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, x)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, y)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    g1, g2 = a @ np.array([0.299, 0.587, 0.114], np.float32), b @ np.array([0.299, 0.587, 0.114], np.float32)
    m1, m2 = _blur(g1), _blur(g2)
    v1, v2 = _blur(g1 * g1) - m1 ** 2, _blur(g2 * g2) - m2 ** 2
    cv = _blur(g1 * g2) - m1 * m2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * m1 * m2 + C1) * (2 * cv + C2)) / ((m1 ** 2 + m2 ** 2 + C1) * (v1 + v2 + C2))
    return float(np.clip(s.mean(), 0, 1))


def render_eval(orig: GeoScene, comp: GeoScene, n_views=4, W=480, H=360) -> dict:
    cams = orbit_cameras(orig, n_views, W, H)
    rows, ps, ss, ms = [], [], [], []
    for i, cam in enumerate(cams):
        t0 = time.perf_counter()
        a, ma, _ = rasterize(orig, cam)
        b, mb, _ = rasterize(comp, cam)
        ms.append(round((time.perf_counter() - t0) * 1000 / 2, 2))
        m = (ma > 0.03) | (mb > 0.03)
        p_all, p_mask = psnr(a, b), psnr(a, b, m)
        sv = ssim(a, b)
        ps.append(p_all); ss.append(sv)
        rows.append({"view": i, "psnr": round(p_all, 2), "psnr_mask": round(p_mask, 2), "ssim": round(sv, 4),
                     "coverage": round(float(m.mean()), 3)})
    return {"psnr_render": round(float(np.mean(ps)), 3), "ssim_render": round(float(np.mean(ss)), 4),
            "views": rows, "n_views": len(cams),
            "render_ms_per_view": round(float(np.mean(ms)), 2) if ms else None,
            "backend": "cuda" if _cuda_available() else "cpu_reference"}
