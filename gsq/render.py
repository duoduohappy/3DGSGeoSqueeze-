"""软件光栅化预览：把原始/压缩后场景并排渲染成 PNG（无 GPU 依赖）。"""
from __future__ import annotations
import numpy as np
from .ir import GeoScene

BG = np.array([0.965, 0.960, 0.945], np.float32)


def _camera(s: GeoScene, az=42.0, el=22.0):
    lo, hi = s.bounds()
    c = (lo + hi) / 2
    ext = float(np.max(hi - lo)) or 1.0
    a, e = np.deg2rad(az), np.deg2rad(el)
    eye = c + np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)], np.float32) * ext * 2.1
    fwd = c - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0, 0, 1], np.float32))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return eye, fwd, right, up


def render(s: GeoScene, W=560, H=420, az=42.0, el=22.0) -> np.ndarray:
    from .raster import Camera, rasterize
    lo, hi = s.bounds()
    c = (lo + hi) / 2
    ext = float(np.max(hi - lo)) or 1.0
    a, e = np.deg2rad(az), np.deg2rad(el)
    eye = c + np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)], np.float32) * ext * 2.1
    img, _, _ = rasterize(s, Camera.look_at(eye, c, W, H), bg=tuple(BG))
    return (img * 255).astype(np.uint8)


def compare(orig: GeoScene, comp: GeoScene, path: str, metrics: dict | None = None) -> str:
    from PIL import Image, ImageDraw
    W, H, pad, hdr, ftr = 560, 420, 14, 40, 34
    left = render(orig, W, H)
    right = render(comp, W, H)
    canvas = Image.new("RGB", (W * 2 + pad * 3, H + hdr + ftr + pad * 2), (28, 28, 26))
    canvas.paste(Image.fromarray(left), (pad, hdr + pad))
    canvas.paste(Image.fromarray(right), (W + pad * 2, hdr + pad))
    d = ImageDraw.Draw(canvas)
    lc, rc = (238, 236, 228), (238, 236, 228)
    if metrics:
        x = metrics.get("ratio", 1)
        d.text((pad, 14), f"original  n={metrics.get('n_orig', orig.n):,}  {metrics.get('orig_mb', 0)} MB", fill=lc)
        d.text((W + pad * 2, 14),
               f"compressed  n={metrics.get('n_comp', comp.n):,}  {metrics.get('comp_mb', 0)} MB  x{x}  "
               f"PSNR {metrics.get('color_psnr', 0)} dB", fill=rc)
        d.text((pad, H + hdr + pad * 2 - 4),
               f"voxel IoU {metrics.get('voxel_iou', 0)}   tiles {metrics.get('tiles', 0)}   "
               f"fps proxy {metrics.get('fps_proxy', 0)}", fill=(180, 178, 170))
    d.text((W + pad * 2, H + hdr + pad * 2 - 4), "GeoSqueeze software splat preview", fill=(150, 148, 140))
    canvas.save(path)
    return path
