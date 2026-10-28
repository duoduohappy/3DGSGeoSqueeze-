"""统一量化算子：所有通道共用的定点编码/解码与误差上界。

此前 GeoScene 通道（pos/opacity/scale/rot/SH）与网格几何（glTF KHR_mesh_quantization）
各写一套 round/clip，误差口径也不一致，导致"量化误差"在不同产物之间不可比。
本模块把两种码型收敛到一处：

- **affine**：任意 ``[lo, hi]`` → ``[0, 2^bits-1]``，反解 ``x = q·step + lo``。
  适合数值范围未知的属性（位置、不透明度、尺度、旋转、SH）。
- **normalized**：固定区间 → 满量程整数，供 glTF normalized accessor
  （浏览端按 ``q/levels·extent + lo`` 还原）。适合已知包围盒的几何坐标、UV。

两者共同约定 ``step = extent/levels``，**误差上界 = step/2**（四舍五入固有上界），
评测侧统一用 :func:`max_abs_error` / :func:`roundtrip_bound` 报告，不再各估一套。
"""
from __future__ import annotations

import numpy as np

MAX_U16 = 65535
MAX_U8 = 255


def levels(bits: int) -> int:
    """bits 位无符号整数的量级数（8→255，16→65535）。"""
    if not 1 <= int(bits) <= 16:
        raise ValueError(f"量化位宽需在 1..16 之间，收到 {bits}")
    return 2 ** int(bits) - 1


def dtype_for(bits: int):
    return np.uint8 if int(bits) <= 8 else np.uint16


def affine_encode(x: np.ndarray, bits: int) -> tuple[np.ndarray, float, float]:
    """affine 量化：返回 ``(q, lo, step)``；step=0 表示常量输入（q 全 0）。"""
    x = np.asarray(x)
    lv = levels(bits)
    if x.size == 0:
        return np.zeros(x.shape, dtype_for(bits)), 0.0, 0.0
    lo, hi = float(x.min()), float(x.max())
    step = (hi - lo) / lv if hi > lo else 0.0
    if step <= 0.0:
        return np.zeros(x.shape, dtype_for(bits)), lo, 0.0
    q = np.round((x - lo) / step).clip(0, lv).astype(dtype_for(bits))
    return q, lo, float(step)


def affine_decode(q: np.ndarray, lo: float, step: float) -> np.ndarray:
    q = np.asarray(q)
    if not step:
        return np.full(q.shape, lo, np.float32)
    return (q.astype(np.float32) * step + lo).astype(np.float32)


def _extent_arr(extent):
    """extent 可为标量或逐轴数组；退化轴（0/近 0）按 1.0 处理避免除零。"""
    ext = np.asarray(extent, np.float64)
    return np.where(np.abs(ext) < 1e-12, 1.0, ext)


def normalized_step(extent, lv: int = MAX_U16):
    """定点码的格距：extent/lv（标量 extent 返回 float，逐轴 extent 返回数组）。"""
    step = _extent_arr(extent) / float(lv)
    return float(step) if np.isscalar(extent) or np.ndim(extent) == 0 else step


def normalized_encode(x: np.ndarray, lo, extent, lv: int = MAX_U16) -> np.ndarray:
    """定点编码：把 ``[lo, lo+extent]`` 映射到 ``[0, lv]``（glTF normalized 语义）。"""
    x = np.asarray(x, np.float64)
    q = np.clip(np.round((x - np.asarray(lo, np.float64)) / _extent_arr(extent) * lv), 0, lv)
    return q.astype(np.uint8 if lv <= MAX_U8 else np.uint16)


def normalized_decode(q: np.ndarray, lo, extent, lv: int = MAX_U16) -> np.ndarray:
    return np.asarray(q, np.float64) / float(lv) * _extent_arr(extent) + np.asarray(lo, np.float64)


def max_abs_error(orig: np.ndarray, recon: np.ndarray) -> float:
    """逐分量最大绝对误差（统一口径，供 tileset / 网格 / 场景三条通道报告）。"""
    a, b = np.asarray(orig, np.float64), np.asarray(recon, np.float64)
    return float(np.abs(a - b).max()) if a.size else 0.0


def roundtrip_bound(step: float) -> float:
    """给定格距的理论误差上界（round 的固有上界）。"""
    return float(step) / 2.0


def practical_bound(step: float, scale: float = 0.0, extent: float | None = None) -> float:
    """实际可达上界：``step/2`` + 4 ULP(量级)。

    量化链路有两处 float32 运算会额外引入误差：编码期 ``(x - lo)`` 的减法，
    与解码期 ``q·step + lo`` 的回代，各自 ≤2 ULP。量级取 max(|scale|, extent)——
    减法结果的量级是 extent，只看坐标绝对值会低估（26 处的 4 ULP ≈ 7.6e-6）。
    断言与验收都该用这个上界，用 ``step/2`` 卡会误判。
    """
    mag = max(abs(float(scale)), abs(float(extent)) if extent else 0.0, 1e-30)
    ulp = float(np.spacing(np.float32(mag)))
    return roundtrip_bound(step) + 4.0 * ulp


def affine_roundtrip(x: np.ndarray, bits: int) -> tuple[np.ndarray, float, float, float]:
    """编码→解码→误差：返回 ``(recon, step, max_err, bound)``，测试与评测共用。"""
    q, lo, step = affine_encode(x, bits)
    rec = affine_decode(q, lo, step)
    return rec, step, max_abs_error(x, rec), roundtrip_bound(step)
