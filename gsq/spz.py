"""SPZ / .splat 格式通道（设计 §3.1 读取器 + §6.2 SPZ 导出 P1）。

- 写出：SPZ v2（legacy gzip 单流格式，Niantic spz 官方读取器 v1-3 只读路径仍支持）；
- 读取：SPZ v2 / v3（v4 为 ZSTD 多流格式，明确报错而非静默失败）；
- 读取：antimatter15 .splat（32 字节/点定点布局）。

字节级约定对齐 nianticlabs/spz src/cc/load-spz.cc（NGSP_MAGIC=0x5053474e，
fractionalBits=12，scales=log 空间 (x+10)*16，alphas=sigmoid*255，
colors=DC*0.15*255+127.5，SH=×128+128 按桶中心量化，v2 四元数 8-bit xyz）。

GeoScene 约定：opacity 已 sigmoid 激活、scale 线性、rot=(w,x,y,z)、sh 含 DC。
"""
from __future__ import annotations
import gzip
import os
import struct
import numpy as np

from .ir import SH_C0

NGSP_MAGIC = 0x5053474E          # 官方常量（legacy 16 字节头同样使用）
COLOR_SCALE = 0.15               # DC 颜色编码增益（官方 colorScale）
FRAC_BITS = 12                   # 位置定点小数位（官方默认，~0.25mm）

_SQ = 1.0 / (2 ** 0.5)           # smallest-three 归一化幅值分母 sqrt(1/2)


def _u8(x):
    return int(min(255.0, max(0.0, round(float(x)))))


def _q_quant(x: np.ndarray, bucket: int) -> np.ndarray:
    """SH 量化：×128+128 后取整到桶中心，钳 [0,255]（对齐官方 quantizeSH）。"""
    q = np.round(x * 128.0 + 128.0).astype(np.int32)
    q = (q + bucket // 2) // bucket * bucket
    return np.clip(q, 0, 255).astype(np.uint8)


def write_spz(path: str, s, version: int = 2) -> dict:
    """GeoScene → SPZ v2 字节流（gzip）。返回指标信息。

    大坐标场景会自动平移到包围盒中心（24-bit 定点量程 ±2048），
    平移量记录在返回值 offset 中；SH 高于 3 阶的部分丢弃（SPZ 最高 3 阶入 v2）。
    """
    deg = min(s.sh_degree, 3)
    sh = s.sh[:, :(deg + 1) ** 2, :]
    lo, hi = s.bounds()
    center = (lo + hi) / 2
    scale_f = float(1 << FRAC_BITS)
    pos = np.round((s.pos - center) * scale_f).astype(np.int64)
    if not (np.abs(pos) < (1 << 23)).all():
        raise ValueError("位置超出 SPZ 24-bit 定点量程（±2048），且自动居中后仍溢出")

    # positions：24-bit 小端定点，符号在低 3 字节补码里天然成立
    p = (pos & 0xFFFFFF).astype(np.uint32)
    positions = np.empty((s.n, 3, 3), np.uint8)
    for j in range(3):
        positions[:, j, 0] = (p[:, j]) & 0xFF
        positions[:, j, 1] = (p[:, j] >> 8) & 0xFF
        positions[:, j, 2] = (p[:, j] >> 16) & 0xFF

    # alphas：官方存 logit，编码 sigmoid(logit)=激活值；GeoScene 已激活，直接 ×255
    alphas = np.clip(np.round(s.opacity * 255.0), 0, 255).astype(np.uint8)

    # colors：DC 颜色（0.5 + SH_C0·c0）×colorScale
    colors = s.colors() * (COLOR_SCALE * 255.0) + (0.5 * 255.0)
    colors = np.clip(np.round(colors), 0, 255).astype(np.uint8)

    # scales：log 空间 (x+10)*16
    sc = np.clip(np.log(np.maximum(s.scale, 1e-8)), -10.0, 5.9375)
    scales = np.clip(np.round((sc + 10.0) * 16.0), 0, 255).astype(np.uint8).reshape(-1)

    # rotations：v2 = 归一化四元数 xyz，w>0 保证，×127.5+127.5
    q = s.rot / np.maximum(np.linalg.norm(s.rot, axis=1, keepdims=True), 1e-12)
    sign = np.where(q[:, 3] < 0, -127.5, 127.5)[:, None]
    rot = np.clip(np.round(q[:, :3] * sign + 127.5), 0, 255).astype(np.uint8).reshape(-1)

    # SH：去掉 DC 的系数，(coef, channel) 顺序 = sh[:,1:,:].reshape(-1)
    k = sh.shape[1]
    sh_flat = sh[:, 1:, :].reshape(-1) if k > 1 else np.zeros(0, np.float32)
    if len(sh_flat):
        idx = np.arange(sh_flat.size)
        coef = idx % (k - 1)                                   # 0..k-2（1..3 阶系数序号）
        b = np.where(coef < 3, 1 << (8 - 5), 1 << (8 - 4))     # sh1Bits=5, shRestBits=4
        qv = np.round(sh_flat * 128.0 + 128.0).astype(np.int64)
        qv = (qv + b // 2) // b * b
        shq = np.clip(qv, 0, 255).astype(np.uint8)
    else:
        shq = np.zeros(0, np.uint8)

    hdr = struct.pack("<IIIBBBB", NGSP_MAGIC, version, s.n, deg, FRAC_BITS, 0, 0)
    payload = bytearray(hdr)
    payload += positions.tobytes()
    payload += alphas.tobytes()
    payload += colors.tobytes()
    payload += scales.tobytes()
    payload += rot.tobytes()
    payload += shq.tobytes()
    data = _gzip_with_comment(bytes(payload), center)
    with open(path, "wb") as f:
        f.write(data)
    return {"path": os.path.abspath(path), "bytes": len(data), "n": int(s.n),
            "sh_degree": int(deg), "version": int(version),
            "offset": [round(float(v), 3) for v in center]}


def _gzip_with_comment(data: bytes, center: np.ndarray) -> bytes:
    """gzip 压缩，FCOMMENT 头字段携带平移中心（官方读取器忽略注释，不影响兼容性）。"""
    import zlib
    comment = ("gsq_center=" + ",".join(f"{float(v):.4f}" for v in center)).encode()
    hdr = b"\x1f\x8b\x08\x10" + struct.pack("<I", 0) + b"\x00\xff" + comment + b"\x00"
    co = zlib.compressobj(6, zlib.DEFLATED, -zlib.MAX_WBITS)
    body = co.compress(data) + co.flush()
    crc = zlib.crc32(data) & 0xFFFFFFFF
    return hdr + body + struct.pack("<II", crc, len(data) & 0xFFFFFFFF)


def _gzip_read_comment(raw: bytes):
    """读 gzip FCOMMENT 里的 gsq_center（没有则 None）。"""
    if len(raw) < 10 or raw[:2] != b"\x1f\x8b":
        return None
    flg = raw[3]
    if not flg & 0x10:
        return None
    off, i = 10, 0
    if flg & 0x04:                                    # FEXTRA
        xlen = struct.unpack_from("<H", raw, off)[0]
        off += 2 + xlen
    if flg & 0x08:                                    # FNAME
        while raw[off]:
            off += 1
        off += 1
    start = off
    while raw[off]:
        off += 1
    comment = raw[start:off].decode("ascii", "ignore")
    if comment.startswith("gsq_center="):
        return np.array([float(x) for x in comment[11:].split(",")], np.float32)
    return None


def _unpack_v3_rot(rot4: np.ndarray) -> np.ndarray:
    """v3 smallest-three：u32 = iLargest(2bit) | 3×(符号1bit + 9bit 幅值/511×√½)。

    u32 高→低：最高 2 位是最大分量索引，其后三个 10-bit 字段按分量下标升序
    （跳过 iLargest）排列。幅值域 [0, √½/2·√2]，iLargest 分量由归一化补出。
    （每行 3 次迭代的行级循环；万点级耗时可忽略，正确性优先）
    """
    comp = (rot4[:, 0].astype(np.uint32) | (rot4[:, 1].astype(np.uint32) << 8) |
            (rot4[:, 2].astype(np.uint32) << 16) | (rot4[:, 3].astype(np.uint32) << 24))
    q = np.zeros((len(comp), 4), np.float32)
    for r in range(len(comp)):
        c = int(comp[r])
        il = c >> 30
        slot = 0
        acc = 0.0
        for i in range(4):
            if i == il:
                continue
            f = (c >> (10 * (2 - slot))) & 0x3FF
            mag = (f & 0x1FF) / 511.0 * _SQ
            v = -mag if (f >> 9) & 1 else mag
            q[r, i] = v
            acc += v * v
            slot += 1
        q[r, il] = float(np.sqrt(max(0.0, 1.0 - acc)))
    return q


def read_spz(path: str) -> tuple:
    """SPZ v2/v3 → GeoScene。v4（ZSTD 多流）明确报错。返回 (GeoScene, info)。"""
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) >= 4 and raw[:4] == b"NGSP":
        raise ValueError("该文件为 SPZ v4（ZSTD 多流格式），本工具暂只支持 v2/v3 legacy gzip 格式")
    try:
        payload = gzip.decompress(raw)
    except OSError as e:
        raise ValueError(f"不是有效的 .spz 文件（gzip 解压失败）：{e}")
    center = _gzip_read_comment(raw)
    if center is None:
        center = np.zeros(3, np.float32)
    magic, version, n, deg, frac, flags, _r = struct.unpack_from("<IIIBBBB", payload, 0)
    if magic != NGSP_MAGIC:
        raise ValueError(f"SPZ magic 不匹配：0x{magic:08x}")
    if not 1 <= version <= 3:
        raise ValueError(f"不支持的 SPZ 版本：{version}（支持 1-3）")
    if version == 1:
        raise ValueError("SPZ v1（float16 位置）从未正式发布，不支持")
    off = 16
    inv = 1.0 / float(1 << frac)

    def take(cnt):
        nonlocal off
        b = np.frombuffer(payload, np.uint8, cnt, off)
        off += cnt
        return b

    # positions（3 字节/分量，符号扩展）
    pb = take(n * 9).reshape(n, 3, 3).astype(np.uint32)
    fixed = pb[:, :, 0] | (pb[:, :, 1] << 8) | (pb[:, :, 2] << 16)
    fixed = fixed.astype(np.int64)
    fixed = np.where((fixed & 0x800000) != 0, fixed - (1 << 24), fixed)
    pos = (fixed * inv + center[None, :]).astype(np.float32)

    alphas = take(n).astype(np.float32) / 255.0                 # 已激活
    colors = (take(n * 3).astype(np.float32).reshape(n, 3) / 255.0 - 0.5) / COLOR_SCALE
    scale_log = take(n * 3).astype(np.float32).reshape(n, 3) / 16.0 - 10.0
    scale = np.exp(scale_log).astype(np.float32)

    rot_b = take(n * (4 if version == 3 else 3))
    if version == 3:
        rot = _unpack_v3_rot(rot_b.reshape(n, 4))
    else:                                                       # v2：xyz 8-bit，w 由归一化补
        xyz = (rot_b.reshape(n, 3).astype(np.float32) - 127.5) / 127.5
        w = np.sqrt(np.maximum(0.0, 1.0 - (xyz ** 2).sum(1)))
        rot = np.concatenate([xyz, w[:, None]], 1).astype(np.float32)

    k = (deg + 1) ** 2
    sh = np.zeros((n, k, 3), np.float32)
    if k > 1:
        shb = take(n * (k - 1) * 3).reshape(n, k - 1, 3).astype(np.float32)
        sh[:, 1:, :] = ((shb - 128.0) / 128.0).astype(np.float32)
    sh[:, 0, :] = ((colors - 0.5) / SH_C0).astype(np.float32)

    from .ir import GeoScene
    sc = GeoScene(pos, sh, alphas.astype(np.float32), scale, rot, "gaussian",
                  {"source": os.path.basename(path), "spz_version": int(version),
                   "sh_degree": int(deg)})
    return sc, {"n": int(n), "sh_degree": int(deg), "version": int(version), "bytes": len(raw)}


# ---------------- .splat（antimatter15，32 字节/点） ----------------
def read_splat(path: str) -> tuple:
    """.splat → GeoScene：f32 pos×3 + f32 scale×3 + u8 RGBA + u8 quat(wxYZ? 顺序 w,x,y,z)。"""
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) % 32:
        raise ValueError(f".splat 大小不是 32 的倍数：{len(raw)}")
    n = len(raw) // 32
    buf = np.frombuffer(raw, np.uint8, n * 32).reshape(n, 32)
    pos = buf[:, 0:12].copy().view(np.float32).reshape(n, 3)
    scale = buf[:, 12:24].copy().view(np.float32).reshape(n, 3)
    rgba = buf[:, 24:28].astype(np.float32)
    sh = np.zeros((n, 1, 3), np.float32)
    sh[:, 0, :] = ((rgba[:, :3] / 255.0 - 0.5) / SH_C0).astype(np.float32)
    opacity = (rgba[:, 3] / 255.0).astype(np.float32)
    q = (buf[:, 28:32].astype(np.float32) - 128.0) / 128.0
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
    from .ir import GeoScene
    sc = GeoScene(pos, sh, opacity, scale.astype(np.float32), q.astype(np.float32),
                  "gaussian", {"source": os.path.basename(path), "sh_degree": 0})
    return sc, {"n": int(n), "bytes": len(raw), "sh_degree": 0}
