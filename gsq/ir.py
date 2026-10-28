"""GeoScene IR: 统一中间表示 + PLY/LAS 读取 + 合成样例。纯 numpy。"""
from __future__ import annotations
import json
import math
import os
import struct
from dataclasses import dataclass, field
import numpy as np

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
         -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
         -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def sh_basis(dirs: np.ndarray, deg: int = 3) -> np.ndarray:
    """实球谐基函数（含 C0），约定与 3DGS / diff-gaussian-rasterization 完全一致。

    dirs: (N,3) 单位向量，方向为「相机 → 基元」；返回 (N,(deg+1)**2)。
    """
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    out = [np.full(x.shape, SH_C0, np.float32)]
    if deg >= 1:
        out += [-SH_C1 * y, SH_C1 * z, -SH_C1 * x]
    if deg >= 2:
        out += [SH_C2[0] * x * y, SH_C2[1] * y * z, SH_C2[2] * (2 * z * z - x * x - y * y),
                SH_C2[3] * x * z, SH_C2[4] * (x * x - y * y)]
    if deg >= 3:
        out += [SH_C3[0] * y * (3 * x * x - y * y), SH_C3[1] * x * y * z,
                SH_C3[2] * y * (4 * z * z - x * x - y * y),
                SH_C3[3] * z * (2 * z * z - 3 * x * x - 3 * y * y),
                SH_C3[4] * x * (4 * z * z - x * x - y * y), SH_C3[5] * z * (x * x - y * y),
                SH_C3[6] * x * (x * x - 3 * y * y)]
    return np.stack(out[: (deg + 1) ** 2], 1).astype(np.float32)


def sh_color(sh: np.ndarray, dirs: np.ndarray | None = None) -> np.ndarray:
    """SH 系数 → RGB。dirs 为 None 时退化为 DC（视角无关）。"""
    if dirs is None:
        return np.clip(0.5 + SH_C0 * sh[:, 0, :], 0, 1)
    deg = min(int(round(math.sqrt(sh.shape[1]))) - 1, 3)
    b = sh_basis(np.asarray(dirs, np.float32), deg)
    return np.clip(0.5 + np.einsum("nk,nkc->nc", b[:, :sh.shape[1]], sh), 0, 1)
_PLY_T = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2",
          "int16": "i2", "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4",
          "uint": "u4", "uint32": "u4", "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


@dataclass
class GeoScene:
    """geometry: pos(N,3) float32; sh(N,K,3); opacity(N,); scale(N,3); rot(N,4)。"""
    pos: np.ndarray
    sh: np.ndarray
    opacity: np.ndarray
    scale: np.ndarray
    rot: np.ndarray
    kind: str = "gaussian"                    # gaussian | point
    meta: dict = field(default_factory=dict)
    codec: dict | None = None                 # 量化编码计划（见 stages.quantize），不参与 json

    def __post_init__(self):
        self.pos = np.ascontiguousarray(self.pos, dtype=np.float32)
        self.sh = np.ascontiguousarray(self.sh, dtype=np.float32)
        self.opacity = np.ascontiguousarray(self.opacity, dtype=np.float32)
        self.scale = np.ascontiguousarray(self.scale, dtype=np.float32)
        self.rot = np.ascontiguousarray(self.rot, dtype=np.float32)
        n = len(self.pos)
        if self.sh.ndim == 2:
            self.sh = self.sh.reshape(n, 1, 3)
        assert len(self.sh) == len(self.opacity) == len(self.scale) == len(self.rot) == n

    @property
    def n(self) -> int:
        return len(self.pos)

    @property
    def sh_degree(self) -> int:
        return int(round(math.isqrt(self.sh.shape[1]))) - 1

    def colors(self, dirs: np.ndarray | None = None) -> np.ndarray:
        """基元颜色；给 dirs 时按视角计算（视角相关 SH），否则取 DC。"""
        return sh_color(self.sh, dirs)

    def subset(self, idx) -> "GeoScene":
        return GeoScene(self.pos[idx], self.sh[idx], self.opacity[idx], self.scale[idx],
                        self.rot[idx], self.kind, dict(self.meta))   # codec 索引失效，不继承

    def bounds(self):
        lo, hi = self.pos.min(0), self.pos.max(0)
        return lo, hi

    def with_degree(self, deg: int) -> "GeoScene":
        k = (deg + 1) ** 2
        if k >= self.sh.shape[1]:
            return self
        sh = self.sh[:, :k, :].copy()
        return GeoScene(self.pos, sh, self.opacity, self.scale, self.rot, self.kind, dict(self.meta))


# ---------------- PLY ----------------
def read_csv(path: str, has_header: bool | None = None) -> GeoScene:
    """CSV/XYZ 点云 → GeoScene（kind=point）。

    列自动识别：x/y/z（大小写不敏感）优先，否则取前 3 个数值列；
    可选 r,g,b（0-255）→ DC 颜色，可选 i（强度）并入颜色亮度。
    """
    import csv as _csv
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for rec in _csv.reader(f):
            if not rec or all(not c.strip() for c in rec):
                continue
            rows.append([c.strip() for c in rec])
    if not rows:
        raise ValueError(f"CSV 为空：{path}")
    header = None
    if has_header is None:
        try:
            [float(v) for v in rows[0][:3]]
        except ValueError:
            header = rows.pop(0)
    elif has_header:
        header = rows.pop(0)
    if not rows:
        raise ValueError(f"CSV 只有表头没有数据：{path}")
    data = np.array([[float(v) for v in r] for r in rows], np.float64)
    if data.shape[1] < 3:
        raise ValueError(f"CSV 至少需要 x,y,z 三列：{path}")
    xi = yi = zi = None
    if header:
        low = [h.lower() for h in header]
        for i, h in enumerate(low):
            if h in ("x", "px") and xi is None:
                xi = i
            elif h in ("y", "py") and yi is None:
                yi = i
            elif h in ("z", "pz") and zi is None:
                zi = i
    if xi is None:                                  # 无表头或未识别 → 前 3 列
        xi, yi, zi = 0, 1, 2
    pos = data[:, [xi, yi, zi]].astype(np.float32)
    n = len(pos)
    sh = np.zeros((n, 1, 3), np.float32)
    ci = None
    if header:
        low = [h.lower() for h in header]
        ci = [low.index(c) for c in ("r", "g", "b") if c in low] or None
        if ci and len(ci) == 3 and max(ci) < data.shape[1]:
            sh[:, 0, :] = (data[:, ci] / 255.0 - 0.5) / SH_C0
        else:
            ci = None
    if ci is None:
        sh[:, 0, :] = (0.5 - 0.5) / SH_C0           # 默认中灰
    lo, hi = pos.min(0), pos.max(0)
    ext = float(np.max(hi - lo)) or 1.0
    scale = np.full((n, 3), ext / 512.0, np.float32)
    return GeoScene(pos, sh, np.ones(n, np.float32), scale,
                    np.tile([1.0, 0, 0, 0], (n, 1)).astype(np.float32),
                    "point", {"source": os.path.basename(path), "sh_degree": 0})


def read_ply(path: str) -> GeoScene:
    with open(path, "rb") as f:
        raw = f.read()
    end = raw.find(b"end_header")
    if end < 0:
        raise ValueError("not a ply file")
    nl = raw.find(b"\n", end) + 1
    header = raw[:nl].decode("ascii", "replace").splitlines()
    fmt, count, props, elem = None, 0, [], None
    for ln in header:
        t = ln.split()
        if not t:
            continue
        if t[0] == "format":
            fmt = t[1]
        elif t[0] == "element":
            elem = t[1]
            if elem == "vertex":
                count = int(t[2])
        elif t[0] == "property" and elem == "vertex":
            if t[1] == "list":
                raise ValueError("list property unsupported in vertex element")
            props.append((t[2], _PLY_T.get(t[1], "f4")))
    if not props:
        raise ValueError("no vertex properties")
    names = [p[0] for p in props]
    if fmt == "ascii":
        rows = raw[nl:].split()
        arr = np.array(rows[: count * len(props)], dtype=np.float64).reshape(count, len(props))
        data = np.zeros(count, dtype=[(n, d) for n, d in props])
        for i, (n, d) in enumerate(props):
            data[n] = arr[:, i]
    else:
        dt = np.dtype([(n, "<" + d) for n, d in props])
        data = np.frombuffer(raw, dtype=dt, count=count, offset=nl)
    return _ply_to_scene(data, names, path)


def _col(names, prefix, k):
    return [names.index(f"{prefix}_{i}") if f"{prefix}_{i}" in names else -1 for i in range(k)]


def _ply_to_scene(data, names, path=None) -> GeoScene:
    pos = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float32)
    n = len(pos)
    gs = "f_dc_0" in names or "opacity" in names
    if gs:
        dc = np.stack([data[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float32)
        rest = [f"f_rest_{i}" for i in range(45) if f"f_rest_{i}" in names]
        if rest:
            r = np.stack([data[c] for c in rest], 1).astype(np.float32)
            m = r.shape[1] // 3
            sh = np.concatenate([dc[:, None, :], r.reshape(n, 3, m).transpose(0, 2, 1)], 1)
        else:
            sh = dc[:, None, :]
        opacity = 1.0 / (1.0 + np.exp(-np.asarray(data["opacity"], dtype=np.float32)))
        scale = np.exp(np.stack([data[f"scale_{i}"] for i in range(3)], 1).astype(np.float32))
        rot = np.stack([data[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)
        kind = "gaussian"
    else:
        sh = np.zeros((n, 1, 3), np.float32)
        cols = [c for c in ("red", "green", "blue") if c in names]
        if len(cols) == 3:
            rgb = np.stack([data[c] for c in cols], 1).astype(np.float32)
            rgb = rgb / 255.0 if rgb.max() > 1.5 else rgb
            sh[:, 0, :] = (rgb - 0.5) / SH_C0
        opacity = np.ones(n, np.float32)
        scale = np.full((n, 3), 0.05, np.float32)
        rot = np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1))
        kind = "point"
    return GeoScene(pos, sh, opacity, scale, rot, kind,
                    {"source": str(path), "format": "ply"})


# ---------------- LAS / LAZ ----------------
# 点格式 → (记录长度, RGB 偏移, 分类偏移)；-1 表示该格式无此字段
_LAS_PT = {0: (20, -1, 15), 1: (28, -1, 15), 2: (26, 20, 15), 3: (34, 28, 15),
           4: (57, -1, 15), 5: (63, 28, 15), 6: (30, -1, 16), 7: (36, 30, 16),
           8: (38, 30, 16), 9: (59, -1, 16), 10: (67, 30, 16)}


def las_header(path: str) -> dict:
    """解析 LAS 头（不改动点数据），返回坐标系参数、点格式与 VLR 摘要。"""
    with open(path, "rb") as f:
        raw = f.read(375)
    if raw[:4] != b"LASF":
        raise ValueError("不是 LAS/LAZ 文件（文件签名非 LASF）")
    major, minor = raw[24], raw[25]
    hsize = struct.unpack_from("<H", raw, 94)[0]
    fmt_raw = raw[104]
    h = {"version": f"{major}.{minor}", "header_size": hsize,
         "offset": struct.unpack_from("<I", raw, 96)[0],
         "n_vlr": struct.unpack_from("<I", raw, 100)[0],
         "point_format": fmt_raw & 0x7F, "reclen": struct.unpack_from("<H", raw, 105)[0],
         "scale": list(struct.unpack_from("<3d", raw, 131)),
         "offset_xyz": list(struct.unpack_from("<3d", raw, 155)),
         "bounds": list(struct.unpack_from("<6d", raw, 179)),
         "laszip_flag": bool(fmt_raw & 0x80)}
    n = struct.unpack_from("<I", raw, 107)[0]
    if (major, minor) >= (1, 4) and len(raw) >= 255:
        n14 = struct.unpack_from("<Q", raw, 247)[0]
        n = n14 or n
    h["n_points"] = int(n)
    # VLR 扫描：识别 laszip 压缩
    with open(path, "rb") as f:
        f.seek(hsize)
        for _ in range(min(h["n_vlr"], 64)):
            vh = f.read(54)
            if len(vh) < 54:
                break
            uid = vh[2:18].rstrip(b"\x00").decode("latin1")
            rid, vlen = struct.unpack_from("<HH", vh, 18)
            if uid.startswith("laszip") or rid == 22204:
                h["laszip_flag"] = True
            f.seek(vlen, 1)
    return h


def _las_decompress(path: str) -> str:
    """调用外部 LASzip/PDAL 解压 .laz 到临时 .las；无可用工具时报清晰错误。"""
    import shutil
    import subprocess
    import tempfile
    out = os.path.join(tempfile.mkdtemp(prefix="gsq_laz_"), "decompressed.las")
    for exe, cmd in (("laszip", ["laszip", "-i", path, "-o", out]),
                     ("pdal", ["pdal", "translate", path, out])):
        if shutil.which(exe):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and os.path.isfile(out) and os.path.getsize(out) > 0:
                return out
            raise RuntimeError(f"{exe} 解压失败：{(r.stderr or r.stdout).strip()[:200]}")
    raise RuntimeError(
        "该文件是 LAZ（LASzip 压缩），纯 Python 无法解码。请先转换为 LAS：\n"
        "  LASzip:  laszip -i input.laz -o output.las\n"
        "  PDAL:    pdal translate input.laz output.las\n"
        "  CloudCompare / QGIS 亦可直接另存为 LAS")


def _height_colors(z: np.ndarray) -> np.ndarray:
    """无 RGB 时按高程着色（地形色带），比全黑更利于目视判读。"""
    t = (z - z.min()) / max(float(z.max() - z.min()), 1e-9)
    stops = np.array([0.0, 0.25, 0.55, 0.78, 1.0])
    ramp = np.array([[0.16, 0.26, 0.42], [0.20, 0.45, 0.32], [0.72, 0.74, 0.36],
                     [0.60, 0.44, 0.30], [0.94, 0.94, 0.90]])
    return np.stack([np.interp(t, stops, ramp[:, c]) for c in range(3)], 1).astype(np.float32)


def read_las(path: str, max_points: int | None = None) -> GeoScene:
    """读取 LAS 1.0~1.4 点云（点格式 0~10）。LAZ 经外部 LASzip/PDAL 解压后读取。

    坐标按头部的 scale/offset 反算为真实世界坐标；RGB 自动识别 8/16 位；
    无颜色时按高程着色。点尺寸按平面密度估算，供后续体素下采样与渲染使用。
    """
    h = las_header(path)
    if h["laszip_flag"]:
        path = _las_decompress(path)
        h = las_header(path)
    n, reclen = h["n_points"], h["reclen"]
    if n == 0:                                    # 头部计数缺失（流式写入）→ 按文件长度推算
        n = max(0, (os.path.getsize(path) - h["offset"]) // max(reclen, 1))
    if n == 0:
        raise ValueError("LAS 文件不含点记录")
    rgb_off, cls_off = _LAS_PT.get(h["point_format"], (reclen, -1, -1))[1:]
    stride = 1
    if max_points and n > max_points:              # 超大文件等距采样，保证内存可控
        stride = int(np.ceil(n / max_points))
        n_read = n // stride
    else:
        n_read = n
    with open(path, "rb") as f:
        f.seek(h["offset"])
        buf = f.read(n_read * stride * reclen)
    rec = np.frombuffer(buf, np.uint8, n_read * stride * reclen).reshape(n_read * stride, reclen)
    if stride > 1:
        rec = np.ascontiguousarray(rec[::stride][:n_read])
    xyz = np.ascontiguousarray(rec[:, 0:12]).view("<i4").reshape(-1, 3).astype(np.float64)
    pos = (xyz * np.array(h["scale"]) + np.array(h["offset_xyz"])).astype(np.float32)
    m = len(pos)

    rgb = None
    if rgb_off >= 0 and reclen >= rgb_off + 6:
        rgb = np.ascontiguousarray(rec[:, rgb_off:rgb_off + 6]).view("<u2").reshape(-1, 3).astype(np.float32)
        rgb = rgb / (65535.0 if rgb.max() > 255 else 255.0)
        if float(rgb.max()) <= 0.004:              # 全 0 视为无颜色
            rgb = None
    if rgb is None:
        rgb = _height_colors(pos[:, 2])

    sh = np.empty((m, 1, 3), np.float32)
    sh[:, 0, :] = (np.clip(rgb, 0, 1) - 0.5) / SH_C0
    ext = pos.max(0) - pos.min(0)
    area = float(ext[0] * ext[1]) if ext[0] > 0 and ext[1] > 0 else 0.0
    spacing = math.sqrt(area / max(m, 1)) if area > 0 else float(np.max(ext)) / max(np.cbrt(m), 1.0)
    spacing = max(spacing, 1e-4)
    meta = {"source": str(path), "format": "las", "las_version": h["version"],
            "point_format": h["point_format"], "scale": h["scale"],
            "offset_xyz": h["offset_xyz"], "has_color": rgb_off >= 0,
            "spacing": round(spacing, 5)}
    if cls_off >= 0 and reclen > cls_off:
        cls = rec[:, cls_off].astype(np.uint8)
        if cls.max() > 0:
            vals, cnts = np.unique(cls, return_counts=True)
            meta["classification"] = {int(v): int(c) for v, c in zip(vals, cnts)}
    if stride > 1:
        meta["sampled"] = {"from": n, "stride": stride}
    return GeoScene(pos, sh, np.ones(m, np.float32), np.full((m, 3), spacing * 0.9, np.float32),
                    np.tile(np.array([1, 0, 0, 0], np.float32), (m, 1)), "point", meta)


def write_las(path: str, s: GeoScene, scale=0.001) -> int:
    """写出 LAS 1.2 点格式 2（含 RGB），便于与 GIS 工具链互通与往返验证。"""
    pos = np.asarray(s.pos, np.float64)
    off = np.floor(pos.min(0) / 1000.0) * 1000.0
    ijk = np.round((pos - off) / scale).astype(np.int32)
    rgb = np.clip(np.round(s.colors() * 65535), 0, 65535).astype(np.uint16)
    n, reclen = s.n, 26
    hsize, poff = 227, 227
    head = bytearray(hsize)
    head[0:4] = b"LASF"
    head[24], head[25] = 1, 2
    struct.pack_into("<H", head, 94, hsize)
    struct.pack_into("<I", head, 96, poff)
    struct.pack_into("<I", head, 100, 0)
    head[104] = 2                                 # 点格式 2
    struct.pack_into("<H", head, 105, reclen)
    struct.pack_into("<I", head, 107, n)
    struct.pack_into("<3d", head, 131, scale, scale, scale)
    struct.pack_into("<3d", head, 155, *off)
    struct.pack_into("<6d", head, 179, pos[:, 0].max(), pos[:, 0].min(), pos[:, 1].max(),
                     pos[:, 1].min(), pos[:, 2].max(), pos[:, 2].min())
    rec = np.empty((n, reclen), np.uint8)
    rec[:, 0:12] = np.ascontiguousarray(ijk).view(np.uint8).reshape(n, 12)
    rec[:, 12:14] = np.ascontiguousarray(np.zeros((n, 1), np.uint16)).view(np.uint8).reshape(n, 2)
    rec[:, 14] = 0x09                             # 回波序号 1 / 回波总数 1
    rec[:, 15] = 1                                # 分类：未分类
    rec[:, 16] = 0
    rec[:, 17] = 0
    rec[:, 18:20] = np.ascontiguousarray(np.zeros((n, 1), np.uint16)).view(np.uint8).reshape(n, 2)
    rec[:, 20:26] = np.ascontiguousarray(rgb).view(np.uint8).reshape(n, 6)
    with open(path, "wb") as f:
        f.write(bytes(head))
        f.write(rec.tobytes())
    return poff + n * reclen


def write_ply(path: str, s: GeoScene) -> None:
    n, k = s.n, s.sh.shape[1]
    rest = k - 1
    props = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] + \
            [f"f_rest_{i}" for i in range(rest * 3)] + ["opacity"] + \
            [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    hdr = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"] + \
          [f"property float {p}" for p in props] + ["end_header", ""]
    dt = np.dtype([(p, "<f4") for p in props])
    d = np.zeros(n, dt)
    for i, c in enumerate("xyz"):
        d[c] = s.pos[:, i]
    dc = s.sh[:, 0, :]
    for i in range(3):
        d[f"f_dc_{i}"] = dc[:, i]
    if rest:
        r = s.sh[:, 1:, :].transpose(0, 2, 1).reshape(n, -1)
        for i in range(rest * 3):
            d[f"f_rest_{i}"] = r[:, i]
    op = np.clip(s.opacity, 1e-6, 1 - 1e-6)
    d["opacity"] = np.log(op / (1 - op))
    ls = np.log(np.maximum(s.scale, 1e-8))
    for i in range(3):
        d[f"scale_{i}"] = ls[:, i]
    for i in range(4):
        d[f"rot_{i}"] = s.rot[:, i]
    with open(path, "wb") as f:
        f.write(("\n".join(hdr)).encode("ascii"))
        f.write(d.tobytes())


# ---------------- 合成样例 ----------------
def make_sample(n_ground=120, n_box=28, seed=7) -> GeoScene:
    """合成一个"城市街区"：地面 + 3 栋楼 + 少量漂浮噪声，用于闭环自测。"""
    rng = np.random.default_rng(seed)
    pts, cols = [], []

    g = np.linspace(-20, 20, n_ground)
    gx, gy = np.meshgrid(g, g)
    ground = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1)
    ground += rng.normal(0, 0.05, ground.shape)
    pts.append(ground)
    cols.append(np.tile([0.42, 0.40, 0.36], (len(ground), 1)))

    boxes = [(-8, -6, 8, 14, [0.75, 0.55, 0.45]), (6, 4, 6, 18, [0.55, 0.62, 0.7]),
             (-2, 10, 5, 9, [0.8, 0.75, 0.55])]
    for cx, cy, w, h, c in boxes:
        for face in range(5):
            u = rng.uniform(-w / 2, w / 2, n_box * n_box)
            v = rng.uniform(-w / 2, w / 2, n_box * n_box)
            zz = rng.uniform(0, h, n_box * n_box)
            if face == 0:
                p = np.stack([cx + u, np.full_like(u, cy - w / 2), zz], 1)
            elif face == 1:
                p = np.stack([cx + u, np.full_like(u, cy + w / 2), zz], 1)
            elif face == 2:
                p = np.stack([np.full_like(u, cx - w / 2), cy + v, zz], 1)
            elif face == 3:
                p = np.stack([np.full_like(u, cx + w / 2), cy + v, zz], 1)
            else:
                p = np.stack([cx + u, cy + v, np.full_like(u, float(h))], 1)
            pts.append(p)
            cols.append(np.tile(c, (len(p), 1)))
    noise = rng.uniform(-20, 20, (240, 3))
    noise[:, 2] = rng.uniform(6, 26, 240)
    pts.append(noise)
    cols.append(np.tile([0.9, 0.9, 0.9], (240, 1)))

    pos = np.concatenate(pts).astype(np.float32)
    col = np.clip(np.concatenate(cols) + rng.normal(0, 0.03, (len(pos), 3)), 0, 1).astype(np.float32)
    n = len(pos)
    deg = 3
    k = (deg + 1) ** 2
    sh = np.zeros((n, k, 3), np.float32)
    sh[:, 0, :] = (col - 0.5) / SH_C0
    sh[:, 1:, :] = rng.normal(0, 0.02, (n, k - 1, 3)).astype(np.float32)
    scale = np.abs(rng.normal(0.16, 0.04, (n, 3))).astype(np.float32)
    opacity = np.clip(rng.normal(0.85, 0.06, n), 0.3, 0.99).astype(np.float32)
    rot = np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1))
    return GeoScene(pos, sh, opacity, scale, rot, "gaussian", {"source": "synthetic"})


# ---------------- 空间索引 ----------------
def voxel_keys(pos: np.ndarray, voxel: float, origin=None):
    lo = pos.min(0) if origin is None else np.asarray(origin, np.float32)
    idx = np.floor((pos - lo) / max(voxel, 1e-9)).astype(np.int64)
    span = idx.max(0) + 1
    key = (idx[:, 0] * span[1] + idx[:, 1]) * span[2] + idx[:, 2]
    return key, idx


def voxel_counts(pos: np.ndarray, voxel: float) -> np.ndarray:
    key, _ = voxel_keys(pos, voxel)
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return cnt[inv].astype(np.int32)


def head_tail(pos: np.ndarray, voxel: float):
    """每个体素保留重要度最高的代表点（按传入顺序取首个）。返回代表点索引。"""
    key, _ = voxel_keys(pos, voxel)
    _, first = np.unique(key, return_index=True)
    return np.sort(first)
