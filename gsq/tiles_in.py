"""3D Tiles 输入侧：读取已有 tileset（glb / pnts / b3dm / cmpt）并按瓦片流式再压缩。

设计要点（城市级可用性）：
1. **逐瓦片流式处理**：解析、压缩、写盘都以单个瓦片为单位，峰值内存只与最大瓦片相关，
   不随城市规模线性增长；扫描统计走同一遍历但不驻留点数据。
2. **坐标精度**：真实 3D Tiles 常为 ECEF（|x|≈6.4e6 m），单独用 float32 会丢到 0.5m 级。
   读入时用 float64 累积 tile.transform，减去全场景原点后存 float32，输出根节点
   `transform` 平移复原 —— 既保住毫米级精度，又让后续量化在局部坐标内进行。
3. **保持 LOD 结构**：输出沿用输入的树形与瓦片边界，逐块重算 geometricError（子层不超过父层），
   因此仍是合法的流式 tileset，可直接被 Cesium 按 SSE 调度。
"""
from __future__ import annotations

import base64
import glob
import json
import math
import os
import struct

import numpy as np

from .ir import SH_C0, GeoScene

# ---------------- 二进制容器解析 ----------------
_CT = {5120: ("i1", 127.0), 5121: ("u1", 255.0), 5122: ("i2", 32767.0),
       5123: ("u2", 65535.0), 5125: ("u4", 4294967295.0), 5126: ("f4", 1.0)}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


_UNSUPPORTED_EXT = {
    "EXT_meshopt_compression": "网格几何压缩（meshopt）",
    "KHR_draco_mesh_compression": "Draco 几何压缩",
}


def _glb_chunks(raw: bytes) -> tuple[dict, bytes]:
    magic, _ver, _total = struct.unpack_from("<III", raw, 0)
    if magic != 0x46546C67:
        raise ValueError("不是 GLB（魔数不匹配）")
    jl, jt = struct.unpack_from("<II", raw, 12)
    if jt != 0x4E4F534A:
        raise ValueError("GLB 首个分块不是 JSON")
    js = json.loads(raw[20:20 + jl])
    # 必需扩展里出现几何压缩扩展时，bufferView 的 byteLength 是「解压后」尺寸，
    # 直接按未压缩长度读取会得到晦涩的 buffer 越界错误 —— 提前给出可行动的提示
    bad = [e for e in (js.get("extensionsRequired") or []) if e in _UNSUPPORTED_EXT]
    if bad:
        raise ValueError(
            "该 glb 使用了 GeoSqueeze 尚不解码的几何压缩扩展："
            + "、".join(f"{e}（{_UNSUPPORTED_EXT[e]}）" for e in bad)
            + "。这类数据（3DBAG、Cesium ion 导出的 meshopt 瓦片等）需先用 "
              "gltf-transform / gltfpack 等工具解压为普通 glb 再压缩；"
              "仅在线渲染不受影响（CesiumJS 原生支持该扩展）。")
    off = 20 + jl
    binary = b""
    if off + 8 <= len(raw):
        bl, bt = struct.unpack_from("<II", raw, off)
        if bt == 0x004E4942:
            binary = raw[off + 8: off + 8 + bl]
    return js, binary


def _accessor(js: dict, binary: bytes, idx: int) -> np.ndarray:
    """读取 accessor → (count, ncomp) 数组；支持 byteStride 交错与 normalized 归一化。"""
    a = js["accessors"][idx]
    if "sparse" in a:
        raise ValueError("暂不支持 sparse accessor")
    dt, denom = _CT[a["componentType"]]
    nc = _NC[a["type"]]
    count = int(a["count"])
    bv = js["bufferViews"][a["bufferView"]]
    base = int(bv.get("byteOffset", 0)) + int(a.get("byteOffset", 0))
    stride = int(bv.get("byteStride", 0)) or np.dtype(dt).itemsize * nc
    if stride == np.dtype(dt).itemsize * nc:
        arr = np.frombuffer(binary, np.dtype(dt), count * nc, base).reshape(count, nc)
    else:                                       # 交错布局 → 逐行取
        out = np.empty((count, nc), dt)
        for i in range(count):
            off = base + i * stride
            out[i] = np.frombuffer(binary, np.dtype(dt), nc, off)
        arr = out
    arr = arr.astype(np.float32 if dt == "f4" else np.float64)
    if a.get("normalized") and denom > 1:
        arr = arr / denom
    return np.ascontiguousarray(arr)


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], np.float64).reshape(4, 4).T   # 列主序
    m = np.eye(4)
    if "scale" in node:
        m[:3, :3] = np.diag(np.asarray(node["scale"], np.float64))
    if "rotation" in node:
        x, y, z, w = [float(v) for v in node["rotation"]]
        r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        m[:3, :3] = r @ m[:3, :3]
    if "translation" in node:
        m[:3, 3] = np.asarray(node["translation"], np.float64)
    return m


def _glb_points(path: str) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """读 GLB 瓦片 → (pos float64, rgb 0..1 或 None, extras)。点图元取全部；三角面取顶点采样。"""
    raw = open(path, "rb").read()
    js, binary = _glb_chunks(raw)
    nodes = js.get("nodes", [{}])
    node = nodes[0] if nodes else {}
    nm = _node_matrix(node)
    imgs = js.get("images", [])
    pts, cols, mesh = [], [], False
    for msh in js.get("meshes", []):
        for prim in msh.get("primitives", []):
            at = prim.get("attributes", {})
            if "POSITION" not in at:
                continue
            p = _accessor(js, binary, at["POSITION"]).astype(np.float64)
            mode = prim.get("mode", 4)
            if mode == 4 and "indices" in prim:          # 三角面 → 顶点去重（点化采样）
                idx = _accessor(js, binary, prim["indices"]).astype(np.int64).ravel()
                p = p[np.unique(idx)]
                mesh = True
            elif mode != 0:
                mesh = True
            p = (nm @ np.column_stack([p, np.ones(len(p))]).T)[:3].T
            pts.append(p)
            if "COLOR_0" in at:
                c = _accessor(js, binary, at["COLOR_0"])
                if c.shape[1] == 3:
                    cols.append(c.astype(np.float32))
                elif c.shape[1] == 4:                    # 预乘 alpha → 还原
                    a = np.clip(c[:, 3:4], 1e-6, 1)
                    cols.append(np.clip(c[:, :3] / a, 0, 1).astype(np.float32))
            else:
                cols.append(np.full((len(p), 3), 0.7, np.float32))
    if not pts:
        return np.zeros((0, 3), np.float64), None, {"mesh_sampled": mesh}
    pos = np.vstack(pts)
    rgb = np.vstack(cols) if cols else None
    extras = {"mesh_sampled": mesh, "materials": len(js.get("materials", [])),
              "images": len(imgs)}
    prim = js.get("meshes", [{}])[0].get("primitives", [{}])[0] if js.get("meshes") else {}
    if prim.get("extras"):
        extras.update(prim["extras"])
    return pos, rgb, extras


def _ft_json(raw: bytes, off: int) -> tuple[dict, int]:
    """读取 3D Tiles 1.0 的 Feature Table JSON（4 字节对齐）。"""
    ln = struct.unpack_from("<I", raw, off)[0]
    js = json.loads(raw[off + 4: off + 4 + ln].decode("utf-8").rstrip(" \t\r\n\x00"))
    return js, off + 4 + ln


def _pnts_points(path: str) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """pnts：28 字节头（含 batch table 长度）→ FT JSON → FT 二进制 → BT → 点二进制。"""
    raw = open(path, "rb").read()
    if raw[:4] != b"pnts":
        raise ValueError("不是 pnts")
    _m, _v, _bl, ftj_l, ftb_l, btj_l, btb_l = struct.unpack_from("<4sIIIIII", raw, 0)
    ftj = json.loads(raw[28:28 + ftj_l].decode("utf-8").rstrip(" \t\r\n\x00"))
    bin_off = 28 + ftj_l                        # 布局：头28 | FT JSON | FT 二进制 | BT JSON | BT 二进制
    ft = ftj.get("POINTS_LENGTH")
    n = int(ft)
    rtc = ftj.get("RTC_CENTER")
    if "POSITION_QUANTIZED" in ftj:               # u16 体量化（本工具产出）
        qo = np.asarray(ftj["QUANTIZED_VOLUME_OFFSET"], np.float64)
        qs = np.asarray(ftj["QUANTIZED_VOLUME_SCALE"], np.float64)
        po = int(ftj["POSITION_QUANTIZED"]["byteOffset"]) + bin_off
        q = np.frombuffer(raw, "<u2", n * 3, po).reshape(n, 3).astype(np.float64)
        pos = q / 65535.0 * qs + qo
    else:                                          # f32 直存
        po = int(ftj.get("POSITION", {}).get("byteOffset", 0)) + bin_off
        pos = np.frombuffer(raw, "<f4", n * 3, po).reshape(n, 3).astype(np.float64)
    if rtc:
        pos += np.asarray(rtc, np.float64)
    rgb = None
    if "RGB" in ftj:
        o = int(ftj["RGB"]["byteOffset"]) + bin_off
        if "RGB565" in str(ftj["RGB"].get("componentType", "")):
            v = np.frombuffer(raw, "<u2", n, o)
            rgb = np.stack([((v >> 11) & 31) / 31.0, ((v >> 5) & 63) / 63.0, (v & 31) / 31.0], 1)
        else:
            c = np.frombuffer(raw, np.uint8, n * 3, o).reshape(n, 3).astype(np.float32)
            rgb = c / 255.0
    return pos, (rgb.astype(np.float32) if rgb is not None else None), {"format": "pnts"}


def _b3dm_points(path: str) -> tuple[np.ndarray, np.ndarray | None, dict]:
    raw = open(path, "rb").read()
    if raw[:4] != b"b3dm":
        raise ValueError("不是 b3dm")
    _magic, _ver, _blen, ftj_l, ftb_l, btj_l, btb_l = struct.unpack_from("<4sIIIIII", raw, 0)
    off = 28 + ftj_l + ftb_l + btj_l + btb_l
    glb = raw[off:]
    tmp = os.path.join(os.path.dirname(path), f"._gsq_{os.getpid()}_{os.path.basename(path)}.glb")
    with open(tmp, "wb") as f:
        f.write(glb)
    try:
        pos, rgb, ex = _glb_points(tmp)
    finally:
        os.remove(tmp)
    ex["format"] = "b3dm"
    return pos, rgb, ex


def _cmpt_points(path: str) -> tuple[np.ndarray, np.ndarray | None, dict]:
    raw = open(path, "rb").read()
    if raw[:4] != b"cmpt":
        raise ValueError("不是 cmpt")
    _m, _v, total, n_inner = struct.unpack_from("<4sIII", raw, 0)
    lens = list(struct.unpack_from(f"<{n_inner}I", raw, 16))
    off = 16 + 4 * n_inner
    P, C = [], []
    for ln in lens:
        chunk = raw[off:off + ln]
        off += ln
        ext = chunk[:4].decode("latin1")
        tmp = os.path.join(os.path.dirname(path), f"._gsq_{os.getpid()}_{ext}.{ext}")
        with open(tmp, "wb") as f:
            f.write(chunk)
        try:
            p, c, _ = read_content(tmp)
        finally:
            os.remove(tmp)
        P.append(p)
        if c is not None:
            C.append(c)
    pos = np.vstack(P) if P else np.zeros((0, 3))
    rgb = np.vstack(C) if C and len(C) == len(P) else None
    return pos, rgb, {"format": "cmpt", "inner": n_inner}


def read_content(path: str) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """按扩展名分派解析单个瓦片内容 → (pos float64, rgb 0..1 | None, 元信息)。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        return _glb_points(path)
    if ext == ".pnts":
        return _pnts_points(path)
    if ext == ".b3dm":
        return _b3dm_points(path)
    if ext == ".cmpt":
        return _cmpt_points(path)
    if ext in (".i3dm",):
        raise ValueError("i3dm（实例化模型）暂不支持，请先转换为 b3dm/glb 点云")
    raise ValueError(f"不支持的瓦片内容类型：{ext}")


# ---------------- tileset 遍历 ----------------
def tile_matrix(t: dict) -> np.ndarray:
    m = t.get("transform")
    if not m:
        return np.eye(4)
    return np.asarray(m, np.float64).reshape(4, 4).T


def iter_tiles(ts: dict, root_dir: str = ""):
    """深度优先遍历瓦片树，产出 (tile, world_matrix, 内容绝对路径或 None, depth)。"""
    def walk(t, parent, depth):
        world = parent @ tile_matrix(t)
        uri = (t.get("content") or {}).get("uri")
        full = os.path.join(root_dir, uri) if uri else None
        yield t, world, full, depth
        for c in t.get("children", []):
            yield from walk(c, world, depth + 1)
    yield from walk(ts["root"], np.eye(4), 0)


def _is_ours(ts: dict) -> bool:
    return bool((ts.get("extras") or {}).get("geosqueeze"))


def yup_to_zup() -> np.ndarray:
    """glTF 规范为 +Y 上，3D Tiles 为 +Z 上：绕 X 轴 +90°，(x,y,z)→(x,−z,y)。"""
    return np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], np.float64)


def load_tileset(path: str) -> tuple[dict, str, bool]:
    p = os.path.join(path, "tileset.json") if os.path.isdir(path) else path
    if not os.path.isfile(p):
        raise ValueError(f"找不到 tileset.json：{path}")
    with open(p, encoding="utf-8") as f:
        ts = json.load(f)
    if "root" not in ts:
        raise ValueError("tileset.json 缺少 root")
    return ts, os.path.dirname(os.path.abspath(p)), _is_ours(ts)


def scan_tileset(path: str, max_tiles: int | None = None) -> dict:
    """只扫描不驻留：统计瓦片数 / 内容类型 / 点数 / 包围盒（float64）/ 深度。"""
    ts, root, ours = load_tileset(path)
    up = np.eye(4) if ours else yup_to_zup()
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    n_pts, n_content, formats, depth_max, bad = 0, 0, {}, 0, []
    n_vis = 0
    for i, (t, world, full, depth) in enumerate(iter_tiles(ts, root)):
        if max_tiles and i >= max_tiles:
            break
        n_vis += 1
        depth_max = max(depth_max, depth)
        if not full or not os.path.isfile(full):
            if full:
                bad.append(full)
            continue
        try:
            pos, _c, meta = read_content(full)
        except (ValueError, OSError) as e:
            bad.append(f"{os.path.basename(full)}: {e}")
            continue
        n_content += 1
        n_pts += len(pos)
        fmt = meta.get("format") or os.path.splitext(full)[1].lstrip(".")
        formats[fmt] = formats.get(fmt, 0) + 1
        if len(pos):
            w = (world @ up @ np.column_stack([pos, np.ones(len(pos))]).T)[:3].T
            lo = np.minimum(lo, w.min(0))
            hi = np.maximum(hi, w.max(0))
    if not np.isfinite(lo).all():
        lo = hi = np.zeros(3)
    return {"tiles": n_vis, "contents": n_content, "points": n_pts,
            "depth": depth_max, "formats": formats, "bbox": [lo.tolist(), hi.tolist()],
            "self_produced": ours, "up_axis": "Z(自产)" if ours else "Y→Z(规范转换)",
            "missing_or_bad": bad[:10], "bad_count": len(bad)}


def load_scene(path: str, max_points: int | None = None) -> GeoScene:
    """把整个 tileset 读成单个 GeoScene（用于小中型数据集评测/可视化，不作为城市级主路径）。"""
    ts, root, ours = load_tileset(path)
    up = np.eye(4) if ours else yup_to_zup()
    P, C, nt = [], [], 0
    for t, world, full, _d in iter_tiles(ts, root):
        if not full or not os.path.isfile(full):
            continue
        pos, rgb, _meta = read_content(full)
        if not len(pos):
            continue
        P.append((world @ up @ np.column_stack([pos, np.ones(len(pos))]).T)[:3].T)
        C.append(rgb if rgb is not None else np.full((len(pos), 3), 0.7, np.float32))
        nt += 1
        if max_points and sum(len(x) for x in P) >= max_points:
            break
    if not P:
        raise ValueError("tileset 中未解析出任何点")
    pos = np.vstack(P)
    rgb = np.vstack(C).astype(np.float32)
    if max_points and len(pos) > max_points:
        sel = np.linspace(0, len(pos) - 1, max_points).astype(np.int64)
        pos, rgb = pos[sel], rgb[sel]
    origin = ((pos.min(0) + pos.max(0)) / 2).astype(np.float64)
    local = (pos - origin).astype(np.float32)
    ext = float(np.max(pos.max(0) - pos.min(0))) or 1.0
    sh = np.empty((len(local), 1, 3), np.float32)
    sh[:, 0, :] = (np.clip(rgb, 0, 1) - 0.5) / SH_C0
    spacing = max(math.sqrt(ext * ext / max(len(local), 1)) * 0.5, ext * 1e-4)
    return GeoScene(local, sh, np.ones(len(local), np.float32),
                    np.full((len(local), 3), spacing, np.float32),
                    np.tile(np.array([1, 0, 0, 0], np.float32), (len(local), 1)), "point",
                    {"source": str(path), "format": "3dtiles", "origin": origin.tolist(),
                     "tiles": nt, "extent": round(ext, 3)})


# ---------------- 逐瓦片再压缩 ----------------
def compress_tileset(src: str, out_dir: str, prune=0.5, pos_bits=16, method="voxel",
                     max_points_per_tile=400_000, quantize=True, verbose=True,
                     keep_mesh=True, tex_max_dim=1024, tex_quality=80) -> dict:
    """按瓦片流式再压缩：保持输入树结构与边界，逐块抽稀 + 属性字节量化。

    内存常驻只有「一块瓦片的点云/网格 + 输出缓冲」，因此可处理城市级数据集。
    网格瓦片（b3dm/glb 三角面）默认走纹理保留路径：几何 u16 量化 + 体素聚类抽稀(保 UV)
    + 纹理重压缩（JPEG/WebP）；keep_mesh=False 或非三角面时回退点云路径。
    """
    from .stages import prune as prune_stage

    ts, root, ours = load_tileset(src)
    up = np.eye(4) if ours else yup_to_zup()
    tdir = os.path.join(out_dir, "tiles")
    os.makedirs(tdir, exist_ok=True)
    for old in glob.glob(os.path.join(tdir, "*")):
        os.remove(old)
    os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)

    # 第一遍：定原点与包围盒（float64），保证后续局部坐标精度
    scan = scan_tileset(src)
    lo, hi = (np.asarray(scan["bbox"][0], np.float64), np.asarray(scan["bbox"][1], np.float64))
    origin = (lo + hi) / 2
    extent_all = float(max(hi - lo)) or 1.0

    stats = {"tiles_in": 0, "tiles_out": 0, "pts_in": 0, "pts_out": 0,
             "bytes_in": 0, "bytes_out": 0, "mesh_tiles": 0, "empty_tiles": 0,
             "ge_min": np.inf, "ge_max": 0.0, "pnts_out": 0,
             "mesh_out": 0, "verts_in": 0, "verts_out": 0, "tris_in": 0, "tris_out": 0,
             "tex_in": 0, "tex_out": 0, "tex_psnr": [], "geom_step": 0.0}
    per_tile = []
    counter = [0]

    def conv(t, world, depth, parent_ge):
        """把输入瓦片转成输出瓦片（含子树）。"""
        world = world @ tile_matrix(t)          # 累积本层变换（源 tileset 可能带 ECEF/旋转）
        tid = counter[0]
        counter[0] += 1
        full = (t.get("content") or {}).get("uri")
        path = os.path.join(root, full) if full else None
        content = None
        n_out = 0
        nbytes = 0
        is_mesh = False
        if path and os.path.isfile(path):
            stats["tiles_in"] += 1
            stats["bytes_in"] += os.path.getsize(path)
            # ---- 网格路径：三角面 + 纹理保留 ----
            mesh = None
            if keep_mesh:
                try:
                    from .mesh import read_mesh_content
                    mesh = read_mesh_content(path)
                except (ValueError, OSError, KeyError, struct.error):
                    mesh = None
            if mesh:
                from .mesh import compress_mesh_tile, write_b3dm
                # 顶点从瓦片局部坐标 → 原点相对坐标（world@up 后减原点，float64 精度）
                for pr in mesh["prims"]:
                    pw = (world @ up @ np.column_stack([pr["pos"],
                                                        np.ones(len(pr["pos"]))]).T)[:3].T
                    pr["pos"] = pw - origin
                glb, mi = compress_mesh_tile(mesh, prune, tex_max_dim, tex_quality, quantize)
                b = os.path.join(tdir, f"{tid}.b3dm")
                nbytes = write_b3dm(b, glb)
                # 原点相对坐标下的精确包围盒
                wl = np.vstack([p["pos"] for p in mesh["prims"]])
                c = (wl.min(0) + wl.max(0)) / 2
                e = float(max(wl.max(0) - wl.min(0))) or max(extent_all * 1e-3, 1e-3)
                box = [float(c[0]), float(c[1]), float(c[2]),
                       e / 2, 0, 0, 0, e / 2, 0, 0, 0, e / 2]
                stats["mesh_out"] += 1
                stats["verts_in"] += mi["verts_in"]
                stats["verts_out"] += mi["verts_out"]
                stats["tris_in"] += mi["tris_in"]
                stats["tris_out"] += mi["tris_out"]
                stats["bytes_out"] += nbytes
                for ti in mi["tex"]:
                    stats["tex_in"] += ti["bytes_in"]
                    stats["tex_out"] += ti["bytes_out"]
                    stats["tex_psnr"].append(ti["psnr"])
                stats["geom_step"] = max(stats["geom_step"], mi["geom_quant_step"])
                n_out = mi["verts_out"]
                content = {"uri": f"tiles/{os.path.basename(b)}", "boundingVolume": {"box": box}}
                per_tile.append({"tile": tid, "src": full, "kind": "mesh",
                                 "verts": mi["verts_out"], "tris": mi["tris_out"],
                                 "bytes": nbytes,
                                 "tex_psnr": round(min((t["psnr"] for t in mi["tex"]),
                                                       default=99.0), 2)})
                is_mesh = True
            else:
                # ---- 点云路径（pnts/glb 点图元，或网格解析失败的回退） ----
                pos, rgb, meta = read_content(path)
                stats["pts_in"] += len(pos)
                if meta.get("mesh_sampled"):
                    stats["mesh_tiles"] += 1
                if len(pos):
                    w = (world @ up @ np.column_stack([pos, np.ones(len(pos))]).T)[:3].T
                    loc = (w - origin).astype(np.float32)
                    if rgb is None:
                        rgb = np.full((len(loc), 3), 0.7, np.float32)
                    sc = GeoScene(loc, np.concatenate(
                        [(np.clip(rgb, 0, 1) - 0.5)[:, None, :] / SH_C0,
                         np.zeros((len(loc), 0, 3), np.float32)], 1),
                        np.ones(len(loc), np.float32), np.full((len(loc), 3), 0.0, np.float32),
                        np.tile(np.array([1, 0, 0, 0], np.float32), (len(loc), 1)), "point",
                        {"src_tile": full})
                    sc.scale[:] = max(0.0005, extent_all * 1e-4)
                    if prune and prune > 0:
                        keep_target = int(len(loc) * (1 - prune))
                        if max_points_per_tile:
                            keep_target = min(keep_target, max_points_per_tile)
                        if keep_target < len(loc):
                            sc, _ = prune_stage(sc, 1 - keep_target / len(loc), method=method)
                    c = (sc.pos.min(0) + sc.pos.max(0)) / 2
                    e = float(max(sc.pos.max(0) - sc.pos.min(0))) or max(extent_all * 1e-3, 1e-3)
                    box = [float(c[0]), float(c[1]), float(c[2]),
                           e / 2, 0, 0, 0, e / 2, 0, 0, 0, e / 2]
                    # 点云一律走 pnts（u16 体量化 + u8 RGB ≈ 9B/点）；含 SH 的高斯场景走 GLB
                    fmt = "pnts" if sc.kind == "point" and sc.sh.shape[1] == 1 else "glb"
                    rgb = (sc.colors() if sc.sh.shape[1] else np.full((sc.n, 3), 0.7, np.float32))
                    if fmt == "pnts":
                        from .export import write_pnts_q
                        b = os.path.join(tdir, f"{tid}.pnts")
                        nbytes = write_pnts_q(b, sc.pos.astype(np.float64), rgb)
                    else:
                        b = os.path.join(tdir, f"{tid}.glb")
                        nbytes = _write_glb(b, sc, c, e, quantize)
                    n_out = sc.n
                    stats["bytes_out"] += nbytes
                    stats["pts_out"] += n_out
                    if fmt == "pnts":
                        stats["pnts_out"] += 1
                    content = {"uri": f"tiles/{os.path.basename(b)}", "boundingVolume": {"box": box}}
                    per_tile.append({"tile": tid, "src": full, "kind": "points",
                                     "primitives": int(n_out), "bytes": nbytes})
                else:
                    stats["empty_tiles"] += 1
        ge = float(t.get("geometricError", 0.0))
        if n_out and prune:
            ge = max(ge, extent_all * 1e-3 * (2 ** max(0, 6 - depth)))
        ge = min(ge, parent_ge)                       # 子层误差不得超过父层
        kids = [conv(c, world, depth + 1, ge) for c in t.get("children", [])]
        if content and content.get("boundingVolume"):
            # 叶子瓦片：直接用内容实际包围盒（与输出坐标系严格一致，
            # 不依赖输入 tileset 的 BV 坐标约定——世界坐标/局部坐标均可稳健处理）
            bv = content["boundingVolume"]
        else:
            bv = _shift_box(t["boundingVolume"], origin) if t.get("boundingVolume") else \
                {"box": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]}
        tile = {"boundingVolume": bv,
                "geometricError": round(max(ge, 0.0), 4), "refine": t.get("refine", "REPLACE")}
        if content:
            tile["content"] = content
        if kids:
            tile["children"] = kids
        tile["extras"] = {"primitives": int(n_out), "src": full}
        return tile

    root_tile = conv(ts["root"], np.eye(4), 0, float(ts.get("geometricError") or 1e9))
    out = {"asset": {"version": "1.1", "generator": "GeoSqueeze",
                     # Cesium 1.1x 从 tileset.asset.gltfUpAxis（字符串）读取上轴，
                     # 不再读 b3dm Feature Table 的 gltfUpAxis；缺省按 Y-up 旋转会放倒 Z-up 内容
                     "gltfUpAxis": "Z",
                     "extras": {"upAxis": "Z", "origin": origin.tolist(),
                                "source_tileset": os.path.abspath(src)}},
           "geometricError": ts.get("geometricError", root_tile["geometricError"] * 4),
           "root": root_tile,
           "extras": {"geosqueeze": {"recompressed": True, "prune": prune,
                                     "pos_bits": pos_bits, "quantize": bool(quantize),
                                     "tiles_out": None, "pts_in": None, "pts_out": None}}}
    # 列主序平移矩阵：把局部坐标平移回世界坐标（float64 精度复原）
    out["root"]["transform"] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                                float(origin[0]), float(origin[1]), float(origin[2]), 1]
    tp = os.path.join(out_dir, "tileset.json")
    with open(tp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    tiles_out = len(per_tile)
    ts_bytes = os.path.getsize(tp)
    out["extras"]["geosqueeze"].update({"tiles_out": tiles_out, "pts_in": stats["pts_in"],
                                        "pts_out": stats["pts_out"]})
    with open(tp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    viewer = None
    if stats["mesh_out"]:
        from .mesh_viewer import write_mesh_viewer
        viewer = write_mesh_viewer(out_dir, title=f"GeoSqueeze · {os.path.basename(out_dir)}")
    # CesiumJS 查看器：数据集不大时内嵌为自包含单文件（双击即可打开，绕开 file:// CORS）
    try:
        from .cesium_viewer import write_cesium_viewer
        cesium_page = write_cesium_viewer(out_dir)
    except Exception:
        cesium_page = None  # Cesium 查看器是附加能力，不阻塞主流程
    ps = stats["tex_psnr"]
    res = {"tileset": tp, "tiles_in": stats["tiles_in"], "tiles_out": tiles_out,
           "pts_in": stats["pts_in"], "pts_out": stats["pts_out"],
           "bytes_in": stats["bytes_in"], "bytes_out": stats["bytes_out"] + ts_bytes,
           "ratio": round((stats["bytes_in"] or 1) / max(stats["bytes_out"] + ts_bytes, 1), 2),
           "point_ratio": round(stats["pts_in"] / max(stats["pts_out"], 1), 2),
           "mesh_tiles": stats["mesh_tiles"], "origin": origin.tolist(),
           "mesh_out": stats["mesh_out"],
           "verts_in": stats["verts_in"], "verts_out": stats["verts_out"],
           "tris_in": stats["tris_in"], "tris_out": stats["tris_out"],
           "tri_keep": round(stats["tris_out"] / max(stats["tris_in"], 1), 4),
           "tex_bytes_in": stats["tex_in"], "tex_bytes_out": stats["tex_out"],
           "tex_ratio": round(stats["tex_in"] / max(stats["tex_out"], 1), 2),
           "tex_psnr_min": round(min(ps), 2) if ps else None,
           "tex_psnr_mean": round(float(np.mean(ps)), 2) if ps else None,
           "geom_quant_step": round(stats["geom_step"], 6),
           "viewer": viewer, "cesium_page": cesium_page,
           "per_tile": per_tile}
    with open(os.path.join(out_dir, "meta", "tiles_report.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    if verbose:
        print(f"  瓦片 {stats['tiles_in']} → {tiles_out} · 点 {stats['pts_in']:,} → {stats['pts_out']:,} "
              f"· 字节 {stats['bytes_in']/1024**2:.2f}MB → {(stats['bytes_out']+ts_bytes)/1024**2:.2f}MB "
              f"· {res['ratio']}x")
        if stats["mesh_out"]:
            print(f"  网格瓦片 {stats['mesh_out']} · 三角形 {stats['tris_in']:,} → {stats['tris_out']:,} "
                  f"(保留 {res['tri_keep']*100:.1f}%) · 纹理 {stats['tex_in']/1024:.0f}KB → "
                  f"{stats['tex_out']/1024:.0f}KB ({res['tex_ratio']}x) · "
                  f"纹理PSNR {res['tex_psnr_min']}~{res['tex_psnr_mean']}dB · "
                  f"几何量化步长 {res['geom_quant_step']*1000:.2f}mm")
    return res


def _shift_box(bv: dict, origin: np.ndarray) -> dict:
    """把输入瓦片的包围体从世界坐标平移到局部坐标（输出根节点 transform 负责复原）。"""
    if "box" in bv:
        b = np.asarray(bv["box"], np.float64).copy()
        b[:3] -= origin
        return {"box": [float(x) for x in b]}
    if "sphere" in bv:
        s = list(bv["sphere"])
        s[0] -= float(origin[0])
        s[1] -= float(origin[1])
        s[2] -= float(origin[2])
        return {"sphere": s}
    if "region" in bv:                        # 经纬度区域体：无法简单平移，退化为盒体
        r = bv["region"]
        return {"region": r}
    return bv


def _write_glb(path: str, s: GeoScene, center, extent, quantize=True) -> int:
    from .export import write_glb
    return write_glb(path, s, np.asarray(center, np.float64), float(extent), bool(quantize), 0)
