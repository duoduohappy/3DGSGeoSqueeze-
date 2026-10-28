"""矢量数据压缩：GeoJSON 读入 → 几何简化 → 整数网格量化 → 顶点池拓扑去重 → 字典+varint 编码 → 分块 LOD 瓦片。

与「模型压缩」的四件套对应关系：
  · 量化   → 坐标整型化（局部网格 4096，等价于 MVT 的 quantized coordinates）
  · 剪枝   → Douglas–Peucker 顶点删除（拓扑保持，面/线闭合性不破坏）
  · 蒸馏   → 多级 LOD 生成（粗层用大容差简化，作为「轻量副本」供远视距使用）
  · 低秩/去重 → 顶点池共享（相邻地块共边只存一份，天然拓扑编码）

产物：
  · `.gvec`  单文件紧凑容器（JSON 头 + varint 列式几何/属性）
  · 分块瓦片  `<out>/vector_tiles.json` + `<out>/L{z}/{x}_{y}.gvec`，供城市级按视距流式渲染
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

EXTENT = 4096            # 局部网格刻度（与 MVT 一致）


# ---------------- 数据模型 ----------------
@dataclass
class VecLayer:
    """矢量图层：环集合 + 属性表。rings 为世界坐标 (m,2) float64。"""
    rings: list[np.ndarray]
    closed: list[bool]
    props: dict[str, np.ndarray] = field(default_factory=dict)
    origin: np.ndarray = field(default_factory=lambda: np.zeros(2))
    srs: str = ""
    kinds: list[str] = field(default_factory=list)      # polygon | line | point

    @property
    def n(self) -> int:
        return len(self.rings)

    def vertex_count(self) -> int:
        return int(sum(len(r) for r in self.rings))


# ---------------- 读入 ----------------
def read_geojson(path: str, max_features: int | None = None) -> VecLayer:
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj.get("features", [gj] if gj.get("type") == "Feature" else [])
    if max_features:
        feats = feats[:max_features]
    rings, closed, kinds, props = [], [], [], {}
    cols: dict[str, list] = {}
    for ft in feats:
        g = ft.get("geometry") or {}
        t = g.get("type")
        c = g.get("coordinates")
        if not c or t is None:
            continue
        if t == "Polygon":
            for i, ring in enumerate(c):
                rings.append(np.asarray(ring, np.float64)[:, :2])
                closed.append(True)
                kinds.append("polygon")
        elif t == "MultiPolygon":
            for poly in c:
                for ring in poly:
                    rings.append(np.asarray(ring, np.float64)[:, :2])
                    closed.append(True)
                    kinds.append("polygon")
        elif t == "LineString":
            rings.append(np.asarray(c, np.float64)[:, :2])
            closed.append(False)
            kinds.append("line")
        elif t == "MultiLineString":
            for line in c:
                rings.append(np.asarray(line, np.float64)[:, :2])
                closed.append(False)
                kinds.append("line")
        elif t in ("Point", "MultiPoint"):
            pts = [c] if t == "Point" else c
            for p in pts:
                rings.append(np.asarray([p], np.float64)[:, :2])
                closed.append(False)
                kinds.append("point")
        else:
            continue
        p = ft.get("properties") or {}
        for k, v in p.items():
            cols.setdefault(k, []).append(v)
    n = len(rings)
    for k, v in cols.items():
        if len(v) < n:                       # 该要素缺此字段 → 补齐
            v = v + [None] * (n - len(v))
        props[k] = _as_array(v)
    for k in props:                          # 长度对齐
        if len(props[k]) != n:
            props[k] = props[k][:n]
    allp = np.vstack([r for r in rings if len(r)]) if rings else np.zeros((1, 2))
    origin = (allp.min(0) + allp.max(0)) / 2
    srs = ""
    if isinstance(gj.get("crs"), dict):
        srs = gj["crs"].get("properties", {}).get("name", "")
    return VecLayer([r for r in rings], closed, props, origin, srs, kinds)


def _as_array(vals: list) -> np.ndarray:
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        if any(isinstance(v, float) for v in vals) or any(v is None for v in vals):
            return np.asarray([np.nan if v is None else float(v) for v in vals], np.float32)
        return np.asarray([0 if v is None else int(v) for v in vals], np.int64)
    return np.asarray(["" if v is None else str(v) for v in vals], dtype=object)


# ---------------- 几何简化（Douglas–Peucker） ----------------
def _dp_mask(pts: np.ndarray, tol: float) -> np.ndarray:
    """迭代式 DP：返回保留掩码（首尾恒保留）。纯 numpy 逐段递归改为显式栈。"""
    m = len(pts)
    keep = np.zeros(m, bool)
    keep[0] = keep[-1] = True
    if m <= 2 or tol <= 0:
        keep[:] = True
        return keep
    stack = [(0, m - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = pts[b] - pts[a]
        L = math.hypot(seg[0], seg[1])
        sub = pts[a + 1:b]
        if L < 1e-12:
            d = np.hypot(sub[:, 0] - pts[a, 0], sub[:, 1] - pts[a, 1])
        else:
            d = np.abs(seg[0] * (pts[a, 1] - sub[:, 1]) - (pts[a, 0] - sub[:, 0]) * seg[1]) / L
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[a + 1 + k] = True
            stack.append((a, a + 1 + k))
            stack.append((a + 1 + k, b))
    return keep


def simplify(layer: VecLayer, tol: float, min_pts_poly=4) -> tuple[VecLayer, dict]:
    """按容差删除冗余顶点；多边形保证闭合且 ≥4 点（退化则保留原环）。"""
    out_r, n_before, n_after = [], layer.vertex_count(), 0
    for r, cl, kd in zip(layer.rings, layer.closed, layer.kinds):
        if kd == "point" or len(r) <= 3:
            out_r.append(r)
            n_after += len(r)
            continue
        closed_ring = cl and len(r) > 3 and np.allclose(r[0], r[-1])
        core = r[:-1] if closed_ring else r
        if len(core) <= 3:
            out_r.append(r)
            n_after += len(r)
            continue
        if closed_ring:                     # 闭合环：以首点为锚拆两段，避免首尾被误删
            half = len(core) // 2
            m1 = _dp_mask(np.vstack([core[:half + 1]]), tol)
            m2 = _dp_mask(np.vstack([core[half:], core[:1]]), tol)
            keep = np.zeros(len(core), bool)
            keep[:half + 1] = m1
            keep[half:] |= m2[:len(core) - half]
            keep[0] = True
            simp = core[keep]
            if len(simp) < min_pts_poly - 1:
                simp = core
            simp = np.vstack([simp, simp[:1]])
        else:
            keep = _dp_mask(core, tol)
            simp = core[keep]
            if len(simp) < 2:
                simp = np.vstack([core[0], core[-1]])
        out_r.append(simp)
        n_after += len(simp)
    nl = VecLayer(out_r, list(layer.closed), layer.props, layer.origin, layer.srs, layer.kinds)
    return nl, {"vertices_in": n_before, "vertices_out": n_after,
                "reduction": round(1 - n_after / max(n_before, 1), 4), "tol": tol}


# ---------------- 量化 + 顶点池去重 ----------------
def _quantize_rings(rings, origin, size: float):
    """世界坐标 → 局部整数网格（0..EXTENT）。size 为该瓦片 bbox 的边长（米）。"""
    out = []
    for r in rings:
        v = np.round((r - origin) / max(size, 1e-12) * EXTENT + EXTENT / 2)
        out.append(np.clip(v, 0, EXTENT).astype(np.int32))
    return out


def build_vertex_pool(qrings) -> tuple[np.ndarray, list[np.ndarray]]:
    """顶点池去重：所有环的整型坐标合并去重，环只存索引（共边/共顶点只存一份）。"""
    if not qrings:
        return np.zeros((0, 2), np.int32), []
    allv = np.vstack(qrings)
    key = allv[:, 0].astype(np.int64) * (EXTENT + 1) + allv[:, 1]
    uniq, inv = np.unique(key, return_inverse=True)
    verts = np.column_stack([uniq // (EXTENT + 1), uniq % (EXTENT + 1)]).astype(np.int32)
    idxs, off = [], 0
    for r in qrings:
        idxs.append(inv[off:off + len(r)].astype(np.int32))
        off += len(r)
    return verts, idxs


# ---------------- varint / zigzag ----------------
def _put_varint(buf: bytearray, v: int):
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            buf.append(b | 0x80)
        else:
            buf.append(b)
            return


def _zig(v: int) -> int:
    return (v << 1) ^ (v >> 63) if v < 0 else (v << 1)


def _put_zig(buf: bytearray, v: int):
    _put_varint(buf, (v << 1) ^ (v >> 63) if v < 0 else (v << 1))


def _get_varint(data: bytes, pos: int) -> tuple[int, int]:
    r = s = 0
    while True:
        b = data[pos]
        pos += 1
        r |= (b & 0x7F) << s
        if not b & 0x80:
            return r, pos
        s += 7


def _get_zig(data: bytes, pos: int) -> tuple[int, int]:
    v, pos = _get_varint(data, pos)
    return (v >> 1) ^ -(v & 1), pos


# ---------------- 编码 / 解码 ----------------
def encode_gvec(layer: VecLayer, level=0, bbox=None, tol=0.0, meta=None) -> tuple[bytes, dict]:
    """编码为 .gvec：头 JSON（几何/属性布局）+ varint 列式载荷。"""
    if bbox is None:
        allp = np.vstack(layer.rings) if layer.n else np.zeros((1, 2))
        bbox = [float(allp[:, 0].min()), float(allp[:, 1].min()),
                float(allp[:, 0].max()), float(allp[:, 1].max())]
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) or 1.0
    origin = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    q = _quantize_rings(layer.rings, origin, size)
    verts, idxs = build_vertex_pool(q)

    body = bytearray()
    # 1) 顶点表：逐点 delta + zigzag（相邻顶点空间相关）
    px = py = 0
    for x, y in verts:
        _put_zig(body, int(x) - px)
        _put_zig(body, int(y) - py)
        px, py = int(x), int(y)
    # 2) 几何：每条环的长度 + 索引（delta）
    _put_varint(body, len(idxs))
    for a in idxs:
        _put_varint(body, len(a))
    for a in idxs:
        prev = 0
        for v in a:
            _put_zig(body, int(v) - prev)
            prev = int(v)
    # 3) 属性列：字符串字典 / 整型 zigzag / 浮点 f32
    cols = []
    for k, arr in layer.props.items():
        if arr.dtype == object:
            vals, codes = np.unique(arr.astype(str), return_inverse=True)
            if len(vals) <= max(16, 0.5 * len(arr)):        # 低基数 → 字典编码
                _put_varint(body, len(vals))
                for sv in vals:
                    bs = str(sv).encode("utf-8")
                    _put_varint(body, len(bs))
                    body += bs
                for c in codes:
                    _put_varint(body, int(c))
                cols.append({"name": k, "type": "dict", "values": [str(v) for v in vals],
                             "codes": int(len(codes))})
            else:                                            # 高基数（编码类字段）→ 原串长度前缀
                for sv in arr:
                    bs = str(sv).encode("utf-8")
                    _put_varint(body, len(bs))
                    body += bs
                cols.append({"name": k, "type": "str", "count": int(len(arr))})
        elif np.issubdtype(arr.dtype, np.integer):
            prev = 0
            for v in arr:
                _put_zig(body, int(v) - prev)
                prev = int(v)
            cols.append({"name": k, "type": "int", "count": int(len(arr))})
        else:
            body += np.asarray(arr, "<f4").tobytes()
            cols.append({"name": k, "type": "f32", "count": int(len(arr))})
    # 几何类型压成每环一字符（g=面 / l=线 / p=点），闭合性由类型决定，省掉两个大数组
    codes = "".join({"polygon": "g", "line": "l", "point": "p"}.get(k, "l") for k in layer.kinds)
    head = {"v": 1, "level": int(level), "bbox": [round(float(x), 6) for x in bbox],
            "origin": [float(origin[0]), float(origin[1])], "size": float(size), "extent": EXTENT,
            "n_vertices": int(len(verts)), "n_rings": len(idxs), "types": codes,
            "props": cols, "tol": tol, "meta": meta or {}}
    hb = json.dumps(head, ensure_ascii=False, separators=(",", ":")).encode()
    out = b"GVEC" + np.uint32(len(hb)).tobytes() + hb + bytes(body)
    return out, {"bytes": len(out), "header_bytes": 8 + len(hb), "payload_bytes": len(body),
                 "vertices": len(verts), "rings": len(idxs), "bbox": head["bbox"]}


def decode_gvec(path_or_bytes) -> VecLayer:
    raw = open(path_or_bytes, "rb").read() if isinstance(path_or_bytes, str) else path_or_bytes
    if raw[:4] != b"GVEC":
        raise ValueError("不是 GVEC 文件")
    hl = int(np.frombuffer(raw[4:8], np.uint32)[0])
    head = json.loads(raw[8:8 + hl])
    pos = 8 + hl
    nv = head["n_vertices"]
    verts = np.zeros((nv, 2), np.int32)
    px = py = 0
    for i in range(nv):
        dx, pos = _get_zig(raw, pos)
        dy, pos = _get_zig(raw, pos)
        px += dx
        py += dy
        verts[i] = (px, py)
    nr, pos = _get_varint(raw, pos)
    lens = []
    for _ in range(nr):
        v, pos = _get_varint(raw, pos)
        lens.append(v)
    idxs = []
    for L in lens:
        prev = 0
        a = np.empty(L, np.int32)
        for i in range(L):
            d, pos = _get_zig(raw, pos)
            prev += d
            a[i] = prev
        idxs.append(a)
    props = {}
    for col in head["props"]:
        if col["type"] == "dict":
            nv2, pos = _get_varint(raw, pos)
            vals = []
            for _ in range(nv2):
                ln, pos = _get_varint(raw, pos)
                vals.append(raw[pos:pos + ln].decode("utf-8"))
                pos += ln
            codes = np.empty(col["codes"], np.int64)
            for i in range(col["codes"]):
                c, pos = _get_varint(raw, pos)
                codes[i] = c
            props[col["name"]] = np.asarray([vals[c] for c in codes], dtype=object)
        elif col["type"] == "str":
            vals = []
            for _ in range(col["count"]):
                ln, pos = _get_varint(raw, pos)
                vals.append(raw[pos:pos + ln].decode("utf-8"))
                pos += ln
            props[col["name"]] = np.asarray(vals, dtype=object)
        elif col["type"] == "str":
            vals = []
            for _ in range(col["count"]):
                ln, pos = _get_varint(raw, pos)
                vals.append(raw[pos:pos + ln].decode("utf-8"))
                pos += ln
            props[col["name"]] = np.asarray(vals, dtype=object)
        elif col["type"] == "int":
            prev = 0
            a = np.empty(col["count"], np.int64)
            for i in range(col["count"]):
                d, pos = _get_zig(raw, pos)
                prev += d
                a[i] = prev
            props[col["name"]] = a
        else:
            n = col["count"]
            props[col["name"]] = np.frombuffer(raw, "<f4", n, pos).astype(np.float32)
            pos += 4 * n
    origin = np.array(head["origin"], np.float64)
    scale = head["size"] / head["extent"]
    codes = head.get("types", "")
    rings = [((verts[a].astype(np.float64) - head["extent"] / 2) * scale + origin) for a in idxs]
    kinds = [{"g": "polygon", "l": "line", "p": "point"}.get(c, "line") for c in codes]
    closed = [k == "polygon" for k in kinds]
    return VecLayer(rings, closed, props, origin, "", kinds)


# ---------------- 评测 ----------------
def eval_vector(orig: VecLayer, dec: VecLayer, bytes_out=None) -> dict:
    """几何保真度：最大偏离（单侧 Hausdorff 近似）、长度/面积相对误差、顶点压缩率。"""
    dev, len_o, len_d, area_o, area_d = 0.0, 0.0, 0.0, 0.0, 0.0
    for ro, rd, cl, kd in zip(orig.rings, dec.rings, orig.closed, orig.kinds):
        if kd == "point" or len(ro) < 2 or len(rd) < 2:
            continue
        d = _max_dev(ro, rd)
        dev = max(dev, d)
        lo = float(np.sum(np.linalg.norm(np.diff(ro, axis=0), axis=1)))
        ld = float(np.sum(np.linalg.norm(np.diff(rd, axis=0), axis=1)))
        len_o += lo
        len_d += ld
        if cl and len(ro) > 3:
            area_o += abs(_shoelace(ro))
            area_d += abs(_shoelace(rd))
    vo, vd = orig.vertex_count(), dec.vertex_count()
    res = {"vertices_in": vo, "vertices_out": vd,
           "vertex_ratio": round(vo / max(vd, 1), 2),
           "vertex_reduction": round(1 - vd / max(vo, 1), 4),
           "max_dev": round(dev, 4),
           "len_rel_err": round(abs(len_d - len_o) / max(len_o, 1e-9), 6),
           "area_rel_err": round(abs(area_d - area_o) / max(area_o, 1e-9), 6) if area_o else 0.0,
           "features": orig.n}
    if bytes_out:
        res.update(bytes_out)
    return res


def _shoelace(r: np.ndarray) -> float:
    x, y = r[:, 0], r[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _max_dev(coarse: np.ndarray, dense: np.ndarray) -> float:
    """dense 各顶点到 coarse 折线的最大距离（单侧 Hausdorff）。"""
    seg_a = coarse[:-1]
    seg_b = coarse[1:]
    ab = seg_b - seg_a
    L2 = np.maximum((ab ** 2).sum(1), 1e-18)
    md = 0.0
    for p in dense:
        ap = p - seg_a
        t = np.clip((ap * ab).sum(1) / L2, 0, 1)
        proj = seg_a + ab * t[:, None]
        d = np.sqrt(((p - proj) ** 2).sum(1)).min()
        md = max(md, float(d))
    return md


# ---------------- 分块 LOD 瓦片 ----------------
def build_vector_tiles(layer: VecLayer, out_dir: str, levels=4, tol_scale=2e-3,
                       max_tiles_per_level=256, verbose=True) -> dict:
    """网格分块 + 逐级简化：level z 的单元边长 = bbox/2^z，容差 ∝ 单元边长。

    城市级渲染时按视距选层、按视锥选块，单次只解码视口内的少量瓦片。
    """
    allp = np.vstack(layer.rings) if layer.n else np.zeros((1, 2))
    lo, hi = allp.min(0), allp.max(0)
    span = float(max(hi - lo)) or 1.0
    os.makedirs(out_dir, exist_ok=True)
    for d in os.listdir(out_dir):
        fp = os.path.join(out_dir, d)
        if d.startswith("L") and os.path.isdir(fp):
            for f in os.listdir(fp):
                os.remove(os.path.join(fp, f))
    manifest = {"format": "gvec-tiles", "v": 1, "bbox": [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])],
                "levels": [], "extent": EXTENT}
    # 特征分配到 cell（按环首点所在单元；跨单元要素仍只存一份，避免切割开销）
    cen = np.array([[r[:, 0].mean(), r[:, 1].mean()] for r in layer.rings]) if layer.n else np.zeros((0, 2))
    total_bytes = 0
    for z in range(levels):
        cells = 2 ** z
        cell_w = span / cells
        tol = span * tol_scale * (cells ** -0.85)
        ij = np.floor((cen - lo) / max(cell_w, 1e-9)).astype(np.int64).clip(0, cells - 1)
        keys = ij[:, 0] * cells + ij[:, 1] if len(ij) else np.zeros(0, np.int64)
        lvl = {"level": z, "cells": cells, "tol": round(float(tol), 5), "tiles": []}
        os.makedirs(os.path.join(out_dir, f"L{z}"), exist_ok=True)
        uniq = np.unique(keys)
        for k in uniq:
            sel = np.flatnonzero(keys == k)
            sub = VecLayer([layer.rings[i] for i in sel], [layer.closed[i] for i in sel],
                           {kk: vv[sel] for kk, vv in layer.props.items()},
                           layer.origin, layer.srs, [layer.kinds[i] for i in sel])
            ss, _info = simplify(sub, tol)
            bx0, by0 = lo[0] + (k // cells) * cell_w, lo[1] + (k % cells) * cell_w
            bb = [bx0, by0, bx0 + cell_w, by0 + cell_w]
            data, st = encode_gvec(ss, level=z, bbox=_clip_bbox(ss, bb), tol=tol,
                                   meta={"features": int(len(sel))})
            uri = f"L{z}/{k // cells}_{k % cells}.gvec"
            with open(os.path.join(out_dir, uri), "wb") as f:
                f.write(data)
            total_bytes += len(data)
            lvl["tiles"].append({"uri": uri, "bbox": st["bbox"], "vertices": st["vertices"],
                                 "rings": st["rings"], "bytes": st["bytes"],
                                 "features": int(len(sel))})
        lvl["bytes"] = sum(t["bytes"] for t in lvl["tiles"])
        lvl["vertices"] = sum(t["vertices"] for t in lvl["tiles"])
        manifest["levels"].append(lvl)
        if verbose:
            print(f"  L{z}: {len(lvl['tiles'])} 块 · {lvl['vertices']:,} 顶点 · "
                  f"{lvl['bytes']/1024:.1f} KB · tol={tol:.3f} m")
    mp = os.path.join(out_dir, "vector_tiles.json")
    manifest["bytes"] = total_bytes
    manifest["props"] = {k: (v.dtype.kind if hasattr(v, "dtype") else "o") for k, v in layer.props.items()}
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    return {"manifest": mp, "bytes": total_bytes + os.path.getsize(mp), "levels": levels,
            "tiles": sum(len(l["tiles"]) for l in manifest["levels"]),
            "vertices": [l["vertices"] for l in manifest["levels"]]}


def _clip_bbox(layer: VecLayer, bb) -> list:
    """瓦片 bbox 与实际内容取并集，保证包围盒覆盖全部几何（便于视锥剔除）。"""
    if not layer.n:
        return list(bb)
    allp = np.vstack(layer.rings)
    return [float(min(bb[0], allp[:, 0].min())), float(min(bb[1], allp[:, 1].min())),
            float(max(bb[2], allp[:, 0].max())), float(max(bb[3], allp[:, 1].max()))]


# ---------------- 一体化压缩入口 ----------------
def compress_vector(path: str, out_dir: str, tol=0.35, levels=4, max_features=None,
                    verbose=True) -> dict:
    """GeoJSON → 简化 + 量化 + 去重编码 → .gvec 单文件 + 分块 LOD 瓦片 + 评测。"""
    os.makedirs(out_dir, exist_ok=True)
    layer = read_geojson(path, max_features=max_features)
    src_bytes = os.path.getsize(path)
    tb = max(layer.rings, key=len) if layer.n else np.zeros((1, 2))
    layer_s, s_info = simplify(layer, tol)
    data, st = encode_gvec(layer_s, level=99, tol=tol,
                           meta={"features": layer.n, "src": os.path.basename(path)})
    gp = os.path.join(out_dir, "layer.gvec")
    with open(gp, "wb") as f:
        f.write(data)
    dec = decode_gvec(gp)
    ev = eval_vector(layer, dec, {"bytes_in": src_bytes, "bytes_out": len(data),
                                  "ratio": round(src_bytes / max(len(data), 1), 2),
                                  "gvec_bytes": len(data), "header_bytes": st["header_bytes"]})
    tl = build_vector_tiles(layer, os.path.join(out_dir), levels=levels, verbose=verbose)
    res = {"gvec": gp, "single": ev, "tiles": tl, "simplify": s_info,
           "srs": layer.srs, "features": layer.n, "vertices": layer.vertex_count()}
    with open(os.path.join(out_dir, "vector_report.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    if verbose:
        print(f"  矢量 {layer.n:,} 要素 / {layer.vertex_count():,} 顶点 → "
              f"顶点 {ev['vertex_reduction']*100:.1f}%↓ · 字节 {src_bytes/1024:.0f}K → "
              f"{len(data)/1024:.0f}K ({ev['ratio']}x) · 最大偏离 {ev['max_dev']:.3f} m")
    return res
