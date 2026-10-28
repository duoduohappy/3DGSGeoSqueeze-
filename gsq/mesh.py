"""网格瓦片纹理保留压缩：GLB 网格解析/重写 · 体素聚类抽稀(保 UV) · 纹理重压缩 · 评测。

设计要点：
1. **几何**：位置 u16 体量化（KHR_mesh_quantization，误差 = extent/65535）、UV u16 量化、
   索引 u32；抽稀用体素顶点聚类，UV/位置取簇均值，拓扑（三角形连接）完整保留重建。
2. **纹理**：解码 PNG/JPEG → 按最大边长等比缩放(LANCZOS) → 不透明重编码 JPEG(q)、
   带 alpha 重编码 WebP(q)；重编码后若反而变大则保留原图（保证只赚不赔）。
3. **验收评测**：逐纹理 PSNR（在压缩纹理分辨率下对比）+ 顶点量化误差上界 + 三角形数守恒。
"""
from __future__ import annotations

import io
import json
import os
import struct

import numpy as np

from .decimate import decimate_vertices
from .export import BIN_CHUNK, COMP, JSON_CHUNK, GLB_MAGIC
from .quant import (MAX_U16, max_abs_error, normalized_decode, normalized_encode,
                    normalized_step)
from .tiles_in import _accessor, _glb_chunks, _node_matrix

COMP_U4 = 5125


# ---------------- 纹理压缩与评测 ----------------
def _sniff_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def compress_texture(data: bytes, mime: str | None = None, max_dim=1024, quality=80) -> tuple[bytes, str]:
    """纹理重压缩：缩放到 max_dim 内 + 重编码；结果更大则原样返回。"""
    from PIL import Image

    im = Image.open(io.BytesIO(data))
    im.load()
    w, h = im.size
    s = max_dim / float(max(w, h))
    if s < 1.0:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    arr = np.asarray(im)
    has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
    if has_alpha and arr.ndim == 3 and arr.shape[-1] == 4 and (arr[..., 3] < 250).any():
        out_fmt, out_mime, save_im = "WEBP", "image/webp", im.convert("RGBA")
    else:
        out_fmt, out_mime, save_im = "JPEG", "image/jpeg", im.convert("RGB")
    bio = io.BytesIO()
    save_im.save(bio, out_fmt, quality=int(quality), optimize=True)
    out = bio.getvalue()
    if len(out) >= len(data):                       # 重编码无收益（已是高效编码）→ 原样保留
        return data, mime or _sniff_mime(data)
    return out, out_mime


def texture_psnr(orig: bytes, comp: bytes, cap=512) -> float:
    """在压缩纹理分辨率下评估保真度（浏览端只能采样到压缩后的分辨率，以此为公平基准）。"""
    from PIL import Image

    a = Image.open(io.BytesIO(orig)).convert("RGB")
    b = Image.open(io.BytesIO(comp)).convert("RGB")
    if b.size != a.size:
        a = a.resize(b.size, Image.LANCZOS)
    if max(a.size) > cap:
        k = cap / max(a.size)
        a = a.resize((max(1, round(a.size[0] * k)), max(1, round(a.size[1] * k))), Image.LANCZOS)
        b = b.resize(a.size, Image.LANCZOS)
    mse = float(((np.asarray(a, np.float32) - np.asarray(b, np.float32)) ** 2).mean())
    return float(min(10 * np.log10(255.0 ** 2 / max(mse, 1e-9)), 99.0))


# ---------------- 网格读取 ----------------
def read_mesh_content(path: str) -> dict:
    """解析 b3dm/glb → {prims, images, mat_tex}。要求三角面(mode=4)；点图元抛错由上层回退点路径。"""
    raw = open(path, "rb").read()
    ext = os.path.splitext(path)[1].lower()
    if ext == ".b3dm":
        if raw[:4] != b"b3dm":
            raise ValueError("不是 b3dm")
        _m, _v, _bl, ftjl, ftbl, btjl, btbl = struct.unpack_from("<4sIIIIII", raw, 0)
        glb = raw[28 + ftjl + ftbl + btjl + btbl:]
    elif ext == ".glb":
        glb = raw
    else:
        raise ValueError(f"不支持的网格容器：{ext}")
    js, binary = _glb_chunks(glb)
    images = []
    for im in js.get("images", []):
        if "uri" in im:
            raise ValueError("外链纹理 uri 暂不支持")
        bv = js["bufferViews"][im["bufferView"]]
        o = int(bv.get("byteOffset", 0))
        data = bytes(binary[o:o + int(bv["byteLength"])])
        images.append((data, im.get("mimeType") or _sniff_mime(data)))
    mats = js.get("materials", [])
    mat_tex = [(m.get("pbrMetallicRoughness") or {}).get("baseColorTexture", {}).get("index")
               for m in mats]
    prims = []
    for node in js.get("nodes", []):
        if "mesh" not in node:
            continue
        nm = _node_matrix(node)
        for prim in js["meshes"][node["mesh"]].get("primitives", []):
            if prim.get("mode", 4) != 4 or "POSITION" not in prim.get("attributes", {}):
                raise ValueError("非三角面图元（走点云回退路径）")
            at = prim["attributes"]
            p = _accessor(js, binary, at["POSITION"]).astype(np.float64)
            p = (nm @ np.column_stack([p, np.ones(len(p))]).T)[:3].T      # 还原量化/节点变换
            idx = _accessor(js, binary, prim["indices"]).astype(np.int64).ravel()
            uv = _accessor(js, binary, at["TEXCOORD_0"]).astype(np.float32) \
                if "TEXCOORD_0" in at else None
            prims.append({"pos": p, "idx": idx, "uv": uv, "mat": int(prim.get("material", 0))})
    if not prims:
        raise ValueError("无三角面图元")
    return {"prims": prims, "images": images, "mat_tex": mat_tex}


# ---------------- 网格抽稀（体素聚类，保 UV） ----------------
def decimate_mesh(pos: np.ndarray, idx: np.ndarray, uv: np.ndarray | None, ratio: float,
                  *, min_kept_ratio: float = 0.4) -> tuple:
    """顶点体素聚类抽稀：簇内位置/UV 取均值，重建连接；退化三角形剔除。

    实现见 `decimate.decimate_vertices`——与点云体素下采样共用同一体素划分，
    差异只在「每簇留质心（网格重建顶点）还是留最近真实点（点云保采样）」。
    质量守卫：若更粗一档会让保留三角形跌破 ``min_kept_ratio``，停在上一档
    （宁可少抽不毁几何）；任何情况下不会抽空。
    """
    p, i, u, _info = decimate_vertices(pos, idx, uv, ratio, min_kept_ratio=min_kept_ratio)
    return p, i, u


# ---------------- GLB 网格写出 ----------------
def geom_bounds(prims: list[dict]) -> tuple[np.ndarray, float]:
    """全部图元共享的量化包围盒：返回 (lo, extent)。

    共用一套 lo/ext 是必要的——节点变换只有一组 translation/scale，
    每图元各自归一化会让浏览端还原错位。
    """
    if not prims:
        return np.zeros(3, np.float64), 1.0
    allp = np.vstack([p["pos"] for p in prims])
    lo = allp.min(0)
    ext = float(max(allp.max(0) - lo)) or 1.0
    return lo, ext


def geom_quant_stats(prims: list[dict], lo: np.ndarray, ext: float, bits=16) -> dict:
    """按实际编码器回算几何量化误差（与写出走同一码型，不是估算）。"""
    lv = MAX_U16 if bits >= 16 else 2 ** bits - 1
    err = 0.0
    for p in prims:
        pos = np.asarray(p["pos"], np.float64)
        rec = normalized_decode(normalized_encode(pos, lo, ext, lv), lo, ext, lv)
        err = max(err, max_abs_error(pos, rec))
    return {"geom_quant_step": round(normalized_step(ext, lv), 8),
            "geom_quant_max_err": round(err, 8), "geom_quant_bits": int(bits)}


def _glb_mesh_bytes(prims: list[dict], images: list[tuple[bytes, str]], mats_tex: list[int | None],
                    quantize=True, bounds: tuple | None = None) -> bytes:
    """组装带纹理的 glTF 三角网格 GLB（KHR_mesh_quantization u16 位置 + u16 UV）。"""
    data = bytearray()
    views: list[dict] = []

    def add(raw: bytes, target=None) -> int:
        off = len(data)
        data.extend(raw)
        while len(data) % 4:
            data.append(0)
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(raw)}
        if target:
            v["target"] = target
        views.append(v)
        return len(views) - 1

    lo, ext = bounds if bounds else geom_bounds(prims)
    acc: list[dict] = []
    gprims = []
    node: dict = {"mesh": 0}
    if quantize:
        node["translation"] = [float(x) for x in lo]
        node["scale"] = [ext] * 3
    for p in prims:
        pos = p["pos"]
        if quantize:
            # 每图元独立 POSITION 访问器（共享全局 lo/ext 量化网格，节点变换复原）
            q = normalized_encode(pos, lo, ext).astype("<u2")
            vi = add(q.tobytes(), 34962)
            acc.append({"bufferView": vi, "componentType": 5123, "count": len(q), "type": "VEC3",
                        "normalized": True, "min": [0, 0, 0], "max": [MAX_U16, MAX_U16, MAX_U16]})
            ai = add(p["idx"].astype("<u4").tobytes(), 34963)
            acc.append({"bufferView": ai, "componentType": COMP_U4, "count": len(p["idx"]),
                        "type": "SCALAR"})
            attrs = {"POSITION": len(acc) - 2}
            ii = len(acc) - 1
        else:
            vi = add(pos.astype("<f4").tobytes(), 34962)
            acc.append({"bufferView": vi, "componentType": 5126, "count": len(pos), "type": "VEC3",
                        "min": [float(x) for x in pos.min(0)], "max": [float(x) for x in pos.max(0)]})
            ai = add(p["idx"].astype("<u4").tobytes(), 34963)
            acc.append({"bufferView": ai, "componentType": COMP_U4, "count": len(p["idx"]),
                        "type": "SCALAR"})
            attrs = {"POSITION": len(acc) - 2}
            ii = len(acc) - 1
        if p["uv"] is not None:
            uq = normalized_encode(np.clip(p["uv"], 0, 1), 0.0, 1.0).astype("<u2")
            ui = add(uq.tobytes(), 34962)
            acc.append({"bufferView": ui, "componentType": 5123, "count": len(uq), "type": "VEC2",
                        "normalized": True})
            attrs["TEXCOORD_0"] = len(acc) - 1
        gprims.append({"mode": 4, "material": p["mat"], "indices": ii, "attributes": attrs})
    images_js, textures = [], []
    for blob, mime in images:
        vi = add(blob)
        images_js.append({"bufferView": vi, "mimeType": mime})
    for i in range(len(images_js)):
        textures.append({"source": i})
    mats = []
    for ti in mats_tex:
        pbr: dict = {"metallicFactor": 0.0, "roughnessFactor": 1.0}
        if ti is not None:
            pbr["baseColorTexture"] = {"index": ti}
        mats.append({"pbrMetallicRoughness": pbr, "doubleSided": True})
    gltf = {
        "asset": {"version": "2.0", "generator": "GeoSqueeze"},
        "extensionsUsed": ["KHR_mesh_quantization"] if quantize else [],
        "extensionsRequired": ["KHR_mesh_quantization"] if quantize else [],
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [node],
        "meshes": [{"primitives": gprims}],
        "materials": mats,
        "textures": textures, "images": images_js,
        "samplers": [{"wrapS": 10497, "wrapT": 10497, "magFilter": 9729, "minFilter": 9987}],
        "accessors": acc, "bufferViews": views,
        "buffers": [{"byteLength": len(data)}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    bn = bytes(data)
    bn += b"\x00" * ((4 - len(bn) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(bn)
    out = struct.pack("<III", GLB_MAGIC, 2, total)
    out += struct.pack("<II", len(js), JSON_CHUNK) + js
    out += struct.pack("<II", len(bn), BIN_CHUNK) + bn
    return out


def write_b3dm(path: str, glb: bytes, batch_json: dict | None = None) -> int:
    # gltfUpAxis=2（Z）：Cesium 默认按 3D Tiles 规范把 b3dm 内嵌 glTF 当 Y-up 并自动旋转；
    # 我们的数据全程 Z-up，显式声明后 Cesium 跳过旋转（2 对应 Cesium Axis.Z）
    bj = json.dumps(batch_json if batch_json is not None else {"BATCH_LENGTH": 1, "gltfUpAxis": 2},
                    separators=(",", ":")).encode()
    bj += b" " * ((-len(bj)) % 8)
    total = 28 + len(bj) + len(glb)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sIIIIII", b"b3dm", 1, total, len(bj), 0, 0, 0))
        f.write(bj)
        f.write(glb)
    return total


def compress_mesh_tile(mesh: dict, prune=0.0, tex_max_dim=1024, tex_quality=80,
                       quantize=True) -> tuple[bytes, dict]:
    """单瓦片网格压缩：抽稀 + 量化 + 纹理重压缩 → glb 字节 + 评测信息。"""
    tex_info, new_images, texmap = [], [], {}
    for i, (blob, mime) in enumerate(mesh["images"]):
        nb, nm = compress_texture(blob, mime, tex_max_dim, tex_quality)
        psnr = texture_psnr(blob, nb)
        tex_info.append({"index": i, "mime": nm, "bytes_in": len(blob), "bytes_out": len(nb),
                         "psnr": round(psnr, 2)})
        texmap[i] = len(new_images)
        new_images.append((nb, nm))
    used_tex = sorted({t for t in mesh["mat_tex"] if t is not None})
    tex_re = {old: new for new, old in enumerate(used_tex)}
    mats_tex = [tex_re[t] if t is not None else None for t in mesh["mat_tex"]]
    prims_out, v_in, v_out, t_in, t_out = [], 0, 0, 0, 0
    for p in mesh["prims"]:
        v_in += len(p["pos"])
        t_in += len(p["idx"]) // 3
        pos, idx, uv = decimate_mesh(p["pos"], p["idx"], p["uv"], prune)
        if len(idx) == 0:
            pos, idx, uv = decimate_mesh(p["pos"], p["idx"], p["uv"], 0.0)   # 兜底：不许清空
        prims_out.append({"pos": pos, "idx": idx, "uv": uv, "mat": p["mat"]})
        v_out += len(pos)
        t_out += len(idx) // 3
    lo, ext = geom_bounds(prims_out)
    glb = _glb_mesh_bytes(prims_out, new_images, mats_tex, quantize, bounds=(lo, ext))
    stats = {"verts_in": v_in, "verts_out": v_out, "tris_in": t_in, "tris_out": t_out,
             "tex": tex_info, "bytes": len(glb)}
    if quantize and prims_out:
        stats.update(geom_quant_stats(prims_out, lo, ext))   # 统一量化口径（见 quant.py）
    else:
        stats.update({"geom_quant_step": 0.0, "geom_quant_max_err": 0.0, "geom_quant_bits": 0})
    return glb, stats
