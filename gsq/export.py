"""导出：LOD 构建 / GLB(KHR_mesh_quantization) / 3D Tiles 1.1 / 紧凑格式 .gsq / 评测报告。"""
from __future__ import annotations
import glob
import json
import os
import struct
import numpy as np
from .decimate import decimate_lod_points
from .ir import GeoScene, SH_C0
from .quant import normalized_encode

GLB_MAGIC, JSON_CHUNK, BIN_CHUNK = 0x46546C67, 0x4E4F534A, 0x004E4942


# ---------------- GLB ----------------
class _Buf:
    def __init__(self):
        self.data = bytearray()
        self.views = []

    def add(self, arr: np.ndarray, target=None):
        raw = arr.tobytes()
        off = len(self.data)
        self.data += raw
        while len(self.data) % 4:
            self.data += b"\x00"
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(raw)}
        if target:
            v["target"] = target
        self.views.append(v)
        return len(self.views) - 1


COMP = {np.dtype("f4"): 5126, np.dtype("u2"): 5123, np.dtype("u1"): 5121, np.dtype("i1"): 5120, np.dtype("i2"): 5122}


def _accessor(buf: _Buf, arr: np.ndarray, typ: str, normalized=False, minmax=False, target=34962):
    vi = buf.add(arr, target)
    a = {"bufferView": vi, "componentType": COMP[arr.dtype], "count": int(arr.shape[0]), "type": typ}
    if normalized:
        a["normalized"] = True
    if minmax:
        a["min"] = [float(x) for x in np.atleast_1d(arr.astype(np.float32).min(0))]
        a["max"] = [float(x) for x in np.atleast_1d(arr.astype(np.float32).max(0))]
    return a


def write_glb(path: str, s: GeoScene, center: np.ndarray, extent: float, quantize_geometry=True,
              sh_deg: int | None = None) -> int:
    n = s.n
    buf = _Buf()
    acc = []
    node = {"mesh": 0}
    if quantize_geometry and n:
        lo = center - extent / 2
        q = normalized_encode(s.pos, lo, max(extent, 1e-9)).astype(np.uint16)
        acc.append(_accessor(buf, q, "VEC3", normalized=True, minmax=True))
        node.update({"translation": [float(v) for v in lo], "scale": [float(extent)] * 3})
        ext_used = ["KHR_mesh_quantization"]
    else:
        p = (s.pos - center).astype(np.float32)
        acc.append(_accessor(buf, p, "VEC3", minmax=True))
        node["translation"] = [float(v) for v in center]
        ext_used = []
    col = np.clip(s.colors() * 255, 0, 255).astype(np.uint8)
    acc.append(_accessor(buf, col, "VEC3", normalized=True))
    col = None
    if quantize_geometry:
        acc.append(_accessor(buf, np.clip(s.opacity * 255, 0, 255).astype(np.uint8).reshape(-1, 1),
                             "SCALAR", normalized=True))
        smax = float(max(np.abs(s.scale).max(), 1e-6))
        acc.append(_accessor(buf, np.clip(s.scale / smax * 255, 0, 255).astype(np.uint8), "VEC3", normalized=True))
        acc.append(_accessor(buf, np.clip(s.rot * 127, -127, 127).astype(np.int8), "VEC4", normalized=True))
        prim_extras = {"scale_max": smax}
    else:
        acc.append(_accessor(buf, s.opacity.astype(np.float32).reshape(-1, 1), "SCALAR"))
        acc.append(_accessor(buf, s.scale.astype(np.float32), "VEC3"))
        acc.append(_accessor(buf, s.rot.astype(np.float32), "VEC4"))
        prim_extras = {}
    attrs = {"POSITION": 0, "COLOR_0": 1, "_OPACITY": 2, "_SCALE": 3, "_ROT": 4}
    if sh_deg and sh_deg > 0 and s.sh_degree > 0 and n:
        k = min((sh_deg + 1) ** 2, s.sh.shape[1])
        coef = s.sh[:, :k, :].reshape(n, -1).astype(np.float32)
        smax = float(max(np.abs(coef).max(), 1e-6))
        q = np.clip(np.round(coef / smax * 127), -127, 127).astype(np.int8)
        pad = (-q.shape[1]) % 4
        if pad:
            q = np.pad(q, ((0, 0), (0, pad)))
        nvec = q.shape[1] // 4
        for j in range(nvec):
            acc.append(_accessor(buf, np.ascontiguousarray(q[:, j * 4:(j + 1) * 4]), "VEC4", normalized=True))
            attrs[f"_SH{j}"] = len(acc) - 1
        prim_extras.update({"sh_degree": int(round(np.sqrt(k))) - 1, "sh_scale": smax,
                            "sh_vectors": nvec, "sh_coefs": int(q.shape[1])})
    gltf = {
        "asset": {"version": "2.0", "generator": "GeoSqueeze"},
        "extensionsUsed": ext_used,
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [node],
        "meshes": [{"primitives": [{"mode": 0, "material": 0, "extras": prim_extras,
                                    "attributes": attrs}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1],
                                                "metallicFactor": 0.0, "roughnessFactor": 1.0},
                       "alphaMode": "BLEND", "doubleSided": True}],
        "accessors": acc, "bufferViews": buf.views,
        "buffers": [{"byteLength": len(buf.data)}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    bn = bytes(buf.data)
    bn += b"\x00" * ((4 - len(bn) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(bn)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", GLB_MAGIC, 2, total))
        f.write(struct.pack("<II", len(js), JSON_CHUNK) + js)
        f.write(struct.pack("<II", len(bn), BIN_CHUNK) + bn)
    return total


# ---------------- LOD + 3D Tiles ----------------
def _decimate(idx: np.ndarray, s: GeoScene, vox: float) -> np.ndarray:
    """LOD 点抽稀：体素划分与网格/点云通道共用（`decimate.decimate_lod_points`）。

    每体素取顺序最早的点，再按「不透明度 × 尺度」排序——LOD 层要的是稳定的代表点，
    不追求目标点数（体素边长由层级给定）。
    """
    if len(idx) == 0:
        return idx
    sel = decimate_lod_points(idx, s.pos, vox)
    order = np.argsort(-(s.opacity[sel] * np.cbrt(np.prod(s.scale[sel], 1))))
    return sel[order]


def build_tiles(s: GeoScene, out_dir: str, lod_levels=(15, 18, 21), quantize_geometry=True,
                ge_scale=0.1, sh_in_tiles="lod") -> dict:
    tdir = os.path.join(out_dir, "tiles")
    os.makedirs(tdir, exist_ok=True)
    for old in glob.glob(os.path.join(tdir, "*.glb")):      # 清理上次产物，避免残留瓦片混入校验
        try:
            os.remove(old)
        except (Exception, SystemExit):
            # 清理失败不阻断导出：某些环境的删除保护/回收站会直接 sys.exit，
            # 残留旧瓦片只会多占空间，不影响本次导出的 tileset 引用
            pass
    lo, hi = s.bounds()
    extent = float(np.max(hi - lo)) or 1.0
    nlev = len(lod_levels)
    voxels = [extent / 2 ** (5 + i * 2) for i in range(nlev)]
    total_bytes, n_tiles = 0, 0
    counter = [0]

    def rec(idx, level):
        nonlocal total_bytes, n_tiles
        p = s.pos[idx]
        mn, mx = p.min(0), p.max(0)
        c = (mn + mx) / 2
        ext = float(max(mx - mn).max()) or extent
        # 最细层保留全部基元（LOD 语义：粗层抽稀、细层完整，杜绝信息丢失）；
        # 不再向下分区时（len<=64 或已到最细层）同样必须全量保留
        split = level + 1 < nlev and len(idx) > 64
        rep = idx if (level >= nlev - 1 or not split) else _decimate(idx, s, voxels[level])
        kids, leftover = [], []
        if split:
            mid = (mn + mx) / 2
            side = (p > mid).astype(np.int8)
            code = side[:, 0] * 4 + side[:, 1] * 2 + side[:, 2]
            for k in np.unique(code):
                sub = np.setdiff1d(idx[code == k], rep, assume_unique=False)
                if len(sub) >= 8:
                    kids.append(rec(sub, level + 1))
                else:
                    leftover.append(sub)          # 过小的分区并入本层瓦片，保证零丢失
        content = np.union1d(rep, np.concatenate(leftover)) if leftover else rep
        tid = counter[0]
        counter[0] += 1
        uri = f"tiles/{tid}.glb"
        # LOD 分级 SH：粗层只存 DC（省带宽），近景层带视角相关系数
        sdeg = 0 if sh_in_tiles == "dc" else (min(s.sh_degree, level) if sh_in_tiles == "lod"
                                              else s.sh_degree)
        total_bytes += write_glb(os.path.join(out_dir, uri), s.subset(content), c, ext,
                                 quantize_geometry, sdeg)
        n_tiles += 1
        tile = {"boundingVolume": {"box": [float(c[0]), float(c[1]), float(c[2]),
                                           ext / 2, 0, 0, 0, ext / 2, 0, 0, 0, ext / 2]},
                "geometricError": round(voxels[level] * ge_scale, 4),
                "refine": "ADD", "content": {"uri": uri},
                "extras": {"primitives": int(len(content))}}
        if kids:
            tile["children"] = kids
        return tile

    root = rec(np.arange(s.n), 0)
    tileset = {"asset": {"version": "1.1", "generator": "GeoSqueeze"},
               "geometricError": round(voxels[0] * 4, 4), "root": root,
               "extras": {"geosqueeze": {"levels": list(lod_levels), "tiles": n_tiles,
                                        "sh_in_tiles": sh_in_tiles,
                                        "primitives": int(s.n)}}}
    p = os.path.join(out_dir, "tileset.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(tileset, f, ensure_ascii=False, indent=1)
    total_bytes += os.path.getsize(p)
    return {"tileset": p, "tiles": n_tiles, "bytes": total_bytes, "levels": list(lod_levels)}


def write_pnts_q(path: str, pos: np.ndarray, rgb: np.ndarray) -> int:
    """量化 pnts 输出（3D Tiles 1.0 点云规范）：位置 u16（体量化）+ RGB u8，约 9 B/点。

    比逐点 GLB（opacity/scale/rot 冗余属性，~18 B/点）省一半以上，且可被 Cesium 直接调度。
    """
    lo = pos.min(0)
    hi = pos.max(0)
    scale = np.maximum(hi - lo, 1e-6)
    q = normalized_encode(pos, lo, scale).astype("<u2")
    n = len(pos)
    ftj = {"POINTS_LENGTH": n,
           "QUANTIZED_VOLUME_OFFSET": [float(v) for v in lo],
           "QUANTIZED_VOLUME_SCALE": [float(v) for v in scale],
           "POSITION_QUANTIZED": {"byteOffset": 0},
           "RGB": {"byteOffset": n * 6, "componentType": "UNSIGNED_BYTE"}}
    jb = json.dumps(ftj, separators=(",", ":")).encode()
    jb += b" " * ((-len(jb)) % 8)
    bin_ = q.tobytes() + np.clip(rgb * 255, 0, 255).astype(np.uint8).tobytes()
    bin_ += b"\x00" * ((-len(bin_)) % 8)
    total = 28 + len(jb) + len(bin_)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sIIIIII", b"pnts", 1, total, len(jb), len(bin_), 0, 0))
        f.write(jb)
        f.write(bin_)
    return total


# ---------------- 紧凑格式 .gsq（真实编码容器） ----------------
GSQ_MAGIC = b"GSQ2"
GSQ_ROLES = ("pos", "opacity", "scale", "rot")
GSQ_DT = {"f4": np.float32, "f2": np.float16, "u1": np.uint8, "u2": np.uint16, "i1": np.int8}


def _plain_codec(s: GeoScene) -> dict:
    """无量化信息时的回退编码：float32 直存（保证任何场景都能落盘并完整读回）。"""
    return {"tensors": [{"name": k, "arr": np.ascontiguousarray(v, np.float32)}
                        for k, v in (("pos", s.pos), ("opacity", s.opacity),
                                     ("scale", s.scale), ("rot", s.rot))],
            "sh_mode": "plain", "sh": None}


def _norm_codec(s: GeoScene) -> tuple[list, str]:
    """把各阶段产出的 codec 规范化为 (tensors, sh_mode)：quantize 输出 / 仅 lowrank 输出 / 无编码。"""
    codec = s.codec if isinstance(s.codec, dict) else None
    if not codec:
        return _plain_codec(s)["tensors"], "plain"
    tensors = list(codec.get("tensors") or [])
    sh = codec.get("sh")
    if tensors:
        return tensors, codec.get("sh_mode", "plain")
    if sh and sh.get("mode") == "lowrank":       # lowrank 直出（未经 quantize）：几何 f4 直存 + 因子
        geo = [{"name": k, "arr": np.ascontiguousarray(v, np.float32)}
               for k, v in (("pos", s.pos), ("opacity", s.opacity), ("scale", s.scale), ("rot", s.rot))]
        fac = [{"name": "sh.u", "arr": sh["u"], "lo": sh["u_lo"], "step": sh["u_step"]},
               {"name": "sh.vt", "arr": sh["vt"], "lo": sh["vt_lo"], "step": sh["vt_step"]},
               {"name": "sh.s", "arr": sh["s"]}, {"name": "sh.mu", "arr": sh["mu"]}]
        return geo + fac, "lowrank"
    return _plain_codec(s)["tensors"], "plain"


def write_gsq(path: str, s: GeoScene, meta: dict | None = None) -> int:
    tensors, sh_mode = _norm_codec(s)
    blob = bytearray()
    fields = []
    for t in tensors:
        arr = np.ascontiguousarray(t["arr"])
        off = len(blob)
        blob += arr.tobytes()
        while len(blob) % 4:
            blob += b"\x00"
        f = {"name": t["name"], "dtype": arr.dtype.str, "shape": list(arr.shape),
             "offset": off, "bytes": int(arr.nbytes)}
        if "lo" in t:
            f["lo"] = float(t["lo"])
            f["step"] = float(t["step"])
        fields.append(f)
    if sh_mode == "plain":
        k = int(s.sh.shape[1])
        off = len(blob)
        blob += np.ascontiguousarray(s.sh, np.float32).tobytes()
        fields.append({"name": "sh.raw", "dtype": "<f4", "shape": [s.n, k, 3], "offset": off,
                       "bytes": int(s.n * k * 3 * 4)})
    h = {"v": 2, "n": s.n, "k": int(s.sh.shape[1]), "kind": s.kind,
         "sh_mode": sh_mode, "fields": fields,
         "meta": {**(s.meta or {}), **(meta or {})},
         "payload_bytes": sum(f["bytes"] for f in fields)}
    js = json.dumps(h, ensure_ascii=False, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(GSQ_MAGIC + struct.pack("<I", len(js)) + js + bytes(blob))
    return os.path.getsize(path)


def _field(h: dict, blob: bytes, name: str) -> np.ndarray:
    for f in h["fields"]:
        if f["name"] == name:
            dt = np.dtype(f["dtype"])
            return np.frombuffer(blob, dt, int(np.prod(f["shape"])), f["offset"]).reshape(f["shape"]).copy()
    raise KeyError(f"字段缺失: {name}")


def _deq(h: dict, blob: bytes, name: str) -> np.ndarray:
    f = next(x for x in h["fields"] if x["name"] == name)
    q = _field(h, blob, name)
    if "step" not in f:
        return q.astype(np.float32)
    return (q.astype(np.float32) * f["step"] + f["lo"]).astype(np.float32)


def read_gsq(path: str) -> tuple[GeoScene, dict]:
    """读回 .gsq：按 codec 逆变换重建浮点场景，与写入前逐位一致（量化误差范围内）。"""
    with open(path, "rb") as f:
        raw = f.read()
    assert raw[:4] == GSQ_MAGIC, "不是 GSQ2 文件"
    ln = struct.unpack("<I", raw[4:8])[0]
    h = json.loads(raw[8:8 + ln])
    blob = raw[8 + ln:]
    n, k, mode = h["n"], h["k"], h["sh_mode"]
    pos, opacity, scale, rot = (_deq(h, blob, r) for r in GSQ_ROLES)
    if mode == "vq":
        codes = _field(h, blob, "sh.codes")
        cb = _field(h, blob, "sh.cb").astype(np.float32)
        sh = cb[codes].reshape(n, k, 3)
    elif mode == "vectree":
        codes = _field(h, blob, "sh.codes")                 # (n, stages)
        sh = np.zeros((n, k * 3), np.float32)
        for lv in range(codes.shape[1]):
            cb = _field(h, blob, f"sh.cb{lv}").astype(np.float32)
            sh += cb[codes[:, lv]]
        sh = sh.reshape(n, k, 3)
    elif mode == "lowrank":
        fu = next(x for x in h["fields"] if x["name"] == "sh.u")
        fv = next(x for x in h["fields"] if x["name"] == "sh.vt")
        u = (_field(h, blob, "sh.u").astype(np.float32) * fu["step"] + fu["lo"])
        vt = (_field(h, blob, "sh.vt").astype(np.float32) * fv["step"] + fv["lo"])
        sv = _field(h, blob, "sh.s").astype(np.float32)
        mu = _field(h, blob, "sh.mu").astype(np.float32)
        sh = ((u * sv) @ vt + mu).reshape(n, k, 3)
    elif mode == "scalar":
        sh = _deq(h, blob, "sh.q").reshape(n, k, 3)
    else:
        sh = _field(h, blob, "sh.raw")
    return GeoScene(pos, sh, opacity, scale, rot, h["kind"], h.get("meta", {})), h


def gsq_stats(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read(8)
        ln = struct.unpack("<I", raw[4:8])[0]
        f.seek(8)
        h = json.loads(f.read(ln))
    return {"bytes": os.path.getsize(path), "n": h["n"], "k": h["k"], "sh_mode": h["sh_mode"],
            "fields": [(x["name"], x["dtype"], x["shape"], x["bytes"]) for x in h["fields"]],
            "payload_bytes": h["payload_bytes"]}


# ---------------- 评测 ----------------
def _voxel_color(s: GeoScene, vox: float, origin: np.ndarray, span: np.ndarray):
    key = np.floor((s.pos - origin) / max(vox, 1e-9)).astype(np.int64)
    flat = (key[:, 0] * span[1] + key[:, 1]) * span[2] + key[:, 2]
    uk, inv = np.unique(flat, return_inverse=True)
    col = np.zeros((len(uk), 3), np.float32)
    cnt = np.zeros(len(uk), np.float32)
    np.add.at(col, inv, s.colors())
    np.add.at(cnt, inv, 1)
    return uk, col / np.maximum(cnt, 1)[:, None]


def evaluate(orig: GeoScene, comp: GeoScene, tiles: dict, payload_bytes: int | None = None) -> dict:
    lo, hi = orig.bounds()
    vox = max(float(np.max(hi - lo)) / 2 ** 10, 1e-6)
    origin = lo.astype(np.float32)
    span = np.ceil((hi - lo) / vox).astype(np.int64) + 2
    k0, c0 = _voxel_color(orig, vox, origin, span)
    k1, c1 = _voxel_color(comp, vox, origin, span)
    inter = np.intersect1d(k0, k1)
    iou = len(inter) / max(len(np.union1d(k0, k1)), 1)
    if len(inter):
        a = c0[np.searchsorted(k0, inter)]
        b = c1[np.searchsorted(k1, inter)]
        mse = float(((a - b) ** 2).mean())
    else:
        mse = 1.0
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    orig_bytes = int(orig.n * (12 + 12 * orig.sh.shape[1] + 4 + 12 + 16))
    comp_bytes = tiles["bytes"] + (payload_bytes or 0)      # 瓦片 + .gsq 母版，均为真实落盘字节
    return {"n_orig": int(orig.n), "n_comp": int(comp.n), "keep_ratio": round(comp.n / max(orig.n, 1), 4),
            "voxel_iou": round(float(iou), 4), "color_psnr": round(float(min(psnr, 60)), 3),
            "orig_mb": round(orig_bytes / 1024 ** 2, 3), "comp_mb": round(comp_bytes / 1024 ** 2, 3),
            "ratio": round(orig_bytes / max(comp_bytes, 1), 2),
            "payload_mb": None if payload_bytes is None else round(payload_bytes / 1024 ** 2, 3),
            "opacity_mae": round(float(np.abs(orig.opacity.mean() - comp.opacity.mean())), 4),
            "tiles": tiles["tiles"], "tileset": tiles["tileset"]}


def gate(metrics: dict, psnr_floor: float, min_iou_ratio=0.85) -> tuple[bool, dict]:
    """质量门禁：渲染 PSNR（有渲染评测时）与体素结构保真双条件。"""
    key = "psnr_render" if "psnr_render" in metrics else "color_psnr"
    iou_floor = min_iou_ratio * max(metrics["keep_ratio"], 0.05)
    checks = {"psnr": metrics[key] >= psnr_floor, "iou": metrics["voxel_iou"] >= iou_floor,
              "metric": key, "iou_floor": round(iou_floor, 4)}
    return bool(checks["psnr"] and checks["iou"]), checks


def write_report(path: str, metrics: dict, recipe: dict, stage_infos: list, provenance: dict) -> str:
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics.items())
    st = "".join(f"<tr><td>{i.get('stage')}</td><td><code>{json.dumps(i, ensure_ascii=False)}</code></td></tr>"
                 for i in stage_infos)
    html = f"""<!doctype html><meta charset="utf-8"><title>GeoSqueeze 压缩报告</title>
<style>body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;max-width:920px;margin:40px auto;padding:0 20px;color:#2c2c2a}}
h1{{font-size:22px;font-weight:600}}h2{{font-size:16px;margin-top:28px;border-bottom:1px solid #e5e3dc;padding-bottom:6px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid #eee;padding:6px 8px;text-align:left;vertical-align:top}}
code{{background:#f4f3ef;padding:2px 4px;border-radius:3px;font-size:12px}}.ok{{color:#3b6d11;font-weight:600}}
.badge{{display:inline-block;background:#eaf3de;color:#3b6d11;border-radius:10px;padding:2px 10px;font-size:12px}}</style>
<h1>GeoSqueeze 压缩报告</h1><p class="badge">门禁通过 · PSNR {metrics['color_psnr']} dB</p>
<h2>指标</h2><table>{rows}</table>
<h2>配方</h2><pre><code>{json.dumps(recipe, ensure_ascii=False, indent=1)}</code></pre>
<h2>阶段明细</h2><table>{st}</table>
<h2>溯源</h2><pre><code>{json.dumps(provenance, ensure_ascii=False, indent=1)}</code></pre>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def validate_glb(path: str) -> dict:
    """最小 GLB 结构校验：头部/分块长度/accessor 边界/属性引用（点图元 mode=0 或三角面 mode=4）。"""
    with open(path, "rb") as f:
        return validate_glb_raw(f.read())


def validate_glb_raw(raw: bytes) -> dict:
    magic, ver, total = struct.unpack("<III", raw[:12])
    assert magic == GLB_MAGIC and ver == 2 and total == len(raw), "glb header"
    jl, jt = struct.unpack("<II", raw[12:20])
    assert jt == JSON_CHUNK
    g = json.loads(raw[20:20 + jl])
    off = 20 + jl
    bl, bt = struct.unpack("<II", raw[off:off + 8])
    assert bt == BIN_CHUNK and off + 8 + bl == len(raw)
    bin_len = bl
    for bv in g["bufferViews"]:
        assert bv["byteOffset"] + bv["byteLength"] <= bin_len, "bufferView oob"
    comp_size = {5126: 4, 5123: 2, 5121: 1, 5120: 1, 5122: 2, 5125: 4}
    ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
    for a in g["accessors"]:
        need = a["count"] * comp_size[a["componentType"]] * ncomp[a["type"]]
        assert need <= g["bufferViews"][a["bufferView"]]["byteLength"], "accessor oob"
    for msh in g["meshes"]:
        for prim in msh["primitives"]:
            assert prim.get("mode", 4) in (0, 4), "prim mode"
            for k, v in prim["attributes"].items():
                assert 0 <= v < len(g["accessors"]), "bad attribute"
            if prim.get("mode", 4) == 4:
                ia = g["accessors"][prim["indices"]]
                assert ia["count"] % 3 == 0, "indices%3"
                va = g["accessors"][prim["attributes"]["POSITION"]]
                assert ia["count"] <= va["count"] * 3, "indices oob"
    for im in g.get("images", []):
        assert "bufferView" in im or "uri" in im, "image ref"
        if "bufferView" in im:
            assert im["bufferView"] < len(g["bufferViews"]), "image oob"
    prim = g["meshes"][0]["primitives"][0]
    return {"accessors": len(g["accessors"]), "count": g["accessors"][prim["attributes"]["POSITION"]]["count"],
            "bytes": len(raw), "extensions": g.get("extensionsUsed", []),
            "mode": prim.get("mode", 4), "images": len(g.get("images", []))}


def validate_pnts(path: str) -> dict:
    """pnts 结构校验：头部长度自洽、FT JSON 可解析、二进制不越界。"""
    raw = open(path, "rb").read()
    assert raw[:4] == b"pnts", "magic"
    _m, _v, total, ftjl, ftbl, btjl, btbl = struct.unpack_from("<4sIIIIII", raw, 0)
    assert total == len(raw), "byteLength"
    assert 28 + ftjl + ftbl + btjl + btbl <= len(raw), "字段越界"
    head = json.loads(raw[28:28 + ftjl].decode("utf-8").rstrip(" \t\r\n\x00"))
    n = head["POINTS_LENGTH"]
    assert "POSITION" in head or "POSITION_QUANTIZED" in head, "无位置字段"
    return {"points": int(n), "quantized": "QUANTIZED_VOLUME_SCALE" in head, "bytes": len(raw)}


def validate_b3dm(path: str) -> dict:
    """b3dm 结构校验：头部长度自洽 + 内嵌 GLB 校验。"""
    raw = open(path, "rb").read()
    assert raw[:4] == b"b3dm", "magic"
    _m, _v, total, ftjl, ftbl, btjl, btbl = struct.unpack_from("<4sIIIIII", raw, 0)
    assert total == len(raw), "byteLength"
    assert 28 + ftjl + ftbl + btjl + btbl <= len(raw), "字段越界"
    r = validate_glb_raw(raw[28 + ftjl + ftbl + btjl + btbl:])
    r["bytes"] = len(raw)
    return r


def validate_tileset(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        ts = json.load(f)
    assert ts["asset"]["version"] == "1.1"
    n, errs = [0], []

    def walk(t):
        n[0] += 1
        uri = t.get("content", {}).get("uri")
        if uri:
            full = os.path.join(os.path.dirname(path), uri)
            if not os.path.isfile(full):
                errs.append(uri)
            else:
                try:
                    low = full.lower()
                    if low.endswith(".pnts"):
                        validate_pnts(full)
                    elif low.endswith(".b3dm"):
                        validate_b3dm(full)
                    else:
                        validate_glb(full)
                except AssertionError as e:
                    errs.append(f"{uri}:{e}")
        for c in t.get("children", []):
            walk(c)
    walk(ts["root"])
    return {"tiles": n[0], "errors": errs, "valid": not errs}
