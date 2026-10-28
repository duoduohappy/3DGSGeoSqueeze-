"""GeoSqueeze 测试工作台：在线 3D Tiles / 矢量地址加载 + 压缩包上传测试。

启动：python -m gsq testbench [--port 8619] [--work out/testbench]

能力（单页界面，四条通道）：
1. 输入在线 3D Tiles tileset.json 地址 → 后端反代（/p/<b64>/...，顺带解决跨域）→
   Cesium 直读渲染 + 结构统计
2. 输入在线矢量（GeoJSON）地址 → 下载 → compress-vector 压缩 + 评测 → 查看器渲染
3. 上传 3D Tiles zip → 解压校验 → 自动生成内嵌 cesium.html（≤8MB 单文件）/ 外链渲染
4. 上传矢量数据 zip（.geojson/.json）→ compress-vector → 查看器渲染

零第三方依赖（标准库 HTTP 服务 + zipfile + urllib），代理自动探测（env → 直连 →
127.0.0.1:7897 回退）。
"""
import base64
import io
import json
import os
import shutil
import struct
import time
import urllib.parse
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .cesium_viewer import _HTML as _CESIUM_HTML, write_cesium_viewer
from .vec_viewer import write_vector_viewer
from .vector import compress_vector
from .tiles_in import scan_tileset

_MAX_UPLOAD = 300 * 1024 * 1024  # 上传 zip 上限
_CACHE_MAX = 1024 ** 3           # 反代磁盘缓存上限（1GB），超出按最旧访问淘汰
_PRESETS = {
    "tiles": [
        ["Cesium 官方 · DiscreteLOD 点云（3 瓦片，可完整压缩）",
         "https://raw.githubusercontent.com/CesiumGS/3d-tiles-samples/refs/heads/main/1.0/TilesetWithDiscreteLOD/tileset.json"],
        ["GeoSqueeze 带纹理城市（16 b3dm，演示纹理+几何压缩）",
         "https://raw.githubusercontent.com/duoduohappy/3DGSGeoSqueeze-/main/out/mesh_city/tileset.json"],
        ["Cesium 官方 · TreeBillboards（i3dm 实例化，仅渲染）",
         "https://raw.githubusercontent.com/CesiumGS/3d-tiles-samples/refs/heads/main/1.0/TilesetWithTreeBillboards/tileset.json"],
        ["3DBAG 荷兰全国建筑 LoD2.2（仅渲染：meshopt 格式，本工具不解码）",
         "https://data.3dbag.nl/v20250903/cesium3dtiles/lod22/tileset.json"],
        ["3DBAG 荷兰全国建筑 LoD1.2（仅渲染：meshopt 格式）",
         "https://data.3dbag.nl/v20250903/cesium3dtiles/lod12/tileset.json"],
    ],
    "vector": [
        ["GeoSqueeze 合成城市矢量（本仓库 raw）",
         "https://raw.githubusercontent.com/duoduohappy/3DGSGeoSqueeze-/main/data/city.geojson"],
        ["世界国家边界（world.geo.json）",
         "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"],
    ],
}


def _proxy_handlers() -> list:
    """候选链路：env 代理 → 直连 → 本地常见代理端口（去重）。"""
    env = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") \
        or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    out = []
    if env:
        out.append(urllib.request.ProxyHandler({"http": env, "https": env}))
    out.append(urllib.request.ProxyHandler({}))  # 直连
    out.append(urllib.request.ProxyHandler({"http": "http://127.0.0.1:7897",
                                            "https": "http://127.0.0.1:7897"}))
    # 按 handler 的 __proxy__ 标记去重（直连与 env 相同值时）
    seen, uniq = set(), []
    for h in out:
        key = tuple(sorted(getattr(h, "proxies", {}).items()))
        if key not in seen:
            seen.add(key)
            uniq.append(h)
    return uniq


def _fetch(url: str, timeout: int = 45) -> bytes:
    """三条链路并发竞速取回远端字节——坏代理不再阻塞后续尝试。

    之前串行回退时，env 中已失效的代理会先挂满超时才轮到可用链路，
    导致浏览器端 Cesium 请求长时间 pending（界面卡在"加载 tileset.json…"）。
    """
    from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait

    handlers = _proxy_handlers()

    def attempt(h):
        req = urllib.request.Request(url, headers={"User-Agent": "GeoSqueeze-TestBench/1.0"})
        with urllib.request.build_opener(h).open(req, timeout=timeout) as r:
            return r.read()

    ex = ThreadPoolExecutor(max_workers=len(handlers))
    try:
        futs = {ex.submit(attempt, h): h for h in handlers}
        last = None
        while futs:
            done, _ = wait(list(futs), return_when=FIRST_COMPLETED)
            for f in done:
                futs.pop(f, None)
                try:
                    return f.result()
                except Exception as e:
                    last = e
        raise RuntimeError(f"获取 {url} 失败（env代理/直连/7897 均不可用）: {last}")
    finally:
        ex.shutdown(wait=False)


def _stats_tiles_json(data: bytes, url: str, fetch=None, max_external: int = 3) -> dict:
    """轻量解析 tileset.json：版本 / 瓦片数 / 内容格式分布 / 根包围盒。

    3D Tiles 1.1 的大规模数据集（如 3DBAG 全国）根层通常只做空间索引，内容以
    「外部 tileset」形式递归引用。此时区分统计：内容瓦片格式 + 外部 tileset 数量，
    并抽样抓取至多 max_external 个外部 tileset 以给出真实内容格式。
    """
    ts = json.loads(data)
    root = ts.get("root") or {}
    fmt: dict[str, int] = {}
    ext_list: list[str] = []
    n = [0]

    def walk(t):
        n[0] += 1
        ct = t.get("content")
        for c in (ct if isinstance(ct, list) else [ct] if ct else []):
            uri = str(c.get("uri") or c.get("url") or "")
            ext = os.path.splitext(uri)[1].lstrip(".").lower() or "?"
            if ext == "json":
                ext_list.append(uri)     # 外部 tileset，内容在别处
            else:
                fmt[ext] = fmt.get(ext, 0) + 1
        for c in (t.get("children") or []):
            walk(c)

    walk(root)
    if root.get("implicitTiling"):
        fmt["implicit"] = fmt.get("implicit", 0) + 1

    sampled = 0
    if fetch and ext_list:
        base = url.rsplit("/", 1)[0] + "/"
        for uri in ext_list[:max_external]:
            try:
                sub = json.loads(fetch(urllib.parse.urljoin(base, uri)))
            except Exception:
                continue
            sampled += 1
            stack = [sub.get("root") or {}]
            while stack:
                t = stack.pop()
                ct = t.get("content")
                for c in (ct if isinstance(ct, list) else [ct] if ct else []):
                    e = os.path.splitext(str(c.get("uri") or ""))[1].lstrip(".").lower() or "?"
                    if e != "json":
                        fmt[e] = fmt.get(e, 0) + 1
                stack.extend(x for x in (t.get("children") or []) if isinstance(x, dict))

    return {"url": url, "version": (ts.get("asset") or {}).get("version"),
            "gltfUpAxis": (ts.get("asset") or {}).get("gltfUpAxis"),
            "tiles": n[0], "formats": fmt,
            "external_tilesets": len(ext_list), "sampled": sampled,
            "root_bv": "box" if "box" in root.get("boundingVolume", {}) else
                       "region" if "region" in root.get("boundingVolume", {}) else
                       "sphere" if "sphere" in root.get("boundingVolume", {}) else "-"}


def _unzip_find_root(zb: bytes, work: str, vid: str, want: str) -> str | None:
    """解压 zip 到 work/<vid>/，返回包含 want 文件（如 tileset.json）的子目录。"""
    zf = zipfile.ZipFile(io.BytesIO(zb))
    dest = os.path.join(work, vid)
    members = [m for m in zf.namelist() if not m.endswith("/")]
    for m in members:
        p = os.path.normpath(os.path.join(dest, m))
        if not p.startswith(os.path.normpath(dest)):  # 防 zip-slip
            raise ValueError(f"zip 内非法路径: {m}")
    zf.extractall(dest)
    cands = [m for m in members if os.path.basename(m) == want]
    if not cands:
        return None
    root = os.path.dirname(min(cands, key=lambda x: x.count("/")))
    return os.path.join(dest, root) if root else dest


def _probe_compressible(path: str) -> None:
    """下载期快速探测：瓦片含 meshopt/Draco 等必需扩展时立即失败（避免白下载整站）。"""
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".glb", ".b3dm"):
        return
    from .tiles_in import _glb_chunks
    raw = open(path, "rb").read()
    if ext == ".b3dm":
        _m, _v, _bl, ftjl, ftbl, btjl, btbl = struct.unpack_from("<4sIIIIII", raw, 0)
        raw = raw[28 + ftjl + ftbl + btjl + btbl:]
    _glb_chunks(raw)   # 遇到不支持的必需扩展时抛 ValueError（信息已含解决方案）


def _download_tileset(url: str, dest: str, max_bytes: int, fetch,
                      deadline: float | None = None) -> tuple[int, bool]:
    """下载 tileset（含外部子 tileset 与内容文件）到本地，保持相对路径结构。

    返回 (总字节数, 是否完整)；超过 max_bytes 或 deadline 时停止并返回 False。
    单个瓦片若为 meshopt/Draco 压缩几何，立即抛错（这类数据本工具不解码）。
    """
    os.makedirs(dest, exist_ok=True)
    total = [0]
    seen: set[str] = set()
    complete = [True]

    def dl(remote: str, rel: str, depth: int = 0) -> None:
        rel = rel.replace("\\", "/").lstrip("/")
        if rel in seen or depth > 12 or not rel:
            return
        seen.add(rel)
        p = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(p) or dest, exist_ok=True)
        if os.path.isfile(p):
            data = open(p, "rb").read()
        else:
            if total[0] > max_bytes or (deadline and time.time() > deadline):
                complete[0] = False
                return
            data = fetch(remote)
            total[0] += len(data)
            with open(p, "wb") as f:
                f.write(data)
            if not rel.lower().endswith(".json"):
                _probe_compressible(p)      # 不可压格式 → 立刻抛出，不再继续下载
                return
        if not rel.lower().endswith(".json"):
            return
        try:
            ts = json.loads(data)
        except ValueError:
            return
        base = remote.rsplit("/", 1)[0] + "/"
        parent = os.path.dirname(rel)
        stack = [ts.get("root") or {}]
        while stack:
            t = stack.pop()
            if not isinstance(t, dict):
                continue
            ct = t.get("content")
            for c in (ct if isinstance(ct, list) else [ct] if ct else []):
                uri = str(c.get("uri") or c.get("url") or "").split("?")[0]
                if not uri:
                    continue
                if total[0] > max_bytes:
                    complete[0] = False
                    continue
                dl(urllib.parse.urljoin(base, uri),
                   os.path.normpath(os.path.join(parent, uri)).replace("\\", "/"), depth + 1)
            stack.extend(x for x in (t.get("children") or []) if isinstance(x, dict))

    dl(url, "tileset.json")
    return total[0], complete[0]


class _Bench:
    def __init__(self, work: str):
        self.work = os.path.abspath(work)
        os.makedirs(self.work, exist_ok=True)
        self.json_cache: dict[str, bytes] = {}
        self._n_write = 0

    def new_id(self) -> str:
        return time.strftime("%H%M%S") + f"{int(time.time()*1000) % 1000:03d}"

    # ---- 在线 3D Tiles：校验 + 反代视图 ----
    def tiles_url(self, url: str) -> dict:
        data = self.json_cache.get(url) or _fetch(url)
        self.json_cache[url] = data
        st = _stats_tiles_json(data, url, fetch=_fetch)
        b64 = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
        return {"stats": st, "view": f"/cesium.html?ts=/p/{b64}/tileset.json"}

    # ---- 在线 3D Tiles：下载 → GeoSqueeze 压缩 → 前后对比 ----
    def tiles_compress(self, url: str, prune: float = 0.5, pos_bits: int = 16,
                       quantize: bool = True, keep_mesh: bool = True,
                       tex_max_dim: int = 512, tex_quality: int = 78,
                       max_mb: int = 60) -> dict:
        from .tiles_in import compress_tileset

        vid = self.new_id()
        src = os.path.join(self.work, vid, "src")
        out = os.path.join(self.work, vid, "out")
        nbytes, complete = _download_tileset(url, src, int(max_mb) * 1024 * 1024, _fetch,
                                             deadline=time.time() + 150)
        res = compress_tileset(src, out, prune=float(prune), pos_bits=int(pos_bits),
                               quantize=bool(quantize), keep_mesh=bool(keep_mesh),
                               tex_max_dim=int(tex_max_dim), tex_quality=int(tex_quality),
                               verbose=False)
        # 逐阶段如实报告：哪些阶段真的跑了、参数是什么、效果多少
        is_mesh = bool(res.get("mesh_out"))
        stages = [
            {"stage": "剪枝 prune", "executed": True,
             "detail": f"比例 {prune}（1-比例=输出保留率）",
             "effect": (f"三角形 {res['tris_in']:,} → {res['tris_out']:,}"
                        f"（保留 {res['tri_keep'] * 100:.1f}%）") if is_mesh else
                       f"点 {res['pts_in']:,} → {res['pts_out']:,}"
                       f"（保留 {res['point_ratio'] and 1 / res['point_ratio'] * 100:.1f}%）"},
            {"stage": "蒸馏 distill",
             "executed": False,
             "detail": "本路径不执行",
             "effect": "球谐阶数降阶只作用于带球谐系数的 3DGS 数据（PLY/LAS 主链路 `gsq compress`）；"
                       + ("该瓦片为网格，无球谐可降。" if is_mesh else "该瓦片点云不含球谐系数。")},
            {"stage": "量化 quantize", "executed": bool(quantize),
             "detail": f"位置 {pos_bits} bit" + ("（KHR_mesh_quantization）" if is_mesh else ""),
             "effect": (f"几何量化步长 {res['geom_quant_step'] * 1000:.2f} mm · "
                        f"实测最大误差 {res.get('geom_quant_max_err', 0) * 1000:.2f} mm")},
        ]
        if res.get("tex_bytes_in"):
            stages.append({"stage": "纹理重压缩", "executed": True,
                           "detail": f"最长边 {tex_max_dim}px · 质量 {tex_quality}",
                           "effect": f"{res['tex_bytes_in'] / 1024:.0f}KB → "
                                     f"{res['tex_bytes_out'] / 1024:.0f}KB（{res['tex_ratio']}x）· "
                                     f"PSNR {res['tex_psnr_min']}~{res['tex_psnr_mean']}dB"})
        page = res.get("cesium_page") or ""
        rel = os.path.relpath(page, out).replace("\\", "/") if page else ""
        b64 = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
        return {"metrics": {k: res[k] for k in
                            ("tiles_in", "tiles_out", "pts_in", "pts_out", "bytes_in",
                             "bytes_out", "ratio", "point_ratio", "mesh_out", "tris_in",
                             "tris_out", "tri_keep", "tex_bytes_in", "tex_bytes_out",
                             "tex_ratio", "tex_psnr_min", "tex_psnr_mean",
                             "geom_quant_step", "geom_quant_max_err") if k in res},
                "stages": stages, "download_mb": round(nbytes / 1024 ** 2, 2),
                "complete": complete,
                "view": f"/w/{vid}/out/{rel}" if rel else None,
                "orig_view": f"/cesium.html?ts=/p/{b64}/tileset.json"}

    # ---- 3DGS / 点云 → 主链路（剪枝 + 蒸馏 + 量化，唯一会做球谐降阶的路径） ----
    def primitives_compress(self, name: str, data: bytes, platform: str = "web",
                            budget_mb: float = 120.0, psnr_floor: float = 28.0,
                            rounds: int = 3, target_degree: int | None = None) -> dict:
        from .analyze import plan, profile
        from .cesium_viewer import write_cesium_viewer
        from .cli import _load
        from .pipeline import run_pipeline

        ext = os.path.splitext(name)[1].lower() or ".ply"
        if ext not in (".ply", ".las", ".laz", ".gsq"):
            raise ValueError(f"不支持的输入格式 {ext}（支持 .ply / .las / .laz / .gsq）")
        vid = self.new_id()
        d = os.path.join(self.work, vid)
        os.makedirs(d, exist_ok=True)
        src = os.path.join(d, "input" + ext)
        with open(src, "wb") as f:
            f.write(data)
        scene = _load(src)
        n_in, deg_in = int(scene.n), int(scene.sh_degree)
        # 指定蒸馏目标阶数时覆盖自动配方（默认自动：预算够就不降阶）
        rec = None
        if target_degree is not None:
            rec = plan(scene, platform, float(budget_mb), float(psnr_floor), profile(scene))
            for st in rec["stages"]:
                if st.get("stage") == "distill":
                    st["target_degree"] = min(int(target_degree), deg_in)
        out = os.path.join(d, "out")
        summ = run_pipeline(scene, out, platform=platform, budget_mb=float(budget_mb),
                            psnr_floor=float(psnr_floor), recipe=rec,
                            max_rounds=int(rounds), verbose=False, do_render=False)
        page = None
        try:
            page = write_cesium_viewer(out)
        except Exception:
            page = None
        return {"n_in": n_in, "degree_in": deg_in, "name": name,
                "passed": summ.get("passed"), "rounds": summ.get("rounds"),
                "metrics": summ.get("metrics", {}), "stages": summ.get("infos", []),
                "history": summ.get("history", []),
                "view": f"/w/{vid}/out/{os.path.basename(page)}" if page else None}

    def proxy(self, b64: str, rel: str):
        import hashlib
        pad = "=" * (-len(b64) % 4)
        url = base64.urlsafe_b64decode(b64 + pad).decode()
        if rel != "tileset.json":  # 内容 uri 相对 tileset.json 所在目录
            base = url.rsplit("/", 1)[0] + "/"
        else:
            base = url
        remote = urllib.parse.urljoin(base, rel)
        # 磁盘缓存：同一远端文件只取一次（代理链路波动时不重复下载）
        key = hashlib.sha1(remote.encode()).hexdigest()
        cp = os.path.join(self.work, "_cache", key)
        if os.path.isfile(cp):
            return open(cp, "rb").read()
        data = _fetch(remote)
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        with open(cp, "wb") as f:
            f.write(data)
        self._n_write += 1
        if self._n_write % 20 == 0:      # 每 20 次写盘检查一次容量，避免频繁遍历目录
            self._evict_cache()
        return data

    def _evict_cache(self) -> None:
        """缓存超过 _CACHE_MAX 时按最旧访问时间淘汰（全国级数据集可达 GB 量级）。"""
        d = os.path.join(self.work, "_cache")
        try:
            items = []
            for f in os.listdir(d):
                p = os.path.join(d, f)
                try:
                    items.append((os.path.getmtime(p), os.path.getsize(p), p))
                except OSError:
                    continue
        except OSError:
            return
        total = sum(i[1] for i in items)
        if total <= _CACHE_MAX:
            return
        items.sort()
        for _, sz, p in items:
            if total <= _CACHE_MAX * 0.8:
                break
            try:
                os.remove(p)
                total -= sz
            except Exception:      # 删除受限的环境（回收站不可用等）跳过即可
                pass

    # ---- 上传 3D Tiles zip ----
    def tiles_zip(self, name: str, zb: bytes) -> dict:
        vid = self.new_id()
        root = _unzip_find_root(zb, self.work, vid, "tileset.json")
        if not root:
            shutil.rmtree(os.path.join(self.work, vid), ignore_errors=True)
            raise ValueError("zip 中未找到 tileset.json")
        st = scan_tileset(os.path.join(root, "tileset.json"))
        page = write_cesium_viewer(root)  # ≤8MB 自动内嵌单文件
        return {"stats": {k: st.get(k) for k in
                          ("tiles_in", "tiles_out", "pts_in", "bytes_in", "formats") if k in st},
                "view": "/w/" + os.path.relpath(page, self.work).replace("\\", "/"),
                "dir": os.path.relpath(root, self.work).replace("\\", "/")}

    # ---- 矢量（在线或 zip 均落到本地文件再压缩） ----
    def vector_run(self, src_path: str, name: str) -> dict:
        vid = self.new_id()
        out = os.path.join(self.work, vid)
        r = compress_vector(src_path, out, tol=0.35, levels=4, verbose=False)
        page = write_vector_viewer(out, title=f"GeoSqueeze · {name}")
        ev = r["single"]
        return {"stats": {"features": r["features"], "vertices": r["vertices"],
                          "bytes_in": ev["bytes_in"], "bytes_out": ev["bytes_out"],
                          "ratio": ev["ratio"], "max_dev": round(ev["max_dev"], 4)},
                "view": "/w/" + vid + "/" + os.path.basename(page)}

    def vector_url(self, url: str) -> dict:
        data = _fetch(url)
        p = os.path.join(self.work, self.new_id() + ".geojson")
        with open(p, "wb") as f:
            f.write(data)
        return self.vector_run(p, os.path.basename(url))

    def vector_zip(self, name: str, zb: bytes) -> dict:
        vid = self.new_id()
        root = _unzip_find_root(zb, self.work, vid, None) or ""
        gjs = None
        for dirpath, _dirs, files in os.walk(os.path.join(self.work, vid)):
            for fn in files:
                if fn.lower().endswith((".geojson", ".json")):
                    try:
                        j = json.load(open(os.path.join(dirpath, fn), encoding="utf-8"))
                    except ValueError:
                        continue
                    if j.get("type") == "FeatureCollection":
                        gjs = os.path.join(dirpath, fn)
                        break
            if gjs:
                break
        if not gjs:
            shutil.rmtree(os.path.join(self.work, vid), ignore_errors=True)
            raise ValueError("zip 中未找到 GeoJSON FeatureCollection")
        return self.vector_run(gjs, name)


_PAGE = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>GeoSqueeze 测试工作台</title><style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#0b0f19;color:#e8ecf5;font:14px/1.6 system-ui;padding:28px}
h1{font-size:20px;margin:0 0 4px} .sub{color:#8b96ab;margin-bottom:22px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:1100px}
.card{background:#121828;border:1px solid #232c42;border-radius:12px;padding:18px}
.card h2{font-size:15px;margin:0 0 10px;color:#7dd3fc}
select,input[type=text]{width:100%;padding:8px 10px;border-radius:8px;border:1px solid #2c3852;
 background:#0d1322;color:#e8ecf5;font:13px system-ui}
input[type=file]{width:100%;color:#8b96ab;margin-top:6px}
button{margin-top:12px;padding:8px 18px;border-radius:8px;border:0;cursor:pointer;
 background:#2563eb;color:#fff;font:13px system-ui}
button:disabled{background:#334155;cursor:wait}
.drop{border:1.5px dashed #31405f;border-radius:10px;padding:14px;text-align:center;color:#8b96ab;margin-top:8px}
.drop.hot{border-color:#2563eb;color:#bfd6ff}
#res{max-width:1100px;margin-top:20px}
.item{background:#121828;border:1px solid #232c42;border-radius:12px;padding:14px 16px;margin-bottom:14px}
.item .st{color:#a8b3c7;font:12.5px/1.8 ui-monospace,monospace;white-space:pre-wrap;word-break:break-all}
.item b{color:#7dd3fc}
iframe{width:100%;height:560px;border:0;border-radius:10px;background:#0b0f19;margin-top:10px}
.err{color:#f8a} .ok{color:#7ee787}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:8px 0;font-size:12px;opacity:.9}
.row label{display:flex;gap:5px;align-items:center}
.row input[type=range]{width:110px}
table.cmp{width:100%;border-collapse:collapse;margin-top:8px;font-size:12.5px}
table.cmp th,table.cmp td{border-bottom:1px solid #2a3242;padding:5px 7px;text-align:left;vertical-align:top}
table.cmp th{color:#8fa3c0;font-weight:600}
.pair{display:flex;gap:10px;flex-wrap:wrap;margin-top:10px}
.pair>div{flex:1 1 420px;min-width:320px}
.pair h4{margin:0 0 4px;font-size:12.5px;color:#8fa3c0}
a{color:#7dd3fc}
</style></head><body>
<h1>GeoSqueeze 测试工作台</h1>
<div class="sub">在线 3D Tiles / 矢量地址加载 · zip 上传测试 —— ① 支持「原始渲染」与「走 GeoSqueeze 管线压缩后对比」两种模式；后端反代解决跨域，渲染层 CesiumJS / 自研 WebGL2</div>
<div class="grid">
<div class="card"><h2>① 在线 3D Tiles</h2>
  <select id="tPreset"></select>
  <input type="text" id="tUrl" placeholder="https://…/tileset.json" spellcheck="false">
  <button id="tGo" onclick="runTilesUrl()">加载并渲染（原始数据，不做任何处理）</button>
  <div class="row">
    <label>剪枝比例 <input type="range" id="cPrune" min="0" max="0.9" step="0.05" value="0.5"
      oninput="document.getElementById('pruneVal').textContent=this.value"></label>
    <b id="pruneVal">0.5</b>
    <label>位置位深 <select id="cBits"><option>8</option><option>12</option><option selected>16</option></select></label>
    <label><input type="checkbox" id="cQuant" checked> 量化</label>
    <label><input type="checkbox" id="cMesh" checked> 保网格</label>
  </div>
  <button id="tCmp" onclick="runTilesCompress()">走 GeoSqueeze 管线压缩 → 前后对比</button>
</div>
<div class="card"><h2>② 在线矢量（GeoJSON）</h2>
  <select id="vPreset"></select>
  <input type="text" id="vUrl" placeholder="https://…/xxx.geojson" spellcheck="false">
  <button id="vGo" onclick="runVectorUrl()">下载 → 压缩 → 渲染</button>
</div>
<div class="card"><h2>③ 上传 3D Tiles 压缩包（zip）</h2>
  <div class="drop" id="tDrop">拖拽 zip 到这里，或<input type="file" id="tZip" accept=".zip"></div>
  <button id="tUp" onclick="runTilesZip()" disabled>解压 → 校验 → 渲染</button>
</div>
<div class="card"><h2>④ 上传矢量数据压缩包（zip）</h2>
  <div class="drop" id="vDrop">拖拽 zip 到这里，或<input type="file" id="vZip" accept=".zip"></div>
  <button id="vUp" onclick="runVectorZip()" disabled>解压 → 压缩 → 渲染</button>
</div>
<div class="card"><h2>⑤ 3DGS / 点云主链路（剪枝 + 蒸馏 + 量化）</h2>
  <div class="drop" id="pDrop">拖拽 .ply / .las / .gsq 到这里，或<input type="file" id="pFile" accept=".ply,.las,.laz,.gsq"></div>
  <div class="row">
    <label>体积预算 <select id="pBudget"><option value="20">20MB</option><option value="120" selected>120MB</option><option value="500">500MB</option></select></label>
    <label>PSNR 底线 <input type="number" id="pFloor" value="28" step="1" min="10" max="60"></label>
    <label>蒸馏目标阶数 <select id="pDeg"><option value="auto" selected>自动</option><option value="2">2 阶</option><option value="1">1 阶</option><option value="0">0 阶</option></select></label>
  </div>
  <button id="pGo" onclick="runPrimitives()" disabled>走主链路 → 指标 + 渲染</button>
</div>
</div>
<div id="res"></div>
<script>
const PRESETS = __PRESETS__;
function fill(sel, list, input){ const s=document.getElementById(sel);
  list.forEach((p,i)=>{ const o=document.createElement('option'); o.value=p[1]; o.textContent=p[0]; s.appendChild(o); });
  s.onchange=()=>{ document.getElementById(input).value=s.value; };
  document.getElementById(input).value=list[0][1]; }
fill('tPreset', PRESETS.tiles, 'tUrl'); fill('vPreset', PRESETS.vector, 'vUrl');
function drop2input(dropId, inputId, btnId){
  const d=document.getElementById(dropId), inp=document.getElementById(inputId);
  d.addEventListener('dragover', e=>{ e.preventDefault(); d.classList.add('hot'); });
  d.addEventListener('dragleave', ()=> d.classList.remove('hot'));
  d.addEventListener('drop', e=>{ e.preventDefault(); d.classList.remove('hot');
    if(e.dataTransfer.files[0]){ inp.files=e.dataTransfer.files; document.getElementById(btnId).disabled=false; } });
  inp.onchange=()=> document.getElementById(btnId).disabled = !inp.files.length; }
drop2input('tDrop','tZip','tUp'); drop2input('vDrop','vZip','vUp');
drop2input('pDrop','pFile','pGo');
function fileB64(f){ return new Promise((res,rej)=>{ const r=new FileReader();
  r.onload=()=>res(r.result.split(',')[1]); r.onerror=rej; r.readAsDataURL(f); }); }
async function post(api, body, btn){
  btn.disabled = true; const old = btn.textContent; btn.textContent = '处理中…';
  try { const r = await fetch(api, {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify(body)});
        const j = await r.json();
        if (!r.ok) throw new Error(j.error || r.statusText); return j; }
  finally { btn.disabled = false; btn.textContent = old; } }
function show(title, stats, view, err){
  const box = document.createElement('div'); box.className='item';
  const st = err ? `<span class="err">✗ ${err}</span>` :
    `<span class="ok">✓ 成功</span>  <span class="st">${JSON.stringify(stats, null, 1)
      .replace(/[{}",\\[\\]]/g,'').trim()}</span>`;
  box.innerHTML = `<b>${title}</b> ${st}` +
    (view ? `<br><a href="${view}" target="_blank">在新窗口打开渲染 →</a>
      <iframe src="${view}" loading="lazy"></iframe>` : '');
  document.getElementById('res').prepend(box); }
async function runTilesUrl(){ try { const j = await post('/api/tiles-url', {url: tUrl.value.trim()}, tGo);
  show('① 在线 3D Tiles（原始数据，未做任何处理）', j.stats, j.view); } catch(e){ show('① 在线 3D Tiles', null, null, e.message); } }
function mbStr(v){ return v==null ? '-' : (v/1048576).toFixed(2)+'MB'; }
function showCompress(title, j){
  const m = j.metrics || {};
  const box = document.createElement('div'); box.className='item';
  const rows = (j.stages||[]).map(function(s){
    return '<tr><td>'+s.stage+'</td><td class="'+(s.executed?'ok':'err')+'">'+
      (s.executed?'已执行':'未执行')+'</td><td>'+(s.detail||'')+'</td><td>'+(s.effect||'')+'</td></tr>';
  }).join('');
  let kv = '';
  kv += '<tr><td>瓦片数</td><td>'+(m.tiles_in||0)+' → '+(m.tiles_out||0)+'</td></tr>';
  kv += '<tr><td>体积</td><td>'+mbStr(m.bytes_in)+' → '+mbStr(m.bytes_out)+'</td></tr>';
  kv += '<tr><td>总压缩比</td><td><b>'+(m.ratio||'-')+'x</b></td></tr>';
  if (m.mesh_out) kv += '<tr><td>网格瓦片</td><td>'+m.mesh_out+' 个 · 三角形 '+m.tris_in+' → '+m.tris_out+'</td></tr>';
  else kv += '<tr><td>点</td><td>'+(m.pts_in||0)+' → '+(m.pts_out||0)+'</td></tr>';
  if (m.tex_psnr_mean!=null) kv += '<tr><td>纹理 PSNR</td><td>'+m.tex_psnr_min+' ~ '+m.tex_psnr_mean+' dB</td></tr>';
  box.innerHTML = '<b>'+title+'</b> <span class="ok">✓ 管线执行完成</span>'+
    '<div class="st">下载 '+(j.download_mb||'?')+'MB'+
    (j.complete?'':' · 已达下载上限，仅压缩了部分瓦片')+'</div>'+
    '<table class="cmp"><tr><th>阶段</th><th>状态</th><th>参数</th><th>效果</th></tr>'+rows+'</table>'+
    '<table class="cmp">'+kv+'</table>'+
    '<div class="pair"><div><h4>原始（未处理，反代直读）</h4>'+
    (j.orig_view?'<iframe src="'+j.orig_view+'" loading="lazy"></iframe>':'')+'</div>'+
    '<div><h4>GeoSqueeze 压缩后</h4>'+
    (j.view?'<iframe src="'+j.view+'" loading="lazy"></iframe>':'')+'</div></div>'+
    (j.view?'<br><a href="'+j.view+'" target="_blank">单独打开压缩后视图 →</a>':'');
  document.getElementById('res').prepend(box);
}
async function runTilesCompress(){
  const body = {url: tUrl.value.trim(),
    prune: +document.getElementById('cPrune').value,
    pos_bits: +document.getElementById('cBits').value,
    quantize: document.getElementById('cQuant').checked,
    keep_mesh: document.getElementById('cMesh').checked};
  try { const j = await post('/api/tiles-compress', body, tCmp); showCompress('① 在线 3D Tiles · GeoSqueeze 管线', j); }
  catch(e){ show('① 管线压缩', null, null, e.message); } }
async function runVectorUrl(){ try { const j = await post('/api/vector-url', {url: vUrl.value.trim()}, vGo);
  show('② 在线矢量（下载 → 压缩 → 评测）', j.stats, j.view); } catch(e){ show('② 在线矢量', null, null, e.message); } }
async function runTilesZip(){ try { const f = tZip.files[0];
  const j = await post('/api/tiles-zip', {name: f.name, b64: await fileB64(f)}, tUp);
  show('③ 上传 3D Tiles zip：' + f.name, j.stats, j.view); } catch(e){ show('③ 上传 3D Tiles zip', null, null, e.message); } }
async function runVectorZip(){ try { const f = vZip.files[0];
  const j = await post('/api/vector-zip', {name: f.name, b64: await fileB64(f)}, vUp);
  show('④ 上传矢量 zip：' + f.name, j.stats, j.view); } catch(e){ show('④ 上传矢量 zip', null, null, e.message); } }
function fmtNum(v){ return (v==null) ? '-' : Number(v).toLocaleString(); }
function showPrimitives(j){
  const box = document.createElement('div'); box.className='item';
  const m = j.metrics || {};
  const rows = (j.stages||[]).map(function(s){
    const st = s.stage || s.name || '?';
    let eff = '';
    if (st === 'prune') eff = '保留 ' + fmtNum(s.kept) + ' 点 · 剪掉比例 ' + (s.ratio!=null?s.ratio:'?');
    else if (st === 'distill') eff = s.skipped
      ? ('跳过（输入已是 ' + s.degree + ' 阶，未指定更低目标）')
      : ('球谐 ' + s.degree_from + ' 阶 → ' + s.degree_to + ' 阶 · 保留系数 ' + s.coef_kept +
         ' · 能量补偿 ' + (s.energy_compensated ? '是' : '否'));
    else if (st === 'quantize') eff = '位置 ' + s.pos_bits + ' bit · SH 模式 ' + s.sh_mode +
      (s.codebook ? (' · 码本 ' + s.codebook) : '') + ' · 位置MSE ' + Number(s.pos_mse).toExponential(2);
    else if (st === 'recover') eff = 'PSNR ' + s.psnr_before + ' → ' + s.psnr_after + ' dB · ' + s.iters + ' 次迭代';
    else eff = Object.keys(s).filter(k=>k!=='stage').map(k=>k+'='+s[k]).join(' · ');
    return '<tr><td><b>'+st+'</b></td><td>'+eff+'</td></tr>';
  }).join('');
  const keyRows = Object.keys(m).map(function(k){
    return '<tr><td>'+k+'</td><td>'+m[k]+'</td></tr>'; }).join('');
  box.innerHTML = '<b>⑤ 主链路：'+j.name+'</b> ' +
    (j.passed ? '<span class="ok">✓ 达标</span>' : '<span class="err">✗ 未达标</span>') +
    '<div class="st">输入 ' + fmtNum(j.n_in) + ' 基元 · 球谐 ' + j.degree_in + ' 阶 · ' +
    (j.rounds||1) + ' 轮迭代</div>' +
    '<table class="cmp"><tr><th>阶段</th><th>执行结果</th></tr>' + rows + '</table>' +
    '<table class="cmp">' + keyRows + '</table>' +
    (j.view ? ('<br><a href="'+j.view+'" target="_blank">打开压缩后 3D Tiles 渲染 →</a>' +
      '<iframe src="'+j.view+'" loading="lazy"></iframe>') : '<div class="st">（无渲染产物）</div>');
  document.getElementById('res').prepend(box);
}
async function runPrimitives(){ const f = pFile.files[0]; if(!f) return;
  try { const j = await post('/api/primitives', {name: f.name, b64: await fileB64(f),
      budget_mb: +document.getElementById('pBudget').value,
      psnr_floor: +document.getElementById('pFloor').value,
      target_degree: document.getElementById('pDeg').value}, pGo);
    showPrimitives(j);
  } catch(e){ show('⑤ 主链路：' + f.name, null, null, e.message); } }
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    bench: _Bench = None

    def log_message(self, fmt, *args):  # 静默访问日志
        pass

    def _send(self, code: int, body: bytes, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False, default=float).encode())

    def do_GET(self):
        b = self.bench
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            html = _PAGE.replace("__PRESETS__", json.dumps(_PRESETS, ensure_ascii=False))
            return self._send(200, html.encode(), "text/html; charset=utf-8")
        if path == "/cesium.html":  # 无内嵌 viewer：供 ?ts= 在线/反代地址使用
            return self._send(200, (_CESIUM_HTML.replace("__CDN__", _CDN_REF)
                                     .replace("__VFS__", "null")).encode(),
                              "text/html; charset=utf-8")
        if path.startswith("/p/"):  # 在线 3D Tiles 反代
            try:
                _, _, b64, rel = path.split("/", 3)
                return self._send(200, b.proxy(b64, rel), "application/octet-stream")
            except Exception as e:
                return self._send(502, str(e).encode(), "text/plain; charset=utf-8")
        if path.startswith("/w/"):  # 工作产物静态服务
            fp = os.path.normpath(os.path.join(b.work, path[3:].split("?")[0]))
            if not fp.startswith(b.work) or not os.path.isfile(fp):
                return self._send(404, b"not found", "text/plain")
            ext = os.path.splitext(fp)[1].lower()
            ctype = {".html": "text/html; charset=utf-8", ".json": "application/json",
                     ".b3dm": "application/octet-stream", ".pnts": "application/octet-stream",
                     ".glb": "model/gltf-binary", ".gvec": "application/octet-stream"}.get(ext)
            if ctype is None:
                ctype = "image/webp" if ext == ".webp" else \
                    "image/jpeg" if ext in (".jpg", ".jpeg") else \
                    "image/png" if ext == ".png" else "application/octet-stream"
            return self._send(200, open(fp, "rb").read(), ctype)
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        b = self.bench
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > _MAX_UPLOAD:
                raise ValueError(f"上传超过上限 {_MAX_UPLOAD//1024//1024}MB")
            buf = bytearray()
            while len(buf) < n:                      # 大请求体分块读，避免短读导致 JSON 截断
                chunk = self.rfile.read(min(1 << 20, n - len(buf)))
                if not chunk:
                    break
                buf += chunk
            if len(buf) != n:
                raise ValueError(f"请求体不完整：收到 {len(buf)}/{n} 字节")
            body = json.loads(buf) if n else {}
            if self.path == "/api/tiles-url":
                return self._json(b.tiles_url(body["url"].strip()))
            if self.path == "/api/tiles-compress":
                return self._json(b.tiles_compress(
                    body["url"].strip(),
                    prune=float(body.get("prune", 0.5)),
                    pos_bits=int(body.get("pos_bits", 16)),
                    quantize=bool(body.get("quantize", True)),
                    keep_mesh=bool(body.get("keep_mesh", True)),
                    tex_max_dim=int(body.get("tex_max_dim", 512)),
                    tex_quality=int(body.get("tex_quality", 78)),
                    max_mb=int(body.get("max_mb", 60))))
            if self.path == "/api/vector-url":
                return self._json(b.vector_url(body["url"].strip()))
            if self.path == "/api/tiles-zip":
                return self._json(b.tiles_zip(body.get("name", "tiles.zip"),
                                              base64.b64decode(body["b64"])))
            if self.path == "/api/vector-zip":
                return self._json(b.vector_zip(body.get("name", "vector.zip"),
                                               base64.b64decode(body["b64"])))
            if self.path == "/api/primitives":
                return self._json(b.primitives_compress(
                    body.get("name", "input.ply"), base64.b64decode(body["b64"]),
                    platform=body.get("platform", "web"),
                    budget_mb=float(body.get("budget_mb", 120)),
                    psnr_floor=float(body.get("psnr_floor", 28)),
                    rounds=int(body.get("rounds", 3)),
                    target_degree=(None if body.get("target_degree") in (None, "", "auto")
                                   else int(body["target_degree"]))))
            self._json({"error": f"未知接口 {self.path}"}, 404)
        except BaseException as e:      # 含环境钩子（如回收站/安全删除）抛出的非 Exception
            import traceback
            traceback.print_exc()
            try:
                self._json({"error": f"{type(e).__name__}: {e}"}, 400)
            except Exception:
                pass


_CDN_REF = "https://unpkg.com/cesium@1.120.0/Build/Cesium"  # 与 cesium_viewer 保持一致


class _Server(ThreadingHTTPServer):
    # Windows 上 SO_REUSEADDR 允许两个进程绑定同一端口（连接被随机分发、症状是偶发卡死），
    # 关掉复用改为独占绑定：端口被占用时明确报错，而不是静默双绑
    allow_reuse_address = False
    daemon_threads = True
    request_queue_size = 64


def serve_testbench(port: int = 8619, work: str = "out/testbench") -> None:
    """启动测试工作台 HTTP 服务（阻塞）。"""
    _Handler.bench = _Bench(work)
    try:
        srv = _Server(("127.0.0.1", port), _Handler)
    except OSError as e:
        raise RuntimeError(f"端口 {port} 已被占用（可能上一次的 testbench 还在跑）: {e}") from e
    print(f"GeoSqueeze 测试工作台 → http://127.0.0.1:{port}/  （Ctrl+C 退出）")
    print(f"  工作目录: {os.path.abspath(work)}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")
