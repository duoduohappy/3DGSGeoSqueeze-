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
        ["3DBAG 荷兰全国建筑 · LoD2.2（真实生产数据，CC BY 4.0）",
         "https://data.3dbag.nl/v20250903/cesium3dtiles/lod22/tileset.json"],
        ["3DBAG 荷兰全国建筑 · LoD1.2（轻量，加载快）",
         "https://data.3dbag.nl/v20250903/cesium3dtiles/lod12/tileset.json"],
        ["GeoSqueeze 带纹理城市（本仓库 raw）",
         "https://raw.githubusercontent.com/duoduohappy/3DGSGeoSqueeze-/main/out/mesh_city/tileset.json"],
        ["Cesium 官方样例 · TreeBillboards（1.0）",
         "https://raw.githubusercontent.com/CesiumGS/3d-tiles-samples/refs/heads/main/1.0/TilesetWithTreeBillboards/tileset.json"],
        ["Cesium 官方样例 · DiscreteLOD（1.0）",
         "https://raw.githubusercontent.com/CesiumGS/3d-tiles-samples/refs/heads/main/1.0/TilesetWithDiscreteLOD/tileset.json"],
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
            except OSError:
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
a{color:#7dd3fc}
</style></head><body>
<h1>GeoSqueeze 测试工作台</h1>
<div class="sub">在线 3D Tiles / 矢量地址加载 · zip 上传测试 —— 后端反代解决跨域，渲染层 CesiumJS / 自研 WebGL2</div>
<div class="grid">
<div class="card"><h2>① 在线 3D Tiles</h2>
  <select id="tPreset"></select>
  <input type="text" id="tUrl" placeholder="https://…/tileset.json" spellcheck="false">
  <button id="tGo" onclick="runTilesUrl()">加载并渲染</button>
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
  show('① 在线 3D Tiles', j.stats, j.view); } catch(e){ show('① 在线 3D Tiles', null, null, e.message); } }
async function runVectorUrl(){ try { const j = await post('/api/vector-url', {url: vUrl.value.trim()}, vGo);
  show('② 在线矢量（下载 → 压缩 → 评测）', j.stats, j.view); } catch(e){ show('② 在线矢量', null, null, e.message); } }
async function runTilesZip(){ try { const f = tZip.files[0];
  const j = await post('/api/tiles-zip', {name: f.name, b64: await fileB64(f)}, tUp);
  show('③ 上传 3D Tiles zip：' + f.name, j.stats, j.view); } catch(e){ show('③ 上传 3D Tiles zip', null, null, e.message); } }
async function runVectorZip(){ try { const f = vZip.files[0];
  const j = await post('/api/vector-zip', {name: f.name, b64: await fileB64(f)}, vUp);
  show('④ 上传矢量 zip：' + f.name, j.stats, j.view); } catch(e){ show('④ 上传矢量 zip', null, null, e.message); } }
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
            body = json.loads(self.rfile.read(n)) if n else {}
            if self.path == "/api/tiles-url":
                return self._json(b.tiles_url(body["url"].strip()))
            if self.path == "/api/vector-url":
                return self._json(b.vector_url(body["url"].strip()))
            if self.path == "/api/tiles-zip":
                return self._json(b.tiles_zip(body.get("name", "tiles.zip"),
                                              base64.b64decode(body["b64"])))
            if self.path == "/api/vector-zip":
                return self._json(b.vector_zip(body.get("name", "vector.zip"),
                                               base64.b64decode(body["b64"])))
            self._json({"error": f"未知接口 {self.path}"}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 400)


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
