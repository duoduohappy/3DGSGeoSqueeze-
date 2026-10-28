"""CesiumJS 查看器：用行业标准 3D Tiles 引擎加载 GeoSqueeze 输出，验证互操作性。

产物 tileset.json + b3dm/pnts 本身就是标准 3D Tiles，Cesium 原生加载、零转换：
- tileset.asset.gltfUpAxis="Z"（Cesium 1.1x 只读该字段），Cesium 不再做 Y→Z 旋转
- pnts POSITION_QUANTIZED 为 Cesium 内置支持
- 本地坐标系渲染（隐藏地球），也可用 ?geo=1 放到地球指定位置

数据加载策略：
- 默认把 tileset.json + 全部瓦片内容 base64 内嵌进 cesium.html（≤ inline_max_mb 时），
  单文件双击即可打开，不受 file:// CORS 限制
- 超过阈值则写外链版，需通过 `python -m gsq serve` 经 HTTP 打开（file:// 会被 CORS 拦截，
  页面会显示醒目提示）

注意：CesiumJS 本体从 CDN 加载（约 3MB + Workers），首次打开需联网。
"""
import base64
import json
import os

_CDN = "https://unpkg.com/cesium@1.120.0/Build/Cesium"
INLINE_MAX_MB = 8  # 数据集总大小低于该值时内嵌进单文件 HTML（超过则回退外链模式）

_HTML = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>GeoSqueeze · CesiumJS 查看器</title>
<link href="__CDN__/Widgets/widgets.css" rel="stylesheet">
<style>
html,body{margin:0;height:100%;background:#0b0f19;overflow:hidden}
#cesiumContainer{width:100%;height:100%}
#hud{position:fixed;top:10px;left:10px;color:#e8ecf5;background:rgba(10,14,25,.78);
 padding:8px 12px;border-radius:8px;font:13px/1.6 system-ui;z-index:9;white-space:pre;
 border:1px solid rgba(255,255,255,.12)}
#hud b{color:#7dd3fc}
#hud .err{color:#fbb}
#hud .warn{color:#fde68a}
</style></head><body>
<div id="cesiumContainer"></div><div id="hud">加载 CesiumJS（CDN，首次需联网）…</div>
<script>window.CESIUM_BASE_URL="__CDN__";</script>
<script src="__CDN__/Cesium.js"></script>
<script>
// 内嵌虚拟文件系统：vfile://<uri> -> base64（write_cesium_viewer 生成；null = 外链模式）
const VFS = __VFS__;
const hud = document.getElementById('hud');
const errs = [];
window.onerror = (m, src, l, c) => { errs.push(m + ' @' + l + ':' + c); };
window.addEventListener('unhandledrejection', e => errs.push(String(e.reason && e.reason.message || e.reason)));
for (const lv of ['error', 'warn']) { const o = console[lv].bind(console);
  console[lv] = (...a) => { errs.push(lv + ': ' + a.map(x => String(x && x.message || x)).join(' ').slice(0, 300)); o(...a); }; }
if (!VFS && location.protocol === 'file:') {
  hud.innerHTML = '<span class="warn">直接双击打开无法加载数据（浏览器 file:// 安全策略拦截网络请求）。\\n' +
    '请在项目目录运行：\\n    python -m gsq serve <输出目录>\\n然后浏览器访问打印出的 http://localhost 地址。</span>';
  throw new Error('file:// 模式需经 HTTP 服务打开');
}
if (VFS) {
  // 内嵌模式：劫持 Cesium 的 XHR 实现，vfile:// 请求直接从内存返回（无需任何服务器）
  const orig = Cesium.Resource._Implementations.loadWithXhr;
  Cesium.Resource._Implementations.loadWithXhr = function(url, responseType, method, data, headers, deferred, overrideMimeType) {
    if (url.indexOf('vfile:') === 0) {
      // Cesium 会做 URI 规范化（加尾斜杠/拆 host），这里统一还原成 vfile:///<rel> 再查表
      const rel = url.slice(8).replace(/^\\/+|\\/+$/g, '');
      const b64 = VFS['vfile:///' + rel];
      if (b64 === undefined) { deferred.reject(new Error('vfile 缺少: ' + url)); return; }
      const bin = atob(b64), buf = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
      deferred.resolve(responseType === 'text' ? new TextDecoder().decode(buf) : buf.buffer);
      return;
    }
    orig.apply(this, arguments);
  };
}
(async function(){
 try {
  const viewer = new Cesium.Viewer('cesiumContainer', {
    baseLayer: false, baseLayerPicker: false, geocoder: false, homeButton: false,
    sceneModePicker: false, navigationHelpButton: false, animation: false,
    timeline: false, fullscreenButton: false, infoBox: false,
    selectionIndicator: false, shadows: false,
  });
  const s = viewer.scene;
  s.globe.show = false; s.skyBox.show = false; s.skyAtmosphere.show = false;
  s.fog.enabled = false; s.sun.show = false; s.moon.show = false;
  s.logarithmicDepthBuffer = new URLSearchParams(location.search).has('nolog');
  s.backgroundColor = Cesium.Color.fromCssColorString('#0b0f19');
  viewer.cesiumWidget.creditContainer.style.display = 'none';
  viewer.scene.debugShowFramesPerSecond = false;

  hud.textContent = '加载 tileset.json…（在线大文件/网络慢时请耐心等待，出错会显示在此）';
  const qsp = new URLSearchParams(location.search);
  const qsts = qsp.get('ts') || 'tileset.json';
  const tsUrl = VFS ? ('vfile:///' + qsts) : qsts;
  // 上轴：?up=z|y|x 强制指定；缺省 auto——读 tileset.asset.gltfUpAxis（"Z" → Z-up 直存，
  // 无声明按 3D Tiles 默认交由 Cesium 做 Y-up→Z-up 旋转），兼容外部标准数据
  async function detectUpAxis() {
    const up = qsp.get('up');
    if (up === 'z') return Cesium.Axis.Z;
    if (up === 'y') return Cesium.Axis.Y;
    if (up === 'x') return Cesium.Axis.X;
    try {
      let asset;
      if (VFS) {
        const b64 = VFS['vfile:///' + qsts];
        if (!b64) return null;
        asset = JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(b64), c => c.charCodeAt(0)))).asset;
      } else {
        asset = (await (await fetch(qsts)).json()).asset;
      }
      return (asset && asset.gltfUpAxis === 'Z') ? Cesium.Axis.Z : null;
    } catch (e) { return null; }
  }
  const opts = {maximumScreenSpaceError: 8, skipLevelOfDetail: false, backFaceCulling: false};
  const ua = await detectUpAxis();
  if (ua) { opts.modelUpAxis = ua; opts.modelForwardAxis = Cesium.Axis.X; }
  const ts = await Cesium.Cesium3DTileset.fromUrl(tsUrl, opts);
  s.primitives.add(ts);
  // geo=1：把本地坐标场景放到椭球面（ENU 框架），避开"场景在地球内部"的深度退化
  if (qsp.has('geo')) {
    ts.modelMatrix = Cesium.Transforms.eastNorthUpToFixedFrame(
        Cesium.Cartesian3.fromDegrees(116.397, 39.908, 10));
  }
  let bs = ts.boundingSphere;
  if (ts.modelMatrix && !Cesium.Matrix4.equals(ts.modelMatrix, Cesium.Matrix4.IDENTITY)) {
    bs = Cesium.BoundingSphere.transform(ts.boundingSphere, ts.modelMatrix, new Cesium.BoundingSphere());
  }
  const c = bs.center, r = bs.radius;
  window._gqR = r;
  s.camera.lookAt(c, new Cesium.HeadingPitchRange(0,
      qsp.get('top') ? -Math.PI/2 : Cesium.Math.toRadians(+(qsp.get('pitch') || -90)),
      qsp.get('d') ? +qsp.get('d') : r * 2.6));
  s.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);   // 解除锁定，保留自由轨道/缩放
  window._gqTileset = ts;

  const t0 = performance.now();
  let dbg = VFS ? ' · 内嵌单文件' : '';
  s.postRender.addEventListener(function tick(){
    const st = ts.statistics || {};
    hud.innerHTML = '<b>GeoSqueeze × CesiumJS</b>（标准 3D Tiles 直读）\\n' +
      `瓦片就绪 ${ts.tilesLoaded ? '是' : '否'} · 总数 ${st.numberOfTilesTotal ?? '-'} · ` +
      `加载中 ${isNaN(ts.tilesRequested - ts.tilesLoaded) ? 0 : ts.tilesRequested - ts.tilesLoaded}\\n` +
      `渲染点/三角形累计 ${(st.numberOfPointsLoaded ?? 0).toLocaleString()} 点 · ` +
      `耗时 ${((performance.now() - t0) / 1000).toFixed(1)}s\\n` +
      'r=' + (window._gqR ? window._gqR.toFixed(1) : '?') + dbg +
      (errs.length ? `\\n<span class="err">错误: ${errs.join(' | ')}</span>` : '');
    requestAnimationFrame(tick);
  });
  ts.tileFailed.addEventListener(t => errs.push(t.uri + ' ' + t.error));
 } catch (e) { errs.push(String(e && e.message || e)); hud.innerHTML = '<span class="err">错误: ' + errs.join(' | ') + '</span>'; }
})();
</script></body></html>"""


def _scan_uris(tileset_dir: str) -> list[str] | None:
    """遍历 tileset.json（含外部 tileset）收集全部内容 uri。

    只解析 JSON（小文件），不读瓦片二进制；任一 JSON 缺失返回 None（回退外链模式）。
    """
    uris: list[str] = []
    queue = ["tileset.json"]
    while queue:
        rel = queue.pop(0).replace("\\", "/")
        if rel in uris:
            continue
        uris.append(rel)
        p = os.path.join(tileset_dir, rel)
        if not os.path.isfile(p):
            return None
        if not rel.lower().endswith(".json"):
            continue
        try:
            ts = json.loads(open(p, "rb").read())
        except ValueError:
            continue
        stack = [ts.get("root") or {}]
        while stack:
            t = stack.pop()
            if not isinstance(t, dict):
                continue
            stack.extend(c for c in (t.get("children") or []) if isinstance(c, dict))
            ct = t.get("content")
            for c in (ct if isinstance(ct, list) else [ct] if ct else []):
                uri = (c.get("uri") or c.get("url") or "").split("?")[0]
                if uri:
                    queue.append(uri)
    return uris


def write_cesium_viewer(tileset_dir: str, inline: bool = True,
                        inline_max_mb: int = INLINE_MAX_MB) -> str:
    """在 tileset 目录写入 cesium.html。

    inline=True 且数据集 ≤ inline_max_mb 时生成自包含单文件（双击即可打开，
    流式写出，峰值内存只取决于单个最大瓦片）；否则生成外链版
    （需 `python -m gsq serve` 经 HTTP 打开）。
    """
    head, tail = _HTML.split("__VFS__", 1)
    if inline:
        uris = _scan_uris(tileset_dir)
        total = sum(os.path.getsize(os.path.join(tileset_dir, u)) for u in uris) if uris else 0
    else:
        uris, total = None, 0
    p = os.path.join(tileset_dir, "cesium.html")
    with open(p, "w", encoding="utf-8") as f:
        f.write(head.replace("__CDN__", _CDN))
        if uris is not None and total <= inline_max_mb * 1024 * 1024:
            f.write("{")  # 流式内嵌：逐文件 base64 写出，不整体驻留
            first = True
            for u in sorted(set(uris)):
                data = open(os.path.join(tileset_dir, u), "rb").read()
                f.write(('' if first else ',') + f'"vfile:///{u}":"')
                f.write(base64.b64encode(data).decode())
                f.write('"')
                first = False
            f.write("}")
        else:
            f.write("null")
        f.write(tail.replace("__CDN__", _CDN))
    return p
