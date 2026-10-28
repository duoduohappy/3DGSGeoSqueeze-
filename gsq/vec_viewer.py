"""矢量分块查看器：零依赖 Canvas2D，按视距选 LOD、按视口剔除瓦片、内置 GVEC 解码。

验证要点（城市级）：只请求视口内的瓦片；放大时自动切到更细层；HUD 显示已加载块数/顶点数/字节数。
"""
from __future__ import annotations

import json
import os

_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
 html,body{margin:0;height:100%;background:#11151a;color:#e8eef5;font:13px/1.5 system-ui,"Segoe UI",sans-serif;overflow:hidden}
 canvas{display:block;width:100%;height:100%;cursor:grab}
 canvas.drag{cursor:grabbing}
 .hud{position:fixed;left:12px;top:12px;background:rgba(12,16,20,.82);border:1px solid #2a333d;border-radius:8px;padding:10px 12px;min-width:190px;backdrop-filter:blur(6px)}
 .hud b{color:#7fd1a6;font-weight:600}
 .hud div{display:flex;justify-content:space-between;gap:14px}
 .hud .k{color:#93a4b5}
 .ctl{position:fixed;right:12px;top:12px;background:rgba(12,16,20,.82);border:1px solid #2a333d;border-radius:8px;padding:10px 12px;width:210px}
 .ctl label{display:block;color:#93a4b5;margin:6px 0 2px}
 input[type=range]{width:100%}
 .ctl button{width:100%;margin-top:8px;background:#1d6b4c;border:0;color:#eafff5;padding:6px;border-radius:6px;cursor:pointer}
 .err{position:fixed;left:12px;bottom:12px;color:#ff9d9d;font-family:ui-monospace,monospace;white-space:pre-wrap;max-width:60%}
</style></head><body>
<canvas id="c"></canvas>
<div class="hud">
 <div><span class="k">图层</span><b id="h-name">-</b></div>
 <div><span class="k">LOD 层</span><b id="h-lvl">-</b></div>
 <div><span class="k">瓦片</span><b id="h-tiles">0/0</b></div>
 <div><span class="k">顶点</span><b id="h-verts">0</b></div>
 <div><span class="k">已下载</span><b id="h-bytes">0 KB</b></div>
 <div><span class="k">帧率</span><b id="h-fps">-</b></div>
 <div><span class="k">比例尺</span><b id="h-scale">-</b></div>
</div>
<div class="ctl">
 <label>简化开关（显示当前 LOD 容差）</label>
 <div id="c-tol" style="color:#e8eef5">-</div>
 <label>LOD 偏移（0 = 自动）</label>
 <input id="c-bias" type="range" min="-2" max="2" value="0" step="1">
 <button id="c-fit">复位视图</button>
</div>
<div class="err" id="err"></div>
<script>
const CFG = __CONFIG__;
const cv = document.getElementById("c"), ctx = cv.getContext("2d");
const err = document.getElementById("err");
function fail(e){ err.textContent = String(e && e.stack || e); }
window.addEventListener("error", ev => fail(ev.error || ev.message));

let MAN = null, view = null, lvl = -1, cache = new Map(), inflight = 0;
let stats = {tiles:0, verts:0, bytes:0, total:0}, frames = 0, lastT = performance.now(), fps = 0;

// ---------- GVEC 解码（与 gsq/vector.py 完全一致的 varint/zigzag 布局） ----------
function dz(v){ return (v >>> 1) ^ -(v & 1); }
function decodeGvec(buf){
  const dv = new DataView(buf), u8 = new Uint8Array(buf);
  if (String.fromCharCode(u8[0],u8[1],u8[2],u8[3]) !== "GVEC") throw new Error("非 GVEC");
  const hl = dv.getUint32(4, true);
  const head = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + hl)));
  let pos = 8 + hl;
  const rv = () => { let r = 0, s = 0, b; do { b = u8[pos++]; r |= (b & 0x7f) << s; s += 7; } while (b & 0x80); return r; };
  const rz = () => dz(rv());
  const nv = head.n_vertices, vx = new Int32Array(nv), vy = new Int32Array(nv);
  let px = 0, py = 0;
  for (let i = 0; i < nv; i++){ px += rz(); py += rz(); vx[i] = px; vy[i] = py; }
  const nr = rv(), lens = new Int32Array(nr);
  for (let i = 0; i < nr; i++) lens[i] = rv();
  const idx = new Array(nr);
  for (let i = 0; i < nr; i++){
    const L = lens[i], a = new Int32Array(L); let prev = 0;
    for (let k = 0; k < L; k++){ prev += rz(); a[k] = prev; }
    idx[i] = a;
  }
  const props = {};
  for (const col of head.props){
    if (col.type === "dict"){
      const n2 = rv(), vals = [];
      for (let i = 0; i < n2; i++){ const ln = rv(); vals.push(new TextDecoder().decode(u8.subarray(pos, pos + ln))); pos += ln; }
      const n3 = col.codes, codes = new Int32Array(n3);
      for (let i = 0; i < n3; i++) codes[i] = rv();
      props[col.name] = codes.map(c => vals[c]);
    } else if (col.type === "str"){
      const out = [];
      for (let i = 0; i < col.count; i++){ const ln = rv(); out.push(new TextDecoder().decode(u8.subarray(pos, pos + ln))); pos += ln; }
      props[col.name] = out;
    } else if (col.type === "int"){
      const out = new Int32Array(col.count); let prev = 0;
      for (let i = 0; i < col.count; i++){ prev += rz(); out[i] = prev; }
      props[col.name] = out;
    } else {
      // varint 流不保证 4 字节对齐 → 先拷贝到独立 buffer
      props[col.name] = new Float32Array(buf.slice(pos, pos + 4 * col.count)); pos += 4 * col.count;
    }
  }
  const ox = head.origin[0], oy = head.origin[1], sc = head.size / head.extent, off = head.extent / 2;
  const rings = idx.map(a => { const o = new Float64Array(a.length * 2);
    for (let k = 0; k < a.length; k++){ o[2*k] = (vx[a[k]] - off) * sc + ox; o[2*k+1] = (vy[a[k]] - off) * sc + oy; }
    return o; });
  return {head, rings, props};
}

// ---------- 视图变换 ----------
function fitView(){
  const b = MAN.bbox;
  const span = Math.max(b[2] - b[0], b[3] - b[1]) || 1;
  view = {cx: (b[0]+b[2])/2, cy: (b[1]+b[3])/2, span: span * 1.1};
  queueTiles();
}
function toScreen(x, y){
  const W = cv.width, H = cv.height, s = Math.min(W, H) / view.span;
  return [(x - view.cx) * s + W/2, H/2 - (y - view.cy) * s];
}
function viewBounds(){
  const s = Math.min(cv.width, cv.height) / view.span;
  const hw = cv.width / 2 / s, hh = cv.height / 2 / s;
  return [view.cx - hw, view.cy - hh, view.cx + hw, view.cy + hh];
}
function pickLevel(){
  if (!MAN) return 0;
  const b = MAN.bbox, span = Math.max(b[2]-b[0], b[3]-b[1]) || 1;
  const vs = view.span / span;
  let z = Math.ceil(-Math.log2(Math.max(vs, 1e-6)) + 1) + Number(document.getElementById("c-bias").value);
  return Math.max(0, Math.min(MAN.levels.length - 1, z));
}
function visibleTiles(lv){
  const vb = viewBounds(), out = [];
  for (const t of lv.tiles){
    const b = t.bbox;
    if (b[2] < vb[0] || b[0] > vb[2] || b[3] < vb[1] || b[1] > vb[3]) continue;
    out.push(t);
  }
  return out;
}

// ---------- 瓦片加载 ----------
function queueTiles(){
  if (!MAN) return;
  const z = pickLevel();
  if (z !== lvl){
    lvl = z;
    document.getElementById("h-lvl").textContent = "L" + z;
    document.getElementById("c-tol").textContent = (MAN.levels[z].tol ?? 0).toFixed(3) + " m";
  }
  const lv = MAN.levels[z], vis = visibleTiles(lv);
  document.getElementById("h-tiles").textContent = vis.length + "/" + lv.tiles.length;
  for (const t of vis){
    if (cache.has(t.uri) || inflight > 6) continue;
    inflight++;
    fetch(t.uri).then(r => r.ok ? r.arrayBuffer() : Promise.reject(r.status))
      .then(buf => { cache.set(t.uri, decodeGvec(buf)); stats.bytes += buf.byteLength;
        document.getElementById("h-bytes").textContent = (stats.bytes/1024).toFixed(0) + " KB"; })
      .catch(e => fail("加载失败 " + t.uri + " " + e))
      .finally(() => { inflight--; });
  }
}

// ---------- 绘制 ----------
const PALETTE = { g: ["#2f6f57", "#37625c", "#4a6b3f", "#31607a"], l: "#c8b184" };
function draw(){
  const W = cv.width, H = cv.height;
  ctx.fillStyle = "#11151a"; ctx.fillRect(0, 0, W, H);
  if (!MAN) return;
  const lv = MAN.levels[lvl] || MAN.levels[0];
  let nv = 0, nt = 0;
  for (const t of lv.tiles){
    const r = cache.get(t.uri);
    if (!r) continue;
    const vb = viewBounds();
    if (t.bbox[2] < vb[0] || t.bbox[0] > vb[2] || t.bbox[3] < vb[1] || t.bbox[1] > vb[3]) continue;
    nt++; nv += t.vertices;
    const kinds = r.head.types;
    for (let i = 0; i < r.rings.length; i++){
      const ring = r.rings[i], m = ring.length / 2;
      if (m < 2) continue;
      const kc = kinds[i] || "l";
      ctx.beginPath();
      for (let k = 0; k < m; k++){
        const p = toScreen(ring[2*k], ring[2*k+1]);
        k ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]);
      }
      if (kc === "g"){
        ctx.closePath();
        ctx.fillStyle = PALETTE.g[i % PALETTE.g.length];
        ctx.globalAlpha = 0.5; ctx.fill(); ctx.globalAlpha = 1;
        ctx.strokeStyle = "#7fd1a6"; ctx.lineWidth = 0.7; ctx.stroke();
      } else if (kc === "l"){
        ctx.strokeStyle = PALETTE.l; ctx.lineWidth = 1.6; ctx.stroke();
      } else {
        const p = toScreen(ring[0], ring[1]);
        ctx.fillStyle = "#e8eef5"; ctx.fillRect(p[0]-1.5, p[1]-1.5, 3, 3);
      }
    }
  }
  stats.verts = nv; stats.tiles = nt;
  document.getElementById("h-verts").textContent = nv.toLocaleString();
  const s = Math.min(W, H) / view.span;
  document.getElementById("h-scale").textContent = (100 / s).toFixed(1) + " m/100px";
}

// ---------- 交互 ----------
let drag = null;
cv.addEventListener("mousedown", e => { drag = {x: e.clientX, y: e.clientY, cx: view.cx, cy: view.cy}; cv.classList.add("drag"); });
window.addEventListener("mouseup", () => { drag = null; cv.classList.remove("drag"); });
window.addEventListener("mousemove", e => {
  if (!drag) return;
  const s = Math.min(cv.width, cv.height) / view.span;
  view.cx = drag.cx - (e.clientX - drag.x) / s;
  view.cy = drag.cy + (e.clientY - drag.y) / s;
  queueTiles();
});
cv.addEventListener("wheel", e => {
  e.preventDefault();
  const k = Math.exp(e.deltaY * 0.0012);
  view.span = Math.max(1, Math.min(MAN.bbox[2]-MAN.bbox[0] ? (MAN.bbox[2]-MAN.bbox[0])*4 : 1e6, view.span * k));
  queueTiles();
}, {passive: false});
document.getElementById("c-bias").addEventListener("input", queueTiles);
document.getElementById("c-fit").addEventListener("click", () => { fitView(); });

function resize(){ cv.width = innerWidth; cv.height = innerHeight; }
addEventListener("resize", () => { resize(); queueTiles(); });
resize();

function loop(t){
  frames++;
  if (t - lastT > 500){ fps = frames * 1000 / (t - lastT); frames = 0; lastT = t;
    document.getElementById("h-fps").textContent = fps.toFixed(0); }
  draw();
  requestAnimationFrame(loop);
}

fetch(CFG.manifest).then(r => r.json()).then(m => {
  MAN = m;
  document.getElementById("h-name").textContent = CFG.title || "矢量图层";
  stats.total = MAN.tiles ?? 0;
  document.getElementById("h-tiles").textContent = "0/" + MAN.levels[0].tiles.length;
  fitView();
  requestAnimationFrame(loop);
}).catch(e => fail("入口清单加载失败：" + e));
</script></body></html>
"""


def write_vector_viewer(out_dir: str, manifest="vector_tiles.json", title="GeoSqueeze · 矢量",
                        name="vector_viewer.html") -> str:
    cfg = {"manifest": manifest, "title": title}
    html = _HTML.replace("__TITLE__", title).replace("__CONFIG__", json.dumps(cfg, ensure_ascii=False))
    p = os.path.join(out_dir, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(html)
    return p
