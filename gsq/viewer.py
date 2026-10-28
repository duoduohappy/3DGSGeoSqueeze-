"""生成自包含 3D Tiles 查看器：按 geometricError 流式加载瓦片，WebGL2 点精灵渲染高斯。

viewer 是交付物的一部分（验证"可流式加载"是否成立），不依赖任何外部库或 CDN。
"""
from __future__ import annotations
import json
import os

_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
  html,body{margin:0;height:100%;background:#1b1b19;color:#e8e6e0;
    font:13px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;overflow:hidden}
  #gl{display:block;width:100%;height:100%;cursor:grab}#gl:active{cursor:grabbing}
  #hud{position:fixed;left:14px;top:12px;background:rgba(20,20,18,.72);padding:10px 13px;
    border-radius:8px;backdrop-filter:blur(8px);min-width:190px;border:1px solid rgba(255,255,255,.08)}
  #hud b{font-weight:600;color:#fff}#hud div{white-space:nowrap;margin-top:2px}
  .k{color:#9a978e}#v{color:#c8e6a0;font-variant-numeric:tabular-nums}
  #ctl{position:fixed;right:14px;top:12px;background:rgba(20,20,18,.72);padding:10px 13px;
    border-radius:8px;backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.08);width:210px}
  #ctl label{display:flex;justify-content:space-between;gap:8px;margin:6px 0 2px;color:#b8b5ac}
  input[type=range]{width:100%;accent-color:#7fa650}
  button{width:100%;margin-top:8px;padding:6px;border-radius:6px;cursor:pointer;
    background:#2f3a24;color:#d6ecc0;border:1px solid #46592f;font:inherit}
  button:hover{background:#3a482c}
  #err{position:fixed;left:50%;top:50%;transform:translate(-50%,-50%);max-width:70%;padding:18px 22px;
    background:#3a1f1f;border:1px solid #7a3a3a;border-radius:10px;display:none;line-height:1.7}
  code{background:rgba(255,255,255,.09);padding:1px 5px;border-radius:3px}
</style></head><body>
<canvas id="gl"></canvas>
<div id="hud">
  <div><b>GeoSqueeze 流式查看器</b></div>
  <div><span class="k">瓦片</span> <span class="v" id="v_tiles">0 / 0</span></div>
  <div><span class="k">基元</span> <span class="v" id="v_pts">0</span></div>
  <div><span class="k">已下载</span> <span class="v" id="v_bytes">0 KB</span></div>
  <div><span class="k">帧率</span> <span class="v" id="v_fps">—</span></div>
  <div><span class="k">绘制</span> <span class="v" id="v_draw">0 ms</span></div>
</div>
<div id="ctl">
  <label>屏幕误差阈值 <span id="l_sse">1.0</span></label>
  <input id="sse" type="range" min="0.2" max="8" step="0.1" value="1">
  <label>点尺寸倍率 <span id="l_size">1.0</span></label>
  <input id="size" type="range" min="0.2" max="3" step="0.1" value="1">
  <button id="all">加载全部瓦片</button>
</div>
<div id="err"></div>
<script>
"use strict";
const CFG = __CONFIG__;

const canvas = document.getElementById("gl");
const gl = canvas.getContext("webgl2", {antialias: false, premultipliedAlpha: true});
if (!gl) fail("浏览器不支持 WebGL2");

const VS = `#version 300 es
precision highp float;
in vec3 a_pos;      // 已还原到世界坐标
in vec4 a_col;      // rgb + alpha
in vec4 a_sh0;      // SH 系数（deg1，12 分量展平为 3 个 vec4）
in vec4 a_sh1;
in vec4 a_sh2;
in float a_size;    // 世界尺度
uniform mat4 u_mvp;
uniform vec3 u_eye;
uniform float u_focal;
uniform float u_size;
uniform float u_shScale;
uniform float u_deg;
uniform float u_minPx;      // 密度自适应最小点径（稀疏基元补齐可见性）
out vec4 v_col;
void main(){
  vec4 vp = u_mvp * vec4(a_pos, 1.0);
  if (vp.w <= 0.0) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }  // 相机背后的点直接裁掉
  gl_Position = vp;
  float depth = max(vp.w, 0.0001);
  float px = max(a_size * u_size * u_focal / depth, u_minPx);
  gl_PointSize = clamp(px, 1.0, 512.0);
  vec3 rgb = a_col.rgb;
  if (u_deg > 0.5) {
    vec3 dir = normalize(a_pos - u_eye);          // 相机 → 基元
    vec3 s0 = a_sh0.xyz;
    vec3 s1 = vec3(a_sh0.w, a_sh1.x, a_sh1.y);
    vec3 s2 = vec3(a_sh1.z, a_sh1.w, a_sh2.x);
    vec3 s3 = a_sh2.yzw;
    vec3 c = s0 * 0.2820948 - s1 * (0.4886025 * dir.y)
           + s2 * (0.4886025 * dir.z) - s3 * (0.4886025 * dir.x);
    rgb = clamp(0.5 + c * u_shScale, 0.0, 1.0);
  }
  v_col = vec4(rgb, a_col.a);
}`;
const FS = `#version 300 es
precision highp float;
in vec4 v_col;
out vec4 o;
void main(){
  vec2 d = gl_PointCoord - vec2(0.5);
  float r2 = dot(d, d);
  if (r2 > 0.25) discard;
  float a = v_col.a * exp(-4.0 * r2);            // 圆形软边，近似高斯核
  if (a < 0.004) discard;
  o = vec4(v_col.rgb * a, a);                    // 预乘 alpha
}`;

function sh(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) fail("着色器编译失败：" + gl.getShaderInfoLog(s));
  return s;
}
const prog = gl.createProgram();
gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS));
gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) fail("着色器链接失败");
gl.useProgram(prog);

const U = {};
for (const n of ["u_mvp", "u_eye", "u_focal", "u_size", "u_shScale", "u_deg", "u_minPx"])
  U[n] = gl.getUniformLocation(prog, n);
const A = {};
for (const n of ["a_pos", "a_col", "a_sh0", "a_sh1", "a_sh2", "a_size"])
  A[n] = gl.getAttribLocation(prog, n);

gl.enable(gl.DEPTH_TEST);
gl.enable(gl.BLEND);
gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
gl.clearColor(0.106, 0.106, 0.098, 1);

// ---------------- 相机（绕 bbox 中心轨道） ----------------
const cam = {yaw: 0.9, pitch: 0.42, dist: 3.0, target: [0, 0, 0], fov: 50 * Math.PI / 180};
let sceneBox = null;

function mat4Mul(a, b) {
  const o = new Float32Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k];
    o[i * 4 + j] = s;
  }
  return o;
}
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
}
function lookAt(eye, ctr, up) {
  const z = norm(sub(eye, ctr)), x = norm(cross(up, z)), y = cross(z, x);
  return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
}
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = a => Math.hypot(a[0], a[1], a[2]);
const norm = a => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const eyePos = () => {
  const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw), cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  return [cam.target[0] + cam.dist * cp * cy, cam.target[1] + cam.dist * cp * sy, cam.target[2] + cam.dist * sp];
};

// ---------------- GLB 解析 ----------------
const COMP = {5120: [Int8Array, 1], 5121: [Uint8Array, 1], 5122: [Int16Array, 2],
              5123: [Uint16Array, 2], 5125: [Uint32Array, 4], 5126: [Float32Array, 4]};
const NCOMP = {SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16};

function readAccessor(g, bin, idx) {
  const a = g.accessors[idx], bv = g.bufferViews[a.bufferView];
  const [Ctor, sz] = COMP[a.componentType];
  const nc = NCOMP[a.type];
  const off = (bv.byteOffset || 0) + (a.byteOffset || 0);
  const n = a.count * nc;
  const stride = bv.byteStride || 0;
  let arr;
  if (stride && stride !== nc * sz) {                 // 交错存储
    const dv = new DataView(bin, off);
    arr = new Ctor(n);
    for (let i = 0; i < a.count; i++)
      for (let c = 0; c < nc; c++)
        arr[i * nc + c] = readComp(dv, i * stride + c * sz, a.componentType);
  } else {
    arr = new Ctor(bin.slice(off, off + n * sz).buffer);   // slice 得到干净的独立 buffer
  }
  return {arr, nc, normalized: !!a.normalized, componentType: a.componentType};
}
function readComp(dv, o, ct) {
  switch (ct) {
    case 5126: return dv.getFloat32(o, true);
    case 5123: return dv.getUint16(o, true);
    case 5121: return dv.getUint8(o);
    case 5120: return dv.getInt8(o);
    case 5122: return dv.getInt16(o, true);
    default: return 0;
  }
}
function decode(acc) {
  let {arr, componentType, normalized} = acc;
  if (!normalized) return arr;
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    if (componentType === 5123) out[i] = arr[i] / 65535;
    else if (componentType === 5121) out[i] = arr[i] / 255;
    else if (componentType === 5120) out[i] = Math.max(arr[i] / 127, -1);
    else if (componentType === 5122) out[i] = Math.max(arr[i] / 32767, -1);
    else out[i] = arr[i];
  }
  return out;
}

async function loadGLB(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("HTTP " + res.status + " " + url);
  const raw = new Uint8Array(await res.arrayBuffer());
  const dv = new DataView(raw.buffer);
  if (dv.getUint32(0, true) !== 0x46546C67) throw new Error("非 GLB 文件: " + url);
  const jl = dv.getUint32(12, true);
  const g = JSON.parse(new TextDecoder().decode(raw.subarray(20, 20 + jl)));
  const binOff = 20 + jl + 8, binLen = dv.getUint32(20 + jl, true);
  const bin = raw.subarray(binOff, binOff + binLen);
  const prim = g.meshes[0].primitives[0];
  const at = prim.attributes;
  const node = g.nodes[0] || {};
  const tr = node.translation || [0, 0, 0];
  const sc = node.scale ? node.scale[0] : 1;

  const P = decode(readAccessor(g, bin, at.POSITION));        // normalized u16 或 float
  const n = P.length / 3;
  const out = {
    n,
    pos: new Float32Array(n * 3),
    col: new Float32Array(n * 4),
    sh: new Float32Array(n * 12),
    size: new Float32Array(n),
    deg: 0, shScale: 1,
  };
  for (let i = 0; i < n; i++)
    for (let c = 0; c < 3; c++) out.pos[i * 3 + c] = P[i * 3 + c] * sc + tr[c];

  if (at.COLOR_0 !== undefined) {
    const C = decode(readAccessor(g, bin, at.COLOR_0));
    for (let i = 0; i < n; i++)
      for (let c = 0; c < 3; c++) out.col[i * 4 + c] = C[i * 3 + c];
  } else out.col.fill(0.7);
  let alpha = 0.9;
  if (at._OPACITY !== undefined) {
    const O = decode(readAccessor(g, bin, at._OPACITY));
    for (let i = 0; i < n; i++) out.col[i * 4 + 3] = Math.min(1, Math.max(0, O[i]));
  } else for (let i = 0; i < n; i++) out.col[i * 4 + 3] = alpha;

  const ex = prim.extras || {};
  const smax = ex.scale_max || (at._SCALE === undefined ? 0.2 : 1);
  if (at._SCALE !== undefined) {
    const S = decode(readAccessor(g, bin, at._SCALE));
    for (let i = 0; i < n; i++) {
      const s = (S[i * 3] + S[i * 3 + 1] + S[i * 3 + 2]) / 3;
      out.size[i] = s * smax;
    }
  } else out.size.fill(0.2);

  if (ex.sh_vectors) {
    const nv = Math.min(ex.sh_vectors, 3);
    const shScale = ex.sh_scale || 1;
    const parts = [];
    for (let j = 0; j < nv; j++) {
      const key = "_SH" + j;
      if (at[key] === undefined) break;
      parts.push(decode(readAccessor(g, bin, at[key])));
    }
    const total = parts.length * 4;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < total; j++) out.sh[i * 12 + j] = parts[Math.floor(j / 4)][i * 4 + (j % 4)];
    out.deg = ex.sh_degree || 1;
    out.shScale = shScale;
  }
  return out;
}

// ---------------- 瓦片流式加载 ----------------
const tiles = new Map();                       // uri -> {buf, count, bytes}
let root = null, totalTiles = 0, loadedTiles = 0, totalPts = 0, totalBytes = 0;

function walkCount(t) { totalTiles++; (t.children || []).forEach(walkCount); }

// pnts 瓦片（量化点云，城市级再压缩产物）：u16 体量化位置 + u8 RGB
async function loadPnts(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("HTTP " + res.status + " " + url);
  const raw = new Uint8Array(await res.arrayBuffer());
  const dv = new DataView(raw.buffer);
  if (String.fromCharCode(raw[0],raw[1],raw[2],raw[3]) !== "pnts") throw new Error("非 pnts: " + url);
  const ftjl = dv.getUint32(12, true), ftbl = dv.getUint32(16, true);
  const head = JSON.parse(new TextDecoder().decode(raw.subarray(28, 28 + ftjl)));
  const bin = 28 + ftjl;                 // FT 二进制紧跟 FT JSON（与 Python 读取端一致）
  const n = head.POINTS_LENGTH;
  const qk = head.POSITION_QUANTIZED ? "POSITION_QUANTIZED" : "POSITION";
  const po = head[qk] ? head[qk].byteOffset : 0;
  const ro = head.RGB ? head.RGB.byteOffset : -1;
  const qo = head.QUANTIZED_VOLUME_OFFSET || [0,0,0];
  const qs = head.QUANTIZED_VOLUME_SCALE || [1,1,1];
  const out = { n, pos: new Float32Array(n*3), col: new Float32Array(n*4),
                sh: new Float32Array(n*12), size: new Float32Array(n), deg: 0, shScale: 1 };
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const q = dv.getUint16(bin + po + (i*3+c)*2, true);
      out.pos[i*3+c] = q / 65535 * qs[c] + qo[c];
    }
    for (let c = 0; c < 3; c++)
      out.col[i*4+c] = ro >= 0 ? raw[bin + ro + i*3 + c] / 255 : 0.7;
    out.col[i*4+3] = 0.95;
    out.size[i] = 1;
  }
  return out;
}

async function ensureTile(t) {
  const uri = t.content && t.content.uri;
  if (!uri || tiles.has(uri)) return tiles.get(uri);
  const data = /\.pnts$/i.test(uri) ? await loadPnts(uri) : await loadGLB(uri);
  const buf = gl.createBuffer();
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const bind = (loc, arr, size) => {
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
    if (loc >= 0) { gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0); }
  };
  bind(A.a_pos, data.pos, 3);
  bind(A.a_col, data.col, 4);
  // SH：12 个 float 交错存储，按 3×vec4、stride 48 绑定
  const shBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, shBuf);
  gl.bufferData(gl.ARRAY_BUFFER, data.sh, gl.STATIC_DRAW);
  for (let j = 0; j < 3; j++) {
    const loc = A["a_sh" + j];
    if (loc >= 0) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 48, j * 16);
    }
  }
  bind(A.a_size, data.size, 1);
  gl.bindVertexArray(null);
  const rec = {vao, n: data.n, deg: data.deg, shScale: data.shScale, bytes: data.n * 20};
  tiles.set(uri, rec);
  loadedTiles++; totalPts += data.n; totalBytes += rec.bytes;
  return rec;
}

function tilesToDraw() {
  const eye = eyePos();
  const out = [];
  const sse = parseFloat(document.getElementById("sse").value);
  const visit = t => {
    const c = boxCenter(t);
    const d = len(sub(c, eye));
    const ge = t.geometricError || 0;
    // 屏幕空间误差：ge(世界) × focal(px) / dist → 投影像素数，超过阈值则需更细层
    const focal = (canvas.height || 780) / (2 * Math.tan(cam.fov / 2));
    if (ge <= 0 || ge * focal / Math.max(d, 1e-3) > sse || forceAll) {
      if (t.content && t.content.uri) out.push(t);
      (t.children || []).forEach(visit);
    }
  };
  if (root) visit(root);
  return out;
}
function boxCenter(t) {
  const b = t.boundingVolume.box;
  if (!b) return [0, 0, 0];
  return [b[0], b[1], b[2]];
}

let forceAll = false;
let pending = false;
async function stream() {
  if (pending) return;
  pending = true;
  try {
    const list = tilesToDraw();
    for (const t of list) await ensureTile(t);
    drawList = list.map(t => tiles.get(t.content.uri)).filter(Boolean);
  } catch (e) {
    console.warn(e);
  }
  pending = false;
}
let drawList = [];

// ---------------- 渲染 ----------------
function frame(now) {
  const eye = eyePos();
  const asp = canvas.width / canvas.height;
  const near = Math.max(cam.dist * 0.01, 0.01), far = cam.dist * 60 + 1000;
  const view = lookAt(eye, cam.target, [0, 0, 1]);
  const mvp = mat4Mul(perspective(cam.fov, asp, near, far), view);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.uniformMatrix4fv(U.u_mvp, false, mvp);
  gl.uniform3fv(U.u_eye, eye);
  gl.uniform1f(U.u_focal, canvas.height / (2 * Math.tan(cam.fov / 2)));
  gl.uniform1f(U.u_size, parseFloat(document.getElementById("size").value));
  gl.uniform1f(U.u_minPx, Math.min(6, Math.max(1, 0.8 * Math.sqrt(canvas.width * canvas.height / Math.max(totalPts, 1)))));
  const t0 = performance.now();
  for (const r of drawList) {
    gl.uniform1f(U.u_shScale, r.shScale);
    gl.uniform1f(U.u_deg, r.deg);
    gl.bindVertexArray(r.vao);
    gl.drawArrays(gl.POINTS, 0, r.n);
  }
  gl.bindVertexArray(null);
  const t1 = performance.now();

  document.getElementById("v_tiles").textContent = loadedTiles + " / " + totalTiles;
  document.getElementById("v_pts").textContent = totalPts.toLocaleString();
  document.getElementById("v_bytes").textContent = (totalBytes / 1024).toFixed(0) + " KB";
  document.getElementById("v_draw").textContent = (t1 - t0).toFixed(1) + " ms";
  frames.push(now);
  while (frames.length && now - frames[0] > 1000) frames.shift();
  if (frames.length > 2) document.getElementById("v_fps").textContent = frames.length + "";

  requestAnimationFrame(frame);
}
const frames = [];

// ---------------- 交互 ----------------
let drag = false, lx = 0, ly = 0;
canvas.addEventListener("pointerdown", e => { drag = true; lx = e.clientX; ly = e.clientY; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener("pointerup", e => { drag = false; });
canvas.addEventListener("pointermove", e => {
  if (!drag) return;
  cam.yaw -= (e.clientX - lx) * 0.006;
  cam.pitch = Math.max(-1.4, Math.min(1.4, cam.pitch + (e.clientY - ly) * 0.006));
  lx = e.clientX; ly = e.clientY;
  stream();
});
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  cam.dist *= Math.exp(e.deltaY * 0.0012);
  stream();
}, {passive: false});
document.getElementById("all").onclick = () => { forceAll = true; stream(); };
document.getElementById("sse").oninput = e => {
  document.getElementById("l_sse").textContent = e.target.value;
  forceAll = false; stream();
};
document.getElementById("size").oninput = e => { document.getElementById("l_size").textContent = e.target.value; };

function resize() {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.floor(innerWidth * dpr);
  canvas.height = Math.floor(innerHeight * dpr);
}
addEventListener("resize", resize);
function fail(msg) {
  const d = document.getElementById("err");
  d.style.display = "block";
  d.innerHTML = "<b>无法启动查看器</b><br>" + msg +
    "<br><br>若通过 <code>file://</code> 打开，浏览器会阻止读取瓦片。请在本目录运行：" +
    "<br><code>python -m gsq serve .</code> 然后访问命令行提示的地址。";
  throw new Error(msg);
}

(async function boot() {
  resize();
  try {
    const res = await fetch(CFG.tileset);
    if (!res.ok) throw new Error("HTTP " + res.status);
    const ts = await res.json();
    root = ts.root;
    walkCount(root);
    const b = root.boundingVolume.box;
    if (b) {
      cam.target = [b[0], b[1], b[2]];
      cam.dist = Math.max(b[3], b[4], b[5]) * 3.2 || 10;
    }
    document.getElementById("sse").value = CFG.sse || 1;
    document.getElementById("l_sse").textContent = document.getElementById("sse").value;
    await stream();
    requestAnimationFrame(frame);
  } catch (e) {
    fail(String(e.message || e));
  }
})();
</script></body></html>
"""


def write_viewer(out_dir: str, tileset: str = "tileset.json", title: str = "GeoSqueeze",
                 sse: float = 1.0) -> str:
    cfg = {"tileset": tileset, "sse": sse}
    html = _HTML.replace("__TITLE__", title).replace("__CONFIG__", json.dumps(cfg))
    p = os.path.join(out_dir, "viewer.html")
    with open(p, "w", encoding="utf-8") as f:
        f.write(html)
    return p
