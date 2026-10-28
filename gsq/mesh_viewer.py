"""城市级纹理网格 tileset 的 WebGL2 流式查看器（零依赖单文件）。

能力：SSE 调度逐瓦片加载 b3dm/glb → 解析 KHR_mesh_quantization u16 位置 + u16 UV +
嵌入 JPEG/PNG/WebP 纹理 → 纹理三角网格渲染（屏幕空间法线 lambert）。
"""
from __future__ import annotations

import json
import os

_TMPL = r"""<!doctype html><html lang="zh"><meta charset="utf-8">
<title>__TITLE__ · GeoSqueeze 纹理网格</title>
<style>
html,body{margin:0;height:100%;overflow:hidden;background:#101418;font-family:system-ui,sans-serif}
#cv{width:100%;height:100%;display:block}
#hud{position:fixed;left:12px;top:12px;color:#dfe7ee;font-size:12px;background:rgba(16,20,24,.72);
padding:8px 12px;border-radius:8px;line-height:1.7;pointer-events:none}
#tip{position:fixed;right:12px;bottom:12px;color:#8fa1b3;font-size:11px}
</style>
<canvas id="cv"></canvas><div id="hud"></div><div id="tip">拖拽旋转 · 滚轮缩放 · 右键平移</div>
<script>
"use strict";
const cv = document.getElementById('cv'), hud = document.getElementById('hud');
const gl = cv.getContext('webgl2', {antialias: true});
if (!gl) { hud.textContent = '需要 WebGL2'; throw new Error('no webgl2'); }
const _errs = [];
function showErr(e){ _errs.push(String(e && e.message || e));
  hud.innerHTML = '错误: ' + _errs.slice(-2).join(' | '); }
addEventListener('error', ev => showErr(ev.error || ev.message));
addEventListener('unhandledrejection', ev => showErr(ev.reason));

const VS = `#version 300 es
layout(location=0) in vec3 aP; layout(location=1) in vec2 aT;
uniform mat4 uMVP; out vec2 vT; out vec3 vW;
void main(){ vT = aT; vW = aP; gl_Position = uMVP * vec4(aP, 1.0); }`;
const FS = `#version 300 es
precision highp float;
in vec2 vT; in vec3 vW; uniform sampler2D uS; uniform vec3 uEye; out vec4 o;
void main(){
  vec3 n = normalize(cross(dFdx(vW), dFdy(vW)));
  float lam = 0.62 + 0.38 * abs(dot(n, normalize(vec3(0.4, 0.3, 0.85))));
  vec4 c = texture(uS, vT);
  if (c.a < 0.05) discard;
  o = vec4(c.rgb * lam, 1.0);
}`;

function sh(t, s){ const o = gl.createShader(t); gl.shaderSource(o, s); gl.compileShader(o);
  if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw gl.getShaderInfoLog(o); return o; }
const prog = gl.createProgram();
gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog); gl.useProgram(prog);
const uMVP = gl.getUniformLocation(prog, 'uMVP'), uEye = gl.getUniformLocation(prog, 'uEye');
gl.enable(gl.DEPTH_TEST);

// ---------- 矩阵 ----------
function m4mul(a, b){ const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++)
    o[c*4+r] = a[r]*b[c*4] + a[4+r]*b[c*4+1] + a[8+r]*b[c*4+2] + a[12+r]*b[c*4+3];
  return o; }
function m4persp(f, ar, n, f4){ const t = 1/Math.tan(f/2);
  return new Float32Array([t/ar,0,0,0, 0,t,0,0, 0,0,(f4+n)/(n-f4),-1, 0,0,2*f4*n/(n-f4),0]); }
function m4look(e, c, u){
  const z = norm3(sub3(e, c)), x = norm3(cross3(u, z)), y = cross3(z, x);
  return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0,
    -(x[0]*e[0]+x[1]*e[1]+x[2]*e[2]), -(y[0]*e[0]+y[1]*e[1]+y[2]*e[2]),
    -(z[0]*e[0]+z[1]*e[1]+z[2]*e[2]), 1]); }
const sub3=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const cross3=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm3=a=>{const l=Math.hypot(...a)||1;return [a[0]/l,a[1]/l,a[2]/l];};

// ---------- 相机（轨道） ----------
let yaw = 0.8, pitch = 0.5, dist = 0, target = [0,0,0], fov = 50*Math.PI/180;
function eye(){ const cp = Math.cos(pitch);
  return [target[0]+dist*cp*Math.cos(yaw), target[1]+dist*cp*Math.sin(yaw), target[2]+dist*Math.sin(pitch)]; }
let drag = null;
cv.onmousedown = e => drag = {x: e.clientX, y: e.clientY, b: e.button};
addEventListener('mouseup', () => drag = null);
addEventListener('mousemove', e => { if (!drag) return;
  const dx = e.clientX-drag.x, dy = e.clientY-drag.y; drag = {x: e.clientX, y: e.clientY, b: drag.b};
  if (drag.b === 2 || drag.b === 1) {
    const s = dist * 0.0016, r = m4rotz(yaw);
    target[0] -= (r[0]*dx - r[3]*dy)*s; target[1] -= (r[1]*dx - r[4]*dy)*s;
  } else { yaw -= dx*0.005; pitch = Math.min(1.52, Math.max(-0.1, pitch + dy*0.005)); } });
cv.oncontextmenu = e => e.preventDefault();
cv.onwheel = e => { e.preventDefault(); dist *= Math.pow(1.1, Math.sign(e.deltaY)); dist = Math.max(2, dist); };
function m4rotz(a){ const c = Math.cos(a), s = Math.sin(a); return [c, s, 0, -s, c, 0, 0, 0, 1]; }

// ---------- b3dm/GLB 解析 ----------
const CT = {5120:[Int8Array,1,127],5121:[Uint8Array,1,255],5122:[Int16Array,2,32767],
            5123:[Uint16Array,2,65535],5125:[Uint32Array,4,1],5126:[Float32Array,4,1]};
const NC = {SCALAR:1, VEC2:2, VEC3:3, VEC4:4};
function acc(j, bin, i){
  const a = j.accessors[i], bv = j.bufferViews[a.bufferView];
  const [T, sz, den] = CT[a.componentType], nc = NC[a.type];
  const base = bin.byteOffset + (bv.byteOffset||0) + (a.byteOffset||0);
  const buf = bin.buffer || bin;
  const st = bv.byteStride || sz*nc;
  const out = new Float32Array(a.count*nc);
  const g = 'get' + T.name.replace('Array','');
  for (let k = 0; k < a.count; k++) for (let c = 0; c < nc; c++)
    out[k*nc+c] = new DataView(buf, base+k*st+c*sz, sz)[g](0, true) / den;
  return out; }
function parseTile(buf){
  let u8 = new Uint8Array(buf);
  if (String.fromCharCode(...u8.slice(0,4)) === 'b3dm') {
    const v = new DataView(buf);
    const fl = v.getUint32(12,1)+v.getUint32(16,1)+v.getUint32(20,1)+v.getUint32(24,1);
    u8 = u8.slice(28+fl);
  }
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const jl = dv.getUint32(12,1), binOff = 20+jl+8, bl = dv.getUint32(20+jl,1);
  const j = JSON.parse(new TextDecoder().decode(u8.slice(20, 20+jl)));
  const bin = u8.slice(binOff, binOff+bl);
  const meshes = [];
  const root = j.scenes[j.scene||0].nodes[0], nd = j.nodes[root];
  const tr = nd.translation||[0,0,0], sc = nd.scale||[1,1,1];
  for (const m of j.meshes) for (const pr of m.primitives) {
    if ((pr.mode??4) !== 4) continue;
    const P = acc(j, bin, pr.attributes.POSITION), n = P.length/3;
    for (let k = 0; k < n; k++) { P[k*3]=P[k*3]*sc[0]+tr[0]; P[k*3+1]=P[k*3+1]*sc[1]+tr[1]; P[k*3+2]=P[k*3+2]*sc[2]+tr[2]; }
    const T = pr.attributes.TEXCOORD_0 != null ? acc(j, bin, pr.attributes.TEXCOORD_0) : new Float32Array(n*2);
    const I = acc(j, bin, pr.indices);
    const mat = j.materials[pr.material||0], texIdx = mat && mat.pbrMetallicRoughness &&
      mat.pbrMetallicRoughness.baseColorTexture ? mat.pbrMetallicRoughness.baseColorTexture.index : -1;
    let tex = null;
    if (texIdx >= 0) { const im = j.images[j.textures[texIdx].source], bv = j.bufferViews[im.bufferView];
      const blob = new Blob([bin.slice(bv.byteOffset||0, (bv.byteOffset||0)+bv.byteLength)], {type: im.mimeType});
      tex = blob; }
    meshes.push({P, T, I, tex});
  }
  return meshes; }

// ---------- tileset 调度 ----------
const tiles = new Map(); let TS = null, ROOT_M = null, bboxR = 1;
function colM(t16){ return new Float32Array(t16); }   // 3D Tiles transform 即列主序，直接用
function boxRadius(bv){
  if (bv.box) return Math.hypot(bv.box[3], bv.box[7], bv.box[11]);
  if (bv.sphere) return bv.sphere[3]; return 10; }
function boxCenter(bv){
  if (bv.box) return bv.box.slice(0,3); if (bv.sphere) return bv.sphere.slice(0,3); return [0,0,0]; }
function loadTileset(url){
  return fetch(url).then(r => r.json()).then(ts => {
    TS = ts; ROOT_M = ts.root.transform ? colM(ts.root.transform) : null;
    // 相机初始化：以 root 包围盒为中心
    const c = boxCenter(ts.root.boundingVolume);
    if (ROOT_M) { target = [c[0]+ROOT_M[12], c[1]+ROOT_M[13], c[2]+ROOT_M[14]]; }
    else target = c;
    bboxR = boxRadius(ts.root.boundingVolume) || 100; dist = bboxR * 2.2; }); }
function worldM(t, parent){
  let m = parent;
  if (t.transform) m = m4mul(m, colM(t.transform));
  return m; }
function ssePx(t, eyeP, h){
  const c = boxCenter(t.boundingVolume), r = boxRadius(t.boundingVolume);
  const m = t._wm || ROOT_M || new Float32Array(16);
  const wc = [c[0]*m[0]+c[1]*m[4]+c[2]*m[8]+m[12], c[0]*m[1]+c[1]*m[5]+c[2]*m[9]+m[13],
              c[0]*m[2]+c[1]*m[6]+c[2]*m[10]+m[14]];
  const d = Math.max(1, Math.hypot(wc[0]-eyeP[0], wc[1]-eyeP[1], wc[2]-eyeP[2]) - r);
  return (t.geometricError||0) * h / (2*d*Math.tan(fov/2)); }
function visit(t, parentM, eyeP, h, out){
  t._wm = worldM(t, parentM);
  const s = ssePx(t, eyeP, h);
  if (s > 1.2 && t.children && t.children.length) {
    for (const c of t.children) visit(c, t._wm, eyeP, h, out);
    return; }
  if (t.content) out.push(t); }
function ensure(t){
  const uri = t.content.uri; let e = tiles.get(uri);
  if (!e) { e = {st: 'loading', meshes: null}; tiles.set(uri, e);
    fetch(uri).then(r => { if (!r.ok) throw new Error(uri + ' HTTP ' + r.status); return r.arrayBuffer(); })
      .then(b => {
        const ms = parseTile(b); let pend = ms.length;
        if (!pend) { e.st = 'ready'; e.meshes = ms; return; }
        for (const m of ms) {
          if (m.tex) {
            createImageBitmap(m.tex).then(bm => { m.bmp = bm; if (--pend === 0) e.st = 'ready'; })
              .catch(err => { m.bmp = null; if (--pend === 0) e.st = 'ready'; showErr(err); });
          } else if (--pend === 0) e.st = 'ready';
        }
        e.meshes = ms;
      }).catch(err => { e.st = 'failed'; showErr(err); });
    e.st = 'loading'; }
  return e; }
const vaos = new Map();
function drawMesh(m, wm){
  let vao = vaos.get(m);
  if (!vao) { vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    const vb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vb);
    gl.bufferData(gl.ARRAY_BUFFER, m.P, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    const tb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, tb);
    gl.bufferData(gl.ARRAY_BUFFER, m.T, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 0, 0);
    const ib = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(m.I), gl.STATIC_DRAW);
    vaos.set(m, vao); }
  gl.bindVertexArray(vao);
  if (m.bmp) { const t = m._tex || (m._tex = (() => { const x = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, x); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, m.bmp);
      gl.generateMipmap(gl.TEXTURE_2D);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT); return x; })());
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, t); }
  // t._wm 已含根 transform（visit 从 ROOT_M 起累积），直接用
  gl.uniformMatrix4fv(uMVP, false, m4mul(vp, wm));
  gl.drawElements(gl.TRIANGLES, m.I.length, gl.UNSIGNED_INT, 0); }

let vp = null, stats = {tiles: 0, tris: 0}, fno = 0;
function frame(){
  fno++;
  try { _frame(); } catch (err) { showErr(err); }
  requestAnimationFrame(frame); }
function _frame(){
  const dpr = devicePixelRatio || 1;
  const w = innerWidth, h = innerHeight;
  if (cv.width !== w*dpr || cv.height !== h*dpr) { cv.width = w*dpr; cv.height = h*dpr; }
  gl.viewport(0, 0, cv.width, cv.height);
  gl.clearColor(0.063, 0.078, 0.094, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  if (!TS) { requestAnimationFrame(frame); return; }
  const e = eye();
  const view = m4look(e, target, [0,0,1]);
  vp = m4mul(m4persp(fov, w/h, dist*0.01, dist*10 + bboxR*4), view);
  gl.uniform3f(uEye, e[0], e[1], e[2]);
  const vis = []; visit(TS.root, ROOT_M || new Float32Array(16), e, h, vis);
  let tris = 0, ready = 0;
  const stc = {};
  for (const t of vis) { const ent = ensure(t); stc[ent.st] = (stc[ent.st]||0)+1;
    if (ent.st === 'ready' && ent.meshes) { ready++;
      for (const m of ent.meshes) { drawMesh(m, t._wm); tris += m.I.length/3; } } }
  stats = {tiles: ready, tris};
  hud.innerHTML = `瓦片 ${ready} · 三角形 ${tris.toLocaleString()} · 距离 ${dist.toFixed(0)}m` +
    (_errs.length ? `<br><span style="color:#ff9">错误: ${_errs.slice(-2).join(' | ')}</span>` : '');
}
loadTileset('tileset.json').then(() => requestAnimationFrame(frame));
</script></html>"""


def write_mesh_viewer(out_dir: str, title: str = "GeoSqueeze 纹理网格") -> str:
    p = os.path.join(out_dir, "mesh_viewer.html")
    with open(p, "w", encoding="utf-8") as f:
        f.write(_TMPL.replace("__TITLE__", title))
    return p
