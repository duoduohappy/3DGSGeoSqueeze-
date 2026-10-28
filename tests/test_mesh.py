"""纹理保留链路验收：带纹理网格城市 → 再压缩 → 几何/纹理/结构/内存全量验收。

运行：python tests/test_mesh.py
"""
import io
import json
import os
import shutil
import sys
import tempfile
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from gsq.cesium_viewer import write_cesium_viewer
from gsq.export import validate_b3dm, validate_tileset
from gsq.mesh import read_mesh_content, texture_psnr
from gsq.mesh_viewer import write_mesh_viewer
from gsq.samples import make_city_mesh
from gsq.tiles_in import compress_tileset

IN = OUT = None


def test_sample_mesh_structure():
    m = read_mesh_content(os.path.join(IN, "tiles", "0_0.b3dm"))
    assert len(m["prims"]) == 2 and len(m["images"]) == 2
    for p in m["prims"]:
        assert p["uv"] is not None and len(p["idx"]) % 3 == 0
        assert p["idx"].max() < len(p["pos"])
    for blob, mime in m["images"]:
        assert mime == "image/png" and len(blob) > 1000
    print("  sample mesh: 2 prims · 2 textures · 结构合法")


def test_zero_prune_roundtrip():
    r = compress_tileset(IN, OUT, prune=0.0, tex_max_dim=512, tex_quality=92, verbose=False)
    assert r["mesh_out"] == 9 and r["tiles_out"] == 9, r
    assert r["tris_out"] == r["tris_in"], "零抽稀必须三角形守恒"
    assert r["verts_out"] == r["verts_in"], "零抽稀必须顶点守恒"
    assert r["geom_quant_step"] < 0.01, f"几何量化步长过大 {r['geom_quant_step']}"
    assert r["tex_psnr_min"] >= 32.0, f"纹理 PSNR 越界 {r['tex_psnr_min']}dB"
    assert r["ratio"] > 1.0, "压缩率必须 >1（只赚不赔）"
    o = read_mesh_content(os.path.join(IN, "tiles", "0_0.b3dm"))
    c = read_mesh_content(os.path.join(OUT, "tiles", "1.b3dm"))
    assert [len(p["pos"]) for p in c["prims"]] == [len(p["pos"]) for p in o["prims"]]
    # 输出为原点相对坐标：世界坐标 = comp.pos + origin，须与输入世界坐标一致
    # （输入瓦片 0_0 的 transform 为平移 (0,0,0)，故其局部坐标即世界坐标）
    with open(os.path.join(OUT, "tileset.json"), encoding="utf-8") as f:
        origin = np.asarray(json.load(f)["asset"]["extras"]["origin"], np.float64)
    for po, pc in zip(o["prims"], c["prims"]):
        assert float(np.abs(po["pos"] - (pc["pos"] + origin)).max()) <= r["geom_quant_step"] + 1e-9
        assert float(np.abs(po["uv"] - pc["uv"]).max()) < 1e-4
    for (bo, _), (bc, mc) in zip(o["images"], c["images"]):
        assert mc in ("image/jpeg", "image/webp", "image/png")
        assert texture_psnr(bo, bc) >= 32.0
    print(f"  零抽稀回读: 三角形 {r['tris_in']} 守恒 · 顶点误差 ≤{r['geom_quant_step']*1000:.2f}mm · "
          f"纹理PSNR {r['tex_psnr_min']}~{r['tex_psnr_mean']}dB · {r['ratio']}x")


def test_prune_and_texture_budget():
    out2 = OUT + "_p"
    r = compress_tileset(IN, out2, prune=0.6, tex_max_dim=128, tex_quality=70, verbose=False)
    assert r["verts_out"] < r["verts_in"] * 0.6
    assert r["tris_out"] > 0, "几何守卫失效（抽空）"
    assert r["tex_bytes_out"] < r["tex_bytes_in"]
    assert r["ratio"] > 2.0
    m = read_mesh_content(os.path.join(out2, "tiles", "1.b3dm"))
    for blob, _mime in m["images"]:
        assert max(Image.open(io.BytesIO(blob)).size) <= 128
    print(f"  抽稀0.6: 顶点 {r['verts_in']}→{r['verts_out']} · 纹理 {r['tex_ratio']}x · {r['ratio']}x")


def test_structural_validation():
    ts = validate_tileset(os.path.join(OUT, "tileset.json"))
    assert ts["valid"], ts["errors"]
    n_mesh = 0
    for f in os.listdir(os.path.join(OUT, "tiles")):
        if f.endswith(".b3dm"):
            info = validate_b3dm(os.path.join(OUT, "tiles", f))
            assert info["mode"] == 4 and info["images"] >= 1
            n_mesh += 1
    assert n_mesh == 9
    print(f"  结构校验: {n_mesh} 网格瓦片全部合法")


def test_geometric_error_monotonic():
    with open(os.path.join(OUT, "tileset.json"), encoding="utf-8") as f:
        ts = json.load(f)

    def walk(t, parent):
        assert t["geometricError"] <= parent + 1e-9
        for c in t.get("children", []):
            walk(c, t["geometricError"])
    walk(ts["root"], ts["geometricError"])
    print("  geometricError 单调性通过")


def test_city_scale_stream_memory():
    big = "out/_test_mesh_big"
    outb = "out/_test_mesh_bigout"
    make_city_mesh(big, grid=6, tile_size=120.0, seed=1)
    tracemalloc.start()
    t0 = time.time()
    r = compress_tileset(big, outb, prune=0.5, tex_max_dim=256, tex_quality=75, verbose=False)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert r["mesh_out"] == 36
    assert peak / 1024 ** 2 < 150, f"峰值内存 {peak/1024**2:.1f}MB 超界"
    print(f"  城市级: 36 瓦片 · {r['ratio']}x · {time.time()-t0:.1f}s · 峰值 {peak/1024**2:.1f}MB")


def test_viewer_generated():
    assert os.path.isfile(os.path.join(OUT, "mesh_viewer.html"))
    html = open(os.path.join(OUT, "mesh_viewer.html"), encoding="utf-8").read()
    assert "tileset.json" in html and "webgl2" in html
    assert write_mesh_viewer(OUT) == os.path.join(OUT, "mesh_viewer.html")
    print("  mesh_viewer.html 生成并幂等")


def test_cesium_viewer():
    # tileset 声明：Cesium 1.1x 只读 asset.gltfUpAxis（字符串），缺失会把 Z-up 内容放倒
    ts = json.load(open(os.path.join(OUT, "tileset.json"), encoding="utf-8"))
    assert ts["asset"].get("gltfUpAxis") == "Z"
    # 查看器生成；小数据集自动内嵌为自包含单文件（双击打开不受 file:// CORS 限制）
    p = write_cesium_viewer(OUT)
    assert p == os.path.join(OUT, "cesium.html")
    html = open(p, encoding="utf-8").read()
    assert "vfile:///tileset.json" in html and "vfile:///tiles/1.b3dm" in html
    assert "modelUpAxis" in html and "Cesium.Axis.Z" in html
    assert write_cesium_viewer(OUT) == p  # 幂等
    print(f"  cesium.html 内嵌单文件 {os.path.getsize(p)/1024:.0f}KB，gltfUpAxis=Z")


def test_pointify_fallback():
    out3 = OUT + "_pt"
    r = compress_tileset(IN, out3, prune=0.0, keep_mesh=False, verbose=False)
    assert r["mesh_out"] == 0 and r["pts_in"] > 0 and r["pts_out"] == r["pts_in"]
    assert all(f.endswith(".pnts") for f in os.listdir(os.path.join(out3, "tiles")))
    print(f"  pointify 回退: {r['pts_in']} 点零丢失 → pnts")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        IN = os.path.join(td, "mesh_in")
        OUT = os.path.join(td, "mesh_out")
        print("GeoSqueeze 纹理保留验收")
        make_city_mesh(IN, grid=3, tile_size=120.0, seed=5)
        test_sample_mesh_structure()
        test_zero_prune_roundtrip()
        test_prune_and_texture_budget()
        test_structural_validation()
        test_geometric_error_monotonic()
        test_city_scale_stream_memory()
        test_viewer_generated()
        test_cesium_viewer()
        test_pointify_fallback()
    print("ALL PASSED")
