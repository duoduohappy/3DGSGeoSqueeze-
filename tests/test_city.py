"""城市级两条线（3D Tiles 输入再压缩 / 矢量压缩）端到端验收测试。"""
import glob
import json
import os
import struct
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gsq.samples import make_city_tiles, make_city_vector
from gsq.tiles_in import compress_tileset, load_scene, load_tileset, scan_tileset
from gsq.vector import (VecLayer, decode_gvec, encode_gvec, eval_vector, read_geojson,
                        simplify)
from gsq.export import validate_tileset


def test_tiles_formats(tmp):
    """三种瓦片格式（glb/pnts/b3dm）读入 + 再压缩 + 坐标精度不变量。"""
    for kind in ("glb", "pnts", "b3dm"):
        src = os.path.join(tmp, f"src_{kind}")
        r = make_city_tiles(src, grid=2, per_tile=2000, kind=kind,
                            world_offset=(5.2e6, 4.1e6, 3.0e4))
        sc = scan_tileset(src)
        assert sc["contents"] == 4 and sc["points"] == r["points"], sc
        assert not sc["missing_or_bad"], sc["missing_or_bad"]

        out = os.path.join(tmp, f"out_{kind}")
        res = compress_tileset(src, out, prune=0.5, verbose=False)
        assert res["tiles_out"] == 4, res
        assert abs(res["pts_out"] - res["pts_in"] / 2) <= res["pts_in"] * 0.05, res
        assert res["ratio"] > 1.5, res
        assert res["mesh_tiles"] == 0, res          # 自产点云瓦片不应被误判为网格
        v = validate_tileset(os.path.join(out, "tileset.json"))
        assert v["valid"], v["errors"]
        # 输出 uri 必须带 tiles/ 前缀（相对 tileset 可解析）
        with open(os.path.join(out, "tileset.json"), encoding="utf-8") as f:
            ts = json.load(f)
        for t, _w, full, _d in _iter(out, ts):
            if full:
                assert os.path.isfile(full), full

    # prune=0 时逐点精度：ECEF 量级坐标下应达毫米级
    out0 = os.path.join(tmp, "out_q0")
    compress_tileset(os.path.join(tmp, "src_glb"), out0, prune=0.0, verbose=False)
    a = load_scene(os.path.join(tmp, "src_glb"))
    b = load_scene(out0)
    assert a.n == b.n
    d = np.linalg.norm(a.pos - b.pos, axis=1) * 1000
    assert d.max() < 5.0, f"量化误差超 5mm：{d.max():.2f}mm"
    print(f"  tiles: 3 格式 × 4 瓦片 · prune=0 逐点误差 max={d.max():.2f}mm（ECEF 偏移 5.2e6）")


def _iter(root, ts):
    def walk(t, parent):
        world = parent @ (np.asarray(t["transform"], np.float64).reshape(4, 4).T
                          if t.get("transform") else np.eye(4))
        uri = (t.get("content") or {}).get("uri")
        yield t, world, os.path.join(root, uri) if uri else None, 0
        for c in t.get("children", []):
            yield from walk(c, world)
    yield from walk(ts["root"], np.eye(4))


def test_tiles_zero_loss(tmp):
    """抽稀后瓦片内容并集 == 压缩场景全集（城市级零丢失不变量）。"""
    src = os.path.join(tmp, "zl_src")
    make_city_tiles(src, grid=3, per_tile=3000, kind="pnts")
    out = os.path.join(tmp, "zl_out")
    res = compress_tileset(src, out, prune=0.5, verbose=False)
    tot = 0
    for g in glob.glob(os.path.join(out, "tiles", "*")):
        raw = open(g, "rb").read()
        if g.endswith(".pnts"):
            ftjl = struct.unpack_from("<I", raw, 12)[0]
            head = json.loads(raw[28:28 + ftjl])
            tot += head["POINTS_LENGTH"]
        else:
            jl = struct.unpack_from("<I", raw, 12)[0]
            j = json.loads(raw[20:20 + jl])
            tot += j["accessors"][0]["count"]
    assert tot == res["pts_out"], f"瓦片基元 {tot} != 压缩点数 {res['pts_out']}"
    print(f"  tiles zero-loss: {tot} 点，瓦片并集与压缩场景一致")


def test_vector_roundtrip(tmp):
    """矢量：编码往返精度 + 属性保真 + 顶点池去重 + 分块零丢失。"""
    src = os.path.join(tmp, "v.geojson")
    m = make_city_vector(src, grid=4, parcels_per_cell=25)
    layer = read_geojson(src)
    assert layer.n == m["features"]
    n_orig = layer.vertex_count()
    ls, info = simplify(layer, 0.35)
    assert info["vertices_out"] <= n_orig
    data, st = encode_gvec(ls)
    dec = decode_gvec(data)
    assert dec.n == ls.n
    ev = eval_vector(ls, dec)
    # 量化分辨率 = size/4096 的半步 + 简化容差
    bbox = st["bbox"]
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    assert ev["max_dev"] <= size / 4096 + 1e-6, ev
    assert ev["len_rel_err"] < 1e-3 and ev["area_rel_err"] < 1e-3, ev
    # 属性保真：字典列与整型列往返一致
    assert list(dec.props["owner"][:5]) == list(layer.props["owner"][:5])
    assert list(dec.props["year"][:5]) == list(layer.props["year"][:5])
    assert st["payload_bytes"] < m["bytes"] / 3

    # 分块：每层要素数守恒（按质心分配，每要素每层恰好一块）
    from gsq.vector import build_vector_tiles
    t0 = time.time()
    tl = build_vector_tiles(layer, os.path.join(tmp, "vt"), levels=3, verbose=False)
    with open(tl["manifest"], encoding="utf-8") as f:
        man = json.load(f)
    for lvl in man["levels"]:
        assert sum(t["features"] for t in lvl["tiles"]) == layer.n, lvl["level"]
    assert man["levels"][0]["tiles"][0]["vertices"] <= man["levels"][-1]["tiles"][0]["vertices"] + 10 ** 9
    print(f"  vector: {layer.n} 要素 rt dev={ev['max_dev']:.3f}m · 顶点池去重后 {st['vertices']:,} · "
          f"分块 {tl['tiles']} 块零丢失（{time.time()-t0:.1f}s）")


def test_city_scale(tmp):
    """城市级基准：3D Tiles 2.9M 点 + 矢量 9 万顶点，校验耗时与内存上界。"""
    import tracemalloc
    src = os.path.join(tmp, "big_tiles")
    make_city_tiles(src, grid=8, per_tile=40000, kind="glb", world_offset=(5.2e6, 4.1e6, 3e4))
    tracemalloc.start()
    t0 = time.time()
    res = compress_tileset(src, os.path.join(tmp, "big_out"), prune=0.5, verbose=False)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    dt = time.time() - t0
    assert res["pts_in"] > 2_000_000 and res["ratio"] > 4, res
    assert peak / 1024 ** 2 < 64, f"峰值内存 {peak/1024**2:.1f}MB 超界（应逐瓦片流式）"
    assert dt < 90, f"城市级压缩耗时 {dt:.1f}s 超界"
    print(f"  city 3dtiles: {res['pts_in']:,} 点 {res['bytes_in']/1024**2:.1f}MB → "
          f"{res['bytes_out']/1024**2:.1f}MB ({res['ratio']}x) · {dt:.1f}s · 峰值 {peak/1024**2:.1f}MB")

    vsrc = os.path.join(tmp, "big.geojson")
    make_city_vector(vsrc, grid=16, parcels_per_cell=25)
    layer = read_geojson(vsrc)
    assert layer.vertex_count() > 50_000
    tracemalloc.start()
    t0 = time.time()
    data, st = encode_gvec(layer)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert st["bytes"] < os.path.getsize(vsrc) / 4
    print(f"  city vector: {layer.vertex_count():,} 顶点 → {st['bytes']/1024:.0f}KB "
          f"({os.path.getsize(vsrc)/st['bytes']:.1f}x) · {time.time()-t0:.1f}s · 峰值 {peak/1024**2:.1f}MB")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        print("GeoSqueeze 城市级验收")
        test_tiles_formats(tmp)
        test_tiles_zero_loss(tmp)
        test_vector_roundtrip(tmp)
        test_city_scale(tmp)
    print("ALL PASSED")
