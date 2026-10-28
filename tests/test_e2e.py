"""端到端自测：python tests/test_e2e.py"""
import glob
import json
import os
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gsq import (GeoScene, evaluate, make_sample, plan, profile, read_gsq, read_ply,
                 run_pipeline, significance, write_gsq, write_ply)
from gsq.ir import las_header, read_las, write_las
from gsq.analyze import relax
from gsq.export import build_tiles, validate_tileset
from gsq.stages import distill, lowrank, prune, quantize


def test_sample_io(tmp):
    p = os.path.join(tmp, "s.ply")
    s0 = make_sample(60, 12)
    write_ply(p, s0)
    s1 = read_ply(p)
    assert s1.n == s0.n and s1.sh_degree == 3
    assert np.allclose(s1.pos, s0.pos, atol=1e-5)
    assert np.abs(s1.colors() - s0.colors()).mean() < 2e-3
    assert np.abs(s1.opacity - s0.opacity).max() < 2e-3
    print(f"  ply roundtrip: n={s1.n} deg={s1.sh_degree}")


def test_analysis():
    s = make_sample(60, 12)
    prof = profile(s)
    assert prof["n_primitives"] == s.n and prof["bottleneck"] == "sh"
    rec = plan(s, "mobile", 5.0, 30.0, prof)
    names = [x["stage"] for x in rec["stages"]]
    assert names == ["prune", "distill", "recover", "quantize"], names   # 代价模型：K=16 时低秩不划算
    assert rec["encoding"]["sh"] == "vq" and rec["encoding"]["cost_model"]["lowrank_cost"] > rec["encoding"]["cost_model"]["vq_cost"]
    assert rec["stages"][0]["method"] == "render"          # 小场景默认真渲染显著性
    assert rec["stages"][2]["stage"] == "recover" and rec["stages"][2]["iters"] >= 2
    # 高维 + 真实低秩结构的 SH：代价模型应选中低秩（随机数据无低秩结构，SVD 不省）
    from gsq.ir import GeoScene
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (4, 64 * 3)).astype(np.float32)          # 秩 4 结构
    coef = rng.normal(0, 0.3, (s.n, 4)).astype(np.float32) @ base + rng.normal(0, 0.01, (s.n, 64 * 3)).astype(np.float32)
    hi = GeoScene(s.pos, coef.reshape(s.n, 64, 3), s.opacity, s.scale, s.rot, "gaussian")
    rh = plan(hi, "mobile", 5.0, 30.0)
    assert rh["encoding"]["sh"] == "lowrank" and any(x["stage"] == "lowrank" for x in rh["stages"]), rh["encoding"]
    assert rh["encoding"]["cost_model"]["lowrank_cost"] < rh["encoding"]["cost_model"]["vq_cost"]
    sc = significance(s)
    assert sc.shape == (s.n,) and np.isfinite(sc).all() and sc.min() >= 0
    r2 = relax(rec, 1)
    assert r2["stages"][0]["ratio"] < rec["stages"][0]["ratio"]
    assert r2["stages"][1]["target_degree"] >= rec["stages"][1]["target_degree"]
    print(f"  analysis: bottleneck={prof['bottleneck']} prune={rec['stages'][0]['ratio']}")


def test_stages():
    s = make_sample(50, 10)
    p, i1 = prune(s, ratio=0.5)
    assert p.n == s.n // 2 and i1["kept"] == p.n
    d, i2 = distill(p, target_degree=1)
    assert d.sh.shape[1] == 4 and i2["degree_to"] == 1
    q, i3 = quantize(d, 16, 8, vq=True, codebook=64)
    assert q.n == d.n and i3["payload_bytes"] > 0
    lr, i4 = lowrank(q, energy=0.99)
    assert lr.n == q.n and i4["rank"] <= min(q.n, q.sh.shape[1] * 3)
    assert i4["factor_bytes"] < q.n * q.sh.shape[1] * 3 * 4        # 因子真实小于全量
    # 点云体素下采样
    from gsq.ir import GeoScene
    pc = GeoScene(s.pos, s.sh[:, :1, :], np.ones(s.n, np.float32),
                  np.full((s.n, 3), 0.05, np.float32), s.rot, "point")
    vp, vi = prune(pc, 0.7, method="voxel")
    assert abs(vp.n - pc.n * 0.3) <= max(16, pc.n * 0.05), vi
    assert vi["method"] == "voxel" and vi["kept"] == vp.n
    from gsq.stages import recover
    rc, ri = recover(s, d, iters=2, lr=1.0, n_views=2)
    assert ri["psnr_after"] >= ri["psnr_before"] - 0.05, ri
    # GSQ2 容器往返：三种 SH 模式均需逐位可逆（量化误差内）
    import os, tempfile
    from gsq.export import write_gsq, read_gsq, gsq_stats
    out = tempfile.mkdtemp()
    for tag, sc in (("vq", q), ("lowrank", lr), ("plain", d)):
        gp = os.path.join(out, f"t_{tag}.gsq")
        b1 = write_gsq(gp, sc)
        back, _ = read_gsq(gp)
        assert back.n == sc.n and gsq_stats(gp)["sh_mode"] == tag
        assert np.abs(back.pos - sc.pos).max() < 1e-3
        assert np.abs(back.sh - sc.sh).max() < 0.05
    print(f"  stages: prune={p.n} distill_k={d.sh.shape[1]} vq_bytes={i3['payload_bytes']} rank={i4['rank']} "
          f"recover={ri['delta_db']:+.2f}dB gsq_roundtrip=3modes-ok")


def test_pipeline_web(tmp):
    s = make_sample(80, 16)
    out = os.path.join(tmp, "web")
    res = run_pipeline(s, out, "web", 120, 24, None, 2, verbose=False, n_views=2)
    m = res["metrics"]
    assert res["passed"], m
    assert m["ratio"] >= 8 and m["psnr_render"] >= 20 and m["tiles"] >= 3
    assert 0 <= m["ssim_render"] <= 1 and len(m["render_views"]) == 2
    v = validate_tileset(os.path.join(out, "tileset.json"))
    assert v["valid"] and v["tiles"] == m["tiles"]
    assert os.path.isfile(os.path.join(out, "report.html"))
    assert os.path.isfile(os.path.join(out, "meta", "provenance.json"))
    back, h = read_gsq(os.path.join(out, "scene.gsq"))
    assert back.n == m["n_comp"] and h["k"] == back.sh.shape[1]
    print(f"  web: X{m['ratio']} PSNR={m['color_psnr']} IoU={m['voxel_iou']} tiles={m['tiles']} ({m['comp_mb']}MB)")


def test_pipeline_mobile(tmp):
    s = make_sample(60, 12)
    out = os.path.join(tmp, "mobile")
    res = run_pipeline(s, out, "mobile", 4.0, 20, None, 1, verbose=False, n_views=2)
    names = [i["stage"] for i in res["infos"]]
    assert "quantize" in names          # 低秩由代价模型决定：K=16 时不启用（README 有说明）
    assert os.path.isfile(os.path.join(out, "tileset.json"))
    assert validate_tileset(os.path.join(out, "tileset.json"))["valid"]
    print(f"  mobile: X{res['metrics']['ratio']} PSNR={res['metrics']['color_psnr']} stages={names}")


def test_pointcloud(tmp):
    s = make_sample(60, 12)
    pc = GeoScene(s.pos, np.zeros((s.n, 1, 3), np.float32), np.ones(s.n, np.float32),
                  np.full((s.n, 3), 0.05, np.float32), np.tile([1, 0, 0, 0], (s.n, 1)), "point",
                  {"source": "point-test"})
    p, _ = prune(pc, 0.4)
    q, info = quantize(p, 12, 8, vq=False)
    out = os.path.join(tmp, "pc")
    t = build_tiles(q, out, (15, 18), True, 0.1)
    assert t["tiles"] >= 2 and validate_tileset(t["tileset"])["valid"]
    m = evaluate(pc, q, t)
    assert m["keep_ratio"] < 0.7 and m["ratio"] > 3
    gsq = os.path.join(tmp, "pc.gsq")
    write_gsq(gsq, q, {"x": 1})
    back, _ = read_gsq(gsq)
    assert back.kind == "point" and np.allclose(back.pos, q.pos, atol=1e-6)
    print(f"  pointcloud: n={q.n} tiles={t['tiles']} X{m['ratio']}")



def _make_las14(path, pts, rgb, cls=2):
    """手工构造 LAS 1.4 / 点格式 7 文件（RGB 在偏移 30、分类在偏移 16），用于验收读取器字段表。"""
    import struct as st
    hsize, reclen, off = 375, 36, 375
    head = bytearray(hsize)
    head[0:4] = b"LASF"
    head[24], head[25] = 1, 4
    st.pack_into("<H", head, 94, hsize)
    st.pack_into("<I", head, 96, off)
    st.pack_into("<I", head, 100, 0)
    head[104] = 7
    st.pack_into("<H", head, 105, reclen)
    st.pack_into("<I", head, 107, 0)                      # 1.4 用 64 位计数
    st.pack_into("<Q", head, 247, len(pts))
    st.pack_into("<3d", head, 131, 0.001, 0.001, 0.001)
    st.pack_into("<3d", head, 155, 0.0, 0.0, 0.0)
    rec = np.zeros((len(pts), reclen), np.uint8)
    ijk = np.round(np.asarray(pts, np.float64) / 0.001).astype(np.int32)
    rec[:, 0:12] = np.ascontiguousarray(ijk).view(np.uint8).reshape(-1, 12)
    rec[:, 14] = 0x09
    rec[:, 16] = cls                                      # 格式 6-10：分类在 16
    rec[:, 30:36] = np.ascontiguousarray(np.asarray(rgb, np.uint16)).view(np.uint8).reshape(-1, 6)
    with open(path, "wb") as f:
        f.write(bytes(head))
        f.write(rec.tobytes())


def test_las(tmp):
    s = make_sample(28, 6)
    p = os.path.join(tmp, "rt.las")
    write_las(p, s)
    h = las_header(p)
    assert h["version"] == "1.2" and h["point_format"] == 2 and h["n_points"] == s.n
    back = read_las(p)
    assert back.kind == "point" and back.n == s.n
    assert np.abs(back.pos - s.pos).max() <= 0.0006          # 1mm 量化级别
    assert np.abs(back.colors() - s.colors()).max() <= 0.002  # 16 位 RGB 往返

    # LAS 1.4 / 格式 7：验证字段偏移表（RGB@30、分类@16、64 位点数）
    p14 = os.path.join(tmp, "v14.las")
    pts = np.array([[100.0, 200.0, 30.0], [101.0, 200.0, 30.5]], np.float64)
    _make_las14(p14, pts, [[65535, 0, 0], [0, 65535, 0]])
    h14 = las_header(p14)
    assert h14["version"] == "1.4" and h14["n_points"] == 2 and h14["point_format"] == 7
    s14 = read_las(p14)
    assert np.abs(s14.pos - pts).max() < 0.002 and s14.n == 2
    assert np.abs(s14.colors()[0] - np.array([1, 0, 0])).max() < 0.01
    assert np.abs(s14.colors()[1] - np.array([0, 1, 0])).max() < 0.01
    assert s14.meta["classification"] == {2: 2}

    # 无 RGB 格式（0-1）→ 高程着色，不出现全黑
    p0 = os.path.join(tmp, "nocolor.las")
    _make_las14(p0, pts, [[0, 0, 0], [0, 0, 0]])
    c0 = read_las(p0).colors()
    assert c0.min() >= 0 and c0.max() > 0.2, c0

    # LAZ：无解码工具时给出可操作的错误（而非崩溃）
    laz = os.path.join(tmp, "fake.laz")
    raw = bytearray(open(p, "rb").read())
    raw[104] |= 0x80
    open(laz, "wb").write(bytes(raw))
    try:
        read_las(laz)
        raise AssertionError("LAZ 未触发解码提示")
    except RuntimeError as e:
        assert "laszip" in str(e).lower() and "pdal" in str(e).lower()

    # 零丢失不变量：构造含小聚簇（<64 点分区）的场景，瓦片内容并集必须等于全集
    from gsq.export import build_tiles
    rng2 = np.random.default_rng(3)
    base = rng2.uniform(0, 100, (3000, 2))
    pts = np.column_stack([base, np.full(3000, 5.0)])
    pts = np.vstack([pts, rng2.uniform(40, 42, (50, 3))])   # 一个 50 点的小聚簇 → 触发 <=64 分支
    tiny = GeoScene(pts.astype(np.float32), np.zeros((3050, 1, 3), np.float32),
                    np.ones(3050, np.float32), np.full((3050, 3), 0.5, np.float32),
                    np.tile([1, 0, 0, 0], (3050, 1)), "point")
    tdir2 = os.path.join(tmp, "lossless")
    t2 = build_tiles(tiny, tdir2, (12, 15, 18), True, 2.0, "dc")
    tot2 = 0
    for g in glob.glob(os.path.join(tdir2, "tiles", "*.glb")):
        raw = open(g, "rb").read()
        j = json.loads(raw[20:20 + struct.unpack("<I", raw[12:16])[0]])
        tot2 += j["accessors"][0]["count"]
    assert tot2 == tiny.n, f"瓦片基元 {tot2} != 场景 {tiny.n}（存在丢失）"
    assert validate_tileset(t2["tileset"])["valid"]

    # LAS 走完整压缩管线
    out = os.path.join(tmp, "las_out")
    res = run_pipeline(read_las(p), out, "mobile", 6.0, 16, None, 3, verbose=False, n_views=2)
    m = res["metrics"]
    assert res["passed"] and m["tiles"] >= 1 and m["gsq_mb"] > 0, m.get("gate")
    assert res["rounds"] >= 2                      # 首轮过激进 → 质量门禁回流后达标
    print(f"  las: 1.2 rt err={np.abs(back.pos - s.pos).max():.4f} 1.4 n={s14.n} "
          f"pipeline X{m['ratio']} PSNR={m['psnr_render']} rounds={res['rounds']}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        print("GeoSqueeze e2e")
        test_sample_io(tmp)
        test_analysis()
        test_stages()
        test_pipeline_web(tmp)
        test_pipeline_mobile(tmp)
        test_pointcloud(tmp)
        test_las(tmp)
    print("ALL PASSED")
