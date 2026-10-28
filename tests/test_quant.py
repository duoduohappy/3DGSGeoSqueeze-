"""统一量化算子验收：affine 与定点码型的误差上界、场景量化口径、网格几何共用同一码型。

运行：python tests/test_quant.py
"""
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gsq.ir import make_sample  # noqa: E402
from gsq.mesh import compress_mesh_tile, read_mesh_content, write_b3dm  # noqa: E402
from gsq.quant import (affine_decode, affine_encode, affine_roundtrip, max_abs_error,  # noqa: E402
                       normalized_decode, normalized_encode, normalized_step,
                       practical_bound, roundtrip_bound)
from gsq.stages import quantize  # noqa: E402

EPS = 1e-9


def test_affine_bound():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 3, (2000, 3)).astype(np.float32)
    for bits in (8, 12, 16):
        q, lo, step = affine_encode(x, bits)
        assert q.dtype == (np.uint8 if bits <= 8 else np.uint16)
        assert q.min() >= 0 and q.max() <= 2 ** bits - 1
        rec, step2, err, bound = affine_roundtrip(x, bits)
        assert step == step2 and abs(bound - step / 2) < EPS
        # step/2 的理论界 + 解码落 float32 的 2 ULP（见 quant.practical_bound）
        assert err <= practical_bound(step, float(np.abs(x).max()), float(np.ptp(x))), (bits, err)
    # 常量输入：step=0，解码还原原值
    q, lo, st = affine_encode(np.full(5, 2.5, np.float32), 8)
    assert st == 0.0 and not q.any()
    assert np.allclose(affine_decode(q, lo, st), 2.5)
    # 16bit 比 8bit 更准（同分布下误差应降 2 个数量级）
    e8 = affine_roundtrip(x, 8)[2]
    e16 = affine_roundtrip(x, 16)[2]
    assert e16 < e8 / 100
    print(f"  affine: 8bit 误差 {e8:.5f} · 16bit {e16:.6f} · 均 ≤ step/2")


def test_normalized_bound():
    rng = np.random.default_rng(1)
    pos = rng.normal(0, 100, (500, 3))
    lo = pos.min(0)
    ext = float(max(pos.max(0) - lo))
    q = normalized_encode(pos, lo, ext)
    assert q.dtype == np.uint16 and q.min() >= 0 and q.max() <= 65535
    rec = normalized_decode(q, lo, ext)
    step = normalized_step(ext)
    err = max_abs_error(pos, rec)
    assert err <= roundtrip_bound(step) + EPS, (err, step)
    # 越界值裁剪到量程内（浏览端 normalized 语义）
    out = normalized_encode(np.array([[lo[0] - ext, lo[1] + 2 * ext, lo[2]]]), lo, ext)
    assert int(out[0, 0]) == 0 and int(out[0, 1]) == 65535
    # UV 口径：lo=0 / ext=1 → 格距 1/65535
    uv = rng.random((100, 2))
    qv = normalized_encode(uv, 0.0, 1.0)
    assert max_abs_error(uv, normalized_decode(qv, 0.0, 1.0)) <= 0.5 / 65535 + EPS
    print(f"  normalized: 位置格距 {step*1000:.3f}mm · 误差 {err*1000:.3f}mm · UV ≤ 1/65535")


def test_scene_quantize_reports_error():
    s = make_sample(200, 8)
    q, info = quantize(s, pos_bits=16)
    assert "pos_max_err" in info and 0 < info["pos_max_err"] < 1e-2
    pos_t = next(t for t in q.codec["tensors"] if t["name"] == "pos")
    assert info["pos_max_err"] <= practical_bound(pos_t["step"], float(np.abs(s.pos).max()),
                                                 float(np.ptp(s.pos)))
    assert set(info["max_err"]) >= {"pos", "opacity", "scale", "rot"}
    # 8bit 位置误差应明显大于 16bit（同一个码型不同位宽）
    q8, info8 = quantize(s, pos_bits=8)
    assert info8["pos_max_err"] > info["pos_max_err"] * 50
    print(f"  scene quantize: 16bit {info['pos_max_err']*1000:.3f}mm · "
          f"8bit {info8['pos_max_err']*1000:.2f}mm · 口径一致")


def _tiny_png() -> bytes:
    from PIL import Image
    bio = io.BytesIO()
    Image.new("RGB", (8, 8), (120, 140, 160)).save(bio, "PNG")
    return bio.getvalue()


def test_mesh_shares_codec(tmp):
    rng = np.random.default_rng(2)
    n = 240
    pos = rng.normal(0, 5, (n, 3))
    idx = np.arange(n, dtype=np.int64)
    uv = rng.random((n, 2)).astype(np.float32)
    mesh = {"prims": [{"pos": pos, "idx": idx, "uv": uv, "mat": 0}],
            "images": [(_tiny_png(), "image/png")], "mat_tex": [0]}
    glb, st = compress_mesh_tile(mesh, prune=0.0, tex_max_dim=64, tex_quality=80)
    assert st["geom_quant_bits"] == 16
    assert b"KHR_mesh_quantization" in glb
    # 报告的最大误差必须不超过格距一半（与场景通道同一判定）
    assert st["geom_quant_max_err"] <= roundtrip_bound(st["geom_quant_step"]) + EPS
    assert st["geom_quant_max_err"] > 0
    # 写盘回读：世界坐标复原误差应等于报告值（同一码型，不是估算）
    p = os.path.join(tmp, "t.b3dm")
    write_b3dm(p, glb)
    back = read_mesh_content(p)
    err = max_abs_error(pos, back["prims"][0]["pos"])
    assert err <= st["geom_quant_max_err"] + 1e-6, (err, st["geom_quant_max_err"])
    print(f"  mesh: 步长 {st['geom_quant_step']*1000:.3f}mm · 实测最大误差 "
          f"{st['geom_quant_max_err']*1000:.3f}mm · 回读 {err*1000:.3f}mm")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        print("GeoSqueeze 统一量化算子测试")
        test_affine_bound()
        test_normalized_bound()
        test_scene_quantize_reports_error()
        test_mesh_shares_codec(tmp)
    print("ALL PASSED")
