"""设计对齐测试：SPZ/splat 通道 · 模型线 ONNX · merge/vectree · checkpoint · YAML/CSV。

运行：python tests/test_design.py
"""
import json
import os
import struct
import sys
import tempfile
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gsq.ir import GeoScene, make_sample, read_csv  # noqa: E402
from gsq.spz import read_spz, read_splat, write_spz  # noqa: E402
from gsq.stages import prune, quantize  # noqa: E402
from gsq.export import read_gsq, write_gsq  # noqa: E402
from gsq.analyze import plan  # noqa: E402
from gsq.pipeline import rerun_stage, run_pipeline  # noqa: E402


def test_spz_roundtrip(tmp):
    s = make_sample(30, 8)
    p = os.path.join(tmp, "t.spz")
    info = write_spz(p, s)
    assert info["n"] == s.n and info["sh_degree"] == 3
    s2, h = read_spz(p)
    assert h["version"] == 2 and h["n"] == s.n
    # 24-bit 定点 + 各属性量化精度上界（scales 1/16 log 步长，SH 桶中心量化）
    assert np.abs(s2.pos - s.pos).max() < 0.001
    assert np.abs(s2.colors() - s.colors()).mean() < 0.01
    assert np.abs(s2.opacity - s.opacity).max() < 0.005
    assert np.abs(np.log(s2.scale) - np.log(s.scale)).max() < 0.04
    assert np.abs(s2.rot - s.rot).max() < 0.01
    # ECEF 大坐标：自动居中 + gzip 注释回读
    big = GeoScene(s.pos + np.array([5.2e6, 4.1e6, 3e4], np.float32), s.sh,
                   s.opacity, s.scale, s.rot)
    p2 = os.path.join(tmp, "big.spz")
    write_spz(p2, big)
    s3, _ = read_spz(p2)
    assert np.abs(s3.pos - big.pos).max() < 0.001
    # v4 / 坏文件明确报错
    open(p2, "wb").write(b"NGSPgarbage")
    try:
        read_spz(p2)
        raise AssertionError("v4 应明确报错")
    except ValueError:
        pass
    print(f"  spz: {info['bytes']}B n={s.n} roundtrip ok · ECEF ok · v4 明确报错")


def test_splat(tmp):
    s = make_sample(20, 6)
    buf = bytearray()
    for i in range(s.n):
        buf += s.pos[i].astype("<f4").tobytes() + s.scale[i].astype("<f4").tobytes()
        c = (s.colors()[i] * 255).astype(np.uint8)
        buf += bytes([int(c[0]), int(c[1]), int(c[2]), int(s.opacity[i] * 255)])
        buf += ((s.rot[i] * 128 + 128).clip(0, 255).astype(np.uint8)).tobytes()
    p = os.path.join(tmp, "t.splat")
    open(p, "wb").write(bytes(buf))
    s2, h = read_splat(p)
    assert h["n"] == s.n
    assert np.abs(s2.pos - s.pos).max() < 1e-5
    assert np.abs(s2.colors() - s.colors()).mean() < 0.005
    print(f"  splat: n={s.n} roundtrip ok")


def test_csv(tmp):
    p = os.path.join(tmp, "a.csv")
    open(p, "w").write("x,y,z,r,g,b\n0,0,0,255,0,0\n1,2,3,0,255,0\n")
    s = read_csv(p)
    assert s.n == 2 and s.kind == "point"
    assert np.abs(s.colors()[0] - [1, 0, 0]).max() < 0.01
    p2 = os.path.join(tmp, "b.csv")
    open(p2, "w").write("1,2,3\n4,5,6\n")          # 无表头 XYZ
    s2 = read_csv(p2)
    assert s2.n == 2 and np.abs(s2.pos[1] - [4, 5, 6]).max() < 1e-6
    print("  csv: 表头/无表头 ok")


def test_merge_and_vectree(tmp):
    s = make_sample(40, 12)
    m, info = prune(s, ratio=0.3, method="merge")
    assert info["method"] == "merge" and m.n < s.n and info["absorbed"] >= 0
    q, qi = quantize(m, method="vectree")
    assert qi["sh_mode"] == "vectree" and qi["rvq_stages"] >= 2
    p = os.path.join(tmp, "t.gsq")
    write_gsq(p, q)
    s2, h = read_gsq(p)
    assert h["sh_mode"] == "vectree"
    assert np.abs(s2.sh - q.sh).max() < 1e-3       # 残差级联重建逐位一致
    print(f"  merge: 吸收 {info['absorbed']} · vectree: {qi['rvq_stages']} 级 gsq 往返 ok")


def test_desktop_vectree_plan():
    s = make_sample(30, 10)
    rec = plan(s, "desktop", 5.0, 30.0)
    assert rec["encoding"]["sh"] == "vectree"       # 设计决策树：desktop → VecTree
    assert rec["stages"][-1].get("method") == "vectree"
    print("  desktop 配方 → vectree ok")


def test_checkpoint_resume_and_rerun(tmp):
    s = make_sample(30, 8)
    out = os.path.join(tmp, "out")
    res = run_pipeline(s, out, "web", 50.0, 20.0, None, 3, verbose=False, n_views=2)
    assert res["passed"]
    ckpt = os.path.join(out, ".gsq", "ckpt")
    assert os.path.isdir(ckpt) and len(os.listdir(ckpt)) >= 4   # 每阶段一个 ckpt
    # resume：从最后 checkpoint 续跑，同样收敛通过
    res2 = run_pipeline(s, out, "web", 50.0, 20.0, None, 3, verbose=False, n_views=2, resume=True)
    assert res2["passed"] and res2["metrics"]["n_comp"] == res["metrics"]["n_comp"]
    # 单阶段重跑：quantize 从其输入状态独立重跑
    r = rerun_stage(out, s, "quantize")
    assert r["info"]["stage"] == "quantize"
    # 错误输入：不存在的阶段
    try:
        rerun_stage(out, s, "nope")
        raise AssertionError("应报错")
    except ValueError:
        pass
    print(f"  checkpoint: {len(os.listdir(ckpt))} 个 · resume ok · rerun-stage ok")


def test_yaml_recipe(tmp):
    p = os.path.join(tmp, "r.yaml")
    open(p, "w", encoding="utf-8").write(
        "target:\n  platform: web\n  quality_floor: { psnr: 20 }\n"
        "stages:\n  - prune: { method: render, ratio: 0.5 }\n"
        "  - quantize: { pos_bits: 16, method: vectree }\n"
        "export:\n  lod_levels: [15, 18]\n")
    from gsq.cli import _load_recipe
    rec = _load_recipe(p)
    assert rec["stages"][0]["stage"] == "prune"      # 设计方言单键写法已适配
    assert rec["target"]["psnr_floor"] == 20
    assert rec["stages"][1]["method"] == "vectree"
    print("  yaml recipe: 方言适配 ok")


def _make_conv_net(path):
    from onnx import TensorProto, helper, numpy_helper, shape_inference
    import onnx
    inits = []

    def init(name, arr):
        inits.append(numpy_helper.from_array(arr, name))
        return name
    rng = np.random.default_rng(0)
    w1 = rng.normal(0, 0.1, (16, 3, 3, 3)).astype(np.float32)
    w2 = rng.normal(0, 0.1, (32, 16, 3, 3)).astype(np.float32)
    w3 = rng.normal(0, 0.1, (10, 512)).astype(np.float32)
    b1 = np.zeros(16, np.float32); b2 = np.zeros(32, np.float32); b3 = np.zeros(10, np.float32)
    bn = [np.ones(16, np.float32), np.zeros(16, np.float32),
          np.zeros(16, np.float32), np.ones(16, np.float32)]
    nodes = [
        helper.make_node("Conv", ["x", init("w1", w1), init("b1", b1)], ["c1"],
                         name="conv1", pads=[1, 1, 1, 1]),
        helper.make_node("BatchNormalization",
                         ["c1"] + [init(f"bnp{i}", v) for i, v in enumerate(bn)], ["n1"]),
        helper.make_node("Relu", ["n1"], ["r1"]),
        helper.make_node("MaxPool", ["r1"], ["p1"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Conv", ["p1", init("w2", w2), init("b2", b2)], ["c2"],
                         name="conv2", pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c2"], ["r2"]),
        helper.make_node("MaxPool", ["r2"], ["p2"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Reshape", ["p2", init("shp", np.array([1, 512], np.int64))], ["flat"]),
        helper.make_node("Gemm", ["flat", init("w3", w3), init("b3", b3)], ["y"],
                         name="fc1", transB=1),
    ]
    graph = helper.make_graph(
        nodes, "net",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 16, 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 10])], inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(shape_inference.infer_shapes(m), path)


def test_model_line(tmp):
    from gsq.model import analyze_model, compress_model
    p = os.path.join(tmp, "net.onnx")
    _make_conv_net(p)
    a = analyze_model(p)
    assert a["prunable_layers"] >= 1
    prunable = {l["name"] for l in a["layers"] if l["prunable"]}
    assert "conv1" in prunable and "conv2" not in prunable    # Reshape 终点 → 保守跳过
    r = compress_model(p, os.path.join(tmp, "out"), ratio=0.4)
    assert r["prune"]["pruned_layers"][0]["out_ch"] == 16
    assert r["final"]["bytes"] < a["bytes"]                    # INT8 后更小
    v = r.get("validate", {})
    assert v.get("cosine_min", 0) > 0.8                        # 前向对拍一致
    print(f"  model line: 剪枝 {r['prune']['pruned_layers']} · "
          f"{a['bytes']}→{r['final']['bytes']}B · cosine {v.get('cosine_min')}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        print("GeoSqueeze 设计对齐测试")
        test_spz_roundtrip(tmp)
        test_splat(tmp)
        test_csv(tmp)
        test_merge_and_vectree(tmp)
        test_desktop_vectree_plan()
        test_checkpoint_resume_and_rerun(tmp)
        test_yaml_recipe(tmp)
        test_model_line(tmp)
    print("ALL PASSED")
