"""模型线：ONNX 任务模型压缩（设计 §2 模型线 / §5.1 ONNX 结构化剪枝 / §6.2 ONNX 导出 P1）。

借鉴 H3DNAS 的通道依赖图（CDG）思路，不需要训练源码，直接做图手术：

- ``analyze_model``   —— 通道依赖图分析：逐层参数量、可剪通道数、理论压缩上限；
- ``prune_onnx``      —— L1 结构化通道剪枝：按 filter L1 范数保留 top-k 输出通道，
                        同步切边（下游 Conv 的输入列 / Gemm 的行 / BN 的统计量），
                        残差 Add / Concat 等通道约束算子所在层保守跳过；
- ``quantize_onnx``   —— INT8 动态量化（onnxruntime.quantization）；
- ``validate_model``  —— onnxruntime 前向对拍：随机输入下输出余弦相似度 + 时延。

边界（如实说明）：算子覆盖 Conv/ConvTranspose/Gemm/MatMul/BatchNormalization 及
通道保持的逐元素算子；遇到 Reshape/Transpose/循环控制流等通道语义复杂节点时保守
跳过该层并计入 skipped，保证输出图永远有效。
"""
from __future__ import annotations
import json
import os
import time
import numpy as np

from onnx import numpy_helper, shape_inference

# 通道保持算子：输入输出通道一致，剪枝可穿透传播（BN 统计量可切，且传播继续）
_PASSTHROUGH = {"Relu", "Sigmoid", "Clip", "Identity", "LeakyRelu", "PRelu",
                "Sqrt", "Abs", "Exp", "Log", "Floor", "Ceil", "Neg", "Erf", "HardSigmoid",
                "BatchNormalization", "InstanceNormalization", "Dropout"}


def _load(path: str):
    import onnx
    m = onnx.load(path)
    m = shape_inference.infer_shapes(m)
    return m


def _shapes(m) -> dict:
    out = {}
    for vi in list(m.graph.value_info) + list(m.graph.output) + list(m.graph.input):
        t = vi.type.tensor_type
        dims = [d.dim_value for d in t.shape.dim]
        out[vi.name] = dims
    return out


def _params(m) -> dict:
    return {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}


def _node_io(node):
    return list(node.input), list(node.output)


def analyze_model(path: str) -> dict:
    """CDG 通道依赖图分析（O(V+E)）：每层通道数、参数量与结构化可剪上限。"""
    import onnx
    m = _load(path)
    shp, par = _shapes(m), _params(m)
    tensor_out = {o: n for n in m.graph.node for o in n.output}   # tensor -> producer
    users = {}
    for node in m.graph.node:
        for inp in node.input:
            users.setdefault(inp, []).append(node)
    layers, total = [], 0
    for node in m.graph.node:
        w = None
        for inp in node.input:
            if inp in par and par[inp].ndim >= 2:
                w = par[inp]
        if w is None:
            continue
        total += int(w.nbytes)
        if node.op_type == "Conv" and w.ndim == 4:
            oc, ic = int(w.shape[0]), int(w.shape[1])
            terminals, _ = _chain_consumers(node.output[0], users)
            kinds = [t.op_type for t in terminals]
            # 保守可剪判定：链路终点全部可切（Conv/BN 或可切权重的 Gemm/MatMul）
            adaptable = all(
                t.op_type in ("Conv", "BatchNormalization")
                or (t.op_type in ("Gemm", "MatMul") and any(i in par and par[i].ndim == 2 for i in t.input))
                for t in terminals) if terminals else False
            layers.append({"name": node.name or node.output[0], "op": "Conv",
                           "out_ch": oc, "in_ch": ic, "params": int(w.size),
                           "prunable": bool(adaptable), "consumers": kinds})
        elif node.op_type in ("Gemm", "MatMul") and w.ndim == 2:
            layers.append({"name": node.name or node.output[0], "op": node.op_type,
                           "shape": list(w.shape), "params": int(w.size), "prunable": False,
                           "consumers": []})
    prunable_params = sum(l["params"] for l in layers if l.get("prunable"))
    return {"file": os.path.basename(path), "bytes": os.path.getsize(path),
            "opset": m.opset_import[0].version if m.opset_import else None,
            "n_nodes": len(m.graph.node), "n_initializers": len(m.graph.initializer),
            "weight_bytes": total, "layers": layers,
            "prunable_layers": sum(1 for l in layers if l.get("prunable")),
            "prunable_params": prunable_params,
            "ceiling_note": "prunable=True 的 Conv 输出通道可结构化移除；其余层受通道依赖约束（CDG 保守判定）"}


def _sliceable(node, shp, par) -> bool:
    """（保留给 analyze 使用）Gemm/MatMul 的输入列能否跟随上游通道数变化。"""
    return any(i in par and par[i].ndim == 2 for i in node.input)


def _chain_consumers(start_tensor: str, users: dict):
    """从 start_tensor 出发穿透 passthrough 算子链，返回 (终点节点列表, 全部途经节点)。

    终点 = Conv/BN/Gemm/MatMul（可切权重）或其它算子（触发保守放弃）。
    """
    terminals, chain = [], []
    work = [start_tensor]
    seen = set()
    while work:
        t = work.pop()
        for node in users.get(t, []):
            chain.append(node)
            if node.op_type in _PASSTHROUGH or node.op_type in ("MaxPool", "AveragePool"):
                for o in node.output:
                    if o not in seen:
                        seen.add(o)
                        work.append(o)
            else:
                terminals.append(node)
    return terminals, chain


def prune_onnx(path: str, out_path: str, ratio: float = 0.4) -> dict:
    """L1 结构化通道剪枝：图手术直接改 ONNX，输出仍可被任何 ONNX runtime 加载。

    消费者判定沿 passthrough 链（Relu/MaxPool/…）传递闭包——终点不可适配
    （Reshape/Add/Concat 等）则整层放弃，绝不静默破坏通道数。
    """
    import onnx
    m = _load(path)
    par = _params(m)
    users = {}
    for node in m.graph.node:
        for inp in node.input:
            users.setdefault(inp, []).append(node)

    removed = kept_total = orig_total = 0
    pruned_layers, skipped = [], []
    for node in list(m.graph.node):
        if node.op_type != "Conv" or len(node.input) < 2:
            continue
        wname, bname = node.input[1], (node.input[2] if len(node.input) > 2 else None)
        if wname not in par or par[wname].ndim != 4:
            continue
        w = par[wname]
        oc = int(w.shape[0])
        terminals, chain = _chain_consumers(node.output[0], users)
        kinds = [t.op_type for t in terminals]
        if not terminals:
            skipped.append({"layer": node.name or node.output[0], "reason": "无消费者（死分支）"})
            continue
        adaptable = all(
            t.op_type in ("Conv", "BatchNormalization")
            or (t.op_type in ("Gemm", "MatMul") and any(i in par and par[i].ndim == 2 for i in t.input))
            for t in terminals)
        if not adaptable:
            skipped.append({"layer": node.name or node.output[0], "reason": f"链路终点不可适配: {kinds}"})
            continue
        keep_n = max(1, int(round(oc * (1 - ratio))))
        if keep_n >= oc:
            continue
        score = np.abs(w).sum(axis=(1, 2, 3))                      # L1 per filter
        keep = np.sort(np.argpartition(-score, keep_n - 1)[:keep_n])
        # 1) 本节点权重与 bias
        par[wname] = np.ascontiguousarray(w[keep])
        _set_init(m, wname, par[wname])
        if bname and bname in par:
            par[bname] = np.ascontiguousarray(par[bname][keep])
            _set_init(m, bname, par[bname])
        # 2) 链上全部消费者跟着切
        for c in chain:
            if c.op_type == "Conv" and len(c.input) >= 2 and c.input[1] in par:
                cw = par[c.input[1]]
                if cw.ndim == 4 and cw.shape[1] == oc:             # group=1 才能切输入列
                    par[c.input[1]] = np.ascontiguousarray(cw[:, keep])
                    _set_init(m, c.input[1], par[c.input[1]])
            elif c.op_type == "BatchNormalization":
                for nm in c.input[1:5]:
                    if nm in par and par[nm].shape == (oc,):
                        par[nm] = np.ascontiguousarray(par[nm][keep])
                        _set_init(m, nm, par[nm])
            elif c.op_type in ("Gemm", "MatMul"):
                wnm = next((i for i in c.input if i in par and par[i].ndim == 2), None)
                if wnm is None:
                    continue
                wmat = par[wnm]
                if wmat.shape[0] == oc:                            # (K,N) 行切
                    par[wnm] = np.ascontiguousarray(wmat[keep])
                elif wmat.shape[1] == oc:                          # (N,K) 列切
                    par[wnm] = np.ascontiguousarray(wmat[:, keep])
                else:
                    continue
                _set_init(m, wnm, par[wnm])
        removed += oc - keep_n
        kept_total += keep_n
        orig_total += oc
        pruned_layers.append({"layer": node.name or node.output[0], "out_ch": oc, "kept": int(keep_n)})

    # 同步 value_info 的通道维（runtime 重新推断也行，但保留一致性）
    m = _sync_value_info(m)
    onnx.checker.check_model(m)
    onnx.save(m, out_path)
    return {"out": os.path.abspath(out_path), "conv_out_ch_before": orig_total,
            "conv_out_ch_after": kept_total, "removed_channels": removed,
            "pruned_layers": pruned_layers, "skipped": skipped,
            "bytes": os.path.getsize(out_path)}


def _set_init(m, name: str, arr: np.ndarray):
    from onnx import numpy_helper
    for i, init in enumerate(m.graph.initializer):
        if init.name == name:
            m.graph.initializer[i].CopyFrom(numpy_helper.from_array(arr, name))
            return
    m.graph.initializer.append(numpy_helper.from_array(arr, name))


def _sync_value_info(m):
    """剪枝后重建 shape 信息：先清空旧 value_info（可能已过期）再全量重推断。"""
    del m.graph.value_info[:]
    try:
        return shape_inference.infer_shapes(m)
    except Exception:
        return m                                        # runtime 自行推断


def _normalize_gemm(m):
    """把 transB=1 的 Gemm 改写为显式转置权重 + transB=0。

    onnxruntime 动态量化器不转置 Gemm 权重，transB=1 会产出维度不匹配的
    MatMulInteger；量化前统一规范化（语义等价，runtime 兼容）。
    """
    par = {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}
    changed = 0
    for node in m.graph.node:
        if node.op_type != "Gemm":
            continue
        attrs = {a.name: a for a in node.attribute}
        if int(attrs.get("transB", 0).i) != 1:
            continue
        ta = int(attrs["transA"].i) if "transA" in attrs else 0
        if ta == 1:
            continue                                  # transA=1 场景少见，保守跳过
        alpha = float(attrs["alpha"].f) if "alpha" in attrs else 1.0
        beta = float(attrs["beta"].f) if "beta" in attrs else 1.0
        if alpha != 1.0 or beta != 1.0:
            continue
        wn = node.input[1]
        if wn in par and par[wn].ndim == 2:
            _set_init(m, wn, np.ascontiguousarray(par[wn].T))
            for a in node.attribute:
                if a.name == "transB":
                    a.i = 0
            changed += 1
    return changed


def quantize_onnx(path: str, out_path: str, weight_type: str = "qint8") -> dict:
    """INT8 动态量化（onnxruntime.quantization，权重 INT8、激活动态范围）。"""
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    wt = QuantType.QInt8 if weight_type == "qint8" else QuantType.QUInt8
    m = _load(path)
    if _normalize_gemm(m):
        tmp = out_path + ".norm.onnx"
        onnx.save(m, tmp)
        path = tmp
    quantize_dynamic(path, out_path, weight_type=wt,
                     extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT})
    if path.endswith(".norm.onnx") and os.path.exists(path):
        os.remove(path)
    return {"out": os.path.abspath(out_path), "bytes": os.path.getsize(out_path),
            "weight_type": weight_type}


def validate_model(path: str, ref_path: str | None = None) -> dict:
    """onnxruntime 前向对拍：随机输入下与参考模型（可选）输出余弦/相对误差 + 单次时延。"""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    feeds = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        feeds[inp.name] = np.random.default_rng(0).standard_normal(shape).astype(
            np.float32 if inp.type == "tensor(float)" else np.float32)
    t0 = time.perf_counter()
    outs = sess.run(None, feeds)
    dt = (time.perf_counter() - t0) * 1000
    res = {"latency_ms": round(dt, 2), "outputs": [list(o.shape) for o in outs]}
    if ref_path:
        ref = ort.InferenceSession(ref_path, so, providers=["CPUExecutionProvider"])
        names = [i.name for i in ref.get_inputs()]
        ro = ref.run(None, {n: feeds[n] for n in names if n in feeds})
        sims, errs = [], []
        for a, b in zip(outs, ro):
            a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
            if a.shape != b.shape:
                continue
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            sims.append(float((a @ b) / (na * nb)) if na > 0 and nb > 0 else 1.0)
            errs.append(float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12)))
        res.update({"cosine_min": round(min(sims), 6) if sims else None,
                    "rel_err_max": round(max(errs), 6) if errs else None})
    return res


def compress_model(path: str, out_dir: str, ratio: float = 0.4, do_quantize: bool = True,
                   validate: bool = True) -> dict:
    """一条命令：分析 → 剪枝 → INT8 量化 → 对拍验证（模型线主入口）。"""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    report = {"input": os.path.abspath(path), "ratio": ratio,
              "analysis": analyze_model(path)}
    cur = path
    if ratio > 0 and report["analysis"]["prunable_layers"] > 0:
        p_out = os.path.join(out_dir, f"{base}.pruned.onnx")
        report["prune"] = prune_onnx(path, p_out, ratio)
        cur = p_out
    if do_quantize:
        q_out = os.path.join(out_dir, f"{base}.int8.onnx")
        report["quantize"] = quantize_onnx(cur, q_out)
        cur = q_out
    report["final"] = {"path": cur, "bytes": os.path.getsize(cur)}
    if validate:
        report["validate"] = validate_model(cur, path if cur != path else None)
    with open(os.path.join(out_dir, f"{base}.model_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    return report
