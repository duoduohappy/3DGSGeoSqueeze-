"""流水线编排：分析 → 配方 → 阶段 DAG → 导出 → 评测 → 质量门禁回调。

阶段间自动落盘 checkpoint（.gsq/ckpt/，设计 §5）：断点可续跑（resume=True）、
单阶段可独立重跑（rerun_stage）。
"""
from __future__ import annotations
import hashlib
import json
import os
import time
import numpy as np
from .analyze import plan, profile, relax
from .export import build_tiles, evaluate, gate, write_gsq, write_report, read_gsq
from .viewer import write_viewer
from .ir import GeoScene
from .stages import recover, run_stage


def _fingerprint(s: GeoScene) -> str:
    h = hashlib.md5()
    h.update(np.ascontiguousarray(s.pos[: min(s.n, 4096)]).tobytes())
    h.update(np.ascontiguousarray(s.sh[:1]).tobytes())
    return h.hexdigest()[:16]


# ---------------- checkpoint（设计 §5：断点续跑 / 单阶段重跑） ----------------
def _ckpt_dir(out_dir: str) -> str:
    return os.path.join(out_dir, ".gsq", "ckpt")


def _ckpt_save(out_dir: str, rnd: int, idx: int, name: str, scene: GeoScene, rec: dict, fp: str):
    d = _ckpt_dir(out_dir)
    os.makedirs(d, exist_ok=True)
    tag = f"r{rnd}_s{idx:02d}_{name}"
    write_gsq(os.path.join(d, tag + ".gsq"), scene)
    with open(os.path.join(d, tag + ".json"), "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fp, "round": rnd, "stage_idx": idx, "stage": name,
                   "recipe": rec}, f, ensure_ascii=False)


def _ckpt_scan(out_dir: str, fp: str) -> list:
    """返回属于同一输入的 checkpoint 列表，按 (round, stage_idx) 升序。"""
    d = _ckpt_dir(out_dir)
    if not os.path.isdir(d):
        return []
    out = []
    for fn in os.listdir(d):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                man = json.load(f)
            if man.get("fingerprint") == fp:
                gpath = os.path.join(d, fn[:-5] + ".gsq")
                if os.path.isfile(gpath):
                    out.append((man["round"], man["stage_idx"], man["stage"], gpath, man))
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    return sorted(out)


def _ckpt_latest(out_dir: str, fp: str):
    cks = _ckpt_scan(out_dir, fp)
    return cks[-1] if cks else None


def rerun_stage(out_dir: str, baseline: GeoScene, stage: str) -> dict:
    """单阶段独立重跑：从该阶段最近 checkpoint 的输入状态重跑这一个阶段（设计 §5）。

    返回 {"before": 场景, "after": 场景, "info": 阶段指标, "round": 轮次}。
    """
    fp = _fingerprint(baseline)
    cks = _ckpt_scan(out_dir, fp)
    hits = [c for c in cks if c[2] == stage]
    if not hits:
        raise ValueError(f"checkpoint 中没有阶段 {stage} 的记录（先跑一次 compress）")
    rnd, idx, _, _, man = hits[-1]
    if idx == 0:
        src = baseline
    else:
        prev = next((c for c in cks if c[0] == rnd and c[1] == idx - 1), None)
        if prev is None:
            raise ValueError(f"缺少阶段 {stage} 的输入 checkpoint（r{rnd}_s{idx - 1:02d}）")
        src = read_gsq(prev[3])[0]
    rec = man["recipe"]
    cfg = next((c for c in rec["stages"] if c["stage"] == stage), None)
    if cfg is None:
        raise ValueError(f"checkpoint 配方中没有阶段 {stage}")
    if stage == "recover":
        after, info = recover(baseline, src, **{k: v for k, v in cfg.items() if k != "stage"})
    else:
        after, info = run_stage(stage, src, cfg)
    _ckpt_save(out_dir, rnd, idx, stage, after, rec, fp)
    return {"round": rnd, "stage_idx": idx, "input_n": src.n, "output_n": after.n,
            "info": info, "recipe": rec}


def run_pipeline(s: GeoScene, out_dir: str, platform="web", budget_mb=120.0, psnr_floor=28.0,
                 recipe=None, max_rounds=3, verbose=True, n_views=4, do_render=True,
                 resume=False, export_spz=False) -> dict:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    prof = profile(s)
    rec = recipe or plan(s, platform, budget_mb, psnr_floor, prof)
    fp = _fingerprint(s)

    # 断点续跑：恢复最近的 checkpoint（同输入指纹；显式配方不一致则从头跑）
    start_round, start_idx, cur = 0, -1, s
    if resume:
        latest = _ckpt_latest(out_dir, fp)
        if latest:
            rnd, idx, name, gpath, man = latest
            if recipe is not None and json.dumps(recipe, sort_keys=True) != json.dumps(man["recipe"], sort_keys=True):
                if verbose:
                    print(f"  resume: checkpoint 配方与显式配方不一致，从头重跑")
            else:
                start_round, start_idx = rnd, idx
                cur = read_gsq(gpath)[0]
                rec = man["recipe"]
                if verbose:
                    print(f"  resume: 从 r{rnd} 阶段 {idx + 1}（{name} 之后）继续")

    history, result = [], None
    for rnd in range(start_round, max_rounds):
        stage_list = rec["stages"]
        begin = start_idx + 1 if rnd == start_round else 0
        if begin == 0:
            cur = s                                  # 每轮从原始场景重跑（回调重跑语义）
        infos = []
        for si in range(begin, len(stage_list)):
            cfg = stage_list[si]
            name = cfg["stage"]
            if name == "recover":
                cur, info = recover(s, cur, **{k: v for k, v in cfg.items() if k != "stage"})
            else:
                cur, info = run_stage(name, cur, cfg)
            infos.append(info)
            _ckpt_save(out_dir, rnd, si, name, cur, rec, fp)
            if verbose:
                print(f"  [{rnd}] {name:<9} -> n={cur.n:<7} {json.dumps({k: v for k, v in info.items() if k != 'stage'}, ensure_ascii=False)}")
        tiles = build_tiles(cur, out_dir, rec["export"]["lod_levels"],
                           rec["export"].get("quantize_geometry", True),
                           rec["export"].get("geometric_error_scale", 0.1),
                           rec["export"].get("sh_in_tiles", "lod"))
        gsq_path = os.path.join(out_dir, "scene.gsq")
        gsq_bytes = write_gsq(gsq_path, cur)          # 先落盘取真实字节，评测后再带元数据回写
        metrics = evaluate(s, cur, tiles, gsq_bytes)
        metrics["gsq_mb"] = round(gsq_bytes / 1024 ** 2, 3)
        if do_render:
            from .raster import render_eval
            m = render_eval(s, cur, n_views)
            metrics.update({"psnr_render": m["psnr_render"], "ssim_render": m["ssim_render"],
                            "render_views": m["views"],
                            "render_ms_per_view": m["render_ms_per_view"],
                            "render_backend": m["backend"]})
        floor = rec["target"]["psnr_floor"]
        passed, checks = gate(metrics, floor)
        metrics["gate"] = checks
        history.append({"round": rnd, "metrics": metrics, "passed": bool(passed)})
        if verbose:
            key = "psnr_render" if "psnr_render" in metrics else "color_psnr"
            print(f"  [{rnd}] metrics: X{metrics['ratio']} {key}={metrics[key]} "
                  f"IoU={metrics['voxel_iou']} -> {'PASS' if passed else 'FAIL'}")
        result = {"scene": cur, "tiles": tiles, "metrics": metrics, "infos": infos, "recipe": rec,
                  "passed": bool(passed), "rounds": rnd + 1}
        if passed or rnd == max_rounds - 1:
            break
        rec = relax(rec, rnd + 1)

    cur = result["scene"]
    try:
        from .render import compare
        compare(s, result["scene"], os.path.join(out_dir, "preview.png"), result["metrics"])
    except Exception as e:            # 预览失败不影响主流程
        if verbose:
            print(f"  preview skipped: {e}")
    gsq = os.path.join(out_dir, "scene.gsq")
    write_gsq(gsq, cur, {"compression": result["metrics"], "recipe": rec})   # 带元数据回写
    spz_path = None
    if export_spz:                                   # 设计 §6.2：SPZ 导出（P1）
        try:
            from .spz import write_spz
            spz_path = os.path.join(out_dir, "scene.spz")
            spz_info = write_spz(spz_path, cur)
            result["metrics"]["spz_bytes"] = spz_info["bytes"]
        except Exception as e:                       # SPZ 失败不影响 3D Tiles 主产物
            if verbose:
                print(f"  spz export skipped: {e}")
    prov = {"tool": "GeoSqueeze", "version": "0.1.0", "input_fingerprint": _fingerprint(s),
            "platform": platform, "rounds": result["rounds"], "elapsed_s": round(time.time() - t0, 2),
            "input": s.meta}
    viewer = write_viewer(out_dir, "tileset.json", f"GeoSqueeze · {os.path.basename(out_dir)}")
    os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)
    with open(os.path.join(out_dir, "meta", "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(prov, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "meta", "recipe.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    if result["passed"]:
        write_report(os.path.join(out_dir, "report.html"), result["metrics"], rec, result["infos"], prov)
    summary = {"passed": result["passed"], "metrics": result["metrics"], "rounds": result["rounds"],
               "viewer": viewer,
               "history": history, "infos": result["infos"], "tileset": result["tiles"]["tileset"], "gsq": gsq,
               "spz": spz_path,
               "out_dir": os.path.abspath(out_dir)}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    if verbose:
        print(f"done in {prov['elapsed_s']}s · {'PASS' if result['passed'] else 'FAIL'} · {result['tiles']['tileset']}")
    return summary


def eval_dir(out_dir: str, baseline: GeoScene) -> dict:
    cur, h = read_gsq(os.path.join(out_dir, "scene.gsq"))
    tdir = os.path.join(out_dir, "tiles")
    tbytes = sum(os.path.getsize(os.path.join(tdir, f)) for f in os.listdir(tdir)) if os.path.isdir(tdir) else 0
    tbytes += os.path.getsize(os.path.join(out_dir, "tileset.json")) if os.path.isfile(os.path.join(out_dir, "tileset.json")) else 0
    tiles = {"bytes": tbytes, "tiles": h.get("meta", {}).get("compression", {}).get("tiles", 0),
             "tileset": os.path.join(out_dir, "tileset.json")}
    m = evaluate(baseline, cur, tiles)
    return m
