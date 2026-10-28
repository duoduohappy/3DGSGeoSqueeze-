"""流水线编排：分析 → 配方 → 阶段 DAG → 导出 → 评测 → 质量门禁回调。"""
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


def run_pipeline(s: GeoScene, out_dir: str, platform="web", budget_mb=120.0, psnr_floor=28.0,
                 recipe=None, max_rounds=3, verbose=True, n_views=4, do_render=True) -> dict:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    prof = profile(s)
    rec = recipe or plan(s, platform, budget_mb, psnr_floor, prof)
    history, result = [], None

    for rnd in range(max_rounds):
        cur, infos = s, []
        for cfg in rec["stages"]:
            name = cfg["stage"]
            if name == "recover":
                cur, info = recover(s, cur, **{k: v for k, v in cfg.items() if k != "stage"})
            else:
                cur, info = run_stage(name, cur, cfg)
            infos.append(info)
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
