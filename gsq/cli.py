"""命令行入口：python -m gsq <cmd>"""
from __future__ import annotations
import argparse
import glob
import json
import os
import platform as _plat
import sys

from .analyze import plan, profile
from .export import gsq_stats, read_gsq, validate_glb, validate_tileset
from .ir import las_header, make_sample, read_las, read_ply, write_las, write_ply
from .pipeline import eval_dir, run_pipeline

PLATFORMS = ["web", "mobile", "desktop", "edge"]


class UserError(Exception):
    """可预期的输入/配置错误：打印简明信息，不抛栈。"""


def _load(path: str):
    if not os.path.isfile(path):
        raise UserError(f"输入文件不存在：{path}")
    if os.path.getsize(path) == 0:
        raise UserError(f"输入文件为空：{path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gsq":
        s = read_gsq(path)[0]
    elif ext == ".ply":
        s = read_ply(path)
    elif ext in (".las", ".laz"):
        try:
            s = read_las(path)                       # LAZ 会自动调用 laszip/pdal 解压
        except RuntimeError as e:                    # LAZ 缺少解码工具 → 用户可读的提示
            raise UserError(str(e))
        except ValueError as e:
            raise UserError(f"LAS 解析失败：{e}")
    else:
        raise UserError(f"不支持的输入格式：{path}（支持 .ply / .las / .laz / .gsq）")
    if s.n == 0:
        raise UserError(f"场景为空（0 个基元）：{path}")
    if not (np_isfinite(s.pos) and np_isfinite(s.sh) and np_isfinite(s.opacity)):
        raise UserError(f"场景包含 NaN/Inf：{path}")
    return s


def np_isfinite(a) -> bool:
    import numpy as np
    return bool(np.isfinite(a).all())


def _validate_recipe(rec: dict) -> dict:
    """配方校验：允许覆写任意字段，但结构必须合法，避免跑一半才崩。"""
    if not isinstance(rec, dict):
        raise UserError("recipe 必须是 JSON 对象")
    for key in ("target", "stages", "export"):
        if key not in rec:
            raise UserError(f"recipe 缺少字段：{key}")
    if not isinstance(rec["stages"], list) or not rec["stages"]:
        raise UserError("recipe.stages 必须是非空数组")
    known = {"prune", "distill", "recover", "quantize", "lowrank"}
    for st in rec["stages"]:
        name = st.get("stage")
        if name not in known:
            raise UserError(f"未知阶段：{name}（可用：{sorted(known)}）")
        if name == "prune":
            r = float(st.get("ratio", 0))
            if not 0 <= r < 1:
                raise UserError(f"prune.ratio 必须在 [0,1)：{r}")
        if name == "quantize":
            pb = int(st.get("pos_bits", 16))
            if pb not in (8, 10, 12, 14, 16):
                raise UserError(f"quantize.pos_bits 建议 8/10/12/14/16：{pb}")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gsq", description="GeoSqueeze：3DGS/点云 → 分析 → 剪枝/蒸馏/量化 → 3D Tiles")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen-sample", help="生成合成点云/3DGS 样例")
    g.add_argument("--out", default="data/sample.ply")
    g.add_argument("--ground", type=int, default=120)
    g.add_argument("--box", type=int, default=28)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--kind", default="gaussian", choices=["gaussian", "point"])
    g.add_argument("--city-tiles", default=None, metavar="DIR",
                   help="改为生成合成城市级 3D Tiles 数据集（--grid/--per-tile/--tile-kind）")
    g.add_argument("--grid", type=int, default=4)
    g.add_argument("--per-tile", type=int, default=20000)
    g.add_argument("--tile-kind", default="glb", choices=["glb", "pnts", "b3dm"])
    g.add_argument("--world-offset", default="0,0,0", help="模拟 ECEF 大坐标，如 5.2e6,4.1e6,3e4")
    g.add_argument("--city-vector", default=None, metavar="PATH",
                   help="改为生成合成城市级 GeoJSON（--grid 控制规模）")
    g.add_argument("--city-mesh", default=None, metavar="DIR",
                   help="改为生成带纹理的 b3dm 城市网格数据集（--grid 控制规模）")

    a = sub.add_parser("analyze", help="属性画像 + 配方建议")
    a.add_argument("input")
    a.add_argument("--platform", default="web", choices=PLATFORMS)
    a.add_argument("--budget", type=float, default=120.0)
    a.add_argument("--floor", type=float, default=28.0)
    a.add_argument("--json", default=None)

    c = sub.add_parser("compress", help="一键压缩并导出 3D Tiles")
    c.add_argument("input")
    c.add_argument("-o", "--out", default="out")
    c.add_argument("--platform", default="web", choices=PLATFORMS)
    c.add_argument("--budget", type=float, default=120.0, help="体积预算 MB")
    c.add_argument("--floor", type=float, default=28.0, help="渲染 PSNR 质量底线 dB")
    c.add_argument("--recipe", default=None, help="自定义 recipe.json")
    c.add_argument("--rounds", type=int, default=3, help="质量门禁最大回调轮数")
    c.add_argument("--views", type=int, default=4, help="渲染评测视位数")
    c.add_argument("--no-render", action="store_true", help="跳过渲染评测（仅体素指标）")
    c.add_argument("--quiet", action="store_true")

    e = sub.add_parser("eval", help="对已压缩结果做验收评测")
    e.add_argument("dir")
    e.add_argument("--baseline", required=True)

    v = sub.add_parser("verify", help="校验产物完整性（tileset / GLB / .gsq）")
    v.add_argument("dir")

    s = sub.add_parser("serve", help="起本地 HTTP 服务以打开流式查看器")
    s.add_argument("dir", nargs="?", default="out")
    s.add_argument("--port", type=int, default=8000)

    t = sub.add_parser("compress-tiles", help="读入已有 3D Tiles 并按瓦片流式再压缩（城市级可用）")
    t.add_argument("input")
    t.add_argument("-o", "--out", default="out_tiles2")
    t.add_argument("--prune", type=float, default=0.5, help="每瓦片点数抽稀比例")
    t.add_argument("--pos-bits", type=int, default=16, choices=(16,))
    t.add_argument("--method", default="voxel", choices=["voxel", "significance"])
    t.add_argument("--max-tile-pts", type=int, default=400000, help="单瓦片点数上限（内存保护）")
    t.add_argument("--no-quantize", action="store_true", help="关闭属性字节量化")
    t.add_argument("--pointify", action="store_true",
                   help="网格瓦片退化为点云处理（默认保留纹理网格）")
    t.add_argument("--tex-max-dim", type=int, default=1024, help="纹理最大边长（像素）")
    t.add_argument("--tex-quality", type=int, default=80, help="纹理 JPEG/WebP 质量 (1-95)")
    t.add_argument("--quiet", action="store_true")

    ti = sub.add_parser("tiles-info", help="扫描 3D Tiles 数据集（不驻留点数据）")
    ti.add_argument("input")
    ti.add_argument("--max-tiles", type=int, default=None)

    cv = sub.add_parser("compress-vector", help="矢量压缩：简化 + 整型化 + 拓扑去重 + 分块 LOD")
    cv.add_argument("input", help="GeoJSON 路径")
    cv.add_argument("-o", "--out", default="out_vec")
    cv.add_argument("--tol", type=float, default=0.35, help="Douglas-Peucker 容差（米）")
    cv.add_argument("--levels", type=int, default=5, help="LOD 层数（2^z 网格）")
    cv.add_argument("--max-features", type=int, default=None)
    cv.add_argument("--quiet", action="store_true")

    sub.add_parser("doctor", help="检查运行环境与可选加速后端")

    args = ap.parse_args(argv)
    try:
        return _dispatch(args)
    except UserError as err:
        print(f"错误：{err}", file=sys.stderr)
        return 1
    except FileNotFoundError as err:
        print(f"错误：找不到文件 {err.filename}", file=sys.stderr)
        return 1


def _dispatch(args) -> int:
    if args.cmd == "gen-sample" and getattr(args, "city_tiles", None):
        from .samples import make_city_tiles
        off = [float(x) for x in str(args.world_offset).split(",")]
        r = make_city_tiles(args.city_tiles, args.grid, args.per_tile, args.tile_kind,
                            args.seed, tuple(off))
        print(json.dumps(r, ensure_ascii=False))
        return 0
    if args.cmd == "gen-sample" and getattr(args, "city_mesh", None):
        from .samples import make_city_mesh
        r = make_city_mesh(args.city_mesh, args.grid, seed=args.seed)
        print(json.dumps(r, ensure_ascii=False))
        return 0
    if args.cmd == "gen-sample" and getattr(args, "city_vector", None):
        from .samples import make_city_vector
        r = make_city_vector(args.city_vector, args.grid, args.seed)
        print(json.dumps(r, ensure_ascii=False))
        return 0
    if args.cmd == "gen-sample":
        d = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(d, exist_ok=True)
        s = make_sample(args.ground, args.box, args.seed)
        if args.kind == "point":
            import numpy as np
            s.kind = "point"
            s.sh = s.sh[:, :1, :].copy()
            s.opacity = np.ones(s.n, np.float32)
        if args.out.lower().endswith(".las"):
            write_las(args.out, s)
            info = las_header(args.out)
            print(json.dumps({"out": args.out, "n": s.n, "degree": s.sh_degree, "kind": "point",
                              "format": "las", "las_version": info["version"],
                              "point_format": info["point_format"]}))
            return 0
        write_ply(args.out, s)
        print(json.dumps({"out": args.out, "n": s.n, "degree": s.sh_degree, "kind": s.kind}))
        return 0

    if args.cmd == "analyze":
        s = _load(args.input)
        prof = profile(s)
        rec = plan(s, args.platform, args.budget, args.floor, prof)
        out = {"profile": prof, "recipe": rec}
        print(json.dumps(out, ensure_ascii=False, indent=1))
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=1)
        return 0

    if args.cmd == "compress":
        s = _load(args.input)
        if not args.quiet:
            print(f"input: {s.n} primitives · degree {s.sh_degree} · kind {s.kind} · platform {args.platform}")
        rec = None
        if args.recipe:
            with open(args.recipe, encoding="utf-8") as f:
                rec = _validate_recipe(json.load(f))
        summary = run_pipeline(s, args.out, args.platform, args.budget, args.floor, rec,
                               args.rounds, verbose=not args.quiet, n_views=args.views,
                               do_render=not args.no_render)
        print(json.dumps({"passed": summary["passed"], **summary["metrics"]}, ensure_ascii=False, indent=1))
        return 0 if summary["passed"] else 2

    if args.cmd == "eval":
        base = _load(args.baseline)
        m = eval_dir(args.dir, base)
        print(json.dumps(m, ensure_ascii=False, indent=1))
        return 0 if m.get("psnr_render", m["color_psnr"]) >= 20 else 2

    if args.cmd == "tiles-info":
        from .tiles_in import scan_tileset
        r = scan_tileset(args.input, args.max_tiles)
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return 0

    if args.cmd == "compress-tiles":
        from .tiles_in import compress_tileset
        r = compress_tileset(args.input, args.out, prune=args.prune, pos_bits=args.pos_bits,
                             method=args.method, max_points_per_tile=args.max_tile_pts,
                             quantize=not args.no_quantize, verbose=not args.quiet,
                             keep_mesh=not args.pointify, tex_max_dim=args.tex_max_dim,
                             tex_quality=args.tex_quality)
        print(json.dumps({k: v for k, v in r.items() if k != "per_tile"},
                         ensure_ascii=False, indent=1))
        return 0

    if args.cmd == "compress-vector":
        from .vector import compress_vector
        from .vec_viewer import write_vector_viewer
        r = compress_vector(args.input, args.out, tol=args.tol, levels=args.levels,
                            max_features=args.max_features, verbose=not args.quiet)
        r["viewer"] = write_vector_viewer(args.out, title=f"GeoSqueeze · {os.path.basename(args.input)}")
        print(json.dumps({"features": r["features"], "vertices": r["vertices"],
                          "ratio": r["single"]["ratio"], "max_dev_m": r["single"]["max_dev"],
                          "len_rel_err": r["single"]["len_rel_err"],
                          "area_rel_err": r["single"]["area_rel_err"],
                          "tiles": r["tiles"]["tiles"], "viewer": r["viewer"]},
                         ensure_ascii=False, indent=1))
        return 0

    if args.cmd == "verify":
        return _verify(args.dir)

    if args.cmd == "serve":
        return _serve(args.dir, args.port)

    if args.cmd == "doctor":
        from .raster import backend_info
        info = {"python": sys.version.split()[0], "platform": _plat.platform(),
                "numpy": __import__("numpy").__version__,
                "raster_backend": backend_info()}
        try:
            import PIL
            info["pillow"] = PIL.__version__
        except Exception:
            info["pillow"] = None
        for mod in ("torch", "diff_gaussian_rasterization", "jsonschema"):
            try:
                info[mod] = getattr(__import__(mod), "__version__", "installed")
            except Exception:
                info[mod] = None
        print(json.dumps(info, ensure_ascii=False, indent=1))
        return 0
    return 1


def _verify(d: str) -> int:
    if not os.path.isdir(d):
        raise UserError(f"目录不存在：{d}")
    ts_path = os.path.join(d, "tileset.json")
    if not os.path.isfile(ts_path):
        raise UserError(f"缺少 tileset.json：{d}")
    rep = {"dir": d, "checks": {}, "errors": []}
    try:
        rep["checks"]["tileset"] = validate_tileset(ts_path)
    except AssertionError as err:
        rep["errors"].append(f"tileset: {err}")
    glbs = sorted(glob.glob(os.path.join(d, "tiles", "*.glb")))
    ok = 0
    for g in glbs:
        try:
            validate_glb(g)
            ok += 1
        except AssertionError as err:
            rep["errors"].append(f"{os.path.basename(g)}: {err}")
    rep["checks"]["glb"] = {"total": len(glbs), "valid": ok}
    gsq = os.path.join(d, "scene.gsq")
    if os.path.isfile(gsq):
        try:
            sc, h = read_gsq(gsq)
            st = gsq_stats(gsq)
            rep["checks"]["gsq"] = {"n": sc.n, "sh_mode": st["sh_mode"], "bytes": st["bytes"],
                                    "fields": len(st["fields"])}
            if sc.n == 0:
                rep["errors"].append("gsq: 0 基元")
        except Exception as err:
            rep["errors"].append(f"gsq: {type(err).__name__} {err}")
    else:
        rep["errors"].append("缺少 scene.gsq")
    for extra in ("viewer.html", "report.html"):
        if not os.path.isfile(os.path.join(d, extra)):
            rep["errors"].append(f"缺少 {extra}")
    rep["valid"] = not rep["errors"]
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0 if rep["valid"] else 2


def _serve(d: str, port: int) -> int:
    import functools
    import http.server
    import socketserver
    if not os.path.isdir(d):
        raise UserError(f"目录不存在：{d}")
    page = "viewer.html" if os.path.isfile(os.path.join(d, "viewer.html")) else \
        "mesh_viewer.html" if os.path.isfile(os.path.join(d, "mesh_viewer.html")) else \
        "cesium.html" if os.path.isfile(os.path.join(d, "cesium.html")) else None
    if not page:
        raise UserError(f"{d} 中没有 viewer.html / mesh_viewer.html / cesium.html，请先执行 compress")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=os.path.abspath(d))
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/{page}"
        print(f"serving {os.path.abspath(d)}\n打开：{url}\nCtrl+C 停止", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n停止")
    return 0


if __name__ == "__main__":
    sys.exit(main())
