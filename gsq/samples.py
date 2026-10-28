"""合成城市级测试数据：多瓦片 3D Tiles（glb/pnts/b3dm）与矢量图层（GeoJSON）。

用途：① 端到端验收的可复现输入；② 城市级性能基准（--grid 8 --per-tile 50000 → 数百万点）。
"""
from __future__ import annotations

import json
import os
import struct

import numpy as np

from .ir import SH_C0, GeoScene


# ---------------- 城市场景（点云） ----------------
def _building_points(rng, cx, cy, w, d, h, n, color):
    """在长方体表面（四墙 + 屋顶）均匀采样点。"""
    per = max(1, n // 5)
    out = []
    for side in range(4):
        u = rng.random(per)
        v = rng.random(per)
        if side == 0:
            p = np.column_stack([cx - w / 2 + u * w, np.full(per, cy - d / 2), v * h])
        elif side == 1:
            p = np.column_stack([cx - w / 2 + u * w, np.full(per, cy + d / 2), v * h])
        elif side == 2:
            p = np.column_stack([np.full(per, cx - w / 2), cy - d / 2 + u * d, v * h])
        else:
            p = np.column_stack([np.full(per, cx + w / 2), cy - d / 2 + u * d, v * h])
        out.append(p)
    roof = np.column_stack([cx - w / 2 + rng.random(per) * w, cy - d / 2 + rng.random(per) * d,
                            np.full(per, h)])
    out.append(roof)
    pts = np.vstack(out)
    col = np.tile(np.asarray(color, np.float32), (len(pts), 1))
    shade = 0.88 + 0.24 * rng.random((len(pts), 1)).astype(np.float32)
    return pts, np.clip(col * shade, 0, 1)


def _ground_points(rng, cx, cy, size, n, color=(0.62, 0.63, 0.58)):
    p = np.column_stack([cx + (rng.random(n) - 0.5) * size, cy + (rng.random(n) - 0.5) * size,
                         0.02 * rng.random(n)])
    c = np.tile(np.asarray(color, np.float32), (n, 1))
    return p, np.clip(c * (0.9 + 0.2 * rng.random((n, 1)).astype(np.float32)), 0, 1)


def _scene_points(grid, per_tile, seed=0, tile_size=200.0):
    """按网格生成每个瓦片的局部点云；返回 [(i,j,pos_local, rgb)]。"""
    rng = np.random.default_rng(seed)
    out = []
    per_bld = max(200, per_tile // 12)
    for i in range(grid):
        for j in range(grid):
            cx, cy = (i + 0.5) * tile_size, (j + 0.5) * tile_size
            P, C = [np.zeros((0, 3))], [np.zeros((0, 3))]
            g_n = int(per_tile * 0.25)
            gp, gc = _ground_points(rng, tile_size / 2, tile_size / 2, tile_size * 0.96, g_n)
            P.append(gp)
            C.append(gc)
            nb = 8
            for _ in range(nb):
                bw = rng.uniform(12, 34)
                bd = rng.uniform(12, 34)
                bh = rng.uniform(8, 60)
                bx = rng.uniform(tile_size * 0.15, tile_size * 0.85)
                by = rng.uniform(tile_size * 0.15, tile_size * 0.85)
                col = (rng.uniform(0.35, 0.85), rng.uniform(0.35, 0.7), rng.uniform(0.3, 0.75))
                bp, bc = _building_points(rng, bx, by, bw, bd, bh, per_bld, col)
                P.append(bp)
                C.append(bc)
            pos = np.vstack(P)
            rgb = np.vstack(C)
            out.append((i, j, pos.astype(np.float32), rgb.astype(np.float32), cx, cy))
    return out


def _write_pnts(path: str, pos: np.ndarray, rgb: np.ndarray) -> int:
    n = len(pos)
    ftj = {"POINTS_LENGTH": n,
           "POSITION": {"byteOffset": 0},
           "RGB": {"byteOffset": n * 12, "componentType": "UNSIGNED_BYTE"}}
    jb = json.dumps(ftj, separators=(",", ":")).encode()
    pad = (-len(jb)) % 8
    jb += b" " * pad
    bin_ = pos.astype("<f4").tobytes() + np.clip(rgb * 255, 0, 255).astype(np.uint8).tobytes()
    bpad = (-len(bin_)) % 8
    bin_ += b"\x00" * bpad
    total = 28 + len(jb) + len(bin_)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sIIIIII", b"pnts", 1, total, len(jb), len(bin_), 0, 0))
        f.write(jb)
        f.write(bin_)
    return total


def _write_b3dm(path: str, glb_path: str) -> int:
    glb = open(glb_path, "rb").read()
    ftj = json.dumps({"BATCH_LENGTH": 1}, separators=(",", ":")).encode()
    ftj += b" " * ((-len(ftj)) % 8)
    btj = json.dumps({"name": ["tile"]}, separators=(",", ":")).encode()
    btj += b" " * ((-len(btj)) % 8)
    total = 28 + len(ftj) + len(btj) + len(glb)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sIIIIII", b"b3dm", 1, total, len(ftj), 0, len(btj), 0))
        f.write(ftj)
        f.write(btj)
        f.write(glb)
    return total


def make_city_tiles(out_dir: str, grid=4, per_tile=20000, kind="glb", seed=0,
                    world_offset=(0.0, 0.0, 0.0), tile_size=200.0, levels=2) -> dict:
    """生成多瓦片 3D Tiles 1.1 点云数据集（可选 pnts / b3dm 格式）。

    world_offset 用于模拟 ECEF/投影大坐标（如 5e6 量级），验证 float32 精度处理。
    """
    from .export import write_glb
    os.makedirs(os.path.join(out_dir, "tiles"), exist_ok=True)
    for f in os.listdir(os.path.join(out_dir, "tiles")):
        os.remove(os.path.join(out_dir, "tiles", f))
    off = np.asarray(world_offset, np.float64)
    data = _scene_points(grid, per_tile, seed, tile_size)
    children, n_tiles, n_pts = [], 0, 0
    for i, j, pos, rgb, cx, cy in data:
        if kind == "pnts":
            name = f"tiles/{i}_{j}.pnts"
            _write_pnts(os.path.join(out_dir, name), pos, rgb)
        elif kind == "b3dm":
            tmp = os.path.join(out_dir, f"._{i}_{j}.glb")
            sh = np.empty((len(pos), 1, 3), np.float32)
            sh[:, 0, :] = (rgb - 0.5) / SH_C0
            s = GeoScene(pos, sh, np.ones(len(pos), np.float32),
                         np.full((len(pos), 3), 0.25, np.float32),
                         np.tile(np.array([1, 0, 0, 0], np.float32), (len(pos), 1)), "point")
            write_glb(tmp, s, np.zeros(3), float(tile_size), False, 0)
            name = f"tiles/{i}_{j}.b3dm"
            _write_b3dm(os.path.join(out_dir, name), tmp)
            os.remove(tmp)
        else:
            name = f"tiles/{i}_{j}.glb"
            sh = np.empty((len(pos), 1, 3), np.float32)
            sh[:, 0, :] = (rgb - 0.5) / SH_C0
            s = GeoScene(pos, sh, np.ones(len(pos), np.float32),
                         np.full((len(pos), 3), 0.25, np.float32),
                         np.tile(np.array([1, 0, 0, 0], np.float32), (len(pos), 1)), "point")
            # 源瓦片不量化（float32），贴近真实数据；节点平移承载瓦片世界位置
            write_glb(os.path.join(out_dir, name), s, np.zeros(3), float(tile_size), False, 0)
        n_tiles += 1
        n_pts += len(pos)
        box = [float(off[0] + cx), float(off[1] + cy), float(off[2] + tile_size * 0.2),
               tile_size / 2, 0, 0, 0, tile_size / 2, 0, 0, 0, tile_size]
        children.append({"boundingVolume": {"box": box},
                         "geometricError": 4.0 if levels > 1 else 0.0,
                         "refine": "REPLACE",
                         "content": {"uri": name},
                         "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                                       float(off[0] + cx), float(off[1] + cy), float(off[2]), 1],
                         "extras": {"primitives": len(pos)}})
    span = grid * tile_size
    ts = {"asset": {"version": "1.1", "generator": "GeoSqueeze-sample", "gltfUpAxis": "Z"},
          "extras": {"geosqueeze": {"generator": "sample", "upAxis": "Z"}},
          "geometricError": max(span, 8.0),
          "root": {"boundingVolume": {"box": [float(off[0] + span / 2), float(off[1] + span / 2),
                                              float(off[2] + tile_size * 0.2),
                                              span / 2, 0, 0, 0, span / 2, 0, 0, 0, tile_size]},
                   "geometricError": max(span / 4, 4.0),
                   "refine": "REPLACE",
                   "children": children}}
    p = os.path.join(out_dir, "tileset.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(ts, f, ensure_ascii=False, indent=1)
    b = sum(os.path.getsize(os.path.join(out_dir, c["content"]["uri"])) for c in children) \
        + os.path.getsize(p)
    return {"tileset": p, "tiles": n_tiles, "points": n_pts, "bytes": b,
            "grid": grid, "kind": kind, "offset": list(off)}


# ---------------- 带纹理城市网格数据（b3dm，验收纹理保留链路） ----------------
def _make_facade_tex(path: str, size=512) -> None:
    """程序化立面纹理：窗户格网 + 底部 1/8 素色条带（供屋顶 UV 复用同一张贴图）。"""
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(3)
    im = Image.new("RGB", (size, size), (168, 162, 150))
    d = ImageDraw.Draw(im)
    roof_h = size // 8
    for y in range(roof_h, size):
        for x in range(0, size, 4):                    # 砖纹底噪
            v = 150 + int(rng.integers(-10, 10))
            d.rectangle([x, y, x + 3, y], fill=(v, v - 6, v - 14))
    cell = size // 4
    for r in range(4):                                 # 4×4 窗格（避开顶部素带）
        for c in range(4):
            x0, y0 = c * cell + 12, roof_h + r * cell + 10
            w, h = cell - 24, cell - 26
            base = rng.integers(60, 140)
            d.rectangle([x0, y0, x0 + w, y0 + h], fill=(int(base * 0.8), int(base), int(base * 1.15)))
            d.rectangle([x0, y0, x0 + w // 2, y0 + h], fill=(int(base * 0.55), int(base * 0.75),
                                                             int(base * 0.95)))
            d.rectangle([x0, y0, x0 + w, y0 + 3], fill=(210, 205, 196))
    d.rectangle([0, 0, size, roof_h], fill=(96, 88, 80))
    im.save(path, "PNG")


def _make_ground_tex(path: str, size=512) -> None:
    """程序化地面纹理：沥青底 + 车道线块。"""
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(4)
    im = Image.new("RGB", (size, size), (74, 74, 76))
    d = ImageDraw.Draw(im)
    for _ in range(2600):
        x, y = int(rng.integers(0, size)), int(rng.integers(0, size))
        v = int(rng.integers(48, 100))
        d.point((x, y), fill=(v, v, v + 2))
    d.rectangle([0, size // 2 - 3, size, size // 2 + 3], fill=(206, 196, 120))
    for x in range(0, size, 64):
        d.rectangle([x, 0, x + 30, 8], fill=(190, 188, 182))
    im.save(path, "PNG")


def _box_walls_uv(x0, y0, x1, y1, z0, z1, uv_scale):
    """长方体四墙+顶：每面两个三角形，UV 按世界尺寸缩放（REPEAT 平铺）。"""
    v = []
    faces = [
        [(x0, y0, z1), (x1, y0, z1), (x1, y0, z0), (x0, y0, z0)],   # -Y 墙
        [(x1, y1, z1), (x0, y1, z1), (x0, y1, z0), (x1, y1, z0)],   # +Y 墙
        [(x0, y1, z1), (x0, y0, z1), (x0, y0, z0), (x0, y1, z0)],   # -X 墙
        [(x1, y0, z1), (x1, y1, z1), (x1, y1, z0), (x1, y0, z0)],   # +X 墙
    ]
    for f in faces:
        (ax, ay, az), (bx, by, bz), (cx, cy, cz), (dx, dy, dz) = f
        wv = max(abs(bx - ax), abs(by - ay))
        hv = az - dz
        s = uv_scale
        base = len(v)
        v += [(ax, ay, az, 0.0, 0.0), (bx, by, bz, wv * s, 0.0),
              (cx, cy, cz, wv * s, hv * s), (dx, dy, dz, 0.0, hv * s)]
        yield base
    top = len(v)                                        # 屋顶：素色条带 UV（v∈[0,0.06]）
    v += [(x0, y0, z1, 0.0, 0.02), (x1, y0, z1, 0.45, 0.02),
          (x1, y1, z1, 0.45, 0.06), (x0, y1, z1, 0.0, 0.06)]
    yield "top"


def make_city_mesh(out_dir: str, grid=3, tile_size=120.0, seed=0, buildings_per_tile=6,
                   uv_scale=1.0 / 3.0, texture_size=512) -> dict:
    """生成带纹理的 b3dm 城市数据集：每瓦片 = 地面(prim0/mat0) + 建筑群(prim1/mat1)。"""
    from .mesh import _glb_mesh_bytes, write_b3dm

    tdir = os.path.join(out_dir, "tiles")
    os.makedirs(tdir, exist_ok=True)
    for f in os.listdir(tdir):
        os.remove(os.path.join(tdir, f))
    tex_f = os.path.join(out_dir, "._facade.png")
    tex_g = os.path.join(out_dir, "._ground.png")
    _make_facade_tex(tex_f, texture_size)
    _make_ground_tex(tex_g, texture_size)
    facade = open(tex_f, "rb").read()
    ground = open(tex_g, "rb").read()
    os.remove(tex_f)
    os.remove(tex_g)
    rng = np.random.default_rng(seed)
    children, n_tris, n_verts, total_bytes = [], 0, 0, 0
    for i in range(grid):
        for j in range(grid):
            ox, oy = i * tile_size, j * tile_size
            V, UVW = [], []                            # 顶点池 + UV（REPEAT 平铺）

            def quad(a, b, c, d, uvs):
                base = len(V)
                V.extend([a, b, c, d])
                UVW.extend(uvs)
                return base

            gb = quad((0.4, 0.4, 0.0), (tile_size - 0.4, 0.4, 0.0),
                      (tile_size - 0.4, tile_size - 0.4, 0.0), (0.4, tile_size - 0.4, 0.0),
                      [(0, 0), (tile_size * 0.02, 0), (tile_size * 0.02, tile_size * 0.02),
                       (0, tile_size * 0.02)])
            gnd = [gb, gb + 1, gb + 2, gb, gb + 2, gb + 3]
            walls, roofs = [], []
            for _ in range(buildings_per_tile):
                bw = rng.uniform(10, 28)
                bd = rng.uniform(10, 28)
                bh = rng.uniform(6, 40)
                bx = rng.uniform(8, tile_size - 8 - bw)
                by = rng.uniform(8, tile_size - 8 - bd)
                x0, y0, x1, y1 = bx, by, bx + bw, by + bd
                s = uv_scale
                wall_defs = [
                    ((x0, y0, bh), (x1, y0, bh), (x1, y0, 0), (x0, y0, 0), bw),
                    ((x1, y1, bh), (x0, y1, bh), (x0, y1, 0), (x1, y1, 0), bw),
                    ((x0, y1, bh), (x0, y0, bh), (x0, y0, 0), (x0, y1, 0), bd),
                    ((x1, y0, bh), (x1, y1, bh), (x1, y1, 0), (x1, y0, 0), bd),
                ]
                for a, b, c, d, wv in wall_defs:
                    bs = quad(a, b, c, d, [(0, 0), (wv * s, 0), (wv * s, bh * s), (0, bh * s)])
                    # 反转绕序：让 front face 朝向墙外（否则被 three.js/Cesium 背面剔除）
                    walls += [bs, bs + 2, bs + 1, bs, bs + 3, bs + 2]
                bs = quad((x0, y0, bh), (x1, y0, bh), (x1, y1, bh), (x0, y1, bh),
                          [(0.02, 0.02), (0.44, 0.02), (0.44, 0.06), (0.02, 0.06)])
                roofs += [bs, bs + 1, bs + 2, bs, bs + 2, bs + 3]
            P = np.asarray(V, np.float64)
            UV = np.asarray(UVW, np.float32)
            W = np.asarray(walls, np.int64)
            R = np.asarray(roofs, np.int64)
            G = np.asarray(gnd, np.int64)
            bld = np.concatenate([W, R])
            prims = [{"pos": P[G], "idx": np.arange(6, dtype=np.int64), "uv": UV[G], "mat": 0},
                     {"pos": P[bld], "idx": np.arange(len(bld), dtype=np.int64), "uv": UV[bld],
                      "mat": 1}]
            name = f"tiles/{i}_{j}.b3dm"
            fp = os.path.join(out_dir, name)
            glb = _glb_mesh_bytes(prims, [(ground, "image/png"), (facade, "image/png")],
                                  [0, 1], quantize=False)
            nb = write_b3dm(fp, glb)
            total_bytes += nb
            tris = (len(W) + len(R) + len(G)) // 3
            n_tris += tris
            n_verts += len(P)
            children.append({"boundingVolume": {"box": [tile_size / 2, tile_size / 2, 20.0,
                                                        tile_size / 2, 0, 0, 0,
                                                        tile_size / 2, 0, 0, 0, 40.0]},
                             "geometricError": 2.0, "refine": "REPLACE",
                             "content": {"uri": name},
                             "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                                           float(ox), float(oy), 0.0, 1],
                             "extras": {"tris": tris, "verts": int(len(P))}})
    span = grid * tile_size
    ts = {"asset": {"version": "1.1", "generator": "GeoSqueeze-sample", "gltfUpAxis": "Z"},
          "extras": {"geosqueeze": {"generator": "sample-mesh", "upAxis": "Z"}},
          "geometricError": span,
          "root": {"boundingVolume": {"box": [span / 2, span / 2, 20.0, span / 2, 0, 0,
                                              0, span / 2, 0, 0, 0, 40.0]},
                   "geometricError": span / 4, "refine": "REPLACE", "children": children}}
    p = os.path.join(out_dir, "tileset.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(ts, f, ensure_ascii=False, indent=1)
    total_bytes += os.path.getsize(p)
    return {"tileset": p, "tiles": len(children), "tris": n_tris, "verts": n_verts,
            "bytes": total_bytes, "tex_bytes": len(ground) + len(facade), "textures_per_tile": 2}


def make_city_vector(path: str, grid=10, seed=0, cell=100.0, offset=(0.0, 0.0),
                     parcels_per_cell=25) -> dict:
    """生成城市矢量图层：地块多边形（相邻共边）+ 道路中心线 + 属性表。"""
    rng = np.random.default_rng(seed)
    feats = []
    names = ["张村", "李庄", "王屯", "赵巷", "孙楼", "周湾", "吴店", "郑埠"]
    for i in range(grid):
        for j in range(grid):
            ox, oy = i * cell, j * cell
            # 地块：规则 5×5 格网切分后扰动顶点，相邻地块共用边界（模拟真实拓扑）
            n = 5
            step = cell / n
            for a in range(n):
                for b in range(n):
                    x0, y0 = ox + a * step, oy + b * step
                    x1, y1 = x0 + step, y0 + step
                    pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                    if rng.random() < 0.6:            # 部分边界扰动，制造简化收益
                        k = rng.integers(0, 4)
                        pts = pts[:k + 1] + [[(pts[k][0] + pts[(k + 1) % 4][0]) / 2 + rng.normal(0, step * 0.06),
                                              (pts[k][1] + pts[(k + 1) % 4][1]) / 2 + rng.normal(0, step * 0.06)]] \
                              + pts[k + 1:]
                    ring = [[round(p[0] + offset[0], 3), round(p[1] + offset[1], 3)] for p in pts]
                    ring.append(ring[0])
                    feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                                  "properties": {"kind": "parcel", "code": f"P{i:02d}{j:02d}{a}{b}",
                                                 "owner": names[int(rng.integers(0, len(names)))],
                                                 "area_m2": round(step * step, 1),
                                                 "year": int(rng.integers(1985, 2024)),
                                                 "floors": int(rng.integers(1, 30))}})
            # 道路：穿过单元的两条中心线，顶点密集（模拟 CAD 折线）
            for axis in (0, 1):
                base = ox + cell * 0.5 if axis == 0 else oy + cell * 0.5
                pts = []
                for t in np.linspace(0, cell, 41):
                    if axis == 0:
                        pts.append([round(ox + t + offset[0], 3),
                                    round(base + 2.5 * np.sin(t / 9.0) + offset[1], 3)])
                    else:
                        pts.append([round(base + 2.5 * np.cos(t / 11.0) + offset[0], 3),
                                    round(oy + t + offset[1], 3)])
                feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": pts},
                              "properties": {"kind": "road", "code": f"R{i:02d}{j:02d}{axis}",
                                             "owner": "市政", "area_m2": 0.0,
                                             "year": 2005, "floors": 0}})
    gj = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::4547"}},
          "features": feats}
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False)
    nv = sum(len(f["geometry"]["coordinates"][0] if f["geometry"]["type"] == "Polygon"
                 else f["geometry"]["coordinates"]) for f in feats)
    return {"geojson": path, "features": len(feats), "vertices": nv,
            "bytes": os.path.getsize(path)}
