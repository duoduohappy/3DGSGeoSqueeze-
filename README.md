# GeoSqueeze

端到端三维资产压缩工具：**3DGS / 点云 → 自动分析 → 剪枝 + 蒸馏 + 恢复微调 + 量化（+ 低秩）→ 可流式加载的 3D Tiles 1.1 / GSQ2 紧凑格式**。

纯 Python（numpy + Pillow），零 GPU 依赖即可跑通全链路；检测到 torch + diff-gaussian-rasterization + CUDA 时自动切换官方光栅化器。

> 模块分层、主链路调用关系与设计稿差异见 **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**。

## 快速开始

```bash
python -m gsq gen-sample --out data/sample.ply        # 合成街景（26,400 高斯）
python -m gsq analyze data/sample.ply                 # 属性画像 + 配方建议
python -m gsq compress data/sample.ply -o out --platform web --floor 24
python -m gsq verify out                              # 产物完整性校验
python -m gsq serve out                               # 打开流式查看器 http://127.0.0.1:8000/viewer.html
```

## 3D Tiles 输入再压缩（城市级）

读入**已有 3D Tiles 数据集**做二次压缩，保持原树结构与瓦片边界（仍是合法的流式 tileset）：

| 输入 | 输出 | 说明 |
|---|---|---|
| `.glb`（点图元） | `.pnts`（量化）或 `.glb` | 点图元全量读入，走点云路径 |
| `.glb`/`.b3dm`（三角面网格） | `.b3dm`（**纹理保留**） | 几何 u16 体量化（KHR_mesh_quantization）+ 体素聚类抽稀（保 UV，退化守卫）+ 纹理重压缩 |
| `.pnts`（f32 或 u16 体量化） | `.pnts` | QUANTIZED_VOLUME 双向支持 |
| `.b3dm` | 同上（按内嵌 glTF 图元类型自动分派） | 剥离 Feature/Batch Table 后取内嵌 glTF |
| `.cmpt` | 递归展开 | 组合瓦片逐子块处理 |

城市级关键设计：
- **逐瓦片流式**：解析/压缩/写盘以单瓦片为单位，**峰值内存只与最大瓦片相关**（2.34M 点实测峰值 9.8 MB）
- **大坐标精度**：ECEF（\|x\|≈5.2e6）用 float64 累积 `tile.transform`，减去场景原点存 float32，
  输出根节点 `transform` 平移复原 —— 实测逐点误差 **2.08 mm**（prune=0）
- **点云输出自动选 pnts**：u16 体量化 + u8 RGB ≈ 9 B/点，比逐点 GLB 再省一半
- **网格瓦片纹理保留**：三角面 b3dm/glb 走独立管线——几何 u16 量化（误差 = extent/65535，120m 瓦片 ≈1.8 mm）、
  UV u16 量化、体素聚类抽稀（簇内位置/UV 均值，三角形大面积退化时自动停在上一档，宁少抽不毁几何）、
  纹理 Pillow 重压缩（等比缩放 + JPEG/WebP 重编码，重编码反而变大则保留原图，只赚不赔）
- **纹理保真验收**：逐纹理 PSNR（在压缩分辨率下对比）+ 三角形/顶点数守恒 + 峰值内存上界，全部进入测试断言
- `geometricError` 按抽稀体素重算，子层 ≤ 父层；`tiles-info` 扫描不驻留点数据

## 矢量数据压缩（城市级）

GeoJSON → **Douglas–Peucker 简化 → 坐标整型化（4096 网格，同 MVT 约定）→ 顶点池拓扑去重
→ 属性字典/varint 编码（GVEC 容器）→ 2^z 网格分块 LOD**：

| 环节 | 对应"模型四件套" |
|---|---|
| 坐标整型化 + zigzag varint | 量化 |
| Douglas–Peucker 顶点删除 | 剪枝（拓扑保持，面不破坏闭合） |
| 多级 LOD（粗层大容差轻量副本） | 蒸馏 |
| 顶点池共享边/角（相邻地块共边只存一份）+ 属性字典 | 去重/低秩 |

- 几何保真以 **单侧 Hausdorff 最大偏离 + 长度/面积相对误差** 评测（PSNR 对矢量无意义）
- 分块按要素质心归属单元（不切割几何，跨块要素在所属块整体绘制）
- `vector_viewer.html`：零依赖 Canvas2D，按视距自动选层 + 视口剔除 + 内置 GVEC JS 解码器

## 输入格式

| 格式 | 支持情况 | 说明 |
|---|---|---|
| `.ply` | ✅ 完整 | 3DGS（`f_dc_*/f_rest_*/opacity/scale/rot`）与普通点云（`x/y/z/red/green/blue`），ASCII 与二进制 |
| `.las` | ✅ 完整 | LAS 1.0~1.4，点格式 **0~10** 全覆盖；按头部 `scale/offset` 反算世界坐标；RGB 自动识别 8/16 位；无颜色时按高程着色；记录分类统计与点间距 |
| `.laz` | ⚠️ 需外部解码器 | 纯 Python 无法解 LASzip 算术编码。检测到 `laszip`/`pdal` 在 PATH 时自动解压；否则给出可操作提示（非崩溃） |
| `.gsq` | ✅ 完整 | 本工具产出的紧凑容器，可直接作为输入继续压缩 |
| `.spz` | ✅ 完整 | Niantic SPZ v2 写出（legacy gzip 格式，官方读取器 v1-3 兼容）+ v2/v3 读取；v4（ZSTD 多流）读取明确报错。大坐标自动居中，偏移量存 gzip 注释字段 |
| `.splat` | ✅ 读取 | antimatter15 32 字节/点格式（pos/scale f32 + RGBA u8 + 四元数 u8） |
| `.csv` | ✅ 读取 | x/y/z（大小写不敏感，无表头取前 3 数值列）+ 可选 r,g,b（0-255）→ 点基元进管线 |

> 已用真实数据验证：laspy 测试集中的 `autzen.las`（1.2 / 格式 1 / 106 点 / 州平面坐标 635k）、
> `1_4_w_evlr.las`（1.4 / 格式 6 / 1000 点 / EVLR）。

```bash
python -m gsq gen-sample --out data/sample.las       # 生成 LAS 1.2 格式 2 样例
python -m gsq compress data/tile.las -o out --platform mobile
```

## 实测指标（合成街景 26,400 高斯 → 7,920 基元）

| 平台 | 压缩率 | 瓦片 | .gsq 母版 | 渲染质量 | 说明 |
|---|---|---|---|---|---|
| web | **15.3x** | 28 | 0.120 MB | PSNR 26.6 dB / SSIM 0.959 | 全产物真实字节（含 .gsq），非理论值 |
| mobile | 27.6x | — | — | — | 更激进剪枝 + 12bit 位置 |
| 点云 | **10.3x** | — | — | — | 体素下采样 + 量化（此前直转 4.7x） |
| LAS 输入 | 4.5x | 7 | 0.029 MB | PSNR 31.5 dB / SSIM 0.966 | 6,000 点 LAS 样例；瓦片零丢失（不变量测试覆盖） |
| **3D Tiles 再压缩（城市级）** | **10.5x** | 64 | — | 逐点误差 2.1 mm | 2.34M 点 / 105MB→10MB / 20.3s / **峰值 9.8 MB** |
| **矢量压缩（城市级）** | **10.9x** | 133 | — | Hausdorff 0.34 m | 10,800 要素 / 3.5MB→0.33MB / 56.8k 顶点 9.9x |
| **纹理网格再压缩（零抽稀 q92）** | 1.04x | 9 | — | 纹理 PSNR 41.8~42.4 dB | 三角形 100% 守恒 / 顶点误差 ≤1.82 mm / 只赚不赔 |
| **纹理网格再压缩（抽稀 0.6 + 纹理 128/q70）** | **7.3x** | 9 | — | 纹理 PSNR ≥36.9 dB | 顶点 1674→486 / 纹理 14.6x / 几何守卫生效 |
| **纹理网格城市级** | **4.45x** | 36 | — | 纹理 PSNR ≥38.7 dB | 16 瓦片城市 0.9MB→0.51MB / 峰值 **2.5 MB** |

压缩率 15.3x 是「瓦片 + .gsq 母版」的真实落盘字节之比——早期版本只统计瓦片（33x）为虚高值。

## 压缩引擎（五个阶段，全部真实生效）

| 阶段 | 实现 | 交付要点 |
|---|---|---|
| **prune 剪枝** | 渲染显著性（LightGaussian global significance 等价）/ 几何代理 / 点云体素下采样 | 渲染打分比几何代理 **+5.5 dB**（A/B 实测） |
| **distill 蒸馏** | SH 降阶 + 方向采样能量补偿；效果由**视角相关 SH 渲染**评测真实反映 | SH 基函数与 diff-gaussian-rasterization 数值一致 |
| **recover 恢复微调** | 以原图渲染为监督，残差按合成权重回传 SH DC（渲染对颜色线性 → Jacobi 迭代），覆盖过滤防白噪，可选 grow 填空洞 | 25.93 → 26.73 dB（+0.8） |
| **quantize 量化** | 位置 8-16bit affine + SH 向量量化（k-means 码本）/ **vectree 残差级联码本** / 标量 / 因子化 | 产物是**真实整型编码**，不再是 float32 转储 |
| **lowrank 低秩** | SH 矩阵 SVD → U·diag(S)·Vᵀ+μ，因子 **int8 量化**存储 | 由编码代价模型决策（见下） |

**编码代价模型**：VQ 每基元 1 字节 vs 低秩每基元 rank 字节——对标准 3DGS（SH≤16 系数）低秩必然不划算，planner 自动选 VQ；高维 SH（如 64 系数）且有真实低秩结构时自动切低秩。盲开低秩会让产物大 5 倍（实测教训）。

### 模型线：ONNX 任务模型压缩（H3DNAS 路线）

`gsq compress-model model.onnx -o out_model --ratio 0.4`——无需训练源码，直接图手术：

| 步骤 | 实现 | 边界 |
|---|---|---|
| CDG 分析 | 通道依赖图：从每个 Conv 沿 passthrough 链（BN/Relu/MaxPool/…）找终点消费者，判定可剪层与理论压缩上限 | `gsq compress-model --no-quantize --no-validate` 只出分析 |
| 结构化剪枝 | filter L1 top-k 保留 → 同步切边：下游 Conv 输入列 / BN 统计量 / Gemm 行列；终点不可适配（Reshape/Add/Concat 等）整层放弃 | 输出图永远可加载（onnx.checker 校验） |
| INT8 量化 | onnxruntime 动态量化；**transB=1 的 Gemm 先规范化**（ort 量化器不转置权重，直接量化产出维度不匹配的 MatMulInteger——已修复） | 权重 INT8、激活动态范围 |
| 对拍验证 | 随机输入前向余弦相似度 + 时延（压缩前 vs 后） | 小网络上 0.4 比例剪枝 cosine ≈ 0.87，可调低 ratio |

### Checkpoint 断点续跑（设计 §5）

每阶段完成自动落盘 `out/.gsq/ckpt/`（场景 .gsq + 配方清单 + 输入指纹）：

```bash
python -m gsq compress big.ply -o out ... --resume   # 中断后从最近 checkpoint 继续
python -m gsq rerun-stage out quantize --baseline big.ply   # 单阶段独立重跑
```

## 产物格式

- **3D Tiles 1.1**：八叉树 LOD（ADD 精化 + 子层增量，最细层零丢失——所有基元可溯源），`KHR_mesh_quantization`（pos u16 / color u8 / opacity u8 / scale u8 / rot i8）；**SH 系数按 LOD 分级写入**（粗层 DC、近景层 deg1，`_SH0.._SH2` VEC4 i8）
- **GSQ2 紧凑格式**：`GSQ2 | header JSON | payload`，字段化张量布局（dtype/shape/offset/lo/step），三种 SH 模式（vq / lowrank / plain）均**可完整逆变换读回**（测试覆盖三种模式往返）
- **viewer.html**：零依赖 WebGL2 流式查看器——按屏幕空间误差（ge·focal/dist，像素单位）流式加载瓦片，顶点着色器计算视角相关 SH 颜色，密度自适应点径

## 渲染与评测

- **CPU 参考光栅化器**：与 diff-gaussian-rasterization 数学一致（EWA splatting：Σ=RSSᵀRᵀ → 投影 conic → 深度排序前向 alpha 合成），支持视角相关 SH、渲染贡献统计、残差回传
- **CUDA 后端**：`rasterize_cuda()` 完整对接官方实现（新旧两代 API 兼容），`consistency_check()` 首用时自动校验 CPU/CUDA 一致性（MAE≤0.06）；本机无 GPU 未实测，`python -m gsq doctor` 报告环境
- **指标**：多视位 PSNR/SSIM + 体素 IoU + 真实落盘字节 + 真实渲染耗时（ms/视位）；无 fps 之类的代理假指标

## 质量门禁

`gate()` 双条件：渲染 PSNR ≥ floor 且 体素 IoU ≥ 0.85×keep_ratio。不达标自动放宽配方重跑（≤3 轮）。产物含 `meta/provenance.json`（输入指纹/配方/轮次/耗时）保证可复现。

## CLI 一览

```
gen-sample   合成样例（--kind point 可生成点云）
analyze      属性画像 + 配方（含编码代价模型决策依据）
compress     一键压缩（--recipe 支持 JSON/YAML；--resume 断点续跑；--export-spz 追加 SPZ）
rerun-stage  单阶段独立重跑（基于 .gsq/ckpt 最近记录）
compress-model  模型线：ONNX 剪枝 + INT8 量化 + 对拍（--ratio/--no-quantize/--no-validate）
eval         对已压缩结果做验收评测
verify       tileset/GLB/gsq 三层结构校验 + 产物清单
serve        本地 HTTP 服务打开流式查看器
compress-tiles  读入已有 3D Tiles 逐瓦片再压缩（--prune/--method/--max-tile-pts/--pointify/--tex-max-dim/--tex-quality）
tiles-info      扫描 3D Tiles 数据集（瓦片/格式/点数/包围盒，不驻留点数据）
compress-vector 矢量压缩：简化 + 整型化 + 去重 + 分块 LOD（--tol/--levels）
testbench    测试工作台：在线 3D Tiles/矢量地址加载 + zip 上传测试（--port/--work）
gen-sample      合成样例；--city-tiles / --city-vector / --city-mesh 生成城市级测试数据
doctor       环境与后端能力报告
```

```bash
# 城市级端到端示例
python -m gsq gen-sample --city-tiles data/city --grid 8 --per-tile 40000 --world-offset 5.2e6,4.1e6,3e4
python -m gsq tiles-info data/city
python -m gsq compress-tiles data/city -o out_city --prune 0.5      # 105MB → 10MB
python -m gsq gen-sample --city-vector data/city.geojson --grid 16
python -m gsq compress-vector data/city.geojson -o out_v --tol 0.35 --levels 6
python -m gsq gen-sample --city-mesh data/mesh_city --grid 4        # 带纹理 b3dm 城市样例
python -m gsq compress-tiles data/mesh_city -o out_mesh --prune 0.35 --tex-max-dim 512 --tex-quality 78
python -m gsq serve out_mesh                                       # 纹理网格流式查看器（mesh_viewer.html）
python -m gsq serve out_v                                          # 矢量流式查看器
```

## CesiumJS 前端接入（标准 3D Tiles 直读）

产物本身就是标准 3D Tiles，Cesium 原生加载、零转换。`compress-tiles`（含 `--city-mesh` 路径）会自动在输出目录生成 `cesium.html`，且**数据集 ≤ 8MB 时内嵌为自包含单文件**——直接双击即可打开，无需任何服务器：

```bash
python -m gsq compress-tiles data/mesh_city -o out_mesh --prune 0.35 --tex-max-dim 512 --tex-quality 78
# 双击 out_mesh/cesium.html 即可（tileset.json + 全部瓦片已 base64 内嵌）
python -m gsq serve out_mesh        # 大数据集（>8MB）回退外链版，必须经 HTTP 打开
```

> **为什么直接双击有时看不到建筑物**：浏览器 `file://` 安全策略会拦截页面发起的本地网络请求（CORS），旧版外链页面会报 "Request has failed."。内嵌单文件模式从内存读取全部瓦片，不受此限制；外链模式则必须用 `python -m gsq serve` 经 HTTP 打开，页面在 file:// 下也会显示醒目提示。

查看器为单文件零构建页面，Cesium 1.120 走 unpkg CDN（首次需联网）。支持 URL 调试参数：`?ts=`（tileset 路径）、`?top=1`（正俯视）、`?pitch=`（弧度）、`?d=`（相机距离）、`?geo=1`（按 extras.origin 做 ENU 放置）。

**接入时踩到的真实互操作坑（对自有数据接入 Cesium 同样适用）：**

- **上轴声明必须写在 `tileset.asset.gltfUpAxis`（字符串 `"Z"`）**：Cesium 1.1x 只读该字段，**完全忽略** b3dm Feature Table 里的 `gltfUpAxis`（旧 1.0 行为已移除）。缺省按 Y-up 自动 +90° X 旋转，会把 Z-up 内容放倒——GeoSqueeze 输出已自动带上该声明
- **瓦片 `boundingVolume.box` 与 `tile.transform` 均为瓦片局部坐标**：把世界坐标写进 BV，Cesium 会再乘一次 transform，出现"台阶状"错位。样例生成与再压缩管线均已改为内容实际包围盒
- **headless 软渲染限制**：无头截图（swiftshader）下 UV>1 的 REPEAT 平铺纹理触发渲染 bug；真实 GPU 不受影响。查看器默认机位为正俯视以保证截图脚本稳定，`?pitch=-1.1` 可切斜视角

实测（合成 4×4 带纹理城市，16 b3dm 瓦片）：内嵌单文件 672KB，**file:// 双击与 HTTP 两种方式均完整渲染**建筑 + 窗户纹理 + 地面车道线，位置正确、无放倒、无错位；10MB 城市级数据集自动回退外链模式且不影响压缩管线流式内存（峰值 9.7MB）。参考截图 `out/cesium_city_final.png`。

## 测试工作台（在线数据 + zip 上传）

`python -m gsq testbench --port 8619` 启动单页测试工作台（零第三方依赖，标准库 HTTP 服务），五条通道：

1. **在线 3D Tiles**：输入 tileset.json 地址 → 后端反代（`/p/<b64>/...`，任意源免 CORS）→ 结构统计 → CesiumJS 直读渲染；同一卡片可**走 GeoSqueeze 管线压缩后前后对比**（下载全套瓦片 → `compress-tiles` → 逐阶段指标 + 双视图并排）
2. **在线矢量（GeoJSON）**：输入地址 → 下载 → `compress-vector` 压缩 + 评测 → 矢量查看器渲染
3. **上传 3D Tiles zip**：解压（防 zip-slip）→ 结构校验 → 自动生成内嵌 cesium.html（≤8MB）/外链渲染
4. **上传矢量 zip**（.geojson/.json）→ compress-vector → 查看器渲染
5. **3DGS / 点云主链路**（.ply/.las/.gsq 上传）→ `run_pipeline` 完整执行**剪枝 + 蒸馏 + 量化**（蒸馏只在主链路：球谐阶数降阶，可显式指定目标阶数）→ 逐阶段指标 + 压缩后 3D Tiles 渲染

界面含预设下拉（已验证的公开源）。实测记录（截图见 `out/testbench_*.png`）：

- **3DBAG 荷兰全国建筑**（真实生产数据，CC BY 4.0，3D Tiles 1.1）：根 tileset 含 474 个外部子 tileset + glb 内容瓦片，反代递归加载、961 瓦片就绪渲染成功。统计通道会识别外部 tileset 并抽样统计真实内容格式；反代磁盘缓存上限 1GB（超出按最旧淘汰），避免全国级数据撑爆磁盘。版本号 `v20250903` 随官方发布更新，若失效请到 [3dbag.nl](https://3dbag.nl/en/download) 取新地址
- **3DGS 主链路实测**（合成 gaussian 样例 26,400 基元 / 3 阶球谐 / 5.94MB）：剪枝 26,400 → 7,920（保留 30%，显著性渲染评分）→ **蒸馏球谐 3 阶 → 1 阶**（保留 12 系数 + 能量补偿）→ 量化 16bit 位置 + SH 向量量化（码本 256）→ **5.94MB → 0.389MB（15.28x）**，PSNR 24.2dB。注意：管线内置质量门禁迭代（`relax`），若首轮未达标会逐轮放宽（剪枝变松、蒸馏阶数回升），最终指标以通过门禁的那一轮为准——想让蒸馏保留到产物，可降低 PSNR 底线或直接指定目标阶数
- 在线自有城市 17 瓦片经反代在线加载，Cesium 完整渲染
- Cesium 官方 DiscreteLOD（dragon，3 b3dm 瓦片 4.6MB）经「管线压缩」通道：三角形 119,974 → 56,702（保留 47.3%）+ u16 量化，**4.71x**，几何量化步长 21mm
- 3DBAG 荷兰全国建筑（meshopt 格式）**仅渲染可测、不可压缩**：其 glb 必需 `EXT_meshopt_compression`，本工具不解码该扩展（会给出明确提示）；解压后可压
- world.geo.json 293 要素 → 压缩 15.0x、最大偏离 0.061m，查看器世界地图正确
- zip 上传（带一层根目录）解压校验 16 b3dm → 内嵌单文件渲染
- Cesium 官方 TilesetWithTreeBillboards（3D Tiles 1.0 / i3dm / ECEF 坐标 / Y-up）正确加载渲染——`cesium.html` 新增上轴自适应：默认读 `asset.gltfUpAxis`，无声明按 3D Tiles 默认（Y-up 旋转），`?up=z|y|x` 可强制

查看器对 **ECEF 地球尺度数据自动适配**：预取根 tileset.json 判断包围盒中心是否接近地球半径，是则自动开启对数深度缓冲（否则深度精度不足会导致远处瓦片闪烁/整片不可见），并把默认相机距离收紧到 `r*0.04`（夹在 1.2~6km），打开即可看到单体建筑；`?lat=52.37&lon=4.90` 可指定观察点，`?log=1` / `?nolog=1` 强制开/关对数深度。

代理说明：后端反代按 `HTTPS_PROXY 环境变量 → 直连 → 127.0.0.1:7897` 顺序自动探测。

three.js 路线：GLTFLoader 原生支持 `KHR_mesh_quantization` 的 glb 可直接加载；pnts 需自定义 loader（读 Feature Table + 3D Tiles 语义），工作量更大，未在本仓库实现。

## 工程边界（如实说明）

- CPU 光栅化约 0.66 s/视位（26k 基元）；大场景评测建议启用 CUDA 后端
- CUDA 路径代码完整但未在本机（无 torch/nvcc）实测，首用时 `consistency_check` 自动校验
- viewer 以点精灵近似高斯 splat（圆形软边 + 预乘 alpha），未做逐像素 EWA 各向异性投影
- 评测视位为 orbit 相机；真实资产建议提供训练视角做更严格的验收
- **LAS 只读不写业务坐标**：写出（`write_las`）使用 scale=0.001 与 1km 对齐的 offset，用于样例生成与往返验证；CRS/投影信息不写入（不依赖 pyproj）
- **LAZ 未内置解码器**：LASzip 是 ~1500 行算术编码实现，超出纯 numpy 交付范围；走外部 `laszip`/`pdal`，缺失时明确报错
- **网格抽稀为体素聚类（非流形边坍缩）**：保 UV、保三角形连接，但簇内顶点合并可能钝化细节；质量守卫保证退化三角形 ≤60% 时自动停止加粗。未实现 Draco/meshopt 熵编码，网格瓦片的压缩率主要来自顶点去重 + 几何/纹理量化
- **纹理压缩为 JPEG/WebP 重编码**（Pillow）：未做 GPU 纹理格式（KTX2/BasisU）；带 alpha 纹理自动走 WebP
- **网格查看器（mesh_viewer.html）按 SSE 流式调度**：支持 u16 量化位置/UV + 嵌入 JPEG/PNG/WebP；屏幕空间法线 lambert 着色（无需法线属性）
- **矢量分块按质心归属**：跨单元的长道路不切割，由所属块整体绘制；极长要素在块边界外仍会绘制（视锥剔除按块 bbox 做，视觉正确）
- **矢量 varint 编解码为 Python 循环**：10.8k 要素全量单文件编码+评测约 50s（分块路径秒级）；城市级建议直接用分块产物
- **自产 tileset 坐标约定为 Z-up 直存**（asset.extras 标记），读入外部数据时按 3D Tiles 规范做 Y-up→Z-up 转换；Cesium 直读已打通（见"CesiumJS 前端接入"，依赖 `asset.gltfUpAxis: "Z"` 声明）
- **SPZ 写出为 v2 legacy gzip 格式**：v4（ZSTD 多流）只读报错；SH 高于 3 阶丢弃（SPZ 规范上限）；scale 量化钳在 log∈[-10, 5.94]，极小尺度（<4.5e-5）会贴底
- **模型线算子覆盖**：Conv/BN/Gemm/MatMul + passthrough 链（Relu/MaxPool/Clip 等）；Reshape/Transpose/Add/Concat 等通道语义复杂节点保守跳过（计入 skipped，绝不产出坏图）；分组卷积（group>1）、控制流子图不支持
- **未实现（设计 P2/可选）**：SOG 导出、QAT（需 torch）、LPIPS（需 torch，现以 SSIM+PSNR+体素 IoU 组合门禁）、e57 读取
