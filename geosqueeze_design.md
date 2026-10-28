# GeoSqueeze 设计文档

> 端到端三维资产压缩与流式化工具：输入 3DGS / 点云 → 自动分析 → 剪枝 + 蒸馏 + 量化（+ 低秩分解）→ 输出可流式加载的 3D Tiles / 紧凑格式。
> 定位：填补"数据压缩工具不懂模型、模型压缩研究不接 GIS 管线"的空白。

---

## 1. 项目定位与设计目标

### 1.1 一句话定位

**面向实景三维 GIS 的"压-评-发"一体化引擎**：把 3DGS 场景、激光点云、以及配套的下游任务模型，按质量底线自动压缩到目标体积与端侧性能预算内，并以 3D Tiles 等流式格式交付。

### 1.2 设计目标（可量化验收）

| 指标 | 目标 |
|---|---|
| 压缩率 | 3DGS 场景 ≥ 15x（对标 LightGaussian）；点云 3D Tiles ≥ 5x（对标 Draco+meshopt） |
| 质量底线 | 用户给定 PSNR/SSIM/LPIPS 阈值，流水线自动保证（不达标自动回调配方重跑） |
| 自动化程度 | 输入资产 + 一条命令 → 输出可直接在 CesiumJS 加载的 tileset，全程无需人工调参 |
| 可扩展性 | 压缩阶段均为插件，新算法（论文复现）≤ 1 个文件的接入成本 |
| 双路径支持 | 既有"场景压缩"路径，也有"任务模型压缩"路径（ONNX 结构化剪枝，借鉴 H3DNAS） |

### 1.3 非目标（明确不做）

- 不做 3DGS 训练/重建本身（用户带已训练好的资产进来）；
- 不做通用神经网络的 AutoML；
- 不做渲染引擎，只做 headless 评测渲染。

---

## 2. 总体架构

架构分六层，数据自上而下流经一条**阶段化 DAG**，右侧质量门禁形成回流闭环：

```
输入层 → 摄取与IR → 自动分析(配方规划) → 压缩引擎(DAG) → 打包与流式化 → 评测验证
                                              ↑________不达标重跑________↓
```

三层角色划分：

| 层 | 职责 | 借鉴来源 |
|---|---|---|
| 数据线 | 3DGS/点云 → GeoScene IR → 压缩 → 3D Tiles/SPZ | LightGaussian + point-tiler + 3dgs.zip |
| 模型线 | 任务模型 ONNX → 图分析 → 结构化剪枝/蒸馏/量化 → ONNX | H3DNAS（CDG 通道依赖图） |
| 横切 | CLI/API/UI、插件注册表、并行运行时、溯源元数据 | 自研 |

完整架构图见同目录 `geosqueeze_architecture.svg`。

---

## 3. 核心数据模型：GeoScene IR

一切阶段都作用于统一中间表示，避免每个插件各写一遍解析器。

```python
@dataclass
class GeoScene:
    geometry: GaussianCloud | PointCloud      # 统一为高斯或点两种基元
    spatial_index: Octree                     # 自适应八叉树（体素大小可配）
    attributes: AttributeTable                # 列式存储，零拷贝给 numpy/torch
    views: RenderViews | None                 # 训练视图（可选，供蒸馏/恢复微调）
    provenance: Provenance                    # 溯源链：来源、每阶段参数与随机种子
    stats: SceneStats                         # 分析器产出的画像缓存
```

关键点：

- **列式属性表**：position(3) / SH 系数（按阶分块 0-3）/ opacity / scale(3) / rotation(4) / color，分别独立量化，对应 3DGS 各属性量化敏感度的天然差异（位置敏感 > SH 中等 > rotation 最不敏感）。
- **八叉树即 IR 的一部分**：后续 LOD 构建、分块、并行调度都复用同一棵树，不重复建索引。
- **溯源链（Provenance）**：记录每一阶段的输入指纹、参数、随机种子，保证压缩结果可复现——GIS 行业交付对可追溯性有硬要求，这是区别于 CV 开源工具的关键设计。

### 3.1 读取器（Readers）

| 格式 | 来源 | 说明 |
|---|---|---|
| `.ply`（3DGS） | INRIA/LightGaussian | 事实标准 |
| `.spz` / `.splat` | Niantic / antimatter15 | 已压缩格式的再压缩与转瓦片 |
| LAS / LAZ / e57 / CSV | 测绘行业 | 复用 point-tiler 的 LAS 解析经验；无高斯时以"点基元"进管线 |
| NeRF checkpoint | nerfstudio 等 | 低优先级：先烘焙成高斯再进管线 |
| ONNX 模型 | 任务模型 | 走模型线，不进 GeoScene |

---

## 4. 自动分析与配方规划（Auto-Analysis → Recipe）

用户只给三样东西：**目标平台、质量底线、体积预算**。分析器负责把"压缩什么、怎么压"决策出来。

### 4.1 四个分析器

1. **属性画像（Profiler）**：高斯数量 N、SH 阶数分布、opacity/scale 直方图、颜色统计；点云则是密度、分类码、强度分布。输出"哪个属性是体积瓶颈"。
2. **空间分析（Spatial Analyzer）**：八叉树每层点数分布 → 判断 LOD 需求与瓦片切分粒度；识别离簇点（噪声/漂浮高斯，可直接清洗）。
3. **重要性打分（Significance Scorer）**：核心算法。
   - 场景基元：借用 LightGaussian 的**全局显著性分数**——修改 diff-gaussian-rasterization，在渲染采样时累计每个高斯对最终像素的贡献（贡献低 → 剪枝候选），比单看 opacity 可靠得多；
   - 点云基元：几何显著性（法向变化率、曲率、体素占用率），无训练成本；
   - 模型线：CDG 可压缩上限分析（H3DNAS 的通道依赖图，O(V+E) 算出理论压缩天花板，先告诉用户"最多能剪多少"再动手）。
4. **配方规划（Recipe Planner）**：决策树 + 资源约束求解，输出一份 YAML 配方（见 4.2）。决策树骨架：

```
目标平台?
├─ Web(浏览器)      → SH 降阶到 1 阶起 + 重剪枝(70%) + PQ 8bit + 3D Tiles 1.1
├─ 移动端/WebGPU     → SH 降阶 + 重剪枝(85%) + VQ 码本 + SPZ 兼容
├─ 桌面/服务器       → 轻剪枝(30%) + VecTree 残差量化
└─ 机载/边缘盒       → 模型线 ONNX 剪枝 + 场景线 INT8 几何量化
质量底线 ≥ 高(PSNR>32)?
└─ 是 → 追加蒸馏/恢复微调阶段，量化位宽上调
```

### 4.2 配方（Recipe）示例

```yaml
# recipe.yaml —— 由 Planner 生成，用户可手工覆写任意字段
target:
  platform: web            # web | mobile | desktop | edge
  quality_floor: { psnr: 30, ssim: 0.9 }
  size_budget_mb: 120
stages:
  - prune:
      method: significance        # significance | merge | voxel_dedup
      ratio: 0.7
      recover_ft_iters: 3000      # 剪后恢复微调
  - distill:
      method: sh_distill          # LightGaussian 式 SH 蒸馏
      iters: 5000
  - quantize:
      position: { bits: 16 }      # KHR_mesh_quantization 兼容
      sh: { method: pq, codebook: 65536, bits: 8 }
      opacity: { bits: 8 }
      rotation: { bits: 8 }
export:
  format: 3dtiles_1.1             # 3dtiles_1.1 | spz | sog | onnx
  lod: { levels: [15, 18, 21], geometric_error_scale: 0.1 }
  compression: [meshopt, gzip]
eval:
  render_views: 40
  gate: strict                    # 不达标自动回调
```

---

## 5. 压缩引擎：阶段化可插拔 DAG

引擎是一个**阶段注册表 + 顺序执行器**。每个阶段满足同一接口：

```python
class Stage(Protocol):
    name: str
    def applicable(self, scene: GeoScene, recipe: dict) -> bool: ...
    def run(self, scene: GeoScene, ctx: RunContext) -> StageResult: ...
```

阶段间自动落盘 checkpoint（`.gsq/ckpt/`），断点可续跑、单阶段可独立重跑。

### 5.1 剪枝阶段（Prune）

| 算法 | 对象 | 说明 |
|---|---|---|
| 显著性剪枝 + 恢复 | 3DGS | 全局显著性分数排序 → 剪掉 bottom-k → 光栅化器微调恢复（LightGaussian prune_finetune 思路，k 由配方的质量底线反推） |
| 合并吸收（merge） | 3DGS | 相邻、同向、近共面的高斯合并为单个大高斯，减数量不丢覆盖 |
| 体素去重 | 点云/3DGS | 每体素保留代表点，LOD 上层天然实现 |
| ONNX 结构化剪枝 | 任务模型 | 通道依赖图分类算子 → L1 重要性通道选择 → 图手术直接改 ONNX（H3DNAS 路线，无需源码） |

### 5.2 蒸馏/恢复阶段（Distill）

- **SH 蒸馏**：以未量化高分辨率模型为 Teacher，对剪枝后的 Student 做 SH 系数蒸馏 + 渲染监督（L1 + LPIPS），把"高阶 SH 的信息"压进低阶+微调参数里；
- **恢复微调（Recovery FT）**：剪枝/量化后的标准动作，迭代数由质量缺口自动决定（先小步试跑评测，不达标再加轮次）；
- **模型线蒸馏**（可选）：Teacher 大模型对 Student 的 logits 做 KL 蒸馏，服务"无人机上跑小模型"场景；
- **无训练视图时的降级策略**：用 Teacher 模型在随机轨迹上渲染生成伪视图（pseudo-views）做监督——这是本项目的工程创新点之一，解决"客户只给了 ply 没给原图"的真实交付困境。

### 5.3 量化阶段（Quantize）

| 属性 | 方法 | 默认 |
|---|---|---|
| position | 标量量化，对齐 KHR_mesh_quantization（16/12 bit） | 16 |
| SH | 乘积量化（PQ）码本 / VecTree 残差树编码 | 8 bit + 64K 码本 |
| opacity / scale | 标量 + min-max 校准 | 8 bit |
| rotation | 四元数量化（不敏感） | 8 bit |
| 几何（点云） | Draco 风格位置量化 + 熵编码，或 meshopt | 可选 |

- **混合精度**：位置 16bit + 其余 8bit 是 3D Tiles 兼容的最优默认；追求极致时 SH 走 VecTree（按重要性自适应位宽）。
- **QAT（量化感知训练）**：当 PTQ 掉点超阈值且存在视图时，可选 torch.ao QAT 路线（nerf-qat 已验证可行性）。

### 5.4 低秩阶段（LowRank，可选）

- SH 残差张量 SVD 截断：剪枝+量化后的残差往往低秩，SVD 再省 20-30%；
- 网格化表示（SOG 风格自组织高斯网格）的张量分解，服务极致压缩模式；
- 借鉴 CCNeRF 的秩截断思想：rank 即细节层级，与 LOD 天然对齐。

---

## 6. 打包与流式输出

### 6.1 LOD 构建器

- **层级八叉树高斯（Hierarchical Gaussians）**：父瓦片存粗化表示（体素去重 + SH 降阶），子瓦片存精细化高斯；渲染端由远及近渐进加载；
- **geometric_error 计算**：`GE = voxel_size × scale_factor`（沿用 point-tiler 的 zoom-level ↔ 几何误差映射，如 L15≈64m、L18≈8m），并针对高斯基元补充屏幕空间误差修正；
- 瓦片内属性走 glTF 扩展：`KHR_mesh_quantization`（位置量化）+ `EXT_meshopt_compression`（缓冲区压缩）+ 自定义扩展 `GSQ_gaussian_attributes`（SH/opacity 扩展属性，随项目提案给 3D Tiles 社区）。

### 6.2 导出器

| 格式 | 用途 | 优先级 |
|---|---|---|
| 3D Tiles 1.1（GLB） | Cesium/CesiumJS 生态，GIS 交付主格式 | P0 |
| SPZ | 移动端/Web 小体积场景 | P1 |
| SOG（自组织网格） | 极致压缩研究向 | P2 |
| ONNX（量化后任务模型） | TensorRT/NPU 部署 | P1 |

---

## 7. 评测验证与质量门禁（Eval & Gate）

流水线的最后一环，也是"自动化"承诺的兑现点：

- **Headless 渲染评测**：固定 40 个评测视位，分别渲染压缩前/后场景，算 PSNR / SSIM / LPIPS；
- **性能评测**：瓦片总体积、首帧加载时间（模拟 WebGPU 端）、峰值显存、FPS（用自带的简化光栅化器在 CPU 上估，不上真机也能横向对比）；
- **质量门禁（Gate）**：`strict` 模式下不达标自动执行"回调策略"——按 Planner 里的梯度表放宽一位（如剪枝比例 -10%、SH 位宽 +2bit）重跑受影响阶段，最多 3 轮；仍不达标则输出带告警的报告交人工决策；
- **对比报告**：HTML 报告并排渲染压缩前后截图 + 全指标表 + 每阶段体积贡献瀑布图，可直接附进交付文档。

---

## 8. 工程结构与技术选型

```
geosqueeze/
├── crates/                     # Rust：高性能无关部分（仿 point-tiler）
│   ├── gsq-core/               #   八叉树、列式属性表、瓦片几何误差
│   ├── gsq-io/                 #   LAS/LAZ/PLY/SPZ 解析（并行）
│   └── gsq-tile/               #   3D Tiles 1.1 打包、meshopt/gzip
├── python/                     # Python：分析与压缩（torch 生态）
│   ├── gsq/analyze/            #   profiler / spatial / significance / planner
│   ├── gsq/stages/             #   prune / distill / quantize / lowrank（插件）
│   ├── gsq/model/              #   ONNX 图手术剪枝（借鉴 H3DNAS CDG）
│   ├── gsq/eval/               #   headless 渲染、指标、gate 回调
│   └── gsq/cli.py              #   typer CLI
├── plugins/                    # 第三方阶段插件注册点
├── ui/                         # (P2) Web UI：配方编辑 + 报告查看
└── tests/                      # golden-set 回归：固定场景+期望指标区间
```

| 组件 | 选型 | 理由 |
|---|---|---|
| 点云/高斯 I/O 与切瓦片 | Rust + Rayon | point-tiler 已验证该栈做 8GB 级 LAS 转瓦片 4 分钟 |
| 分析/蒸馏/量化 | Python + PyTorch + 修改版 diff-gaussian-rasterization | LightGaussian 同款，改显著性累计 |
| ONNX 图手术 | onnx + 自研 CDG 模块 | H3DNAS 路线 |
| CLI / 配置 | typer + pydantic | 配方即强类型 schema |
| 并行 | 瓦片级 Rayon 多线程 + 阶段级单进程（GPU 独占） | 避免多进程抢卡 |
| 评测渲染 | 自带简化 splat 光栅化器 + CesiumJS 抽查脚本 | headless 可跑 CI |

### 8.1 CLI 使用示例

```bash
# 场景压缩：一条命令出 3D Tiles
gsq compress scene.ply --platform web --budget 120mb --floor psnr:30 -o ./out/

# 点云直转瓦片（跳过模型相关阶段）
gsq compress city.las --recipe recipes/pointcloud_web.yaml -o ./out/

# 任务模型压缩（模型线）
gsq compress-model pointnet2.onnx --target jetson-orin --max-speedup 1.5

# 只评测，不压缩（对交付件做验收）
gsq eval ./out/ --baseline scene.ply --report report.html
```

---

## 9. Roadmap

| 里程碑 | 内容 | 验收标准 |
|---|---|---|
| **M0 PoC（4 周）** | GeoScene IR + 3DGS 读取 + 显著性剪枝 + 恢复微调 + 单瓦片导出 | room 场景 10x 压缩，PSNR 降幅 < 1.5dB |
| **M1 流式（+4 周）** | 层级八叉树 LOD + 3D Tiles 1.1 + meshopt + CesiumJS 在线 demo | 城市级场景浏览器渐进加载可用 |
| **M2 量化+门禁（+4 周）** | PQ/VecTree 量化 + 评测 harness + 自动回调 gate | 全流程一条命令，报告自动生成 |
| **M3 模型线（+4 周）** | ONNX 结构化剪枝 + 点云任务模型双路径 | PointNet++ 参数 -40% 且精度损失 < 1% |
| **M4 生态（持续）** | 插件注册表、SPZ/SOG 导出、Web UI、论文级 benchmark | 3 个外部插件接入案例 |

## 10. 风险与应对

| 风险 | 应对 |
|---|---|
| 光栅化器改动（显著性累计）维护成本高 | 上游跟随 gaussian-splatting 官方仓库，改动收敛为独立 patch |
| 大场景蒸馏阶段显存爆炸 | 按八叉树分区蒸馏 + 梯度检查点；瓦片间无依赖天然可分批 |
| 3D Tiles 对高斯属性无标准扩展 | 先用自定义扩展（`GSQ_` 前缀）+ 同时提供点云降级渲染；向 OGC 社区提案 |
| "无训练视图"场景蒸溜质量不可控 | 伪视图策略 + 报告中显著标注置信度，提醒用户补充视图 |
| Rust/Python 双语言维护成本 | 边界收敛在 IR 序列化（列式二进制 + Arrow schema），两端只认这一份契约 |

---

## 附：与现有开源项目的关系

| 借鉴项目 | 拿什么 | 本项目增量 |
|---|---|---|
| LightGaussian | 显著性剪枝、SH 蒸馏、VecTree 量化 | LOD/瓦片化、自动配方、质量门禁 |
| point-tiler | LAS 解析、3D Tiles 打包、geometric error 映射 | 高斯基元支持、压缩阶段前置 |
| H3DNAS | ONNX 无源码结构化剪枝、CDG 压缩上限 | 与场景线合入同一配方体系 |
| Draco/meshopt | 几何量化与熵编码 | 仅作导出后端 |
| 3dgs.zip 综述 | 新压缩算法跟踪 | 插件机制承接论文复现 |
