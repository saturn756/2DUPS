# 2DUPS · 单模态 2D 感知系统

在公开道路 RGB 图像 / 视频上离线复现单模态 2D 感知流程的课程项目仓库。

![图 1-1 全新设计的 2D 感知系统架构](docs/figures/图1-1_全新设计_2D感知系统架构.svg)

数据流：`输入 → M1 输入接入与标定状态管理 → M2 输入规范化与质量评价 →｛M3 帧间几何 ∥ M4 检测跟踪 ∥ M5 语义分割｝→ M6 2D 场景结构融合 → SceneGraph2D`

## 文档

| 文档 | 内容 |
|---|---|
| [系统架构](docs/architecture/01_系统架构.md) | 目标与范围、架构风格、六个模块的职责 |
| [模块规范](docs/architecture/02_模块规范.md) | 每个模块的职责、边界、输入输出、约束与验收维度 |
| [数据流与连接逻辑](docs/dataflow/03_数据流与连接逻辑.md) | 模块之间怎么连、缺了会怎样（图 1-2） |
| [接口规范](docs/interface/04_接口规范.md) | 9 个接口的字段、不变量与降级语义 |
| [数据集与格式调研](docs/datasets/06_数据集与格式调研.md) | 候选数据集、标定信息和原始格式，当前不下载 |
| [四类数据集格式清单](docs/datasets/07_四类数据集格式清单.md) | BDD100K、KITTI、Cityscapes、nuScenes 的具体格式示例 |
| [BDD100K 五段小样本](docs/datasets/08_BDD100K_五段小样本.md) | 第一轮视频和标签选择、路径、下载方式与能力边界 |
| [实施规划](docs/architecture/05_实施规划.md) | 填充顺序、选型流程、评测体系与待填清单 |
| [方法](docs/methods/README.md) | 各模块方法说明与候选对比 |
| [图与资产](docs/figures/README.md) | 图 1-1、图 1-2（PNG + SVG） |

文档总索引见 [docs/README.md](docs/README.md)。

## 目录

```
configs/            各模块配置（一模块一份 + common.yaml）
docs/               架构、数据流、接口、方法、图
src/twodups/        代码包
scripts/            命令行入口
tests/              测试
notebooks/          实验与可视化
logs/               运行日志（不入库）
outputs/            链路输出与评测记录（不入库）
data/               数据 raw / interim / processed（不入库）
requirements.txt    依赖
requirements-dev.txt  开发与测试依赖（含项目可编辑安装）
environment.yml     项目专用 Conda 环境规格
pyproject.toml      包与工具配置
```

## 代码结构

```
src/twodups/
├── contracts/    接口契约 I1–I9：字段、状态枚举与大对象引用
├── modules/      六个模块，每个含 module.py（入口）与 impl_*.py（选定 / 对照实现）
├── pipeline/     编排：registry（实现注册表）+ runner（按边串联）
├── evaluation/   数据切片、模块指标与 EvaluationRecord
├── data/         数据清单（I1）读取、BDD100K 视频流式解码与 I2 元数据
└── utils/        配置加载与日志
```

约定：模块之间只通过 `contracts` 中的接口通信；换实现不改接口，靠配置里的 `impl` 切换。

## 开发

在服务器 `zsf` 账号下使用项目专用环境，不借用其他项目的 `demo` 等环境：

```bash
cd /home/zsf/2DUPS
/opt/anaconda3/bin/conda env create -f environment.yml -p /home/zsf/.conda/envs/2dups
/home/zsf/.conda/envs/2dups/bin/python -m pip install -r requirements-dev.txt
```

环境已创建后，直接使用专用 Python 即可；若要在交互式 Shell 中激活，先执行
`source /opt/anaconda3/etc/profile.d/conda.sh`，再执行
`conda activate /home/zsf/.conda/envs/2dups`。

```bash
cd /home/zsf/2DUPS
python -m pytest -q                                                  # 激活环境后运行

python scripts/check_contracts.py                                   # 打印 9 个接口与字段
python scripts/run_dataloader.py --sample-id 359ce11c-e2f58b33 --max-frames 30
python scripts/run_dataloader.py                                    # 检查 5 段视频和 10 秒标签
python scripts/run_m2.py --sample-id 359ce11c-e2f58b33 --max-frames 20
python scripts/run_m2.py --max-frames 320 --stride 100             # 五段视频抽帧冒烟测试
python scripts/run_pipeline.py --module-config configs/m5_segmentation.yaml --dry-run
python scripts/evaluate.py --list-slices
pytest                                                              # 契约与配置冒烟测试
```

## 状态

阶段一（文献调研与系统架构设计）已完成。M1 的第一步 BDD100K DataLoader、
M2 的首版质量评价与简单增强已有独立入口；完整 M1 标定管理及 M3–M6 运行链路
仍待接通。M2 尚未启用去畸变，阈值也未通过下游指标校准。填充顺序见
[实施规划](docs/architecture/05_实施规划.md)。

许可：Apache License 2.0，见 [LICENSE](LICENSE)。
