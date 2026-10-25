# 2DUPS · 单目 RGB 道路视频的 2D 感知原型

项目目标是把单目 RGB 道路视频整理为可复用的 2D 感知链路：数据接入、图像预处理、目标检测、语义/车道分割、多目标跟踪、系统集成，以及独立评测层。系统不输出真实距离、深度、TTC、碰撞风险、规划或控制结论。

![主架构图](docs/figures/图1-1_全新设计_2D感知系统架构.svg)

当前实现状态：

- M0：BDD100K 五段样本适配和流式解码可用；其他数据集适配器待补齐。
- M1：RGB 校验、质量评价、去噪和 CLAHE 低照度增强可用，阈值仍是初始值。
- M2：官方 YOLO11n 检测适配器可用，已在一台服务器验证真实 GPU 推理，未做精度评测。
- M3：语义/车道分割接口和配置骨架已建立，算法待接入。
- M4：最小同类 IoU tracking-by-detection 已实现，尚未进行正式跟踪评测。
- M5、E：接口和配置骨架已建立，算法和指标实现待补齐。

## 开始开发

每位成员在自己的 Linux x86_64 服务器上独立 clone 仓库、创建项目专用环境、取得数据和权重。仓库内不写服务器绝对路径，也不依赖其他成员的服务器或 Conda 环境。

```bash
git clone https://github.com/saturn756/2DUPS.git
cd 2DUPS
conda env create -f environment.yml
conda activate 2dups
python -m pip install -r requirements-lock-linux-py311.txt
python -m pip install --no-deps -e .
python -m pip check
python -m pytest -q
```

获取五段 BDD100K 视频和对应 JSON 的命令见[样本说明](docs/datasets/08_BDD100K_五段小样本.md)。数据放在 `data/raw/bdd100k/`，不进入 Git；取得后先校验：

```bash
sha256sum -c configs/datasets/bdd100k-five.sha256
python scripts/run_dataloader.py --sample-id 359ce11c-e2f58b33 --max-frames 30
python scripts/run_m1_preprocess.py --sample-id 359ce11c-e2f58b33 --max-frames 20
```

完整的五段数据接入检查：

```bash
python scripts/run_dataloader.py
python scripts/run_m1_preprocess.py --max-frames 320 --stride 100
```

## M0 → M1 → M2 检测冒烟链路

M2 配置固定了官方 YOLO11n 权重的仓库相对路径和 SHA-256。权重不提交 Git；每台服务器按[协作规范](docs/team/01_协作开发与复现规范.md)分别取得并核对。

```bash
python scripts/run_m2_detection.py --sample-id 359ce11c-e2f58b33 --max-frames 3
python scripts/run_m2_detection.py --sample-id 359ce11c-e2f58b33 --start-frame 302 --max-frames 1 --json
```

第二条命令输出 I5 JSON，用于检查真实检测框和接口对齐，不计算准确率。`python scripts/run_pipeline.py --module-config configs/m2_detection.yaml --dry-run` 只解析配置。

## 维护入口

| 内容 | 权威位置 |
|---|---|
| 协作者进入项目后的操作和检查 | [AGENTS.md](AGENTS.md) |
| M0–M5 主链路、模块职责和降级 | [模块规范](docs/architecture/02_模块规范.md) |
| I1–I9 字段、JSON 示例和对齐规则 | [接口规范](docs/interface/04_接口规范.md) + `src/twodups/contracts/` |
| 环境、数据、checkpoint 和协作约定 | [协作与复现规范](docs/team/01_协作开发与复现规范.md) |
| 当前 BDD100K 样本与下载方法 | [BDD100K 五段样本](docs/datasets/08_BDD100K_五段小样本.md) |
| 当前模块参数 | `configs/` |

`docs/reference/` 是历史调研，不作为当前实现要求。数据、权重、日志和输出均放在被 Git 忽略的目录中。

许可证见 [LICENSE](LICENSE)。
