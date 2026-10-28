# 2DUPS · 单目 RGB 道路视频的 2D 感知原型

项目目标是把单目 RGB 道路视频整理为可复用的 2D 感知链路：数据接入、图像预处理、目标检测、语义/车道分割、多目标跟踪、系统集成，以及独立评测层。系统不输出真实距离、深度、TTC、碰撞风险、规划或控制结论。

![主架构图](docs/figures/图1-1_全新设计_2D感知系统架构.svg)

当前实现状态：

- M0：BDD100K 五段视频流式解码、KITTI Road 单图与标定接入可用；KITTI 当前只通过合成数据测试，其他数据集适配器待补齐。
- M1：RGB 校验、质量评价、去噪和 CLAHE 低照度增强可用，阈值仍是初始值。
- M2：官方 YOLO11n 检测适配器可用，已在一台服务器验证真实 GPU 推理，未做精度评测。
- M3：语义/车道分割接口和配置骨架已建立，算法待接入。
- M4：已接入 Ultralytics ByteTrack，按类别独立关联并输出 I6，支持 ID、丢失与恢复；尚未进行正式跟踪评测。
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

## 检测与 ByteTrack 跟踪

```bash
python scripts/run_m4_tracking.py --sample-id 359ce11c-e2f58b33 --max-frames 300 --output outputs/tracking/coco-demo
python scripts/run_m4_tracking.py --sample-id 359ce11c-e2f58b33 --detection-config configs/m2_detection_bdd100k.yaml --max-frames 300 --output outputs/tracking/bdd-demo
```

输出目录必须是仓库内的新目录，包含 `tracking.mp4`、I5/I6 JSONL 和有效配置。ByteTrack 不需要额外权重，使用锁定的 Ultralytics 8.3.184 与 `lap==0.5.12`。跟踪入口将 M2 置信度阈值设为 `track_low_thresh=0.10` 并保存有效配置，以保留第二阶段关联所需的低分框；默认 0.25 以上的检测才能建立新轨迹。原有检测入口及 M0/M1 不变。

BDD100K 微调权重按 `configs/m2_detection_bdd100k.yaml` 的路径与哈希单独取得：

```bash
mkdir -p checkpoints/m2_detection/yolo11n-bdd100k/v1
curl -fL https://huggingface.co/dronefreak/bdd100k-yolo11n/resolve/2b5554133522f96f50ce87a4df0a2de556e4b7ad/best.pt -o checkpoints/m2_detection/yolo11n-bdd100k/v1/best.pt
sha256sum checkpoints/m2_detection/yolo11n-bdd100k/v1/best.pt
```

M4 必须逐帧调用，运行时 `frame_index` 来自 I2；跳帧或倒序会被拒绝。序列切换、图像版本、坐标、尺寸或类别模式改变会重置关联。丢失轨迹仅输出状态和历史检测引用，不绘制旧框。当前不使用几何补偿或 ReID，KITTI Road 独立图片不能用于跟踪验证。

## KITTI Road 图片输入

按 KITTI 原始目录准备 `data/raw/kitti_road/training/` 下的 `image_2/`、`gt_image_2/` 和 `calib/`，再运行：

```bash
python scripts/make_kitti_road_manifest.py
python scripts/run_m1_preprocess.py --manifest data/raw/kitti_road/manifest.yaml --sample-id um_000000 --max-frames 1
python scripts/run_m2_detection.py --manifest data/raw/kitti_road/manifest.yaml --sample-id um_000000 --max-frames 1
```

每张图片独立成样本，不提供视频时序或跟踪真值。检测仍用当前 M2 权重，不读取道路真值掩码。`evaluation/kitti_road.py` 仅提供本地图像平面的二值指标辅助，不等同于官方 BEV 评测或完整 E 层。通用自采视频入口尚未实现。

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
