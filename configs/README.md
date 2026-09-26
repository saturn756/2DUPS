# 配置

各模块的配置集中在本目录，**一个模块一份**，命名 `m<编号>_<简称>.yaml`；跨模块公共项放 `common.yaml`。

M1 的 DataLoader、M2 的简单增强已开始实现；其余多数仍是骨架。
共享参数、个人实验和版本记录遵循[协作开发与复现规范](../docs/team/01_协作开发与复现规范.md)。

| 文件 | 归属 | 状态 |
|---|---|---|
| `common.yaml` | 全链路公共项：数据清单、日志、对齐键与版本策略 | 骨架 |
| `m1_calibration.yaml` | M1 输入接入与标定状态管理 | 骨架 |
| `m2_preprocess.yaml` | M2 图像预处理与质量评价 | 初版阈值，未用下游指标校准 |
| `m3_geometry.yaml` | M3 帧间几何估计 | 骨架 |
| `m4_detection_tracking.yaml` | M4 多目标检测与跟踪 | 骨架 |
| `m5_segmentation.yaml` | M5 语义分割 | 骨架 |
| `m6_topology.yaml` | M6 2D 场景结构融合 | 骨架 |
| `dataset_manifest.yaml` | BDD100K 五段样本与切分 | 已建立；文件校验见 `datasets/bdd100k-five.sha256` |

## 约定

- 一个配置文件只描述一个模块；模块之间不共享配置对象，公共项经 `common.yaml` 组合；
- 配置里出现的接口名必须与 `docs/interface/` 一致，类别语义必须引用类别模式；
- 阈值、数据集版本、实现标识（`producer_ref` / `config_ref`）必须写进配置，保证评测可复现；
- 运行日志与评测产物写到 `logs/`，不入库。
- 实验用完整配置放 `configs/experiments/<模块>/<实验名>.yaml`；变更共享默认值需评审。运行时记录配置哈希、Git 提交、环境锁、数据及权重校验和。

## 如果以后要改成模块目录

若模块数量增多、单个模块出现多份配置，可把 `m*_*.yaml` 改为 `configs/<module>/` 子目录；届时只需同步本说明与
`common.yaml` 中的引用方式，其他文档不受影响。
