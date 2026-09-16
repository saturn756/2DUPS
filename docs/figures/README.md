# 图 1-1 单模态 2D 感知系统六大模块架构图

| 文件 | 说明 |
|---|---|
| `fig1-1-2d-perception-architecture.svg` | 矢量源文件（可用 Illustrator / Inkscape / 浏览器打开编辑） |
| `fig1-1-2d-perception-architecture.png` | 位图，5067 × 2373，可直接插入报告与 PPT |

## 六个模块与选定模型

| 模块 | 课程阶段 | 对应经典环节 | 选定模型 | 输出接口 |
|---|---|---|---|---|
| M1 道路图像采集与相机标定 | 阶段二 | §4.1 传感硬件与标定 | OpenCV Zhang 平面标定 + cv2.undistort | DatasetManifest、CalibratedFrame |
| M2 图像预处理与质量评价 | 阶段三 | 感知前端预处理 | Zero-DCE Tiny + CLAHE | ImageVariantSet、QualityReport |
| M3 特征检测与局部特征匹配 | 阶段四 | §4.2.3 定位与自运动估计 | SIFT + FLANN + RANSAC | MatchResult |
| M4 2D 目标检测 | 阶段四 | §4.2.1 目标检测 | YOLOv12-N | DetectionSet |
| M5 多目标跟踪 | 阶段四–五 | 多目标跟踪 | ByteTrack + UCMCTrack | TrackSet |
| M6 2D 语义分割与场景拓扑推理 | 阶段五 | §4.2.2 语义分割 + §4.3 场景表示 | PIDNet-S + 规则化 2D 场景图 | SegmentationMap、SceneGraph2D |
| 验证层 | 阶段六 | Safety Monitor | 切片评测 + 失败标签 + 降级策略 | EvaluationRecord |

数据流：`输入 → M1 → M2 →｛M3 ∥ M4｝→ M5 → M6 → 结构化场景描述`；阶段六为环绕六个模块的验证层，按数据切片汇总指标、延迟与失败标签，并把质量门控结论回注 M2。

## 几条约定

- 实线为运行时数据流，虚线为可选几何依赖（M3 的单应矩阵用于 M5 相机运动补偿与 M6 几何证据）与评测依赖。
- 模块之间只传字段级结构化契约，不传高维特征张量；接口字段与不变量见阶段一《架构决策文档》第六节。
- 裁剪说明：规划、行为仲裁与运动控制不在课程范围内；经典架构中的定位建图（SLAM）由 M1 标定与 M3 几何估计承担。

## 依据

经典架构划分参照：Grigorescu S, Trasnea B, Cocias T, Macesanu G. *A Survey of Deep Learning Techniques for Autonomous Driving*. Journal of Field Robotics, 2020, 37(3): 362-386（其 Fig.1(a) 模块化感知-规划-动作流水线与 §4 感知与定位小节）。

## 修订记录

| 日期 | 说明 |
|---|---|
| 2026-09-15 | 初版：按经典模块化流水线确定六个模块、模型选型与数据流，用于阶段一评审 |

> 本图后续可能随实现与实验结论调整；修改时请同步更新本说明与仓库 README 中的引用。
