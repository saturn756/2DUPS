# M2 图像预处理与质量评价

| 项 | 位置 |
|---|---|
| 职责、边界与验收维度 | [`docs/architecture/02_模块规范.md`](../../../../docs/architecture/02_模块规范.md) |
| 接口字段与不变量 | [`docs/interface/04_接口规范.md`](../../../../docs/interface/04_接口规范.md) |
| 选定方法与对照 | [`docs/methods/README.md`](../../../../docs/methods/README.md) |
| 配置 | [`configs/m2_preprocess.yaml`](../../../../configs/m2_preprocess.yaml) |

- 模块入口：`module.py`；实现文件：`quality_metrics.py`、`impl_clahe.py`、
  `impl_denoise.py`。`impl_zero_dce_tiny.py` 仍是未启用的占位文件。
- 阶段三（S2）首版：接收 M1 DataLoader 的 `LoadedFrame`（I2 元数据 + RGB 像素），
  检查尺寸、类型和坐标语义，计算亮度/曝光/对比度/模糊/噪声/色偏指标；
  按初始阈值选择 raw、条件去噪或 CLAHE-LAB + 受限 Gamma。
- 输入接口：I2；输出接口：I3。

`ImageVariantSet` 总含 `raw`，生成版本保留父版本、处理参数、原像素坐标语义和
带 SHA-256 的 PNG 引用。`QualityReport` 保存指标、质量标签、门控理由、操作状态
及耗时。生成文件写入 `outputs/m2/<配置哈希>/<序列>/`，由 `.gitignore` 排除；
内存中的 `ProcessedFrame.images_rgb` 可供同进程下一模块消费。

当前 BDD100K 无可信标定，因此不生成 `rectified`。即使上游标记 `valid`，在
标定文件格式和适用性检查完成前也只记录 `rectification_deferred`；不得把它
当作已去畸变的输出。Zero-DCE、下游指标对照、时序门控平滑均未实现；当前
阈值只供小样本冒烟测试，不代表增强一定改善检测或跟踪。

```bash
python scripts/run_m2.py --max-frames 320 --stride 100
```
