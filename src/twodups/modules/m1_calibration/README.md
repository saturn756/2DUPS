# M1 输入接入与标定状态管理

| 项 | 位置 |
|---|---|
| 职责、边界与验收维度 | [`docs/architecture/02_模块规范.md`](../../../../docs/architecture/02_模块规范.md) |
| 接口字段与不变量 | [`docs/interface/04_接口规范.md`](../../../../docs/interface/04_接口规范.md) |
| 选定方法与对照 | [`docs/methods/README.md`](../../../../docs/methods/README.md) |
| 配置 | [`configs/m1_calibration.yaml`](../../../../configs/m1_calibration.yaml) |

- 模块入口：`module.py`；实现文件：见本目录 `impl_*.py`。
- 阶段二（S1）第一步：`src/twodups/data/bdd100k_loader.py` 已提供 BDD100K
  `DatasetManifest → LoadedFrame(I2 元数据 + RGB 像素 + 稀疏标注)` 的流式入口。
  `module.py` 的通用 M1 编排与外部标定参数接入尚未实现；当前请使用
  `scripts/run_dataloader.py` 检查数据。
- 输入接口：外部输入；输出接口：I2, I1。

DataLoader 逐视频读取 MOV；根据 display matrix 将编码帧旋转为显示方向，
输出 RGB `uint8`，不缓存全视频、不保存逐帧图片。`media_ref` 指向原 MOV
和帧序号，像素仅在内存中随 `LoadedFrame.image_rgb` 传递。
源 PTS 保存在 `source_timestamp`；若重复、回退或缺失，运行时间戳
`timestamp` 以最小 1 ms 步长补齐为严格递增（原始 PTS 缺失时用帧号/平均帧率），
标注以该时间轴匹配最近帧，
误差必须不大于 `max(0.05 秒, 1.5/fps)`。截断读取时尚未到达的标签不会报错。
只有含 `box2d` 的原始对象会转换为检测框；原始标注中的 `id` 只作为单帧
`local_id`，不能用作跨帧跟踪 ID。
这批视频没有已验证相机参数，因此逐帧标定状态均为 `missing`。
