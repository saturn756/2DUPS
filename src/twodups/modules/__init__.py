"""六个运行时模块。

持久化/跨进程数据只使用 contracts；当前同进程 M1→M2 的 RGB ndarray
是显式 LoadedFrame 运行时载体，不属于可序列化接口。
每个模块的职责与验收维度见 docs/architecture/02_模块规范.md。
"""
MODULES = ("M1", "M2", "M3", "M4", "M5", "M6")
