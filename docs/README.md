# 文档索引

本目录只放设计与方法类文档：架构、数据流、接口、方法，以及图与资产。

| 目录 | 内容 | 入口 |
|---|---|---|
| `architecture/` | 总体架构、模块规范、实施规划 | [01 系统架构](architecture/01_系统架构.md) |
| `dataflow/` | 模块之间的连接、顺序、对齐与失败传播 | [03 数据流与连接逻辑](dataflow/03_数据流与连接逻辑.md) |
| `interface/` | 9 个接口的字段、不变量与降级语义 | [04 接口规范](interface/04_接口规范.md) |
| `datasets/` | 候选数据集、标定信息与原始格式调研 | [06 数据集与格式调研](datasets/06_数据集与格式调研.md) |
| `datasets/` | 四类候选数据集的目录、标签和标定字段清单 | [07 四类数据集格式清单](datasets/07_四类数据集格式清单.md) |
| `methods/` | 各模块方法说明与候选对比 | [方法索引](methods/README.md) |
| `figures/` | 图 1-1、图 1-2（PNG + SVG） | [图与资产](figures/README.md) |

## 阅读顺序

1. 图 1-1（`figures/`）→ 2. [系统架构](architecture/01_系统架构.md) → 3. [数据流与连接逻辑](dataflow/03_数据流与连接逻辑.md)
   → 4. [模块规范](architecture/02_模块规范.md) / [接口规范](interface/04_接口规范.md) → 5. [数据集与格式调研](datasets/06_数据集与格式调研.md) → 6. [实施规划](architecture/05_实施规划.md)

## 改动顺序

| 改了什么 | 先改哪里 |
|---|---|
| 模块职责、边界 | `architecture/01_系统架构.md`、`architecture/02_模块规范.md` |
| 模块之间的连接 | `dataflow/03_数据流与连接逻辑.md` |
| 接口字段、不变量 | `interface/04_接口规范.md` |

图与下游材料在上表对应文档改完后同步。
