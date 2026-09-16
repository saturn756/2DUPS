"""M5 选定实现：PIDNet-S（Cityscapes 预训练）。

机制：把双分支结构类比为比例—积分控制器，两路直接融合会在边界产生过冲；
增加第三条浅分支 D 由高频信息预测边界；Pag 用 Sigmoid 计算两路像素同类置信度并
让细节分支有选择地吸收语义，Bag 以 D 的边界注意力在边界处信任细节、其余用上下文填充。
对照实现：PP-MobileSeg、SCTNet（移动端）。
"""
from __future__ import annotations


def segment(image, class_schema, **kwargs):
    """返回 SegmentationMap。TODO(阶段五)。"""
    raise NotImplementedError
