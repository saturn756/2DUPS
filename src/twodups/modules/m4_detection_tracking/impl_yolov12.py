"""M4 检测子块：YOLOv12-N。

机制：区域注意力把特征图按竖直/水平等分为 4 段（只用一次 reshape，避免 window partition），
R-ELAN 引入贯穿 block 的残差 shortcut 缓解梯度阻断，改用 FlashAttention 并去掉位置编码。
同族 N/S/M/L/X 尺度对照后选定 N：精度增益远慢于时延与参数增长。
"""
from __future__ import annotations


def detect(image, **kwargs):
    """返回 DetectionSet。TODO(阶段四)。"""
    raise NotImplementedError
