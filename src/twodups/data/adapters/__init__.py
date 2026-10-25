"""数据源适配器：每个数据集只负责解析原始格式。"""

from .base import DatasetAdapter  # noqa: F401
from .bdd100k import BDD100KAdapter  # noqa: F401
