"""日志：统一写到 logs/（不入库）。"""
from __future__ import annotations

import logging
import pathlib


def get_logger(name: str, log_dir: str | pathlib.Path = "logs", level: str = "INFO") -> logging.Logger:
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.FileHandler(pathlib.Path(log_dir) / f"{name}.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
