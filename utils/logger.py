# -*- coding: utf-8 -*-
"""
日志工具模块
统一的日志记录，支持文件和控制台输出
"""
import os
import sys
import logging
from datetime import datetime
from typing import Optional


class ColorFormatter(logging.Formatter):
    """彩色日志格式器"""

    COLORS = {
        logging.DEBUG: '\033[36m',     # cyan
        logging.INFO: '\033[32m',      # green
        logging.WARNING: '\033[33m',   # yellow
        logging.ERROR: '\033[31m',     # red
        logging.CRITICAL: '\033[1;31m', # bold red
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelno, '')
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = 'ginkgo_sense',
    level: int = logging.INFO,
    log_dir: str = 'logs',
    log_to_file: bool = True,
    console_output: bool = True,
) -> logging.Logger:
    """
    配置日志系统

    Args:
        name: 日志器名称
        level: 日志级别
        log_dir: 日志文件目录
        log_to_file: 是否写入文件
        console_output: 是否输出到控制台

    Returns:
        配置好的Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # 格式
    fmt = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'

    # 控制台输出
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColorFormatter(fmt, datefmt=datefmt))
        logger.addHandler(console_handler)

    # 文件输出
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'{name}_{datetime.now():%Y%m%d}.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        logger.addHandler(file_handler)

    return logger


class TrainingLogger:
    """训练过程日志记录器"""

    def __init__(self, log_dir: str = 'logs/training'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.logger = setup_logger('training', log_dir=log_dir, console_output=False)
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  train_acc: float, val_acc: float, lr: float):
        """记录一个epoch的训练指标"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        self.history['lr'].append(lr)

        self.logger.info(
            f"Epoch {epoch:>3d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} | "
            f"LR: {lr:.2e}"
        )

    def log_config(self, config: dict):
        """记录训练配置"""
        self.logger.info("=" * 50)
        self.logger.info("训练配置:")
        for k, v in config.items():
            self.logger.info(f"  {k}: {v}")
        self.logger.info("=" * 50)

    def save_history(self, path: Optional[str] = None):
        """保存训练历史"""
        import json
        path = path or os.path.join(self.log_dir, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        self.logger.info(f"训练历史已保存: {path}")
