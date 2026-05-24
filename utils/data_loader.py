# -*- coding: utf-8 -*-
"""
数据加载工具
"""
import os
from pathlib import Path
from PIL import Image


def load_images(directory, extensions=None):
    """加载目录下所有图片"""
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    images = []
    dir_path = Path(directory)

    for ext in extensions:
        for img_path in dir_path.glob(f'*{ext}'):
            images.append(str(img_path))

    return sorted(images)


def get_dataset_info(data_dir):
    """获取数据集统计信息"""
    info = {}
    data_path = Path(data_dir)

    for split in ['train', 'val', 'test']:
        split_path = data_path / split
        if split_path.exists():
            classes = {}
            for cls_dir in split_path.iterdir():
                if cls_dir.is_dir():
                    count = len(list(cls_dir.glob('*.jpg'))) + len(list(cls_dir.glob('*.png')))
                    classes[cls_dir.name] = count
            info[split] = classes

    return info
