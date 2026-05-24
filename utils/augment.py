# -*- coding: utf-8 -*-
"""
数据增强模块
针对白果图像特点设计的增强策略
支持训练时在线增强和离线增强
"""
import random
import math
from typing import Tuple, Optional
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as T


class GinkgoAugmentation:
    """
    白果图像专用数据增强

    针对白果特点:
    - 通常为黄白色，背景多为深色
    - 可能有裂纹纹理，需要保留细节
    - 表面反光，需要模拟不同光照
    """

    def __init__(
        self,
        size: int = 224,
        is_train: bool = True,
        color_jitter: float = 0.3,
        rotation: int = 30,
        flip_prob: float = 0.5,
        cutout_prob: float = 0.2,
        cutout_size: int = 40,
        mixup_alpha: float = 0.2,
    ):
        self.size = size
        self.is_train = is_train
        self.mixup_alpha = mixup_alpha

        if is_train:
            self.transform = T.Compose([
                T.Resize(size + 32),
                T.RandomCrop(size),
                T.RandomHorizontalFlip(p=flip_prob),
                T.RandomVerticalFlip(p=flip_prob * 0.3),
                T.RandomRotation(rotation, fillcolor=(128, 128, 128)),
                T.ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter * 0.5,
                    hue=color_jitter * 0.2,
                ),
                T.RandomGrayscale(p=0.05),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomErasing(p=cutout_prob, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            ])
        else:
            self.transform = T.Compose([
                T.Resize(size + 32),
                T.CenterCrop(size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __call__(self, image: Image.Image):
        return self.transform(image)


class CutoutTransform:
    """Cutout数据增强的PIL实现"""

    def __init__(self, size: int = 40, probability: float = 0.5):
        self.size = size
        self.probability = probability

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return img

        w, h = img.size
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)

        x1 = max(0, x - self.size // 2)
        y1 = max(0, y - self.size // 2)
        x2 = min(w, x + self.size // 2)
        y2 = min(h, y + self.size // 2)

        img = img.copy()
        pixels = img.load()
        for i in range(x1, x2):
            for j in range(y1, y2):
                pixels[i, j] = (128, 128, 128)

        return img


class RandomSolarization:
    """随机太阳化效果，模拟强光照射"""

    def __init__(self, threshold: int = 128, probability: float = 0.3):
        self.threshold = threshold
        self.probability = probability

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return img

        return img.point(lambda x: self.threshold if x < self.threshold else x)


class GinkgoOfflineAugment:
    """
    离线数据增强
    生成增强后的图片并保存到指定目录
    """

    STRATEGIES = {
        'brightness_up': lambda img: ImageEnhance.Brightness(img).enhance(1.3),
        'brightness_down': lambda img: ImageEnhance.Brightness(img).enhance(0.7),
        'contrast_up': lambda img: ImageEnhance.Contrast(img).enhance(1.3),
        'contrast_down': lambda img: ImageEnhance.Contrast(img).enhance(0.7),
        'sharpness_up': lambda img: ImageEnhance.Sharpness(img).enhance(2.0),
        'blur': lambda img: img.filter(ImageFilter.GaussianBlur(radius=2)),
        'rotate_90': lambda img: img.rotate(90, fillcolor=(128, 128, 128)),
        'rotate_180': lambda img: img.rotate(180, fillcolor=(128, 128, 128)),
        'rotate_270': lambda img: img.rotate(270, fillcolor=(128, 128, 128)),
        'flip_h': lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
        'flip_v': lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
    }

    def __init__(self, output_dir: str, strategies: Optional[list] = None):
        self.output_dir = output_dir
        self.strategies = strategies or list(self.STRATEGIES.keys())

    def augment_image(self, image_path: str, prefix: str = '') -> list:
        """
        对单张图片进行多种增强

        Returns:
            生成的文件路径列表
        """
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        img = Image.open(image_path).convert('RGB')
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_paths = []

        for strategy_name in self.strategies:
            if strategy_name in self.STRATEGIES:
                augmented = self.STRATEGIES[strategy_name](img)
                out_name = f"{prefix}{base_name}_{strategy_name}.jpg"
                out_path = os.path.join(self.output_dir, out_name)
                augmented.save(out_path, quality=95)
                output_paths.append(out_path)

        return output_paths

    def augment_directory(self, input_dir: str, output_dir: Optional[str] = None):
        """对整个目录的图片进行增强"""
        import os
        output_dir = output_dir or self.output_dir
        self.output_dir = output_dir

        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        count = 0

        for root, dirs, files in os.walk(input_dir):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in extensions:
                    img_path = os.path.join(root, fname)
                    self.augment_image(img_path)
                    count += 1

        print(f"增强完成: {count} 张图片, 每张生成 {len(self.strategies)} 个增强版本")
