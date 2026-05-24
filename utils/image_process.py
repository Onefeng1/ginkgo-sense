# -*- coding: utf-8 -*-
"""
图像预处理工具
"""
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


def preprocess_image(image_path, target_size=(224, 224)):
    """标准预处理"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.BILINEAR)
    return img


def enhance_image(image_path):
    """图像增强 - 提升对比度和清晰度"""
    img = Image.open(image_path).convert('RGB')

    # 提升对比度
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)

    # 提升锐度
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.3)

    return img


def segment_ginkgo(image_path):
    """
    简单的白果分割（基于颜色阈值）
    返回：二值掩码
    """
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)

    # 白果通常是浅黄色/白色，基于HSV空间分割
    r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]

    # 简单的颜色阈值：高亮度、偏黄色
    mask = (
        (r > 150) & (g > 130) & (b > 100) &
        (r > b) & ((r.astype(int) - b.astype(int)) > 20)
    )

    return Image.fromarray(mask.astype(np.uint8) * 255)


def image_to_base64(image):
    """PIL Image转base64字符串"""
    import io, base64
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=85)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
