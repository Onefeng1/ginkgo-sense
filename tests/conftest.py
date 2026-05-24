# -*- coding: utf-8 -*-
"""
Pytest测试配置
提供通用的测试fixtures
"""
import os
import sys
import pytest
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from unittest.mock import MagicMock

# 添加项目根目录到path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def sample_image():
    """生成一张随机测试图片"""
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    return img


@pytest.fixture
def sample_image_file(tmp_path):
    """生成一张临时测试图片文件"""
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    path = tmp_path / "test_image.jpg"
    img.save(str(path))
    return str(path)


@pytest.fixture
def sample_batch():
    """生成一个batch的随机输入"""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def device():
    """获取计算设备"""
    return torch.device('cpu')


@pytest.fixture
def mock_model():
    """模拟的分类模型"""
    model = MagicMock()
    model.eval.return_value = model
    model.return_value = torch.randn(1, 3)
    return model


@pytest.fixture
def sample_prediction():
    """模拟的预测结果"""
    return {
        'label': 'intact',
        'label_cn': '完整果',
        'confidence': 0.95,
        'probabilities': {
            '完整果': 0.95,
            '轻微裂纹': 0.03,
            '严重破损': 0.02,
        },
        'elapsed_ms': 12.5,
    }
