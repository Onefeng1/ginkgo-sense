# -*- coding: utf-8 -*-
"""
白果图像分类模型
基于ResNet50迁移学习的白果完整性识别
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

from config import MODEL_CONFIG, PREPROCESS_CONFIG


class GinkgoClassifier:
    """白果图像分类器"""

    def __init__(self, weights=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = MODEL_CONFIG['class_names']
        self.class_names_cn = MODEL_CONFIG['class_names_cn']
        self.num_classes = MODEL_CONFIG['num_classes']

        # 构建模型
        self.model = self._build_model()

        # 加载权重
        if weights and os.path.exists(weights):
            self.model.load_state_dict(torch.load(weights, map_location=self.device))
            print(f"已加载模型权重: {weights}")

        self.model.to(self.device)
        self.model.eval()

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(PREPROCESS_CONFIG['resize']),
            transforms.CenterCrop(PREPROCESS_CONFIG['crop']),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=PREPROCESS_CONFIG['mean'],
                std=PREPROCESS_CONFIG['std']
            ),
        ])

    def _build_model(self):
        """构建ResNet50分类模型"""
        model = models.resnet50(pretrained=False)

        # 替换最后的全连接层
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes),
        )

        return model

    def predict(self, image_path):
        """
        对单张图片进行预测
        Args:
            image_path: 图片路径
        Returns:
            dict: 包含label, confidence, probabilities的结果
        """
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        idx = predicted.item()
        return {
            'label': self.class_names[idx],
            'label_cn': self.class_names_cn[idx],
            'confidence': confidence.item(),
            'probabilities': {
                self.class_names_cn[i]: round(probabilities[0][i].item(), 4)
                for i in range(self.num_classes)
            },
        }

    def predict_batch(self, image_paths):
        """
        批量预测
        Args:
            image_paths: 图片路径列表
        Returns:
            dict: {路径: 预测结果}
        """
        results = {}
        for path in image_paths:
            try:
                results[path] = self.predict(path)
            except Exception as e:
                results[path] = {'error': str(e)}
        return results
