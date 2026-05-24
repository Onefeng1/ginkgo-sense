# -*- coding: utf-8 -*-
"""
GinkgoSense 配置文件
"""
import os

# 基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 模型配置
MODEL_CONFIG = {
    'backbone': 'resnet50',
    'num_classes': 3,
    'input_size': 224,
    'class_names': ['intact', 'cracked', 'broken'],
    'class_names_cn': ['完整果', '轻微裂纹', '严重破损'],
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': 50,
    'weight_decay': 1e-4,
    'lr_scheduler': 'cosine',
    'optimizer': 'adam',
    'early_stopping': 10,
}

# Web应用配置
WEB_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'max_upload_size': 16 * 1024 * 1024,  # 16MB
    'allowed_extensions': {'png', 'jpg', 'jpeg', 'bmp', 'webp'},
}

# 预处理配置
PREPROCESS_CONFIG = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'resize': 256,
    'crop': 224,
}
