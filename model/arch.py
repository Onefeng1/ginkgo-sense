# -*- coding: utf-8 -*-
"""
模型架构定义模块
定义GinkgoSense中使用的所有神经网络架构
支持ResNet, EfficientNet, MobileNet等主流backbone
"""
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, List, Dict


class ChannelAttention(nn.Module):
    """通道注意力模块 (Squeeze-and-Excitation)"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1, 1) * x


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x


class CBAM(nn.Module):
    """卷积块注意力模块 (Convolutional Block Attention Module)"""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class GinkgoResNet(nn.Module):
    """
    基于ResNet的白果分类网络
    支持ResNet18/34/50/101/152，集成CBAM注意力机制
    """

    RESNET_VERSIONS = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
    }

    def __init__(
        self,
        num_classes: int = 3,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        dropout: float = 0.3,
        use_cbam: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.use_cbam = use_cbam

        # 加载预训练backbone
        if backbone not in self.RESNET_VERSIONS:
            raise ValueError(f"不支持的backbone: {backbone}, 可选: {list(self.RESNET_VERSIONS.keys())}")

        weights = 'IMAGENET1K_V1' if pretrained else None
        base_model = self.RESNET_VERSIONS[backbone](weights=weights)

        # 拆分网络结构
        self.stem = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
        )
        self.layer1 = base_model.layer1  # 256 channels (resnet50)
        self.layer2 = base_model.layer2  # 512 channels
        self.layer3 = base_model.layer3  # 1024 channels
        self.layer4 = base_model.layer4  # 2048 channels

        # 获取各层通道数
        self.channels = self._get_channels(backbone)

        # CBAM注意力模块
        if use_cbam:
            self.cbam3 = CBAM(self.channels[2])
            self.cbam4 = self.cbam3  # 共享参数减少开销

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类头
        in_features = self.channels[3]
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

        # 权重初始化
        self._init_weights()

    def _get_channels(self, backbone: str) -> List[int]:
        """获取各层输出通道数"""
        channel_map = {
            'resnet18': [64, 64, 128, 256, 512],
            'resnet34': [64, 64, 128, 256, 512],
            'resnet50': [64, 256, 512, 1024, 2048],
            'resnet101': [64, 256, 512, 1024, 2048],
            'resnet152': [64, 256, 512, 1024, 2048],
        }
        return channel_map[backbone]

    def _init_weights(self):
        """初始化分类头权重"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.use_cbam:
            x = self.cbam3(x)
        x = self.layer4(x)
        if self.use_cbam:
            x = self.cbam4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取中间特征图（用于可视化）"""
        features = {}
        x = self.stem(x)
        features['stem'] = x
        x = self.layer1(x)
        features['layer1'] = x
        x = self.layer2(x)
        features['layer2'] = x
        x = self.layer3(x)
        features['layer3'] = x
        x = self.layer4(x)
        features['layer4'] = x
        return features

    def count_parameters(self) -> Dict[str, int]:
        """统计模型参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
        }


class GinkgoEfficientNet(nn.Module):
    """
    基于EfficientNet的白果分类网络
    适用于资源受限场景下的轻量化部署
    """

    EFFICIENTNET_VERSIONS = {
        'efficientnet_b0': models.efficientnet_b0,
        'efficientnet_b1': models.efficientnet_b1,
        'efficientnet_b2': models.efficientnet_b2,
    }

    def __init__(
        self,
        num_classes: int = 3,
        backbone: str = 'efficientnet_b0',
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        if backbone not in self.EFFICIENTNET_VERSIONS:
            raise ValueError(f"不支持的backbone: {backbone}")

        weights = 'IMAGENET1K_V1' if pretrained else None
        base_model = self.EFFICIENTNET_VERSIONS[backbone](weights=weights)

        # 移除原始分类器
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 获取特征维度
        feature_dim = base_model.classifier[1].in_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_model(config: dict) -> nn.Module:
    """
    根据配置构建模型的工厂函数

    Args:
        config: 模型配置字典，包含backbone, num_classes等字段

    Returns:
        构建好的PyTorch模型
    """
    model_type = config.get('model_type', 'resnet')
    num_classes = config.get('num_classes', 3)
    backbone = config.get('backbone', 'resnet50')
    pretrained = config.get('pretrained', True)
    dropout = config.get('dropout', 0.3)

    if model_type == 'resnet':
        use_cbam = config.get('use_cbam', True)
        return GinkgoResNet(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            dropout=dropout,
            use_cbam=use_cbam,
        )
    elif model_type == 'efficientnet':
        return GinkgoEfficientNet(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            dropout=dropout,
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
