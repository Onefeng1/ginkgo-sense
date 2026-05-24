# -*- coding: utf-8 -*-
"""
GradCAM可视化模块
生成类激活映射图，展示模型关注区域
"""
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Optional, List

from config import PREPROCESS_CONFIG


class GradCAM:
    """
    GradCAM - 梯度加权类激活映射

    通过分析模型梯度，生成热力图展示模型决策依据
    对于白果检测场景，可以直观看到模型关注的是白果的哪些区域
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册钩子
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        生成GradCAM热力图

        Args:
            input_tensor: 输入张量 (1, C, H, W)
            target_class: 目标类别，None则使用预测类别

        Returns:
            热力图数组 (H, W)，值范围0-1
        """
        self.model.eval()

        # 前向传播
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # 计算权重
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = torch.relu(cam)

        # 归一化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # 缩放到输入尺寸
        cam = torch.nn.functional.interpolate(
            cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False
        )

        return cam[0, 0].cpu().numpy()

    def overlay(
        self,
        image: Image.Image,
        cam: np.ndarray,
        alpha: float = 0.4,
    ) -> Image.Image:
        """
        将热力图叠加到原图上

        Args:
            image: 原始PIL图片
            cam: 热力图数组
            alpha: 透明度

        Returns:
            叠加后的PIL图片
        """
        import matplotlib.cm as cm

        # 缩放热力图到图片尺寸
        h, w = image.size[1], image.size[0]
        cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR))
        cam_resized = cam_resized.astype(np.float32) / 255.0

        # 应用colormap
        colormap = cm.get_cmap('jet')
        cam_color = colormap(cam_resized)[:, :, :3]
        cam_color = (cam_color * 255).astype(np.uint8)
        cam_image = Image.fromarray(cam_color)

        # 叠加
        overlay = Image.blend(image.convert('RGB'), cam_image, alpha)
        return overlay


def apply_gradcam(
    model: nn.Module,
    image_path: str,
    output_path: str,
    target_class: Optional[int] = None,
    layer_name: str = 'layer4',
):
    """
    对单张图片应用GradCAM并保存结果

    Args:
        model: PyTorch模型
        image_path: 输入图片路径
        output_path: 输出图片路径
        target_class: 目标类别
        layer_name: 目标层名称
    """
    from torchvision import transforms

    # 加载图片
    image = Image.open(image_path).convert('RGB')

    # 预处理
    transform = transforms.Compose([
        transforms.Resize(PREPROCESS_CONFIG['resize']),
        transforms.CenterCrop(PREPROCESS_CONFIG['crop']),
        transforms.ToTensor(),
        transforms.Normalize(mean=PREPROCESS_CONFIG['mean'], std=PREPROCESS_CONFIG['std']),
    ])
    input_tensor = transform(image).unsqueeze(0)

    # 获取目标层
    target_layer = getattr(model, layer_name, None)
    if target_layer is None:
        raise ValueError(f"模型中未找到层: {layer_name}")

    # 生成GradCAM
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate(input_tensor, target_class)
    overlay = grad_cam.overlay(image, cam)

    # 保存
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    overlay.save(output_path, quality=95)
    print(f"GradCAM已保存: {output_path}")
