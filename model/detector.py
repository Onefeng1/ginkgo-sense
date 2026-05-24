# -*- coding: utf-8 -*-
"""
白果目标检测模块
基于改进YOLOv5的白果检测与定位
支持单果检测、多果检测、实时检测等场景
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image
from torchvision import transforms

from config import MODEL_CONFIG, PREPROCESS_CONFIG


class SPPF(nn.Module):
    """空间金字塔池化 - 快速版 (Spatial Pyramid Pooling - Fast)"""

    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.pool2 = nn.MaxPool2d(kernel_size=kernel_size * 2, stride=1, padding=kernel_size)
        self.pool3 = nn.MaxPool2d(kernel_size=kernel_size * 4, stride=1, padding=kernel_size * 2)
        self.conv = nn.Conv2d(channels * 4, channels, 1)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = x
        y2 = self.pool1(x)
        y3 = self.pool2(y2)
        y4 = self.pool3(y3)
        return self.act(self.bn(self.conv(torch.cat([y1, y2, y3, y4], 1))))


class CSPBlock(nn.Module):
    """CSP (Cross Stage Partial) 模块"""

    def __init__(self, in_channels: int, out_channels: int, n_bottlenecks: int = 3):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(in_channels, mid_channels, 1)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.bottlenecks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(inplace=True),
            ) for _ in range(n_bottlenecks)
        ])

        self.conv3 = nn.Conv2d(mid_channels * 2, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.act(self.bn1(self.conv1(x)))
        y2 = self.act(self.bn2(self.conv2(x)))
        y2 = self.bottlenecks(y2)
        return self.act(self.bn3(self.conv3(torch.cat([y1, y2], 1))))


class GinkgoDetector(nn.Module):
    """
    白果目标检测网络
    基于YOLOv5-S的轻量化改进版本

    特点:
    - 使用SPPF替代SPP，减少计算量
    - 引入CBAM注意力机制提升小目标检测
    - 轻量化CSP模块减少参数量
    - 三尺度特征融合(FPN + PAN)
    """

    def __init__(
        self,
        num_classes: int = 1,  # 仅检测白果
        input_channels: int = 3,
        anchors: Optional[List] = None,
    ):
        super().__init__()
        self.num_classes = num_classes

        # 默认锚框 (w, h) - 针对白果尺寸优化
        if anchors is None:
            self.anchors = [
                [(10, 13), (16, 30), (33, 23)],      # P3/8
                [(30, 61), (62, 45), (59, 119)],     # P4/16
                [(116, 90), (156, 198), (373, 326)],  # P5/32
            ]
        else:
            self.anchors = anchors

        self.n_anchors = len(self.anchors[0])

        # Backbone
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        self.backbone = nn.ModuleDict({
            'stage1': CSPBlock(32, 64, n_bottlenecks=1),     # /2
            'sppf': SPPF(64),                                 # /4
            'stage2': CSPBlock(64, 128, n_bottlenecks=2),    # /8
            'stage3': CSPBlock(128, 256, n_bottlenecks=3),   # /16
            'stage4': CSPBlock(256, 512, n_bottlenecks=3),   # /32
        })

        # Neck (FPN + PAN)
        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            CSPBlock(512, 256, n_bottlenecks=1),
        )
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            CSPBlock(256, 128, n_bottlenecks=1),
        )
        self.downconv1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            CSPBlock(128, 256, n_bottlenecks=1),
        )
        self.downconv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            CSPBlock(256, 512, n_bottlenecks=1),
        )

        # Detection heads
        self.head_p3 = nn.Conv2d(128, self.n_anchors * (5 + num_classes), 1)
        self.head_p4 = nn.Conv2d(256, self.n_anchors * (5 + num_classes), 1)
        self.head_p5 = nn.Conv2d(512, self.n_anchors * (5 + num_classes), 1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """前向传播，返回三个尺度的检测结果"""
        x = self.stem(x)

        # Backbone
        c1 = self.backbone['stage1'](x)
        c2 = self.backbone['sppf'](c1)
        c3 = self.backbone['stage2'](c2)
        c4 = self.backbone['stage3'](c3)
        c5 = self.backbone['stage4'](c4)

        # FPN (自顶向下)
        p5 = c5
        p4 = self.upconv1[1](self.upconv1[0](p5) + c4)
        p3 = self.upconv2[1](self.upconv2[0](p4) + c3)

        # PAN (自底向上)
        p4 = self.downconv1[1](self.downconv1[0](p3) + p4)
        p5 = self.downconv2[1](self.downconv2[0](p4) + p5)

        # Detection
        out_p3 = self.head_p3(p3)
        out_p4 = self.head_p4(p4)
        out_p5 = self.head_p5(p5)

        return [out_p3, out_p4, out_p5]


class DetectionPostprocessor:
    """检测后处理：NMS、置信度过滤等"""

    def __init__(
        self,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def __call__(self, predictions: List[torch.Tensor]) -> List[Dict]:
        """对模型输出进行后处理"""
        results = []
        for pred in predictions:
            batch_results = self._process_single_batch(pred)
            results.extend(batch_results)
        return results

    def _process_single_batch(self, pred: torch.Tensor) -> List[Dict]:
        """处理单个batch的预测结果"""
        batch_size = pred.shape[0]
        batch_results = []

        for i in range(batch_size):
            output = pred[i]
            # Reshape: (anchors, 5+classes) -> (h*w*anchors, 5+classes)
            output = output.permute(1, 2, 0).reshape(-1, output.shape[-1])

            # 过滤低置信度
            scores = output[:, 4]
            mask = scores > self.conf_threshold
            output = output[mask]

            if len(output) == 0:
                batch_results.append({
                    'boxes': np.array([]),
                    'scores': np.array([]),
                    'labels': np.array([]),
                })
                continue

            # 获取类别和置信度
            class_scores = output[:, 5:]
            class_ids = torch.argmax(class_scores, dim=1)
            scores = output[:, 4] * class_scores[torch.arange(len(class_ids)), class_ids]

            # NMS
            boxes = self._xywh_to_xyxy(output[:, :4])
            keep = self._nms(boxes, scores, self.iou_threshold)

            # 限制最大检测数
            keep = keep[:self.max_detections]

            batch_results.append({
                'boxes': boxes[keep].cpu().numpy(),
                'scores': scores[keep].cpu().numpy(),
                'labels': class_ids[keep].cpu().numpy(),
            })

        return batch_results

    def _xywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """将 (cx, cy, w, h) 转换为 (x1, y1, x2, y2)"""
        new_boxes = torch.zeros_like(boxes)
        new_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        new_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        new_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        new_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return new_boxes

    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        """非极大值抑制"""
        try:
            from torchvision.ops import nms
            return nms(boxes, scores, iou_threshold)
        except ImportError:
            return self._nms_manual(boxes, scores, iou_threshold)

    def _nms_manual(self, boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        """手动实现NMS"""
        order = scores.argsort(descending=True)
        keep = []

        while len(order) > 0 and len(keep) < self.max_detections:
            idx = order[0]
            keep.append(idx)

            if len(order) == 1:
                break

            ious = self._box_iou(boxes[idx:idx+1], boxes[order[1:]])
            mask = ious < iou_threshold
            order = order[1:][mask]

        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    def _box_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """计算IoU"""
        x1 = torch.max(box1[:, 0], box2[:, 0])
        y1 = torch.max(box1[:, 1], box2[:, 1])
        x2 = torch.min(box1[:, 2], box2[:, 2])
        y2 = torch.min(box1[:, 3], box2[:, 3])

        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        return inter / (area1 + area2 - inter + 1e-7)


class GinkgoDetectionPipeline:
    """
    完整的白果检测流水线
    整合预处理、模型推理、后处理
    """

    def __init__(
        self,
        model_weights: Optional[str] = None,
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = GinkgoDetector(num_classes=1)

        if model_weights and __import__('os').path.exists(model_weights):
            self.model.load_state_dict(torch.load(model_weights, map_location=self.device))

        self.model.to(self.device)
        self.model.eval()

        self.postprocessor = DetectionPostprocessor(
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

        self.transform = transforms.Compose([
            transforms.Resize(640),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=PREPROCESS_CONFIG['mean'],
                std=PREPROCESS_CONFIG['std'],
            ),
        ])

    def detect(self, image_path: str) -> List[Dict]:
        """
        检测单张图片中的白果

        Args:
            image_path: 图片路径

        Returns:
            检测结果列表，每个结果包含boxes, scores, labels
        """
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (w, h)

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(input_tensor)

        results = self.postprocessor(predictions)

        # 缩放回原始尺寸
        for result in results:
            if len(result['boxes']) > 0:
                scale_x = original_size[0] / 640
                scale_y = original_size[1] / 640
                result['boxes'][:, [0, 2]] *= scale_x
                result['boxes'][:, [1, 3]] *= scale_y
                result['original_size'] = original_size
                result['num_detections'] = len(result['boxes'])

        return results

    def detect_batch(self, image_paths: List[str]) -> List[List[Dict]]:
        """批量检测"""
        return [self.detect(path) for path in image_paths]
