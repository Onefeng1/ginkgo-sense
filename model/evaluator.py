# -*- coding: utf-8 -*-
"""
模型评估模块
提供分类模型和检测模型的完整评估流程
包括精度指标计算、混淆矩阵、ROC曲线等
"""
import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from config import MODEL_CONFIG, DATA_DIR


class ClassificationEvaluator:
    """
    分类模型评估器
    计算Accuracy, Precision, Recall, F1, 混淆矩阵等
    """

    def __init__(self, class_names: Optional[List[str]] = None, class_names_cn: Optional[List[str]] = None):
        self.class_names = class_names or MODEL_CONFIG['class_names']
        self.class_names_cn = class_names_cn or MODEL_CONFIG['class_names_cn']
        self.num_classes = len(self.class_names)
        self.reset()

    def reset(self):
        """重置评估状态"""
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []
        self.all_image_ids = []

    def update(self, predictions: List[int], labels: List[int],
               probabilities: Optional[np.ndarray] = None,
               image_ids: Optional[List[str]] = None):
        """
        更新评估数据

        Args:
            predictions: 预测类别列表
            labels: 真实标签列表
            probabilities: 预测概率矩阵 (N, num_classes)
            image_ids: 图片标识列表
        """
        self.all_predictions.extend(predictions)
        self.all_labels.extend(labels)
        if probabilities is not None:
            self.all_probabilities.extend(probabilities.tolist())
        if image_ids is not None:
            self.all_image_ids.extend(image_ids)

    def compute_metrics(self) -> Dict:
        """计算所有评估指标"""
        preds = np.array(self.all_predictions)
        labels = np.array(self.all_labels)

        assert len(preds) == len(labels), "预测和标签数量不一致"

        # 混淆矩阵
        confusion = self._confusion_matrix(preds, labels)

        # 各类别指标
        per_class = {}
        for i, (name_en, name_cn) in enumerate(zip(self.class_names, self.class_names_cn)):
            tp = confusion[i, i]
            fp = confusion[:, i].sum() - tp
            fn = confusion[i, :].sum() - tp
            tn = confusion.sum() - tp - fp - fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            per_class[name_en] = {
                'name_cn': name_cn,
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'support': int(confusion[i, :].sum()),
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
            }

        # 总体指标
        accuracy = (preds == labels).mean()
        macro_precision = np.mean([v['precision'] for v in per_class.values()])
        macro_recall = np.mean([v['recall'] for v in per_class.values()])
        macro_f1 = np.mean([v['f1'] for v in per_class.values()])

        # 加权平均
        supports = np.array([v['support'] for v in per_class.values()])
        weighted_precision = np.average([v['precision'] for v in per_class.values()], weights=supports)
        weighted_recall = np.average([v['recall'] for v in per_class.values()], weights=supports)
        weighted_f1 = np.average([v['f1'] for v in per_class.values()], weights=supports)

        return {
            'accuracy': round(float(accuracy), 4),
            'macro_precision': round(float(macro_precision), 4),
            'macro_recall': round(float(macro_recall), 4),
            'macro_f1': round(float(macro_f1), 4),
            'weighted_precision': round(float(weighted_precision), 4),
            'weighted_recall': round(float(weighted_recall), 4),
            'weighted_f1': round(float(weighted_f1), 4),
            'per_class': per_class,
            'confusion_matrix': confusion.tolist(),
            'total_samples': len(labels),
            'correct_samples': int((preds == labels).sum()),
        }

    def _confusion_matrix(self, preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """计算混淆矩阵"""
        matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for pred, label in zip(preds, labels):
            matrix[label][pred] += 1
        return matrix

    def save_report(self, output_path: str, extra_info: Optional[Dict] = None):
        """保存评估报告为JSON文件"""
        metrics = self.compute_metrics()
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_config': {
                'backbone': MODEL_CONFIG['backbone'],
                'num_classes': self.num_classes,
                'class_names': self.class_names,
            },
            'metrics': metrics,
        }
        if extra_info:
            report.update(extra_info)

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"评估报告已保存: {output_path}")
        return report

    def print_report(self):
        """打印评估报告到终端"""
        metrics = self.compute_metrics()

        print("\n" + "=" * 70)
        print("                    GinkgoSense 模型评估报告")
        print("=" * 70)
        print(f"\n  总样本数: {metrics['total_samples']}")
        print(f"  正确预测: {metrics['correct_samples']}")
        print(f"  总体准确率: {metrics['accuracy']:.2%}")

        print(f"\n  {'类别':<12} {'精确率':>8} {'召回率':>8} {'F1分数':>8} {'样本数':>8}")
        print("  " + "-" * 50)
        for name_en, info in metrics['per_class'].items():
            print(f"  {info['name_cn']:<10} {info['precision']:>8.2%} {info['recall']:>8.2%} {info['f1']:>8.2%} {info['support']:>8}")

        print("  " + "-" * 50)
        print(f"  {'宏平均':<10} {metrics['macro_precision']:>8.2%} {metrics['macro_recall']:>8.2%} {metrics['macro_f1']:>8.2%}")
        print(f"  {'加权平均':<10} {metrics['weighted_precision']:>8.2%} {metrics['weighted_recall']:>8.2%} {metrics['weighted_f1']:>8.2%}")

        print("\n  混淆矩阵:")
        print("  " + "  ".join(["  " + name[:6] for name in self.class_names]))
        for i, row in enumerate(metrics['confusion_matrix']):
            print(f"  {self.class_names_cn[i][:6]:<8}  " + "  ".join([f"{v:>6}" for v in row]))

        print("\n" + "=" * 70)


class DetectionEvaluator:
    """
    检测模型评估器
    计算mAP@0.5, mAP@0.5:0.95等检测指标
    """

    def __init__(self, iou_thresholds: Optional[List[float]] = None):
        self.iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)
        self.all_predictions = []
        self.all_ground_truths = []

    def reset(self):
        self.all_predictions = []
        self.all_ground_truths = []

    def update(self, predictions: List[Dict], ground_truths: List[Dict]):
        """
        更新评估数据

        Args:
            predictions: [{'boxes': array, 'scores': array, 'labels': array}]
            ground_truths: [{'boxes': array, 'labels': array}]
        """
        self.all_predictions.extend(predictions)
        self.all_ground_truths.extend(ground_truths)

    def compute_ap(self, pred_scores: np.ndarray, pred_boxes: np.ndarray,
                   gt_boxes: np.ndarray, iou_threshold: float) -> float:
        """计算单个类别的AP"""
        if len(pred_scores) == 0:
            return 0.0
        if len(gt_boxes) == 0:
            return 0.0

        # 按置信度排序
        sorted_indices = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sorted_indices]

        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        matched_gt = set()

        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            for j, gt_box in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = self._compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[i] = 1
                matched_gt.add(best_gt_idx)
            else:
                fp[i] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recall = tp_cumsum / len(gt_boxes)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        # 计算AP (11点插值)
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            precisions_at_recall = precision[recall >= t]
            if len(precisions_at_recall) > 0:
                ap += np.max(precisions_at_recall) / 11

        return ap

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """计算两个框的IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter / (area1 + area2 - inter + 1e-7)

    def compute_map(self) -> Dict:
        """计算mAP"""
        aps_per_threshold = {}

        for iou_thresh in self.iou_thresholds:
            class_aps = defaultdict(list)

            for pred, gt in zip(self.all_predictions, self.all_ground_truths):
                unique_labels = set(gt['labels'].tolist()) | set(pred['labels'].tolist())
                for label in unique_labels:
                    pred_mask = pred['labels'] == label
                    gt_mask = gt['labels'] == label
                    ap = self.compute_ap(
                        pred['scores'][pred_mask],
                        pred['boxes'][pred_mask],
                        gt['boxes'][gt_mask],
                        iou_thresh,
                    )
                    class_aps[label].append(ap)

            mean_ap = np.mean([np.mean(aps) for aps in class_aps.values()]) if class_aps else 0
            aps_per_threshold[f'mAP@{iou_thresh:.2f}'] = round(float(mean_ap), 4)

        return {
            'mAP@0.5': aps_per_threshold.get('mAP@0.50', 0),
            'mAP@0.5:0.95': round(float(np.mean(list(aps_per_threshold.values()))), 4),
            'per_iou_threshold': aps_per_threshold,
            'total_images': len(self.all_ground_truths),
        }


def evaluate_classifier(
    model,
    dataloader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    完整的分类模型评估流程

    Args:
        model: PyTorch模型
        dataloader: 数据加载器
        device: 计算设备
        class_names: 类别名称列表

    Returns:
        评估结果字典
    """
    evaluator = ClassificationEvaluator(class_names)

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy())

    evaluator.update(all_preds, all_labels, np.array(all_probs))
    metrics = evaluator.compute_metrics()
    evaluator.print_report()

    return metrics
