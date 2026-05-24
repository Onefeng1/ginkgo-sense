# -*- coding: utf-8 -*-
"""
评估指标计算模块
提供分类和检测任务的各种指标
"""
import numpy as np
from typing import List, Dict, Optional


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算准确率"""
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict:
    """计算每个类别的Precision, Recall, F1"""
    results = {}
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        results[c] = {'precision': p, 'recall': r, 'f1': f1, 'support': int(np.sum(y_true == c))}
    return results


def top_k_accuracy(y_true: np.ndarray, y_scores: np.ndarray, k: int = 3) -> float:
    """计算Top-K准确率"""
    top_k_preds = np.argsort(-y_scores, axis=1)[:, :k]
    correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
    return float(np.mean(correct))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """计算混淆矩阵"""
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """计算两个框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-7)


def compute_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
) -> float:
    """
    计算mAP@IoU

    Args:
        predictions: [{'boxes': array(N,4), 'scores': array(N)}]
        ground_truths: [{'boxes': array(M,4)}]
        iou_threshold: IoU阈值

    Returns:
        mAP值
    """
    aps = []
    for pred, gt in zip(predictions, ground_truths):
        if len(gt['boxes']) == 0:
            continue
        if len(pred['scores']) == 0:
            aps.append(0.0)
            continue

        sorted_idx = np.argsort(-pred['scores'])
        sorted_boxes = pred['boxes'][sorted_idx]

        tp = np.zeros(len(sorted_boxes))
        matched = set()

        for i, pbox in enumerate(sorted_boxes):
            best_iou = 0
            best_j = -1
            for j, gbox in enumerate(gt['boxes']):
                if j in matched:
                    continue
                cur_iou = iou(pbox, gbox)
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_j = j

            if best_iou >= iou_threshold and best_j >= 0:
                tp[i] = 1
                matched.add(best_j)

        tp_cumsum = np.cumsum(tp)
        precision = tp_cumsum / np.arange(1, len(tp) + 1)
        recall = tp_cumsum / len(gt['boxes'])

        # AP = area under precision-recall curve (11-point interpolation)
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            p_at_r = precision[recall >= t]
            if len(p_at_r) > 0:
                ap += np.max(p_at_r) / 11
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0
