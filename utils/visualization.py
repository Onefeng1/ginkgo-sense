# -*- coding: utf-8 -*-
"""
可视化工具模块
提供训练曲线、混淆矩阵、ROC曲线、预测结果等可视化
"""
import os
import numpy as np
from typing import Dict, List, Optional, Tuple


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = '混淆矩阵',
    normalize: bool = True,
    cmap: str = 'Blues',
):
    """
    绘制混淆矩阵热力图

    Args:
        confusion_matrix: 混淆矩阵数组
        class_names: 类别名称列表
        save_path: 保存路径
        title: 图表标题
        normalize: 是否归一化
        cmap: 颜色映射
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if normalize:
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1, keepdims=True)
    else:
        cm = confusion_matrix.astype('float')

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='真实标签',
        xlabel='预测标签',
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:.2f}' if normalize else f'{int(cm[i, j])}',
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"混淆矩阵已保存: {save_path}")
    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None,
):
    """
    绘制训练曲线（Loss和Accuracy）

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accs: 训练准确率列表
        val_accs: 验证准确率列表
        save_path: 保存路径
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss曲线
    ax1.plot(epochs, train_losses, 'b-o', label='训练损失', markersize=3)
    ax1.plot(epochs, val_losses, 'r-o', label='验证损失', markersize=3)
    ax1.set_title('损失曲线', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy曲线
    ax2.plot(epochs, train_accs, 'b-o', label='训练准确率', markersize=3)
    ax2.plot(epochs, val_accs, 'r-o', label='验证准确率', markersize=3)
    ax2.set_title('准确率曲线', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.suptitle('GinkgoSense 训练过程', fontsize=16, y=1.02)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存: {save_path}")
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
):
    """
    绘制多分类ROC曲线（One-vs-Rest）

    Args:
        y_true: 真实标签 (N,)
        y_scores: 预测概率 (N, num_classes)
        class_names: 类别名称列表
        save_path: 保存路径
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    num_classes = len(class_names)
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db', '#9b59b6']

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(num_classes):
        binary_true = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_true, y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假阳性率 (FPR)', fontsize=12)
    ax.set_ylabel('真阳性率 (TPR)', fontsize=12)
    ax.set_title('ROC 曲线 (One-vs-Rest)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC曲线已保存: {save_path}")
    plt.close()


def plot_prediction_result(
    image: 'Image.Image',
    result: Dict,
    save_path: Optional[str] = None,
):
    """
    可视化单张图片的预测结果

    Args:
        image: PIL图片
        result: 预测结果字典
        save_path: 保存路径
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 原图
    ax1.imshow(image)
    ax1.set_title('输入图片', fontsize=13)
    ax1.axis('off')

    # 概率条形图
    probs = result.get('probabilities', {})
    names = list(probs.keys())
    values = list(probs.values())
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    bars = ax2.barh(names, values, color=colors[:len(names)], height=0.5)
    ax2.set_xlim(0, 1.0)
    ax2.set_title(f"预测: {result.get('label_cn', 'N/A')} ({result.get('confidence', 0):.1%})", fontsize=13)

    for bar, val in zip(bars, values):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1%}', va='center', fontsize=11)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"预测结果已保存: {save_path}")
    plt.close()


def plot_feature_maps(
    feature_maps: np.ndarray,
    save_path: Optional[str] = None,
    max_channels: int = 16,
    title: str = '特征图可视化',
):
    """
    可视化卷积层特征图

    Args:
        feature_maps: 特征图数组 (C, H, W)
        save_path: 保存路径
        max_channels: 最多显示的通道数
        title: 图表标题
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_channels = min(feature_maps.shape[0], max_channels)
    n_cols = 4
    n_rows = (n_channels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(title, fontsize=14)

    for i in range(n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        if i < n_channels:
            axes[row, col].imshow(feature_maps[i], cmap='viridis')
            axes[row, col].set_title(f'Channel {i}', fontsize=10)
        axes[row, col].axis('off')

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"特征图已保存: {save_path}")
    plt.close()
