# -*- coding: utf-8 -*-
"""
模型评估脚本
独立运行的模型评估工具，支持生成完整报告
"""
import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import MODEL_CONFIG, PREPROCESS_CONFIG, DATA_DIR
from model.classifier import GinkgoClassifier
from model.evaluator import ClassificationEvaluator, evaluate_classifier
from utils.visualization import plot_confusion_matrix, plot_roc_curve, plot_training_curves


def main():
    parser = argparse.ArgumentParser(description='GinkgoSense 模型评估')
    parser.add_argument('--data_dir', type=str, default=os.path.join(DATA_DIR, 'test'), help='测试数据目录')
    parser.add_argument('--weights', type=str, default='weights/best_model.pth', help='模型权重路径')
    parser.add_argument('--output_dir', type=str, default='results/evaluation', help='评估结果输出目录')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default=None, help='计算设备')
    args = parser.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"使用设备: {device}")

    # 数据加载
    transform = transforms.Compose([
        transforms.Resize(PREPROCESS_CONFIG['resize']),
        transforms.CenterCrop(PREPROCESS_CONFIG['crop']),
        transforms.ToTensor(),
        transforms.Normalize(mean=PREPROCESS_CONFIG['mean'], std=PREPROCESS_CONFIG['std']),
    ])

    if os.path.exists(args.data_dir):
        test_dataset = datasets.ImageFolder(args.data_dir, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        class_names = test_dataset.classes
    else:
        print(f"警告: 测试数据目录不存在 ({args.data_dir})")
        print("将使用模拟数据进行评估演示")
        test_dataset = datasets.FakeData(size=100, image_size=(3, 224, 224), num_classes=3, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        class_names = MODEL_CONFIG['class_names']

    # 加载模型
    print(f"加载模型权重: {args.weights}")
    classifier = GinkgoClassifier(weights=args.weights, device=device)

    # 评估
    os.makedirs(args.output_dir, exist_ok=True)

    evaluator = ClassificationEvaluator(class_names=class_names, class_names_cn=MODEL_CONFIG['class_names_cn'])
    metrics = evaluate_classifier(classifier.model, test_loader, device, class_names)

    # 保存报告
    evaluator.update([], [])  # 确保有数据
    report_path = os.path.join(args.output_dir, 'evaluation_report.json')
    evaluator.save_report(report_path, extra_info={'test_data_dir': args.data_dir})

    # 生成可视化
    if 'confusion_matrix' in metrics:
        import numpy as np
        cm = np.array(metrics['confusion_matrix'])
        plot_confusion_matrix(
            cm, MODEL_CONFIG['class_names_cn'],
            save_path=os.path.join(args.output_dir, 'confusion_matrix.png'),
        )

    print(f"\n评估完成！结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()
