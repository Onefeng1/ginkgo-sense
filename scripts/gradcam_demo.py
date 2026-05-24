# -*- coding: utf-8 -*-
"""
GradCAM可视化演示脚本
生成模型决策区域的热力图
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.classifier import GinkgoClassifier
from utils.gradcam import apply_gradcam


def main():
    parser = argparse.ArgumentParser(description='GinkgoSense GradCAM可视化')
    parser.add_argument('--image', type=str, required=True, help='输入图片路径')
    parser.add_argument('--weights', type=str, default='weights/best_model.pth')
    parser.add_argument('--output', type=str, default='results/gradcam')
    parser.add_argument('--layer', type=str, default='layer4', help='目标层名称')
    parser.add_argument('--class_id', type=int, default=None, help='目标类别ID')
    args = parser.parse_args()

    classifier = GinkgoClassifier(weights=args.weights)
    model = classifier.model

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, f'gradcam_{os.path.basename(args.image)}')

    apply_gradcam(model, args.image, output_path, target_class=args.class_id, layer_name=args.layer)
    print(f"GradCAM可视化完成: {output_path}")


if __name__ == '__main__':
    main()
