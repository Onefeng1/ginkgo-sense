# -*- coding: utf-8 -*-
"""
模型基准测试脚本
测试不同模型的推理速度、参数量、模型大小
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.classifier import GinkgoClassifier
from model.benchmark import ModelBenchmark


def main():
    parser = argparse.ArgumentParser(description='GinkgoSense 模型基准测试')
    parser.add_argument('--weights', type=str, default='weights/best_model.pth')
    parser.add_argument('--output_dir', type=str, default='results/benchmark')
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[1, 4, 8, 16])
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    benchmark = ModelBenchmark(device=args.device)

    # 测试当前模型
    classifier = GinkgoClassifier(weights=args.weights, device=args.device or 'cpu')
    result = benchmark.benchmark_model(
        classifier.model,
        model_name='GinkgoResNet50_CBAM',
        batch_sizes=args.batch_sizes,
        num_iterations=args.iterations,
    )

    benchmark.print_report()

    os.makedirs(args.output_dir, exist_ok=True)
    benchmark.save_report(os.path.join(args.output_dir, 'benchmark_report.json'))


if __name__ == '__main__':
    main()
