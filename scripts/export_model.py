# -*- coding: utf-8 -*-
"""
模型导出脚本
将训练好的模型导出为多种部署格式
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.classifier import GinkgoClassifier
from model.export import ModelExporter
from config import MODEL_CONFIG


def main():
    parser = argparse.ArgumentParser(description='GinkgoSense 模型导出')
    parser.add_argument('--weights', type=str, default='weights/best_model.pth', help='模型权重')
    parser.add_argument('--output_dir', type=str, default='weights/exports', help='导出目录')
    parser.add_argument('--formats', nargs='+', default=['torchscript', 'onnx', 'quantized'],
                        choices=['torchscript', 'onnx', 'quantized', 'all'])
    parser.add_argument('--input_size', type=int, default=224, help='输入尺寸')
    parser.add_argument('--model_name', type=str, default='ginkgo_resnet50', help='模型名称')
    args = parser.parse_args()

    print("加载模型...")
    classifier = GinkgoClassifier(weights=args.weights)

    print("导出模型...")
    exporter = ModelExporter(classifier.model, output_dir=args.output_dir)

    if 'all' in args.formats:
        exporter.export_all(input_size=(args.input_size, args.input_size), model_name=args.model_name)
    else:
        for fmt in args.formats:
            if fmt == 'torchscript':
                exporter.export_torchscript(input_size=(args.input_size, args.input_size), model_name=args.model_name)
            elif fmt == 'onnx':
                exporter.export_onnx(input_size=(args.input_size, args.input_size), model_name=args.model_name)
            elif fmt == 'quantized':
                exporter.export_torchscript_quantized(input_size=(args.input_size, args.input_size), model_name=args.model_name)

    print("导出完成！")


if __name__ == '__main__':
    main()
