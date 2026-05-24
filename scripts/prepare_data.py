# -*- coding: utf-8 -*-
"""
数据准备脚本
自动下载、解压、划分训练/验证/测试集
支持多种数据源格式
"""
import os
import json
import shutil
import random
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional
from collections import Counter


def scan_dataset(data_dir: str) -> dict:
    """
    扫描数据集目录结构

    Args:
        data_dir: 数据集根目录

    Returns:
        数据集统计信息
    """
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    stats = {'total': 0, 'classes': {}, 'splits': {}}

    for root, dirs, files in os.walk(data_dir):
        rel_path = os.path.relpath(root, data_dir)
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in extensions:
                stats['total'] += 1
                class_name = os.path.basename(root)
                stats['classes'][class_name] = stats['classes'].get(class_name, 0) + 1

                split = rel_path.split(os.sep)[0] if os.sep in rel_path else 'unknown'
                if split not in stats['splits']:
                    stats['splits'][split] = {'total': 0, 'classes': {}}
                stats['splits'][split]['total'] += 1
                stats['splits'][split]['classes'][class_name] = \
                    stats['splits'][split]['classes'].get(class_name, 0) + 1

    return stats


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    copy: bool = True,
):
    """
    将数据集按比例划分为训练/验证/测试集

    Args:
        source_dir: 源数据目录（每个子目录为一个类别）
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        copy: True=复制, False=移动
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

    random.seed(seed)
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # 收集所有类别和图片
    class_dirs = [d for d in os.listdir(source_dir)
                  if os.path.isdir(os.path.join(source_dir, d))]

    print(f"发现 {len(class_dirs)} 个类别: {class_dirs}")

    total_files = 0
    for class_name in class_dirs:
        class_path = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(class_path)
                  if os.path.splitext(f)[1].lower() in extensions]

        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:],
        }

        for split_name, file_list in splits.items():
            split_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(split_dir, exist_ok=True)

            for fname in file_list:
                src = os.path.join(class_path, fname)
                dst = os.path.join(split_dir, fname)
                if copy:
                    shutil.copy2(src, dst)
                else:
                    shutil.move(src, dst)
                total_files += 1

        print(f"  {class_name}: {n} 张 -> train={n_train}, val={n_val}, test={n - n_train - n_val}")

    print(f"\n数据集划分完成: 共 {total_files} 张图片")
    print(f"输出目录: {output_dir}")

    # 生成统计报告
    report = {
        'source': source_dir,
        'output': output_dir,
        'ratios': {'train': train_ratio, 'val': val_ratio, 'test': test_ratio},
        'total_images': total_files,
        'classes': {},
    }
    for class_name in class_dirs:
        class_path = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(class_path)
                  if os.path.splitext(f)[1].lower() in extensions]
        report['classes'][class_name] = len(images)

    report_path = os.path.join(output_dir, 'split_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"统计报告已保存: {report_path}")


def validate_dataset(data_dir: str) -> dict:
    """
    验证数据集完整性和质量

    Returns:
        验证结果
    """
    from PIL import Image

    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    issues = []
    corrupted = 0
    total = 0

    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() not in extensions:
                continue

            total += 1
            path = os.path.join(root, f)

            try:
                img = Image.open(path)
                img.verify()
                img = Image.open(path)
                img.load()
            except Exception as e:
                corrupted += 1
                issues.append({'file': path, 'error': str(e)})

    result = {
        'total_images': total,
        'corrupted': corrupted,
        'healthy': total - corrupted,
        'integrity_rate': (total - corrupted) / total if total > 0 else 0,
        'issues': issues[:50],  # 最多显示50个问题
    }

    print(f"\n数据集验证结果:")
    print(f"  总图片数: {total}")
    print(f"  损坏图片: {corrupted}")
    print(f"  完好率: {result['integrity_rate']:.2%}")

    if issues:
        print(f"\n  损坏文件列表:")
        for issue in issues[:10]:
            print(f"    {issue['file']}: {issue['error']}")
        if len(issues) > 10:
            print(f"    ... 还有 {len(issues) - 10} 个问题文件")

    return result


def generate_data_statistics(data_dir: str, output_path: str):
    """生成数据集统计报告"""
    stats = scan_dataset(data_dir)

    report = {
        'dataset_path': data_dir,
        'total_images': stats['total'],
        'num_classes': len(stats['classes']),
        'class_distribution': stats['classes'],
        'splits': stats['splits'],
    }

    # 计算类别不平衡度
    if stats['classes']:
        counts = list(stats['classes'].values())
        report['imbalance_ratio'] = max(counts) / min(counts) if min(counts) > 0 else float('inf')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"统计报告已保存: {output_path}")
    return report


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='白果数据集准备工具')
    subparsers = parser.add_subparsers(dest='command')

    # scan命令
    scan_parser = subparsers.add_parser('scan', help='扫描数据集')
    scan_parser.add_argument('data_dir', help='数据目录')

    # split命令
    split_parser = subparsers.add_parser('split', help='划分数据集')
    split_parser.add_argument('source_dir', help='源数据目录')
    split_parser.add_argument('output_dir', help='输出目录')
    split_parser.add_argument('--train-ratio', type=float, default=0.7)
    split_parser.add_argument('--val-ratio', type=float, default=0.15)
    split_parser.add_argument('--test-ratio', type=float, default=0.15)
    split_parser.add_argument('--seed', type=int, default=42)

    # validate命令
    validate_parser = subparsers.add_parser('validate', help='验证数据集')
    validate_parser.add_argument('data_dir', help='数据目录')

    args = parser.parse_args()

    if args.command == 'scan':
        stats = scan_dataset(args.data_dir)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    elif args.command == 'split':
        split_dataset(args.source_dir, args.output_dir, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    elif args.command == 'validate':
        validate_dataset(args.data_dir)
    else:
        parser.print_help()
