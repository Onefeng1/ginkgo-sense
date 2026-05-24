# -*- coding: utf-8 -*-
"""
模型基准测试模块
对比不同backbone的性能、速度、参数量
支持导出ONNX/TorchScript格式
"""
import os
import time
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from config import MODEL_CONFIG


class ModelBenchmark:
    """模型性能基准测试"""

    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.results = []

    def benchmark_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        input_size: Tuple[int, int] = (224, 224),
        batch_sizes: Optional[List[int]] = None,
        num_iterations: int = 100,
        warmup_iterations: int = 20,
    ) -> Dict:
        """
        对单个模型进行完整基准测试

        Args:
            model: PyTorch模型
            model_name: 模型名称
            input_size: 输入尺寸
            batch_sizes: 测试的batch size列表
            num_iterations: 推理迭代次数
            warmup_iterations: 预热迭代次数

        Returns:
            基准测试结果
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32]

        model.to(self.device)
        model.eval()

        # 参数量统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 模型大小估算 (MB)
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

        batch_results = {}
        for bs in batch_sizes:
            try:
                dummy_input = torch.randn(bs, 3, *input_size).to(self.device)

                # 预热
                with torch.no_grad():
                    for _ in range(warmup_iterations):
                        _ = model(dummy_input)

                # 正式测试
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                latencies = []
                with torch.no_grad():
                    for _ in range(num_iterations):
                        start = time.perf_counter()
                        _ = model(dummy_input)
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        end = time.perf_counter()
                        latencies.append((end - start) * 1000)

                batch_results[bs] = {
                    'mean_latency_ms': round(np.mean(latencies), 2),
                    'std_latency_ms': round(np.std(latencies), 2),
                    'p50_latency_ms': round(np.percentile(latencies, 50), 2),
                    'p95_latency_ms': round(np.percentile(latencies, 95), 2),
                    'p99_latency_ms': round(np.percentile(latencies, 99), 2),
                    'throughput_fps': round(1000 / np.mean(latencies) * bs, 1),
                }
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"  batch_size={bs} 显存不足，跳过")
                    torch.cuda.empty_cache()
                    continue
                raise

        result = {
            'model_name': model_name,
            'device': str(self.device),
            'input_size': input_size,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': round(model_size_mb, 2),
            'batch_benchmarks': batch_results,
            'timestamp': datetime.now().isoformat(),
        }

        self.results.append(result)
        return result

    def export_onnx(
        self,
        model: torch.nn.Module,
        save_path: str,
        input_size: Tuple[int, int] = (224, 224),
        opset_version: int = 14,
        dynamic_batch: bool = True,
    ) -> str:
        """
        导出ONNX格式模型

        Args:
            model: PyTorch模型
            save_path: 保存路径
            input_size: 输入尺寸
            opset_version: ONNX opset版本
            dynamic_batch: 是否支持动态batch

        Returns:
            保存路径
        """
        model.eval()
        model.to('cpu')
        dummy = torch.randn(1, 3, *input_size)

        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
            }

        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

        torch.onnx.export(
            model,
            dummy,
            save_path,
            opset_version=opset_version,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
        )

        # 验证导出
        import onnx
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)

        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"ONNX模型已导出: {save_path} ({size_mb:.2f} MB)")
        return save_path

    def export_torchscript(
        self,
        model: torch.nn.Module,
        save_path: str,
        input_size: Tuple[int, int] = (224, 224),
        quantize: bool = False,
    ) -> str:
        """
        导出TorchScript格式模型

        Args:
            model: PyTorch模型
            save_path: 保存路径
            input_size: 输入尺寸
            quantize: 是否量化

        Returns:
            保存路径
        """
        model.eval()
        model.to('cpu')
        dummy = torch.randn(1, 3, *input_size)

        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

        if quantize:
            model_q = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            traced = torch.jit.trace(model_q, dummy)
        else:
            traced = torch.jit.trace(model, dummy)

        traced.save(save_path)
        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"TorchScript模型已导出: {save_path} ({size_mb:.2f} MB)")
        return save_path

    def save_report(self, output_path: str):
        """保存基准测试报告"""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        report = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'results': self.results,
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"基准测试报告已保存: {output_path}")

    def print_report(self):
        """打印基准测试报告"""
        print("\n" + "=" * 80)
        print("                    GinkgoSense 模型基准测试报告")
        print("=" * 80)
        print(f"  设备: {self.device}")

        for result in self.results:
            print(f"\n  模型: {result['model_name']}")
            print(f"  参数量: {result['total_params']:,} ({result['model_size_mb']:.2f} MB)")
            print(f"  可训练参数: {result['trainable_params']:,}")

            print(f"\n  {'Batch':>6} {'平均延迟':>10} {'P95延迟':>10} {'P99延迟':>10} {'吞吐量':>10}")
            print("  " + "-" * 55)
            for bs, bench in result['batch_benchmarks'].items():
                print(f"  {bs:>6} {bench['mean_latency_ms']:>9.2f}ms {bench['p95_latency_ms']:>9.2f}ms {bench['p99_latency_ms']:>9.2f}ms {bench['throughput_fps']:>9.1f}fps")

        print("\n" + "=" * 80)
