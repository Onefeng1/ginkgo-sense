# -*- coding: utf-8 -*-
"""
模型导出模块
支持将训练好的模型导出为多种格式用于部署
"""
import os
import json
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


class ModelExporter:
    """模型格式导出器"""

    def __init__(self, model: nn.Module, output_dir: str = 'weights/exports'):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_all(
        self,
        input_size: Tuple[int, int] = (224, 224),
        model_name: str = 'ginkgo_model',
    ) -> Dict[str, str]:
        """
        导出所有支持的格式

        Returns:
            格式 -> 文件路径的映射
        """
        self.model.eval()
        self.model.cpu()
        results = {}

        # TorchScript
        try:
            ts_path = self.export_torchscript(input_size, model_name)
            results['torchscript'] = ts_path
        except Exception as e:
            print(f"TorchScript导出失败: {e}")

        # ONNX
        try:
            onnx_path = self.export_onnx(input_size, model_name)
            results['onnx'] = onnx_path
        except Exception as e:
            print(f"ONNX导出失败: {e}")

        # 量化TorchScript
        try:
            q_path = self.export_torchscript_quantized(input_size, model_name)
            results['torchscript_quantized'] = q_path
        except Exception as e:
            print(f"量化TorchScript导出失败: {e}")

        # 保存元数据
        meta = {
            'model_name': model_name,
            'input_size': list(input_size),
            'formats': results,
            'total_params': sum(p.numel() for p in self.model.parameters()),
        }
        meta_path = os.path.join(self.output_dir, f'{model_name}_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"\n导出完成，共 {len(results)} 个格式")
        for fmt, path in results.items():
            print(f"  {fmt}: {path}")

        return results

    def export_torchscript(
        self,
        input_size: Tuple[int, int] = (224, 224),
        model_name: str = 'ginkgo_model',
    ) -> str:
        """导出TorchScript格式"""
        dummy = torch.randn(1, 3, *input_size)
        path = os.path.join(self.output_dir, f'{model_name}.torchscript')
        traced = torch.jit.trace(self.model, dummy)
        traced.save(path)
        print(f"TorchScript已导出: {path}")
        return path

    def export_torchscript_quantized(
        self,
        input_size: Tuple[int, int] = (224, 224),
        model_name: str = 'ginkgo_model',
    ) -> str:
        """导出量化TorchScript格式"""
        model_q = torch.quantization.quantize_dynamic(
            self.model, {nn.Linear}, dtype=torch.qint8
        )
        dummy = torch.randn(1, 3, *input_size)
        path = os.path.join(self.output_dir, f'{model_name}_quantized.torchscript')
        traced = torch.jit.trace(model_q, dummy)
        traced.save(path)
        print(f"量化TorchScript已导出: {path}")
        return path

    def export_onnx(
        self,
        input_size: Tuple[int, int] = (224, 224),
        model_name: str = 'ginkgo_model',
        opset_version: int = 14,
    ) -> str:
        """导出ONNX格式"""
        dummy = torch.randn(1, 3, *input_size)
        path = os.path.join(self.output_dir, f'{model_name}.onnx')

        torch.onnx.export(
            self.model, dummy, path,
            opset_version=opset_version,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        )
        print(f"ONNX已导出: {path}")
        return path
