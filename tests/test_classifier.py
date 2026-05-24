# -*- coding: utf-8 -*-
"""
分类模型单元测试
"""
import os
import sys
import pytest
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.classifier import GinkgoClassifier
from model.arch import GinkgoResNet, ChannelAttention, SpatialAttention, CBAM, build_model
from model.evaluator import ClassificationEvaluator
from utils.metrics import accuracy, precision_recall_f1, confusion_matrix


class TestGinkgoResNet:
    """GinkgoResNet模型测试"""

    def test_build_resnet50(self):
        model = GinkgoResNet(num_classes=3, backbone='resnet50', pretrained=False, use_cbam=False)
        assert model is not None
        assert model.num_classes == 3

    def test_build_resnet18(self):
        model = GinkgoResNet(num_classes=3, backbone='resnet18', pretrained=False, use_cbam=False)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 3)

    def test_build_resnet50_with_cbam(self):
        model = GinkgoResNet(num_classes=3, backbone='resnet50', pretrained=False, use_cbam=True)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 3)

    def test_invalid_backbone(self):
        with pytest.raises(ValueError, match="不支持的backbone"):
            GinkgoResNet(backbone='invalid_model')

    def test_feature_maps(self):
        model = GinkgoResNet(num_classes=3, backbone='resnet18', pretrained=False, use_cbam=False)
        x = torch.randn(1, 3, 224, 224)
        features = model.get_feature_maps(x)
        assert 'stem' in features
        assert 'layer4' in features
        assert features['layer4'].shape[1] > 0

    def test_count_parameters(self):
        model = GinkgoResNet(num_classes=3, backbone='resnet18', pretrained=False, use_cbam=False)
        params = model.count_parameters()
        assert 'total' in params
        assert 'trainable' in params
        assert params['total'] > 0

    def test_build_model_factory(self):
        config = {
            'model_type': 'resnet',
            'backbone': 'resnet50',
            'num_classes': 3,
            'pretrained': False,
            'use_cbam': False,
        }
        model = build_model(config)
        assert isinstance(model, GinkgoResNet)


class TestAttentionModules:
    """注意力模块测试"""

    def test_channel_attention(self):
        ca = ChannelAttention(64)
        x = torch.randn(1, 64, 14, 14)
        out = ca(x)
        assert out.shape == x.shape

    def test_spatial_attention(self):
        sa = SpatialAttention()
        x = torch.randn(1, 64, 14, 14)
        out = sa(x)
        assert out.shape == x.shape

    def test_cbam(self):
        cbam = CBAM(128)
        x = torch.randn(1, 128, 7, 7)
        out = cbam(x)
        assert out.shape == x.shape


class TestGinkgoClassifier:
    """GinkgoClassifier测试"""

    def test_init_demo_mode(self):
        classifier = GinkgoClassifier()
        assert classifier is not None
        assert classifier.model is not None

    def test_predict_single(self, sample_image_file):
        classifier = GinkgoClassifier()
        result = classifier.predict(sample_image_file)
        assert 'label' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert result['label'] in ['intact', 'cracked', 'broken']
        assert 0 <= result['confidence'] <= 1

    def test_predict_batch(self, tmp_path):
        # 创建多张测试图片
        paths = []
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            path = tmp_path / f"test_{i}.jpg"
            img.save(str(path))
            paths.append(str(path))

        classifier = GinkgoClassifier()
        results = classifier.predict_batch(paths)
        assert len(results) == 3
        for path in paths:
            assert path in results
            assert 'label' in results[path]


class TestMetrics:
    """评估指标测试"""

    def test_accuracy(self):
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 1, 1])
        acc = accuracy(y_true, y_pred)
        assert acc == 0.8

    def test_precision_recall_f1(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        metrics = precision_recall_f1(y_true, y_pred, num_classes=3)
        assert 0 in metrics
        assert 1 in metrics
        assert 2 in metrics
        for cls_metrics in metrics.values():
            assert 'precision' in cls_metrics
            assert 'recall' in cls_metrics
            assert 'f1' in cls_metrics

    def test_confusion_matrix(self):
        y_true = np.array([0, 0, 1, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2])
        cm = confusion_matrix(y_true, y_pred, num_classes=3)
        assert cm.shape == (3, 3)
        assert cm[0, 0] == 1
        assert cm[0, 1] == 1
        assert cm[1, 1] == 1
        assert cm[1, 0] == 1
        assert cm[2, 2] == 1


class TestEvaluator:
    """评估器测试"""

    def test_evaluator_init(self):
        evaluator = ClassificationEvaluator()
        assert evaluator.num_classes == 3

    def test_evaluator_update_and_compute(self):
        evaluator = ClassificationEvaluator(
            class_names=['intact', 'cracked', 'broken'],
            class_names_cn=['完整果', '轻微裂纹', '严重破损'],
        )
        preds = [0, 0, 1, 1, 2, 2]
        labels = [0, 1, 1, 1, 2, 0]
        evaluator.update(preds, labels)

        metrics = evaluator.compute_metrics()
        assert 'accuracy' in metrics
        assert 'macro_f1' in metrics
        assert metrics['total_samples'] == 6

    def test_evaluator_reset(self):
        evaluator = ClassificationEvaluator()
        evaluator.update([0, 1], [0, 1])
        assert len(evaluator.all_predictions) == 2
        evaluator.reset()
        assert len(evaluator.all_predictions) == 0
