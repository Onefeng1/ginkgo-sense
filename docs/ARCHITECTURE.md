# GinkgoSense 系统架构文档

## 总体架构

```
┌─────────────────────────────────────────────────────┐
│                    Web Layer                         │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Flask App │  │  REST API    │  │  HTML/JS UI  │  │
│  └─────┬────┘  └──────┬───────┘  └──────────────┘  │
├────────┼───────────────┼────────────────────────────┤
│                    Core Layer                        │
│  ┌─────┴────┐  ┌──────┴───────┐  ┌──────────────┐  │
│  │Classifier│  │  Detector    │  │  Evaluator   │  │
│  └─────┬────┘  └──────┬───────┘  └──────────────┘  │
├────────┼───────────────┼────────────────────────────┤
│                  Model Layer                         │
│  ┌─────┴───────────────┴────────────────────────┐   │
│  │  GinkgoResNet / GinkgoEfficientNet           │   │
│  │  (Backbone + CBAM + Classification Head)     │   │
│  └──────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────┤
│                  Utils Layer                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │Augment   │  │GradCAM   │  │ Visualization    │  │
│  │Metrics   │  │Logger    │  │ Benchmark        │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## 模块职责

### Model Layer（模型层）

| 模块 | 文件 | 职责 |
|------|------|------|
| GinkgoResNet | `model/arch.py` | 基于ResNet+CBAM的分类网络 |
| GinkgoEfficientNet | `model/arch.py` | 轻量化EfficientNet变体 |
| GinkgoClassifier | `model/classifier.py` | 分类推理封装 |
| GinkgoDetector | `model/detector.py` | YOLOv5-S目标检测网络 |
| ModelExporter | `model/export.py` | 模型格式导出 |
| ModelBenchmark | `model/benchmark.py` | 性能基准测试 |
| ClassificationEvaluator | `model/evaluator.py` | 分类指标评估 |
| DetectionEvaluator | `model/evaluator.py` | 检测指标评估 |

### Utils Layer（工具层）

| 模块 | 文件 | 职责 |
|------|------|------|
| Data Augmentation | `utils/augment.py` | 在线/离线数据增强 |
| GradCAM | `utils/gradcam.py` | 类激活映射可视化 |
| Visualization | `utils/visualization.py` | 训练曲线、混淆矩阵等 |
| Metrics | `utils/metrics.py` | 评估指标计算 |
| Logger | `utils/logger.py` | 统一日志系统 |

### Web Layer（Web层）

| 模块 | 文件 | 职责 |
|------|------|------|
| Flask App | `app.py` | Web服务主程序 |
| API Routes | `app.py` | RESTful API接口 |
| Frontend | `templates/` | HTML/CSS/JS前端 |

## 数据流

```
用户上传图片
    │
    ▼
Flask路由接收 → 文件验证 → 保存临时文件
    │
    ▼
PIL加载图片 → 预处理(Resize/Crop/Normalize) → Tensor
    │
    ▼
模型推理 → Softmax概率 → 选择最高置信类别
    │
    ▼
后处理(置信度阈值) → 构造响应JSON → 返回结果
    │
    ▼
清理临时文件
```

## 模型架构详解

### GinkgoResNet + CBAM

```
Input (3, 224, 224)
    │
    ▼
Stem: Conv3x3 → BN → ReLU → MaxPool
    │
    ▼
Layer1: 3x Bottleneck (256ch)  ─┐
Layer2: 4x Bottleneck (512ch)   │  ResNet50 Backbone
Layer3: 6x Bottleneck (1024ch) ─┤
    │                            │
    ▼                            │
CBAM (Channel Attention)      ─┘
CBAM (Spatial Attention)
    │
    ▼
Layer4: 3x Bottleneck (2048ch)
    │
    ▼
Global Average Pooling
    │
    ▼
Classifier: Dropout → FC(2048→512) → BN → ReLU
           → Dropout → FC(512→128) → BN → ReLU
           → FC(128→3)
    │
    ▼
Output: [完整果, 轻微裂纹, 严重破损]
```

### CBAM注意力机制

```
特征图 F (C, H, W)
    │
    ├─→ Channel Attention
    │     ├─→ AvgPool → FC → Sigmoid → M_c
    │     └─→ MaxPool → FC → Sigmoid ─┘
    │     F' = M_c ⊗ F
    │
    └─→ Spatial Attention
          ├─→ Channel-wise Avg
          └─→ Channel-wise Max → Conv → Sigmoid → M_s
          F'' = M_s ⊗ F'
```

## 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| 深度学习框架 | PyTorch | >= 1.12 |
| 计算机视觉 | TorchVision | >= 0.13 |
| Web框架 | Flask | >= 2.3 |
| 图像处理 | Pillow | >= 9.0 |
| 数值计算 | NumPy | >= 1.21 |
| 容器化 | Docker | >= 20.10 |
| CI/CD | GitHub Actions | - |
