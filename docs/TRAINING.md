# GinkgoSense 模型训练指南

## 1. 数据集准备

### 目录结构

```
data/
├── train/
│   ├── intact/      # 完整果图片
│   ├── cracked/     # 轻微裂纹图片
│   └── broken/      # 严重破损图片
├── val/
│   ├── intact/
│   ├── cracked/
│   └── broken/
└── test/
    ├── intact/
    ├── cracked/
    └── broken/
```

### 数据准备脚本

```bash
# 扫描数据集统计信息
python scripts/prepare_data.py scan data/raw

# 自动划分训练/验证/测试集
python scripts/prepare_data.py split data/raw data/processed --train-ratio 0.7

# 验证数据集完整性
python scripts/prepare_data.py validate data/processed
```

### 数据增强

项目内置多种数据增强策略，针对白果图像特点优化：

| 增强方式 | 说明 | 概率 |
|---------|------|------|
| 随机裁剪 | 256→224 CenterCrop + RandomCrop | - |
| 水平翻转 | 镜像翻转 | 50% |
| 垂直翻转 | 上下翻转 | 15% |
| 随机旋转 | ±30° | - |
| 颜色抖动 | 亮度/对比度/饱和度 | - |
| 高斯模糊 | 模拟对焦不准 | - |
| 随机擦除 | Cutout模拟遮挡 | 20% |
| 太阳化 | 模拟强光照射 | 30% |

---

## 2. 训练配置

### 配置文件 `configs/train.yaml`

```yaml
model:
  backbone: resnet50
  num_classes: 3
  pretrained: true
  use_cbam: true
  dropout: 0.3

training:
  batch_size: 32
  learning_rate: 0.0001
  epochs: 50
  weight_decay: 0.0001
  optimizer: adam
  lr_scheduler: cosine
  early_stopping: 10

data:
  train_dir: data/train
  val_dir: data/val
  test_dir: data/test
  num_workers: 4
```

---

## 3. 开始训练

### 使用内置训练脚本

```bash
# 默认配置训练
python model/train.py --data_dir data --epochs 50

# 自定义配置
python model/train.py \
  --data_dir data \
  --backbone resnet50 \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.0001 \
  --use_cbam

# GPU训练
python model/train.py --data_dir data --epochs 50 --device cuda
```

### 训练过程输出

```
Epoch  1 | Train Loss: 1.2345 Acc: 45.23% | Val Loss: 0.9876 Acc: 62.10% | LR: 1.00e-04
Epoch  2 | Train Loss: 0.8765 Acc: 65.43% | Val Loss: 0.7654 Acc: 72.30% | LR: 9.89e-05
...
Epoch 50 | Train Loss: 0.1234 Acc: 96.80% | Val Loss: 0.1456 Acc: 95.20% | LR: 1.00e-06
```

---

## 4. 评估模型

```bash
# 在测试集上评估
python scripts/evaluate.py --data_dir data/test --weights weights/best_model.pth

# 生成GradCAM可视化
python scripts/gradcam_demo.py --image test.jpg --weights weights/best_model.pth

# 基准测试（推理速度、参数量）
python scripts/benchmark.py --weights weights/best_model.pth
```

---

## 5. 模型导出

```bash
# 导出所有格式
python scripts/export_model.py --weights weights/best_model.pth --formats all

# 仅导出ONNX
python scripts/export_model.py --weights weights/best_model.pth --formats onnx

# 量化导出（模型更小，推理更快）
python scripts/export_model.py --weights weights/best_model.pth --formats quantized
```

---

## 6. 调参建议

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| learning_rate | 1e-5 ~ 1e-3 | 建议从1e-4开始 |
| batch_size | 16 ~ 64 | 显存不足可减小 |
| dropout | 0.2 ~ 0.5 | 过拟合时增大 |
| epochs | 30 ~ 100 | 配合Early Stopping |
| weight_decay | 1e-5 ~ 1e-3 | L2正则化强度 |

### 常见问题

**过拟合**：
- 增大dropout
- 增加数据增强强度
- 增加训练数据
- 减小模型复杂度

**欠拟合**：
- 增加训练轮数
- 增大学习率
- 使用更大的backbone
- 检查数据质量

**训练不稳定**：
- 减小学习率
- 使用学习率warmup
- 检查数据标签是否正确
