# GinkgoSense 测试数据说明

## 目录结构

```
data/
├── raw/              # 原始未处理数据
├── processed/        # 处理后的标准数据
│   ├── train/        # 训练集 (70%)
│   │   ├── intact/
│   │   ├── cracked/
│   │   └── broken/
│   ├── val/          # 验证集 (15%)
│   │   ├── intact/
│   │   ├── cracked/
│   │   └── broken/
│   └── test/         # 测试集 (15%)
│       ├── intact/
│       ├── cracked/
│       └── broken/
└── README.md         # 本文件
```

## 数据来源

训练数据采集自实际白果加工产线环境，包含三种类别：

| 类别 | 标签 | 描述 | 建议数量 |
|------|------|------|---------|
| 完整果 | intact | 外壳完整无损 | 2000+ |
| 轻微裂纹 | cracked | 外壳有细小裂纹 | 1500+ |
| 严重破损 | broken | 外壳明显破损或缺失 | 1500+ |

## 图片规格

- **格式**: JPG/PNG
- **分辨率**: 建议 >= 512x512
- **色彩**: RGB
- **背景**: 深色/浅色均可，建议包含多种背景

## 数据准备

```bash
# 扫描统计
python scripts/prepare_data.py scan data/raw

# 自动划分
python scripts/prepare_data.py split data/raw data/processed

# 验证完整性
python scripts/prepare_data.py validate data/processed
```

## 注意事项

1. 确保图片清晰，无明显模糊
2. 每张图片应只包含一个白果
3. 避免重复图片（同一白果不同角度可以保留）
4. 标注应准确，裂纹/破损边界清晰
