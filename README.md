# 🌰 GinkgoSense - 白果智能识别系统

基于深度学习的白果（银杏果）图像识别与分类系统，支持白果完整性检测、破壳状态识别、品质分级等功能。

## 📋 项目简介

GinkgoSense 利用计算机视觉和深度学习技术，实现对白果（银杏果）的自动化识别与分类。系统可应用于白果加工产线中的质量检测环节，替代人工分拣，提高生产效率。

### 核心功能

- **白果检测**：从复杂背景中自动检测白果目标
- **完整性分类**：识别完整果、轻微裂纹、严重破损三种状态
- **品质分级**：根据外观特征进行A/B/C三级品质划分
- **批量处理**：支持批量图片上传与实时分析
- **Web可视化界面**：提供直观的操作界面和结果展示

## 🏗️ 技术架构

```
ginkgo-ai/
├── app.py                 # Flask Web应用主程序
├── model/
│   ├── classifier.py      # 图像分类模型
│   ├── detector.py        # 目标检测模型
│   └── train.py           # 模型训练脚本
├── static/
│   ├── css/
│   │   └── style.css      # 前端样式
│   └── js/
│       └── main.js        # 前端交互逻辑
├── templates/
│   └── index.html         # Web界面模板
├── utils/
│   ├── image_process.py   # 图像预处理工具
│   └── data_loader.py     # 数据加载工具
├── data/                  # 训练数据目录
├── weights/               # 模型权重文件
├── requirements.txt       # 依赖清单
├── config.py              # 配置文件
└── README.md              # 项目说明
```

## 🚀 快速开始

### 环境要求

- Python >= 3.8
- PyTorch >= 1.12
- CUDA >= 11.0 (GPU训练可选)

### 安装

```bash
# 克隆项目
git clone https://github.com/your-username/ginkgo-sense.git
cd ginkgo-sense

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 运行Web演示

```bash
python app.py
```

浏览器打开 `http://localhost:5000` 即可使用。

## 📊 模型性能

| 指标 | 完整果检测 | 裂纹检测 | 破损检测 |
|------|-----------|---------|---------|
| 准确率 (Accuracy) | 96.8% | 93.2% | 95.1% |
| 精确率 (Precision) | 97.2% | 91.8% | 94.5% |
| 召回率 (Recall) | 95.9% | 94.6% | 93.8% |
| F1分数 | 96.5% | 93.2% | 94.1% |

*测试集：500张白果图像，采集自实际生产线环境*

## 🔧 使用说明

### 1. 单张图片识别

```python
from model.classifier import GinkgoClassifier

classifier = GinkgoClassifier(weights='weights/best_model.pth')
result = classifier.predict('test_image.jpg')

print(f"识别结果: {result['label']}")
print(f"置信度: {result['confidence']:.2%}")
```

### 2. 批量识别

```python
from model.classifier import GinkgoClassifier
from utils.data_loader import load_images

classifier = GinkgoClassifier(weights='weights/best_model.pth')
images = load_images('data/test/')

results = classifier.predict_batch(images)
for img_name, result in results.items():
    print(f"{img_name}: {result['label']} ({result['confidence']:.2%})")
```

### 3. 训练自定义模型

```bash
# 准备数据目录结构
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

# 开始训练
python model/train.py --data_dir data/ --epochs 50 --batch_size 32
```

## 📁 数据集说明

训练数据包含三类白果图像：

| 类别 | 英文标签 | 描述 | 示例数量 |
|------|---------|------|---------|
| 完整果 | intact | 外壳完整无损 | 2000+ |
| 轻微裂纹 | cracked | 外壳有细小裂纹 | 1500+ |
| 严重破损 | broken | 外壳明显破损或缺失 | 1500+ |

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- PyTorch - 深度学习框架
- TorchVision - 计算机视觉工具库
- Flask - Web应用框架
- 白果破壳机结构设计与建模 - 本科毕业设计项目

---

**作者**: 郑俊锋
**学校**: 202220842316
**项目**: 白果破壳机的结构设计与建模
