# 🌰 GinkgoSense — 白果智能识别系统

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3+-000000?style=flat-square&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue?style=flat-square&logo=githubactions&logoColor=white)

**基于深度学习的白果（银杏果）图像识别与分类系统**

*支持白果完整性检测、破壳状态识别、品质分级、目标检测、模型可解释性分析*

</div>

---

## 📋 目录

- [项目简介](#-项目简介)
- [核心功能](#-核心功能)
- [系统架构](#-系统架构)
- [快速开始](#-快速开始)
- [API接口](#-api接口)
- [模型性能](#-模型性能)
- [训练指南](#-训练指南)
- [部署方案](#-部署方案)
- [项目结构](#-项目结构)
- [技术栈](#-技术栈)
- [贡献指南](#-贡献指南)
- [许可证](#-许可证)
- [致谢](#-致谢)

---

## 🎯 项目简介

GinkgoSense 是一个面向白果（银杏果）加工产线的智能视觉检测系统。系统利用深度学习和计算机视觉技术，实现对白果的自动化识别、分类和质量检测，旨在替代传统人工分拣，提高生产效率和检测一致性。

### 应用场景

| 场景 | 说明 |
|------|------|
| 🏭 **产线质检** | 白果加工流水线上的实时质量检测 |
| 📦 **分拣分级** | 根据外观特征自动进行A/B/C品质分级 |
| 🔍 **缺陷检测** | 检测裂纹、破损等外观缺陷 |
| 📊 **数据分析** | 统计分析产线良品率和缺陷分布 |

### 核心能力

- **白果检测**：从复杂背景中自动检测白果目标（基于改进YOLOv5）
- **完整性分类**：识别完整果、轻微裂纹、严重破损三种状态（基于ResNet50+CBAM）
- **品质分级**：根据外观特征进行A/B/C三级品质划分
- **批量处理**：支持批量图片上传与实时分析
- **模型可解释性**：GradCAM热力图可视化模型决策依据
- **多格式导出**：支持TorchScript、ONNX、量化模型导出

---

## 🧩 核心功能

### 1. 图像分类

基于ResNet50 + CBAM注意力机制的迁移学习模型，三分类白果完整性检测：

| 类别 | 英文标签 | 描述 | 样本数 |
|------|---------|------|--------|
| 完整果 | `intact` | 外壳完整无损 | 2000+ |
| 轻微裂纹 | `cracked` | 外壳有细小裂纹 | 1500+ |
| 严重破损 | `broken` | 外壳明显破损或缺失 | 1500+ |

### 2. 目标检测

基于改进YOLOv5-S的轻量化检测网络，支持：
- 单果/多果检测
- 实时检测（30+ FPS）
- 自定义锚框适配不同尺寸白果

### 3. 模型可解释性

通过GradCAM生成类激活映射热力图，直观展示模型决策依据：
- 通道注意力可视化
- 空间注意力可视化
- 特征图提取与展示

### 4. 数据增强

针对白果图像特点设计的增强策略：
- 随机裁剪/翻转/旋转
- 颜色抖动（模拟不同光照）
- 高斯模糊（模拟对焦不准）
- Cutout/随机擦除（模拟遮挡）
- 太阳化（模拟强光照射）

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────┐
│                    Web Layer                         │
│  Flask App │ REST API │ HTML/JS UI │ Nginx Proxy    │
├─────────────────────────────────────────────────────┤
│                    Core Layer                        │
│  Classifier │ Detector │ Evaluator │ Benchmark      │
├─────────────────────────────────────────────────────┤
│                  Model Layer                         │
│  GinkgoResNet+CBAM │ GinkgoEfficientNet │ YOLOv5-S  │
├─────────────────────────────────────────────────────┤
│                  Utils Layer                         │
│  Augment │ GradCAM │ Visualization │ Metrics │ Log  │
├─────────────────────────────────────────────────────┤
│                  Infra Layer                         │
│  Docker │ CI/CD │ Tests │ Configs │ Docs            │
└─────────────────────────────────────────────────────┘
```

**详细架构文档** → [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## 🚀 快速开始

### 环境要求

- Python >= 3.8
- PyTorch >= 1.12
- CUDA >= 11.0（GPU训练可选，CPU推理可用）

### 安装

```bash
# 克隆项目
git clone https://github.com/Onefeng1/ginkgo-sense.git
cd ginkgo-sense

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 安装开发依赖（可选）
pip install -e ".[dev]"
```

### 运行Web演示

```bash
python app.py
```

浏览器打开 `http://localhost:5000` 即可使用。

### Docker一键部署

```bash
docker compose up -d
```

---

## 📡 API接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | Web可视化界面 |
| `GET` | `/info` | 系统信息 |
| `POST` | `/predict` | 单张图片识别 |
| `POST` | `/batch_predict` | 批量图片识别 |

### 单张识别

```bash
curl -X POST -F "file=@test.jpg" http://localhost:5000/predict
```

**响应**:
```json
{
  "label": "intact",
  "label_cn": "完整果",
  "confidence": 0.9681,
  "probabilities": {
    "完整果": 0.9681,
    "轻微裂纹": 0.0203,
    "严重破损": 0.0116
  },
  "elapsed_ms": 12.5
}
```

### Python SDK

```python
import requests

with open('test.jpg', 'rb') as f:
    result = requests.post('http://localhost:5000/predict', files={'file': f}).json()

print(f"识别结果: {result['label_cn']} ({result['confidence']:.1%})")
```

**完整API文档** → [docs/API.md](docs/API.md)

---

## 📊 模型性能

### 分类模型（ResNet50 + CBAM）

| 指标 | 完整果 | 裂纹 | 破损 | 宏平均 |
|------|--------|------|------|--------|
| 准确率 | 96.8% | 93.2% | 95.1% | - |
| 精确率 | 97.2% | 91.8% | 94.5% | 94.5% |
| 召回率 | 95.9% | 94.6% | 93.8% | 94.8% |
| F1分数 | 96.5% | 93.2% | 94.1% | 94.6% |

*测试集：500张白果图像，采集自实际生产线环境*

### 模型规格

| 规格 | 数值 |
|------|------|
| 总参数量 | 23.5M |
| 可训练参数 | 23.5M |
| 模型大小 | 89.7 MB |
| 推理延迟 (batch=1) | ~12ms (GPU) / ~85ms (CPU) |
| 吞吐量 (batch=32) | ~280 FPS (GPU) |

### 基准测试

```bash
# 运行基准测试
python scripts/benchmark.py --weights weights/best_model.pth
```

---

## 🔧 训练指南

### 数据准备

```bash
# 扫描数据集
python scripts/prepare_data.py scan data/raw

# 自动划分训练/验证/测试集
python scripts/prepare_data.py split data/raw data/processed

# 验证数据完整性
python scripts/prepare_data.py validate data/processed
```

### 开始训练

```bash
# 默认配置训练
python model/train.py --data_dir data --epochs 50

# 自定义配置
python model/train.py \
  --data_dir data \
  --backbone resnet50 \
  --epochs 100 \
  --batch_size 16 \
  --use_cbam
```

### 评估与可视化

```bash
# 模型评估
python scripts/evaluate.py --data_dir data/test --weights weights/best_model.pth

# GradCAM可视化
python scripts/gradcam_demo.py --image test.jpg --weights weights/best_model.pth

# 模型导出
python scripts/export_model.py --weights weights/best_model.pth --formats all
```

**完整训练文档** → [docs/TRAINING.md](docs/TRAINING.md)

---

## 📦 部署方案

| 方式 | 适用场景 | 文档 |
|------|---------|------|
| 本地运行 | 开发调试 | [快速开始](#快速开始) |
| Docker | 标准部署 | [Docker部署](#docker一键部署) |
| Gunicorn + Nginx | 生产环境 | [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) |
| Docker Compose | 微服务架构 | [docker-compose.yml](docker-compose.yml) |

### Docker Compose 服务

```bash
# 启动完整服务（Web + Nginx + Redis）
docker compose up -d

# 仅Web服务
docker compose up -d web
```

---

## 📁 项目结构

```
ginkgo-sense/
├── app.py                      # Flask Web应用主程序
├── config.py                   # 全局配置文件
├── requirements.txt            # Python依赖清单
├── setup.py                    # 包安装配置
├── pyproject.toml              # 项目元数据
│
├── model/                      # 🧠 模型层
│   ├── __init__.py
│   ├── arch.py                 # 模型架构定义 (ResNet+CBAM, EfficientNet)
│   ├── classifier.py           # 分类推理封装
│   ├── detector.py             # 目标检测 (YOLOv5-S)
│   ├── evaluator.py            # 模型评估 (分类+检测)
│   ├── benchmark.py            # 性能基准测试
│   ├── export.py               # 模型格式导出
│   └── train.py                # 模型训练脚本
│
├── utils/                      # 🔧 工具层
│   ├── __init__.py
│   ├── augment.py              # 数据增强 (在线+离线)
│   ├── data_loader.py          # 数据加载器
│   ├── gradcam.py              # GradCAM可解释性分析
│   ├── image_process.py        # 图像预处理
│   ├── logger.py               # 统一日志系统
│   ├── metrics.py              # 评估指标计算
│   └── visualization.py        # 可视化工具
│
├── scripts/                    # 🛠️ 脚本层
│   ├── prepare_data.py         # 数据准备与划分
│   ├── evaluate.py             # 模型评估
│   ├── export_model.py         # 模型导出
│   ├── benchmark.py            # 基准测试
│   └── gradcam_demo.py         # GradCAM演示
│
├── tests/                      # 🧪 测试层
│   ├── conftest.py             # 测试fixtures
│   ├── test_classifier.py      # 模型单元测试
│   └── test_api.py             # API接口测试
│
├── templates/                  # 🎨 Web模板
│   └── index.html              # 识别界面
│
├── static/                     # 📦 静态资源
│   ├── css/
│   ├── js/
│   └── img/
│
├── configs/                    # ⚙️ 配置文件
│   ├── resnet50.yaml           # ResNet50模型配置
│   └── train.yaml              # 训练超参数配置
│
├── docs/                       # 📖 文档
│   ├── API.md                  # API接口文档
│   ├── ARCHITECTURE.md         # 系统架构文档
│   ├── DEPLOYMENT.md           # 部署指南
│   └── TRAINING.md             # 训练指南
│
├── Dockerfile                  # 🐳 Docker镜像
├── docker-compose.yml          # Docker编排
├── .github/workflows/          # 🔄 CI/CD
│   ├── ci.yml                  # 持续集成
│   └── deploy.yml              # 自动部署
├── LICENSE                     # MIT许可证
└── README.md                   # 项目说明
```

---

## 🛠️ 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 深度学习 | PyTorch 1.12+ | 模型训练与推理 |
| 计算机视觉 | TorchVision | 图像预处理、预训练模型 |
| Web框架 | Flask | REST API服务 |
| 图像处理 | Pillow | 图像加载与增强 |
| 可视化 | Matplotlib | 训练曲线、热力图 |
| 容器化 | Docker | 应用打包与部署 |
| CI/CD | GitHub Actions | 自动化测试与部署 |
| 代码规范 | Black + Flake8 | 代码格式化与检查 |

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

### 开发环境

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v

# 代码检查
flake8 app.py model/ utils/ tests/
black --check app.py model/ utils/ tests/

# 代码格式化
black app.py model/ utils/ tests/
isort app.py model/ utils/ tests/
```

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [TorchVision](https://pytorch.org/vision/) - 计算机视觉工具库
- [Flask](https://flask.palletsprojects.com/) - Web应用框架
- [ResNet](https://arxiv.org/abs/1512.03385) - 残差网络架构
- [CBAM](https://arxiv.org/abs/1807.06521) - 卷积块注意力模块
- [YOLOv5](https://github.com/ultralytics/yolov5) - 目标检测框架
- [GradCAM](https://arxiv.org/abs/1610.02391) - 可解释性分析

---

**作者**: 郑俊锋 ([@Onefeng1](https://github.com/Onefeng1))  
**项目**: 白果破壳机的结构设计与建模 — 本科毕业设计  
**学校编号**: 202220842316

<div align="center">

⭐ 如果这个项目对你有帮助，请点个Star支持一下！

</div>
