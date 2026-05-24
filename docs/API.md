# GinkgoSense API文档

## 概述

GinkgoSense提供RESTful API接口，支持白果图像的单张识别和批量识别。

**Base URL**: `http://localhost:5000`

**Content-Type**: `application/json` (响应) / `multipart/form-data` (上传)

---

## 接口列表

### 1. 系统信息 `GET /info`

获取系统基本信息，无需认证。

**响应示例**:
```json
{
  "project": "GinkgoSense",
  "version": "1.0.0",
  "model": "resnet50",
  "classes": ["完整果", "轻微裂纹", "严重破损"],
  "description": "白果智能识别系统"
}
```

---

### 2. 单张识别 `POST /predict`

上传一张白果图片进行识别分类。

**请求参数** (multipart/form-data):

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | ✅ | 白果图片文件 |

**支持格式**: PNG, JPG, JPEG, BMP, WebP  
**最大文件**: 16MB

**响应示例**:
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
  "elapsed_ms": 12.5,
  "image_size": "224x224"
}
```

**错误响应**:

| 状态码 | 说明 |
|--------|------|
| 400 | 未上传文件 / 文件格式不支持 |
| 500 | 识别过程出错 |

**cURL示例**:
```bash
curl -X POST -F "file=@ginkgo.jpg" http://localhost:5000/predict
```

---

### 3. 批量识别 `POST /batch_predict`

同时上传多张图片进行批量识别。

**请求参数** (multipart/form-data):

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| files | File[] | ✅ | 多个图片文件 |

**响应示例**:
```json
{
  "results": [
    {
      "filename": "ginkgo1.jpg",
      "label": "intact",
      "label_cn": "完整果",
      "confidence": 0.9512,
      "probabilities": { ... }
    },
    {
      "filename": "ginkgo2.jpg",
      "label": "cracked",
      "label_cn": "轻微裂纹",
      "confidence": 0.8734,
      "probabilities": { ... }
    }
  ],
  "total": 2
}
```

**cURL示例**:
```bash
curl -X POST -F "files=@img1.jpg" -F "files=@img2.jpg" http://localhost:5000/batch_predict
```

---

### 4. 首页 `GET /`

返回Web可视化界面HTML页面。

---

## Python SDK调用示例

```python
import requests

# 单张识别
with open('test.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict',
        files={'file': f}
    )
result = response.json()
print(f"识别结果: {result['label_cn']} ({result['confidence']:.1%})")

# 批量识别
files = [('files', open(f, 'rb')) for f in ['img1.jpg', 'img2.jpg']]
response = requests.post('http://localhost:5000/batch_predict', files=files)
results = response.json()
for r in results['results']:
    print(f"{r['filename']}: {r['label_cn']}")
```

## 错误码说明

| 错误码 | 说明 | 处理建议 |
|--------|------|---------|
| 400 | 请求参数错误 | 检查文件是否正确上传 |
| 413 | 文件过大 | 压缩图片或减小尺寸 |
| 500 | 服务器内部错误 | 查看服务端日志 |
