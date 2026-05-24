# GinkgoSense 部署指南

## 方式一：本地部署

### 环境要求

- Python >= 3.8
- CUDA >= 11.0 (GPU加速可选)
- 内存 >= 4GB

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/Onefeng1/ginkgo-sense.git
cd ginkgo-sense

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载模型权重（如有）
mkdir -p weights
# 将 best_model.pth 放入 weights/ 目录

# 5. 启动服务
python app.py
```

访问 `http://localhost:5000` 即可使用。

---

## 方式二：Docker部署

### 单容器部署

```bash
# 构建镜像
docker build -t ginkgo-sense .

# 运行容器
docker run -d \
  --name ginkgo \
  -p 5000:5000 \
  -v $(pwd)/weights:/app/weights:ro \
  ginkgo-sense
```

### Docker Compose部署

```bash
# 完整服务（Web + Nginx + Redis）
docker compose up -d

# 仅Web服务
docker compose up -d web

# 查看日志
docker compose logs -f web

# 停止服务
docker compose down
```

---

## 方式三：生产环境部署

### Gunicorn + Nginx

```bash
# 安装Gunicorn
pip install gunicorn

# 启动Gunicorn
gunicorn -w 4 -b 127.0.0.1:5000 --timeout 120 app:app
```

### Nginx配置示例

```nginx
server {
    listen 80;
    server_name ginkgo.example.com;

    client_max_body_size 20M;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }
}
```

### Systemd服务

```ini
[Unit]
Description=GinkgoSense Web Service
After=network.target

[Service]
Type=simple
User=ginkgo
WorkingDirectory=/opt/ginkgo-sense
ExecStart=/opt/ginkgo-sense/venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 app:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

---

## GPU部署注意事项

```bash
# 确认CUDA可用
python -c "import torch; print(torch.cuda.is_available())"

# Docker GPU支持（需要nvidia-container-toolkit）
docker run --gpus all -d -p 5000:5000 ginkgo-sense

# Docker Compose GPU配置
# 在docker-compose.yml的web服务下添加:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

---

## 监控与日志

```bash
# 查看应用日志
tail -f logs/ginkgo_sense_*.log

# Docker日志
docker logs -f ginkgo-sense

# 健康检查
curl http://localhost:5000/info
```
