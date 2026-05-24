# ==========================================
# GinkgoSense - Docker多阶段构建
# ==========================================

# --- Stage 1: 构建阶段 ---
FROM python:3.10-slim as builder

WORKDIR /build

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: 运行阶段 ---
FROM python:3.10-slim as runtime

# 设置时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 创建非root用户
RUN groupadd -r ginkgo && useradd -r -g ginkgo -d /app -s /sbin/nologin ginkgo

WORKDIR /app

# 从构建阶段复制依赖
COPY --from=builder /install /usr/local

# 复制应用代码
COPY . .

# 创建必要目录
RUN mkdir -p weights uploads logs results \
    && chown -R ginkgo:ginkgo /app

USER ginkgo

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/info')" || exit 1

# 暴露端口
EXPOSE 5000

# 环境变量
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 启动命令
CMD ["python", "app.py"]
