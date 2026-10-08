# CamDigest 主应用：流水线 + 调度（insightface sidecar 见 docker-compose.yml）
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir ".[cv]"
ENTRYPOINT ["camdigest"]
CMD ["web", "--config", "/app/config/config.yaml"]   # Web UI + 日批调度（8080）
