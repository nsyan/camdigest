# CamDigest

把已录制的家庭监控录像（4 台摄像头 · 5 机位 · 含音频，其中一台为双摄设备，出固定+云台两路）离线分析为每日四档精华视频与带图飞书日报。Docker Compose 部署。

模型双角色（详见 ADR-0001）：**识别**（全模态，仅 OpenAI 兼容协议）与**报告**（文本，OpenAI 或 Anthropic 协议可选）。

## 文档

- [CONTEXT.md](./CONTEXT.md) — 统一语言（术语表）
- [docs/architecture.md](./docs/architecture.md) — 技术架构（部署、流水线、数据模型、配置）
- [docs/adr/](./docs/adr/) — 架构决策记录

## 状态

M1 CLI 全链路已实现（`camdigest run --date` 一键 S1-S10）· 设计已定稿（三轮设计访谈收敛 + 双轴评审修复，2026-09）· 双摄适配修订（4 设备 5 机位）

## 运行

依赖：Docker Compose（或本机 Python 3.12 + ffmpeg + `pip install -e ".[cv]"`）。

```bash
# 1. 配置：复制样例到 config/config.yaml，填入 API key / 飞书凭据 / 真机目录
cp config.example.yaml config/config.yaml

# 2. 人脸建档：/data/faces/{身份名}/*.jpg → registry
camdigest enroll-faces --config config/config.yaml

# 3. 首跑（任意历史日期，S1-S10 一键）
camdigest run --date 2026-09-29 --config config/config.yaml

# 4. 日常：每日 03:00 自动跑前一天（容器默认入口）
docker compose up -d --build
```

产物在 `/data`：`highlights/{日期}/精华_5|10|30|60min.mp4` + `reports/{日期}.md` + 飞书在线文档（家庭群卡片推送）。断点续跑：中断后重跑同一日期自动跳过已完成阶段。

> 校准点：CAL-1（双摄目录形态）、CAL-2（双机位音轨）首跑后按真机实测改 `config.yaml`，无需改代码（见 config.example.yaml 内注释）。
