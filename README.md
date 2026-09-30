# CamDigest

把已录制的家庭监控录像（4 台摄像头 · 5 机位 · 含音频，其中一台为双摄设备，出固定+云台两路）离线分析为每日四档精华视频与带图飞书日报。Docker Compose 部署。

模型双角色（详见 ADR-0001）：**识别**（全模态，仅 OpenAI 兼容协议）与**报告**（文本，OpenAI 或 Anthropic 协议可选）。

## 文档

- [CONTEXT.md](./CONTEXT.md) — 统一语言（术语表）
- [docs/architecture.md](./docs/architecture.md) — 技术架构（部署、流水线、数据模型、配置）
- [docs/adr/](./docs/adr/) — 架构决策记录

## 状态

M1 开发前 · 设计已定稿（三轮设计访谈收敛 + 双轴评审修复，2026-09）· 双摄适配修订（4 设备 5 机位）
