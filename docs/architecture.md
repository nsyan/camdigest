# CamDigest 技术架构

> 3 机位监控录像 → 每日精华视频（5/10/30/60 分钟四档）+ 带图飞书日报
> 术语见 [CONTEXT.md](../CONTEXT.md)；关键决策见 [ADR 目录](./adr/)

## 1. 目标与非目标

**目标**
- 只读消费已有录像目录，不负责录制
- 本地免费预筛承担 95%+ 过滤，API 成本压在 ¥1-3/天
- 人脸识别（家人身份）驱动精华筛选与日报叙事
- 视频+音频一起理解（识别模型为全模态）
- 每日产物：四档精华视频（NAS 本地）+ 飞书在线文档日报 + 家庭群推送

**非目标（明确不做）**
- 录制、实时告警、企业微信、腾讯文档、飞书云盘视频上传、模型主备降级

## 2. 运行环境

| 项 | 值 |
|---|---|
| 硬件 | 绿联 DXP4800 Pro · i3-1315U（2P+4E/8T，睿频 4.5GHz）· 无独显 |
| 部署 | Docker Compose，双容器 |
| 技术栈 | Python 3.12 · FastAPI（M2 起）· APScheduler · FFmpeg 子进程 · ONNX Runtime CPU · SQLite |
| 语言选型依据 | InsightFace/faster-whisper 仅 Python 原生；人脸走 REST sidecar 后保留了换语言余地 |

## 3. 部署拓扑

```yaml
services:
  camdigest:            # 主应用：流水线 + 调度 + (M2) Web
    image: camdigest:latest
    volumes:
      - /vol1/nvr:/media/nvr:ro        # 3 机位录像根目录（只读）
      - ./data:/data                    # 产物 + SQLite + 人脸库
      - ./config:/app/config
    depends_on: [insightface]

  insightface:          # 人脸识别 sidecar（ADR-0003）
    image: sthphoenix/insightface-rest
    environment: [USE_ONNX=1]         # CPU 模式，无 TensorRT
    volumes:
      - ./models/insightface:/models   # buffalo_l 模型缓存
```

`/data` 目录规划：

```
/data
├── camdigest.db          # SQLite（唯一数据库）
├── faces/                # 人脸库建档：/faces/{身份名}/*.jpg
├── keyframes/{date}/     # 事件关键帧（日报贴图源）
├── highlights/{date}/    # 四档精华：精华_5min.mp4 ...
└── reports/{date}.md     # 日报 Markdown 原稿
```

## 4. 流水线（日批，默认 03:00 跑前一天）

```
┌─────────────── 本地·全免费（CPU）───────────────┐
│ S1 索引    扫描机位目录 → ffprobe 元数据入库      │
│ S2 预筛    FFmpeg scene/motion 检测 → 候选区间    │
│ S3 人形    YOLO11n ONNX：区间内采样帧人形计数      │
│ S4 人脸    有人的帧 → InsightFace-REST → 身份标签  │
│ S5 音频    silencedetect 标记 → 有声段 faster-     │
│            whisper(int8) 本地转写为文本提示        │
└──────────────┬───────────────────────────────────┘
               ↓ 候选片段 ~200/天（含标签元数据）
┌─────────────── API 层 ──────────────────────────┐
│ S6 识别   识别模型逐段：视频+音频 → 事件 JSON      │
│           （单角色，见 ADR-0001；失败重试2次后     │
│             标记失败，人工补跑）                   │
└──────────────┬───────────────────────────────────┘
               ↓ 事件库
┌─────────────── 产物层 ──────────────────────────┐
│ S7 归并   跨段/跨机位合并事件（人脸ID为key）、     │
│           异常判定、关键帧提取                     │
│ S8 日报   报告模型：事件清单 → 五板块 Markdown     │
│ S9 发布   飞书：建docx→传关键帧图→写blocks→推群卡片│
│ S10 剪辑  单池选段(60min池) → FFmpeg 切片拼接     │
│           → 四档导出（ADR-0002）                   │
└─────────────────────────────────────────────────┘
```

各阶段落库可断点续跑（`jobs` 表按 stage 记录状态）；阶段内串行、阶段间顺序执行，S3-S5 可按片段并行（进程池 2-3 并发，避免挤占 NAS 转码）。

**CPU 吞吐预估（i3-1315U）**：YOLO11n ~30-80ms/帧、buffalo_l ~100ms/帧；日批约 900 帧 YOLO + 数百帧人脸 + 有声段转写（估 20-30 分钟），整夜窗口绰绰有余。若转写拖尾，降级为"仅 S6 选中的片段补转写"。

## 5. 数据模型（SQLite）

```sql
cameras     (id, name, dir, pattern, timezone)
media_files (id, camera_id, path, start_ts, end_ts, duration)
segments    (id, media_file_id, start_s, end_s, motion_score,
             person_count, face_labels,     -- JSON: [{identity, conf, ts}]
             audio_active, transcript, recognition_status, created_at)
events      (id, date, start_ts, end_ts, category, score, title,
             description, camera_ids, segment_ids, keyframe_path,
             is_anomaly, created_at)
highlights  (id, date, tier_minutes, file_path, duration, bytes,
             event_ids, created_at)
reports     (id, date, md_path, feishu_doc_url, feishu_msg_id, created_at)
jobs        (id, date, stage, status, error, started_at, finished_at)
identities  (id, name, dir, unknown_cluster, enrolled_at)  -- 人脸库登记
```

全部产物**永久保留**（用户决策），`/data` 挂大容量卷（精华约 2-4GB/天）。

## 6. 模型接入层（核心抽象）

```python
class RecognitionModel(Protocol):
    def analyze(self, video_path: Path, hint: SegmentHint) -> EventDraft:
        """视频+音频 → 结构化事件草稿。hint 含人脸标签/转写文本，注入 prompt。"""

class ReportModel(Protocol):
    def write(self, date: str, events: list[Event]) -> str:
        """事件清单 → 日报 Markdown（含 {{img:keyframe}} 图位标记）"""
```

- 每角色一个配置节点，`protocol: openai | anthropic`（识别角色仅 openai）
- OpenAI 适配器处理百炼私有 content-part（`video_url`/`input_audio`）与标准协议的差异
- Anthropic 适配器仅实现 Messages API 文本调用（供报告角色换 Claude）
- 响应强制 JSON schema 解析；解析失败重试 2 次（附错误反馈），仍失败则该片段标记 `recognition_status=failed`

**默认配置**：

| 角色 | 默认 | 端点 |
|---|---|---|
| 识别 | `qwen3.8-omni-flash` | DashScope compatible-mode（OpenAI 协议） |
| 报告 | `deepseek-v4.1-flash` | DeepSeek API（OpenAI 协议，可切 anthropic） |

**识别 prompt 要点**：身份标签直填（"画面中为：妈妈、未知-02"），音频转写作上下文；输出 `{category: family|stranger|visitor|animal|vehicle|empty, people: [...], score: 0-100, title, description}`，规则表兜底（family 70 起步、empty 10，模型分高于规则分 20+ 才采信——沿用 TimeCut 验证过的混合打分）。

## 7. 精华剪辑与档位（ADR-0002）

单池四步选段（复用 TimeCut 验证过的算法，参数化）：

1. 每小时保底 1 个最高分事件（全天覆盖含夜间）
2. 每小时最多补到 N 段（默认 3），与已选间隔 <30 分钟视为同一事件跳过
3. 按分数补齐至 60 分钟目标
4. 仍不足则放开去重兜底

单事件 ≤20s（取**家人出现最密集**窗口——人脸时间戳密度优先，scene 密度次之）；按事件真实时间排序；FFmpeg 先切 TS 片段再 concat 统一时间轴，`-c:a aac` 重编码保原声。四档 = 段池 Top-N 子集分别导出。

## 8. 飞书发布器

自建应用（配置 `app_id`/`app_secret`/`folder_token`/`chat_id`）：

```
tenant_access_token
→ POST /docx/v1/documents                    # 建日报（folder_token 指定目录）
→ POST /drive/v1/medias/upload_all           # 传关键帧（parent_type=docx_image）
→ PATCH /docx/v1/documents/{id}/blocks/...   # Markdown→blocks 写入（图片块）
→ POST  /im/v1/messages                      # interactive 卡片推家庭群
```

所需权限：`docx:document`、`drive:file`、`im:message:send_as_bot`。日报五板块：①今日总览（报告模型生成）②时间线（带关键帧图）③人物出没统计 ④异常事件 ⑤精华清单（MVP 为本地文件信息 + LAN 提示，M2 起变 `http://NAS:端口` 点播链接）。

## 9. 配置结构

```yaml
schedule: { daily_at: "03:00", timezone: Asia/Shanghai }
cameras:
  - { id: gate,  name: 大门, dir: /media/nvr/gate,  pattern: "*.mp4" }
  - { id: yard,  name: 院子, dir: /media/nvr/yard,  pattern: "*.mp4" }
  - { id: hall,  name: 大厅, dir: /media/nvr/hall,  pattern: "*.mp4" }
models:
  recognition: { protocol: openai, base_url: https://dashscope.aliyuncs.com/compatible-mode/v1,
                 model: qwen3.8-omni-flash, api_key: ${DASHSCOPE_API_KEY} }
  report:      { protocol: openai, base_url: https://api.deepseek.com/v1,
                 model: deepseek-v4.1-flash, api_key: ${DEEPSEEK_API_KEY} }
prefilter:
  scene_threshold: 0.03
  whisper: { enabled: true, model: small-int8 }
faces:   { rest_url: http://insightface:18080, threshold: 0.4,
           registry_dir: /data/faces }
highlight: { tiers: [5, 10, 30, 60], max_segment_seconds: 20,
             min_per_hour: 1, max_per_hour: 3, event_gap_minutes: 30 }
publish:
  feishu: { folder_token: xxx, chat_id: xxx, app_id: ${FEISHU_APP_ID},
            app_secret: ${FEISHU_APP_SECRET} }
storage: { data_dir: /data, retention: permanent }
```

## 10. 代码结构（M1）

```
src/camdigest/
├── cli.py            # camdigest run [--date] / enroll-faces / backfill
├── config.py         # pydantic-settings + ${ENV} 展开
├── db.py             # SQLAlchemy 2.0 + SQLite
├── scheduler.py      # APScheduler 日批入口
├── pipeline/         # s1_index ... s10_clip（每阶段独立模块）
├── llm/              # contracts + openai_adapter + anthropic_adapter
├── publish/          # feishu.py（M1 唯一渠道）
└── web/              # M2
tests/                # 每阶段单测：fixture 用短样本录像
```

## 11. 里程碑

| 里程碑 | 内容 | 验收 |
|---|---|---|
| **M1** | CLI 全链路：S1-S10 + 人脸文件夹建档 + 补跑 | 对任意历史日期一键产出四档精华 + 飞书日报 |
| **M2** | Web UI：进度/历史浏览、精华 LAN 点播、人脸库界面化管理、日报板块⑤变可点播链接 | 家人手机浏览器直接看 |
| **M3** | watch 增量：录像落盘即预筛索引，日批只做识别汇总 | 批次耗时段差明显缩短 |

## 12. 风险与待校准

| 项 | 说明 | 校准方式 |
|---|---|---|
| qwen3.8-omni-flash 视频计费率 | 挂牌价待接入实测（视频 token/秒未公布确认） | 首周记录每日 token 用量，超 ¥5/天则收紧预筛阈值 |
| 未知脸体验 | 聚类编号需人工归档，漏归档会以"未知-XX"出现在日报 | M2 做"代表照点选归档"交互 |
| CPU 夜批窗口 | whisper 是最大变量 | 有声段占比高时只转写 S6 选中片段 |
| 飞书图片频控 | 单文档贴图量大时注意 upload QPS | 关键帧压缩至 ≤200KB，批量间 sleep |
