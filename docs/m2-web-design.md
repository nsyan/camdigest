# CamDigest M2（Web UI）设计

> 术语见 [CONTEXT.md](../CONTEXT.md)；M1 架构见 [architecture.md](./architecture.md)。本文是 M2 的补充设计，只覆盖增量，不重复 M1 内容。

## 1. 目标与非目标

**目标**（= spec §11 M2 里程碑五件，一次做完）：
1. 精华视频 LAN 点播（按日期浏览、浏览器内播放）
2. 日报渲染 + 板块⑤路径变可点播链接
3. 进度与历史浏览（jobs 可视化）
4. 手动触发 / 任选日期补跑
5. 人脸库界面化管理（未知脸聚类 + 代表照归档）

**验收**：家人手机连家里 Wi-Fi，浏览器打开 `http://NAS_IP:8080`，两步内（首页 → 点日期）播到当天精华；任选历史日期补跑成功且进度可见；归档一个未知簇后，后续日报人物名正确显示。

**非目标**：
- 登录/鉴权、HTTPS、外网访问（LAN-only 已确认；不做内网穿透，预留仅限于"将来加反代不需改代码"）
- 实时监控画面、录像转码/多码率（mp4 直拷 + faststart 已够）
- 重新生成历史日报（重生成要重调报告模型花 token，不做进 M2）
- 前端构建链（无 Node/npm）

## 2. 方案决策

| 决策 | 选择 | 否决项与原因 |
|---|---|---|
| 前端形态 | FastAPI + Jinja2 服务端渲染 + 原生 JS | React SPA：构建链对 LAN-only 家人工具过度工程；现成应用（filebrowser 等）：日报/触发/人脸库落空 |
| 访问范围 | 仅局域网，无鉴权 | 外网：需鉴权+HTTPS+穿透，留作 M2 后独立迭代 |
| 进程模型 | `camdigest web` 单进程：uvicorn + 同进程 APScheduler 后台调度 | 双容器/双进程：SQLite 单写者下徒增协调 |
| 日报链接化 | 渲染时动态替换路径 → URL，不改日报生成与 md 文件 | 重写报告模型 prompt：破坏 M1 产物不可变性 |

## 3. 架构

```
src/camdigest/web/
├── __init__.py
├── app.py          # create_app(settings)：FastAPI 工厂、StaticFiles 挂载、异常处理
├── views.py        # 路由：/ /day/{date} /jobs /run /faces /faces/archive
├── faces.py        # 未知脸聚类扫描 + 归档（唯一新算法）
├── runner.py       # 后台补跑线程 + 与调度的互斥锁
├── templates/      # base.html / index.html / day.html / jobs.html / run.html / faces.html
└── static/         # style.css（归档用原生表单 POST，无需独立 JS——实现时归档）
```

- **模块落位**：spec §10 预留的 `web/`。
- **进程模型**：CLI 新增 `camdigest web [--config]` 子命令 → uvicorn(0.0.0.0:8080，单 worker) + `BackgroundScheduler`（复用 scheduler.build_trigger，job 加互斥锁）。`schedule` 子命令保留，纯调度部署仍可用；Dockerfile 默认 CMD 改 `web`。
- **数据库**：同一个 SQLite，零迁移（M1 尚无生产数据，直接 `create_all` 增表）。**新增一张表**：
  `unknown_faces (id, cluster_id, embedding JSON, segment_id, media_path, ts_in_seg, created_at)`
  ——这是对 spec §5 的增补（M1 统一标「未知」未存向量，聚类需补收集）。
  `identities.unknown_cluster` 语义钉死：归档时该身份行记录它吸收的簇号（历史映射锚点）。
- **并发控制**：`runner.py` 持有进程级 `threading.Lock`；手动补跑线程与 APScheduler 日批 job 竞争同一把锁，拿不到立即返回"已有任务在跑"（jobs 表可见状态）。SQLite 单写者，读路径无阻塞。
- **新增依赖**（进核心依赖）：`fastapi`、`uvicorn`、`jinja2`、`python-multipart`（表单）、`mistune`（日报 md→HTML 渲染）。

## 4. 页面与路由

| 路由 | 页面 | 功能 | 数据源 |
|---|---|---|---|
| `GET /` | 日期列表（倒序：报告✓、四档精华有无、事件数/异常数） | 历史浏览 | reports/highlights/events |
| `GET /day/{date}` | 四档 `<video>` 播放器 + 日报渲染（板块⑤路径→链接）+ 事件时间线（关键帧图、身份名、异常标红） | **点播（家人主页面）** | highlights/events + reports md |
| `GET /highlights/{date}/{file}` 等 | StaticFiles 只读（仅 highlights/、keyframes/、reports/ 三子目录） | 点播 | /data 文件 |
| `GET /jobs` | 日期 × 阶段状态矩阵；failed 标红 + error 悬浮 | 进度浏览 | jobs |
| `GET/POST /run` | 日期表单 + 可选"跑到阶段"（--until 同名值）；POST 后台补跑 → 303 跳 /jobs | 手动触发 | orchestrator.run_day |
| `GET /faces` | 已知身份网格 + 未知簇网格（代表照、出现次数）；"扫描历史"按钮 | 人脸库 | identities/unknown_faces |
| `POST /faces/archive` | 簇 → 归入已有身份或新建身份名 | 人脸库 | 见 §5 |

- 手机优先极简 CSS；家人主路径 = `/` → `/day/{date}` 两步。
- 日期/阶段参数校验：非法日期 404，非法阶段 422。

## 5. 人脸库：聚类与归档

**M1 约束**：S4 对未匹配脸统一标「未知」，未存向量 → M2 需补收集。

**聚类扫描**（`faces.py::scan_unknown_faces(date)`）：
1. 取该日期 `person_count>0` 的候选片段，3 采样帧抽帧 → InsightFace `/extract`；
2. 仅收集 `label_faces` 结果为「未知」的检测向量，落 `unknown_faces`；
3. **贪心近邻聚类**：向量与簇内任一向量余弦 ≥ **0.45**（高于识别阈值 0.4，避免错并）归入该簇，否则新建簇；簇代表照 = det_score 最高的帧（jpg 存 `/data/keyframes/.clusters/{cluster_id}.jpg`）；
4. 触发：日批 `run_day` 完成后自动扫当天（调度 job 的收尾步骤，不进 STAGES——S1-S10 流水线保持稳定）；`/faces` 页手动按钮按日期范围扫历史。

**归档**（`faces.py::archive_cluster(cluster_id, name)`）：
1. 簇代表照写入 `/data/faces/{name}/`（已存在身份则追加）；
2. 复用 `enroll_faces` 重建 `registry.npz` + `identities` 行（`unknown_cluster = cluster_id`）；
3. **历史回填**：按 `unknown_faces` 的（segment_id, ts_in_seg）精确映射，把相关 segments `face_labels` 中的「未知」替换为该身份名；
4. 幂等：重复归档同簇先解绑（清理旧身份对该簇的引用）再执行。

**边界**：回填只改数据库（影响后续 S8 payload 与 /faces 页）；**已生成的日报 md 不重写**；**旧事件的 title/description 保持 S6 生成时原样**（时间线身份名不追溯——与"旧日报不可变"同一立场）；"重新生成历史日报"留待后续。

## 6. 部署

- compose camdigest 服务加 `ports: "8080:8080"`；Dockerfile `CMD ["web", "--config", "/app/config/config.yaml"]`。
- GHCR 镜像自动构建已就绪（推 main 触发），M2 合并即出新镜像，NAS `docker compose pull && up -d` 升级。
- 绿联不设端口映射不出外网；LAN-only 边界靠网络拓扑保证。

## 7. 安全边界（LAN-only 的具体含义）

- 无登录、无 HTTPS：内网明文，明示接受；
- StaticFiles **只挂** `highlights/`、`keyframes/`、`reports/`，**不挂整个 /data**（库与人脸 registry 不出网）；
- `/run` 表单白名单校验（日期格式、阶段名）；
- 上传面为零：归档不传照片（代表照由服务端抽帧），无文件上传攻击面。

## 8. 测试策略

- **TestClient 路由全覆盖**：首页/单日页渲染与空态（404）、日报板块⑤链接替换、jobs 矩阵与 failed 展示、`/run` 触发（后台线程启动、锁重入拒绝、非法参数 422/404）、`/faces` 渲染。
- **聚类纯函数重单测**：贪心近邻归簇、阈值边界（0.45 上下）、代表照选取——向量注入假体，不依赖 sidecar。
- **归档端到端**：假 FaceClient 下归档 → faces 目录出现代表照 → registry.npz 重建 → identities.unknown_cluster 落值 → face_labels 回填断言。
- 播放/Range 请求由 Starlette StaticFiles 保证，不重复测；日批锁互斥用注入两个并发 job 验证。

## 9. 实施切分（供 writing-plans 输入）

Phase A：依赖 + app 骨架 + `/` 与 `/day` 点播页（/day 先只做播放器 + 静态文件；家人价值先落地）
Phase B：/jobs + /run + runner 互斥
Phase C：/day 补入日报渲染（mistune + 板块⑤链接替换）+ 事件时间线
Phase D：聚类扫描 + /faces + 归档（最重）
Phase E：compose/Dockerfile/CLI web 子命令 + E2E
