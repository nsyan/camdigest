# 人脸识别采用 InsightFace-REST sidecar 容器

人脸识别不嵌入流水线进程，而是部署为独立 sidecar 容器（SthPhoenix/InsightFace-REST，FastAPI 封装），主流水线通过 HTTP 调用。模型用 buffalo_l（SCRFD 检测 + ArcFace 特征，LFW 99.86%）。

**本 NAS（绿联 DXP4800 Pro，i3-1315U）无 NVIDIA GPU**，TensorRT 加速路径无效，容器以 ONNX Runtime CPU 模式运行（单帧约 100ms，日批约千帧级，够用）。选 sidecar 而非直接嵌入 insightface Python 库：① 视觉能力与主流水线解耦，未来换 GPU 机器只改容器不改代码；② 人脸库的增删查有现成 REST 界面。家庭自用属非商用，buffalo_l 预训练模型许可无碍；若未来商用需购 insightface.ai 授权或以开源数据自训练。

## Considered Options

- 直接 pip 嵌入 insightface 库：少一个容器，但视觉库与流水线强耦合，且无现成人脸库管理
- Exadel CompreFace：UI 成熟但更重（Java 全家桶），模型同为 ArcFace 系，无增量收益
