# Mini SD 技术约定（2026-10-29）

## 目标与术语

- 用户选定路线 1：**从零实现轻量扩散模型**。本项目不是复刻 Stable Diffusion 或 MiniMax H3 的权重/质量。
- `生图`：输入训练过的模型和可选文本条件，输出 PNG；`生视频`：输出 MP4，无默认声音；`训练`：先训练图像 VAE，再分别训练图像去噪器和视频去噪器；`推理`：从各自检查点采样并用 VAE 解码。
- 明天的 GPU 服务器细节未知；不得写死 CUDA 型号、绝对路径或预训练权重。今天只做本机小样本端到端运行，不声称真实画质。
- 内容：Python + PyTorch，所有工程自写代码的关键逻辑逐行中文注释；源码中禁止假造训练指标。

## 范围与取舍

- VAE：RGB 32/64 方图 → 下采样 4 倍的 4 通道潜变量，重参数化，重建 + KL；使用固定缩放 1.0，不承诺与 SD VAE 兼容。
- 图像扩散：自写轻量 DiT 风格 Transformer 去噪器，将潜变量切成 2×2 patch token；多头自注意力 + 前馈残差块联合时间嵌入与字节哈希文本条件；噪声预测 DDPM 训练，DDIM 推理，CFG 可调。无预训练语言语义。
- 视频扩散：同结构但独立权重的时空 patch Transformer，每帧按 2×2 patch tokenize，再把 T×H×W 全部 token 放进同一次自注意力，允许跨帧信息流；同一 VAE 逐帧编码/解码。固定帧数 4/8，不是独立图片拼接、长视频或实时渲染。注意力显存随 token 数平方增长。
- 旧版卷积去噪器检查点与新架构**不兼容**；VAE 权重结构不变，可复用。新 image/video 检查点显式保存 `architecture=dit-v1`，恢复训练及推理都需提前拒绝旧检查点并给出中文错误；禁止静默混用。
- 验收：测试确认真实 `nn.MultiheadAttention` 的参数参与图像/视频扩散训练梯度，视频改变某帧后其它帧 token 的预测会受影响；CPU 端到端 PNG/MP4 可解码；HTML 书不再称 U-Net/三维卷积为当前去噪骨干。
- 数据：`data/images` 的 PNG/JPG + 同名 `.txt`；`data/videos` 的 MP4 + 同名 `.txt`，用 OpenCV 解码，固定裁剪/缩放；缺字幕按空条件训练。
- 无外部权重、无模型自动下载、无付费调用、无服务器操作。安全加载检查点仅接收本工程可信产物，`weights_only=True`。
- CLI：`python -m mini_sd train-vae|train-image|train-video|sample-image|sample-video`；`--device auto|cpu|mps|cuda`、seed、数据/检查点/尺寸等显式选项。
- 输出：检查点（配置/权重/优化器与训练步数）、PNG、无声 H.264/yuv420p MP4；模型生成结果仅供实验，不承诺人物真实感。
- HTML 书：`book/index.html` 自包含，无外链字体/JS；原理先于代码，章节含实际项目模块、数据准备、训练/推理命令、逐行核心代码走读、GPU 次日运行清单、限制与故障排查。

## 验收命令

1. `python3 -m pytest -q`：数据集错误边界、VAE/DiT 张量形状、图像与视频注意力梯度、跨帧信息流、旧检查点拒绝、DDIM 可复现、CLI 生成文件、视频流帧数。
2. 用本地程序绘制小型几何测试集（明确非真实画质基准）；`train-vae --steps 1` → `train-image --steps 1` → `sample-image --steps 2` → `train-video --steps 1` → `sample-video --steps 2`，实际输出可解码 PNG/MP4；不得替换成 mock。
3. `ffprobe` 验证 MP4 真实视频流、画幅、帧数；PIL 验证 PNG；检查检查点可在新进程加载。
4. 静态检查 Python 文件可编译、关键函数中文行级注释；HTML 标题/目录/章节与 DiT 真实源码及命令相符且 UTF-8 有效。

## 明日待决（不阻塞离线开发）

- GPU/显存、驱动、CUDA、训练图片/视频数量与授权、目标画幅、期望时长与质量；据此调 batch/steps/尺寸并做真实数据训练。严禁将今天的几何烟测描述成画质验证。
