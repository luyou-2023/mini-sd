# Mini SD Lab

从零实现的 **Python / PyTorch 潜空间扩散 Transformer 实验**：训练 VAE、图像 DiT、视频时空 DiT；DDPM 加噪训练、DDIM/CFG 推理；输出 PNG 和**无声** MP4。Transformer 是预测噪声的骨干，扩散是训练/采样机制，二者都在。源码关键逻辑附中文行级注释。**没有预训练权重；一步烟测只验证代码链路，不能产生真实人物或高质量短剧。**

## 独立边界

本工程只位于 `/Users/luke/workspace/mini-sd`；不依赖、读取或修改 LiveShop。服务器尚未接入，所有 GPU 性能、显存和画质都待真实服务器验证。

## 安装

Linux GPU：先按服务器的 CUDA/驱动安装官方 PyTorch 对应轮子，再在仓库根目录执行 `python3 -m venv .venv && . .venv/bin/activate && python -m pip install -e '.[test]'`。或使用已安装的 `uv venv .venv && uv pip install --python .venv/bin/python -e '.[test]'`，但需在服务器上选择与 CUDA 对应的 PyTorch 安装源。`ffmpeg` / `ffprobe` 必须在 PATH 可用。不要在未确认 GPU/驱动前假定 CUDA 可用。Mac 可用 `--device mps`，CPU 通用。

## 五个命令

在仓库根目录运行。以下命令只做**1 步几何烟测**，不代表训练收敛：

```bash
python3 scripts/make_smoke_data.py --out data/smoke
python3 -m mini_sd train-vae --data data/smoke/images --steps 1 --size 32 --base 8 --device cpu --out checkpoints/vae.pt
python3 -m mini_sd train-image --data data/smoke/images --vae checkpoints/vae.pt --steps 1 --base 8 --diffusion-steps 16 --device cpu --out checkpoints/image.pt
python3 -m mini_sd sample-image --vae checkpoints/vae.pt --model checkpoints/image.pt --prompt '红 方块' --sample-steps 2 --seed 7 --device cpu --out outputs/sample.png
python3 -m mini_sd train-video --data data/smoke/videos --vae checkpoints/vae.pt --steps 1 --base 8 --frames 4 --diffusion-steps 16 --device cpu --out checkpoints/video.pt
python3 -m mini_sd sample-video --vae checkpoints/vae.pt --model checkpoints/video.pt --prompt '蓝 圆形' --sample-steps 2 --fps 8 --seed 7 --device cpu --out outputs/sample.mp4
python3 -m pytest -q
```

## 用真实数据训练

- 图片：`data/images/*.png`、`*.jpg`、`*.webp`；视频：`data/videos/*.mp4`。每个媒体文件可放一个同名 UTF-8 `.txt` 提示词，例如 `0001.png` 与 `0001.txt`；缺失时为空条件。**中文词之间加空格**，例如 `红 衣 女子`，因为本项目使用空格分词＋稳定哈希嵌入，绝非预训练语言模型。
- 图像会中心裁为方形、缩至 `--size`；视频按整段均匀抽取 `--frames` 帧，输出方形无声短片。输入视频宜预切成数秒短片：解码顺序遍历真实帧，但仅保留被抽中的目标帧张量。
- 建议先从少量**有权使用**的统一风格数据训练 VAE，再训练图像/视频扩散 Transformer。增大到 `--size 64 --base 32 --steps ...` 前先试一个批次，并根据 GPU 显存调整 `--batch-size` / `--frames`；全局注意力对总 token 数很敏感，视频显存尤其容易迅速增长。Transformer 的 `--base` 必须为 4 的倍数；训练与对应 VAE/checkpoint 的尺寸必须一致。
- 断点续训加 `--resume checkpoints/<kind>.pt`；`--steps` 表示本次新增的更新次数。所有样本来自自己的授权数据，不带模型自动下载。
- 推理可用 `--guidance 2.0`（训练时默认 10% 空条件）；大于 1 并非保证更好，不经足量配对数据训练时文本几乎没有语义。视频模型与图像模型**不同权重**，不能混用。
- 检查点是 PyTorch 数据字典，只从可信来源使用；加载时启用 `weights_only=True`。新图像/视频权重有 `architecture=dit-v1` 标记，**旧版卷积去噪权重不能恢复或推理，须重新训练**；VAE 结构不变，可以复用。默认 `--device auto` 优先 CUDA，然后 MPS，否则 CPU。

## 检查与学习

- `ffprobe -v error -show_entries stream=codec_name,nb_frames,width,height -of json outputs/sample.mp4` 查看真实视频流。
- `python3 -m mini_sd train-video --help` 可查看全部参数；`python3 -m pytest -q` 可验证本机实现。
- 教学书：`book/index.html`，完全内联、可离线打开；源码与真实运行命令均有说明。

## 明确限制

这是教育/研究用小模型，不是 Stable Diffusion 或 MiniMax H3 权重；**没有角色一致性参考、跨镜头续接、长视频分段、音轨/配音与商业可用的语义质量**。视频采用跨帧时空自注意力而非静态图拼接；训练收敛、输出动作质量和 GPU 显存峰值仍需服务器上的真实数据测试。不要用一次测试输出做作品效果承诺。
