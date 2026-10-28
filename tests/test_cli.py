"""命令行端到端测试：验证文件不是伪造的返回值。"""

import json  # 解析 FFprobe 的真实媒体元数据。
import subprocess  # 启动独立 Python 进程与 FFprobe。
import sys  # 使用当前测试解释器保持环境一致。

from PIL import Image  # 检验生成图片可被标准解码器打开。
import torch  # 检查检查点可以安全加载。
import pytest  # 验证旧格式被拒绝时的错误提示。
from mini_sd.__main__ import _load  # 对加载边界做最小回归验证。


def _run(*args):  # 用新进程执行真实命令行入口。
    return subprocess.run([sys.executable, "-m", "mini_sd", *map(str, args)], check=True, capture_output=True, text=True)  # 失败时返回完整异常。


def test_legacy_convolution_checkpoint_is_rejected_before_model_load(tmp_path):  # 不允许旧 U-Net 权重伪装成 DiT。
    path = tmp_path / "legacy.pt"  # 为测试准备旧版风格的权重路径。
    torch.save({"kind": "image", "base": 8, "model": {}}, path)  # 缺失新架构字段的真实 PyTorch 检查点。
    with pytest.raises(ValueError, match="旧版卷积|架构不兼容"):  # 要求错误指向架构而非生硬的权重键不匹配。
        _load(str(path), "image")  # 尝试载入旧图像检查点。


def test_image_and_video_train_and_generate(tmp_path):  # 执行训练到媒体导出的完整流程。
    pictures = tmp_path / "images"  # 准备图片训练目录。
    movies = tmp_path / "videos"  # 准备视频训练目录。
    pictures.mkdir()  # 创建图片测试目录。
    movies.mkdir()  # 创建视频测试目录。
    Image.new("RGB", (32, 32), "red").save(pictures / "red.png")  # 创建确实可读的训练图。
    (pictures / "red.txt").write_text("红 色", encoding="utf-8")  # 配对短文本。
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(pictures / "red.png"), "-frames:v", "5", "-pix_fmt", "yuv420p", str(movies / "red.mp4")], check=True)  # 编码真实训练视频。
    (movies / "red.txt").write_text("红 色", encoding="utf-8")  # 配对视频短文本。
    vae = tmp_path / "vae.pt"  # 定义独立 VAE 检查点。
    image_model = tmp_path / "image.pt"  # 定义图像扩散检查点。
    video_model = tmp_path / "video.pt"  # 定义视频扩散检查点。
    image_out = tmp_path / "output.png"  # 定义最终图片文件。
    video_out = tmp_path / "output.mp4"  # 定义最终 MP4 文件。
    _run("train-vae", "--data", pictures, "--out", vae, "--steps", 1, "--size", 32, "--base", 8, "--device", "cpu")  # 训练真实 VAE 一步。
    _run("train-image", "--data", pictures, "--vae", vae, "--out", image_model, "--steps", 1, "--base", 8, "--diffusion-steps", 16, "--device", "cpu")  # 更新图像模型。
    _run("sample-image", "--vae", vae, "--model", image_model, "--out", image_out, "--prompt", "红 色", "--sample-steps", 2, "--device", "cpu")  # 从检查点真实推理。
    _run("train-video", "--data", movies, "--vae", vae, "--out", video_model, "--steps", 1, "--base", 8, "--frames", 4, "--diffusion-steps", 16, "--device", "cpu")  # 更新时空模型。
    _run("sample-video", "--vae", vae, "--model", video_model, "--out", video_out, "--prompt", "红 色", "--sample-steps", 2, "--device", "cpu")  # 从视频模型导出 MP4。
    with Image.open(image_out) as image:  # 由独立图片解码器打开结果。
        assert image.size == (32, 32)  # 验证输出分辨率。
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=codec_name,nb_read_frames,width,height", "-of", "json", str(video_out)], capture_output=True, text=True, check=True)  # 由 FFprobe 读取真实视频流。
    stream = json.loads(probe.stdout)["streams"][0]  # 获取媒体流第一项。
    assert (stream["width"], stream["height"], int(stream["nb_read_frames"])) == (32, 32, 4)  # 验证尺寸与真实帧数。
    assert torch.load(video_model, map_location="cpu", weights_only=True)["kind"] == "video"  # 验证模型文件可安全载入。
    assert torch.load(video_model, map_location="cpu", weights_only=True)["architecture"] == "dit-v1"  # 确认不会误认旧卷积模型。
