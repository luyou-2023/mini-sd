"""真实 PNG/MP4 数据与旁路文本标注测试。"""

import subprocess  # 调用系统 FFmpeg 制作真实测试视频。

from PIL import Image  # 创建可读图片文件。
import pytest  # 验证错误边界。

from mini_sd.data import ImageDataset, VideoDataset  # 引入待实现的数据集。
import mini_sd.data as data_module  # 记录真实视频预处理调用次数。


def test_image_dataset_reads_caption_and_normalizes(tmp_path):  # 验证图像训练输入范围。
    Image.new("RGB", (40, 32), (255, 0, 0)).save(tmp_path / "red.png")  # 保存红色测试图。
    (tmp_path / "red.txt").write_text("红 色", encoding="utf-8")  # 保存与图片同名的训练条件。
    image, prompt = ImageDataset(tmp_path, size=32)[0]  # 经真实图片解码读取一个样本。
    assert image.shape == (3, 32, 32)  # 确认中心裁剪与通道次序。
    assert image.min() >= -1 and image.max() <= 1  # 确认归一化范围。
    assert prompt == "红 色"  # 确认文本配对成功。


def test_video_dataset_decodes_temporal_clip(tmp_path):  # 验证视频帧真正来自 MP4。
    Image.new("RGB", (32, 32), (255, 0, 0)).save(tmp_path / "frame.png")  # 准备真实编码素材。
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(tmp_path / "frame.png"), "-frames:v", "5", "-pix_fmt", "yuv420p", str(tmp_path / "clip.mp4")], check=True)  # 编出五帧视频。
    (tmp_path / "clip.txt").write_text("红 色", encoding="utf-8")  # 添加与视频同名的提示词。
    video, prompt = VideoDataset(tmp_path, size=32, frames=4)[0]  # 解码真实 MP4 的四帧。
    assert video.shape == (3, 4, 32, 32)  # 保持 [颜色, 帧, 高, 宽] 顺序。
    assert prompt == "红 色"  # 保留训练配对文本。


def test_long_video_only_preprocesses_selected_frames(tmp_path, monkeypatch):  # 限制长视频的图片张量驻留量。
    Image.new("RGB", (32, 32), (0, 0, 255)).save(tmp_path / "frame.png")  # 创建真实视频源帧。
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(tmp_path / "frame.png"), "-frames:v", "40", "-pix_fmt", "yuv420p", str(tmp_path / "long.mp4")], check=True)  # 编出超过目标帧数的媒体。
    original = data_module._prepare  # 保留真实像素预处理逻辑。
    count = [0]  # 记录被实际转换为张量的帧数。
    def counted(image, size):  # 为真实预处理加可计数包装。
        count[0] += 1  # 每处理一帧就记录一次。
        return original(image, size)  # 仍执行原本的真实预处理。
    monkeypatch.setattr(data_module, "_prepare", counted)  # 仅观察预处理量，不伪造视频帧。
    clip, _ = VideoDataset(tmp_path, size=32, frames=4)[0]  # 解码完整媒体并选择四帧。
    assert clip.shape == (3, 4, 32, 32)  # 确认真实训练数据仍然完整。
    assert count[0] == 4  # 不把其余三十余帧转换成驻留张量。


def test_empty_image_folder_is_rejected(tmp_path):  # 验证空文件夹不可静默训练。
    with pytest.raises(ValueError, match="没有"):  # 检验对使用者可读的错误提示。
        ImageDataset(tmp_path, size=32)  # 尝试加载空训练集。
