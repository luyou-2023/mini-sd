"""烟测素材生成器验证：每个样本均可由真实媒体解码器读取。"""

import subprocess  # 用独立进程验证脚本入口。
import sys  # 沿用执行测试的 Python 环境。

from mini_sd.data import ImageDataset, VideoDataset  # 使用真实媒体数据集回读。


def test_smoke_data_script_creates_paired_media(tmp_path):  # 验证脚本产出可直接训练。
    result = subprocess.run([sys.executable, "scripts/make_smoke_data.py", "--out", str(tmp_path)], capture_output=True, text=True)  # 在新进程执行生成器。
    assert result.returncode == 0, result.stderr  # 暴露真实错误而非吞掉异常。
    assert len(ImageDataset(tmp_path / "images", size=32)) >= 2  # 验证确实生成多张图片。
    assert len(VideoDataset(tmp_path / "videos", size=32, frames=4)) >= 1  # 验证至少有一个可训练视频。
    video, caption = VideoDataset(tmp_path / "videos", size=32, frames=4)[0]  # 实际解码视频。
    assert video.shape == (3, 4, 32, 32)  # 验证烟测视频确有四帧。
    assert caption  # 验证视频训练时有可学习的文本标签。
