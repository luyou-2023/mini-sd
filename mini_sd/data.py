"""从图片和 MP4 读取有文本配对的真实训练样本。"""

from pathlib import Path  # 使用跨平台路径管理训练目录。

import cv2  # 解码 MP4 中真实的视频帧。
import numpy as np  # 排列图像数组并均匀抽帧。
from PIL import Image  # 裁剪、缩放与规范化图片。
import torch  # 将像素转换为训练张量。
from torch.utils.data import Dataset  # 接入 PyTorch DataLoader。


def _prepare(image: Image.Image, size: int) -> torch.Tensor:  # 对图片和视频帧执行相同预处理。
    if size < 16 or size % 4:  # 确保 VAE 两次下采样后仍有有效尺寸。
        raise ValueError("图片边长须大于等于 16 且可被 4 整除")  # 及早拒绝坏配置。
    image = image.convert("RGB")  # 统一输入到三通道 RGB。
    side = min(image.size)  # 找到可裁剪的最大中心方形边长。
    left = (image.width - side) // 2  # 计算中心裁剪的左边界。
    top = (image.height - side) // 2  # 计算中心裁剪的上边界。
    image = image.crop((left, top, left + side, top + side))  # 保持主体的中心部分。
    image = image.resize((size, size), Image.Resampling.LANCZOS)  # 统一训练画幅。
    array = np.asarray(image, dtype=np.float32).copy()  # 转换为可写的浮点数组。
    return torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1.0  # 输出 CHW 且范围为 [-1,1]。


def _caption(path: Path) -> str:  # 读取与媒体同名的短文字说明。
    text = path.with_suffix(".txt")  # 根据媒体文件名定位旁路标注。
    return text.read_text(encoding="utf-8").strip() if text.exists() else ""  # 无标注时显式使用空条件。


class ImageDataset(Dataset):  # 使用单张图片训练 VAE 和图像扩散器。
    def __init__(self, folder: str | Path, size: int = 32):  # 确定样本位置与输出尺寸。
        self.folder = Path(folder)  # 将调用参数归一化为路径对象。
        self.size = size  # 保存生成与训练一致的像素边长。
        self.paths = sorted(path for path in self.folder.iterdir() if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")) if self.folder.is_dir() else []  # 只选择可读图像。
        if not self.paths:  # 对空目录及时失败。
            raise ValueError(f"目录中没有图片：{self.folder}")  # 避免空批次伪装成功。

    def __len__(self) -> int:  # 提供 DataLoader 所需样本数。
        return len(self.paths)  # 返回真实文件数量。

    def __getitem__(self, index: int):  # 按索引加载并规范化一张图。
        path = self.paths[index]  # 定位单个训练素材。
        with Image.open(path) as image:  # 避免在 DataLoader 中泄漏文件句柄。
            tensor = _prepare(image, self.size)  # 做中心裁剪、缩放和像素归一化。
        return tensor, _caption(path)  # 将画面与对应短文本一起返回。


class VideoDataset(Dataset):  # 读取固定帧数的视频片段用于三维扩散训练。
    def __init__(self, folder: str | Path, size: int = 32, frames: int = 4):  # 配置目录和时空尺寸。
        self.folder = Path(folder)  # 归一化数据目录。
        self.size = size  # 保持与 VAE 训练尺寸一致。
        self.frames = frames  # 规定每个样本的时间长度。
        if frames < 2:  # 视频分支要求至少两帧。
            raise ValueError("视频至少需要两帧")  # 阻止伪视频训练。
        self.paths = sorted(self.folder.glob("*.mp4")) if self.folder.is_dir() else []  # 只选择 MP4 数据。
        if not self.paths:  # 防止没有媒体时空转。
            raise ValueError(f"目录中没有 MP4：{self.folder}")  # 标注真实错误来源。

    def __len__(self) -> int:  # 提供真实视频文件数量。
        return len(self.paths)  # 每个文件对应一个训练片段。

    def __getitem__(self, index: int):  # 从实际 MP4 解码并均匀抽帧。
        path = self.paths[index]  # 定位当前视频素材。
        capture = cv2.VideoCapture(str(path))  # 打开 OpenCV 视频解码器。
        if not capture.isOpened():  # 立即检测无法打开的文件。
            raise ValueError(f"无法解码视频：{path}")  # 禁止用静态占位帧掩盖损坏视频。
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # 从 MP4 元数据读取帧数而不缓存帧。
        if frame_count < self.frames:  # 媒体没有可靠且足够长的帧数信息时拒绝。
            capture.release()  # 在报错前释放视频句柄。
            raise ValueError(f"视频帧数不足或元数据不可用：{path}")  # 提醒先用 FFmpeg 重封装。
        positions = np.linspace(0, frame_count - 1, self.frames).round().astype(int)  # 在整段时间均匀选帧。
        selected = set(int(position) for position in positions)  # 只为目标帧做图片预处理。
        decoded = {}  # 最多保留用户要求的帧数，不缓存整段视频。
        frame_index = 0  # 为顺序解码的每个真实帧标记索引。
        try:  # 即使解码异常也释放文件句柄。
            while frame_index <= int(positions[-1]):  # 最后一帧取齐后不再读取其余画面。
                ok, frame = capture.read()  # 顺序提取真实压缩视频帧。
                if not ok:  # 播放到片尾或遇到解码错误。
                    break  # 在下方报告缺帧，不制造占位帧。
                if frame_index in selected:  # 只预处理均匀选择的目标帧。
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转为 RGB 顺序。
                    decoded[frame_index] = _prepare(Image.fromarray(rgb), self.size)  # 最多持有目标数量的张量。
                frame_index += 1  # 推进真实视频帧位置。
        finally:  # 不依赖 Python 回收机制管理句柄。
            capture.release()  # 释放操作系统视频资源。
        if len(decoded) != self.frames:  # 核对容器声称的帧数与真实可解码帧数。
            raise ValueError(f"视频实际可解码帧数不足 {self.frames}：{path}")  # 防止损坏 MP4 被静默补帧。
        clip = torch.stack([decoded[int(position)] for position in positions], dim=1)  # 形成 [C,T,H,W] 视频片段。
        return clip, _caption(path)  # 返回真实片段及配对提示词。
