"""构造几何烟测样本：仅用于验证训练/推理链路，不代表真实画质。"""

import argparse  # 读取烟测数据的输出目录。
from pathlib import Path  # 管理图片与视频目录。

import cv2  # 将实际移动的几何帧编码成 MP4。
import numpy as np  # 转换 RGB 图片为编码器输入。
from PIL import Image, ImageDraw  # 画可识别的方块与圆形。


def make_data(root: Path) -> None:  # 生成配对的图像和短视频。
    images = root / "images"  # 定义单帧素材目录。
    videos = root / "videos"  # 定义多帧素材目录。
    images.mkdir(parents=True, exist_ok=True)  # 创建图像数据目录。
    videos.mkdir(parents=True, exist_ok=True)  # 创建视频数据目录。
    variants = [("red", "红 方块", "rectangle"), ("blue", "蓝 圆形", "ellipse")]  # 仅使用合成的两个明确类别。
    for color, caption, shape in variants:  # 为每一种类别准备图像和时间序列。
        for offset in (0, 3, 6, 9):  # 为图像类别画多个位置变化。
            image = Image.new("RGB", (32, 32), (225, 225, 225))  # 建立实际图片像素。
            pen = ImageDraw.Draw(image)  # 创建图像绘图器。
            bounds = (4 + offset, 9, 13 + offset, 18)  # 明确主体所在矩形区域。
            if shape == "rectangle":  # 为红色类别选方块。
                pen.rectangle(bounds, fill=color)  # 实际画出红方块。
            else:  # 为蓝色类别选圆形。
                pen.ellipse(bounds, fill=color)  # 实际画出蓝圆形。
            stem = f"{color}_{offset}"  # 保持图与标注文件同名。
            image.save(images / f"{stem}.png")  # 输出可解码 PNG 训练图。
            (images / f"{stem}.txt").write_text(caption, encoding="utf-8")  # 记录训练条件。
        target = videos / f"{color}.mp4"  # 建立独立视频输出路径。
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (32, 32))  # 初始化真实 MP4 视频编码器。
        if not writer.isOpened():  # 检测系统编解码器不可用的情况。
            raise RuntimeError(f"无法创建烟测 MP4：{target}")  # 阻止出现空白伪文件。
        try:  # 保证编码器总是完成封装。
            for offset in range(8):  # 视频使用八张实际不同位置的帧。
                image = Image.new("RGB", (32, 32), (225, 225, 225))  # 为每个时间点独立建立图片。
                pen = ImageDraw.Draw(image)  # 获得逐帧绘图器。
                bounds = (3 + offset * 2, 9, 12 + offset * 2, 18)  # 让主体随时间向右移动。
                if shape == "rectangle":  # 生成矩形运动轨迹。
                    pen.rectangle(bounds, fill=color)  # 写入当前帧方块。
                else:  # 生成圆形运动轨迹。
                    pen.ellipse(bounds, fill=color)  # 写入当前帧圆形。
                writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))  # 转成 BGR 交给视频编码器。
        finally:  # 无论写帧是否出错都封装容器。
            writer.release()  # 写出可由 FFprobe 读取的媒体索引。
        (videos / f"{color}.txt").write_text(caption, encoding="utf-8")  # 视频提示词与文件名配对。
    print(f"已生成几何烟测数据：{images} 与 {videos}")  # 明确标注这不是实拍素材。


if __name__ == "__main__":  # 支持从命令行直接创建烟测素材。
    parser = argparse.ArgumentParser(description="只生成合成的几何测试素材")  # 显示脚本用途。
    parser.add_argument("--out", required=True)  # 要求显式指定工作目录。
    options = parser.parse_args()  # 解析命令行配置。
    make_data(Path(options.out))  # 创建真实可读的合成媒体文件。
