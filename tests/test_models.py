"""模型行为测试：先证明模块缺失，再实现可运行模型。"""

import torch  # 引入张量运算以检查真实前向传播。

from mini_sd.models import TinyVAE, ImageDenoiser, VideoDenoiser  # 引入待实现的模型。


def test_vae_roundtrip_shapes():  # 验证编码与解码对图像尺寸的契约。
    model = TinyVAE(base=8)  # 构造足够小的真实网络供 CPU 测试。
    image = torch.rand(2, 3, 32, 32)  # 创建两张三通道图像。
    latent, mean, logvar = model.encode(image)  # 真实执行变分编码。
    reconstructed = model.decode(latent)  # 真实执行潜变量解码。
    assert latent.shape == (2, 4, 8, 8)  # 要求潜变量缩小四倍。
    assert mean.shape == logvar.shape == latent.shape  # 要求分布参数一致。
    assert reconstructed.shape == image.shape  # 要求还原尺寸与输入一致。


def test_image_denoiser_preserves_latent_shape():  # 检验二维去噪器输出维度。
    model = ImageDenoiser(base=16)  # 创建二维时空条件去噪器。
    noise = torch.randn(2, 4, 8, 8)  # 模拟被加噪的图像潜变量。
    output = model(noise, torch.tensor([1, 2]), ["红色", "蓝色"])  # 执行条件预测。
    assert output.shape == noise.shape  # 噪声预测必须与输入同维。


def test_video_denoiser_keeps_temporal_dimension():  # 检验三维时空去噪器。
    model = VideoDenoiser(base=16)  # 创建带跨帧卷积的真实网络。
    noise = torch.randn(2, 4, 4, 8, 8)  # 模拟四帧视频潜变量。
    output = model(noise, torch.tensor([1, 2]), ["向左", "向右"])  # 执行预测。
    assert output.shape == noise.shape  # 保证帧数与通道数不丢失。
