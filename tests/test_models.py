"""模型行为测试：先证明模块缺失，再实现可运行模型。"""

import torch  # 引入张量运算以检查真实前向传播。
from torch import nn  # 检查网络确实含有多头注意力。

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


def test_image_transformer_attention_receives_gradient():  # 要求图像扩散确实经过可训练自注意力。
    model = ImageDenoiser(base=8)  # 用教学尺度进行真实前向与反向。
    attention = [module for module in model.modules() if isinstance(module, nn.MultiheadAttention)]  # 查找 PyTorch 真注意力模块。
    assert attention, "图像去噪器必须实际包含多头注意力"  # 旧卷积 U-Net 不应通过该断言。
    prediction = model(torch.randn(2, 4, 8, 8), torch.tensor([1, 2]), ["红 方块", "蓝 圆形"])  # 计算真实图像噪声预测。
    prediction.square().mean().backward()  # 将误差从输出反传到注意力参数。
    assert attention[0].in_proj_weight.grad is not None  # 确认该层位于计算图中。
    assert attention[0].in_proj_weight.grad.abs().sum() > 0  # 确认权重确有非零梯度。


def test_video_attention_uses_all_frames_and_receives_gradient():  # 验证视频不是逐帧独立去噪。
    torch.manual_seed(7)  # 固定模型初始化与测试噪声。
    model = VideoDenoiser(base=8).eval()  # 使用同一组确定性 Transformer 权重。
    attention = [module for module in model.modules() if isinstance(module, nn.MultiheadAttention)]  # 检查真注意力模块。
    assert attention  # 视频去噪核心不得退回纯逐帧卷积。
    baseline = torch.randn(1, 4, 4, 8, 8)  # 准备四帧连续潜变量。
    modified = baseline.clone()  # 保持末帧以外的所有输入相同。
    modified[:, :, -1] += 2.0  # 只改变时间距离最远的最后一帧。
    time = torch.tensor([3])  # 使用完全相同的扩散时间步。
    first = model(baseline, time, ["动作"])[:, :, 0]  # 预测原视频首帧噪声。
    second = model(modified, time, ["动作"])[:, :, 0]  # 预测末帧变化后的首帧噪声。
    assert not torch.allclose(first, second, atol=1e-7)  # 末帧应能通过时空自注意力影响首帧。
    model(baseline, time, ["动作"]).square().mean().backward()  # 将真实损失反传经过时空注意力。
    assert attention[0].in_proj_weight.grad is not None  # 确认视频注意力参数在计算图中。
    assert attention[0].in_proj_weight.grad.abs().sum() > 0  # 确认时空注意力获得非零梯度。


def test_video_denoiser_keeps_temporal_dimension():  # 检验时空 Transformer 去噪器。
    model = VideoDenoiser(base=16)  # 创建有跨帧自注意力的真实网络。
    noise = torch.randn(2, 4, 4, 8, 8)  # 模拟四帧视频潜变量。
    output = model(noise, torch.tensor([1, 2]), ["向左", "向右"])  # 执行预测。
    assert output.shape == noise.shape  # 保证帧数与通道数不丢失。
