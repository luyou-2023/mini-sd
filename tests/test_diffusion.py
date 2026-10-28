"""扩散目标与确定性采样的行为测试。"""

import torch  # 使用真实张量作为输入。

from mini_sd.diffusion import Diffusion  # 引入待实现的扩散调度器。
from mini_sd.models import ImageDenoiser  # 引入真实网络验证参数更新。


def test_training_step_updates_denoiser_parameters():  # 验证反向传播确实更新模型。
    model = ImageDenoiser(base=8)  # 使用快速可运行的小网络。
    schedule = Diffusion(timesteps=16)  # 构建有限步的扩散噪声表。
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 建立真实优化器。
    original = model.output.weight.detach().clone()  # 保存尚未更新的权重。
    loss = schedule.train_step(model, torch.randn(2, 4, 8, 8), ["蓝 天", "红 花"], optimizer)  # 执行一次真实训练。
    assert torch.isfinite(torch.tensor(loss))  # 损失必须是有限数值。
    assert not torch.equal(original, model.output.weight)  # 权重必须发生变化。


def test_ddim_is_reproducible_with_same_seed():  # 保证同模型、同种子可重现。
    model = ImageDenoiser(base=8)  # 创建未训练但固定权重的真实网络。
    schedule = Diffusion(timesteps=16)  # 构建采样所需时间表。
    first = schedule.sample(model, shape=(1, 4, 8, 8), prompts=["红 花"], steps=3, seed=7)  # 首次推理。
    second = schedule.sample(model, shape=(1, 4, 8, 8), prompts=["红 花"], steps=3, seed=7)  # 重复同一条件。
    assert first.shape == (1, 4, 8, 8)  # 输出保持潜变量形状。
    assert torch.equal(first, second)  # DDIM eta=0 时逐位一致。
