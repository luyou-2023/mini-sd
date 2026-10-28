"""DDPM 噪声学习与 DDIM 无随机反向采样。"""

import torch  # 提供随机采样、张量运算和梯度能力。
from torch.nn import functional as F  # 提供均方误差损失。


class Diffusion:  # 保留独立于模型权重的扩散噪声时间表。
    def __init__(self, timesteps: int = 100):  # 配置训练扩散的离散步数。
        if timesteps < 2:  # 避免只有一步时无法定义有效噪声过程。
            raise ValueError("扩散步数至少为 2")  # 提示可修复的配置错误。
        self.timesteps = timesteps  # 记录检查点必须复现的训练参数。
        betas = torch.linspace(0.0001, 0.02, timesteps, dtype=torch.float32)  # 使用线性方差计划。
        self.alpha_bar = torch.cumprod(1.0 - betas, dim=0)  # 预计算各步保留的原始信号比例。

    def train_step(self, model, clean: torch.Tensor, prompts: list[str], optimizer) -> float:  # 执行一次真实梯度更新。
        model.train()  # 启用模型训练模式。
        batch = clean.shape[0]  # 让时间和文本条件与批量对应。
        if batch == 0 or len(prompts) != batch:  # 阻止空批次和错位文本。
            raise ValueError("非空样本批次必须与提示词数量一致")  # 显示明确错误。
        step = torch.randint(0, self.timesteps, (batch,), device=clean.device)  # 对每个样本随机选择噪声等级。
        shape = (batch,) + (1,) * (clean.ndim - 1)  # 准备对图像或视频广播的系数。
        alpha = self.alpha_bar.to(clean.device)[step].reshape(shape)  # 查询累积信号比例。
        noise = torch.randn_like(clean)  # 采样训练目标高斯噪声。
        noisy = alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise  # 按 DDPM 前向公式加噪。
        prediction = model(noisy, step, prompts)  # 让网络预测刚加入的噪声。
        loss = F.mse_loss(prediction, noise)  # 最小化噪声预测误差。
        optimizer.zero_grad(set_to_none=True)  # 清空上一轮梯度。
        loss.backward()  # 将误差回传给网络参数。
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 避免小数据集上梯度爆炸。
        optimizer.step()  # 实际更新模型权重。
        return float(loss.detach().cpu())  # 输出可记录的标量训练损失。

    @torch.no_grad()  # 推理阶段不构建梯度图。
    def sample(self, model, shape: tuple[int, ...], prompts: list[str], steps: int = 20, seed: int = 0, guidance: float = 1.0) -> torch.Tensor:  # 从噪声生成潜变量。
        if steps < 1 or steps > self.timesteps:  # 要求推理步数不超过已定义的噪声表。
            raise ValueError("推理步数须位于 1 与训练扩散步数之间")  # 防止重复或越界时间索引。
        if len(shape) not in (4, 5) or shape[0] < 1 or len(prompts) != shape[0]:  # 检查输入形状和提示词对齐。
            raise ValueError("输入须为有效批量图像/视频潜变量，且提示词数量匹配")  # 拒绝不可推理配置。
        if guidance < 0:  # 避免误传负引导系数。
            raise ValueError("引导系数不可为负数")  # 明确参数含义。
        device = next(model.parameters()).device  # 从真实模型读取 CPU/MPS/CUDA 设备。
        generator = torch.Generator(device="cpu").manual_seed(seed)  # 使用 CPU 种子保持跨调用稳定。
        latent = torch.randn(shape, generator=generator, device="cpu").to(device)  # 创建相同种子初始噪声。
        times = torch.linspace(self.timesteps - 1, 0, steps).round().long().tolist()  # 选取降序 DDIM 时间步。
        alpha_bar = self.alpha_bar.to(device)  # 将噪声计划移到模型设备。
        model.eval()  # 关闭潜在的训练态随机层。
        for index, current in enumerate(times):  # 依次逆向消除噪声。
            batch_step = torch.full((shape[0],), current, device=device, dtype=torch.long)  # 构造模型所需时间条件。
            predicted = model(latent, batch_step, prompts)  # 估计有文本条件下的噪声。
            if guidance != 1.0:  # 仅在需要 CFG 时增加一次前向计算。
                unconditional = model(latent, batch_step, [""] * shape[0])  # 估计空提示词噪声。
                predicted = unconditional + guidance * (predicted - unconditional)  # 应用分类器自由引导。
            alpha = alpha_bar[current]  # 获取当前时间步的信号比例。
            previous = times[index + 1] if index + 1 < len(times) else -1  # 最后一步视为原始图像。
            alpha_prev = alpha_bar[previous] if previous >= 0 else torch.ones_like(alpha)  # 为终点使用全信号比例。
            estimate = (latent - (1.0 - alpha).sqrt() * predicted) / alpha.sqrt()  # 根据预测噪声还原干净潜变量。
            latent = alpha_prev.sqrt() * estimate + (1.0 - alpha_prev).sqrt() * predicted  # 执行 eta=0 的确定性 DDIM 更新。
        return latent  # 返回供 VAE 解码的潜变量。
