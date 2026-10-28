"""轻量 VAE 与图像/视频去噪网络：所有权重均从零初始化。"""

import hashlib  # 使用稳定摘要将短文本映射到可训练的词表索引。
import math  # 构造正弦时间嵌入所需的频率。

import torch  # 提供张量、设备与随机采样。
from torch import nn  # 提供神经网络组件。
from torch.nn import functional as F  # 提供插值与激活函数。


class TinyVAE(nn.Module):  # 将像素压到四分之一空间分辨率的变分自编码器。
    def __init__(self, base: int = 32, latent_channels: int = 4):  # 允许按显存缩放宽度。
        super().__init__()  # 注册父类的参数管理能力。
        self.base = base  # 保留宽度以便写入检查点配置。
        self.latent_channels = latent_channels  # 保留潜变量通道配置。
        self.encoder = nn.Sequential(  # 建立两次下采样的像素编码器。
            nn.Conv2d(3, base, 3, padding=1), nn.SiLU(),  # 将 RGB 投影到隐藏特征。
            nn.Conv2d(base, base, 4, stride=2, padding=1), nn.SiLU(),  # 将边长减半。
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1), nn.SiLU(),  # 再将边长减半。
            nn.Conv2d(base * 2, latent_channels * 2, 3, padding=1),  # 同时预测均值与对数方差。
        )  # 完成编码器定义。
        self.decoder = nn.Sequential(  # 建立与编码器空间尺寸对应的解码器。
            nn.Conv2d(latent_channels, base * 2, 3, padding=1), nn.SiLU(),  # 扩展潜变量通道。
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1), nn.SiLU(),  # 恢复两倍边长。
            nn.ConvTranspose2d(base, base, 4, stride=2, padding=1), nn.SiLU(),  # 再恢复两倍边长。
            nn.Conv2d(base, 3, 3, padding=1), nn.Tanh(),  # 输出 [-1,1] 的 RGB 像素。
        )  # 完成解码器定义。

    def encode(self, image: torch.Tensor, sample: bool = True):  # 返回潜变量及分布参数。
        mean, logvar = self.encoder(image).chunk(2, dim=1)  # 沿通道拆出均值和对数方差。
        logvar = logvar.clamp(-10.0, 10.0)  # 防止标准差溢出或数值下溢。
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean) if sample else mean  # 训练时重参数化，推理时取均值。
        return z, mean, logvar  # 交还可用于重建与 KL 损失的量。

    def decode(self, latent: torch.Tensor) -> torch.Tensor:  # 从潜变量还原 RGB 图像。
        return self.decoder(latent)  # 输出范围由 Tanh 保证为 [-1,1]。

    def forward(self, image: torch.Tensor):  # 统一提供训练所需的重建结果。
        z, mean, logvar = self.encode(image)  # 执行变分编码。
        return self.decode(z), mean, logvar  # 将重建与分布参数交给损失函数。


def timestep_embedding(timestep: torch.Tensor, dim: int) -> torch.Tensor:  # 生成确定性的扩散时间特征。
    half = dim // 2  # 为正弦与余弦各留一半通道。
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timestep.device) / max(half - 1, 1))  # 建立对数间隔频率。
    phase = timestep.float()[:, None] * freqs[None, :]  # 将每个时间步乘以频率。
    embedding = torch.cat([phase.sin(), phase.cos()], dim=-1)  # 拼合正弦和余弦特征。
    return F.pad(embedding, (0, dim % 2))  # 奇数维时补一个通道。


class PromptCondition(nn.Module):  # 轻量哈希词表，仅学习数据集内的文字关联。
    def __init__(self, dim: int):  # 配置条件特征宽度。
        super().__init__()  # 注册可训练词嵌入。
        self.table = nn.Embedding(257, dim, padding_idx=0)  # 0 代表无条件，其他索引代表哈希词。

    def forward(self, prompts: list[str], device: torch.device) -> torch.Tensor:  # 将同批提示词转为条件向量。
        indices = []  # 收集每条文本对应的最多 16 个词。
        for prompt in prompts:  # 对每条文本独立分词。
            words = prompt.lower().strip().split()[:16]  # 用空格分词；中文需数据集自身约定词边界。
            hashed = [hashlib.sha256(word.encode("utf-8")).digest()[0] + 1 for word in words]  # 稳定映射到小词表。
            indices.append(hashed + [0] * (16 - len(hashed)))  # 将不同长度的文本补齐。
        tokens = torch.tensor(indices, dtype=torch.long, device=device)  # 让索引与模型处在同一设备。
        mask = (tokens != 0).unsqueeze(-1)  # 忽略补齐槽位。
        return (self.table(tokens) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # 对真实词取平均。


class _ConditionalUNet(nn.Module):  # 二维或三维共用的紧凑条件去噪器。
    def __init__(self, video: bool, base: int = 32):  # 用 video 选择时间轴上的卷积模式。
        super().__init__()  # 注册子网络和训练参数。
        self.video = video  # 记录模型模式供运行时检查。
        self.base = base  # 记录通道数供检查点重建。
        conv = nn.Conv3d if video else nn.Conv2d  # 视频保留显式的帧维度。
        transpose = nn.ConvTranspose3d if video else nn.ConvTranspose2d  # 为视频和图像选择对应的上采样。
        kernel = (3, 3, 3) if video else 3  # 视频卷积覆盖相邻帧。
        padding = (1, 1, 1) if video else 1  # 保持非下采样层的空间尺寸。
        down_kernel = (1, 4, 4) if video else 4  # 下采样只作用于空间维。
        down_stride = (1, 2, 2) if video else 2  # 保持视频帧数原样。
        down_pad = (0, 1, 1) if video else 1  # 使空间尺寸精确减半。
        self.input = conv(4, base, kernel, padding=padding)  # 提取初始潜变量特征。
        self.down = conv(base, base * 2, down_kernel, stride=down_stride, padding=down_pad)  # 构建低分辨率分支。
        self.middle = conv(base * 2, base * 2, kernel, padding=padding)  # 在低分辨率融合邻域信息。
        self.up = transpose(base * 2, base, down_kernel, stride=down_stride, padding=down_pad)  # 恢复输入的空间尺寸。
        self.output = conv(base * 2, 4, kernel, padding=padding)  # 拼接跳连后预测同尺寸噪声。
        self.time = nn.Sequential(nn.Linear(base, base), nn.SiLU(), nn.Linear(base, base))  # 学习时间特征映射。
        self.text = PromptCondition(base)  # 学习文本和目标画面的数据集内关联。
        self.condition = nn.Linear(base, base * 2)  # 将文本与时间投影到低分辨率特征。

    def forward(self, latent: torch.Tensor, timestep: torch.Tensor, prompts: list[str]) -> torch.Tensor:  # 预测潜变量上的噪声。
        expected_dim = 5 if self.video else 4  # 区分批量视频与批量图片。
        if latent.ndim != expected_dim or latent.shape[1] != 4:  # 拒绝输入错误的模型格式。
            raise ValueError(f"预期 {expected_dim} 维且通道为 4 的潜变量")  # 用中文说明失败原因。
        if len(prompts) != latent.shape[0] or timestep.numel() != latent.shape[0]:  # 保证条件逐样本对齐。
            raise ValueError("提示词、时间步与批大小不一致")  # 避免条件静默串样。
        time = self.time(timestep_embedding(timestep, self.base))  # 取得每个样本的扩散时间向量。
        text = self.text(prompts, latent.device)  # 取得每个样本的文本向量。
        shape = (-1, self.base * 2, 1, 1, 1) if self.video else (-1, self.base * 2, 1, 1)  # 选择条件广播形状。
        condition = self.condition(time + text).reshape(shape)  # 把两个条件送入低分辨率特征。
        skip = F.silu(self.input(latent))  # 缓存高分辨率的局部信息。
        hidden = F.silu(self.down(skip) + condition)  # 在低分辨率注入条件。
        hidden = F.silu(self.middle(hidden) + condition)  # 进一步聚合空间或时空上下文。
        hidden = F.silu(self.up(hidden))  # 还原潜变量空间尺寸。
        return self.output(torch.cat([hidden, skip], dim=1))  # 输出真实网络预测的噪声。


class ImageDenoiser(_ConditionalUNet):  # 仅沿二维空间进行扩散预测。
    def __init__(self, base: int = 32):  # 允许测试和 GPU 训练采用不同网络宽度。
        super().__init__(video=False, base=base)  # 初始化二维卷积分支。


class VideoDenoiser(_ConditionalUNet):  # 沿时间和二维空间共同预测噪声。
    def __init__(self, base: int = 32):  # 允许测试和 GPU 训练采用不同网络宽度。
        super().__init__(video=True, base=base)  # 初始化三维时空卷积分支。
