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


class TransformerBlock(nn.Module):  # 用多头自注意力与前馈网络处理潜空间 patch 序列。
    def __init__(self, width: int):  # 为小模型固定四个注意力头。
        super().__init__()  # 注册注意力、归一化和前馈权重。
        self.attention_norm = nn.LayerNorm(width)  # 保持注意力输入的数值稳定。
        self.attention = nn.MultiheadAttention(width, num_heads=4, batch_first=True)  # 让每个 token 关注全部时空 token。
        self.feedforward_norm = nn.LayerNorm(width)  # 为前馈残差分支单独归一化。
        self.feedforward = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))  # 学习非线性的 token 变换。

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # 输入与输出都是 [B,N,D]。
        normalized = self.attention_norm(tokens)  # 在注意力计算前规范化特征。
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)  # 真正执行全局自注意力。
        tokens = tokens + attended  # 将注意力信息与输入残差相加。
        return tokens + self.feedforward(self.feedforward_norm(tokens))  # 再通过前馈残差变换。


class _ConditionalDiT(nn.Module):  # 在图像或视频潜变量的 patch token 上执行扩散 Transformer。
    def __init__(self, video: bool, base: int = 32):  # 图像与视频共享骨干规则但不共享权重。
        super().__init__()  # 注册本模型的可训练模块。
        if base < 4 or base % 4:  # 每个注意力头必须得到整数个通道。
            raise ValueError("Transformer 宽度 base 必须是至少为 4 的四的倍数")  # 在创建模型时拦住无效配置。
        self.video = video  # 记录是否具有显式时间轴。
        self.base = base  # 保存检查点重建所需的隐藏宽度。
        kernel = (1, 2, 2) if video else 2  # 每帧切 2×2 patch，时间轴不下采样。
        conv = nn.Conv3d if video else nn.Conv2d  # 选择视频或图像的 patch 投影算子。
        unpatch = nn.ConvTranspose3d if video else nn.ConvTranspose2d  # 将 token 网格还原成潜变量。
        self.input = conv(4, base, kernel, stride=kernel)  # 把四通道潜变量切成可训练 token。
        self.blocks = nn.Sequential(TransformerBlock(base), TransformerBlock(base))  # 堆叠两层真实自注意力。
        self.final_norm = nn.LayerNorm(base)  # 解码前规范化每个 token。
        self.output = unpatch(base, 4, kernel, stride=kernel)  # 还原并预测每个潜变量位置的噪声。
        self.time = nn.Sequential(nn.Linear(base, base), nn.SiLU(), nn.Linear(base, base))  # 学习扩散时间条件。
        self.text = PromptCondition(base)  # 复用配对数据训练得到的短词条件。
        self.condition = nn.Linear(base, base)  # 将时间和文本合成 token 条件。

    def forward(self, latent: torch.Tensor, timestep: torch.Tensor, prompts: list[str]) -> torch.Tensor:  # 从带噪潜变量预测噪声。
        expected_dim = 5 if self.video else 4  # 视频比图像额外包含一维帧数。
        if latent.ndim != expected_dim or latent.shape[1] != 4:  # 核对批量和四通道潜变量格式。
            raise ValueError(f"预期 {expected_dim} 维且通道为 4 的潜变量")  # 拒绝错误架构输入。
        if any(length % 2 for length in latent.shape[-2:]):  # 切 2×2 patch 必须能整除宽高。
            raise ValueError("潜变量宽高必须是 2 的倍数")  # 防止输出尺寸与目标噪声不一致。
        if len(prompts) != latent.shape[0] or timestep.numel() != latent.shape[0]:  # 一条提示词和一个时间步对应一个样本。
            raise ValueError("提示词、时间步与批大小不一致")  # 防止跨样本条件混淆。
        patches = self.input(latent)  # 把输入投影成 [B,D,(T),h/2,w/2] 网格。
        grid_shape = patches.shape[2:]  # 记录时空网格形状以便还原。
        tokens = patches.flatten(2).transpose(1, 2)  # 形成 [B,N,D] 的图像/全视频 token 序列。
        positions = torch.arange(tokens.shape[1], device=latent.device)  # 为每个时空位置建立固定索引。
        position = timestep_embedding(positions, self.base).to(tokens.dtype).unsqueeze(0)  # 区分不同 patch 的位置。
        time = self.time(timestep_embedding(timestep, self.base))  # 计算逐样本扩散时间特征。
        text = self.text(prompts, latent.device)  # 计算逐样本提示词特征。
        condition = self.condition(time + text).unsqueeze(1)  # 广播相同条件到该样本的全部 token。
        tokens = self.blocks(tokens + position + condition)  # 用全局注意力融合不同空间/时间位置。
        patches = self.final_norm(tokens).transpose(1, 2).reshape(latent.shape[0], self.base, *grid_shape)  # 重建原 token 网格。
        return self.output(patches)  # 输出与带噪潜变量同形状的噪声预测。


class ImageDenoiser(_ConditionalDiT):  # 在二维潜空间 patch 上进行 Transformer 扩散预测。
    def __init__(self, base: int = 32):  # 设置可按显存缩放的 Transformer 宽度。
        super().__init__(video=False, base=base)  # 创建图像专用权重。


class VideoDenoiser(_ConditionalDiT):  # 在整段视频时空 patch 上执行自注意力。
    def __init__(self, base: int = 32):  # 视频宽度与帧数可由调用方控制。
        super().__init__(video=True, base=base)  # 创建独立于图像模型的视频权重。
