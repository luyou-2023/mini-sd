"""训练和推理 CLI：模型权重、数据和输出均显式指定。"""

import argparse  # 定义可发现的命令行训练参数。
import os  # 对检查点执行同文件系统原子替换。
from pathlib import Path  # 统一处理跨平台输出路径。
import subprocess  # 使用已安装的 FFmpeg 编码真正的 MP4。

import numpy as np  # 将解码张量转换为媒体编码字节。
from PIL import Image  # 将图像推理结果保存为 PNG。
import torch  # 实现优化器、随机种子与设备管理。
from torch.nn import functional as F  # 计算 VAE 重建误差。
from torch.utils.data import DataLoader  # 顺序批量读取图片或视频。

from .data import ImageDataset, VideoDataset  # 读取真实训练素材和文本条件。
from .diffusion import Diffusion  # 执行噪声训练和 DDIM 推理。
from .models import ImageDenoiser, TinyVAE, VideoDenoiser  # 实例化三种轻量模型。


def _device(name: str) -> torch.device:  # 按实际硬件选择运算设备。
    if name == "auto":  # 自动选择已可用的 GPU 后端。
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"  # CUDA 优先于 Apple GPU。
    if name == "cuda" and not torch.cuda.is_available():  # 防止静默回退到 CPU。
        raise ValueError("当前 Python 环境不可用 CUDA")  # 显示服务器侧驱动配置问题。
    if name == "mps" and not torch.backends.mps.is_available():  # 防止错误配置 Apple GPU。
        raise ValueError("当前 Python 环境不可用 MPS")  # 显示设备限制。
    return torch.device(name)  # 交给 PyTorch 解析设备名称。


def _load(path: str, kind: str) -> dict:  # 仅加载本项目可信的检查点格式。
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)  # 禁用普通 pickle 对象反序列化。
    if not isinstance(checkpoint, dict) or checkpoint.get("kind") != kind:  # 检查不同模型的权重没有被混用。
        raise ValueError(f"检查点不是 {kind}：{path}")  # 阻止图像权重当成视频权重。
    return checkpoint  # 返回可复原的纯数据字典。


def _save(path: str, payload: dict) -> None:  # 以原子方式保存训练进度。
    destination = Path(path)  # 归一化输出路径。
    destination.parent.mkdir(parents=True, exist_ok=True)  # 支持新建的检查点目录。
    temporary = destination.with_name(destination.name + ".tmp")  # 将未完成写入留在临时文件。
    torch.save(payload, temporary)  # 存储模型、优化器、配置与训练步数。
    os.replace(temporary, destination)  # 避免中断时留下半截正式检查点。


def _images(video: torch.Tensor) -> torch.Tensor:  # 将视频批次拆成逐帧的 VAE 输入。
    batch, channels, frames, height, width = video.shape  # 解包视频时空尺寸。
    return video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)  # 形成 [B*T,C,H,W]。


def _videos(latents: torch.Tensor, batch: int, frames: int) -> torch.Tensor:  # 将逐帧潜变量恢复视频格式。
    _, channels, height, width = latents.shape  # 解包每帧潜变量尺寸。
    return latents.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)  # 形成 [B,C,T,h,w]。


def _next_batches(loader):  # 为有限步数训练循环供给数据。
    while True:  # 在数据集用完以后继续下一轮训练。
        for batch in loader:  # 让 DataLoader 控制打乱和批大小。
            yield batch  # 交还一个真实媒体批次。


def _validate_size(size: int) -> None:  # 保证 VAE 和 UNet 都能整除空间尺寸。
    if size < 16 or size % 8:  # 两次 VAE 与一次 UNet 下采样合计要求八倍对齐。
        raise ValueError("训练尺寸至少为 16，且必须是 8 的倍数")  # 让配置错误在训练前暴露。


def train_vae(args) -> None:  # 以真实图片优化共享的 VAE。
    _validate_size(args.size)  # 防止训练到中途才触发尺寸不匹配。
    device = _device(args.device)  # 确定此次训练所用实际设备。
    dataset = ImageDataset(args.data, size=args.size)  # 建立用户图片数据集。
    loader = _next_batches(DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0))  # 迭代真实图片样本。
    model = TinyVAE(base=args.base).to(device)  # 从头创建轻量编码与解码器。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # 训练 VAE 的全部参数。
    finished = 0  # 默认从第零步开始训练。
    if args.resume:  # 仅在用户显式要求时恢复进度。
        checkpoint = _load(args.resume, "vae")  # 从可信 VAE 检查点恢复。
        if checkpoint["size"] != args.size or checkpoint["base"] != args.base:  # 防止错配导致权重加载失败。
            raise ValueError("恢复训练的尺寸或宽度与检查点不一致")  # 说明具体不一致字段。
        model.load_state_dict(checkpoint["model"])  # 恢复已学习的 VAE 权重。
        optimizer.load_state_dict(checkpoint["optimizer"])  # 恢复优化器内部动量。
        finished = checkpoint["step"]  # 延续全局训练步数。
    model.train()  # 启用训练时重参数化。
    for local_step in range(args.steps):  # 仅执行明确要求的新增训练步数。
        images, _ = next(loader)  # 取得真实训练图片，VAE 阶段不使用文字。
        images = images.to(device)  # 将数据与网络放在同一设备。
        reconstructed, mean, logvar = model(images)  # 同时生成重建图与潜分布。
        reconstruction = F.mse_loss(reconstructed, images)  # 计算可微分的像素重建误差。
        kl = -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).mean()  # 将潜分布约束靠近单位正态。
        loss = reconstruction + args.kl_weight * kl  # 平衡重建品质与采样结构。
        optimizer.zero_grad(set_to_none=True)  # 清空旧梯度。
        loss.backward()  # 将损失回传到编码器与解码器。
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 限制梯度峰值。
        optimizer.step()  # 实际更新 VAE 权重。
        if local_step % max(1, args.steps // 20) == 0:  # 定期记录可核验的训练标量。
            print(f"VAE 步数 {finished + local_step + 1}，损失 {loss.item():.5f}", flush=True)  # 不捏造画质指标。
    _save(args.out, {"kind": "vae", "size": args.size, "base": args.base, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": finished + args.steps})  # 保存 VAE 和真实训练状态。
    print(f"已保存 VAE：{args.out}", flush=True)  # 显示实际写入位置。


def train_diffusion(args, video: bool) -> None:  # 为图像和视频训练各自的噪声预测器。
    device = _device(args.device)  # 选择运行设备。
    vae_checkpoint = _load(args.vae, "vae")  # 获取已训练的共享 VAE。
    size = vae_checkpoint["size"]  # 以 VAE 的训练尺寸约束新数据。
    vae = TinyVAE(base=vae_checkpoint["base"]).to(device)  # 重建相同宽度的 VAE。
    vae.load_state_dict(vae_checkpoint["model"])  # 载入实际训练过的编码器。
    vae.eval()  # 冻结随机训练态特性。
    vae.requires_grad_(False)  # 防止扩散阶段反向更新 VAE。
    dataset = VideoDataset(args.data, size=size, frames=args.frames) if video else ImageDataset(args.data, size=size)  # 选择与模型对应的真实媒体。
    loader = _next_batches(DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0))  # 建立可循环的训练批次。
    model = (VideoDenoiser(base=args.base) if video else ImageDenoiser(base=args.base)).to(device)  # 选择 3D 或 2D 模型。
    schedule = Diffusion(timesteps=args.diffusion_steps)  # 固定训练的噪声计划。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # 仅优化对应的噪声预测器。
    kind = "video" if video else "image"  # 标记权重类别以免混淆。
    finished = 0  # 默认本次从头训练。
    if args.resume:  # 显式恢复用户已有的扩散训练。
        checkpoint = _load(args.resume, kind)  # 严格校验训练类别。
        expected = {"size": size, "base": args.base, "diffusion_steps": args.diffusion_steps}  # 明确要求架构和训练步数一致。
        if video:  # 视频还须保持时间维度一致。
            expected["frames"] = args.frames  # 把帧数加入结构检查。
        if any(checkpoint[key] != value for key, value in expected.items()):  # 排除不兼容恢复。
            raise ValueError("恢复训练配置与检查点不一致")  # 给出直观的失败提示。
        model.load_state_dict(checkpoint["model"])  # 恢复扩散网络参数。
        optimizer.load_state_dict(checkpoint["optimizer"])  # 恢复优化器内部状态。
        finished = checkpoint["step"]  # 继续累计步数。
    for local_step in range(args.steps):  # 执行明确要求的新增训练步骤。
        pixels, prompts = next(loader)  # 从真实数据集中读媒体和标注。
        with torch.no_grad():  # 不更新 VAE 参数或缓存无用梯度。
            source = _images(pixels) if video else pixels  # 视频以同一个 VAE 编码逐帧图像。
            latent, _, _ = vae.encode(source.to(device), sample=False)  # 获取确定性的媒体潜变量。
            if video:  # 视频模型还原完整的时间维度。
                latent = _videos(latent, pixels.shape[0], pixels.shape[2])  # 在潜空间连接相邻帧。
        prompts = [prompt if torch.rand(()) >= args.condition_dropout else "" for prompt in prompts]  # 用随机空条件训练 CFG 分支。
        loss = schedule.train_step(model, latent, prompts, optimizer)  # 执行实际噪声预测梯度更新。
        if local_step % max(1, args.steps // 20) == 0:  # 定期反馈可验证训练进度。
            print(f"{kind} 步数 {finished + local_step + 1}，噪声损失 {loss:.5f}", flush=True)  # 不将损失当成画质指标。
    payload = {"kind": kind, "size": size, "base": args.base, "diffusion_steps": args.diffusion_steps, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": finished + args.steps}  # 封装完整恢复信息。
    if video:  # 视频权重需记录固定推理帧数。
        payload["frames"] = args.frames  # 保证训练与推理时序一致。
    _save(args.out, payload)  # 原子保存已训练的模型检查点。
    print(f"已保存 {kind} 模型：{args.out}", flush=True)  # 回报真实生成文件位置。


def _to_rgb(tensor: torch.Tensor) -> np.ndarray:  # 将 VAE 的 [-1,1] 输出转换成无符号 RGB。
    pixels = ((tensor.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)  # 限制溢出并量化为 8 位。
    return pixels.permute(1, 2, 0).numpy()  # 转换为编码器期望的 HWC 顺序。


def sample(args, video: bool) -> None:  # 从训练后的图像或视频检查点推理。
    device = _device(args.device)  # 选择 CPU 或现有 GPU。
    vae_checkpoint = _load(args.vae, "vae")  # 加载共享潜空间解码器。
    kind = "video" if video else "image"  # 决定所需模型类型。
    checkpoint = _load(args.model, kind)  # 严格禁止交叉使用权重。
    if vae_checkpoint["size"] != checkpoint["size"]:  # 检查推理阶段的像素尺寸一致性。
        raise ValueError("VAE 与扩散模型的尺寸不一致")  # 阻止损坏或错误的输出。
    vae = TinyVAE(base=vae_checkpoint["base"]).to(device)  # 重建对应架构的 VAE。
    vae.load_state_dict(vae_checkpoint["model"])  # 装载实际训练过的解码权重。
    vae.eval()  # 以固定模型行为推理。
    model = (VideoDenoiser(base=checkpoint["base"]) if video else ImageDenoiser(base=checkpoint["base"])).to(device)  # 重建正确的去噪网络。
    model.load_state_dict(checkpoint["model"])  # 装载真实训练检查点。
    schedule = Diffusion(timesteps=checkpoint["diffusion_steps"])  # 复现与训练一致的噪声计划。
    size = checkpoint["size"]  # 从训练检查点确定输出画幅。
    shape = (1, 4, checkpoint["frames"], size // 4, size // 4) if video else (1, 4, size // 4, size // 4)  # 明确潜变量的时空形状。
    latent = schedule.sample(model, shape, [args.prompt], steps=args.sample_steps, seed=args.seed, guidance=args.guidance)  # 完成真正的 DDIM 反向采样。
    destination = Path(args.out)  # 归一化媒体输出路径。
    destination.parent.mkdir(parents=True, exist_ok=True)  # 创建属于此任务的输出目录。
    with torch.no_grad():  # 解码不需保存梯度。
        decoded = vae.decode(_images(latent) if video else latent)  # 视频逐帧解码共用的 VAE。
    if not video:  # 图像直接保存单张 PNG。
        Image.fromarray(_to_rgb(decoded[0])).save(destination, format="PNG")  # 写出可由 PIL 打开的真实图片。
    else:  # 视频使用 FFmpeg 将实际生成帧编码成 MP4。
        frames = np.stack([_to_rgb(frame) for frame in decoded], axis=0)  # 构造 [T,H,W,RGB] 连续帧。
        command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size}x{size}", "-r", str(args.fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]  # 显式选择无声 H.264 兼容输出。
        subprocess.run(command, input=frames.tobytes(), capture_output=True, check=True)  # 任何编码失败均返回真实错误。
    print(f"已生成 {kind}：{destination}", flush=True)  # 回报已完成的文件路径。


def main() -> None:  # 解析命令并分派到正确的训练/推理流程。
    parser = argparse.ArgumentParser(description="Mini SD：从零训练的轻量图像/视频潜扩散实验")  # 提供用户可发现的帮助。
    commands = parser.add_subparsers(dest="command", required=True)  # 创建互斥子命令。
    for name in ("train-vae", "train-image", "train-video", "sample-image", "sample-video"):  # 逐项建立命令参数。
        sub = commands.add_parser(name)  # 创建命令专属解析器。
        sub.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")  # 允许明确指定设备。
        if name.startswith("train"):  # 训练命令需要数据集和优化配置。
            sub.add_argument("--data", required=True)  # 指定输入训练媒体目录。
            sub.add_argument("--steps", type=int, default=100)  # 指定新增梯度更新次数。
            sub.add_argument("--batch-size", type=int, default=1)  # 控制显存占用。
            sub.add_argument("--lr", type=float, default=0.0002)  # 指定优化器学习率。
            sub.add_argument("--base", type=int, default=32)  # 指定网络宽度。
            sub.add_argument("--resume")  # 可选恢复已有的同类检查点。
            sub.add_argument("--out", required=True)  # 指定新检查点文件路径。
        if name != "train-vae":  # 扩散训练与所有采样都需要 VAE 权重。
            sub.add_argument("--vae", required=True)  # 指定预训练的本项目 VAE。
        if name == "train-vae":  # VAE 阶段需要显式确定像素尺寸。
            sub.add_argument("--size", type=int, default=32)  # 默认教学用小画幅。
            sub.add_argument("--kl-weight", type=float, default=0.001)  # 控制潜变量正则强度。
        if name in ("train-image", "train-video"):  # 扩散训练独有的噪声参数。
            sub.add_argument("--diffusion-steps", type=int, default=100)  # 固定训练噪声计划长度。
            sub.add_argument("--condition-dropout", type=float, default=0.1)  # 为 CFG 训练空条件分支。
        if name == "train-video":  # 三维模型需要固定时间长度。
            sub.add_argument("--frames", type=int, default=4)  # 默认生成短片段以适配小显存。
        if name.startswith("sample"):  # 推理命令需要模型与提示词。
            sub.add_argument("--model", required=True)  # 指定图像或视频扩散检查点。
            sub.add_argument("--out", required=True)  # 指定 PNG 或 MP4 输出路径。
            sub.add_argument("--prompt", default="")  # 指定本训练集对应的短文本条件。
            sub.add_argument("--sample-steps", type=int, default=20)  # 控制 DDIM 推理步数。
            sub.add_argument("--seed", type=int, default=0)  # 允许固定随机种子。
            sub.add_argument("--guidance", type=float, default=1.0)  # 控制文本 CFG 权重。
        if name == "sample-video":  # 视频容器还需要播放帧率。
            sub.add_argument("--fps", type=int, default=8)  # 指定无声输出播放速度。
    args = parser.parse_args()  # 真正读取命令行输入。
    if args.command.startswith("train"):  # 训练前统一检查数值边界。
        if args.steps < 1 or args.batch_size < 1 or args.lr <= 0 or args.base < 4:  # 阻止不可能完成的训练配置。
            parser.error("steps/batch-size 必须为正，lr 须大于零，base 至少为 4")  # 报错而非写入空模型。
    if args.command == "train-vae":  # 将训练限制在指定 VAE 功能。
        train_vae(args)  # 执行实际 VAE 训练。
    elif args.command in ("train-image", "train-video"):  # 区分不同维度的扩散网络。
        if not 0 <= args.condition_dropout <= 1:  # 拒绝无效概率。
            parser.error("condition-dropout 必须位于 0 到 1")  # 显示配置错误。
        train_diffusion(args, video=args.command == "train-video")  # 执行实际噪声训练。
    else:  # 剩余子命令均为媒体推理。
        if args.command == "sample-video" and args.fps < 1:  # 防止视频帧率非法。
            parser.error("fps 至少为 1")  # 在解码前给出可读错误。
        sample(args, video=args.command == "sample-video")  # 输出 PNG 或 MP4。


if __name__ == "__main__":  # 支持以 Python 模块方式直接运行。
    main()  # 执行命令行入口。
