"""把实际项目源码嵌入一册可离线阅读的中文 HTML 教学书。"""

import argparse  # 允许按命令行指定最终 HTML 文件。
import ast  # 从真实源码中提取类与函数，不复制过期片段。
import html  # 对源码和正文中的特殊字符做安全转义。
from pathlib import Path  # 找到项目根目录和源码路径。

ROOT = Path(__file__).resolve().parent.parent  # 以书籍脚本的位置定位项目根。


def block(text: str) -> str:  # 给真实代码生成可滚动的浅色展示块。
    return '<pre class="code"><code>' + html.escape(text) + "</code></pre>"  # 避免把 Python 源码误解析成 HTML。


def excerpt(file: str, name: str) -> str:  # 从项目文件里抽取一个完整函数或类。
    source = (ROOT / file).read_text(encoding="utf-8")  # 读取真正参与测试的源码。
    tree = ast.parse(source)  # 解析 Python 语法树确认片段边界。
    for node in tree.body:  # 只匹配顶层的类或函数定义。
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name:  # 按精确名称检索实际实现。
            segment = ast.get_source_segment(source, node)  # 从实际文件提取完整类或函数。
            if segment is None:  # 语法树定位失败时阻止生成空白示例。
                raise ValueError(f"无法提取源码：{file}:{name}")  # 明确报告失效的示例位置。
            return block(segment)  # 保留源码的逐行中文注释。
    raise ValueError(f"源码里找不到 {file}:{name}")  # 书籍构建必须因过时引用而失败。


def appendix(file: str) -> str:  # 将完整可运行源码以折叠附录收入书内。
    source = (ROOT / file).read_text(encoding="utf-8")  # 从最终项目读取原文。
    return '<details class="source"><summary>' + html.escape(file) + " · 完整源码</summary>" + block(source) + "</details>"  # 让正文优先于长代码。


def chapter(number: int, title: str, body: str) -> str:  # 生成带目录锚点的书籍章节。
    return f'<section id="c{number}" class="chapter"><div class="eyebrow">CHAPTER {number:02d}</div><h2>{title}</h2>{body}</section>'  # 固定阅读层级。


def build() -> str:  # 组织原理先行、真实代码随后、可执行命令最后的内容。
    sections = []  # 保留书内章节以便生成目录。
    sections.append((1, "先决定要造什么", '''
<p class="lead">Mini SD Lab 是一台可以拆开的微型图像/视频生成机器：亲自训练一个图像压缩器、两个独立的噪声预测器，再从纯噪声回到图片和短视频。</p>
<div class="note"><strong>首先看清边界。</strong>这不是 Stable Diffusion 官方权重的轻量复刻，也没有 MiniMax H3 的语义能力。今天能够验证“梯度真的更新、文件真的可解码”；好看的真人角色、长时间动作、实时桌面互动都不在承诺内。</div>
<p>本书采用“原理 → 对应源码 → 验证命令”的顺序。图像与视频都采用真正的 Transformer 扩散去噪器：图像 token 彼此自注意力，视频 token 覆盖整段的空间与时间；潜变量先由逐帧图像 VAE 编码。两个去噪器结构相近但权重独立，并非把单张图复制四次。</p>
<figure class="diagram"><svg role="img" aria-label="从数据到图像和视频的完整训练推理链" viewBox="0 0 920 218"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8Z" fill="#23756c"/></marker></defs><rect x="12" y="50" width="175" height="118" rx="16" fill="#e4efe9"/><text x="35" y="95">授权图片 / 短视频</text><text x="35" y="123">同名 .txt 标注</text><rect x="238" y="50" width="164" height="118" rx="16" fill="#dbeae5"/><text x="280" y="101">图像 VAE</text><text x="267" y="129">像素 → 潜变量</text><rect x="456" y="10" width="194" height="83" rx="16" fill="#d9ece5"/><text x="478" y="47">图像 DiT</text><text x="478" y="71">独立图像权重</text><rect x="456" y="126" width="194" height="83" rx="16" fill="#e8e4d8"/><text x="478" y="164">视频 DiT</text><text x="478" y="189">独立视频权重</text><rect x="716" y="54" width="185" height="112" rx="16" fill="#fbebd7"/><text x="751" y="100">VAE 解码</text><text x="747" y="128">PNG / MP4</text><path d="M187 109 H238 M402 107 L456 52 M402 107 L456 168 M650 52 L716 103 M650 168 L716 116" stroke="#23756c" stroke-width="3" fill="none" marker-end="url(#arrow)"/></svg><figcaption>一条潜空间，两种生成器；尺寸必须在检查点之间一致。</figcaption></figure>
<h3>怎样判断“做完了”</h3><p>文件存在不是画质合格。最低技术验收是：训练后的权重发生变化；新进程可以加载检查点；生成的 PNG 可被图像库打开；生成的 MP4 有真实 H.264 视频流及目标帧数。画质、文本遵循度和时间连续性必须在服务器上用授权数据另行评估。</p>'''))  # 解释工程目的与验收边界。
    sections.append((2, "把图片压成潜变量：VAE", '''
<p>如果直接对每一个 RGB 像素去噪，计算量随空间大小迅速增加。VAE 先把图片的宽高各缩成四分之一、通道变成四个；去噪网络只处理这个更小的潜空间。推理的最后一步再用同一 VAE 把潜变量翻译回 RGB。</p>
<p>编码器输出两组值：均值 μ 和对数方差 log σ²。训练时从该分布采样，解码后比较与原图的像素误差；同时用 KL 项防止潜变量分布漂移。<code>损失 = MSE(重建图, 原图) + kl_weight × KL</code>。这里的 <code>kl_weight=0.001</code> 是本项目默认配置，不是普适最优值。</p>
<div class="callout">为什么先训练 VAE？如果解码器连红方块都无法还原，扩散器即使完美预测噪声，也没有办法输出清晰图。图像/视频扩散器随后共享这套 VAE，但训练扩散器时将 VAE 冻结。</div>
<p>下面是完整的 VAE 类源码。看 <code>encode</code> 的重参数化和 <code>decode</code> 的 Tanh 边界；训练循环在附录的 CLI 源码 <code>train_vae</code> 中。</p>
''' + excerpt("mini_sd/models.py", "TinyVAE")))  # 教授压缩概念并展示完整类实现。
    sections.append((3, "前向加噪与反向采样", '''
<p>扩散模型不是直接画画。训练时先选一个时间步 <em>t</em>，采样纯高斯噪声 ε，把干净潜变量 z₀ 变成 <code>zₜ = √ᾱₜ · z₀ + √(1−ᾱₜ) · ε</code>。网络学习预测刚刚加入的 ε。知道 ε 后，推理才能从随机噪声逐步还原潜变量。</p>
<p>本项目用线性 β 计划构造 ᾱ 表；用均方误差训练 ε 预测；使用确定性的 DDIM（η=0）跳步采样。采样步数不能超过训练噪声时间表。<code>--seed</code> 固定初始随机噪声，但不同设备/库版本不保证跨设备逐位相同。</p>
<figure class="formula"><span>干净潜变量</span><b>＋噪声</b><span>带噪潜变量</span><b>→ 预测噪声</b><span>逐步还原</span></figure>
<p><code>train_step</code> 每调用一次才更新一次权重；模型的损失下降与观众觉得好看是不同的判断。<code>sample</code> 里对每个时间点计算预测噪声、估计干净潜变量，再得到上一时间点状态。以下是完整的扩散调度器源码。</p>
''' + excerpt("mini_sd/diffusion.py", "Diffusion")))  # 说明训练与推理的数学链条。
    sections.append((4, "文字条件：能输入词，不等于懂中文", '''
<p>文本条件的目的，是让数据里的“红 方块”和图像里的红色方块形成可训练关联。本项目先用空格把一句文本分成短词，再通过 SHA-256 的首字节映射到一个小词表。它只提供稳定索引，不预先理解语义；不同词可能哈希碰撞，训练数据很少时自然无法泛化。</p>
<p>每个词对应一个可训练向量；所有有效词向量取平均，加上时间向量，广播到图像/视频 Transformer 的全部 patch token。扩散训练时默认有概率丢弃条件，用于让模型同时学到有词和无词预测。推理可用 CFG：<code>ε = ε_uncond + guidance × (ε_cond − ε_uncond)</code>。引导系数增加只会放大两种预测的差异；模型没学会对应词时不会凭空长出知识。</p>
<div class="note"><strong>中文分词提醒：</strong>标注和推理应统一用空格拆词，例如 <code>红 衣 女子</code>，否则整句会成为一个哈希 token。短句、统一词表比堆长提示词更适合这个教学模型。</div>
''' + excerpt("mini_sd/models.py", "PromptCondition")))  # 解释轻量文本条件的能力上限。
    sections.append((5, "图像与视频扩散 Transformer", '''
<p>图像潜变量为 <code>[B, 4, H/4, W/4]</code>；视频潜变量为 <code>[B, 4, T, H/4, W/4]</code>。先把每帧的潜变量切成 <code>2×2</code> patch，经可训练投影形成 token，加入位置、扩散时间和文字条件，然后送入两层 Transformer。每层都是真正的多头自注意力、LayerNorm、前馈层和残差连接，不是把卷积网络改名。</p>
<p>视频模式把 <code>T×(H/8)×(W/8)</code> 个 token 放在同一个自注意力序列里，所以第一帧 token 可以看见最后一帧；图像只有空间 token。最终把 token 还原为噪声张量，交给前章的 DDPM 训练和 DDIM 采样。<strong>Transformer 是噪声预测器，扩散是训练/采样机制：二者共同构成模型。</strong></p>
<div class="callout">全局注意力的计算和显存随 token 数增长得很快，视频尤其明显。当前只适用于小画幅和短片段；未加入高效注意力、第一帧控制、角色一致性模块或预训练文本编码器。VAE 编解码中的卷积与 DiT 的 patch 投影并不改变去噪主干是 Transformer 的事实。</div>
''' + excerpt("mini_sd/models.py", "TransformerBlock") + excerpt("mini_sd/models.py", "_ConditionalDiT") + excerpt("mini_sd/models.py", "ImageDenoiser") + excerpt("mini_sd/models.py", "VideoDenoiser")))  # 对照真实代码解释时空自注意力。
    sections.append((6, "准备训练材料", '''
<p>把有授权的图片放进 <code>data/images/</code>，MP4 短片放进 <code>data/videos/</code>。同名 UTF-8 <code>.txt</code> 是条件，例如 <code>scene01.mp4</code> 对应 <code>scene01.txt</code>；缺标注会变成空条件。图片和视频帧都先中心裁方，再缩到 VAE 的训练尺寸，像素从 0–255 变到 −1 至 1。</p>
<p>视频数据集按元数据计算均匀采样位置，顺序解码真实帧，但只保留目标 <code>--frames</code> 帧，不把整段视频转为张量；源视频仍需逐帧解码，因此服务器数据宜先切成数秒短片。它不会用一张图片冒充视频。视频训练不使用原声音轨，导出的视频也无声；配音、字幕与剪辑属于另一条制作流程。</p>
<h3>目录示例</h3>
''' + block("data/images/scene01.png\ndata/images/scene01.txt\ndata/videos/action01.mp4\ndata/videos/action01.txt") + '''
<p>没有授权数据时，可先用工程内的几何烟测生成器。它画的是红方块、蓝圆形和简单平移，不代表人物数据集。</p>
''' + block("python3 scripts/make_smoke_data.py --out data/smoke") + excerpt("mini_sd/data.py", "VideoDataset")))  # 介绍真实数据契约与读取方式。
    sections.append((7, "第一遍训练：三种检查点", '''
<p>训练必须按顺序：先训练 VAE，再用同一个 VAE 分别训练图片模型和视频模型。<code>--steps</code> 表示新增优化器更新次数，不是轮数；<code>--diffusion-steps</code> 决定训练噪声表长度，和推理时的 <code>--sample-steps</code> 不是同一个参数。</p>
<h3>只验证能运行的最小命令</h3>
''' + block("python3 -m mini_sd train-vae --data data/smoke/images --steps 1 --size 32 --base 8 --device cpu --out checkpoints/vae.pt\npython3 -m mini_sd train-image --data data/smoke/images --vae checkpoints/vae.pt --steps 1 --base 8 --diffusion-steps 16 --device cpu --out checkpoints/image.pt\npython3 -m mini_sd train-video --data data/smoke/videos --vae checkpoints/vae.pt --steps 1 --base 8 --frames 4 --diffusion-steps 16 --device cpu --out checkpoints/video.pt") + '''
<p>检查点包含模型权重、优化器状态、网络宽度、图像尺寸、训练扩散步数和累计更新步数；图像/视频检查点还保存 <code>architecture=dit-v1</code>，视频额外保存帧数。旧版卷积去噪检查点不兼容，须重新训练；原 VAE 结构不变可继续使用。中断后可通过 <code>--resume</code> 延续同架构检查点。恢复时必须保持尺寸、宽度和扩散时间表不变。当前版本只恢复权重/优化器/步数，不保存数据加载器或随机数状态，不能逐位重放训练轨迹。</p>
<div class="callout">一次更新的数值损失只能证明梯度路径工作；判断画质必须看训练多步的独立验证集和逐帧视频结果。不要把烟测输出当成“生成能力已训成”。</div>'''))  # 给出有顺序且能复核的训练命令。
    sections.append((8, "第二遍推理：图片与视频", '''
<p>推理从随机潜变量开始，使用图像或视频各自的去噪器进行 DDIM 采样，再让 VAE 解码。图像输出 PNG；视频把真实生成的连续帧交给 FFmpeg，编码 H.264/yuv420p 的无声 MP4。</p>
''' + block("python3 -m mini_sd sample-image --vae checkpoints/vae.pt --model checkpoints/image.pt --prompt '红 方块' --sample-steps 2 --seed 7 --device cpu --out outputs/sample.png\npython3 -m mini_sd sample-video --vae checkpoints/vae.pt --model checkpoints/video.pt --prompt '蓝 圆形' --sample-steps 2 --fps 8 --seed 7 --device cpu --out outputs/sample.mp4\nffprobe -v error -count_frames -show_entries stream=codec_name,nb_read_frames,width,height -of json outputs/sample.mp4") + '''
<p>模型检查点限制采样尺寸，视频检查点还限制帧数；不能只改导出参数就让模型无代价生成长片。<code>--guidance</code> 可以显式设置大于 1 的 CFG，但至少要先在真实标注数据上训练，且仍需看成片。媒体编码器不可用会直接报错，不会伪装生成成功。</p>
<h3>完整调用链走读</h3><ol><li>CLI 从检查点读取 <code>kind</code>，拒绝把 image 权重塞进 video 命令。</li><li>读取网络宽度、VAE 尺寸及扩散时间表；新建同结构网络并载入权重。</li><li>随机噪声变成图片潜变量或 <code>[1,4,T,h,w]</code> 视频潜变量，按 DDIM 反向迭代。</li><li>视频潜变量拆成逐帧图像潜变量，交给共享 VAE 解码；随后送往 FFmpeg。</li><li>独立工具重新打开 PNG/MP4；能播放是技术通过，不是审美通过。</li></ol>'''))  # 解释从命令入口到媒体导出的实际路径。
    sections.append((9, "明天在服务器上怎么开工", '''
<p>服务器还未开启，因此本章是**操作顺序**，不是已验证的 GPU 性能报告。先确认操作系统、显卡型号和显存、驱动、CUDA 对应的 PyTorch，再检查 <code>ffmpeg</code>、Python 和可用磁盘。不要照抄陌生整合包或为教程临时拉满卡。</p>
<ol class="steps"><li><strong>建立隔离环境。</strong>用 Python 3.10+ 创建虚拟环境；先按官方指引安装与服务器驱动相符的 PyTorch CUDA 轮子，再在仓库根目录安装本项目。<code>torch.cuda.is_available()</code> 必须为真。</li><li><strong>跑单步技术烟测。</strong>先用几何素材跑三种检查点与两个输出，<code>ffprobe</code> 验证真实帧数。若 CUDA OOM，先把批次设为 1、尺寸退回 32、视频帧数退回 4；DiT 注意力尤其要控制总 token 数。</li><li><strong>接授权素材。</strong>按目录与同名 txt 规则准备短图/短片；先核对画面版权、帧率、静音情况和标注语义。分训练与验证素材，避免只看训练数据。</li><li><strong>先做 VAE 验证。</strong>检查重建图再决定是否扩大尺寸；重建基础差则继续训练 VAE，不能靠扩散参数弥补。</li><li><strong>再训练图像与视频模型。</strong>记录 GPU 峰值、每步耗时、损失曲线和固定种子的输出对照；仅当一个小试验可信时才扩大样本/步数。每次中断用 <code>--resume</code>，并保留原检查点副本。</li><li><strong>独立审片。</strong>真正打开最终图像/视频，核角色、服装、动作连续性、闪烁、时长与画幅；不足就报告“不合格”，不要用“已导出”代替。</li></ol>
<div class="note"><strong>GPU 参数留白：</strong>服务器的显存、驱动和数据量未知，书中不编造训练小时数、最大分辨率或商业画质。当前版本也不包含混合精度、分布式、数据流式读取或高级文本编码器。</div>'''))  # 将未来验证项与今天完成的结果区分开。
    sections.append((10, "故障排查与限制", '''
<table><thead><tr><th>现象</th><th>先核实</th><th>处理边界</th></tr></thead><tbody>
<tr><td>看起来像噪声</td><td>是否只训练一两步；VAE 能否重建授权训练图</td><td>先提高真实训练量；不承诺短时间得到人物质量</td></tr>
<tr><td>提示词没有作用</td><td>同名 txt 是否存在；中文词是否空格分隔</td><td>哈希嵌入不预训练语言语义，需配对数据学习</td></tr>
<tr><td>视频不动或闪烁</td><td>训练短片是否有真实动作；是否采用视频检查点</td><td>增加有意义的连续帧数据，不能靠 FFmpeg 修复内容</td></tr>
<tr><td>尺寸或权重报错</td><td>VAE 与扩散检查点的 size/base/kind</td><td>使用同一训练系列，严禁强行加载错权重</td></tr>
<tr><td>CUDA 不可用</td><td>驱动、torch CUDA 轮子及 Python 环境</td><td>先校准服务器环境，不把 CPU 跑通说成 GPU 验收</td></tr>
<tr><td>显存不足</td><td>视频帧数、批次、尺寸、网络宽度</td><td>小批次优先；目前无混精和分布式承诺</td></tr>
<tr><td>MP4 无法编码</td><td>FFmpeg 是否包含 libx264</td><td>安装可信 FFmpeg 或明确报阻塞，禁写空文件</td></tr>
</tbody></table><p>本项目专注教学与明天的服务器实验，既没有从 H3/SD 权重迁移，也没有训练角色 LoRA、RefMod 或把生成的短片剪成完整作品。下个阶段是否接现成高质量模型，应以用户新的明确方向决定。</p>'''))  # 给出可操作的排障顺序。
    sections.append((11, "完整源码与练习", '''
<p>前文依次讲了模型、噪声、数据与命令。下面四份代码是构建时从当前项目原文嵌入的完整实现，不是另写的伪代码。阅读路线：先读 <code>models.py</code> 看张量形状，再读 <code>diffusion.py</code> 看训练/采样，随后读 <code>data.py</code> 看真实媒体，最后在 <code>__main__.py</code> 串起 CLI 到输出文件。</p>
<p><strong>练习 A：</strong>保持数据集和种子不变，只增加训练步数，比较固定提示词输出；先记录 VAE 重建结果，再看生成结果。<strong>练习 B：</strong>固定提示词，改变视频末帧的潜变量，观察全局注意力如何影响第一帧的预测。<strong>练习 C：</strong>让图片标注中的中文不加空格，再用分词后的提示词推理，解释为什么输出可能不符合文字。</p>
''' + "".join(appendix(file) for file in ("mini_sd/models.py", "mini_sd/diffusion.py", "mini_sd/data.py", "mini_sd/__main__.py"))))  # 附上构建时真实的完整项目代码。
    toc = "".join(f'<a href="#c{number}"><span>{number:02d}</span>{html.escape(title)}</a>' for number, title, _ in sections)  # 由实际章节自动生成导航。
    content = "".join(chapter(number, html.escape(title), body) for number, title, body in sections)  # 组装书籍正文。
    css = '''
:root{--bg:#f7f6f2;--paper:#fff;--ink:#1b2b2b;--soft:#566b69;--line:#dce4df;--accent:#176e62;--cream:#fff5e5}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.85 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}
a{color:var(--accent)}.layout{display:grid;grid-template-columns:264px minmax(0,1fr);min-height:100vh}nav{position:sticky;top:0;height:100vh;overflow:auto;padding:34px 17px 24px 25px;background:#eaf0e9;border-right:1px solid var(--line)}nav .brand{display:block;font-weight:800;letter-spacing:.04em;font-size:19px;color:var(--ink);margin:0 0 4px}nav .sub{font-size:12px;color:var(--soft);margin:0 0 32px}nav a{display:flex;gap:12px;align-items:center;padding:9px 10px;margin:2px 0;border-radius:7px;color:#334742;text-decoration:none;font-size:13px;line-height:1.45}nav a:hover,nav a:focus{background:#dbe9e0;color:#114e44}nav a span{color:var(--accent);font-size:11px;font-weight:700}
main{min-width:0;padding:0 max(30px,calc((100vw - 1300px)/2)) 80px 52px}.hero{padding:70px 0 67px;border-bottom:1px solid var(--line)}.kicker,.eyebrow{font-size:12px;font-weight:800;letter-spacing:.15em;color:var(--accent)}h1{font-size:clamp(37px,5vw,63px);line-height:1.12;letter-spacing:-.05em;margin:17px 0 15px}h2{font-size:clamp(27px,3vw,38px);letter-spacing:-.04em;line-height:1.25;margin:8px 0 25px}h3{font-size:21px;line-height:1.4;margin:36px 0 12px}p{margin:15px 0;color:#2b3d3a}.hero .intro{font-size:20px;max-width:800px;color:#41534d}.hero .meta{color:var(--soft);font-size:14px;margin-top:30px}.chapter{padding:56px 0 62px;border-bottom:1px solid var(--line);max-width:1050px}.lead{font-size:21px;color:#26433a}.note,.callout{padding:20px 24px;border-radius:10px;border:1px solid #cbded5;background:#eaf4ec;margin:25px 0;color:#234138}.callout{background:var(--cream);border-color:#ecd8b8;color:#604b29}.diagram{width:100%;margin:30px 0;padding:22px;background:white;border:1px solid var(--line);border-radius:12px}.diagram svg{display:block;width:100%;height:auto}.diagram text{font-family:inherit;font-size:16px;fill:#24483f}.diagram figcaption{font-size:13px;color:var(--soft);margin:12px 0 0}.formula{display:flex;flex-wrap:wrap;gap:10px;align-items:center;background:white;border:1px solid var(--line);border-radius:10px;padding:18px;margin:28px 0}.formula span{background:#e7f1ea;border-radius:6px;padding:6px 10px}.formula b{font-size:12px;color:var(--accent)}code,pre{font-family:SFMono-Regular,Consolas,Menlo,monospace}p code,li code,td code{font-size:.82em;background:#edf2ec;padding:2px 5px;border-radius:4px;word-break:break-all}.code{background:#f8fafc;color:#172033;border:1px solid #cbd5e1;padding:19px 22px;border-radius:9px;overflow:auto;max-width:100%;font-size:12px;line-height:1.65;white-space:pre;tab-size:4}.code code{background:none;padding:0}.source{border:1px solid var(--line);background:white;border-radius:10px;margin:15px 0;overflow:hidden}.source summary{cursor:pointer;padding:13px 18px;font-size:14px;font-weight:700;color:var(--accent)}.source .code{margin:0;border:0;border-top:1px solid var(--line);border-radius:0;max-height:520px}.steps{padding-left:28px}.steps li{padding:8px 0 8px 10px}table{width:100%;table-layout:fixed;border-collapse:collapse;background:white;margin:25px 0;font-size:14px}th,td{border:1px solid var(--line);padding:12px;vertical-align:top;word-break:break-word}th{background:#edf4ef;text-align:left;color:#245146}footer{padding:35px 0;color:var(--soft);font-size:13px}body{overflow-x:hidden}
@media(max-width:850px){.layout{display:block}nav{position:static;height:auto;max-height:260px;padding:18px 22px;border-right:0;border-bottom:1px solid var(--line)}nav .sub{margin-bottom:8px}nav .links{display:flex;overflow:auto;gap:5px}nav a{flex:none;max-width:140px}main{padding:0 21px 55px}.hero{padding:43px 0}.chapter{padding:45px 0}.diagram{padding:8px}.diagram text{font-size:18px}.code{font-size:11px}}
@media print{nav{display:none}.layout{display:block}main{padding:0}.hero{padding:25px 0}.chapter{break-inside:auto;padding:24px 0}.source{display:none}body{background:#fff}.code{white-space:pre-wrap;overflow-wrap:anywhere}}
'''  # 页面样式全内联，不需要服务器或外部字体。
    return "<!DOCTYPE html>\n<html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Mini SD Lab｜从零训练图像与视频扩散 Transformer</title><style>" + css + "</style></head><body><div class=\"layout\"><nav aria-label=\"章节导航\"><div class=\"brand\">Mini SD Lab</div><p class=\"sub\">PYTORCH · 从零到一 · 离线可读</p><div class=\"links\">" + toc + "</div></nav><main><header class=\"hero\"><div class=\"kicker\">THE PRACTICAL BOOK</div><h1>从零训练<br>图像与视频扩散模型</h1><p class=\"intro\">一册对照真实 PyTorch 工程的中文实验手册：VAE、图像/视频 Transformer、扩散训练与推理。</p><div class=\"meta\">Mini SD Lab · 代码版本 0.2.0 · 本地技术烟测已跑通，GPU 训练待服务器验证</div></header>" + content + "<footer>Mini SD Lab · 只读当前项目真实源码生成 · 离线 HTML 教学书</footer></main></div></body></html>"  # 返回完全自包含的 HTML。


def main() -> None:  # 从命令行输出独立可分享的书籍文件。
    parser = argparse.ArgumentParser(description="从真实源码生成 Mini SD 中文 HTML 教学书")  # 声明脚本用途。
    parser.add_argument("--out", default=str(ROOT / "book/index.html"))  # 允许测试用临时目录构建。
    args = parser.parse_args()  # 获取输出路径。
    target = Path(args.out)  # 规范化书籍输出位置。
    target.parent.mkdir(parents=True, exist_ok=True)  # 创建尚不存在的目录。
    target.write_text(build(), encoding="utf-8")  # 将完整 HTML 以 UTF-8 落盘。
    print(f"已生成 HTML 教学书：{target}")  # 报告可核实的交付路径。


if __name__ == "__main__":  # 支持从工程根目录启动书籍构建器。
    main()  # 真正生成完整文档。
