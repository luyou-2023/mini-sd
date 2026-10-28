"""HTML 教学书必须由真实源码生成并保持离线可阅读。"""

import subprocess  # 在独立进程运行图书构建器。
import sys  # 使用当前 Python 环境。


def test_book_is_self_contained_and_contains_real_source(tmp_path):  # 验证内容与工程源代码关联。
    result = subprocess.run([sys.executable, "book/build.py", "--out", str(tmp_path / "index.html")], capture_output=True, text=True)  # 构建实际交付文件。
    assert result.returncode == 0, result.stderr  # 构建失败时输出真实异常。
    html = (tmp_path / "index.html").read_text(encoding="utf-8")  # 回读最终 HTML 内容。
    assert html.startswith("<!DOCTYPE html>")  # 验证完整 HTML 文档结构。
    assert "VAE" in html and "DDIM" in html and "三维卷积" in html  # 验证核心概念均有教学说明。
    assert "class TinyVAE" in html and "class VideoDenoiser" in html  # 验证真实网络源码内嵌。
    assert "train-video" in html and "sample-video" in html  # 验证服务器操作可按书执行。
    assert "src=\"http" not in html and "href=\"http" not in html  # 验证不依赖外部静态资源。
