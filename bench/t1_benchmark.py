"""T1 · VoxCPM2 四项实测（RTF / 流式首包 / 显存 / 音色一致性）

对应 docs/开发/32 §5.0 的 T1，以及 docs/通用/24 §3.2 的四个未知数。

用法：
    F:\\VoxCPM\\venv\\Scripts\\python.exe F:\\VoxCPM\\t1_benchmark.py

⚠️ 产物写到 F:\\VoxCPM\\out\\，音频交给耳朵判断（音色一致性没法自动打分）。
"""

import os
import sys
import time
from pathlib import Path

# ⚠️ Windows 上 stdout 默认 GBK —— 输出中文/符号会直接崩。
#    实测：UnicodeEncodeError: 'gbk' codec can't encode '\u2705'
#    别删（删了在 Windows 上必崩）。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from voxcpm import VoxCPM  # noqa: E402

MODEL_DIR = os.environ.get("VOXCPM_MODEL_DIR", r"F:\VoxCPM\models\VoxCPM2")
OUT_DIR = Path(os.environ.get("VOXCPM_OUT_DIR", "out"))

# 中文测试句 —— 长度递增，用来看 RTF 是否随长度变化
SENTENCES = [
    "你好呀。",
    "今天天气不错，要不要出去走走？",
    "我刚才在整理桌面上的文件，发现了一份很久以前的报告，你要看看吗？",
]


def gpu_mem() -> str:
    """当前显存占用。"""
    if not torch.cuda.is_available():
        return "无 CUDA"
    used = torch.cuda.memory_allocated() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"已用 {used:.2f} GB / 峰值 {peak:.2f} GB / 总量 {total:.2f} GB"


def section(title: str) -> None:
    print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")


def load() -> VoxCPM:
    """加载模型并报告加载耗时与显存。"""
    section("① 加载模型")
    t0 = time.perf_counter()
    model = VoxCPM.from_pretrained(MODEL_DIR, load_denoiser=False)
    print(f"  加载耗时: {time.perf_counter() - t0:.1f} s")
    print(f"  采样率  : {model.tts_model.sample_rate} Hz")
    print(f"  {gpu_mem()}")
    return model


def bench_rtf(model: VoxCPM) -> None:
    """逐句测 RTF（生成耗时 / 音频时长）。"""
    section("② RTF（生成耗时 ÷ 音频时长）")
    print(f"  {'文本':<28} {'音频时长':>9} {'生成耗时':>9} {'RTF':>7}")
    for text in SENTENCES:
        t0 = time.perf_counter()
        wav = model.generate(text=text, cfg_value=2.0, inference_timesteps=10)
        cost = time.perf_counter() - t0
        dur = len(wav) / model.tts_model.sample_rate
        print(f"  {text[:26]:<28} {dur:>8.2f}s {cost:>8.2f}s {cost / dur:>7.2f}")
    print(f"  {gpu_mem()}")


def bench_streaming(model: VoxCPM) -> None:
    """测流式的首包延迟与分块情况 —— 全双工的关键。"""
    section("③ 流式首包延迟（全双工的关键）")
    text = SENTENCES[-1]
    t0 = time.perf_counter()
    first, chunks, total = None, 0, 0
    for chunk in model.generate_streaming(text=text):
        if first is None:
            first = time.perf_counter() - t0
        chunks += 1
        total += len(chunk)
    cost = time.perf_counter() - t0
    dur = total / model.tts_model.sample_rate
    print(f"  首包延迟: {first:.2f} s" if first else "  ⚠️ 没有产出任何分块")
    print(f"  分块数  : {chunks}")
    print(f"  总耗时  : {cost:.2f} s / 音频 {dur:.2f} s / RTF {cost / dur:.2f}")
    print(f"  {gpu_mem()}")


def save_samples(model: VoxCPM) -> None:
    """落盘音频 —— 音色一致性靠耳朵判断。"""
    section("④ 音色一致性（落盘，需人耳判断）")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(SENTENCES, start=1):
        wav = model.generate(text=text, cfg_value=2.0, inference_timesteps=10)
        path = OUT_DIR / f"sample_{i}.wav"
        sf.write(str(path), wav, model.tts_model.sample_rate)
        print(f"  {path}  ({len(wav) / model.tts_model.sample_rate:.2f}s)")
    print("  ⭐ 请听：三句的『音色』是否一致（语速/音高起伏可不同，但音色要像同一个人）")


def main() -> None:
    if not Path(MODEL_DIR).exists():
        print(f"❌ 模型目录不存在: {MODEL_DIR}\n   先跑 download_model.py")
        return
    print(f"CUDA: {torch.cuda.is_available()} | torch {torch.__version__}")
    model = load()
    bench_rtf(model)
    bench_streaming(model)
    save_samples(model)
    section("完成")
    print(f"  {gpu_mem()}")


if __name__ == "__main__":
    main()
