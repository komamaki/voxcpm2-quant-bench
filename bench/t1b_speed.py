"""T1b · 零依赖加速：inference_timesteps / cfg_value 参数扫描

背景：Nano-vLLM 在 Windows 上卡在 flash-attn（无预编译 wheel + 需 CUDA Toolkit）。
本脚本只调**已有参数**，看能不能把 RTF 从 1.55 压到 < 1.0。

⭐ 同时测 Voice Design（文字生成音色，本项目核心需求 —— 不是克隆）

用法：
    python bench/t1b_speed.py
"""

import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from voxcpm import VoxCPM  # noqa: E402

MODEL_DIR = os.environ.get("VOXCPM_MODEL_DIR", "models/VoxCPM2")
OUT_DIR = Path(os.environ.get("VOXCPM_OUT_DIR", "out"))

# ⭐ 本项目要的是「文字生成音色」（Voice Design），不是克隆
VOICE_DESC = "(年轻男孩，清爽开朗，语速自然)"
TEXT = "我刚才在整理桌面上的文件，发现了一份很久以前的报告，你要看看吗？"

# (timesteps, cfg, 说明)  —— cfg=1.0 相当于关掉 CFG（省一次前向）
COMBOS = [
    (10, 2.0, "基准（默认）"),
    (10, 1.0, "关 CFG"),
    (5, 2.0, "减半步数"),
    (5, 1.0, "半步数 + 关 CFG"),
    (3, 1.0, "极少步数"),
]


def main() -> None:
    print(f"CUDA: {torch.cuda.is_available()} | torch {torch.__version__}")
    print("加载模型…")
    model = VoxCPM.from_pretrained(MODEL_DIR, load_denoiser=False)
    rate = model.tts_model.sample_rate
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n文本: {TEXT[:30]}…")
    print(f"音色描述: {VOICE_DESC}\n")
    print(f"  {'步数':>4} {'cfg':>5}  {'音频':>7} {'耗时':>7} {'RTF':>6}  说明")
    print(f"  {'-' * 62}")

    for steps, cfg, note in COMBOS:
        t0 = time.perf_counter()
        wav = model.generate(
            text=f"{VOICE_DESC}{TEXT}",
            cfg_value=cfg,
            inference_timesteps=steps,
        )
        cost = time.perf_counter() - t0
        dur = len(wav) / rate
        print(f"  {steps:>4} {cfg:>5}  {dur:>6.2f}s {cost:>6.2f}s {cost / dur:>6.2f}  {note}")
        sf.write(str(OUT_DIR / f"speed_s{steps}_cfg{cfg}.wav"), wav, rate)

    print(f"\n⭐ 音频在 {OUT_DIR}\\speed_*.wav —— 请听：哪些还能听、哪些已经坏了")
    print("📌 取舍：步数越少越快，但**音质/稳定性下降** —— 要按听感定")


if __name__ == "__main__":
    main()
