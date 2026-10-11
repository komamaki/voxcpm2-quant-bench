"""T1d · 全流程显存账：LLM(llama-server) 与 VoxCPM2 **同时加载**

用户提醒：「本机支持不代表整个流程跑起来还撑得住」——
所以本脚本用 nvidia-smi 测**真实总占用**，不是靠估算相加。

前置：llama-server 已在 8899 端口运行（由外部命令启动）。
用法：
    python bench/t1d_coexist.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402
from voxcpm import VoxCPM  # noqa: E402

MODEL_DIR = os.environ.get("VOXCPM_MODEL_DIR", "models/VoxCPM2")
OUT_DIR = Path(os.environ.get("VOXCPM_OUT_DIR", "out"))
TOTAL_MIB = int(os.environ.get("GPU_TOTAL_MIB", "8188"))


def smi() -> int:
    """nvidia-smi 报告的已用显存（MiB）—— ⭐ 含 llama-server 等所有进程。"""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
        capture_output=True, encoding="utf-8", errors="replace",
    ).stdout
    return int(out.strip().split()[0])


def report(tag: str) -> None:
    used = smi()
    pct = used / TOTAL_MIB * 100
    print(f"  {tag:<28} {used:>5} MiB  ({pct:>4.1f}%)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report("① 当前（含 llama-server）")

    print("\n② 加载 VoxCPM2 …")
    try:
        model = VoxCPM.from_pretrained(MODEL_DIR, load_denoiser=False)
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ 加载失败: {type(e).__name__}: {e}")
        return
    report("③ 两者共存")

    print("\n④ 共存下实际生成一段 …")
    try:
        t0 = time.perf_counter()
        wav = model.generate(
            text="(年轻男孩，清爽开朗)我在，你说吧。",
            cfg_value=2.0, inference_timesteps=5,
        )
        dur = len(wav) / model.tts_model.sample_rate
        print(f"  ✅ 成功：音频 {dur:.2f}s / 耗时 {time.perf_counter() - t0:.2f}s")
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ 生成失败: {type(e).__name__}: {e}")
    report("⑤ 生成时峰值")

    print(f"\n  torch 侧峰值: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    free = TOTAL_MIB - smi()
    print(f"  ⭐ 剩余可用: {free} MiB")
    if free < 500:
        print("  ⚠️⚠️ 余量过小 —— 长时间运行有 OOM 风险")


if __name__ == "__main__":
    main()
