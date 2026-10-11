"""T1e · bitsandbytes 4-bit 量化 VoxCPM2（只量化两个 LM，保住音质）

背景（T1d 实测）：LLM(3.3G) + VoxCPM2(4.5G) 共存 = 7814 MiB（95%），
且 RTF 从 1.0 劣化到 14.4 —— ❌ 不可用。

策略：VoxCPM2 的两个 MiniCPM 主干（base_lm / residual_lm）是参数大头，
       把它们的 nn.Linear 换成 bnb.nn.Linear4bit；
       ⚠️ AudioVAE / 扩散解码器保持 bf16（对精度敏感，动它们会毁音质）。

用法：
    python bench/t1e_quant.py
"""

import os
import gc
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import bitsandbytes as bnb  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import soundfile as sf  # noqa: E402
from voxcpm import VoxCPM  # noqa: E402

MODEL_DIR = os.environ.get("VOXCPM_MODEL_DIR", "models/VoxCPM2")
OUT_DIR = Path(os.environ.get("VOXCPM_OUT_DIR", "out"))
TEXT = "(年轻男孩，清爽开朗)我在，你说吧，有什么事？"

# ⭐ 只量化这两个前缀下的层 —— 其余保持 bf16
QUANT_PREFIXES = ("base_lm.", "residual_lm.")


def gb() -> float:
    return torch.cuda.memory_allocated() / 1024**3


def should_quantize(name: str) -> bool:
    return name.startswith(QUANT_PREFIXES)


def quantize_lms(model: VoxCPM) -> int:
    """把两个 LM 主干里的 nn.Linear 换成 4-bit，返回替换个数。"""
    inner = model.tts_model
    replaced = 0
    for parent_name, parent in list(inner.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if not should_quantize(full):
                continue
            has_bias = child.bias is not None
            q = bnb.nn.Linear4bit(
                child.in_features, child.out_features, bias=has_bias,
                quant_type="nf4", compute_dtype=torch.bfloat16,
            )
            q.weight = bnb.nn.Params4bit(
                child.weight.data, requires_grad=False, quant_type="nf4",
            )
            if has_bias:
                q.bias = nn.Parameter(child.bias.data.clone(), requires_grad=False)
            q = q.to(child.weight.device)
            setattr(parent, child_name, q)
            replaced += 1
    return replaced


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"bitsandbytes {bnb.__version__} | 量化前缀 {QUANT_PREFIXES}\n")

    print("① 加载（bf16 全量）…")
    model = VoxCPM.from_pretrained(MODEL_DIR, load_denoiser=False)
    print(f"   bf16 加载后: {gb():.2f} GB")

    print("\n② 量化两个 LM …")
    t0 = time.perf_counter()
    n = quantize_lms(model)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"   替换 {n} 个 Linear，耗时 {time.perf_counter() - t0:.1f}s")
    print(f"   ⭐ 量化后: {gb():.2f} GB")

    print("\n③ 生成（看是否还能用 + 音质）")
    try:
        t1 = time.perf_counter()
        wav = model.generate(text=TEXT, cfg_value=2.0, inference_timesteps=5)
        cost = time.perf_counter() - t1
        rate = model.tts_model.sample_rate
        dur = len(wav) / rate
        sf.write(str(OUT_DIR / "quant_nf4.wav"), wav, rate)
        print(f"   ✅ {dur:.2f}s 音频 / {cost:.2f}s 耗时 → RTF {cost / dur:.2f}")
    except Exception as e:  # noqa: BLE001
        print(f"   ❌ 失败: {type(e).__name__}: {e}")
        return

    print(f"\n   torch 峰值: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"\n⭐ 音频: {OUT_DIR}\\quant_nf4.wav —— 请听音质是否劣化")


if __name__ == "__main__":
    main()
