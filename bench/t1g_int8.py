"""T1g · 换更温和的量化：int8（而非 nf4）+ 部分量化对照

用户反馈（Q72）：nf4 4-bit 后音质像"打电话"（金属感/电音）—— ❌ 不可接受。
⭐ 音质是硬需求，所以策略改为「**能省一点是一点，但不能伤音质**」。

本脚本对比三条路（每次都重新加载，互不污染）：
  A. int8（8-bit）—— 量化噪声远小于 nf4，显存省一半
  B. nf4 但**只量化 base_lm**（residual_lm 保持 bf16）
  C. bf16 基线（对照，音质参照）

用法：
    python bench/t1g_int8.py
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


def gb() -> float:
    return torch.cuda.memory_allocated() / 1024**3


def replace_linears(model: VoxCPM, prefixes: tuple[str, ...], mode: str) -> int:
    """把指定前缀下的 nn.Linear 换成量化版；mode = 'int8' | 'nf4'。"""
    n = 0
    for pname, parent in list(model.tts_model.named_modules()):
        for cname, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{pname}.{cname}" if pname else cname
            if not full.startswith(prefixes):
                continue
            if mode == "int8":
                q = bnb.nn.Linear8bitLt(
                    child.in_features, child.out_features,
                    bias=child.bias is not None, has_fp16_weights=False, threshold=6.0,
                )
                q.weight = bnb.nn.Int8Params(
                    child.weight.data, requires_grad=False, has_fp16_weights=False,
                )
            else:
                q = bnb.nn.Linear4bit(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    quant_type="nf4", compute_dtype=torch.bfloat16,
                )
                q.weight = bnb.nn.Params4bit(
                    child.weight.data, requires_grad=False, quant_type="nf4",
                )
            if child.bias is not None:
                q.bias = nn.Parameter(child.bias.data.clone(), requires_grad=False)
            setattr(parent, cname, q.to(child.weight.device))
            n += 1
    return n


def run_case(tag: str, mode: str | None, prefixes: tuple[str, ...]) -> None:
    print("\n" + "=" * 64)
    print(f"【{tag}】")
    print("=" * 64)
    model = VoxCPM.from_pretrained(MODEL_DIR, load_denoiser=False)
    before = gb()
    n = 0
    if mode:
        t0 = time.perf_counter()
        n = replace_linears(model, prefixes, mode)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  {mode} 量化 {n} 层，耗时 {time.perf_counter() - t0:.1f}s")
    print(f"  加载 {before:.2f} GB → ⭐ 稳态 {gb():.2f} GB")

    try:
        t1 = time.perf_counter()
        wav = model.generate(text=TEXT, cfg_value=2.0, inference_timesteps=10)
        cost = time.perf_counter() - t1
        rate = model.tts_model.sample_rate
        dur = len(wav) / rate
        name = f"q_{tag}.wav"
        sf.write(str(OUT_DIR / name), wav, rate)
        print(f"  ✅ {dur:.2f}s 音频 / {cost:.2f}s → RTF {cost / dur:.2f}   [{name}]")
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ 生成失败: {type(e).__name__}: {e}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # ⭐ A：int8 全 LM —— 最温和的压缩，显存省一半
    run_case("A_int8", "int8", ("base_lm.", "residual_lm."))
    # ⭐ B：nf4 只动 base_lm —— residual_lm 疑与音频细节有关，保留 bf16
    run_case("B_nf4_baseonly", "nf4", ("base_lm.",))
    # C：bf16 基线
    run_case("C_bf16", None, ())

    print("\n⭐ 音频都在 $VOXCPM_OUT_DIR/q_*.wav —— 请重点对比音质")
    print("   A_int8 / B_nf4_baseonly / C_bf16（基线参照）")


if __name__ == "__main__":
    main()
