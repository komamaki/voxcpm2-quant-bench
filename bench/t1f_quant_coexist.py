"""T1f · 决定性验证：LLM(llama-server) + **量化后**的 VoxCPM2 共存

T1d 结论：bf16 的 VoxCPM2 + LLM = 7814 MiB（95%），RTF 劣化到 14.4 ❌
T1e 结论：量化后 VoxCPM2 稳态仅 2.78 GB，RTF 仍 1.04 ✅
本脚本：验证两者能否**舒适共存**。

⚠️ 关键顺序：先加载 bf16 → 立刻量化（峰值会短暂升高）→ 稳定态低
   如果共存时在"量化瞬间"OOM，就需要改成"CPU 量化后再上 GPU"。

前置：llama-server 已在 8899 运行。
用法：
    F:\\VoxCPM\\venv\\Scripts\\python.exe F:\\VoxCPM\\t1f_quant_coexist.py
"""

import os
import gc
import subprocess
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

MODEL_DIR = os.environ.get("VOXCPM_MODEL_DIR", r"F:\VoxCPM\models\VoxCPM2")
OUT_DIR = Path(os.environ.get("VOXCPM_OUT_DIR", "out"))
TOTAL_MIB = int(os.environ.get("GPU_TOTAL_MIB", "8188"))
PREFIXES = ("base_lm.", "residual_lm.")


def smi() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
        capture_output=True, encoding="utf-8", errors="replace",
    ).stdout
    return int(out.strip().split()[0])


def report(tag: str) -> None:
    used = smi()
    print(f"  {tag:<30} {used:>5} MiB  ({used / TOTAL_MIB * 100:>4.1f}%)  余 {TOTAL_MIB - used}")


def quantize(model: VoxCPM) -> int:
    n = 0
    for pname, parent in list(model.tts_model.named_modules()):
        for cname, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{pname}.{cname}" if pname else cname
            if not full.startswith(PREFIXES):
                continue
            q = bnb.nn.Linear4bit(
                child.in_features, child.out_features, bias=child.bias is not None,
                quant_type="nf4", compute_dtype=torch.bfloat16,
            )
            q.weight = bnb.nn.Params4bit(child.weight.data, requires_grad=False, quant_type="nf4")
            if child.bias is not None:
                q.bias = nn.Parameter(child.bias.data.clone(), requires_grad=False)
            setattr(parent, cname, q.to(child.weight.device))
            n += 1
    return n


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report("① 基线（含 llama-server）")

    print("\n② 加载 bf16 → 量化 ⚠️ 这是峰值最高的时刻")
    model = VoxCPM.from_pretrained(MODEL_DIR, load_denoiser=False)
    report("   bf16 已加载")
    n = quantize(model)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"   量化 {n} 层完成")
    report("③ ⭐ 共存（量化稳态）")

    print("\n④ 共存下生成 3 句（看 RTF 是否仍可用）")
    texts = ["我在，你说吧。", "好，我记下了。", "那你先忙，我等你。"]
    for i, t in enumerate(texts, start=1):
        try:
            t0 = time.perf_counter()
            wav = model.generate(text=f"(年轻男孩，清爽开朗){t}", cfg_value=2.0, inference_timesteps=5)
            cost = time.perf_counter() - t0
            dur = len(wav) / model.tts_model.sample_rate
            sf.write(str(OUT_DIR / f"qc_{i}.wav"), wav, model.tts_model.sample_rate)
            print(f"   {i}. {dur:>5.2f}s 音频 / {cost:>5.2f}s → ⭐ RTF {cost / dur:.2f}")
        except Exception as e:  # noqa: BLE001
            print(f"   {i}. ❌ {type(e).__name__}: {e}")
    report("⑤ 生成后")

    free = TOTAL_MIB - smi()
    print(f"\n  ⭐ 剩余可用: {free} MiB")
    if free > 2000:
        print("  ✅✅ 余量充足 —— 语音模式可用")
    elif free > 1000:
        print("  ⭐ 余量够用")
    else:
        print("  ⚠️ 余量紧张")


if __name__ == "__main__":
    main()
