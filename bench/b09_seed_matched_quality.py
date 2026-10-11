# -*- coding: utf-8 -*-
"""B09 · 同 seed 对照：让音质从「靠耳朵」变成「可自动评分」

背景
----
T1/T1e/T1g 的样本都是**不固定 seed** 生成的 —— 同一句话两次采样本来就会不同，
所以「量化前后差异」里混着「随机采样差异」，谱距离这类指标**没有意义**。

⚠️ 版本差异（本仓库已实测确认）：
  - 本机安装的 `voxcpm==2.0.3`（site-packages）**没有** `generate(seed=...)` 参数，
    调用会直接 `TypeError: _generate() got an unexpected keyword argument 'seed'`。
  - 上游较新的源码克隆（`src/src/voxcpm/model/utils.py`）才有 `apply_generation_seed()`。
所以本脚本对 2.0.3 **用全局 RNG 播种** 达到同样效果：
`torch.manual_seed(N)` + `torch.cuda.manual_seed_all(N)`，每次生成前重置。

另外显式关掉 `retry_badcase`：它的重试循环会多消耗 RNG，
且「是否触发重试」可能因量化与否而不同 —— 那会把量化差异和重试差异混在一起。

本脚本
------
1. 载入 bf16 → 固定 seed 生成参考音频
2. 原地量化 base_lm / residual_lm 为 nf4（AudioVAE、扩散解码器保持 bf16）
3. **同一 seed** 再生成一次 → 候选音频
4. 落盘 + 写 run_meta.json（seed / 参数 / 版本 / 显存）

用法：
    # 可选：覆盖默认路径（默认 models/VoxCPM2 与 results/samples/seedmatched）
    set VOXCPM_MODEL_DIR=D:\\models\\VoxCPM2
    set VOXCPM_OUT_DIR=results\\samples\\seedmatched
    python bench/b09_seed_matched_quality.py
"""

import gc
import json
import os
import platform
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import bitsandbytes as bnb  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from voxcpm import VoxCPM  # noqa: E402

MODEL_DIR = os.environ.get("VOXCPM_MODEL_DIR", "models/VoxCPM2")
OUT_DIR = Path(os.environ.get("VOXCPM_OUT_DIR", "results/samples/seedmatched"))

SEED = int(os.environ.get("BENCH_SEED", "20261011"))
VOICE_DESC = os.environ.get("BENCH_VOICE_DESC", "(年轻男孩，清爽开朗，语速自然)")
TEXT = os.environ.get("BENCH_TEXT", "我在，你说吧，有什么事？")
CFG = float(os.environ.get("BENCH_CFG", "2.0"))
STEPS = int(os.environ.get("BENCH_STEPS", "10"))
PREFIXES = ("base_lm.", "residual_lm.")


def seed_all(seed: int) -> None:
    """重置 CPU + CUDA 全局 RNG —— 等价于新版上游的 apply_generation_seed()。

    2.0.3 的 `generate()` 不接受 seed 参数，只能这样在外部播种。
    必须在**每次** generate 之前调用，否则第二次生成会接着消耗上一次的 RNG 状态。
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_mem() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "device": props.name,
        "total_mib": round(props.total_memory / 1024**2, 1),
        "allocated_mib": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }


def quantize_nf4(model: VoxCPM) -> int:
    """把 base_lm / residual_lm 下的 nn.Linear 换成 4-bit（nf4），其余保持 bf16。"""
    n = 0
    for pname, parent in list(model.tts_model.named_modules()):
        for cname, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{pname}.{cname}" if pname else cname
            if not full.startswith(PREFIXES):
                continue
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


def gen(model: VoxCPM, name: str) -> dict:
    torch.cuda.reset_peak_memory_stats()
    seed_all(SEED)  # ⭐ 每次生成前重置 RNG —— A/B 唯一的变量是量化
    t0 = time.perf_counter()
    wav = model.generate(
        text=f"{VOICE_DESC}{TEXT}", cfg_value=CFG,
        inference_timesteps=STEPS, retry_badcase=False,
    )
    cost = time.perf_counter() - t0
    rate = model.tts_model.sample_rate
    dur = len(wav) / rate
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(str(OUT_DIR / f"{name}.wav"), wav, rate)
    print(f"  {name:<10} 音频 {dur:>5.2f}s / 耗时 {cost:>5.2f}s → RTF {cost / dur:.2f}")
    return {
        "file": f"{name}.wav",
        "audio_seconds": round(dur, 4),
        "cost_seconds": round(cost, 4),
        "rtf": round(cost / dur, 4),
        "sample_rate": rate,
        "seed": SEED,
        "cfg_value": CFG,
        "inference_timesteps": STEPS,
    }


def main() -> int:
    if not Path(MODEL_DIR).exists():
        print(f"❌ 模型目录不存在: {MODEL_DIR}")
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"seed={SEED} cfg={CFG} steps={STEPS}")
    print(f"文本: {VOICE_DESC}{TEXT}")
    print(f"CUDA={torch.cuda.is_available()} torch={torch.__version__}\n")

    runs = []
    print("① bf16 参考")
    model = VoxCPM.from_pretrained(MODEL_DIR, load_denoiser=False)
    ref = gen(model, "bf16")
    ref["vram"] = gpu_mem()
    runs.append({"id": "bf16", "quantization": "none", **ref})

    print("\n② 原地量化 → nf4 候选（同一 seed）")
    t0 = time.perf_counter()
    n = quantize_nf4(model)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  量化 {n} 层，耗时 {time.perf_counter() - t0:.1f}s")
    cand = gen(model, "nf4_lm")
    cand["vram"] = gpu_mem()
    runs.append({"id": "nf4_lm", "quantization": "nf4(base_lm,residual_lm)",
                 "quantized_layers": n, **cand})

    del model
    gc.collect()
    torch.cuda.empty_cache()

    meta = {
        "schema": "voxcpm2-quant-bench/run_meta/v1",
        "purpose": "同 seed 对照 —— 供 score/score.py 计算音质代理",
        "seed_matched": True,
        "seeding": {
            "method": "global_rng_reset",
            "detail": "torch.manual_seed(N) + torch.cuda.manual_seed_all(N) before each generate",
            "reason": "installed voxcpm==2.0.3 的 generate() 无 seed 参数",
            "retry_badcase": False,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "model_dir": MODEL_DIR,
        },
        "runs": runs,
    }
    (OUT_DIR / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n✅ 完成 → {OUT_DIR}（含 run_meta.json）")
    print("   下一步: python score/score.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
