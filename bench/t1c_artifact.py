"""T1c · 追「电音/卡顿」+ 试 fp16

用户反馈：sample_2.wav（长句）出现「卡顿 / 电音」，1 和 3 正常。
本次要分辨：① 偶发？ ② retry_badcase 造成？ ③ 长句造成？

用法：
    python bench/t1c_artifact.py
"""

import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402
from voxcpm import VoxCPM  # noqa: E402

MODEL_DIR = os.environ.get("VOXCPM_MODEL_DIR", "models/VoxCPM2")
OUT_DIR = Path(os.environ.get("VOXCPM_OUT_DIR", "out"))

DESC = "(年轻男孩，清爽开朗，语速自然)"
LONG = "我刚才在整理桌面上的文件，发现了一份很久以前的报告，你要看看吗？"
SHORT = ["我刚才在整理桌面上的文件。", "发现了一份很久以前的报告。", "你要看看吗？"]


def mem() -> str:
    if not torch.cuda.is_available():
        return "无 CUDA"
    return f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB / 峰值 {torch.cuda.max_memory_allocated() / 1024**3:.2f}"


def gen(model, text: str, name: str, **kw):
    """生成一段并落盘，返回 (时长, 耗时, RTF)。"""
    t0 = time.perf_counter()
    wav = model.generate(text=text, **kw)
    cost = time.perf_counter() - t0
    rate = model.tts_model.sample_rate
    dur = len(wav) / rate
    import soundfile as sf
    sf.write(str(OUT_DIR / f"{name}.wav"), wav, rate)
    print(f"  {name:<26} {dur:>6.2f}s {cost:>6.2f}s  RTF {cost / dur:>5.2f}")
    return dur, cost


def load(dtype=None) -> VoxCPM:
    kw = {"load_denoiser": False}
    if dtype:
        kw["dtype"] = dtype
    return VoxCPM.from_pretrained(MODEL_DIR, **kw)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"torch {torch.__version__} | bf16 基准\n")

    print("=" * 62)
    print("① 长句重复 3 次 —— 电音是偶发还是必然？")
    print("=" * 62)
    model = load()
    print(f"  加载后显存: {mem()}")
    for i in (1, 2, 3):
        gen(model, f"{DESC}{LONG}", f"art_long_try{i}", cfg_value=2.0, inference_timesteps=5)
    print(f"  显存: {mem()}")

    print("\n" + "=" * 62)
    print("② 关 retry_badcase —— 看是否它的重试拼接造成电音")
    print("=" * 62)
    gen(model, f"{DESC}{LONG}", "art_noretry", cfg_value=2.0,
        inference_timesteps=5, retry_badcase=False)

    print("\n" + "=" * 62)
    print("③ 拆短句 —— 短句是否稳定（排除「长句」是主因）")
    print("=" * 62)
    for i, s in enumerate(SHORT, start=1):
        gen(model, f"{DESC}{s}", f"art_short{i}", cfg_value=2.0, inference_timesteps=5)

    del model
    torch.cuda.empty_cache()

    print("\n" + "=" * 62)
    print("④ fp16 对照 —— 更快吗？（显存不会省，都是 2 字节）")
    print("=" * 62)
    try:
        m16 = load("float16")
        print(f"  加载后显存: {mem()}")
        gen(m16, f"{DESC}{LONG}", "art_fp16", cfg_value=2.0, inference_timesteps=5)
        print(f"  显存: {mem()}")
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ fp16 加载/生成失败: {type(e).__name__}: {e}")

    print(f"\n⭐ 音频在 {OUT_DIR}\\art_*.wav —— 请听哪些有电音")


if __name__ == "__main__":
    main()
