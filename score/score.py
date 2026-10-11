# -*- coding: utf-8 -*-
"""VoxCPM2 · 8GB 共存量化任务 —— 自动评分脚本（CPU-only，无需 GPU）

评分分两层：

  ① 硬门槛（来自 results/results.json 的实测数字）
     - 共存峰值显存 <= peak_vram_mib_max
     - 共存下 TTS 实时率 RTF <= rtf_max

  ② 音质代理（来自 results/samples/ 的 wav 对，相对 bf16 基线的漂移）
     - log-mel 谱距离（LSD，dB）        —— 音色/细节整体漂移
     - 高频能量占比相对变化（HF ratio） —— 金属感 / 电音
     - 谱通量 p95 相对变化（flux）      —— 卡顿 / 爆音
     - 时长漂移（duration drift）       —— 长尾失控

  ⚠️ 音质代理**只有样本同 seed 时才有效**。若样本对的 seed 不一致，
     脚本会照常输出数字，但把该次判定标为 INVALID，不参与最终结论。

退出码：0 = 全部通过；1 = 有门槛未过；2 = 输入缺失/不可评。

用法：
    python score/score.py --results results/results.json --samples results/samples \
        --rubric score/rubric.json --out score/score.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

N_FFT = 1024
HOP = 256
N_MELS = 80
HF_CUTOFF_HZ = 4000.0


# ─────────────────────────── 信号处理（仅 numpy，无 librosa 依赖）─────────────

def _hz2mel(f: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def _mel2hz(m: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int = N_FFT, n_mels: int = N_MELS) -> np.ndarray:
    """HTK 风格三角梅尔滤波器组，形状 (n_mels, n_fft//2+1)。"""
    n_bins = n_fft // 2 + 1
    hz = _mel2hz(np.linspace(_hz2mel(0.0), _hz2mel(sr / 2.0), n_mels + 2))
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for i in range(n_mels):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        center = max(center, left + 1)
        right = min(max(right, center + 1), n_bins)
        for k in range(left, center):
            fb[i, k] = (k - left) / (center - left)
        for k in range(center, right):
            fb[i, k] = (right - k) / (right - center)
    return fb


def stft_mag(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """幅度谱，形状 (frames, n_fft//2+1)。"""
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    win = np.hanning(n_fft)
    n_frames = 1 + (len(x) - n_fft) // hop
    out = np.empty((n_frames, n_fft // 2 + 1), dtype=np.float64)
    for i in range(n_frames):
        out[i] = np.abs(np.fft.rfft(x[i * hop : i * hop + n_fft] * win))
    return out


def log_mel(x: np.ndarray, sr: int, fb: np.ndarray) -> np.ndarray:
    """log10 梅尔谱，单位近似 dB。"""
    mag = stft_mag(x)
    mel = mag @ fb.T
    return 10.0 * np.log10(np.maximum(mel, 1e-10))


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float64), int(sr)


def hf_energy_ratio(x: np.ndarray, sr: int, cutoff: float = HF_CUTOFF_HZ) -> float:
    """4 kHz 以上能量占总能量的比例 —— 量化噪声/金属感会把它推高。"""
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)) if len(x) > 1 else x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    total = float(spec.sum()) or 1e-12
    return float(spec[freqs >= cutoff].sum() / total)


def spectral_flux_p95(x: np.ndarray) -> float:
    """谱通量 p95（归一化）—— 卡顿/爆音会在少数帧上产生尖峰。"""
    mag = stft_mag(x)
    if len(mag) < 2:
        return 0.0
    ref = np.maximum(mag[:-1].sum(axis=1), 1e-12)
    diff = np.maximum(mag[1:] - mag[:-1], 0.0).sum(axis=1) / ref
    return float(np.percentile(diff, 95))


# ─────────────────────────── 指标与打分 ──────────────────────────────────────

def _rel(a: float, b: float) -> float:
    """相对变化 |a-b|/|b|，b 为基线。"""
    return abs(a - b) / max(abs(b), 1e-12)


def _clamp01(v: float) -> float:
    return float(min(max(v, 0.0), 1.0))


def _sub_score(value: float, limit: float) -> float:
    """value <= 0 满分；value >= limit 零分；线性衰减。"""
    if limit <= 0:
        return 0.0
    return _clamp01(1.0 - value / limit)


def compare_pair(ref_path: Path, cand_path: Path, qcfg: dict) -> dict:
    """参考（bf16）与候选（量化）的音质代理对比。"""
    ref, sr = load_wav(ref_path)
    cand, sr_c = load_wav(cand_path)
    if sr != sr_c:
        raise ValueError(f"采样率不一致: {sr} vs {sr_c}")

    fb = mel_filterbank(sr)
    lm_ref, lm_cand = log_mel(ref, sr, fb), log_mel(cand, sr, fb)
    n_common = min(len(lm_ref), len(lm_cand))
    lsd_db = float(np.abs(lm_ref[:n_common] - lm_cand[:n_common]).mean())

    dur_ref = len(ref) / sr
    dur_cand = len(cand) / sr
    dur_drift = _rel(dur_cand, dur_ref)

    hf_ref = hf_energy_ratio(ref, sr)
    hf_cand = hf_energy_ratio(cand, sr)
    hf_rel = _rel(hf_cand, hf_ref)

    flux_ref = spectral_flux_p95(ref)
    flux_cand = spectral_flux_p95(cand)
    flux_rel = _rel(flux_cand, flux_ref)

    w = qcfg["weights"]
    score = (
        w["lsd"] * _sub_score(lsd_db, qcfg["lsd_db_max"])
        + w["hf"] * _sub_score(hf_rel, qcfg["hf_ratio_rel_max"])
        + w["flux"] * _sub_score(flux_rel, qcfg["flux_p95_rel_max"])
        + w.get("duration", 0.0) * _sub_score(dur_drift, qcfg["duration_drift_rel_max"])
    )

    return {
        "reference": ref_path.name,
        "candidate": cand_path.name,
        "sample_rate": sr,
        "duration_ref_s": round(dur_ref, 4),
        "duration_cand_s": round(dur_cand, 4),
        "logmel_lsd_db": round(lsd_db, 4),
        "hf_ratio_ref": round(hf_ref, 6),
        "hf_ratio_cand": round(hf_cand, 6),
        "hf_ratio_rel": round(hf_rel, 4),
        "flux_p95_ref": round(flux_ref, 6),
        "flux_p95_cand": round(flux_cand, 6),
        "flux_p95_rel": round(flux_rel, 4),
        "duration_drift_rel": round(dur_drift, 4),
        "quality_score": round(_clamp01(score), 4),
        "pass": bool(score >= qcfg["pass_score"]),
    }


# ─────────────────────────── 主流程 ──────────────────────────────────────────

def gate_vram(peak_mib: float | None, limit: int) -> dict:
    """peak_mib 为 None 表示提交者未提供该测量 —— 视为未通过，并说明原因。"""
    if peak_mib is None:
        return {
            "metric": "peak_vram_mib",
            "value": None,
            "limit": limit,
            "pass": False,
            "headroom_mib": None,
            "note": "未提供共存峰值显存测量（results.coexist.peak_vram_mib 为空）",
        }
    return {
        "metric": "peak_vram_mib",
        "value": peak_mib,
        "limit": limit,
        "pass": bool(peak_mib <= limit),
        "headroom_mib": round(limit - peak_mib, 1),
    }


def gate_rtf(rtf: float | None, limit: float) -> dict:
    if rtf is None:
        return {
            "metric": "rtf",
            "value": None,
            "limit": limit,
            "pass": False,
            "note": "未提供共存 RTF 测量（results.coexist.rtf 为空）",
        }
    return {"metric": "rtf", "value": rtf, "limit": limit, "pass": bool(rtf <= limit)}


def main() -> int:
    ap = argparse.ArgumentParser(description="VoxCPM2 8GB 共存量化任务 · 自动评分")
    ap.add_argument("--results", default="results/results.json")
    ap.add_argument("--samples", default="results/samples")
    ap.add_argument("--rubric", default="score/rubric.json")
    ap.add_argument("--out", default="score/score.json")
    args = ap.parse_args()

    results_p, rubric_p = Path(args.results), Path(args.rubric)
    if not results_p.exists() or not rubric_p.exists():
        print(f"❌ 缺少输入: {results_p} / {rubric_p}", file=sys.stderr)
        return 2

    results = json.loads(results_p.read_text(encoding="utf-8"))
    rubric = json.loads(rubric_p.read_text(encoding="utf-8"))
    gates_cfg = rubric["hard_gates"]
    qcfg = rubric["quality"]

    # ① 硬门槛 —— 取"共存"这一组测量
    coexist = results["coexist"]
    gate_list = [
        gate_vram(coexist.get("peak_vram_mib"), gates_cfg["peak_vram_mib_max"]),
        gate_rtf(coexist.get("rtf"), gates_cfg["rtf_max"]),
    ]
    gates_pass = all(g["pass"] for g in gate_list)

    # ② 音质代理
    samples_dir = Path(args.samples)
    pairs = results.get("quality_pairs", [])
    comparisons, quality_valid = [], True
    for pr in pairs:
        ref_p, cand_p = samples_dir / pr["reference"], samples_dir / pr["candidate"]
        if not ref_p.exists() or not cand_p.exists():
            print(f"⚠️ 样本缺失，跳过: {ref_p.name} / {cand_p.name}")
            continue
        if pr.get("seed_reference") != pr.get("seed_candidate"):
            quality_valid = False
            print(f"⚠️ {cand_p.name}: 与基线的 seed 不一致 → 本次音质对比 INVALID")
        cmp_ = compare_pair(ref_p, cand_p, qcfg)
        cmp_["seed_matched"] = pr.get("seed_reference") == pr.get("seed_candidate")
        cmp_["quantization"] = pr.get("quantization", "unknown")
        comparisons.append(cmp_)

    quality_scores = [c["quality_score"] for c in comparisons if c["seed_matched"]]
    mean_quality = round(float(np.mean(quality_scores)), 4) if quality_scores else None
    quality_pass = (
        quality_valid and mean_quality is not None and mean_quality >= qcfg["pass_score"]
    )

    verdict = "PASS" if (gates_pass and quality_pass) else "FAIL"
    if gates_pass and not quality_valid:
        verdict = "FAIL(音质未验证)"

    out = {
        "schema": "voxcpm2-quant-bench/score/v1",
        "task": rubric["task"],
        "objective": rubric["objective"],
        "hard_gates": gate_list,
        "hard_gates_pass": gates_pass,
        "quality_comparisons": comparisons,
        "quality_valid": quality_valid,
        "mean_quality_score": mean_quality,
        "quality_pass": quality_pass,
        "verdict": verdict,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print(f"任务: {rubric['task']}")
    print("=" * 66)
    print("\n① 硬门槛")
    for g in gate_list:
        mark = "✅" if g["pass"] else "❌"
        shown = "未测量" if g["value"] is None else g["value"]
        head = g.get("headroom_mib")
        extra = f"  余量 {head} MiB" if head is not None else ""
        note = f"  （{g['note']}）" if g.get("note") else ""
        print(f"   {mark} {g['metric']}: {shown} (上限 {g['limit']}){extra}{note}")
    print("\n② 音质代理（相对 bf16 基线）")
    if not comparisons:
        print("   ⚠️ 没有可比对的样本对")
    for c in comparisons:
        flag = "✅" if c["pass"] else "❌"
        seed = "" if c["seed_matched"] else "  ⚠️seed 不一致"
        print(f"   {flag} {c['candidate']}  score={c['quality_score']:.3f}"
              f"  LSD={c['logmel_lsd_db']:.2f}dB  HF+{c['hf_ratio_rel'] * 100:.1f}%"
              f"  flux+{c['flux_p95_rel'] * 100:.1f}%{seed}")
    if mean_quality is not None:
        print(f"   → 平均音质分 {mean_quality}（门槛 {qcfg['pass_score']}）")
    if not quality_valid:
        print("   ⚠️ 音质维度 INVALID：样本未同 seed 生成，"
              "请先跑 bench/b09_seed_matched_quality.py")

    print("\n" + "=" * 66)
    print(f"结论: {verdict}     （明细已写入 {args.out}）")
    print("=" * 66)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
