# VoxCPM2 · 8 GB VRAM Coexistence Quantization Benchmark

> Run **VoxCPM2 TTS** and a **4B-class Qwen LLM** on the same 8 GB GPU,
> while keeping speech synthesis **real-time** (RTF ≤ 1.0) and the audio **usable**.
>
> A self-contained, **automatically-scorable** task package:
> task statement + reproducible environment + measured baseline + scoring script.

[![Task](https://img.shields.io/badge/task-open%20problem-orange)](#task-status)
[![Scoring](https://img.shields.io/badge/scoring-automatic-brightgreen)](score/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

---

## What this is

A **single-GPU memory-budget optimization task**, packaged so that it can be
reproduced and scored without human judgment:

- **The task**: on an 8188 MiB RTX 4060 Laptop, make a resident 4B-class LLM
  and VoxCPM2 TTS usable **at the same time**.
- **Why it is hard**: VoxCPM2 in full precision (bf16) already needs ~4.5 GB;
  the LLM adds ~3.3 GB. Together they hit 95 % of the card — and generation
  slows down by **14×** (RTF 1.0 → 14.4). Memory and latency fail **simultaneously**.
- **Not a "run the demo" task**: it requires understanding what VoxCPM2 is
  actually made of (LM backbone / AudioVAE / diffusion decoder), quantizing only
  what should be quantized, and putting a **verifiable** number on the audio cost.

This repository is also the **author's own experiment record**: the baseline is
measured, one viable route has been validated up to "almost passing", and the
remaining gap is **open** (see [Task status](#task-status)).

---

## Task status

**Coexistence measurements** (true total usage via `nvidia-smi`, including the
separate `llama-server` process):

| Configuration | Peak VRAM (coexist) | RTF (coexist) | Same-seed quality | Verdict |
|---|---|---|---|---|
| bf16 full precision (baseline) | **7814 MiB** (95.4 %) | **14.4** | — | ❌ memory ceiling + 14× slower |
| nf4 on `base_lm` / `residual_lm` only | VoxCPM2 steady **2.78 GB** (coexist peak not logged separately) | **1.04** | **0.59** | ❌ both RTF and quality below the gate |

**Single-model measurements** (VoxCPM2 alone, no LLM; same seed, 10 steps):

| Precision | Peak VRAM (torch) | RTF |
|---|---|---|
| bf16 | 5457 MiB | 1.60 |
| nf4 (LM only) | **3054 MiB** | 1.63 |

> **Key finding**: quantizing only the LM backbone cuts VoxCPM2's own peak VRAM
> by **44 %** (5457 → 3054 MiB) at almost no speed cost (RTF +1.8 %).
> **But quality pays**: the same-seed comparison shows high-frequency energy up
> **17.6 %** (the signature of quantization noise) and a quality score of 0.59,
> below the 0.7 threshold.

> **RTF** = generation time ÷ audio duration. RTF ≤ 1.0 means real-time
> (generation faster than playback).

**Conclusion: this task is not solved on 8 GB.** The known route brings memory
down substantially and pulls RTF from 14.4 back to 1.04, but **both** the RTF
gate and the quality gate are narrowly missed — which is exactly the room left
for a solver.

> **Why "unsolved" is the value of this package**: it proves the task is
> **actually doable** (the baseline runs reproducibly, the metrics are measured
> automatically, and the improvement direction is verified — 44 % VRAM saved)
> while **clearly leaving room for improvement** (both gates are close but unmet).
> A solved problem has no optimization space; an unrunnable one is not a task.
> This repository deliberately sits between the two.

### How this maps to "a complete task"

| A complete task needs | Where it lives here |
|---|---|
| Task statement | `README.md` + `docs/task-spec.md` (objective, what may be modified, what to submit) |
| Runnable environment | `requirements.txt`, `Dockerfile`, `docker-compose.yml` (all paths parameterized) |
| Scoring script | `score/score.py` + `score/rubric.json` (CPU-only, automatic verdict) |
| Verified results | `results/` (measured baseline + improvement route + same-seed sample pairs) |

---

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop, **8188 MiB** |
| Python | 3.12 |
| PyTorch | 2.6.0+cu124 |
| voxcpm | **2.0.3** |
| bitsandbytes | 0.50.2 |
| Model | VoxCPM2 (~4.7 GB, ModelScope `OpenBMB/VoxCPM2`) |
| Resident LLM | 4B-class Qwen served by `llama-server` (~3.3 GB, port 8899) |

---

## Quickstart

```bash
# 0) dependencies
pip install -r requirements.txt

# 1) download the model (~4.7 GB)
python bench/download_model.py

# 2) start the resident LLM first (coexistence measurements need it listening on 8899)
#    llama-server -m <your-4B-model.gguf> --port 8899 ...
#    Without it, coexistence scripts measure too little VRAM and the result is invalid

# 3) single-model baseline: RTF / first-chunk latency / VRAM / timbre
python bench/t1_benchmark.py

# 4) coexistence, bf16 (expected to fail)
python bench/t1d_coexist.py

# 5) coexistence after quantization (the known route, expect RTF ≈ 1.04)
python bench/t1f_quant_coexist.py

# 6) ⭐ same-seed A/B: turns audio quality from "by ear" into "automatically scorable"
python bench/b09_seed_matched_quality.py

# 7) automatic scoring
python score/score.py --results results/results.json --samples results/samples \
    --rubric score/rubric.json --out score/score.json
```

> **Paths are parameterized**: model dir, output dir and total VRAM can all be
> overridden by environment variables — no machine-specific paths are hardcoded:
> `VOXCPM_MODEL_DIR`, `VOXCPM_OUT_DIR`, `GPU_TOTAL_MIB`.

---

## Repository layout

```
.
├── bench/                      # runnable measurement scripts (author's originals, parameterized)
│   ├── download_model.py       #   fetch VoxCPM2 weights
│   ├── t1_benchmark.py         #   single-model: RTF / first chunk / VRAM / timbre
│   ├── t1b_speed.py            #   squeeze RTF using existing knobs only (steps / cfg)
│   ├── t1c_artifact.py         #   write listenable samples to disk
│   ├── t1d_coexist.py          #   coexistence memory accounting (bf16, failing baseline)
│   ├── t1e_quant.py            #   quantization itself: VRAM / RTF
│   ├── t1f_quant_coexist.py    #   ⭐ coexistence after quantization (known route)
│   ├── t1g_int8.py             #   int8 control
│   └── b09_seed_matched_quality.py  # ⭐ same-seed A/B generator
├── score/
│   ├── score.py                # ⭐ automatic scoring (CPU-only, no GPU needed)
│   └── rubric.json             #   hard gates + quality-proxy weights and thresholds
├── results/
│   ├── results.json            # measured results (submitters replace `coexist` with their own)
│   └── samples/                # listenable samples (seedmatched/ holds the same-seed pair)
├── docs/
│   └── task-spec.md            # formal task definition
└── requirements.txt
```

---

## How scoring works

`score/score.py` has two layers, **fully automatic — no GPU and no human listening**:

**1. Hard gates** (from the measured numbers in `results/results.json`)

- peak coexistence VRAM ≤ **8188 MiB**
- coexistence TTS real-time factor RTF ≤ **1.0**

**2. Audio-quality proxies** (from wav pairs in `results/samples/`, drift relative to the bf16 baseline)

| Metric | What it captures | Threshold |
|---|---|---|
| log-mel spectral distance (LSD, dB) | overall timbre / detail drift | ≤ 3.0 dB |
| high-frequency energy ratio change | quantization noise / metallic artifacts | ≤ 30 % |
| spectral flux p95 change | stutter / clipping | ≤ 50 % |

These are weighted into a 0–1 quality score; **≥ 0.7 passes**.

> ⚠️ **The quality proxies are only valid when the samples share a seed.**
> Two samples drawn with different seeds differ on their own, and spectral
> distance would then mistake *sampling* variance for *quantization* loss.
> `b09_seed_matched_quality.py` produces the same-seed pair; if the seeds do not
> match, the scorer still prints numbers but marks that dimension **INVALID** and
> excludes it from the verdict.

Exit codes: `0` all passed / `1` a gate failed / `2` missing input.

---

## Known levers (reference, not the answer)

- `inference_timesteps` from 10 down to 5 / 3 — measured to cut RTF, at a quality cost
- `cfg_value=1.0` to disable CFG — saves one forward pass
- `int8` instead of `nf4` — less quantization noise, smaller memory win
- Narrow the quantization scope (only `base_lm`, keep `residual_lm` in bf16)
- **Quantize on CPU before moving to GPU** — lowers the peak during the quantization instant

Key engineering detail: **keep AudioVAE and the diffusion decoder in bf16.**
They are a small fraction of the parameters but are highly sensitive to
quantization; quantizing them buys very little memory and costs a lot of quality.

---

## Provenance

This repository was **packaged on 2026-10-11**, but the underlying work is older —
the experiments were run on **2026-09-18**.

| Date | What happened |
|---|---|
| 2026-09-18 | Upstream source cloned, environment built, VoxCPM2 weights downloaded, all `bench/` scripts written and run; raw audio outputs produced. Motivation: evaluating whether a high-quality real-time TTS model could coexist with a resident 4B LLM on an 8 GB laptop GPU, for a local assistant project. |
| 2026-10-11 | Repackaged into this self-contained task package: paths parameterized, an automatic scoring layer added, results consolidated, and the same-seed A/B generator written. |

What that means concretely:

- The **measurement scripts and their outputs are the originals** from the
  2026-09-18 session (with paths parameterized so they run off this machine).
- The **scoring layer, the task framing, and this README were written on
  2026-10-11.**
- The repository's commit history starts on 2026-10-11 because that is when the
  packaging happened — not because the experiments are that recent.

Stated plainly: **the experiments are from September; the task package is from
October.** Both statements are true and neither is inflated.

---

## Scope of original work (important)

**This project has no affiliation with OpenBMB or the official VoxCPM project.**
It is not an official VoxCPM repository.

The author **did not** participate in the design, training or code development of
VoxCPM2, and has **not** submitted any issue or pull request upstream.
VoxCPM2's role here is **the object of study**, not the author's output.

What is original to this repository:

| Original contribution | Description |
|---|---|
| Measurement scripts (`bench/`) | The methodology for measuring VRAM / RTF / first-chunk latency / coexistence, including the rule that VRAM must be read with `nvidia-smi`, not `torch.cuda` |
| Quantization strategy | Modular trade-off: quantize only `base_lm` / `residual_lm`, keep AudioVAE and the diffusion decoder in bf16 |
| Automatic scoring (`score/`) | CPU-only quality proxies plus hard gates, with weights and thresholds in `rubric.json` |
| Same-seed A/B methodology | Found that version 2.0.3 exposes no `seed` parameter, so global RNG seeding plus disabling `retry_badcase` is used to make quantization the only variable |
| Measured conclusions | The 44 % VRAM / +1.8 % RTF trade-off, and the honest finding that neither gate is met |

In one line: **upstream provides the model; this repository provides the study
of the model.** This is the same category of work as paper reproduction,
benchmarking, or writing an evaluation harness — it does not claim to be the
model author, but the evaluation itself is the author's own work.

---

## License and attribution

The code in this repository is released under **Apache-2.0**. It builds on the
following upstream work; model and weights remain the property of their authors:

- **VoxCPM / VoxCPM2** — OpenBMB · <https://github.com/OpenBMB/VoxCPM> (Apache-2.0)
- Weights: ModelScope `OpenBMB/VoxCPM2`
- Quantization: `bitsandbytes` (nf4 / int8)

This repository contains **no model weights** — only measurement and scoring code.
