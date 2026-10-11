# Task specification · VoxCPM2 coexisting with a 4B LLM on 8 GB VRAM

> This file is the **formal definition** of the task package: objective, inputs,
> outputs, judging criteria, and why it is not easy.
> Implementation lives in `bench/`; automatic scoring lives in `score/`.

---

## 1. Objective

On a single **8188 MiB** GPU, make both of the following **usable at the same time**:

1. **A resident LLM** — a 4B-class Qwen model served by `llama-server` (~3.3 GB), port 8899.
2. **VoxCPM2 TTS** — speech synthesis, ~4.5 GB at full precision.

All of the following must hold **simultaneously**:

- peak VRAM across the whole pipeline **≤ 8188 MiB** (no OOM, no swapping to survive)
- TTS **RTF ≤ 1.0** while coexisting (real-time)
- audio drift relative to the bf16 baseline within the quality-proxy thresholds

Subject to those three constraints, **maximize the remaining VRAM headroom**.

---

## 2. Why it is not easy

| Obstacle | Detail |
|---|---|
| Memory ceiling | In bf16 the two add up to 7814 MiB — 95.4 % of the card, not enough headroom for sustained operation |
| Speed collapse | RTF degrades from 1.0 to **14.4** while coexisting — 14 seconds to generate 1 second of audio |
| Quantization is not free | 4-bit quantization introduces noise; for TTS this shows up directly as timbre drift, metallic artifacts and clipping |
| Module heterogeneity | VoxCPM2 is not a single Transformer: the LM backbone, the AudioVAE and the diffusion decoder have completely different sensitivity to quantization |
| Peak vs steady state | The VRAM peak during the "quantization instant" can exceed the post-quantization steady state — and that instant is exactly where coexistence is most likely to OOM |

**Key insight**: memory and latency are two independent constraints; optimizing
only one does not count. Squeezing the model into VRAM while RTF grows 14× is not
a solution.

---

## 3. Inputs

| Input | Detail |
|---|---|
| Hardware | A single 8 GB-class GPU (this benchmark: RTX 4060 Laptop, 8188 MiB) |
| Model A | VoxCPM2 weights (ModelScope `OpenBMB/VoxCPM2`, ~4.7 GB) |
| Model B | 4B-class Qwen, served resident by `llama-server` (~3.3 GB) |
| Code | measurement scripts under `bench/`; `bitsandbytes` provides the quantization primitives |
| Environment variables | `VOXCPM_MODEL_DIR`, `VOXCPM_OUT_DIR`, `GPU_TOTAL_MIB` |

---

## 4. Outputs (what to submit)

Submitters should provide:

1. **`results/results.json`** — at minimum, coexistence measurements for your own
   configuration:
   ```json
   { "coexist": { "peak_vram_mib": <int>, "rtf": <float> } }
   ```
   (the other fields may be kept as-is; see the file for the full structure)

2. **A wav pair under `results/samples/`** — candidate audio **sharing a seed**
   with the bf16 baseline. Running `bench/b09_seed_matched_quality.py` produces
   this directly and records the seed in `run_meta.json`.

3. **A short method note** — which modules you quantized, why, and what it cost
   (in the PR description or under `docs/`).

---

## 5. Judging criteria

All of this is done automatically by `score/score.py` — **no human listening required**:

### 1. Hard gates

| Metric | Limit | Source |
|---|---|---|
| Peak coexistence VRAM | 8188 MiB | `results.coexist.peak_vram_mib` |
| Coexistence RTF | 1.0 | `results.coexist.rtf` |

### 2. Audio-quality proxies (relative to the bf16 baseline)

| Metric | Threshold | Weight |
|---|---|---|
| log-mel spectral distance (LSD) | ≤ 3.0 dB | 0.50 |
| high-frequency energy ratio change | ≤ 30 % | 0.25 |
| spectral flux p95 change | ≤ 50 % | 0.25 |

A weighted score of **≥ 0.7** counts as passing on quality.

### 3. Final verdict

- hard gates **and** quality both pass → `PASS`
- hard gates pass but sample seeds do not match (quality not evaluable) → `FAIL(quality unverified)`
- otherwise → `FAIL`

---

## 6. Measurement conventions (important)

**VRAM must be measured with `nvidia-smi`, not estimated with
`torch.cuda.memory_allocated()`.** The latter only accounts for PyTorch
allocations inside the current process; it cannot see the 3.3 GB held by
`llama-server`, which is a **separate process**. This is the central trap of this
task (see the design notes in `bench/t1d_coexist.py`).

**Quality comparisons require a shared seed.** Otherwise spectral distance mixes
in sampling variance and the metric is meaningless. Note that the locally
installed `voxcpm==2.0.3` `generate()` has **no** `seed` parameter, so seeding
must be done externally with `torch.manual_seed()` +
`torch.cuda.manual_seed_all()`, with `retry_badcase` disabled (its retry loop
consumes additional RNG).

---

## 7. Reference starting point

A route the author has already measured (**does not pass** — this is a starting
point, not the answer):

```
Quantization scope: only nn.Linear under base_lm. / residual_lm. → bitsandbytes nf4
Kept in bf16:       AudioVAE, diffusion decoder
Result:             VoxCPM2 steady state 2.78 GB, RTF 1.04
```

It still misses the RTF gate by a small margin, and the same-seed quality score
is 0.59 against a 0.7 threshold. Candidate directions are listed under "Known
levers" in the README.

---

## 8. Scorer exit codes

| Code | Meaning |
|---|---|
| 0 | everything passed |
| 1 | a gate failed |
| 2 | missing / non-evaluable input |
