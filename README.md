# VoxCPM2 · 8 GB 显存共存量化基准

> 让 **VoxCPM2 语音合成** 与 **一个 4B 级 Qwen 大模型** 在同一张 8 GB 显卡上共存，
> 且 TTS 保持**实时**（RTF ≤ 1.0）与**可用音质**。
>
> 一个自包含、**可自动评分**的任务包（task package）：任务描述 + 可复现环境 + 实测基线 + 评分脚本。

[![Task](https://img.shields.io/badge/task-open%20problem-orange)](#任务状态)
[![Scoring](https://img.shields.io/badge/scoring-automatic-brightgreen)](score/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

---

## 这是什么

这是一个**单卡显存预算优化任务**，被完整地封装成可复现、可自动评分的形态：

- **任务**：在 8188 MiB 的 RTX 4060 Laptop 上，让常驻的 4B 级 LLM 与 VoxCPM2 TTS 同时可用。
- **难点**：VoxCPM2 全精度（bf16）本身就要 ~4.5 GB，加上 LLM 的 ~3.3 GB 直接贴顶（95%），
  并且生成速度劣化 **14 倍**（RTF 1.0 → 14.4）。显存和延迟**同时**被卡死。
- **不是"跑个 demo"**：需要真正理解 VoxCPM2 的模块构成（LM 主干 / AudioVAE / 扩散解码器），
  只量化该量化的部分，并给出**可验证**的音质代价。

本仓库同时是**作者自己的实验记录**：基线已实测，一条可行的技术路线已验证到"差一点通过"，
剩下的差距是**开放的**（见 [任务状态](#任务状态)）。

---

## 任务状态

**共存测量**（`nvidia-smi` 真实总占用，含 llama-server 独立进程）：

| 配置 | 共存峰值显存 | 共存 RTF | 同 seed 音质分 | 结论 |
|---|---|---|---|---|
| bf16 全精度（基线） | **7814 MiB**（95.4%） | **14.4** | — | ❌ 显存贴顶 + 慢 14 倍 |
| nf4 仅量化 `base_lm` / `residual_lm` | VoxCPM2 稳态 **2.78 GB**（共存峰值未单独记录） | **1.04** | **0.59** | ❌ RTF 与音质双双未达标 |

**单模型测量**（仅 VoxCPM2，无 LLM 共存；同 seed，10 步）：

| 精度 | torch 峰值显存 | RTF |
|---|---|---|
| bf16 | 5457 MiB | 1.60 |
| nf4（仅 LM） | **3054 MiB** | 1.63 |

> **关键发现**：只量化 LM 主干，能把 VoxCPM2 自身的峰值显存砍掉 **44%**（5457 → 3054 MiB），
> 而速度代价几乎为零（RTF +1.8%）。**但音质有代价** —— 同 seed 对照显示
> 高频能量上升 **17.6%**（量化噪声的典型特征），音质分 0.59，低于 0.7 门槛。

> **RTF** = 生成耗时 ÷ 音频时长。RTF ≤ 1.0 才算实时（生成比播放快）。

**结论：本题在 8 GB 上尚未被解决。** 已知路线能大幅压下显存、把 RTF 从 14.4 拉回 1.04，
但 RTF 与音质**两条线都还差一点** —— 这正是留给解题者的空间。

> **为什么"没解决"反而是这个任务包的价值**：它证明了任务**确实可做**
> （基线可复现运行、指标可自动测量、改进方向已被验证有效——显存 −44%），
> 同时**明确存在提升空间**（两条门槛都逼近但未达）。一个已被解决的题目没有优化空间，
> 一个跑不通的题目不可做；本仓库刻意停在这两者之间。

### 本任务包与"一套完整任务"的对应

| 一套完整任务 | 本仓库对应物 |
|---|---|
| 任务说明 | `README.md` + `docs/任务说明.md`（明确优化目标、允许修改范围、提交物） |
| 运行环境 | `requirements.txt`、`Dockerfile`、`docker-compose.yml`（路径全部参数化） |
| 评分脚本 | `score/score.py` + `score/rubric.json`（CPU-only，程序自动判定） |
| 验证结果 | `results/`（实测基线 + 改进方案 + 同 seed 对照样本，见上表） |

---

## 环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop，**8188 MiB** |
| Python | 3.12 |
| PyTorch | 2.6.0+cu124 |
| voxcpm | **2.0.3** |
| bitsandbytes | 0.50.2 |
| 模型 | VoxCPM2（约 4.7 GB，ModelScope `OpenBMB/VoxCPM2`） |
| 常驻 LLM | 4B 级 Qwen，由 `llama-server` 提供（约 3.3 GB，端口 8899） |

---

## 快速开始

```bash
# 0) 依赖
pip install -r requirements.txt

# 1) 下载模型（约 4.7 GB）
python bench/download_model.py

# 2) 先起常驻 LLM（共存测量需要它已在 8899 监听）
#    llama-server -m <你的4B模型.gguf> --port 8899 ...
#    未启动时，共存类脚本会测到偏低的显存，结论无效

# 3) 单模型基线：RTF / 首包 / 显存 / 音色
python bench/t1_benchmark.py

# 4) 共存（bf16，预期失败）
python bench/t1d_coexist.py

# 5) 量化后共存（已知路线，预期 RTF≈1.04）
python bench/t1f_quant_coexist.py

# 6) ⭐ 同 seed A/B：让音质从「靠耳朵」变成「可自动评分」
python bench/b09_seed_matched_quality.py

# 7) 自动评分
python score/score.py --results results/results.json --samples results/samples \
    --rubric score/rubric.json --out score/score.json
```

> **路径参数化**：所有脚本的模型目录/输出目录/显存总量都可用环境变量覆盖，不写死本机路径：
> `VOXCPM_MODEL_DIR`、`VOXCPM_OUT_DIR`、`GPU_TOTAL_MIB`。

---

## 仓库结构

```
.
├── bench/                      # 可运行的测量脚本（作者原始实验，已参数化）
│   ├── download_model.py       #   拉取 VoxCPM2 权重
│   ├── t1_benchmark.py         #   单模型四项实测：RTF / 首包 / 显存 / 音色
│   ├── t1b_speed.py            #   只调已有参数压 RTF（步数 / cfg）
│   ├── t1c_artifact.py         #   落盘可试听样本
│   ├── t1d_coexist.py          #   共存显存账（bf16，失败基线）
│   ├── t1e_quant.py            #   量化本身：显存 / RTF
│   ├── t1f_quant_coexist.py    #   ⭐ 量化后共存（已知路线）
│   ├── t1g_int8.py             #   int8 对照
│   └── b09_seed_matched_quality.py  # ⭐ 同 seed A/B 生成器
├── score/
│   ├── score.py                # ⭐ 自动评分（CPU-only，无需 GPU）
│   └── rubric.json             #   硬门槛 + 音质代理的权重与阈值
├── results/
│   ├── results.json            # 实测结果（提交者需替换 coexist 为自己的测量）
│   └── samples/                # 试听样本（seedmatched/ 为同 seed 对照）
├── docs/
│   └── 任务说明.md              # 任务包定义：输入 / 输出 / 判定 / 提示
└── requirements.txt
```

---

## 评分方式

`score/score.py` 分两层，**全部自动、无需 GPU、无需人工听**：

**① 硬门槛**（来自 `results/results.json` 的实测数字）

- 共存峰值显存 ≤ **8188 MiB**
- 共存下 TTS 实时率 RTF ≤ **1.0**

**② 音质代理**（来自 `results/samples/` 的 wav 对，相对 bf16 基线的漂移）

| 指标 | 捕捉什么 | 阈值 |
|---|---|---|
| log-mel 谱距离（LSD, dB） | 音色 / 细节整体漂移 | ≤ 3.0 dB |
| 高频能量占比相对变化 | 量化噪声 / 金属感 | ≤ 30% |
| 谱通量 p95 相对变化 | 卡顿 / 爆音 | ≤ 50% |

加权合成 0~1 的音质分，**≥ 0.7 通过**。

> ⚠️ **音质代理只在样本同 seed 时有效。** 不同 seed 的两次采样本来就有差异，
> 谱距离会把"随机采样差异"误判成"量化损失"。
> `b09_seed_matched_quality.py` 负责产出同 seed 对照；
> 若样本 seed 不一致，评分脚本会照常输出数字但把该维度标为 **INVALID**，不参与结论。

退出码：`0` 全通过 / `1` 有门槛未过 / `2` 输入缺失。

---

## 已知可用的"杠杆"（供参考，非答案）

- `inference_timesteps` 从 10 降到 5 / 3 —— 已实测可降 RTF，代价是音质
- `cfg_value=1.0` 关闭 CFG —— 省一次前向
- `int8` 替代 `nf4` —— 量化噪声更小，显存收益略低
- 缩小量化范围（只动 `base_lm`，保留 `residual_lm` 为 bf16）
- **CPU 侧先量化再上 GPU** —— 压低"量化瞬间"的显存峰值

关键工程细节：**AudioVAE 与扩散解码器应保持 bf16**。它们参数量占比小，
但对音质敏感；把它们一起量化会换来很小的显存收益和很大的音质代价。

---

## English Summary

**VoxCPM2 · 8 GB VRAM Coexistence Quantization Benchmark** — a self-contained,
automatically-scorable task package.

The task: run VoxCPM2 (TTS) alongside a resident ~4B Qwen LLM on a single
8 GB laptop GPU (RTX 4060, 8188 MiB), keeping TTS **real-time** (RTF ≤ 1.0)
and the audio **usable**.

Measured baseline: bf16 uses **7814 MiB (95.4%)** and degrades RTF to **14.4**.
The known route (nf4 on `base_lm`/`residual_lm` only, keeping AudioVAE and the
diffusion decoder in bf16) brings steady VRAM down to **2.78 GB** and RTF to
**1.04** — close, but still above the gate. **The task remains open.**

Scoring is fully automatic and CPU-only: hard gates on peak VRAM and RTF, plus
audio-quality proxies (log-mel spectral distance, HF energy ratio, spectral flux
p95) computed against a bf16 reference **generated with the same seed**, so
quantization is the only variable.

---

## 本仓库的原创范围（重要）

**本项目与 OpenBMB / VoxCPM 官方没有任何隶属关系**，不是 VoxCPM 的官方仓库。

作者**没有**参与 VoxCPM2 的模型设计、训练或代码开发，也**没有**向上游提交过 issue 或 PR。
VoxCPM2 在本仓库中的角色是**研究对象**，不是作者的产出。

本仓库的原创部分是：

| 原创内容 | 说明 |
|---|---|
| 测量脚本（`bench/`） | 显存 / RTF / 首包 / 共存的实测方法，含"必须用 `nvidia-smi` 而非 `torch.cuda`"的口径 |
| 量化策略 | 模块化取舍：只量化 `base_lm` / `residual_lm`，AudioVAE 与扩散解码器保持 bf16 |
| 自动评分（`score/`） | CPU-only 音质代理指标 + 硬门槛，权重与阈值见 `rubric.json` |
| 同 seed A/B 方法学 | 发现 2.0.3 无 `seed` 参数，改用全局 RNG 播种 + 关闭 `retry_badcase`，使量化成为唯一变量 |
| 实测结论 | 显存 −44% 与 RTF +1.8% 的权衡，以及"两条门槛均未达"的诚实结论 |

一句话：**上游提供模型，本仓库提供对模型的研究。**
这属于"论文复现 / Benchmark / 评测脚本"这一类工作 —— 不声称是原模型的作者，
但评测与优化本身是作者自己的工作。

---

## 许可与署名

本仓库代码以 **Apache-2.0** 发布。它建立在以下上游工作之上，模型与权重版权归原作者：

- **VoxCPM / VoxCPM2** — OpenBMB · <https://github.com/OpenBMB/VoxCPM> （Apache-2.0）
- 权重：ModelScope `OpenBMB/VoxCPM2`
- 量化：`bitsandbytes`（nf4 / int8）

本仓库**不包含**任何模型权重，只包含测量与评分代码。
