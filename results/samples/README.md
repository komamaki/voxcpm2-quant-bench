# 试听样本说明

本目录下的 wav 分两类，**只有 seedmatched/ 参与自动评分**。

## ✅ seedmatched/ —— 同 seed 对照（有效）

由 `bench/b09_seed_matched_quality.py` 生成。两次生成前都重置了全局 RNG
（`torch.manual_seed` + `torch.cuda.manual_seed_all`），因此**量化是唯一变量**，
谱距离类指标才有意义。

| 文件 | 说明 |
|---|---|
| `bf16.wav` | 参考：全精度 |
| `nf4_lm.wav` | 候选：仅 `base_lm`/`residual_lm` 量化为 nf4（252 层） |
| `run_meta.json` | seed / 参数 / 版本 / 显存，评分脚本据此校验 seed 一致性 |

生成参数：seed `20261011`、`cfg_value=2.0`、`inference_timesteps=10`、48 kHz、2.40 s。

## ⚠️ 根目录下的三个 —— 非同 seed（仅供试听）

来自早期探索，**没有固定 seed**，两次采样本身就有差异，因此
**不能**用于谱距离对比（会混入随机采样差异）。保留它们只为人工试听。

| 文件 | 说明 |
|---|---|
| `baseline_bf16.wav` | 全精度基线 |
| `quant_int8.wav` | int8 量化（`Linear8bitLt`） |
| `quant_nf4_baseonly.wav` | 仅 `base_lm` 量化为 nf4 |

`score/score.py` **只**处理 `results/results.json` 里 `quality_pairs` 列出的样本对，
不会自动读取本目录下其它文件。
