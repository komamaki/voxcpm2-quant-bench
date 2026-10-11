# VoxCPM2 · 8GB 共存量化基准 —— 可复现环境
#
# 目的：把**依赖版本**与**入口**固化下来，便于在 Linux 服务器上复现本基准。
#
# ⚠️ 两点如实说明：
#   1) 需要 NVIDIA GPU + nvidia-container-toolkit：
#        docker run --gpus all ...
#      Windows 下建议用 WSL2 + Docker Desktop（已启用 GPU 支持）。
#   2) 本 Dockerfile 在作者本机（Windows 原生环境）**未做端到端构建验证**。
#      作者的实际测量环境见 README 的「环境」表。此文件是复现路径，不是已验产物。
#
# 模型权重（约 4.7 GB）**不打进镜像**，运行时挂载到 /models/VoxCPM2。

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VOXCPM_MODEL_DIR=/models/VoxCPM2 \
    VOXCPM_OUT_DIR=/work/results/samples \
    GPU_TOTAL_MIB=8188

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        libsndfile1 git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# 依赖先行，利用层缓存
COPY requirements.txt ./
RUN python3 -m pip install --break-system-packages --upgrade pip \
 && python3 -m pip install --break-system-packages \
        torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
 && python3 -m pip install --break-system-packages -r requirements.txt

COPY . .

# 默认：跑自动评分（CPU-only，无需 GPU；用于验证镜像内评分链路）
CMD ["python3", "score/score.py", \
     "--results", "results/results.json", \
     "--samples", "results/samples", \
     "--rubric", "score/rubric.json", \
     "--out", "score/score.json"]
