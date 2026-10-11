"""下载 VoxCPM2 权重到 F:\\VoxCPM\\models\\VoxCPM2

⭐ 用 ModelScope（国内源）—— GitHub / HuggingFace 本机不通，见
   docs/开发/34 §1（本机连不上 github.com:443）

用法：
    F:\\VoxCPM\\venv\\Scripts\\python.exe F:\\VoxCPM\\download_model.py

⚠️ 可重复运行 —— ModelScope 会跳过已下载的文件。
"""

import os
from pathlib import Path

from modelscope import snapshot_download

MODEL_ID = "OpenBMB/VoxCPM2"
LOCAL_DIR = os.environ.get("VOXCPM_MODEL_DIR", r"F:\VoxCPM\models\VoxCPM2")


def main() -> None:
    print(f"下载 {MODEL_ID} → {LOCAL_DIR}")
    path = snapshot_download(MODEL_ID, local_dir=LOCAL_DIR)
    print(f"✅ 完成: {path}")

    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    count = sum(1 for f in Path(path).rglob("*") if f.is_file())
    print(f"   文件 {count} 个，合计 {total / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
