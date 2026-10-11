"""测量本机 llama-server 的响应速度（用服务端自报的 timings）

用法：python bench_llm.py [端口]
"""

import os
import json
import sys
import time
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PORT = sys.argv[1] if len(sys.argv) > 1 else "8899"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
PROMPT = "用一句话介绍你自己。"


def main() -> None:
    body = json.dumps({
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.7, "max_tokens": 64, "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ 请求失败: {type(e).__name__}: {e}")
        return
    wall = time.perf_counter() - t0

    tim = data.get("timings") or {}
    usage = data.get("usage") or {}
    txt = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    n_out = usage.get("completion_tokens") or tim.get("predicted_n") or 0

    print(f"  输出 {n_out} tokens | 总耗时 {wall:.2f}s")
    if tim:
        pp = tim.get("prompt_per_second")
        tg = tim.get("predicted_per_second")
        ms = tim.get("prompt_ms")
        if pp:
            print(f"  ⭐ 预填充 {pp:.1f} tok/s（{ms / 1000:.2f}s）| ⭐ 生成 {tg:.1f} tok/s")
        else:
            print(f"  ⭐ 生成 {tg:.1f} tok/s" if tg else "  ⚠️ 无 timings")
    else:
        print("  ⚠️ 服务端未返回 timings")
    if txt:
        print(f"  样例输出: {txt.strip()[:60]}")


if __name__ == "__main__":
    main()
