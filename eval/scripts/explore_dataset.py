"""資料觀察（規劃書 v2 §3.2）。

這支腳本的用途是「複驗」：規劃書裡的每一個數字都應該能由它重跑出來。
若哪天數字對不上，代表資料來源被換過了。

  python scripts/explore_dataset.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from evalkit import config as cf  # noqa: E402

PATTERNS = {
    "@使用者": r"@[^\s@]",
    "URL": r"https?://",
    "email": r"[\w.]+@[\w.]+\.\w+",
    "方括號表情 [心]": r"\[[^\]]{1,6}\]",
    "話題 #話題#": r"#[^#]{1,30}#",
    "全形標點": r"[，。！？；：、（）【】]",
}


def main():
    cfg = cf.load_config()
    src = cf.resolve_source(cfg)
    data = json.loads(src.read_text(encoding="utf-8"))

    print(f"來源：{src}")
    print(f"總筆數：{len(data)}\n")

    print("=== 標籤分佈 ===")
    counts = Counter(x["label"] for x in data)
    for label, n in counts.most_common():
        print(f"  {label:10s} {n:5d}  ({n / len(data):5.1%})")
    least = counts.most_common()[-1]
    print(f"  → 抽樣硬上限由最少的 {least[0]} 決定：每類最多 {least[1]} 句\n")

    print("=== 字數分佈（決定 max_length）===")
    lengths = np.array([len(x["content"]) for x in data])
    p = np.percentile(lengths, [50, 90, 95, 99])
    print(f"  min {lengths.min()}｜median {p[0]:.0f}｜mean {lengths.mean():.1f}"
          f"｜p90 {p[1]:.0f}｜p95 {p[2]:.0f}｜p99 {p[3]:.0f}｜max {lengths.max()}")
    print(f"  → 設定檔目前 max_length={cfg['runtime']['max_length']}"
          "（字數不等於 token 數，實際截斷率由 runner 回報）\n")

    print("=== 噪音樣態 ===")
    for name, pattern in PATTERNS.items():
        n = sum(1 for x in data if re.search(pattern, x["content"]))
        print(f"  {name:18s} {n:5d} 筆  ({n / len(data):5.1%})")
    print()

    print("=== 資料品質 ===")
    texts = [x["content"] for x in data]
    empty = sum(1 for t in texts if not t.strip())
    print(f"  空值      {empty} 筆")
    print(f"  完全重複  {len(texts) - len(set(texts))} 筆（抽樣前會先去重）\n")

    print("=== 各類別範例（每類 2 句）===")
    shown = Counter()
    for x in data:
        if shown[x["label"]] < 2:
            shown[x["label"]] += 1
            print(f"  [{x['label']:8s}] {x['content'][:48]}")


if __name__ == "__main__":
    main()
