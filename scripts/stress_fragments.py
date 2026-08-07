"""片段壓力測試：不完整輸入下，分類器的表現會掉多少。

    python scripts/stress_fragments.py

固定秒數切分音訊必然會切在句子中間，模型拿到的是半句話。這支腳本在還沒接上
麥克風的前提下先模擬同一件事：把完整句子在**非標點位置**隨機截斷，餵進分類器，
比對完整句與片段的預測與信心。

目的不是評測準確率，而是回答兩個問題：
  1. buffer 與分類流程碰到不完整輸入會不會壞（例如空字串、單字、只剩標點）
  2. 信心分數掉多少 —— 若掉得夠明顯，之後可以拿信心當「這句可能被切斷」的訊號

逐句結果留在本地，摘要寫 results/fragment_robustness.md（不含資料集原文）。
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imood_stream.classifier import MODEL_ID, EmotionClassifier, resolve_device  # noqa: E402

DEFAULT_SAMPLES = Path("_local/samples.jsonl")
DEFAULT_OUT = Path("results/fragment_robustness.md")
# 刻意不放 _local/out/：那裡是 run_stream.py 的執行結果，
# 混進去會被 scripts/summarize.py 一起讀走
DETAIL_OUT = Path("_local/stress/fragment_detail.jsonl")

# 保留比例的取樣範圍。低於 0.3 幾乎只剩幾個字，那已經不是「被切斷的句子」
# 而是雜訊；高於 0.9 與完整句沒有差別，測不出東西。
MIN_KEEP, MAX_KEEP = 0.3, 0.9

# 中文標點：截斷點刻意避開這些位置。
# 在標點處切等於切在語意邊界上，那是 VAD 做得好的情況，
# 而這支腳本要測的正是「切在最糟的地方」。
PUNCT = set("，。！？；：、「」『』（）…—,.!?;:")


def truncate_mid(text: str, keep_ratio: float, rng: Random) -> str:
    """截斷到 keep_ratio 長度，並把切點推離標點。"""
    n = max(1, int(len(text) * keep_ratio))
    if n >= len(text):
        return text
    # 若切點剛好落在標點（或標點的下一個字），往前挪到非標點處
    while n > 1 and (text[n - 1] in PUNCT or text[n] in PUNCT):
        n -= 1
    return text[:n]


def main():
    p = argparse.ArgumentParser(description="不完整輸入的穩定度壓力測試")
    p.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--seed", type=int, default=20260807)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    if not args.samples.exists():
        raise SystemExit(
            f"找不到 {args.samples}\n"
            "請先執行：docker compose run --rm app-cpu python scripts/prepare_samples.py --limit 200"
        )
    rows = [json.loads(l) for l in args.samples.read_text(encoding="utf-8").splitlines() if l.strip()]

    device = resolve_device(args.device)
    clf = EmotionClassifier(device=device)
    clf.load()
    print(f"\n對 {len(rows)} 句各做一次完整／截斷比對\n" + "-" * 60)

    rng = Random(args.seed)
    records, errors = [], []

    for i, row in enumerate(rows, start=1):
        text = row["text"]
        keep = rng.uniform(MIN_KEEP, MAX_KEEP)
        frag = truncate_mid(text, keep, rng)

        full_pred = clf.predict(text)
        try:
            frag_pred = clf.predict(frag)
        except Exception as exc:  # 這正是壓力測試要抓的：不完整輸入讓流程炸掉
            errors.append({"text_len": len(text), "frag_len": len(frag),
                           "error": f"{type(exc).__name__}: {exc}"})
            continue

        records.append({
            "id": row.get("id", i),
            "full_text": text,
            "frag_text": frag,
            "keep_ratio": round(len(frag) / len(text), 3),
            "full_label": full_pred.label,
            "frag_label": frag_pred.label,
            "full_conf": full_pred.confidence,
            "frag_conf": frag_pred.confidence,
            "flipped": full_pred.label != frag_pred.label,
        })
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}")

    # 額外的邊界輸入：這些是固定秒數切分真的會產生的東西
    edge_cases = {
        "空字串": "",
        "單一空白": " ",
        "只有標點": "。",
        "單一字元": "我",
        "兩個字": "為什",
        "重複字元": "啊" * 200,
    }
    print("\n邊界輸入：")
    edge_results = {}
    for name, text in edge_cases.items():
        try:
            pred = clf.predict(text)
            edge_results[name] = f"{pred.label}（信心 {pred.confidence:.3f}）"
            print(f"  {name:　<6} → {edge_results[name]}")
        except Exception as exc:
            edge_results[name] = f"**例外：{type(exc).__name__}: {exc}**"
            print(f"  {name:　<6} → 例外 {type(exc).__name__}: {exc}")

    write_outputs(records, errors, edge_results, device, args.out)


def write_outputs(records, errors, edge_results, device, out_path):
    DETAIL_OUT.parent.mkdir(parents=True, exist_ok=True)
    DETAIL_OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )

    full_c = [r["full_conf"] for r in records]
    frag_c = [r["frag_conf"] for r in records]
    flipped = [r for r in records if r["flipped"]]

    # 依保留比例分組，看信心與翻轉率隨截斷程度的變化
    buckets = {"0.3–0.5": [], "0.5–0.7": [], "0.7–0.9": []}
    for r in records:
        k = r["keep_ratio"]
        key = "0.3–0.5" if k < 0.5 else ("0.5–0.7" if k < 0.7 else "0.7–0.9")
        buckets[key].append(r)

    lines = [
        "# 不完整輸入的穩定度（片段壓力測試）",
        "",
        f"模型：`{MODEL_ID}`　裝置：`{device}`　樣本：{len(records)} 句",
        "",
        "固定秒數切分音訊必然切在句子中間。這份測試在接上麥克風前先模擬同一件事：",
        "把完整句子在**非標點位置**隨機截斷（保留 30%~90%），比對完整句與片段的預測。",
        "",
        "> 這不是準確率評測，是穩定度檢查：確認流程不會因不完整輸入而中斷，",
        "> 並量出信心分數的下降幅度。",
        "",
        "## 整體",
        "",
        "| 指標 | 完整句 | 截斷片段 |",
        "| --- | ---: | ---: |",
        f"| 平均信心 | {statistics.mean(full_c):.3f} | {statistics.mean(frag_c):.3f} |",
        f"| 中位信心 | {statistics.median(full_c):.3f} | {statistics.median(frag_c):.3f} |",
        f"| 最低信心 | {min(full_c):.3f} | {min(frag_c):.3f} |",
        "",
        f"- 信心平均下降 **{(statistics.mean(full_c) - statistics.mean(frag_c)):.3f}**"
        f"（{(1 - statistics.mean(frag_c) / statistics.mean(full_c)) * 100:.1f}%）",
        f"- 預測類別改變：**{len(flipped)} / {len(records)}"
        f"（{len(flipped) / len(records) * 100:.1f}%）**",
        f"- 執行期例外：**{len(errors)} 次**",
        "",
        "## 依截斷程度分組",
        "",
        "| 保留比例 | 句數 | 完整句平均信心 | 片段平均信心 | 類別改變率 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, group in buckets.items():
        if not group:
            continue
        fc = statistics.mean(r["full_conf"] for r in group)
        gc = statistics.mean(r["frag_conf"] for r in group)
        fl = sum(r["flipped"] for r in group) / len(group) * 100
        lines.append(f"| {key} | {len(group)} | {fc:.3f} | {gc:.3f} | {fl:.1f}% |")

    if flipped:
        flip_pairs = Counter((r["full_label"], r["frag_label"]) for r in flipped)
        lines += ["", "## 最常見的類別改變", "",
                  "| 完整句判為 | 片段判為 | 次數 |", "| --- | --- | ---: |"]
        for (a, b), n in flip_pairs.most_common(8):
            lines.append(f"| {a} | {b} | {n} |")

    lines += ["", "## 邊界輸入", "",
              "固定秒數切分真的會產生這些東西（靜音段、只切到一個字、切在標點上）。",
              "", "| 輸入 | 結果 |", "| --- | --- |"]
    for name, result in edge_results.items():
        lines.append(f"| {name} | {result} |")

    if errors:
        lines += ["", "## ⚠️ 執行期例外", ""]
        for e in errors[:10]:
            lines.append(f"- 原長 {e['text_len']} → 片段 {e['frag_len']}：`{e['error']}`")

    lines += ["", "---", "",
              "由 `scripts/stress_fragments.py` 產出。逐句結果含資料集原文，不進版控。", ""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "-" * 60)
    print(f"完整句平均信心 {statistics.mean(full_c):.3f} → 片段 {statistics.mean(frag_c):.3f}")
    print(f"類別改變 {len(flipped)}/{len(records)}"
          f"（{len(flipped) / len(records) * 100:.1f}%）"
          f"　執行期例外 {len(errors)} 次")
    print(f"摘要 → {out_path}")
    print(f"逐句 → {DETAIL_OUT}")


if __name__ == "__main__":
    main()
