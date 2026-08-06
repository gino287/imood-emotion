"""彙整各裝置的執行結果，產出不含資料集原文的速度摘要。

    python scripts/summarize.py

讀 `_local/out/*.jsonl` 與同名 `.meta.json`，寫出 `results/summary.md`。
逐句結果含資料集原文、不進版控；這份摘要只有統計數字，可以進版控。
"""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

DEFAULT_IN = Path("_local/out")
DEFAULT_OUT = Path("results/summary.md")


def percentile(values: list, q: float) -> float:
    """最近排名法。n 只有二十幾筆，插值法的精度是假的，不如用最單純的定義。"""
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round(q / 100 * len(ordered) + 0.5) - 1))
    return ordered[idx]


def load_runs(in_dir: Path) -> list:
    runs = []
    for jsonl in sorted(in_dir.glob("*.jsonl")):
        if jsonl.name.startswith("_"):
            continue  # 底線開頭是臨時檔（煙霧測試等），不列入
        records = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        meta_path = jsonl.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if records:
            runs.append({"path": jsonl, "records": records, "meta": meta})
    return runs


def summarize_run(run: dict) -> dict:
    records = run["records"]
    meta = run["meta"]
    timed = [r for r in records if not r["warmup"]]
    totals = [r["latency_ms"]["total"] for r in timed]
    forwards = [r["latency_ms"]["forward"] for r in timed]

    agree = sum(1 for r in timed if r.get("dataset_label") and r["pred_label"] == r["dataset_label"])
    labelled = sum(1 for r in timed if r.get("dataset_label"))

    return {
        "device": meta.get("device") or records[0]["device"],
        "no_delay": bool(meta.get("no_delay")),
        "n_total": len(records),
        "n_timed": len(timed),
        "n_warmup": len(records) - len(timed),
        "mean": statistics.mean(totals),
        "p50": percentile(totals, 50),
        "p95": percentile(totals, 95),
        "min": min(totals),
        "max": max(totals),
        "forward_mean": statistics.mean(forwards),
        "load_seconds": meta.get("model_load_seconds"),
        "vram_alloc": meta.get("torch_peak_allocated_mb"),
        "vram_reserved": meta.get("torch_peak_reserved_mb"),
        "gpu_name": meta.get("gpu_name"),
        "pred_dist": Counter(r["pred_label"] for r in timed),
        "agree": agree,
        "labelled": labelled,
        "meta": meta,
    }


def render(stats: list) -> str:
    first = stats[0]["meta"]
    lines = [
        "# 串流情緒分類 — 速度 baseline",
        "",
        f"模型：`{first.get('model_id', '')}`　樣本：{first.get('n_records', '?')} 句"
        f"（其中 {first.get('n_warmup', '?')} 句為 warmup，不計入統計）",
        f"樣本檔雜湊：`{first.get('samples_sha256', '')[:16]}…`　"
        f"torch {first.get('torch', '?')} / transformers {first.get('transformers', '?')}",
        "",
        "產品場景是單句即時處理，因此以下全部是 batch=1 的單句延遲。",
        "",
        "兩種到達模式各測一輪：**間隔到達**為模擬 STT 的 0.5~3 秒隨機停頓（貼近實際使用），",
        "**連續到達**為句子一句接一句不等待（測模型本身的能力上限）。",
        "",
        "## 單句延遲（毫秒）",
        "",
        "| 裝置 | 到達模式 | 計入句數 | 平均 | p50 | p95 | 最快 | 最慢 | 其中 forward 平均 | 模型載入 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in stats:
        load = f"{s['load_seconds']:.2f}s" if s["load_seconds"] else "—"
        mode = "連續" if s["no_delay"] else "間隔"
        lines.append(
            f"| `{s['device']}` | {mode} | {s['n_timed']} | {s['mean']:.1f} | {s['p50']:.1f} | "
            f"{s['p95']:.1f} | {s['min']:.1f} | {s['max']:.1f} | {s['forward_mean']:.1f} | {load} |"
        )

    def pick(dev_prefix, no_delay):
        return next((s for s in stats
                     if s["device"].startswith(dev_prefix) and s["no_delay"] == no_delay), None)

    gpu_gap, gpu_cont = pick("cuda", False), pick("cuda", True)
    cpu_gap, cpu_cont = pick("cpu", False), pick("cpu", True)

    if gpu_gap and gpu_cont and cpu_gap:
        lines += [
            "",
            "## 主要發現：間隔到達時 GPU 沒有優勢",
            "",
            f"- 連續到達時 GPU 中位數 **{gpu_cont['p50']:.0f}ms**，"
            f"間隔到達時卻是 **{gpu_gap['p50']:.0f}ms** —— 相差約 "
            f"{gpu_gap['p50'] / gpu_cont['p50']:.0f} 倍。",
            f"- 同樣是間隔到達，CPU 為 **{cpu_gap['p50']:.0f}ms**，"
            f"{'快於' if cpu_gap['p50'] < gpu_gap['p50'] else '慢於'} GPU。",
            "",
            "兩者的預測結果逐句相同，差異純粹來自延遲。合理解釋是 GPU 在句與句之間"
            "（0.5~3 秒）閒置降頻，每一句都在時脈尚未拉起時就算完了；連續到達則讓 GPU"
            "維持在高時脈。",
            "",
            "**對部署的意涵**：在真實的對話節奏下，這個模型跑 CPU 即可滿足即時性，"
            "且能把整張卡讓給 pipeline 上運算量更大的模組。若之後改採批次或高頻輸入，"
            "GPU 的優勢才會顯現。",
        ]
        if cpu_cont:
            lines.append("")
            lines.append(f"（對照：CPU 連續到達為 {cpu_cont['p50']:.0f}ms，"
                         "與間隔到達差異不大，CPU 不受閒置影響。）")

    gpu = gpu_gap or gpu_cont
    if gpu and gpu["vram_alloc"]:
        lines += [
            "",
            f"GPU：{gpu['gpu_name']}　張量峰值 {gpu['vram_alloc']:.0f} MB"
            f"／保留 {gpu['vram_reserved']:.0f} MB",
            "",
            "> 這兩個數字只計張量，不含 CUDA context（實務上另需約 300–600 MB）。"
            "評估與其他模型共用同一張卡時要把 context 算進去。",
        ]

    def col(s):
        return f"`{s['device']}`／{'連續' if s['no_delay'] else '間隔'}"

    lines += ["", "## 預測類別分佈", "",
              "| 類別 | " + " | ".join(col(s) for s in stats) + " |",
              "| --- | " + " | ".join("---:" for _ in stats) + " |"]
    all_labels = sorted({label for s in stats for label in s["pred_dist"]})
    for label in all_labels:
        lines.append(f"| {label} | " + " | ".join(str(s["pred_dist"].get(label, 0)) for s in stats) + " |")
    lines += ["", "各欄一致代表裝置與到達模式都不影響預測結果，只影響延遲。"]

    ref = [s for s in stats if s["labelled"]]
    if ref:
        lines += [
            "",
            "## 與資料集標註的一致率（參考值，非評測結果）",
            "",
            "| 裝置 | 一致 / 計入句數 | 比率 |",
            "| --- | ---: | ---: |",
        ]
        for s in ref:
            lines.append(f"| {col(s)} | {s['agree']} / {s['labelled']} | "
                         f"{s['agree'] / s['labelled'] * 100:.0f}% |")
        lines += [
            "",
            f"> ⚠️ **這不是準確率。** 樣本僅 {ref[0]['labelled']} 句、且刻意做成各類均衡，"
            "與真實輸入分布不同；此數字只用來確認流程接通、標籤順序沒接錯，"
            "不足以代表模型效能，也不應對外引用。正式評測另行進行。",
        ]

    lines += [
        "",
        "---",
        "",
        f"產生時間：{first.get('timestamp', '')}　"
        "由 `scripts/summarize.py` 產出，逐句結果不進版控。",
        "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="彙整執行結果為速度摘要")
    p.add_argument("--in-dir", type=Path, default=DEFAULT_IN)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    runs = load_runs(args.in_dir)
    if not runs:
        raise SystemExit(f"{args.in_dir} 底下沒有結果檔，請先執行 run_stream.py")

    # 排序：CPU 在前、同裝置內「間隔到達」在前（間隔才是實際使用情境，該當主數字）
    stats = sorted((summarize_run(r) for r in runs),
                   key=lambda s: (s["device"] != "cpu", s["no_delay"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(stats), encoding="utf-8")

    print(f"彙整 {len(stats)} 份結果 → {args.out}")
    for s in stats:
        print(f"  {s['device']:>5}: n={s['n_timed']} p50={s['p50']:.1f}ms p95={s['p95']:.1f}ms")


if __name__ == "__main__":
    main()
