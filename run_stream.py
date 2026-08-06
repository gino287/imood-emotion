"""串流情緒分類 MVP —— 模擬上游逐句送入，即時分類並記錄。

    docker compose run --rm app-cpu python run_stream.py --device cpu
    docker compose run --rm app     python run_stream.py --device cuda

前置：先跑 scripts/prepare_samples.py 產出樣本檔。
"""
import argparse
import statistics
from pathlib import Path

from imood_stream import source
from imood_stream.classifier import MODEL_ID, EmotionClassifier, resolve_device
from imood_stream.recorder import Recorder

DEFAULT_SAMPLES = Path("_local/samples.jsonl")

# 前幾句只跑不計入統計：CUDA kernel autotune 會讓最初幾句明顯偏慢。
# 用 3 句而非評測慣用的 20 句 —— 這裡總共才 25 句，砍 20 句不合理，
# 3 句足以吃掉主要的 autotune 成本。這些句子仍會寫進 jsonl 並標記 warmup。
WARMUP_N = 3


def parse_args():
    p = argparse.ArgumentParser(description="串流情緒分類 MVP")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                   help="指定 cuda 但 GPU 不可用時直接失敗，不靜默降級；auto 才會降級")
    p.add_argument("--limit", type=int, default=25, help="處理句數（預設 25）")
    p.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    p.add_argument("--out", type=Path, default=None,
                   help="預設 _local/out/stream_<device>.jsonl")
    p.add_argument("--seed", type=int, default=20260806,
                   help="固定句序與間隔，讓不同裝置跑在相同條件下")
    p.add_argument("--no-delay", action="store_true",
                   help="跳過句間的隨機等待。不影響延遲數字（等待在計時區間之外）")
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    # 檔名帶上執行條件：有無句間間隔會讓 GPU 數字差一個量級（閒置降頻），
    # 兩種條件的結果混在同一個檔名下會分不出來
    suffix = "_nogap" if args.no_delay else ""
    out_path = args.out or Path(f"_local/out/stream_{device}{suffix}.jsonl")

    samples = source.load_samples(args.samples)
    n = min(args.limit, len(samples))

    print(f"載入模型 {MODEL_ID} → {device}")
    clf = EmotionClassifier(device=device)
    clf.load()
    print(f"  載入耗時 {clf.load_seconds:.2f}s\n")

    print(f"開始串流：{n} 句"
          + ("（--no-delay，不等待）" if args.no_delay else "，句間隔 0.5~3 秒隨機"))
    print("-" * 72)

    with Recorder(out_path, device, MODEL_ID, args.samples) as rec:
        stream = source.stream(samples, limit=n, seed=args.seed, delay=not args.no_delay)
        for utt in stream:
            pred = clf.predict(utt.text)
            warmup = utt.seq <= WARMUP_N
            rec.write(utt, pred, warmup)

            tag = " [warmup]" if warmup else ""
            preview = utt.text if len(utt.text) <= 24 else utt.text[:23] + "…"
            print(f"[{utt.seq:>2}/{n}] +{utt.gap_sec:>4.1f}s  {preview:　<25}"
                  f"→ {pred.label}  {pred.confidence:.3f}"
                  f"  {pred.latency_ms['total']:>7.1f}ms{tag}")

        rec.write_meta(clf.load_seconds, extra={
            "warmup_n": WARMUP_N,
            "seed": args.seed,
            "no_delay": args.no_delay,
        })

    timed = [r["latency_ms"]["total"] for r in rec.records if not r["warmup"]]
    print("-" * 72)
    if timed:
        print(f"{device}：計入統計 {len(timed)} 句（另有 {WARMUP_N} 句 warmup 排除）")
        print(f"  平均 {statistics.mean(timed):.1f}ms"
              f" / 中位數 {statistics.median(timed):.1f}ms"
              f" / 最快 {min(timed):.1f}ms / 最慢 {max(timed):.1f}ms")
    print(f"逐句結果 → {out_path}")
    print(f"執行環境 → {out_path.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
