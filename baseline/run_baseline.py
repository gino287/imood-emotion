# 待GINO改寫
"""串流情緒分類 baseline —— 模擬上游逐句送入，即時分類並記錄。

    docker compose -f docker/docker-compose.yml run --rm app-cpu \
        python baseline/run_baseline.py --device cpu --limit 200

走完整條 pipeline：模擬上游 → 前處理 → BERT 推論 → 輸出封包。

前置：要先跑 baseline/prepare_samples.py 產出樣本檔。

（2026-08-07 曾經有 --input mic 讓它吃真實麥克風，08/10 檢討時發現把 STT 塞進
 BERT 容器違反單一職責，整套回滾。現在音檔轉文字獨立在 stt/，不接進這條路徑。）
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotion import preprocess  # noqa: E402
from emotion.classifier import (  # noqa: E402
    MODEL_ID,
    EmotionClassifier,
    resolve_device,
)
from emotion.downstream import DownstreamWriter  # noqa: E402

import fake_stt  # noqa: E402
from recorder import Recorder  # noqa: E402

DEFAULT_SAMPLES = Path("_local/samples.jsonl")

# 前幾句只跑不計入統計：CUDA kernel autotune 會讓最初幾句明顯偏慢。
WARMUP_N = 3


def parse_args():
    p = argparse.ArgumentParser(description="串流情緒分類 baseline")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                   help="分類器裝置。指定 cuda 但 GPU 不可用時直接失敗，不靜默降級")
    p.add_argument("--limit", type=int, default=25, help="處理句數（預設 25）")
    p.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    p.add_argument("--seed", type=int, default=20260806)
    p.add_argument("--no-delay", action="store_true",
                   help="句間不等待。等待不在計時區間內，但會影響 GPU 的時脈狀態")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def run(args, clf, rec, down):
    """跑完整條 pipeline，回傳要塞進 .meta.json 的統計。"""
    samples = fake_stt.load_samples(args.samples)
    n = min(args.limit, len(samples))
    print(f"開始串流：{n} 句"
          + ("（--no-delay，不等待）" if args.no_delay else "，句間隔 0.5~3 秒隨機"))
    print("-" * 76)

    skipped = 0
    stream = fake_stt.stream(samples, limit=n, seed=args.seed, delay=not args.no_delay)
    for utt in stream:
        # 前處理。simulated 來源不做簡繁轉換：樣本資料集本來就是繁體，
        # 轉換有機會改到用詞，會讓數字跟已發布的 baseline 不能比。
        t0 = time.perf_counter()
        text, reason = preprocess.prepare(utt.text, traditional=False)
        preprocess_ms = (time.perf_counter() - t0) * 1000

        if reason:
            # 資料集的句子照理不會被擋。真的被擋了要看得見，不能默默少幾句。
            skipped += 1
            print(f"[{utt.seq:>3}/{n}] · 前處理略過：{reason}")
            continue

        utt.text = text
        pred = clf.predict(utt.text)
        warmup = utt.seq <= WARMUP_N
        rec.write(utt, pred, warmup, preprocess_ms=preprocess_ms)
        if not warmup:
            down.write(utt.text, pred.label, pred.confidence)

        tag = " [warmup]" if warmup else ""
        preview = utt.text if len(utt.text) <= 24 else utt.text[:23] + "…"
        print(f"[{utt.seq:>3}/{n}] +{utt.gap_sec:>4.1f}s  {preview:　<25}"
              f"→ {pred.label}  {pred.confidence:.3f}"
              f"  {pred.latency_ms['total']:>7.1f}ms{tag}")

    return {"preprocess_skipped": skipped}


def summarize(rec, down, out_path, down_path, device, extra):
    timed = [r["latency_ms"]["total"] for r in rec.records if not r["warmup"]]
    print("-" * 76)
    if timed:
        print(f"{device}：計入統計 {len(timed)} 句")
        print(f"  平均 {statistics.mean(timed):.1f}ms"
              f" / 中位數 {statistics.median(timed):.1f}ms"
              f" / 最快 {min(timed):.1f}ms / 最慢 {max(timed):.1f}ms")
    else:
        print("沒有任何計入統計的句子。")

    if extra.get("preprocess_skipped"):
        print(f"  前處理略過 {extra['preprocess_skipped']} 句")

    print(f"逐句結果 → {out_path}")
    print(f"執行環境 → {out_path.with_suffix('.meta.json')}")
    print(f"下游封包 → {down_path}（{down.count} 筆）")


def main():
    args = parse_args()
    device = resolve_device(args.device)

    if args.out:
        out_path = args.out
    else:
        suffix = "_nogap" if args.no_delay else ""
        out_path = Path(f"_local/out/stream_{device}{suffix}.jsonl")
    # 下游封包放獨立資料夾：與評估用的逐句結果是兩種東西，混在同一個目錄
    # 會被 baseline/checks/summarize.py 當成評估結果讀進去
    down_path = Path("_local/downstream") / out_path.name

    print(f"載入分類器 {MODEL_ID} → {device}")
    clf = EmotionClassifier(device=device)
    clf.load()
    print(f"  載入耗時 {clf.load_seconds:.2f}s\n")

    interrupted = False
    extra = {}
    with Recorder(out_path, device, MODEL_ID, args.samples) as rec, \
            DownstreamWriter(down_path) as down:
        try:
            extra.update(run(args, clf, rec, down))
        except KeyboardInterrupt:
            # 中斷時 write_meta 還是要執行，否則這一輪的統計全部遺失
            interrupted = True
            print("\n（收到中斷）")

        extra.update({
            "warmup_n": WARMUP_N,
            "seed": args.seed,
            "no_delay": args.no_delay,
            "interrupted": interrupted,
        })
        rec.write_meta(clf.load_seconds, extra=extra)

    summarize(rec, down, out_path, down_path, device, extra)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # 載入模型階段就被中斷，還沒有任何東西要收尾
        print("\n已中斷。")
        sys.exit(130)
