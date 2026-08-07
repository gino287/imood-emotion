"""串流情緒分類 —— 逐句即時分類並記錄。

模擬上游（第一階段，demo 備援路徑）：
    docker compose run --rm app-cpu python run_stream.py --input simulated --device cpu

真實麥克風（第二階段）：主機另開一個視窗跑錄音，這裡消費它寫出的音檔
    主機： _local/hostenv/Scripts/python scripts/record_mic.py
    容器： docker compose run --rm app python run_stream.py --input mic --device cpu

前置：模擬模式需先跑 scripts/prepare_samples.py 產出樣本檔。
"""
import argparse
import statistics
import sys
from datetime import datetime
from pathlib import Path

from imood_stream import source
from imood_stream.classifier import MODEL_ID, EmotionClassifier, resolve_device
from imood_stream.downstream import DownstreamWriter
from imood_stream.recorder import Recorder

DEFAULT_SAMPLES = Path("_local/samples.jsonl")

# 前幾句只跑不計入統計：CUDA kernel autotune 會讓最初幾句明顯偏慢。
WARMUP_N = 3

# 麥克風模式的暖機句：在使用者開口**之前**先跑完，
# 這樣使用者說的每一句都計入統計。模擬模式沿用原本的做法
# （拿樣本的前 3 句當暖機並標記），維持與第一階段 baseline 的可比性。
WARMUP_TEXTS = [
    "今天天氣看起來還不錯。",
    "我剛剛把資料整理了一遍。",
    "等一下再確認一次好了。",
]


def parse_args():
    p = argparse.ArgumentParser(description="串流情緒分類")
    p.add_argument("--input", choices=["simulated", "mic"], default="simulated",
                   help="輸入來源。mic 需先在主機執行 scripts/record_mic.py")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                   help="分類器裝置。指定 cuda 但 GPU 不可用時直接失敗，不靜默降級")
    p.add_argument("--out", type=Path, default=None)

    sim = p.add_argument_group("模擬上游（--input simulated）")
    sim.add_argument("--limit", type=int, default=25, help="處理句數（預設 25）")
    sim.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    sim.add_argument("--seed", type=int, default=20260806)
    sim.add_argument("--no-delay", action="store_true",
                     help="句間不等待。等待不在計時區間內，但會影響 GPU 的時脈狀態")

    mic = p.add_argument_group("真實麥克風（--input mic）")
    mic.add_argument("--audio-dir", type=Path, default=Path("_local/audio"))
    mic.add_argument("--queue-size", type=int, default=8,
                     help="有界佇列長度，滿了丟最舊的（預設 8 ≈ 32 秒語音）")
    mic.add_argument("--whisper-model", default="small",
                     help="faster-whisper 模型大小（預設 small）")
    mic.add_argument("--whisper-device", choices=["cuda", "cpu"], default="cuda",
                     help="Whisper 跑在哪。預設 cuda：BERT 在間隔輸入下 CPU 較快，"
                          "兩者剛好不搶資源")
    mic.add_argument("--no-traditional", action="store_true",
                     help="關閉簡轉繁。⚠️ 模型與資料集是繁體，關掉會靜默拉低準確率")
    mic.add_argument("--vad-filter", action="store_true",
                     help="開啟 faster-whisper 內建的 Silero VAD 過濾靜音段。"
                          "預設關閉（VAD 整合列為下一階段工作）")
    mic.add_argument("--keep-audio", action="store_true", help="保留轉錄後的音檔")
    return p.parse_args()


def run_simulated(args, clf, rec, down, stats):
    """第一階段的模擬串流。行為與 phase 1 完全一致，維持 baseline 可比性。"""
    samples = source.load_samples(args.samples)
    n = min(args.limit, len(samples))
    print(f"開始串流：{n} 句"
          + ("（--no-delay，不等待）" if args.no_delay else "，句間隔 0.5~3 秒隨機"))
    print("-" * 76)

    stream = source.stream(samples, limit=n, seed=args.seed, delay=not args.no_delay)
    for utt in stream:
        pred = clf.predict(utt.text)
        warmup = utt.seq <= WARMUP_N
        rec.write(utt, pred, warmup)
        if not warmup:
            down.write(utt.text, pred.label, pred.confidence)

        tag = " [warmup]" if warmup else ""
        preview = utt.text if len(utt.text) <= 24 else utt.text[:23] + "…"
        print(f"[{utt.seq:>3}/{n}] +{utt.gap_sec:>4.1f}s  {preview:　<25}"
              f"→ {pred.label}  {pred.confidence:.3f}"
              f"  {pred.latency_ms['total']:>7.1f}ms{tag}")


def run_mic(args, clf, rec, down, stats):
    """真實麥克風。轉錄在背景執行緒，分類在主執行緒，中間隔一個有界佇列。"""
    from imood_stream.mic_source import MicSource
    from imood_stream.stt import SpeechToText

    stt = SpeechToText(
        model_size=args.whisper_model,
        device=args.whisper_device,
        to_traditional=not args.no_traditional,
        vad_filter=args.vad_filter,
    )
    print(f"載入 Whisper {args.whisper_model} → {args.whisper_device}"
          + ("（含簡轉繁 s2twp）" if not args.no_traditional else "　⚠️ 已關閉簡轉繁"))
    stt.load()
    print(f"  載入耗時 {stt.load_seconds:.2f}s")

    # 暖機放在使用者開口之前：麥克風模式若沿用「前 3 句排除」，
    # 會吃掉使用者真正講的前三句，demo 時既浪費又難解釋
    for text in WARMUP_TEXTS:
        clf.predict(text)
    print(f"  分類器已用 {len(WARMUP_TEXTS)} 句內建文字暖機，"
          f"以下每一句都計入統計\n")

    mic = MicSource(stt, audio_dir=args.audio_dir, queue_size=args.queue_size,
                    keep_audio=args.keep_audio)
    mic.start()

    print(f"等待音檔 → {args.audio_dir}　佇列長度 {args.queue_size}（滿了丟最舊）")
    print("請在主機另開視窗執行：_local/hostenv/Scripts/python scripts/record_mic.py")
    print("Ctrl+C 結束。")
    print("-" * 76)

    try:
        for utt in mic.stream():
            pred = clf.predict(utt.text)
            rec.write(utt, pred, warmup=False)
            down.write(utt.text, pred.label, pred.confidence)

            preview = utt.text if len(utt.text) <= 24 else utt.text[:23] + "…"
            drop = f"  丟棄{mic.dropped}" if mic.dropped else ""
            print(f"[{utt.seq:>3}] +{utt.gap_sec:>5.1f}s  {preview:　<25}"
                  f"→ {pred.label}  {pred.confidence:.3f}"
                  f"  {pred.latency_ms['total']:>7.1f}ms"
                  f"  (轉錄 {utt.transcribe_ms:.0f}ms){drop}")
    finally:
        # 在 finally 裡回填而不是用 return：麥克風模式唯一的結束方式是 Ctrl+C，
        # 例外一拋出，函式尾端的 return 就永遠不會執行，佇列丟棄次數等統計會全部遺失。
        mic.stop()
        stats.update(mic.stats())


def main():
    args = parse_args()
    device = resolve_device(args.device)

    if args.out:
        out_path = args.out
    elif args.input == "mic":
        out_path = Path(f"_local/out/mic_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
    else:
        suffix = "_nogap" if args.no_delay else ""
        out_path = Path(f"_local/out/stream_{device}{suffix}.jsonl")
    # 下游封包放獨立資料夾：與評估用的逐句結果是兩種東西，混在同一個目錄
    # 會被 scripts/summarize.py 當成評估結果讀進去
    down_path = Path("_local/downstream") / out_path.name

    print(f"載入分類器 {MODEL_ID} → {device}")
    clf = EmotionClassifier(device=device)
    clf.load()
    print(f"  載入耗時 {clf.load_seconds:.2f}s\n")

    interrupted = False
    extra = {}
    with Recorder(out_path, device, MODEL_ID, args.samples, source=args.input) as rec, \
            DownstreamWriter(down_path) as down:
        try:
            if args.input == "mic":
                run_mic(args, clf, rec, down, extra)
            else:
                run_simulated(args, clf, rec, down, extra)
        except KeyboardInterrupt:
            # 麥克風模式唯一的結束方式就是 Ctrl+C。若不在這裡接住，
            # 下面的 write_meta 永遠不會執行，佇列丟棄次數等統計會一併遺失。
            interrupted = True
            print("\n（收到中斷）")

        extra.update({
            "warmup_n": WARMUP_N if args.input == "simulated" else 0,
            "warmup_mode": "前 N 句標記排除" if args.input == "simulated" else "啟動時預熱",
            "interrupted": interrupted,
        })
        if args.input == "simulated":
            extra.update({"seed": args.seed, "no_delay": args.no_delay})
        else:
            extra.update({
                "whisper_model": args.whisper_model,
                "whisper_device": args.whisper_device,
                "to_traditional": not args.no_traditional,
                "vad_filter": args.vad_filter,
                "queue_size": args.queue_size,
            })
        rec.write_meta(clf.load_seconds, extra=extra)

    summarize(rec, down, out_path, down_path, device, args, extra)


def summarize(rec, down, out_path, down_path, device, args, extra):
    timed = [r["latency_ms"]["total"] for r in rec.records if not r["warmup"]]
    print("-" * 76)
    if timed:
        print(f"{device}：計入統計 {len(timed)} 句")
        print(f"  平均 {statistics.mean(timed):.1f}ms"
              f" / 中位數 {statistics.median(timed):.1f}ms"
              f" / 最快 {min(timed):.1f}ms / 最慢 {max(timed):.1f}ms")
    else:
        print("沒有任何計入統計的句子。")

    if args.input == "mic":
        print(f"  音檔轉錄 {extra.get('audio_chunks_transcribed', 0)} 段"
              f"　產出語句 {extra.get('utterances_emitted', 0)} 句"
              f"　文字過濾略過 {extra.get('text_skipped', 0)} 段")
        print(f"  佇列丟棄 {extra.get('queue_dropped', 0)} 段"
              f"（佇列長度 {args.queue_size}）")
        if "transcribe_ms_mean" in extra:
            print(f"  轉錄耗時平均 {extra['transcribe_ms_mean']:.0f}ms"
                  f"（**不計入**上面的延遲，屬上游模組職責）")

    print(f"逐句結果 → {out_path}")
    print(f"執行環境 → {out_path.with_suffix('.meta.json')}")
    print(f"下游封包 → {down_path}（{down.count} 筆）")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # 載入模型階段就被中斷，還沒有任何東西要收尾
        print("\n已中斷。")
        sys.exit(130)
