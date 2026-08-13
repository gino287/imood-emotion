"""實驗 1：間隔長度 vs 推論延遲 vs GPU 實際時脈。

要回答 open_questions 裡「降頻是推論出來的、沒有直接量過時脈」這一條。
做法：掃一系列的閒置間隔，每次推論前後都用 NVML 讀 SM 時脈，
把「間隔多長 → 時脈掉到多少 → 延遲變多少」直接列成表。

另外附一個 decay 模式：連續打滿讓時脈上去之後停手，每 50ms 取樣一次時脈，
畫出降頻的時間曲線 —— 這決定了心跳（keep-alive）該多久打一次。

    python research/gpu-warmup/experiments/exp1_gap_sweep.py --mode sweep
    python research/gpu-warmup/experiments/exp1_gap_sweep.py --mode decay
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpulib import NullProbe, NvmlProbe, env_meta, load_samples, summarize  # noqa: E402
from imood_emotion.classifier import EmotionClassifier  # noqa: E402

DEFAULT_GAPS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]


def build(device: str):
    clf = EmotionClassifier(device=device)
    clf.load()
    return clf


def warmup(clf, texts, n=30):
    for i in range(n):
        clf.predict(texts[i % len(texts)])


def run_sweep(clf, probe, texts, gaps, n_min, device, min_seconds=30.0, cooldown=9.0):
    """每個間隔量到穩態為止。

    v1 版是「每個間隔固定跑 30 句、長短間隔交錯」，結果被前一個條件的餘溫汙染：
    時脈調節器的時間常數是「好幾秒」等級，30 句在小間隔下只有 1.5 秒，
    整段都還在爬升／衰減的過渡期，量到的 p50 其實是上一個條件的殘影。
    （v1 的資料留著沒刪，改名 _v1_interleaved，它正好記錄了過渡行為。）

    v2 改成：
      1. 每個條件開始前先閒置 cooldown 秒，讓時脈掉到底（實測 ~4.5s 到 210MHz），
         確保每個條件都從同一個狀態出發。
      2. 每個條件跑滿 n_min 句「且」至少 min_seconds 秒，讓它有時間走到穩態。
      3. 摘要分「全部」與「後半段（穩態）」兩種，過渡期不汙染穩態數字。
    """
    records, throttle_marks = [], {}
    for gap in sorted(gaps):
        print(f"  gap={gap}s：先閒置 {cooldown}s 讓時脈落底…", flush=True, end=" ")
        time.sleep(cooldown)
        print(f"起始 SM={probe.sm_clock()}MHz，開始量測", flush=True)
        t_cond = time.perf_counter()
        i = 0
        while i < n_min or time.perf_counter() - t_cond < min_seconds:
            if gap > 0:
                time.sleep(gap)
            before = probe.snapshot()
            t0 = time.perf_counter()
            pred = clf.predict(texts[i % len(texts)])
            total_ms = (time.perf_counter() - t0) * 1000
            after = probe.snapshot()
            records.append({
                "gap_sec": gap,
                "i": i,
                "t_in_cond": round(time.perf_counter() - t_cond, 3),
                "device": device,
                "total_ms": round(total_ms, 3),
                "forward_ms": pred.latency_ms["forward"],
                "tokenize_ms": pred.latency_ms["tokenize"],
                "sm_before": before["sm_mhz"],
                "sm_after": after["sm_mhz"],
                "power_before": before["power_w"],
                "power_after": after["power_w"],
                "temp_before": before["temp_c"],
            })
            i += 1
        # throttle_reasons 一次要 18ms，只在條件結束後抓一次，不進迴圈
        full = probe.snapshot_full()
        throttle_marks[gap] = full["throttle"]
        print(f"    → {i} 句，結束 SM={full['sm_mhz']}MHz "
              f"{full['power_w']}W {full['temp_c']}°C throttle={full['throttle']}", flush=True)
    return records, throttle_marks


def run_decay(clf, probe, texts, busy_sec=8.0, idle_sec=12.0, sample_ms=50):
    """先連續推論把時脈拉上去，再放手觀察時脈怎麼掉。"""
    trace = []
    t_start = time.perf_counter()

    print(f"  熱身階段：連續推論 {busy_sec}s", flush=True)
    i = 0
    while time.perf_counter() - t_start < busy_sec:
        clf.predict(texts[i % len(texts)])
        i += 1
        s = probe.snapshot()
        trace.append({"t": round(time.perf_counter() - t_start, 3), "phase": "busy",
                      "sm_mhz": s["sm_mhz"], "power_w": s["power_w"],
                      "temp_c": s["temp_c"]})

    print(f"  閒置階段：停手 {idle_sec}s，每 {sample_ms}ms 取樣", flush=True)
    t_idle = time.perf_counter()
    while time.perf_counter() - t_idle < idle_sec:
        s = probe.snapshot()
        trace.append({"t": round(time.perf_counter() - t_start, 3),
                      "t_idle": round(time.perf_counter() - t_idle, 3), "phase": "idle",
                      "sm_mhz": s["sm_mhz"], "power_w": s["power_w"],
                      "temp_c": s["temp_c"]})
        time.sleep(sample_ms / 1000.0)

    print("  恢復階段：重新開始連續推論，看時脈多快回來", flush=True)
    t_re = time.perf_counter()
    i = 0
    while time.perf_counter() - t_re < 4.0:
        before = probe.snapshot()
        t0 = time.perf_counter()
        clf.predict(texts[i % len(texts)])
        ms = (time.perf_counter() - t0) * 1000
        after = probe.snapshot()
        trace.append({"t": round(time.perf_counter() - t_start, 3),
                      "t_recover": round(time.perf_counter() - t_re, 3), "phase": "recover",
                      "i": i, "latency_ms": round(ms, 3),
                      "sm_mhz": before["sm_mhz"], "sm_after": after["sm_mhz"],
                      "power_w": after["power_w"], "temp_c": after["temp_c"]})
        i += 1
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sweep", "decay"], default="sweep")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-per-gap", type=int, default=30)
    ap.add_argument("--gaps", type=float, nargs="*", default=None)
    ap.add_argument("--out-dir", default="research/gpu-warmup/results")
    args = ap.parse_args()

    texts = load_samples()
    probe = NvmlProbe() if args.device.startswith("cuda") else NullProbe()
    print(f"載入模型（device={args.device}）…", flush=True)
    clf = build(args.device)
    print("熱身 30 句…", flush=True)
    warmup(clf, texts)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = env_meta()
    meta["gpu_static"] = probe.static_info()
    meta["mode"] = args.mode
    meta["device"] = args.device

    if args.mode == "sweep":
        gaps = args.gaps if args.gaps else DEFAULT_GAPS
        print(f"開始掃描 {len(gaps)} 種間隔 × {args.n_per_gap} 句", flush=True)
        recs, throttle_marks = run_sweep(clf, probe, texts, gaps, args.n_per_gap, args.device)
        name = f"exp1_sweep_{args.device}"
        # 順手把每個間隔的摘要算好，report 階段不用再算一次。
        # steady = 後半段，過渡期不算進去。
        summary = {}
        for gap in sorted(set(r["gap_sec"] for r in recs)):
            sub = [r for r in recs if r["gap_sec"] == gap]
            steady = sub[len(sub) // 2:]
            summary[str(gap)] = {
                "n": len(sub),
                "latency_all": summarize([r["total_ms"] for r in sub]),
                "latency_steady": summarize([r["total_ms"] for r in steady]),
                "sm_steady": summarize([float(r["sm_before"]) for r in steady]),
                "power_steady": summarize([float(r["power_before"]) for r in steady]),
                "temp_steady": summarize([float(r["temp_before"]) for r in steady]),
                "throttle_at_cond_end": throttle_marks.get(gap),
            }
        meta["summary"] = summary
    else:
        recs = run_decay(clf, probe, texts)
        name = f"exp1_decay_{args.device}"

    (out_dir / f"{name}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
    (out_dir / f"{name}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n寫出 {out_dir / (name + '.jsonl')}（{len(recs)} 筆）")

    if args.mode == "sweep":
        print(f"\n{'間隔':>7}{'n':>5}{'p50(穩態)':>11}{'p95(穩態)':>11}{'p50(全)':>10}"
              f"{'SM MHz':>9}{'功耗W':>8}{'溫度':>6}")
        for gap, s in meta["summary"].items():
            print(f"{gap:>7}{s['n']:>5}{s['latency_steady']['p50']:>11}"
                  f"{s['latency_steady']['p95']:>11}{s['latency_all']['p50']:>10}"
                  f"{s['sm_steady']['p50']:>9}{s['power_steady']['p50']:>8}"
                  f"{s['temp_steady']['p50']:>6}")


if __name__ == "__main__":
    main()
