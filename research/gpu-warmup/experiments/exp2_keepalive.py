"""實驗 2：各種「保持熱啟動」的做法，實際有沒有用、代價多少。

exp1 已經證實間歇輸入下 SM 時脈會掉到 210MHz、延遲跟著爛掉。
這支就是試各種把時脈維持住的手段，在固定的到達間隔下比延遲：

  none            對照組，什麼都不做
  matmul_*        背景執行緒每隔 N 毫秒丟一個小矩陣乘法（最便宜的心跳）
  model_*         背景執行緒每隔 N 毫秒跑一次完整 dummy 推論（最貼近真實負載）
  load            模擬 Moshi/JoyGen 也在用這張卡（持續佔用），
                  用來驗證 open_questions 裡那個但書：真實 pipeline 下
                  GPU 可能根本不會閒置，「間歇降頻」的前提可能不成立

每個條件都同時記 SM 時脈與功耗 —— 熱啟動不是免費的，
這張卡要跟 Moshi/JoyGen 共用，代價要量出來才有得討論。

    python research/gpu-warmup/experiments/exp2_keepalive.py --gap 1.5 --n 30
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpulib import NvmlProbe, env_meta, load_samples, summarize  # noqa: E402
from imood_emotion.classifier import EmotionClassifier  # noqa: E402


class Heartbeat:
    """背景執行緒，週期性丟一點 GPU 工作進去。

    torch 的 CUDA 呼叫會放掉 GIL，所以放在 python thread 裡不會卡住主執行緒。
    刻意用獨立的 CUDA stream，避免跟主推論在同一個 stream 上排隊。
    """

    def __init__(self, kind: str, interval_ms: float, size: int = 512, clf=None, text=""):
        self.kind = kind
        self.interval = interval_ms / 1000.0
        self.size = size
        self.clf = clf
        self.text = text
        self._stop = threading.Event()
        self._thread = None
        self.beats = 0
        self.stream = torch.cuda.Stream() if kind != "none" else None
        if kind.startswith("matmul") or kind == "load":
            self.a = torch.randn(size, size, device="cuda")
            self.b = torch.randn(size, size, device="cuda")

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            with torch.cuda.stream(self.stream):
                if self.kind.startswith("matmul"):
                    torch.matmul(self.a, self.b)
                elif self.kind.startswith("model"):
                    with torch.inference_mode():
                        enc = self.clf.tokenizer(self.text, return_tensors="pt",
                                                 truncation=True, max_length=self.clf.max_length)
                        enc = {k: v.to("cuda") for k, v in enc.items()}
                        self.clf.model(**enc)
                elif self.kind == "load":
                    # 模擬另一個模型持續在用卡：連跑幾次大矩陣乘法
                    for _ in range(8):
                        torch.matmul(self.a, self.b)
            self.stream.synchronize()
            self.beats += 1
            rest = self.interval - (time.perf_counter() - t0)
            if rest > 0:
                self._stop.wait(rest)

    def start(self):
        if self.kind == "none":
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


CONDITIONS = [
    # (名稱, kind, interval_ms, size)
    ("none", "none", 0, 0),
    ("matmul_256_@500ms", "matmul", 500, 256),
    ("matmul_256_@250ms", "matmul", 250, 256),
    ("matmul_256_@100ms", "matmul", 100, 256),
    ("matmul_256_@50ms", "matmul", 50, 256),
    ("matmul_1024_@100ms", "matmul", 100, 1024),
    ("model_fwd_@500ms", "model", 500, 0),
    ("model_fwd_@250ms", "model", 250, 0),
    ("model_fwd_@100ms", "model", 100, 0),
    ("model_fwd_@50ms", "model", 50, 0),
    ("load_1024x8_@50ms", "load", 50, 1024),
]


def run_condition(clf, probe, texts, name, kind, interval, size, gap, n):
    hb = Heartbeat(kind, interval, size, clf=clf, text="今天天氣還不錯")
    hb.start()
    # 心跳剛開始需要一點時間把時脈推上去，前 2 秒不計
    if kind != "none":
        time.sleep(2.0)
    else:
        # 對照組要先讓時脈掉下來，否則量到的是上一個條件留下的餘溫
        time.sleep(6.0)

    recs = []
    try:
        for i in range(n):
            time.sleep(gap)
            before = probe.snapshot()
            t0 = time.perf_counter()
            clf.predict(texts[i % len(texts)])
            ms = (time.perf_counter() - t0) * 1000
            after = probe.snapshot()
            recs.append({
                "condition": name, "kind": kind, "interval_ms": interval, "size": size,
                "gap_sec": gap, "i": i, "total_ms": round(ms, 3),
                "sm_before": before["sm_mhz"], "sm_after": after["sm_mhz"],
                "power_before": before["power_w"], "temp_before": before["temp_c"],
            })
        # throttle_reasons 一次 18ms，只在最後抓一次，不能進迴圈（見 gpulib 說明）
        throttle = probe.snapshot_full()["throttle"]
    finally:
        hb.stop()
    return recs, hb.beats, throttle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=1.5, help="模擬的句間到達間隔")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--only", nargs="*", default=None, help="只跑指定條件名稱")
    ap.add_argument("--fp16", action="store_true",
                    help="主推論與心跳都改用半精度（exp3 已證實準確率代價為零）")
    ap.add_argument("--out-dir", default="research/gpu-warmup/results")
    args = ap.parse_args()

    texts = load_samples()
    probe = NvmlProbe()
    print("載入模型…", flush=True)
    clf = EmotionClassifier(device="cuda")
    clf.load()
    if args.fp16:
        clf.model.half()
        print("模型已轉為 fp16", flush=True)
    for i in range(30):
        clf.predict(texts[i % len(texts)])

    conds = [c for c in CONDITIONS if not args.only or c[0] in args.only]
    all_recs, summary = [], {}
    for name, kind, interval, size in conds:
        print(f"→ {name} …", flush=True, end=" ")
        recs, beats, throttle = run_condition(clf, probe, texts, name, kind, interval, size,
                                              args.gap, args.n)
        all_recs += recs
        lat = [r["total_ms"] for r in recs]
        summary[name] = {
            "latency": summarize(lat),
            "sm_before_p50": summarize([float(r["sm_before"]) for r in recs])["p50"],
            "power_before_p50": summarize([float(r["power_before"]) for r in recs])["p50"],
            "temp_p50": summarize([float(r["temp_before"]) for r in recs])["p50"],
            "heartbeats": beats,
            "throttle_at_end": throttle,
        }
        print(f"p50={summary[name]['latency']['p50']}ms "
              f"SM={summary[name]['sm_before_p50']}MHz "
              f"{summary[name]['power_before_p50']}W", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = env_meta()
    meta.update({"gap_sec": args.gap, "n_per_condition": args.n,
             "dtype": "fp16" if args.fp16 else "fp32",
                 "gpu_static": probe.static_info(), "summary": summary})
    tag = f"gap{args.gap}" + ("_fp16" if args.fp16 else "")
    (out_dir / f"exp2_keepalive_{tag}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in all_recs) + "\n", encoding="utf-8")
    (out_dir / f"exp2_keepalive_{tag}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'條件':<22}{'p50':>8}{'p95':>8}{'SM MHz':>9}{'功耗W':>8}{'心跳次數':>9}")
    for name, s in summary.items():
        print(f"{name:<22}{s['latency']['p50']:>8}{s['latency']['p95']:>8}"
              f"{s['sm_before_p50']:>9}{s['power_before_p50']:>8}{s['heartbeats']:>9}")
    print(f"\n寫出 {out_dir / f'exp2_keepalive_{tag}.jsonl'}")


if __name__ == "__main__":
    main()
