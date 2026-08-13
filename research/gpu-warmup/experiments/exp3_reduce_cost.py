"""實驗 3：C 路線 —— 不對抗降頻，改成把單次推論做便宜。

如果單句在低時脈下也只要 30ms，那降不降頻就沒那麼要緊了。
這支比較幾種降低單次成本的做法，重點看「間歇到達」那一欄
（連續模式本來就快，不是問題所在）：

  eager              現況
  eager_padded       固定長度 padding 的對照組 —— CUDA Graph 一定要固定形狀，
                     所以要先知道「光是 padding」本身的代價是多少
  fp16               半精度
  compile_default    torch.compile 預設模式（只做圖優化與 kernel fusion）
  compile_reduce_ov  torch.compile mode="reduce-overhead"（= 開 CUDA Graph）
  compile_ro_fp16    上面兩個一起

每個條件都在「連續」與「間歇 1.5 秒」兩種到達模式下各量一次。

    python research/gpu-warmup/experiments/exp3_reduce_cost.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpulib import NvmlProbe, env_meta, load_samples, summarize  # noqa: E402
from imood_emotion.classifier import MAX_LENGTH, MODEL_ID  # noqa: E402
from imood_emotion.labels import MODEL_REVISION  # noqa: E402

PAD_LEN = 64  # 實測字數 p99=139 但那是字元；token 數更少。64 涵蓋絕大多數句子


class Variant:
    """一種推論方式。統一介面：encode(text) → run(enc) → logits。"""

    def __init__(self, name, model, tokenizer, padded, dtype):
        self.name = name
        self.model = model
        self.tok = tokenizer
        self.padded = padded
        self.dtype = dtype

    def encode(self, text):
        if self.padded:
            return self.tok(text, return_tensors="pt", truncation=True,
                            max_length=PAD_LEN, padding="max_length")
        return self.tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)

    def predict(self, text):
        enc = self.encode(text)
        enc = {k: v.to("cuda") for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc).logits
        torch.cuda.synchronize()
        return logits


def load_base(dtype):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=dtype)
    model.to("cuda").eval()
    return tok, model


# ⚠️ 一定要一次只放一個變體在 VRAM 裡。
#
# 第一版是先把六個變體全部建好再逐一量，結果 5 份權重（fp32 約 1.08GB/份）
# 加起來超過這張卡的 4GB，Windows 的 WDDM 會默默把超出的部分搬到系統記憶體，
# 不會 OOM 但速度慘不忍睹——量到的「eager 連續 91.8ms」比 exp1 的 17.5ms 慢五倍，
# 那不是變體的差異，是爆 VRAM 的假象。
#
# 所以改成工廠函式：量完一個就整個丟掉、清快取，再建下一個。
VARIANT_FACTORIES = {
    # tf32：Ampere 以後的卡可以用 TensorFloat-32 跑 fp32 的矩陣乘法，
    # 尾數只有 10 bit 但指數範圍跟 fp32 一樣。torch 預設是關的
    # （2.x 之後為了數值可重現性改成預設關閉），開起來只要一行。
    # 放在最前面跑，因為它是全域開關，開了之後要記得關回去。
    "tf32": lambda: (torch.float32, False, None),
    "eager": lambda: (torch.float32, False, None),
    "eager_padded": lambda: (torch.float32, True, None),
    "fp16": lambda: (torch.float16, False, None),
    "compile_default": lambda: (torch.float32, True, {}),
    "compile_reduce_ov": lambda: (torch.float32, True, {"mode": "reduce-overhead"}),
    "compile_ro_fp16": lambda: (torch.float16, True, {"mode": "reduce-overhead"}),
}


def build_one(name):
    dtype, padded, compile_kwargs = VARIANT_FACTORIES[name]()
    # tf32 是全域開關而不是模型屬性，所以只能在這裡切，而且每個變體都要明確設定，
    # 不然前一個變體留下的狀態會汙染後面的比較
    torch.backends.cuda.matmul.allow_tf32 = (name == "tf32")
    torch.backends.cudnn.allow_tf32 = (name == "tf32")
    tok, model = load_base(dtype)
    if compile_kwargs is not None:
        model = torch.compile(model, **compile_kwargs)
    return Variant(name, model, tok, padded=padded,
                   dtype="fp16" if dtype is torch.float16 else "fp32")


def free(variant):
    del variant.model
    del variant.tok
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def measure(variant, probe, texts, gap, n, cooldown):
    """跑一個 (變體 × 到達模式) 組合。"""
    time.sleep(cooldown)
    recs = []
    for i in range(n):
        if gap > 0:
            time.sleep(gap)
        before = probe.snapshot()
        t0 = time.perf_counter()
        variant.predict(texts[i % len(texts)])
        ms = (time.perf_counter() - t0) * 1000
        recs.append({"variant": variant.name, "dtype": variant.dtype,
                     "padded": variant.padded, "gap_sec": gap, "i": i,
                     "total_ms": round(ms, 3), "sm_before": before["sm_mhz"],
                     "power_before": before["power_w"], "temp_before": before["temp_c"]})
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--gap", type=float, default=1.5, help="間歇模式的到達間隔")
    ap.add_argument("--cont-n", type=int, default=200, help="連續模式跑幾句")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--tag", default="", help="輸出檔名後綴，避免只跑部分變體時蓋掉完整結果")
    ap.add_argument("--out-dir", default="research/gpu-warmup/results")
    args = ap.parse_args()

    texts = load_samples()
    probe = NvmlProbe()
    names = [n for n in VARIANT_FACTORIES if not args.only or n in args.only]

    all_recs, summary = [], {}
    for name in names:
        print(f"→ {name}：載入 + 暖機（compile 的第一次很慢，屬正常）…", flush=True, end=" ")
        v = build_one(name)
        t0 = time.perf_counter()
        # 編譯型變體的前幾句包含編譯時間，一定要先燒掉
        for i in range(40):
            v.predict(texts[i % len(texts)])
        warm_sec = time.perf_counter() - t0
        vram_mb = round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
        print(f"{warm_sec:.1f}s，VRAM {vram_mb}MB", flush=True)

        cont = measure(v, probe, texts, 0.0, args.cont_n, cooldown=0.0)
        inter = measure(v, probe, texts, args.gap, args.n, cooldown=9.0)
        all_recs += cont + inter
        summary[name] = {
            "dtype": v.dtype, "padded": v.padded,
            "warmup_seconds": round(warm_sec, 1),
            "vram_allocated_mb": vram_mb,
            "continuous": summarize([r["total_ms"] for r in cont[len(cont) // 2:]]),
            f"intermittent_{args.gap}s": summarize([r["total_ms"] for r in inter]),
            "sm_intermittent_p50": summarize([float(r["sm_before"]) for r in inter])["p50"],
        }
        s = summary[name]
        print(f"   連續 p50={s['continuous']['p50']}ms   "
              f"間歇 p50={s[f'intermittent_{args.gap}s']['p50']}ms "
              f"p95={s[f'intermittent_{args.gap}s']['p95']}ms", flush=True)
        free(v)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = env_meta()
    meta.update({"gap_sec": args.gap, "n_intermittent": args.n,
                 "n_continuous": args.cont_n, "pad_len": PAD_LEN,
                 "gpu_static": probe.static_info(), "summary": summary})
    suffix = f"_{args.tag}" if args.tag else ""
    (out_dir / f"exp3_reduce_cost{suffix}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in all_recs) + "\n", encoding="utf-8")
    (out_dir / f"exp3_reduce_cost{suffix}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'變體':<20}{'連續p50':>10}{'間歇p50':>10}{'間歇p95':>10}{'暖機秒':>9}")
    for name, s in summary.items():
        i = s[f"intermittent_{args.gap}s"]
        print(f"{name:<20}{s['continuous']['p50']:>10}{i['p50']:>10}{i['p95']:>10}"
              f"{s['warmup_seconds']:>9}")


if __name__ == "__main__":
    main()
