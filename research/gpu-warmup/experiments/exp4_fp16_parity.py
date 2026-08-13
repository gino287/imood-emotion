"""實驗 4：fp16 到底賠了多少準確率？

exp3 量到 fp16 在間歇模式下把延遲從 126ms 砍到 62ms、VRAM 從 1073MB 砍到 547MB。
這兩個都是本專案最缺的東西（延遲、4GB 卡要跟 Moshi/JoyGen 共用）。
但速度的便宜不能白拿 —— 這支檢查 fp16 與 fp32 的輸出差多少。

比的是「同一批句子、同一個模型、只差 dtype」，所以任何差異都純粹來自精度：

  * 預測標籤一致率（最重要 —— 不一致代表下游會拿到不同的情緒）
  * 信心值的最大差、平均絕對差
  * 整個機率向量的最大差
  * 兩者各自的準確率（有標註的那份）

    python research/gpu-warmup/experiments/exp4_fp16_parity.py
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpulib import env_meta  # noqa: E402
from imood_emotion.classifier import MAX_LENGTH, MODEL_ID  # noqa: E402
from imood_emotion.labels import MODEL_REVISION, NATIVE_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


def run_all(dtype, texts):
    """一個 dtype 跑完整批，回傳每句的完整機率向量。跑完就釋放，避免兩份權重同時佔 VRAM。"""
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=dtype)
    model.to("cuda").eval()

    out = []
    for text in texts:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        enc = {k: v.to("cuda") for k, v in enc.items()}
        with torch.inference_mode():
            logits = model(**enc).logits
        # 一律轉回 float32 再做 softmax：fp16 的 softmax 本身也有精度損失，
        # 那一段跟「模型權重用 fp16」是兩回事，這裡只想量後者
        probs = torch.softmax(logits.float(), dim=-1)[0].tolist()
        out.append(probs)

    del model, tok
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="_local/samples.jsonl")
    ap.add_argument("--text-field", default="text",
                    help="eval 測試集用 text_zh_tw，samples.jsonl 用 text")
    ap.add_argument("--gold-field", default="dataset_label",
                    help="eval 測試集的 true_label 是 6 類，跟原生 8 類對不上，"
                         "那種情況只比一致率不算準確率")
    ap.add_argument("--out", default="research/gpu-warmup/results/exp4_fp16_parity.json")
    args = ap.parse_args()

    rows = [json.loads(x) for x in
            (ROOT / args.samples).read_text(encoding="utf-8").splitlines() if x.strip()]
    texts = [r[args.text_field] for r in rows]
    gold = [r.get(args.gold_field) for r in rows]

    print(f"fp32 跑 {len(texts)} 句…", flush=True)
    p32 = run_all(torch.float32, texts)
    print(f"fp16 跑 {len(texts)} 句…", flush=True)
    p16 = run_all(torch.float16, texts)

    lab32 = [NATIVE_LABELS[max(range(8), key=lambda i: p[i])] for p in p32]
    lab16 = [NATIVE_LABELS[max(range(8), key=lambda i: p[i])] for p in p16]
    conf32 = [max(p) for p in p32]
    conf16 = [max(p) for p in p16]

    agree = sum(a == b for a, b in zip(lab32, lab16))
    conf_abs = [abs(a - b) for a, b in zip(conf32, conf16)]
    vec_abs = [max(abs(a - b) for a, b in zip(x, y)) for x, y in zip(p32, p16)]

    disagreements = [
        {"text": t, "fp32": a, "fp32_conf": round(ca, 4),
         "fp16": b, "fp16_conf": round(cb, 4), "gold": g}
        for t, a, b, ca, cb, g in zip(texts, lab32, lab16, conf32, conf16, gold)
        if a != b
    ]

    res = {
        "n": len(texts),
        "label_agreement": round(agree / len(texts), 4),
        "n_disagree": len(texts) - agree,
        "confidence_diff": {
            "mean_abs": round(sum(conf_abs) / len(conf_abs), 6),
            "max_abs": round(max(conf_abs), 6),
        },
        "prob_vector_diff": {
            "mean_of_max_abs": round(sum(vec_abs) / len(vec_abs), 6),
            "max_abs": round(max(vec_abs), 6),
        },
        "disagreements": disagreements,
        "env": env_meta(),
    }
    if gold and all(g in NATIVE_LABELS for g in gold):
        res["accuracy_fp32"] = round(sum(a == g for a, g in zip(lab32, gold)) / len(gold), 4)
        res["accuracy_fp16"] = round(sum(a == g for a, g in zip(lab16, gold)) / len(gold), 4)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n標籤一致率      {res['label_agreement']:.4f}"
          f"（{res['n_disagree']}/{res['n']} 句不一致）")
    print(f"信心平均絕對差  {res['confidence_diff']['mean_abs']:.6f}"
          f"  最大 {res['confidence_diff']['max_abs']:.6f}")
    print(f"機率向量最大差  平均 {res['prob_vector_diff']['mean_of_max_abs']:.6f}"
          f"  最大 {res['prob_vector_diff']['max_abs']:.6f}")
    if "accuracy_fp32" in res:
        print(f"準確率          fp32 {res['accuracy_fp32']}  fp16 {res['accuracy_fp16']}")
    for d in disagreements[:10]:
        print(f"  不一致：「{d['text']}」 fp32={d['fp32']}({d['fp32_conf']}) "
              f"fp16={d['fp16']}({d['fp16_conf']}) gold={d['gold']}")
    print(f"\n寫出 {out}")


if __name__ == "__main__":
    main()
