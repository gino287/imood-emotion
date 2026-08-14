# 待GINO改寫
"""驗證模型輸出順序與標籤名的對應是否正確。

    docker compose -f docker/docker-compose.yml run --rm app-cpu \
        python baseline/checks/verify_labels.py [--per-class 60]

模型的 config.json 只有 `LABEL_0..LABEL_7` 佔位符，順序取自 model card，
**無法由模型檔案本身驗證**。而順序接錯是分類任務最典型的靜默錯誤 ——
分數會低得莫名其妙，但不會有任何錯誤訊息。

改以行為驗證：拿資料集的標註句跑一輪，看混淆矩陣的對角線在不在。
順序正確時對角線會明顯浮出；接錯時整體對角率會掉到隨機水準（八類為 12.5%）。

任何時候改動 `emotion/labels.py` 的順序、或換模型／換資料集版本，
都應該重跑這支腳本。
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from emotion.classifier import MAX_LENGTH, MODEL_ID  # noqa: E402
from emotion.labels import (  # noqa: E402
    MODEL_REVISION,
    NATIVE_LABELS,
    SAMPLE_DATASET_ID,
    SAMPLE_DATASET_REVISION,
)

# 對角率低於這個值就視為失敗。隨機水準是 12.5%，實測為 87.9%，
# 門檻取 50% —— 遠高於隨機、又留足模型本身表現不佳的空間。
MIN_DIAGONAL_RATE = 50.0
BATCH = 32


def main():
    p = argparse.ArgumentParser(description="以混淆矩陣驗證標籤順序")
    p.add_argument("--per-class", type=int, default=60, help="每類抽幾句（預設 60）")
    p.add_argument("--seed", type=int, default=20260806)
    args = p.parse_args()

    ds = load_dataset(SAMPLE_DATASET_ID, split="train", revision=SAMPLE_DATASET_REVISION)
    rows = [{"text": r["text"], "emotion": r["emotion"]} for r in ds]

    print(f"資料集 {SAMPLE_DATASET_ID}（{len(rows)} 筆）")
    found = Counter(r["emotion"] for r in rows)
    extra = set(found) - set(NATIVE_LABELS)
    if extra:
        raise SystemExit(f"資料集含模型沒有的類別 {sorted(extra)}，無法對照，請先確認標籤定義")

    rng = Random(args.seed)
    by = defaultdict(list)
    for r in rows:
        by[r["emotion"]].append(r)

    picked = []
    for label in NATIVE_LABELS:
        bucket = by[label][:]
        rng.shuffle(bucket)
        picked.extend(bucket[:args.per_class])
    print(f"抽樣 {len(picked)} 句（每類最多 {args.per_class}）\n")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model.eval()  # 關掉 dropout，否則同一句每次分數不同

    preds = []
    for i in range(0, len(picked), BATCH):
        enc = tok([r["text"] for r in picked[i:i + BATCH]], return_tensors="pt",
                  truncation=True, max_length=MAX_LENGTH, padding=True)
        with torch.inference_mode():
            logits = model(**enc).logits
        preds.extend(NATIVE_LABELS[i] for i in logits.argmax(dim=-1).tolist())

    mat = defaultdict(Counter)
    for r, pred in zip(picked, preds):
        mat[r["emotion"]][pred] += 1

    print("混淆矩陣（列＝資料集標註，欄＝模型預測）")
    print("        " + "".join(f"{lb[:2]:>6}" for lb in NATIVE_LABELS) + "   對角率")
    for lb in NATIVE_LABELS:
        total = sum(mat[lb].values())
        cells = "".join(f"{mat[lb][p]:>6}" for p in NATIVE_LABELS)
        rate = mat[lb][lb] / total * 100 if total else 0.0
        print(f"{lb[:4]:　<6}{cells}   {rate:5.1f}%")

    overall = sum(mat[lb][lb] for lb in NATIVE_LABELS) / len(picked) * 100
    chance = 100 / len(NATIVE_LABELS)
    print(f"\n整體對角率：{overall:.1f}%（隨機水準 {chance:.1f}%，門檻 {MIN_DIAGONAL_RATE:.0f}%）")

    if overall < MIN_DIAGONAL_RATE:
        raise SystemExit(
            f"\n✗ 對角率過低，標籤順序極可能接錯。\n"
            f"  請核對 emotion/labels.py 的 NATIVE_LABELS 與 model card 的 label_mapping。"
        )
    print("✓ 標籤順序正確：八類的對角線皆浮出，遠高於隨機水準")


if __name__ == "__main__":
    main()
