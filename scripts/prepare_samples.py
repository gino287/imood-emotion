"""從 Hugging Face 抽出串流 demo 用的樣本句。

    docker compose -f docker/docker-compose.yml run --rm app-cpu \
        python scripts/prepare_samples.py --limit 25

輸出 `_local/samples.jsonl`（不進版控）並印出 SHA-256。
seed 固定，任何人重跑得到位元組完全相同的一份 —— 存腳本比存資料有意義，
demo 當天也不必再連網。
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset  # noqa: E402

from imood_emotion.labels import (  # noqa: E402
    NATIVE_LABELS,
    SAMPLE_DATASET_ID,
    SAMPLE_DATASET_REVISION,
)

DEFAULT_OUT = Path("_local/samples.jsonl")


def parse_args():
    p = argparse.ArgumentParser(description="抽出串流 demo 用的樣本句")
    p.add_argument("--limit", type=int, default=25, help="抽幾句（預設 25）")
    p.add_argument("--seed", type=int, default=20260806, help="抽樣 seed，固定才可重現")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def assert_labels(rows) -> None:
    """核對資料集的標籤集合與模型原生 8 類一致。

    這個資料集的 README 列的標籤清單與實際 data.csv 不符（README 寫有「恐懼語調」，
    實際 0 筆）。與其相信文件，不如每次執行都用實際資料驗一次 —— 資料集是活的，
    作者哪天更新了內容，這裡會立刻叫出來。
    """
    found = Counter(r["emotion"] for r in rows)
    expected = set(NATIVE_LABELS)
    actual = set(found)

    if actual != expected:
        raise SystemExit(
            "資料集標籤與模型原生 8 類不符，中止。\n"
            f"  模型原生 : {sorted(expected)}\n"
            f"  資料集   : {sorted(actual)}\n"
            f"  只在模型 : {sorted(expected - actual) or '無'}\n"
            f"  只在資料 : {sorted(actual - expected) or '無'}\n"
            "資料集內容可能已更新，請重新核對 imood_emotion/labels.py。"
        )

    print("標籤核對通過：資料集 8 類與模型原生 8 類相符")
    for label in NATIVE_LABELS:
        print(f"  {label}  {found[label]:>5} 筆")


def sample(rows, limit: int, seed: int):
    """每類輪流各取一句，取滿 limit 為止，最後打散順序。

    刻意做成各類均衡而非純隨機：25 句純隨機抽很可能漏掉某幾類，
    demo 時展示不到模型在 8 個類別上的行為。
    打散順序則是因為上游 STT 不會照情緒分組送句子。
    """
    rng = Random(seed)
    by_label = {label: [] for label in NATIVE_LABELS}
    for r in rows:
        by_label[r["emotion"]].append(r)
    for bucket in by_label.values():
        rng.shuffle(bucket)

    picked, cursor = [], 0
    while len(picked) < limit:
        label = NATIVE_LABELS[cursor % len(NATIVE_LABELS)]
        cursor += 1
        if by_label[label]:
            picked.append(by_label[label].pop())
        elif all(not b for b in by_label.values()):
            break  # 資料用完了（limit 大於資料集總量）

    rng.shuffle(picked)
    return picked


def main():
    args = parse_args()

    print(f"下載資料集：{SAMPLE_DATASET_ID}")
    print(f"  revision: {SAMPLE_DATASET_REVISION}（釘住 commit，避免上游更新後數字失去可比性）")
    ds = load_dataset(SAMPLE_DATASET_ID, split="train", revision=SAMPLE_DATASET_REVISION)

    missing = {"text", "emotion"} - set(ds.column_names)
    if missing:
        raise SystemExit(f"資料集欄位不符，缺少 {missing}；實際欄位：{ds.column_names}")

    rows = [{"text": r["text"], "emotion": r["emotion"]} for r in ds]
    print(f"總筆數：{len(rows)}\n")
    assert_labels(rows)

    picked = sample(rows, args.limit, args.seed)
    if len(picked) < args.limit:
        print(f"\n⚠️ 資料集只夠抽 {len(picked)} 句，少於要求的 {args.limit} 句")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, r in enumerate(picked, start=1):
        # dataset_label 是資料集的標註，本階段只作為人工檢視時的參考，
        # 不參與任何計算 —— MVP 量的是速度，不是準確率。
        lines.append(json.dumps(
            {"id": i, "text": r["text"], "dataset_label": r["emotion"]},
            ensure_ascii=False,
        ))
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    args.out.write_bytes(payload)

    print(f"\n寫出 {len(picked)} 句 → {args.out}")
    print(f"SHA-256: {hashlib.sha256(payload).hexdigest()}")
    print(f"seed   : {args.seed}（固定；重跑應得到相同雜湊）")


if __name__ == "__main__":
    main()
