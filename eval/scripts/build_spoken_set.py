# 待GINO改寫
"""從實際影片的逐字稿抽出一份「口語領域」的待標註測試集。

  python eval/scripts/build_spoken_set.py --per-class 40
  python eval/scripts/build_spoken_set.py --finalize     # 標註完成後轉成正式測試集

**為什麼需要這一份。**

現在的凍結測試集是 SMP2020-EWECT，那是**微博的書面貼文**。
但產品的實際輸入是 麥克風 → STT → BERT，也就是**口語的逐字稿**。
兩者的差異不是小事：

  微博：「想去日本台湾冰岛英国北欧，签证说no，小短假也说no」
  口語：「就是那個時候我真的覺得 嗯 有點撐不下去了」

口語有語助詞、有重複、有沒講完的句子、還有 STT 的錯字
（實測看到「大專→大標」「幹滿→乾滿」「專任→專人」）。
拿微博的數字去推論產品表現，方向可能整個是錯的 ——
johnson-small 在 SMP2020 上 Macro-F1 只有 0.40，但同一個模型在它自己的
對話語氣資料集上對角率有 87.9%。差距大半來自領域不合，不是模型不行。

所以「怎麼讓準確率更好」的第一步，是**先有一把量對領域的尺**。

素材是現成的：sourcing/ 的 L4A 已經對 31 支影片跑完 STT + BERT，
共一萬多段逐字稿，每一段都有模型的預測與信心值。這支腳本從中分層抽樣，
產出一份待人工標註的表；標註完就是一份領域相符的測試集。

⚠️ 抽樣刻意**依模型的預測**分層，不是隨機抽。理由是隨機抽會有一半以上
   都是平淡語氣（實測佔 50.1%），標了也看不出各類的差異。依預測分層可以
   讓每一類都有夠多樣本，代價是這份集合的類別分布不等於真實分布 ——
   所以它適合看 per-class 的精確率，不適合直接拿來報整體準確率。
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from emotion.labels import NATIVE_LABELS  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
TIMELINES = REPO / "_local" / "sourcing" / "timelines"
OUT_DIR = REPO / "eval" / "data" / "spoken"
SHEET = OUT_DIR / "to_label.jsonl"
FINAL = OUT_DIR / "spoken_v1.jsonl"

# 太短的句子沒有足以判斷情緒的內容，人工也標不出來，不要浪費標註成本
MIN_CHARS = 6


def collect() -> list:
    if not TIMELINES.exists():
        raise SystemExit(
            f"找不到 {TIMELINES}\n先跑 sourcing/detect/text_timeline.py 產出逐字稿"
        )
    rows = []
    for f in sorted(TIMELINES.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for i, seg in enumerate(d["timeline"]):
            if not seg.get("raw_label"):
                continue
            text = (seg.get("text") or "").strip()
            if len(text) < MIN_CHARS:
                continue
            rows.append({
                "src_uid": d["uid"],
                "src_title": d.get("title", ""),
                "seg_index": i,
                "start": seg["start"],
                "end": seg["end"],
                "text": text,
                "pred_raw_label": seg["raw_label"],
                "pred_confidence": seg["confidence"],
            })
    return rows


def build_sheet(per_class: int, seed: int) -> None:
    rows = collect()
    print(f"逐字稿共 {len(rows)} 段（已濾掉少於 {MIN_CHARS} 字的）")
    dist = Counter(r["pred_raw_label"] for r in rows)
    print("模型預測的分布：")
    for k in NATIVE_LABELS:
        n = dist[k]
        print(f"  {k:6s} {n:6d}  {n / len(rows) * 100:5.1f}%")

    by_label = defaultdict(list)
    for r in rows:
        by_label[r["pred_raw_label"]].append(r)

    rnd = random.Random(seed)
    picked = []
    for label in NATIVE_LABELS:
        pool = by_label[label]
        rnd.shuffle(pool)
        take = pool[:per_class]
        picked.extend(take)
        if len(take) < per_class:
            print(f"  ⚠️ {label} 只有 {len(take)} 段，不足 {per_class}")

    rnd.shuffle(picked)   # 打散，避免標註時被同一類連續出現帶著走
    for i, r in enumerate(picked):
        r["id"] = i
        r["true_label"] = ""      # 由人填
        r["note"] = ""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SHEET.open("w", encoding="utf-8") as fh:
        for r in picked:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n抽出 {len(picked)} 段 → {SHEET}")
    print("\n請把每一列的 true_label 填成六類之一："
          "\n  happy / angry / sad / fear / surprise / neutral"
          "\n判斷依據是**這句話聽起來的情緒**，不是話題的情緒。"
          "\n完全看不出來的填 unclear，之後會被排除。"
          "\n填完跑：python eval/scripts/build_spoken_set.py --finalize")


def finalize() -> None:
    if not SHEET.exists():
        raise SystemExit(f"找不到 {SHEET}，先跑一次不帶 --finalize 的版本")
    rows = [json.loads(l) for l in SHEET.open(encoding="utf-8") if l.strip()]
    labeled = [r for r in rows if r.get("true_label") and r["true_label"] != "unclear"]
    if not labeled:
        raise SystemExit(f"{SHEET} 裡還沒有任何 true_label，先標註")

    out = []
    for i, r in enumerate(labeled):
        # 欄位對齊 dataset_v1.jsonl，evalkit 才能直接吃。
        # 口語逐字稿本來就是繁體（preprocess 已做過 s2twp），
        # 兩個變體先都放同一份，等真的要做繁簡 A/B 再說。
        out.append({
            "id": i,
            "text_raw": r["text"],
            "text_zh_cn": r["text"],
            "text_zh_tw": r["text"],
            "true_label": r["true_label"],
            "src": {"uid": r["src_uid"], "start": r["start"], "end": r["end"]},
        })

    FINAL.parent.mkdir(parents=True, exist_ok=True)
    with FINAL.open("w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "created": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "n_total": len(out),
        "counts": dict(Counter(r["true_label"] for r in out)),
        "n_skipped_unclear": len(rows) - len(labeled),
        "source": "sourcing/ 的 L4A 逐字稿（faster-whisper small + OpenCC s2twp）",
        "note": ("依模型預測分層抽樣，類別分布不等於真實口語分布。"
                 "適合看 per-class 表現，不適合報整體準確率。"),
    }
    (FINAL.parent / "spoken_v1.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已標註 {len(labeled)} 段（跳過 unclear {meta['n_skipped_unclear']} 段）")
    print("類別分布：", meta["counts"])
    print(f"\n→ {FINAL}")
    print("\n要用它評測，在 configs/models.yaml 的 dataset.path 指向這份，"
          "並把 expect_sha256 設成 null（或填新的雜湊）")


def main():
    p = argparse.ArgumentParser(description="建立口語領域的待標註測試集")
    p.add_argument("--per-class", type=int, default=40,
                   help="每個「模型預測類別」抽幾段（預設 40，八類共 320 段）")
    p.add_argument("--seed", type=int, default=20260818)
    p.add_argument("--finalize", action="store_true",
                   help="標註完成後，把 to_label.jsonl 轉成正式測試集")
    args = p.parse_args()

    if args.finalize:
        finalize()
    else:
        build_sheet(args.per_class, args.seed)


if __name__ == "__main__":
    main()
