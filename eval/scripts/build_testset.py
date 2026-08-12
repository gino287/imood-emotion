"""建立凍結測試集（規劃書 v2 §3.3–3.6）。

  python scripts/build_testset.py

流程：raw → 前處理 → 去空值去重 → 分層抽樣（每類 200，seed 固定）
      → OpenCC s2twp 產生繁體變體 → 寫檔 → 算 SHA-256

固定 seed 的意義：資料集本身不進版控（SMP2020 有自己的授權），但任何人重跑這支
腳本都能得到 hash 完全一致的同一份資料。
"""
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit import config as cf          # noqa: E402
from evalkit import dataset as ds         # noqa: E402
from evalkit import preprocess as pp      # noqa: E402


def main():
    cfg = cf.load_config()
    dcfg = cfg["dataset"]
    seed = dcfg["seed"]
    per_class = dcfg["per_class"]
    labels = cfg["labels"]

    src = cf.resolve_source(cfg)
    raw = json.loads(src.read_text(encoding="utf-8"))
    print(f"來源：{src}（{len(raw)} 筆）")

    # --- 前處理 -------------------------------------------------------------
    # 先清洗再去重：清洗後才知道會不會撞在一起（例如差異只在 @名稱 的兩句）
    cleaned, seen = [], set()
    n_empty = n_dup = 0
    for item in raw:
        text = pp.clean(item["content"])
        if not text:
            n_empty += 1
            continue
        if text in seen:
            n_dup += 1
            continue
        seen.add(text)
        cleaned.append({"id": item["id"], "true_label": item["label"],
                        "text_raw": item["content"], "text_zh_cn": text})
    print(f"清洗後：{len(cleaned)} 筆（清洗後變空 {n_empty}、重複 {n_dup}）")

    # --- 分層抽樣 -----------------------------------------------------------
    pool = defaultdict(list)
    for row in cleaned:
        pool[row["true_label"]].append(row)

    rng = random.Random(seed)
    picked = []
    for label in labels:  # 依設定檔順序迭代，確保 rng 消耗順序固定
        candidates = sorted(pool[label], key=lambda r: r["id"])
        if len(candidates) < per_class:
            raise SystemExit(
                f"標籤 {label} 只有 {len(candidates)} 句，抽不出 {per_class} 句。\n"
                f"請把 configs/models.yaml 的 dataset.per_class 調小。"
            )
        picked.extend(rng.sample(candidates, per_class))

    picked.sort(key=lambda r: r["id"])
    print(f"抽樣：每類 {per_class} 句，共 {len(picked)} 句（seed={seed}）")

    # --- 繁體變體 -----------------------------------------------------------
    # 從清洗後的簡體轉出來，不是從原文轉 —— 兩個變體除了字形用詞之外必須完全相同，
    # 繁簡才會是唯一變因（規劃書 v2 §3.4）
    for row in picked:
        row["text_zh_tw"] = pp.to_traditional(row["text_zh_cn"])

    out = cf.testset_path(cfg)
    ds.write_jsonl(picked, out)
    digest = ds.sha256_of(out)

    meta = {
        "created": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "seed": seed,
        "per_class": per_class,
        "n_total": len(picked),
        "counts": dict(Counter(r["true_label"] for r in picked)),
        "source": {
            "path": str(src),
            "sha256": ds.sha256_of(src),
            "n_records": len(raw),
            "note": "用 data/raw/ 而非 data/clean/：clean 版把 [惊恐] 改成《惊恐》，"
                    "毀掉方括號表情這個情緒訊號",
        },
        "pipeline": {
            "clean_rules": [
                "全形轉半形（U+FF01–U+FF5E 與全形空格；不動 。、「」等 CJK 標點）",
                "移除 @使用者（@ 後最多 6 字元，遇分隔符提前停）",
                "#話題# 去井號留內文",
                "空白正規化",
                "方括號表情保留",
            ],
            "traditional": "OpenCC s2twp（含台灣慣用詞轉換）",
            "opencc": pp.opencc_version(),
            "dedup": "清洗後的 text_zh_cn 完全相同者只留 id 最小的一筆",
        },
        "dropped": {"empty_after_clean": n_empty, "duplicates": n_dup},
        "dataset_sha256": digest,
    }
    cf.testset_meta_path(cfg).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n→ {out}")
    print(f"→ {cf.testset_meta_path(cfg)}")
    print(f"\nSHA-256: {digest}")

    if dcfg.get("expect_sha256") != digest:
        print("\n⚠️ 請把上面這串填進 configs/models.yaml 的 dataset.expect_sha256，")
        print("   之後每次評測都會核對，確保所有模型跑的是同一份資料。")

    print("\n=== 抽樣後前 3 句（人工抽查用）===")
    for row in picked[:3]:
        print(f"  [{row['true_label']}] id={row['id']}")
        print(f"    raw   : {row['text_raw'][:60]}")
        print(f"    zh_cn : {row['text_zh_cn'][:60]}")
        print(f"    zh_tw : {row['text_zh_tw'][:60]}")


if __name__ == "__main__":
    main()
