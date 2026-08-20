# 待GINO改寫
"""L5：把候選影片排出優先順序，產出人工看表情用的短名單。

    python sourcing/deliver/shortlist.py                    # 排序並印出前 30
    python sourcing/deliver/shortlist.py --female-only      # 只留女性訊號的
    python sourcing/deliver/shortlist.py --pick 12          # 印出可直接餵給 L4A 的 --uid 參數

目標很單純：**快速做出一個有四種情緒表情的角色**，所以這裡排的是
「哪幾支最有機會一支就湊齊」，不是統計上的完整性。

排序依據分兩段：

  已跑過 L4A 的 → 直接用實測的 non_neutral_ratio 與各類切點數，這是真數字
  還沒跑過的   → 用 metadata 估分，決定接下來要跑哪幾支

估分的三個訊號（實測有效性由高到低）：

  1. **題材**：人生經歷型敘事（離職、分手、霸凌、失敗、經歷）遠比
     平穩經驗分享型口播有情緒。實測 50% vs 20% 的差距。
  2. **時長**：太短湊不出多種情緒。20 分鐘以上的命中率明顯較高。
  3. **性別**：最終要交的是女性角色，所以女性訊號加分。
     ⚠️ 這只是關鍵字猜測，準確度有限 —— 真正的確認要看封面圖，
     所以這支腳本會一併把候選的封面挑出來給 covers.py 拼縮圖牆。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sourcing.common import paths, store  # noqa: E402

# 人生經歷型敘事的訊號。實測這類影片的非中性段落佔比 40–50%，
# 而「經驗分享」型口播只有 20% 上下且多為偽陽性。
NARRATIVE = [
    "经历", "經歷", "故事", "离职", "離職", "裸辞", "裸辭", "辞职", "辭職",
    "分手", "失恋", "失戀", "霸凌", "校园暴力", "校園暴力", "抑郁", "抑鬱",
    "社恐", "焦虑", "焦慮", "后悔", "後悔", "失败", "失敗", "崩溃", "崩潰",
    "改变", "改變", "转行", "轉行", "创业", "創業", "北漂", "毕业", "畢業",
    "Q&A", "qa", "问答", "問答", "掏心", "真相", "自述", "心路", "反思",
    "第一次", "那一年", "这一年", "這一年", "人生", "成长", "成長",
]

# 女性訊號。分兩組：說話者自述類（可信度較高）與題材類（可信度中等）。
FEMALE_STRONG = [
    "我老公", "我男朋友", "我男友", "怀孕", "懷孕", "生理期", "月经", "月經",
    "宝妈", "寶媽", "当妈", "當媽", "婆婆", "闺蜜", "閨蜜", "生娃", "产后", "產後",
    "女生", "女孩", "女性", "空姐", "空乘", "小姐姐", "姐妹",
]
FEMALE_WEAK = [
    "美妆", "美妝", "护肤", "護膚", "穿搭", "化妆", "化妝", "减肥", "減肥",
    "姐姐", "妹妹", "girl", "她", "婚姻", "相亲", "相親", "择偶", "擇偶",
    "grwm", "vlog", "日常",
]
# 作者暱稱裡的女性字樣
FEMALE_NAME = ["姐", "妹", "娘", "女", "婆", "妈", "媽", "囡", "喵", "兔", "花"]


def text_of(row: dict) -> str:
    return " ".join([
        row.get("title") or "",
        row.get("desc") or "",
        " ".join(row.get("tags") or []),
        (row.get("extra") or {}).get("dynamic") or "",
    ]).lower()


def hits(text: str, words: list) -> list:
    return [w for w in words if w.lower() in text]


def score(row: dict) -> dict:
    text = text_of(row)
    name = row.get("author_name") or ""
    dur = row.get("duration_s") or 0

    nar = hits(text, NARRATIVE)
    fem_s = hits(text, FEMALE_STRONG)
    fem_w = hits(text, FEMALE_WEAK)
    fem_n = [w for w in FEMALE_NAME if w in name]

    # 時長分帶：20 分鐘以上最好，10–20 分鐘次之，5 分鐘以下幾乎湊不齊
    if dur >= 1200:
        dur_pts = 3
    elif dur >= 600:
        dur_pts = 2
    elif dur >= 300:
        dur_pts = 1
    else:
        dur_pts = 0

    female_pts = 3 * len(fem_s) + 1 * len(fem_w) + 1 * len(fem_n)

    return {
        "narrative_pts": min(len(nar), 5) * 2,
        "duration_pts": dur_pts,
        "female_pts": min(female_pts, 6),
        "narrative_hits": nar[:5],
        "female_hits": (fem_s + fem_w + fem_n)[:5],
        "likely_female": bool(fem_s) or female_pts >= 3,
    }


def main():
    store.enable_utf8_stdout()
    p = argparse.ArgumentParser(description="L5 候選短名單")
    p.add_argument("--top", type=int, default=30, help="印出前 N 名")
    p.add_argument("--female-only", action="store_true", help="只留有女性訊號的")
    p.add_argument("--min-duration", type=int, default=600,
                   help="最短秒數（預設 600，短片湊不出多種情緒）")
    p.add_argument("--pick", type=int,
                   help="另外印出前 N 名的 --uid 參數，可直接貼給 emotion_timeline.py")
    args = p.parse_args()

    meta = store.read_jsonl(paths.METADATA)
    if not meta:
        raise SystemExit(f"找不到 {paths.METADATA}，先跑 L1")
    verdicts = {r["uid"]: r["verdict"] for r in store.read_jsonl(paths.FILTERED)}

    # 已經跑過 L4A 的，用實測數字而不是估分
    measured = {}
    if paths.TIMELINE_DIR.exists():
        for f in paths.TIMELINE_DIR.glob("*.json"):
            d = json.loads(f.read_text(encoding="utf-8"))
            measured[d["uid"]] = d

    rows = []
    for r in meta:
        if verdicts.get(r["uid"]) == "drop":
            continue
        if (r.get("duration_s") or 0) < args.min_duration:
            continue
        s = score(r)
        m = measured.get(r["uid"])
        rows.append({
            **{k: r.get(k) for k in
               ("uid", "url", "title", "author_name", "author_id", "duration_s", "cover_url")},
            **s,
            "measured": bool(m),
            "non_neutral_ratio": m["non_neutral_ratio"] if m else None,
            "n_labels_present": m["n_labels_present"] if m else None,
            "clip_counts": m.get("clip_counts") if m else None,
            "est_score": s["narrative_pts"] + s["duration_pts"] + s["female_pts"],
        })

    if args.female_only:
        rows = [r for r in rows if r["likely_female"]]

    # 已實測的排前面（真數字勝過估分），各自再依分數排
    rows.sort(key=lambda r: (
        not r["measured"],
        -(r["non_neutral_ratio"] or 0),
        -r["est_score"],
    ))

    store.write_jsonl(paths.SHORTLIST, rows)

    n_fem = sum(1 for r in rows if r["likely_female"])
    print(f"候選 {len(rows)} 支（其中有女性訊號 {n_fem} 支、已實測 "
          f"{sum(1 for r in rows if r['measured'])} 支）\n")

    for i, r in enumerate(rows[:args.top], 1):
        mark = "♀" if r["likely_female"] else " "
        if r["measured"]:
            counts = "、".join(f"{k}{v}" for k, v in (r["clip_counts"] or {}).items())
            head = f"實測 非中性{r['non_neutral_ratio'] * 100:.0f}% {r['n_labels_present']}類"
        else:
            counts = "／".join(r["narrative_hits"][:3]) or "-"
            head = f"估分 {r['est_score']:2d}"
        print(f"{i:3d}.{mark} [{head}] {r['duration_s'] // 60:2d}分 "
              f"{r['author_name'][:11]:13s} {r['title'][:34]}")
        print(f"       {counts}")

    if args.pick:
        picks = [r for r in rows if not r["measured"]][:args.pick]
        print(f"\n接下來要跑 L4A 的 {len(picks)} 支：\n")
        print("  " + " ".join(f"--uid {r['uid']}" for r in picks))

    print(f"\n→ {paths.SHORTLIST}")


if __name__ == "__main__":
    main()
