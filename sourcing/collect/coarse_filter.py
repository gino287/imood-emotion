# 待GINO改寫
"""L3：免下載的粗篩，把 1188 支收斂到人工看得完的量。

    python sourcing/collect/coarse_filter.py
    python sourcing/collect/coarse_filter.py --exclude-tid 4 --exclude-tid 172   # 排除指定分區
    python sourcing/collect/coarse_filter.py --show drop                        # 看被刷掉的長什麼樣

這一層完全不連網，只吃 L1 抓回來的 metadata。改規則重跑是零成本的。

篩選思路有一件事要先講清楚：**metadata 看不出情緒，但看得出片型。**
所以這裡找的不是「開心」「生氣」這種情緒詞（實測對標題幾乎沒有作用），
而是「一個人對著鏡頭連續講話」的片型詞 —— vlog、口播、經驗分享、訪談、
脫口秀之類。片型對了，情緒起伏才有機會出現；片型不對（混剪、教程、
遊戲實況、純音樂），再怎麼篩也不會有正臉說話的素材。

判定分三檔而不是二分：
  drop   有明確的排除理由（時長不合、片型明顯不對）
  keep   命中片型詞，優先看
  maybe  兩邊都沒命中，留著但排後面
留 maybe 這一檔是因為關鍵字一定有漏 —— 直接二分的話，漏掉的會被靜默丟棄。
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sourcing.common import paths, store  # noqa: E402

# 時長界線。太短湊不出多種情緒；太長則是抽音訊、跑 STT 的成本會爆掉，
# 而且多半是直播回放、講座這類鏡頭不動的內容。
MIN_DURATION_S = 15
MAX_DURATION_S = 30 * 60

# 片型詞：這種影片通常是一個人對著鏡頭連續說話。簡繁都列，素材以簡體為主，
# 但清單裡混了繁體來源，只寫一種會漏。
POSITIVE = [
    "vlog", "訪談", "访谈", "采访", "採訪", "对话", "對話", "口播",
    "经验分享", "經驗分享", "分享", "心得", "感受", "聊聊", "唠嗑", "嘮嗑",
    "聊天", "自述", "独白", "獨白", "单口", "單口", "脱口秀", "脫口秀",
    "演讲", "演講", "吐槽", "reaction", "生活记录", "生活記錄", "日常",
    "故事", "采访实录", "一人分饰", "一人分飾",
]

# 排除詞：這種影片鏡頭裡多半沒有正臉真人，或人臉不是連續說話的。
NEGATIVE = [
    "合集", "混剪", "教程", "教学", "教學", "无人声", "無人聲",
    "纯音乐", "純音樂", "游戏实况", "遊戲實況", "实况", "實況",
    "攻略", "测评", "測評", "评测", "評測", "开箱", "開箱",
    "直播回放", "录播", "錄播", "鬼畜", "手书", "手書", "翻唱",
    "舞蹈", "电影解说", "電影解說", "影视解说", "影視解說",
    "速看", "盘点", "盤點", "动画", "動畫", "asmr", "白噪音",
]


def haystack(row: dict) -> str:
    """把所有能當文字訊號的欄位串成一串來比對。

    tag 特別重要：B站訪客模式拿不到分區名，但 UP 主自訂標籤照樣拿得到，
    而且常常直接寫「经验分享」「生活记录」；抖音那邊 hashtag 也比文案好用。
    dynamic 是 B站 UP 主發布時寫的動態文字，等於多一段免費的描述。
    """
    parts = [
        row.get("title") or "",
        row.get("desc") or "",
        " ".join(row.get("tags") or []),
        (row.get("extra") or {}).get("dynamic") or "",
    ]
    return " ".join(parts).lower()


def hits(text: str, words: list) -> list:
    return [w for w in words if w.lower() in text]


def judge(row: dict, exclude_tids: set, min_s: int, max_s: int) -> dict:
    reasons = []
    text = haystack(row)

    dur = row.get("duration_s")
    if dur is None:
        reasons.append("沒有時長資訊")
    elif dur < min_s:
        reasons.append(f"太短（{dur}s < {min_s}s）")
    elif dur > max_s:
        reasons.append(f"太長（{dur}s > {max_s}s）")

    tid = (row.get("extra") or {}).get("tid")
    if tid is not None and tid in exclude_tids:
        reasons.append(f"分區排除（tid={tid}）")

    neg = hits(text, NEGATIVE)
    if neg:
        reasons.append(f"片型排除詞：{'/'.join(neg[:4])}")

    pos = hits(text, POSITIVE)

    if reasons:
        verdict = "drop"
    elif pos:
        verdict = "keep"
    else:
        verdict = "maybe"

    return {
        "uid": row["uid"],
        "platform": row["platform"],
        "url": row["url"],
        "title": row.get("title") or "",
        "author_name": row.get("author_name") or "",
        "author_id": row.get("author_id") or "",
        "duration_s": dur,
        "verdict": verdict,
        "positive_hits": pos,
        "drop_reasons": reasons,
    }


def main():
    store.enable_utf8_stdout()
    p = argparse.ArgumentParser(description="L3 免下載粗篩")
    p.add_argument("--min-duration", type=int, default=MIN_DURATION_S)
    p.add_argument("--max-duration", type=int, default=MAX_DURATION_S)
    p.add_argument("--exclude-tid", type=int, action="append", default=[],
                   help="要排除的 B站分區 tid，可重複給。先看下面印出的分區分布再決定")
    p.add_argument("--show", choices=["keep", "maybe", "drop"],
                   help="印出該檔的前幾筆，方便檢查規則有沒有誤殺")
    p.add_argument("--show-n", type=int, default=15)
    args = p.parse_args()

    rows = store.read_jsonl(paths.METADATA)
    if not rows:
        raise SystemExit(f"找不到 {paths.METADATA}，先跑 L1")

    results = [judge(r, set(args.exclude_tid), args.min_duration, args.max_duration)
               for r in rows]
    store.write_jsonl(paths.FILTERED, results)

    verdicts = Counter(r["verdict"] for r in results)
    print(f"輸入 {len(rows)} 支\n")
    for v in ("keep", "maybe", "drop"):
        n = verdicts[v]
        print(f"  {v:6s} {n:5d}  ({n / len(rows) * 100:5.1f}%)")

    # 分區分布：訪客模式拿不到分區名，只有數字 tid。不硬編對照表 ——
    # 記錯一個數字就會靜默篩掉一整區的好素材。先把分布印出來，
    # 要排除哪些由人看過再用 --exclude-tid 指定。
    tids = Counter((r.get("extra") or {}).get("tid") for r in rows
                   if r["platform"] == "bilibili")
    if tids:
        print(f"\nB站分區分布（tid → 支數，前 15 名）："
              f"\n  ⚠️ 訪客模式的 tname 是空字串，只拿得到數字。"
              f"要排除哪些分區，開一支該 tid 的影片看是什麼類型再用 --exclude-tid 指定")
        for tid, n in tids.most_common(15):
            sample = next((r["title"] for r in rows
                           if (r.get("extra") or {}).get("tid") == tid), "")
            print(f"  tid={str(tid):6s} {n:4d} 支   例：{sample[:40]}")

    # 哪一條規則刷掉最多，用來判斷規則是不是下得太重
    reason_kinds = Counter()
    for r in results:
        for reason in r["drop_reasons"]:
            reason_kinds[reason.split("（")[0].split("：")[0]] += 1
    if reason_kinds:
        print("\n各項排除理由命中次數（一支可能命中多條）：")
        for kind, n in reason_kinds.most_common():
            print(f"  {kind:16s} {n:5d}")

    pos_kinds = Counter()
    for r in results:
        for w in r["positive_hits"]:
            pos_kinds[w] += 1
    if pos_kinds:
        print("\n片型詞命中次數（前 15）：")
        for w, n in pos_kinds.most_common(15):
            print(f"  {w:14s} {n:5d}")

    if args.show:
        picked = [r for r in results if r["verdict"] == args.show][:args.show_n]
        print(f"\n【{args.show}】前 {len(picked)} 筆：")
        for r in picked:
            note = ("命中 " + "/".join(r["positive_hits"][:3])) if r["positive_hits"] \
                else ("；".join(r["drop_reasons"]) or "無命中")
            print(f"  {r['duration_s'] or '?':>5}s  {r['title'][:44]:46s} [{note}]")

    print(f"\n→ {paths.FILTERED}")


if __name__ == "__main__":
    main()
