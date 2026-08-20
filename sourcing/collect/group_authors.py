# 待GINO改寫
"""L2：以「人」為單位分組，產出候選人物排行。

    python sourcing/collect/group_authors.py
    python sourcing/collect/group_authors.py --min-videos 3     # 只列出有 3 支以上的作者

原本的用意是「找出在清單裡出現多次的作者，那些人才是值得往下挖的礦脈」。

⚠️ **2026-08-18 實測結果推翻了這個前提，這支腳本的結論比它原本的用途更重要。**
   857 支 B站影片跑出來是 **857 個不同作者，零重複**（唯一重複的名字是
   「账号已注销」，但 mid 各不相同）。也就是 JoyGen 這份清單是刻意
   **一人只收一支** —— 對 lip-sync 訓練集來說很合理，那種模型要的是
   說話人多樣性，不是同一個人的多支影片。

   所以「在清單內找同一個人的多支影片」這條路是**走不通的**，不是資料不夠，
   是結構上就不存在。剩下兩條路：

   A. **單支影片內找情緒起伏** —— 一支 30 分鐘的 Q&A 或人生經歷分享，
      本來就可能同時有笑有淚。這條由 L4A 的 non_neutral_ratio 量。
      實測這批的時長中位數 283 秒、167 支超過 10 分鐘，有機會。
   B. **拿清單當「人的名冊」，再去抓那個人的其他作品** —— 抖音有
      `fetch_user_post_videos(sec_user_id)`、B站有 `/x/space/wbi/arc/search?mid=`。
      這條原本被當成加分項，現在是必要條件。

   這支腳本因此變成「產出人的名冊」，而不是「排出誰的影片多」。

⚠️ 作者 ID 只是「同一個人」的**代理**，不是保證。B站的剪輯號、影視解說號
   裡出現的臉常常不是 UP 主本人。最終還是要靠 L4 的臉部特徵確認。
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sourcing.common import paths, store  # noqa: E402


def build(rows: list) -> list:
    """把影片列表捲成作者列表，依影片數由多到少排序。"""
    groups = defaultdict(list)
    for r in rows:
        # 沒有 author_id 的（抓失敗或平台沒給）另外歸一類，不要靜默丟掉 ——
        # 數量異常時要看得出來是「沒抓到作者」而不是「作者只有一支片」
        key = (r["platform"], r.get("author_id") or "")
        groups[key].append(r)

    authors = []
    for (platform, author_id), vids in groups.items():
        durations = [v["duration_s"] for v in vids if v.get("duration_s")]
        authors.append({
            "author_key": f"{platform}:{author_id}" if author_id else f"{platform}:(未知)",
            "platform": platform,
            "author_id": author_id,
            # 同一個 ID 偶爾會有不同的顯示名（改名），取出現最多次的那個
            "author_name": max(
                {v["author_name"] for v in vids},
                key=lambda n: sum(1 for v in vids if v["author_name"] == n),
            ) if vids else "",
            "n_videos": len(vids),
            "total_duration_s": sum(durations),
            "median_duration_s": sorted(durations)[len(durations) // 2] if durations else None,
            "uids": [v["uid"] for v in vids],
            "titles": [v["title"] for v in vids],
        })

    authors.sort(key=lambda a: (-a["n_videos"], -a["total_duration_s"]))
    return authors


def main():
    store.enable_utf8_stdout()
    p = argparse.ArgumentParser(description="L2 作者分組")
    p.add_argument("--min-videos", type=int, default=1,
                   help="只列出影片數 >= N 的作者（預設 1，全列）")
    p.add_argument("--top", type=int, default=20, help="終端機印出前 N 名（預設 20）")
    args = p.parse_args()

    rows = store.read_jsonl(paths.METADATA)
    if not rows:
        raise SystemExit(
            f"找不到 {paths.METADATA}，先跑 L1：\n"
            "  python sourcing/collect/fetch_bilibili.py\n"
            "  python sourcing/collect/fetch_douyin.py"
        )

    authors = build(rows)
    store.write_jsonl(paths.AUTHORS, authors)

    n_by_platform = defaultdict(int)
    for r in rows:
        n_by_platform[r["platform"]] += 1

    print(f"影片 {len(rows)} 支"
          f"（{'、'.join(f'{k} {v}' for k, v in sorted(n_by_platform.items()))}）")
    print(f"作者 {len(authors)} 人\n")

    # 影片數的分布：判斷這批素材到底有沒有礦，看這個比看平均值有用得多
    dist = defaultdict(int)
    for a in authors:
        dist[a["n_videos"]] += 1
    print("每位作者在清單中的影片數分布：")
    for n in sorted(dist, reverse=True)[:12]:
        bar = "█" * min(dist[n], 50)
        print(f"  {n:3d} 支 × {dist[n]:4d} 人  {bar}")

    multi = [a for a in authors if a["n_videos"] >= 3]
    covered = sum(a["n_videos"] for a in multi)
    print(f"\n有 3 支以上的作者：{len(multi)} 人，涵蓋 {covered} 支影片"
          f"（占全部的 {covered / len(rows) * 100:.1f}%）")

    # 一人一支的話，「在清單內湊同一個人的多種情緒」這條路結構上就不存在，
    # 不是資料不夠。這件事必須講清楚，不然看到「作者 857 人」只會覺得數字很多。
    if len(authors) == len(rows):
        print(
            "\n  ⚠️ 每位作者都只有 1 支影片（作者數 = 影片數）。\n"
            "     這份清單是刻意一人只收一支的 —— lip-sync 訓練集要的是說話人多樣性。\n"
            "     所以「在清單內找同一個人的多支影片」走不通，剩下兩條路：\n"
            "       A. 單支影片內找情緒起伏（跑 sourcing/detect/text_timeline.py 看 non_neutral_ratio）\n"
            "       B. 拿這份名冊去抓每個人在自己頻道的其他作品（尚未實作）"
        )

    shown = [a for a in authors if a["n_videos"] >= args.min_videos][:args.top]
    print(f"\n前 {len(shown)} 名候選人物：")
    for i, a in enumerate(shown, 1):
        mins = a["total_duration_s"] // 60 if a["total_duration_s"] else 0
        print(f"  {i:2d}. {a['author_name'][:20]:22s} {a['platform']:9s}"
              f" {a['n_videos']:3d} 支 / 共 {mins:4d} 分")
        for t in a["titles"][:3]:
            print(f"        - {t[:56]}")
        if a["n_videos"] > 3:
            print(f"        … 另外 {a['n_videos'] - 3} 支")

    print(f"\n→ {paths.AUTHORS}")


if __name__ == "__main__":
    main()
