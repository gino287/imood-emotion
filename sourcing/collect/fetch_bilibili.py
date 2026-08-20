# 待GINO改寫
"""L1（B站）：用官方公開 API 抓影片 metadata。

    python sourcing/collect/fetch_bilibili.py --limit 3        # 小樣本先驗欄位
    python sourcing/collect/fetch_bilibili.py                  # 全量（會續跑）
    python sourcing/collect/fetch_bilibili.py --rebuild        # 不連網，只從 raw 重抽欄位

為什麼不用 f2：f2 目前只支援抖音／TikTok／Twitter／微博，B站排在 0.0.1.8
還沒實作。也沒用 yt-dlp —— 官方 API 一個請求就直接給 owner.mid（之後
group by 作者的主鍵）跟 cid（要取字幕時會用到），yt-dlp 較重而且 B站
有已知的 412 問題。yt-dlp 留到真的要下載媒體時再用。

端點 `/x/web-interface/view` 公開影片免登入，一支一個請求（沒有批次端點）。
帶 SESSDATA 可以降低風控機率、也才拿得到部分受限影片，但不是必要。

預設會多打一次 `/x/tag/archive/tags` 拿 UP 主自訂標籤。多花一倍請求數是
值得的：實測 view 端點在訪客模式下 `tname`（分區名）回空字串，只剩數字
`tid`；而 tag 直接就是「经验分享 / 职场 / 职业规划」這種東西，是 L3 判斷
片型最好的訊號。不想多花時間可以加 --no-tags。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sourcing.common import paths, store  # noqa: E402

API = "https://api.bilibili.com/x/web-interface/view"
API_TAGS = "https://api.bilibili.com/x/tag/archive/tags"
PLATFORM = "bilibili"

# 不帶 Referer 的話 B站會直接擋掉，這不是反爬蟲的猜測，是它的既定行為
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"),
    "Referer": "https://www.bilibili.com",
}

# API 回傳的 code，非 0 都代表這支拿不到。分開列出來是因為要採取的行動不同：
# 影片被刪／私有是永久性的，重試多少次都一樣；風控是暫時的，等一下再來就好。
PERMANENT = {
    -404: "影片不存在或已刪除",
    62002: "稿件不可見（私有或已下架）",
    62004: "稿件審核中",
    62012: "稿件僅 UP 主自己可見",
    -403: "存取被拒（可能需要登入）",
}
RATE_LIMITED = {-352: "觸發風控", -509: "請求過於頻繁"}


def fetch_one(session: requests.Session, bvid: str, timeout: float = 15.0) -> dict:
    """回傳完整的 API 回應（含 code）。網路層失敗才會拋例外。"""
    resp = session.get(API, params={"bvid": bvid}, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_tags(session: requests.Session, bvid: str, timeout: float = 15.0) -> list:
    """拿 UP 主自訂標籤。失敗就回空清單 —— 這是加分項不是必要項，
    為了它讓整支影片算抓失敗並不划算。"""
    try:
        resp = session.get(API_TAGS, params={"bvid": bvid},
                           headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return []
    if payload.get("code") != 0:
        return []
    return [t.get("tag_name") for t in (payload.get("data") or []) if t.get("tag_name")]


def extract(bvid: str, payload: dict) -> dict:
    """從 raw 回應抽出各平台共用的統一欄位。

    欄位取名刻意跟抖音那邊對齊（author_id / duration_s / cover_url ...），
    下游 group by 作者、粗篩時長才不用先分平台再各寫一套。
    """
    d = payload["view"]["data"]
    tags = payload.get("tags") or []
    owner = d.get("owner") or {}
    stat = d.get("stat") or {}
    subtitle = (d.get("subtitle") or {}).get("list") or []

    # 訪客模式拿到的 tname 常常是空字串，只剩數字 tid。
    # 空的就退回 tid 字串，讓下游至少分得出「這兩支不同區」。
    category = d.get("tname") or d.get("tname_v2") or ""
    if not category and d.get("tid"):
        category = f"tid:{d['tid']}"

    return {
        "uid": f"bili:{bvid}",
        "platform": PLATFORM,
        "vid": bvid,
        "url": f"https://www.bilibili.com/video/{bvid}",
        "title": d.get("title") or "",
        "desc": d.get("desc") or "",
        "author_name": owner.get("name") or "",
        "author_id": str(owner.get("mid") or ""),
        "duration_s": d.get("duration"),        # B站本來就是秒，不必換算
        "cover_url": d.get("pic") or "",
        "publish_ts": d.get("pubdate"),
        "category": category,                   # 分區，粗篩時用來排除遊戲／音樂等
        "tags": tags,                            # UP 主自訂標籤，L3 判斷片型的主力訊號
        "stats": {
            "view": stat.get("view"),
            "like": stat.get("like"),
            "reply": stat.get("reply"),
            "favorite": stat.get("favorite"),
        },
        # 平台特有、之後可能用得上的：cid 取字幕要用，pages 判斷是不是多 P，
        # dynamic 是 UP 主發布時寫的動態文字，等於多一段免費的描述
        "extra": {
            "aid": d.get("aid"),
            "cid": d.get("cid"),
            "tid": d.get("tid"),
            "pages": len(d.get("pages") or []),
            "has_subtitle": len(subtitle) > 0,
            "dynamic": d.get("dynamic") or "",
        },
    }


def rebuild() -> None:
    rows = [extract(vid, payload) for vid, payload in store.iter_raw(PLATFORM)]
    total = store.merge_metadata(rows)
    print(f"從 {len(rows)} 份 raw 重抽欄位，metadata.jsonl 現有 {total} 筆")


def main():
    store.enable_utf8_stdout()
    p = argparse.ArgumentParser(description="L1 B站 metadata 抓取")
    p.add_argument("--limit", type=int, help="只抓前 N 支（小樣本驗證用）")
    p.add_argument("--interval", type=float, default=1.0, help="請求間隔秒數（預設 1.0）")
    p.add_argument("--retries", type=int, default=3, help="遇到風控時的重試次數")
    p.add_argument("--rebuild", action="store_true",
                   help="不連網，只從已存的 raw 重新抽取欄位")
    p.add_argument("--refetch", action="store_true", help="忽略已存的 raw，全部重抓")
    p.add_argument("--no-tags", action="store_true",
                   help="不抓 UP 主標籤（少一半請求，但 L3 會失去主要的片型訊號）")
    args = p.parse_args()

    if args.rebuild:
        rebuild()
        return

    sources = store.load_sources(PLATFORM)
    todo = sources if args.refetch else [
        s for s in sources if not store.has_raw(PLATFORM, s["vid"])
    ]
    if args.limit:
        todo = todo[:args.limit]

    print(f"B站共 {len(sources)} 支，本次要抓 {len(todo)} 支"
          f"（已抓過 {len(sources) - len([s for s in sources if not store.has_raw(PLATFORM, s['vid'])])} 支）")
    if not todo:
        print("沒有待抓的，直接重抽欄位")
        rebuild()
        return

    session = requests.Session()
    sessdata = os.environ.get("BILI_SESSDATA")
    if sessdata:
        session.cookies.set("SESSDATA", sessdata, domain=".bilibili.com")
        print("已帶入 BILI_SESSDATA")
    else:
        print("未設定 BILI_SESSDATA，以訪客模式抓（公開影片本來就不需要登入）")

    throttle = store.Throttle(args.interval)
    ok = failed = 0

    for i, s in enumerate(todo, 1):
        bvid = s["vid"]
        for attempt in range(1, args.retries + 1):
            throttle.wait()
            try:
                payload = fetch_one(session, bvid)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                if attempt < args.retries:
                    time.sleep(2 ** attempt)     # 網路層失敗退避後再試
                    continue
                store.append_error(PLATFORM, bvid, reason)
                failed += 1
                print(f"  [{i}/{len(todo)}] {bvid} 網路失敗：{reason}")
                break

            code = payload.get("code")
            if code == 0:
                tags = []
                if not args.no_tags:
                    throttle.wait()
                    tags = fetch_tags(session, bvid)
                # raw 存成 {"view":…, "tags":…} 兩段，之後想加第三個端點
                # 也不用動已經抓好的檔
                store.save_raw(PLATFORM, bvid, {"view": payload, "tags": tags})
                ok += 1
                break
            if code in RATE_LIMITED and attempt < args.retries:
                wait = 10 * attempt
                print(f"  [{i}/{len(todo)}] {bvid} {RATE_LIMITED[code]}（code={code}），"
                      f"等 {wait}s 後重試")
                time.sleep(wait)
                continue

            reason = PERMANENT.get(code) or payload.get("message") or f"code={code}"
            store.append_error(PLATFORM, bvid, f"code={code} {reason}")
            failed += 1
            print(f"  [{i}/{len(todo)}] {bvid} 拿不到：{reason}")
            break

        if i % 50 == 0:
            print(f"  ... {i}/{len(todo)}（成功 {ok} / 失敗 {failed}）")

    print(f"\n抓取完成：成功 {ok}、失敗 {failed}")
    if failed:
        print(f"失敗明細見 {paths.FETCH_ERRORS}；重跑同一行指令會只補沒抓到的")
    rebuild()


if __name__ == "__main__":
    main()
