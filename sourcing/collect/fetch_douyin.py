    # 待GINO改寫
"""L1（抖音）：用 f2 抓影片 metadata。

    python sourcing/collect/fetch_douyin.py --limit 3        # 小樣本先驗欄位
    python sourcing/collect/fetch_douyin.py                  # 全量（會續跑）
    python sourcing/collect/fetch_douyin.py --rebuild        # 不連網，只從 raw 重抽欄位

前置：pip install -r sourcing/requirements.txt

抖音跟 B站不同，公開影片也多半需要 cookie 才拿得到完整資料。取得方式：
  1. 瀏覽器登入抖音後，開發者工具複製 document.cookie
  2. 或用 f2 自己的工具：`f2 dy -k`（會從本機瀏覽器讀）
拿到之後放進環境變數 DOUYIN_COOKIE（別寫進程式碼，.env 已經在 .gitignore）。

沒有 cookie 也可以先跑跑看 —— 這支腳本會把成功率印出來，
量到實際數字再決定要不要補 cookie，不用先猜。

⚠️ 抖音的 duration 單位是**毫秒**，B站是**秒**。這裡統一換算成秒再存，
   下游做時長粗篩時才不會把 3 分鐘的片子當成 3 毫秒。
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sourcing.common import paths, store  # noqa: E402

PLATFORM = "douyin"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"),
    "Referer": "https://www.douyin.com/",
}


def build_kwargs(cookie: str) -> dict:
    """f2 的 handler 吃一整包設定，缺欄位會在執行時才炸，所以一次給齊。

    naming / path 是下載用的，這裡只抓 metadata 不下載，但 handler 初始化
    仍會讀，給預設值即可。
    """
    return {
        "headers": HEADERS,
        "cookie": cookie or "",
        "proxies": {"http://": None, "https://": None},
        "timeout": 15,
        "max_retries": 3,
        "max_connections": 4,
        "max_tasks": 4,
        "max_counts": None,
        "page_counts": 20,
        "naming": "{create}_{desc}_{aweme_id}",
        "path": str(paths.OUT_DIR / "_f2_downloads"),
        "mode": "one",
    }


def to_seconds(duration) -> int:
    """抖音給的是毫秒。偶爾會拿到已經是秒的數值（不同版本欄位不一致），
    用「大於 1000 就當毫秒」來判斷 —— 沒有哪支影片只有 1 秒卻標成 1000。"""
    if duration is None:
        return None
    try:
        d = int(duration)
    except (TypeError, ValueError):
        return None
    return d // 1000 if d > 1000 else d


def extract(aweme_id: str, raw: dict) -> dict:
    """從 f2 的 _to_dict() 抽出與 B站對齊的統一欄位。

    f2 的欄位名跟 B站完全不同（desc/nickname/sec_user_id vs title/owner.name/mid），
    在這裡就對齊，下游 group by 作者、粗篩時長才不用先分平台再各寫一套。
    """
    hashtags = raw.get("hashtag_names") or []
    if isinstance(hashtags, str):
        hashtags = [hashtags]

    return {
        "uid": f"dy:{aweme_id}",
        "platform": PLATFORM,
        "vid": aweme_id,
        "url": f"https://www.douyin.com/video/{aweme_id}",
        # 抖音沒有標題欄位，desc 就是我們一般看到的那行文案
        "title": raw.get("desc") or "",
        "desc": raw.get("caption") or "",
        "author_name": raw.get("nickname") or "",
        # 用 sec_user_id 而不是 uid：抖音的 uid 有些介面拿不到，
        # sec_user_id 才是各處通用、也是之後抓該作者全部作品要用的那個
        "author_id": str(raw.get("sec_user_id") or raw.get("uid") or ""),
        "duration_s": to_seconds(raw.get("duration")),
        "cover_url": raw.get("cover") or "",
        "publish_ts": raw.get("create_time"),      # f2 已轉成字串格式
        "category": "",                            # 抖音沒有分區概念
        "tags": hashtags,                          # 常直接寫 #情感 #演技，比文案好用
        "stats": {
            "view": None,                          # 抖音不公開播放數
            "like": raw.get("digg_count"),
            "reply": raw.get("comment_count"),
            "favorite": raw.get("collect_count"),
        },
        "extra": {
            "aweme_type": raw.get("aweme_type"),
            "media_type": raw.get("media_type"),
            "animated_cover": raw.get("animated_cover") or "",
            "music_title": raw.get("music_title") or "",
        },
    }


def rebuild() -> None:
    rows = [extract(vid, raw) for vid, raw in store.iter_raw(PLATFORM)]
    total = store.merge_metadata(rows)
    print(f"從 {len(rows)} 份 raw 重抽欄位，metadata.jsonl 現有 {total} 筆")


async def fetch_all(todo: list, cookie: str, interval: float) -> tuple:
    try:
        from f2.apps.douyin.handler import DouyinHandler
    except ImportError:
        raise SystemExit(
            "沒有安裝 f2。\n"
            "  pip install -r sourcing/requirements.txt\n"
            "（B站那份不需要 f2，可以先跑 sourcing/collect/fetch_bilibili.py）"
        )

    handler = DouyinHandler(build_kwargs(cookie))
    throttle = store.Throttle(interval)
    ok = failed = 0

    for i, s in enumerate(todo, 1):
        aweme_id = s["vid"]
        throttle.wait()
        try:
            video = await handler.fetch_one_video(aweme_id=aweme_id)
            raw = video._to_dict()
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            store.append_error(PLATFORM, aweme_id, reason)
            failed += 1
            print(f"  [{i}/{len(todo)}] {aweme_id} 失敗：{reason[:120]}")
            continue

        # 拿到空殼也算失敗：沒有 cookie 時 f2 常常回一個沒有 desc/nickname 的
        # 空物件而不是拋例外，不擋掉的話會存進一堆看起來成功的空資料
        if not raw or not (raw.get("desc") or raw.get("nickname")):
            store.append_error(PLATFORM, aweme_id, "回傳空資料（多半是缺 cookie 或影片已下架）")
            failed += 1
            print(f"  [{i}/{len(todo)}] {aweme_id} 回傳空資料")
            continue

        store.save_raw(PLATFORM, aweme_id, raw)
        ok += 1

        if i % 25 == 0:
            print(f"  ... {i}/{len(todo)}（成功 {ok} / 失敗 {failed}）")

    return ok, failed


def main():
    store.enable_utf8_stdout()
    p = argparse.ArgumentParser(description="L1 抖音 metadata 抓取")
    p.add_argument("--limit", type=int, help="只抓前 N 支（小樣本驗證用）")
    p.add_argument("--interval", type=float, default=1.0, help="請求間隔秒數（預設 1.0）")
    p.add_argument("--rebuild", action="store_true",
                   help="不連網，只從已存的 raw 重新抽取欄位")
    p.add_argument("--refetch", action="store_true", help="忽略已存的 raw，全部重抓")
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

    print(f"抖音共 {len(sources)} 支，本次要抓 {len(todo)} 支")
    if not todo:
        print("沒有待抓的，直接重抽欄位")
        rebuild()
        return

    cookie = os.environ.get("DOUYIN_COOKIE", "")
    if cookie:
        print(f"已帶入 DOUYIN_COOKIE（長度 {len(cookie)}）")
    else:
        print("未設定 DOUYIN_COOKIE，以訪客模式抓 —— 成功率可能很低，"
              "先看這次的數字再決定要不要補")

    ok, failed = asyncio.run(fetch_all(todo, cookie, args.interval))

    rate = ok / (ok + failed) * 100 if (ok + failed) else 0
    print(f"\n抓取完成：成功 {ok}、失敗 {failed}（成功率 {rate:.0f}%）")
    if failed:
        print(f"失敗明細見 {paths.FETCH_ERRORS}；重跑同一行指令會只補沒抓到的")
    rebuild()


if __name__ == "__main__":
    main()
