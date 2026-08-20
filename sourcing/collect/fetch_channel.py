# 待GINO改寫
"""L2b：抓某位作者在自己頻道的全部作品。

    python sourcing/collect/fetch_channel.py --mid 20960244            # 四月吨吨_
    python sourcing/collect/fetch_channel.py --mid 20960244 --merge    # 併進 metadata.jsonl

**為什麼需要這一支。**
JoyGen 那份清單是刻意一人只收一支的（857 支 = 857 個不同作者），
所以「同一個人的多種情緒」在清單內結構上就找不到。
再加上實測發現一支影片只有一種情緒基調（已釋懷的只有樂、還在憂愁的只有哀怒），
單支影片內湊四種情緒也不存在。

剩下唯一的路：**把清單當成「人的名冊」**，鎖定臉已經確認可用的人，
再回他自己的頻道找缺的那幾種情緒。

B站的 space API 要 WBI 簽名，比 view 端點麻煩，但公開資料不需要登入。
"""
import argparse
import hashlib
import os
import sys
import time
import urllib.parse
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sourcing.common import paths, store  # noqa: E402

API_NAV = "https://api.bilibili.com/x/web-interface/nav"
API_SEARCH = "https://api.bilibili.com/x/space/wbi/arc/search"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"),
    "Referer": "https://space.bilibili.com/",
}

# WBI 簽名用的固定重排表。這是 B站前端寫死的一組順序，沒有規律可循，
# 照抄即可；抄錯任何一個數字，簽出來的 w_rid 就過不了。
MIXIN_KEY_ENC_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
]


def get_mixin_key(session: requests.Session) -> str:
    """從 nav 端點取兩把 key，重排後截 32 字元。

    nav 未登入時會回 code -101，但 wbi_img 這段照樣有 —— 只取簽名用的 key，
    所以不必理會那個錯誤碼。
    """
    payload = session.get(API_NAV, headers=HEADERS, timeout=15).json()
    wbi = (payload.get("data") or {}).get("wbi_img") or {}
    if not wbi.get("img_url") or not wbi.get("sub_url"):
        raise SystemExit("nav 端點沒有回 wbi_img，拿不到簽名用的 key")
    stem = lambda u: u.rsplit("/", 1)[-1].split(".")[0]   # noqa: E731
    raw = stem(wbi["img_url"]) + stem(wbi["sub_url"])
    return "".join(raw[i] for i in MIXIN_KEY_ENC_TAB)[:32]


def sign(params: dict, mixin_key: str) -> dict:
    """加上時間戳與 w_rid。參數要照鍵名排序後才算 md5，順序錯就過不了。"""
    params = dict(params)
    params["wts"] = int(time.time())
    query = "&".join(
        f"{k}={urllib.parse.quote(str(v), safe='')}"
        for k, v in sorted(params.items())
    )
    params["w_rid"] = hashlib.md5((query + mixin_key).encode()).hexdigest()
    return params


def duration_to_seconds(text: str) -> int:
    """space API 的 length 是 "12:34" 或 "1:02:03" 這種字串，不是秒數。"""
    parts = [int(p) for p in str(text).split(":")]
    out = 0
    for p in parts:
        out = out * 60 + p
    return out


API_SPI = "https://api.bilibili.com/x/frontend/finger/spi"


def make_session() -> requests.Session:
    """建一個帶瀏覽器指紋 cookie 的 session。

    space API 跟 view 端點不一樣：沒有 buvid3 這個 cookie 的話，
    大概第二頁就會回 -352 风控校验失败。真的瀏覽器是進站時自然拿到的，
    這裡用官方的 spi 端點補一個。
    有 BILI_SESSDATA 就一併帶上，風控機率更低。
    """
    session = requests.Session()
    try:
        spi = session.get(API_SPI, headers=HEADERS, timeout=15).json()
        data = spi.get("data") or {}
        if data.get("b_3"):
            session.cookies.set("buvid3", data["b_3"], domain=".bilibili.com")
        if data.get("b_4"):
            session.cookies.set("buvid4", data["b_4"], domain=".bilibili.com")
        print(f"  取得指紋 cookie：buvid3={str(data.get('b_3'))[:18]}…")
    except Exception as exc:
        print(f"  ⚠️ 取指紋 cookie 失敗（{type(exc).__name__}），"
              "可能很快就會被風控擋")

    sessdata = os.environ.get("BILI_SESSDATA")
    if sessdata:
        session.cookies.set("SESSDATA", sessdata, domain=".bilibili.com")
        print("  已帶入 BILI_SESSDATA")
    return session


def fetch_all(mid: int, interval: float, max_pages: int) -> list:
    session = make_session()
    mixin = get_mixin_key(session)
    throttle = store.Throttle(interval)

    videos, page, total = [], 1, None
    while page <= max_pages:
        payload = None
        for attempt in range(1, 4):
            throttle.wait()
            params = sign({
                "mid": mid, "ps": 30, "pn": page, "order": "pubdate",
                "platform": "web", "web_location": 1550101,
            }, mixin)
            resp = session.get(API_SEARCH, params=params,
                               headers=HEADERS, timeout=20)
            try:
                payload = resp.json()
                break
            except ValueError:
                # space API 的風控比 view 端點嚴，被擋時回的是 HTML 而不是
                # 帶錯誤碼的 JSON，所以只能從「解不出 JSON」判斷。
                # 退避重試，順便換一把新的 mixin key（舊的可能過期了）。
                wait = 8 * attempt
                print(f"  第 {page} 頁被擋（回傳不是 JSON），等 {wait}s 後重試 "
                      f"{attempt}/3")
                time.sleep(wait)
                mixin = get_mixin_key(session)
        if payload is None:
            # 抓到多少算多少：這一步的目的是「找幾支可用的素材」，
            # 不是「完整鏡像整個頻道」，半途被擋不該讓已經抓到的白費。
            print(f"  第 {page} 頁重試三次都被擋，就用已經抓到的 {len(videos)} 支")
            break
        if payload.get("code") != 0:
            print(f"  space API 回 code={payload.get('code')}："
                  f"{payload.get('message')}，停在已抓到的 {len(videos)} 支")
            break
        data = payload.get("data") or {}
        vlist = ((data.get("list") or {}).get("vlist")) or []
        total = (data.get("page") or {}).get("count", total)
        if not vlist:
            break
        videos.extend(vlist)
        print(f"  第 {page} 頁：{len(vlist)} 支（累計 {len(videos)}/{total}）")
        if total is not None and len(videos) >= total:
            break
        page += 1
    return videos


def to_metadata(v: dict) -> dict:
    """轉成跟 fetch_bilibili.py 一樣的統一欄位，下游才不用分兩套處理。

    ⚠️ space API 給的欄位比 view 端點少：沒有 tag、沒有 cid、desc 也只有摘要。
       真的要細節就對挑中的 bvid 再跑一次 fetch_bilibili.py。
    """
    bvid = v.get("bvid")
    return {
        "uid": f"bili:{bvid}",
        "platform": "bilibili",
        "vid": bvid,
        "url": f"https://www.bilibili.com/video/{bvid}",
        "title": v.get("title") or "",
        "desc": v.get("description") or "",
        "author_name": v.get("author") or "",
        "author_id": str(v.get("mid") or ""),
        "duration_s": duration_to_seconds(v.get("length") or "0"),
        "cover_url": (v.get("pic") or "").replace("http://", "https://"),
        "publish_ts": v.get("created"),
        "category": v.get("typeid"),
        "tags": [],
        "stats": {"view": v.get("play"), "like": None,
                  "reply": v.get("comment"), "favorite": None},
        "extra": {"from": "space_api"},
    }


def main():
    store.enable_utf8_stdout()
    p = argparse.ArgumentParser(description="L2b 抓某作者的全部作品")
    p.add_argument("--mid", type=int, required=True, help="B站使用者 mid")
    p.add_argument("--interval", type=float, default=1.5)
    p.add_argument("--max-pages", type=int, default=20, help="最多翻幾頁（一頁 30 支）")
    p.add_argument("--merge", action="store_true",
                   help="把結果併進 metadata.jsonl，後續 L3/L4 才吃得到")
    p.add_argument("--min-duration", type=int, default=180,
                   help="只保留這個秒數以上的（預設 180）")
    args = p.parse_args()

    print(f"抓 mid={args.mid} 的作品列表")
    raw = fetch_all(args.mid, args.interval, args.max_pages)
    if not raw:
        raise SystemExit("一支都沒抓到，確認 mid 是否正確")

    rows = [to_metadata(v) for v in raw]
    rows = [r for r in rows if r["duration_s"] >= args.min_duration]
    rows.sort(key=lambda r: -(r["publish_ts"] or 0))

    out = paths.CHANNEL_DIR / f"{args.mid}.jsonl"
    store.write_jsonl(out, rows)

    name = rows[0]["author_name"] if rows else ""
    print(f"\n{name}：{len(raw)} 支作品，其中 {len(rows)} 支超過 "
          f"{args.min_duration} 秒\n")
    for r in rows[:30]:
        print(f"  {r['duration_s'] // 60:3d}分  {r['uid']:22s} {r['title'][:44]}")
    if len(rows) > 30:
        print(f"  … 另外 {len(rows) - 30} 支")

    if args.merge:
        total = store.merge_metadata(rows)
        print(f"\n已併入 metadata.jsonl（現有 {total} 筆），"
              f"可以直接對這些 uid 跑 sourcing/detect/face_timeline.py")
    else:
        print(f"\n→ {out}\n（加 --merge 才會併進 metadata.jsonl 給下游用）")


if __name__ == "__main__":
    main()
