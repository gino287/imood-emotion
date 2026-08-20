# 待GINO改寫
"""L3 附帶：下載封面並拼成縮圖牆，供肉眼快篩。

    python sourcing/collect/covers.py --download          # 抓封面（1188 張約數十 MB）
    python sourcing/collect/covers.py --sheet             # 拼成縮圖牆
    python sourcing/collect/covers.py --download --sheet --verdict keep

這是整條漏斗裡 CP 值最高的一步。封面圖是 metadata 階段唯一能看到「畫面」
的東西，而且非常便宜 —— 一千多張縮圖只有幾十 MB，一頁排 100 張，
肉眼掃 30 秒就能刷掉一半「根本沒有人臉／是動畫／是遊戲畫面」的影片。
這件事關鍵字做不到，但眼睛一秒就看得出來。

刻意跟 coarse_filter.py 分開：那支不連網、改規則重跑零成本；
這支要下載檔案，混在一起的話每次調關鍵字都得重抓一輪圖。

進階做法（還沒做）：直接對封面跑人臉偵測，沒有臉的自動淘汰。
JoyGen 那邊本來就有 insightface 那套環境，可以借過來用。
"""
import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sourcing.common import paths, store  # noqa: E402

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"),
    "Referer": "https://www.bilibili.com",
}

THUMB_W, THUMB_H = 320, 180      # 16:9，一頁 10×10 張時整張圖是 3200×1800
COLS, ROWS = 10, 10


def cover_path(uid: str) -> Path:
    return paths.COVER_DIR / f"{uid.replace(':', '_')}.jpg"


def download(rows: list, interval: float) -> None:
    paths.COVER_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    throttle = store.Throttle(interval)
    ok = skip = failed = 0

    for i, r in enumerate(rows, 1):
        dst = cover_path(r["uid"])
        if dst.exists():
            skip += 1
            continue
        url = r.get("cover_url")
        if not url:
            failed += 1
            continue
        throttle.wait()
        try:
            resp = session.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            dst.write_bytes(resp.content)
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"  [{i}/{len(rows)}] {r['uid']} 下載失敗：{type(exc).__name__}")
        if i % 100 == 0:
            print(f"  ... {i}/{len(rows)}（新增 {ok} / 已有 {skip} / 失敗 {failed}）")

    print(f"封面下載完成：新增 {ok}、已存在 {skip}、失敗 {failed}")
    print(f"→ {paths.COVER_DIR}")


def make_sheets(rows: list) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise SystemExit(
            "沒有安裝 Pillow：pip install -r sourcing/requirements.txt"
        )

    have = [r for r in rows if cover_path(r["uid"]).exists()]
    if not have:
        raise SystemExit("還沒有任何封面，先跑 --download")

    paths.SHEET_DIR.mkdir(parents=True, exist_ok=True)
    per_sheet = COLS * ROWS
    n_sheets = (len(have) + per_sheet - 1) // per_sheet

    for s in range(n_sheets):
        chunk = have[s * per_sheet:(s + 1) * per_sheet]
        sheet = Image.new("RGB", (COLS * THUMB_W, ROWS * THUMB_H), (20, 20, 20))
        draw = ImageDraw.Draw(sheet)

        for idx, r in enumerate(chunk):
            try:
                img = Image.open(cover_path(r["uid"])).convert("RGB")
            except Exception:
                continue
            img = img.resize((THUMB_W, THUMB_H))
            x = (idx % COLS) * THUMB_W
            y = (idx // COLS) * THUMB_H
            sheet.paste(img, (x, y))
            # 標上格號：看到「第 3 排第 7 個有臉」時，要能對回是哪支影片。
            # 對照表印在下面的 index 檔裡。
            draw.text((x + 4, y + 4), str(s * per_sheet + idx), fill=(255, 255, 0))

        out = paths.SHEET_DIR / f"sheet_{s:03d}.jpg"
        sheet.save(out, quality=85)
        print(f"→ {out}（{len(chunk)} 張）")

    # 格號 → 影片的對照表，沒有這個縮圖牆看完也不知道要挑哪支
    index = paths.SHEET_DIR / "index.tsv"
    with index.open("w", encoding="utf-8") as fh:
        fh.write("格號\tuid\t標題\t網址\n")
        for i, r in enumerate(have):
            fh.write(f"{i}\t{r['uid']}\t{r.get('title', '')}\t{r['url']}\n")
    print(f"→ {index}")


def main():
    store.enable_utf8_stdout()
    p = argparse.ArgumentParser(description="L3 封面下載與縮圖牆")
    p.add_argument("--download", action="store_true", help="下載封面")
    p.add_argument("--sheet", action="store_true", help="拼縮圖牆")
    p.add_argument("--verdict", choices=["keep", "maybe", "drop"],
                   help="只處理 L3 判定為此檔的影片（預設全部）")
    p.add_argument("--from-shortlist", action="store_true",
                   help="改吃 shortlist.jsonl 並沿用它的排序 —— "
                        "縮圖牆第一頁就是最有機會的候選，不必整批看完")
    p.add_argument("--limit", type=int, help="最多處理幾支")
    p.add_argument("--interval", type=float, default=0.2,
                   help="下載間隔秒數（圖片是靜態資源，可以比 API 快）")
    args = p.parse_args()

    if not (args.download or args.sheet):
        raise SystemExit("要指定 --download 或 --sheet（可以一起）")

    if args.from_shortlist:
        rows = store.read_jsonl(paths.SHORTLIST)
        if not rows:
            raise SystemExit("找不到 shortlist.jsonl，先跑 sourcing/deliver/shortlist.py")
        print(f"沿用 shortlist 排序的 {len(rows)} 支")
    else:
        rows = store.read_jsonl(paths.METADATA)
        if not rows:
            raise SystemExit(f"找不到 {paths.METADATA}，先跑 L1")

    if args.verdict:
        filtered = {r["uid"]: r["verdict"] for r in store.read_jsonl(paths.FILTERED)}
        if not filtered:
            raise SystemExit(f"找不到 {paths.FILTERED}，先跑 sourcing/collect/coarse_filter.py")
        rows = [r for r in rows if filtered.get(r["uid"]) == args.verdict]
        print(f"只處理 verdict={args.verdict} 的 {len(rows)} 支")

    if args.limit:
        rows = rows[:args.limit]

    if args.download:
        download(rows, args.interval)
    if args.sheet:
        make_sheets(rows)


if __name__ == "__main__":
    main()
