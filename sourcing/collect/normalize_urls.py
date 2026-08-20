# 待GINO改寫
"""L0：把兩份裸網址清單整理成一份乾淨的影片清單。

  python sourcing/collect/normalize_urls.py
  python sourcing/collect/normalize_urls.py --dry-run          # 只印統計不寫檔

原始清單有三個坑，這支腳本就是為了處理它們：

1. **bili_urls.txt 裡混了抖音網址**。1012 行裡有 10 行其實是抖音，
   而且影片 ID 不在路徑而在 query 的 modal_id 參數
   （`douyin.com/user/self?modal_id=735...`、`douyin.com/?modal_id=...`）。
   只用 `grep BV` 會安靜地漏掉這 9 支（其中 1 行是純搜尋頁、沒有 ID）。

2. **兩份都有重複行**。B站 1002 行含 BV 但只有 1000 支不重複，
   抖音 181 行只有 179 支不重複。

3. **網址帶一堆追蹤參數**（`spm_id_from`、`vd_source`），
   同一支影片在不同行可能長得不一樣，不清掉就去不了重。

正規化後的總數應該是 1188 支：B站 1000 + 抖音 188（179 + 混進來的 9）。
這個數字寫成斷言，對不上就是解析漏了東西。
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sourcing.common import store  # noqa: E402
from sourcing.common.paths import DEFAULT_URL_FILES, SOURCES  # noqa: E402

# B站 BV 號固定是 BV + 10 碼英數
RE_BV = re.compile(r"BV[0-9A-Za-z]{10}")
# 抖音影片 ID 的兩種寫法：路徑式與 modal_id 參數式
RE_DY_PATH = re.compile(r"douyin\.com/video/(\d+)")
RE_DY_MODAL = re.compile(r"[?&]modal_id=(\d+)")

# 正規化後的標準網址，之後所有階段都用這個，不用原始那串帶追蹤參數的
CANONICAL = {
    "bilibili": "https://www.bilibili.com/video/{vid}",
    "douyin": "https://www.douyin.com/video/{vid}",
}

EXPECTED_TOTAL = 1188
EXPECTED_BY_PLATFORM = {"bilibili": 1000, "douyin": 188}


def extract(line: str):
    """從一行網址抽出 (platform, vid)，抽不出來回傳 None。

    先判抖音再判 B站：混進 bili_urls.txt 的那幾行網域是 douyin.com，
    順序反過來不會有影響，但先處理特例讀起來比較清楚。
    """
    m = RE_DY_PATH.search(line)
    if m:
        return "douyin", m.group(1)

    if "douyin.com" in line:
        m = RE_DY_MODAL.search(line)
        if m:
            return "douyin", m.group(1)
        return None      # 純搜尋頁／個人頁，沒有指向特定影片

    m = RE_BV.search(line)
    if m:
        return "bilibili", m.group(0)

    return None


def main():
    store.enable_utf8_stdout()
    p = argparse.ArgumentParser(description="L0 清單正規化")
    p.add_argument("files", nargs="*", type=Path, default=None,
                   help="網址清單（預設 tmp/bili_urls.txt 與 tmp/dy_urls.txt）")
    p.add_argument("--out", type=Path, default=SOURCES)
    p.add_argument("--dry-run", action="store_true", help="只印統計，不寫檔")
    p.add_argument("--no-assert", action="store_true",
                   help="不檢查總數是否為預期的 1188（換清單時用）")
    args = p.parse_args()

    files = args.files or DEFAULT_URL_FILES
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        raise SystemExit(f"找不到清單檔：{', '.join(str(m) for m in missing)}")

    # 用 dict 而不是 set：要記住每支影片在哪些檔案出現過幾次，
    # 「有 10 行抖音混在 B站清單裡」這種事才看得出來
    records = {}
    unparsed = []
    per_file = Counter()

    for f in files:
        f = Path(f)
        for lineno, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            per_file[f.name] += 1
            got = extract(line)
            if not got:
                unparsed.append((f.name, lineno, line))
                continue
            platform, vid = got
            uid = f"{'bili' if platform == 'bilibili' else 'dy'}:{vid}"
            rec = records.setdefault(uid, {
                "uid": uid,
                "platform": platform,
                "vid": vid,
                "url": CANONICAL[platform].format(vid=vid),
                "seen_in": [],
            })
            rec["seen_in"].append(f"{f.name}:{lineno}")

    by_platform = Counter(r["platform"] for r in records.values())
    dups = {u: r for u, r in records.items() if len(r["seen_in"]) > 1}
    # 出現在「不是自己那份」清單裡的抖音影片
    crossed = [r for r in records.values()
               if r["platform"] == "douyin"
               and any(s.startswith("bili_urls") for s in r["seen_in"])]

    print("讀入行數：")
    for name, n in per_file.items():
        print(f"  {name:20s} {n:5d} 行")
    print(f"\n解析出不重複影片 {len(records)} 支")
    for plat, n in sorted(by_platform.items()):
        print(f"  {plat:10s} {n:5d}")
    print(f"\n重複列（同一支影片出現多次）：{len(dups)} 支")
    for u, r in list(dups.items())[:5]:
        print(f"  {u}  ← {', '.join(r['seen_in'])}")
    print(f"\n混在 bili_urls.txt 裡的抖音影片：{len(crossed)} 支")
    for r in crossed:
        print(f"  {r['uid']}  ← {r['seen_in'][0]}")
    print(f"\n解析不出影片 ID 的行：{len(unparsed)} 行")
    for name, lineno, line in unparsed:
        print(f"  {name}:{lineno}  {line[:90]}")

    if not args.no_assert:
        problems = []
        if len(records) != EXPECTED_TOTAL:
            problems.append(f"總數 {len(records)} ≠ 預期 {EXPECTED_TOTAL}")
        for plat, want in EXPECTED_BY_PLATFORM.items():
            if by_platform[plat] != want:
                problems.append(f"{plat} {by_platform[plat]} ≠ 預期 {want}")
        if problems:
            raise SystemExit(
                "\n數量與預期不符，解析可能漏了東西：\n  " + "\n  ".join(problems)
                + "\n（若是換了新的清單檔，加 --no-assert 略過這項檢查）"
            )
        # 這裡刻意不用 ✓ 之類的符號：Windows 主機的終端機是 cp950，
        # 打不出 U+2713 會直接 UnicodeEncodeError 崩掉（容器內是 UTF-8 才沒事）
        print(f"\n[OK] 數量檢查通過：{EXPECTED_TOTAL} 支"
              f"（bilibili {EXPECTED_BY_PLATFORM['bilibili']}"
              f" / douyin {EXPECTED_BY_PLATFORM['douyin']}）")

    if args.dry_run:
        print("\n--dry-run，未寫檔")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in records.values():
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
