# 待GINO改寫
"""L1 兩支抓取腳本共用的存檔與續跑邏輯。

存檔刻意分兩層：

  raw/{platform}/{vid}.json   API 原始回應，**原封不動、不再改**
  metadata.jsonl              抽取後的統一欄位，一支一列

分兩層的理由是抓取很貴（1188 支、節流之下要跑二十幾分鐘）而抽取很便宜。
把兩者混在一起的話，之後想多留一個欄位就得整批重抓一次。留著 raw，
改抽取邏輯只要重跑 rebuild，不必再碰網路。

續跑也是靠 raw 判斷：raw 檔存在就跳過。中途斷線、被風控擋、或是想
分幾天慢慢抓，都直接重跑同一行指令即可。
"""
import json
import sys
import time
from pathlib import Path

from sourcing.common import paths


def enable_utf8_stdout() -> None:
    """讓終端機印得出簡體字。每支 CLI 的 main() 第一行都要呼叫。

    這批素材的標題與作者名幾乎都是簡體，而 Windows 主機的終端機是 cp950
    （繁體中文碼頁），碰到「点」「洁」這種字會直接 UnicodeEncodeError 整支崩掉——
    不是印成亂碼，是崩掉。容器裡因為 dockerfile 設了 PYTHONIOENCODING=utf-8
    所以不會，但這些腳本兩邊都會跑。

    errors="replace" 是保險：真的遇到連 UTF-8 也處理不了的東西時印成問號，
    不要讓一個字元中斷整批處理。
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def read_jsonl(path: Path) -> list:
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_sources(platform: str = None) -> list:
    rows = read_jsonl(paths.SOURCES)
    if not rows:
        raise SystemExit(
            f"找不到 {paths.SOURCES}，先跑：python sourcing/collect/normalize_urls.py"
        )
    if platform:
        rows = [r for r in rows if r["platform"] == platform]
    return rows


def has_raw(platform: str, vid: str) -> bool:
    return paths.raw_path(platform, vid).exists()


def save_raw(platform: str, vid: str, payload: dict) -> None:
    p = paths.raw_path(platform, vid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_raw(platform: str, vid: str) -> dict:
    return json.loads(paths.raw_path(platform, vid).read_text(encoding="utf-8"))


def iter_raw(platform: str):
    """走訪某平台已經抓下來的 raw 檔，給 rebuild 用。"""
    d = paths.RAW_DIR / platform
    if not d.exists():
        return
    for p in sorted(d.glob("*.json")):
        yield p.stem, json.loads(p.read_text(encoding="utf-8"))


def append_error(platform: str, vid: str, reason: str) -> None:
    """抓失敗的紀錄另外存一份，才知道哪些要重試、以及為什麼失敗。

    附加寫入而不是覆蓋：同一支影片重試多次的話，歷次失敗原因都留著，
    「一直是同一種錯」跟「每次錯的都不一樣」要採取的行動不同。
    """
    paths.FETCH_ERRORS.parent.mkdir(parents=True, exist_ok=True)
    with paths.FETCH_ERRORS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "platform": platform,
            "vid": vid,
            "reason": reason,
        }, ensure_ascii=False) + "\n")


def merge_metadata(new_rows: list) -> int:
    """把新抽出來的欄位併回 metadata.jsonl，以 uid 為主鍵覆蓋同一支的舊資料。

    不是直接附加：重跑時同一支會再抽一次，附加的話會出現重複列，
    下游 group by 作者的數字就會膨脹。
    """
    existing = {r["uid"]: r for r in read_jsonl(paths.METADATA)}
    for r in new_rows:
        existing[r["uid"]] = r
    write_jsonl(paths.METADATA, existing.values())
    return len(existing)


class Throttle:
    """兩次請求之間至少間隔 interval 秒。

    B站與抖音都有風控，抓太快會被擋（B站會回 -352 或 412）。
    這裡用「距離上次請求多久」而不是每次固定 sleep，
    因為請求本身就要花時間，固定 sleep 會白白慢一倍。
    """

    def __init__(self, interval: float):
        self.interval = interval
        self._last = 0.0

    def wait(self) -> None:
        gap = time.perf_counter() - self._last
        if gap < self.interval:
            time.sleep(self.interval - gap)
        self._last = time.perf_counter()
