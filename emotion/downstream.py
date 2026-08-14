# 待GINO改寫
"""給下游模組（Emotion Video Selector / JoyGen）的精簡封包。

與評估用的 .jsonl 分開的理由：那份為了事後分析存了完整 8 類機率、四段耗時、
裝置別、原始轉錄等等；下游只需要「這句話是什麼情緒、有多確定」。
把兩者混在一起，下游得先學會忽略一堆與它無關的欄位。

格式刻意只有四個欄位。品靜那邊的實際需求還沒定案，先給最小可用集合，
不預先設計還沒被提出的需求 —— 加欄位很容易，拿掉已經被依賴的欄位很難。
"""
import json
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class DownstreamWriter:
    """一行一筆串流寫入，下游 `tail -f` 就能消費，不需要任何協定。"""

    def __init__(self, path: Path):
        self.path = path
        self.count = 0
        self._fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        if self._fh:
            self._fh.close()
        return False

    def write(self, text: str, emotion: str, confidence: float) -> dict:
        packet = {
            "ts": now_iso(),
            "text": text,
            "emotion": emotion,
            "confidence": round(confidence, 4),
        }
        self._fh.write(json.dumps(packet, ensure_ascii=False) + "\n")
        # 逐筆 flush：下游是即時消費，緩衝住就失去意義；
        # 也讓 Ctrl+C 之後檔案最後一行仍是完整的 JSON
        self._fh.flush()
        self.count += 1
        return packet
