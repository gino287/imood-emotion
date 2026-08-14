# 待GINO改寫
"""假上游：模擬 STT 以不規律間隔逐句吐出文字。

用 generator 而非 thread + queue：buffer 邏輯是「每收到一句立刻送進模型」，
句間隔 0.5~3 秒、單句推論數十毫秒，消費端永遠跟得上，佇列不可能積壓，
引入執行緒同步只會多一個出錯的地方。

⚠️ 這個判斷有到期條件：若日後改成「累積多句才送」，或上游換成會主動 push
   而不等待消費的真實模組（例如接上真的麥克風），就必須改成 queue。
   2026-08-07 曾經因為接麥克風而改成有界佇列，08/10 隨麥克風功能一起回滾；
   之後要重做時，判斷標準還是這一條。屆時只需替換這一支。
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterator

MIN_GAP_SEC = 0.5
MAX_GAP_SEC = 3.0


@dataclass
class Utterance:
    """上游送來的一句話。"""

    seq: int
    text: str
    gap_sec: float           # 與前一句的間隔，用來對照「上游多快」與「我們多快」
    dataset_label: str = ""  # 資料集原標註，僅供人工檢視，不參與計算


def load_samples(path: Path) -> list:
    if not path.exists():
        raise SystemExit(
            f"找不到樣本檔 {path}\n"
            "請先執行：docker compose -f docker/docker-compose.yml run --rm app-cpu \
"
            "               python baseline/prepare_samples.py"
        )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"樣本檔 {path} 是空的")
    return rows


def stream(
    samples: list,
    limit: int | None = None,
    seed: int = 20260806,
    delay: bool = True,
) -> Iterator[Utterance]:
    """逐句 yield，每句之前等待 0.5~3 秒的隨機間隔。

    間隔由固定 seed 決定：兩個裝置跑到完全相同的句序與節奏，速度對照才公平。
    delay=False 時仍會算出並記錄 gap_sec，只是不真的睡 —— 這樣 demo 趕時間
    可以跳過等待，而輸出的欄位仍與正常執行一致。
    """
    rng = Random(seed)
    rows = samples if limit is None else samples[:limit]

    for i, row in enumerate(rows, start=1):
        gap = rng.uniform(MIN_GAP_SEC, MAX_GAP_SEC)
        if delay:
            time.sleep(gap)
        yield Utterance(
            seq=i,
            text=row["text"],
            gap_sec=round(gap, 3),
            dataset_label=row.get("dataset_label", ""),
        )
