"""串流情緒分類 MVP。

模組分工：
  source.py      假上游：模擬 STT 以不規律間隔逐句吐出文字
  classifier.py  模型載入、標籤核對、單句推論與計時
  recorder.py    逐句結果落地（.jsonl）與執行環境紀錄（.meta.json）
  labels.py      模型原生 8 類標籤的唯一來源

之所以把「輸入來源」獨立成 source.py：日後上游換成真的 STT 模組時，
只需替換這一支，classifier 與 recorder 不必更動。
"""

from .labels import NATIVE_LABELS

__all__ = ["NATIVE_LABELS"]
