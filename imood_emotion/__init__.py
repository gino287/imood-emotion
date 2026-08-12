"""imood.ai 的 BERT 情緒分類模組。

pipeline 位置：麥克風／音檔 → STT（可替換的前置模組，見 stt/）
              → **前處理 → BERT 推論 → 輸出封包**（就是這個套件）

模組分工：
  source.py      假上游：模擬 STT 以不規律間隔逐句吐出文字
  preprocess.py  前處理：文字層過濾與簡繁轉換
  classifier.py  模型載入、標籤核對、單句推論與計時
  recorder.py    逐句結果落地（.jsonl）與執行環境紀錄（.meta.json）
  downstream.py  給下游模組的精簡封包
  labels.py      模型原生 8 類標籤與 HF 版本釘選的唯一來源

之所以把「輸入來源」獨立成 source.py：日後上游換成真的 STT 模組時，
只需替換這一支，其餘不必更動。同理，前處理獨立成 preprocess.py，
換掉 STT 不會連前處理一起換掉。
"""

from .labels import NATIVE_LABELS

__all__ = ["NATIVE_LABELS"]
