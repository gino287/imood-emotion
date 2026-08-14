"""imood.ai 的 BERT 情緒分類模組。

pipeline 位置：麥克風／音檔 → STT（可替換的前置模組，見 stt/）
              → **前處理 → BERT 推論 → 輸出封包**（就是這個套件）

模組分工：
  preprocess.py  前處理：文字層過濾與簡繁轉換
  classifier.py  模型載入、標籤核對、單句推論與計時
  downstream.py  給下游模組的精簡封包
  labels.py      模型原生 8 類標籤與 HF 版本釘選的唯一來源

這個套件只收上線之後真的會被呼叫到的東西。假上游（fake_stt.py）與逐句錄檔
（recorder.py）只有跑測試才用得到，所以放在 baseline/，不放這裡。

前處理獨立成 preprocess.py 的理由不變：上游換成真的 STT 模組時只換上游那一支，
前處理不必跟著換。
"""

from .labels import NATIVE_LABELS

__all__ = ["NATIVE_LABELS"]
