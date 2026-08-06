"""模型原生標籤的唯一來源。

⚠️ `Johnson8187/Chinese-Emotion-Small` 的 config.json 只有 LABEL_0..LABEL_7
   佔位符，沒有真實標籤名，所以**順序無法從模型本身驗證**。
   下面的順序取自 model card 的 label_mapping（2026-07-31 查核）：
   https://huggingface.co/Johnson8187/Chinese-Emotion-Small

   標籤順序對不上是分類任務最典型的靜默錯誤：分數會低得莫名其妙，
   但不會有任何錯誤訊息。若哪天結果離譜，第一個要懷疑的就是這裡。
"""

NATIVE_LABELS = [
    "平淡語氣",
    "關切語調",
    "開心語調",
    "憤怒語調",
    "悲傷語調",
    "疑問語調",
    "驚奇語調",
    "厭惡語調",
]

NATIVE_LABELS_SOURCE = (
    "https://huggingface.co/Johnson8187/Chinese-Emotion-Small （model card 的 label_mapping）"
)

# 樣本資料集（同作者發布）。其 README 列出的標籤清單與實際 data.csv 不符
# ——README 寫有「恐懼語調」「驚訝語調」，實際資料裡是 0 筆與不存在，
# 真正出現的是「關切語調」「驚奇語調」，與模型的 8 類一致。
# 以實際資料為準，並由 scripts/prepare_samples.py 在執行時斷言。
SAMPLE_DATASET_ID = "Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset"
