"""模型原生標籤的唯一來源。

模型的 config.json 只有 LABEL_0..LABEL_7 佔位符，沒有真實標籤名，
因此順序無法從模型檔案本身讀出，只能取自 model card 的 label_mapping。

標籤順序對不上是分類任務最典型的靜默錯誤：分數會低得莫名其妙，
但不會有任何錯誤訊息。因此改以行為驗證：`scripts/verify_labels.py` 對
480 句計算混淆矩陣，八類的對角線全數浮出、整體對角率 87.9%
（順序若接錯應接近隨機的 12.5%）。順序正確，2026-08-06 實測。
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

# 樣本資料集（與模型同作者發布）。
#
# ⚠️ 該資料集 README 列出的標籤清單與實際 data.csv 不符：README 寫有
#    「恐懼語調」「驚訝語調」，實際資料中兩者皆為 0 筆；真正存在的是
#    「關切語調」「驚奇語調」。以實際資料為準 —— 實際的 8 類與模型完全對應，
#    並由 scripts/prepare_samples.py 在每次執行時斷言。
SAMPLE_DATASET_ID = "Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset"

# 釘住 Hugging Face 上的 commit。HF repo 是可變的，作者隨時可能更新內容；
# 不釘版本的話，同一份程式在不同時間跑會拿到不同的模型權重或資料，
# 先前量到的數字就失去比較基礎。兩者皆為 2024-12 的版本。
MODEL_REVISION = "2c04ce86de44d232f0fbe31413868eb31d791aea"
SAMPLE_DATASET_REVISION = "119d246a1595b44fd2cdccff0d9b288eafee25d1"
