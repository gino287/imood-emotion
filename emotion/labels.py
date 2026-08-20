# 待GINO改寫
"""模型原生標籤的唯一來源。

模型的 config.json 只有 LABEL_0..LABEL_7 佔位符，沒有真實標籤名，
因此順序無法從模型檔案本身讀出，只能取自 model card 的 label_mapping。

標籤順序對不上是分類任務最典型的靜默錯誤：分數會低得莫名其妙，
但不會有任何錯誤訊息。因此改以行為驗證：`baseline/checks/verify_labels.py` 對
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
#    並由 baseline/prepare_samples.py 在每次執行時斷言。
SAMPLE_DATASET_ID = "Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset"

# 釘住 Hugging Face 上的 commit。HF repo 是可變的，作者隨時可能更新內容；
# 不釘版本的話，同一份程式在不同時間跑會拿到不同的模型權重或資料，
# 先前量到的數字就失去比較基礎。兩者皆為 2024-12 的版本。
MODEL_REVISION = "2c04ce86de44d232f0fbe31413868eb31d791aea"
SAMPLE_DATASET_REVISION = "119d246a1595b44fd2cdccff0d9b288eafee25d1"

# ---------------------------------------------------------------------------
# 五類收斂（JoyGen 素材篩選 / 情緒影片生成用）
# ---------------------------------------------------------------------------
# 需求方要的是「中性 / 喜 / 怒 / 哀 / 樂」。其中「喜」與「樂」照字義區分：
# 喜偏遇事而生的欣喜（突發、反應性），樂偏處之而安的快樂（持續、狀態性）。
#
# ⚠️ 但這條界線在文字層畫不出來，實測如下（1200 句 SMP2020，
#    eval/results/johnson-small/cpu/zh_cn/predictions.jsonl 交叉比對）：
#
#      驚奇語調 90 筆 → surprise 66.7%、happy 僅 6.7%、sad 0.0%
#
#    也就是「驚奇語調」抓到的是價性中立的**驚訝**，不是**驚喜**。
#    另外試過用 raw_probs 做價性閘門（p[開心] 當正向、p[憤怒+悲傷+厭惡] 當負向），
#    T=0.05 時正向組 happy 13.3%、負向組 happy 11.1%，兩組幾乎相同；
#    T=0.2 時負向組的 happy 反而比正向組高。softmax 下非勝出類的殘餘機率
#    是雜訊，沒有為「當第二軸讀」而校準過，這條路走不通。
#
# 所以這一層只收斂到「驚」為止，喜/哀 的分家交給下游有訊號的地方裁決：
#   1. 臉部表情（FER）—— 驚喜（挑眉＋嘴角上揚）與驚嚇（挑眉＋嘴部緊繃）
#      在畫面上明顯可分，而且素材篩選最終要的本來就是表情不是語意
#   2. 時間軸上的前後鄰段 —— 單句分類器丟掉了上下文，一段「驚」的鄰居
#      若是「樂」，它是驚喜的機率高很多
#
# 值為 None 代表棄權，語意與 eval/evalkit/mapping.py 的 None 一致：
# 模型結構上答不出這一類，不硬塞。「關切語調」實測完全沒有歸屬
#（fear 29.9% / happy 19.7% / sad 16.2% 全糊在一起），是模型最弱的一類。

FIVE_CLASS_LABELS = ["中性", "喜", "怒", "哀", "樂"]

# 這一層實際產得出來的類別。「喜」不在其中 —— 它要靠下游把「驚」拆開才會出現。
TEXT_LAYER_LABELS = ["中性", "驚", "怒", "哀", "樂"]

NATIVE_TO_FIVE = {
    "平淡語氣": "中性",
    "關切語調": None,      # 棄權：實測無明確歸屬
    "開心語調": "樂",
    "憤怒語調": "怒",
    "悲傷語調": "哀",      # ⚠️ 高召回低精確：341 筆預測裡只有 32% 真的是 sad，
                          #    在吸收 fear(20.8%) 與 angry(15%)
    # 「疑問語調」原本也映到「驚」（SMP2020 上它 40.5% 是 surprise、0% 是 happy，
    # 看起來與驚奇同一軸）。但 2026-08-18 拿 7 支真實影片跑完之後推翻了：
    # 口語長篇敘事裡疑問語調佔全部段落的 13%，而且貢獻了「驚」的 68% ——
    # 它抓到的是講話的語氣習慣（「因為…」「就是…」這種轉折），不是驚訝。
    # SMP2020 是微博**書面文字**，這裡的使用情境是**口語影片**，兩者不一樣。
    # 改成棄權之後，「驚」就純粹來自驚奇語調，訊號乾淨得多。
    "疑問語調": None,
    "驚奇語調": "驚",      # 待下游裁決成 喜 或 哀
    "厭惡語調": "怒",      # 沿用 models.yaml 既有 to_angry 的理由：負向高喚醒，
                          #    語意距離比 fear 近
}

# 缺一類就是靜默錯誤 —— 映射表漏掉某個原生類別時，那一類的句子會安靜地
# 被跳過或 KeyError，不會有人發現。比照 mapping.py:apply() 的作法在載入時就擋掉。
assert set(NATIVE_TO_FIVE) == set(NATIVE_LABELS), (
    f"NATIVE_TO_FIVE 的鍵與 NATIVE_LABELS 不一致："
    f"缺 {set(NATIVE_LABELS) - set(NATIVE_TO_FIVE)}、"
    f"多 {set(NATIVE_TO_FIVE) - set(NATIVE_LABELS)}"
)
assert set(v for v in NATIVE_TO_FIVE.values() if v) <= set(TEXT_LAYER_LABELS)
