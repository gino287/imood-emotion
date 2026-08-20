# 待GINO改寫
"""四套標籤體系與它們之間的映射表，集中在這一個檔案。

這裡不做任何運算，只放「哪一類對到哪一類」。要看內容跑：

    python sourcing/mapping/show_mapping.py

**為什麼要集中。**
這條線上同時有四套標籤在跑：BERT 的 8 類（中文語氣詞）、FER 的 8 類
（AffectNet 英文）、RAVDESS 的 8 類（數字代號）、我們要的 5 類。
之前這幾張表散在三個檔案裡（emotion/labels.py、face_timeline.py、
prepare_ravdess.py），改一張要記得去對另外兩張，很容易漏。
集中之後「怒到底吃進了哪些東西」一眼看得完。

**五類是唯一的共同語言。** 各家的原生類別不必也不該互相對齊 ——
AffectNet 的 Contempt 在 BERT 那邊沒有對應，BERT 的關切語調在 AffectNet
那邊也沒有。硬對齊只會製造假資料。所以全部各自收斂到五類，
只在五類這一層比較。

**None 一律代表棄權**，語意與 eval/evalkit/mapping.py 的 None 相同：
這一套體系裡沒有對應的位置，或模型結構上答不出來，就不硬塞。

⚠️ 唯一不在這裡的是 BERT 的 8→5 那張表（NATIVE_TO_FIVE）。它留在
   emotion/labels.py，因為那張表上線之後真的會被呼叫到，而 sourcing/
   是離線的一次性流程。這裡只是 re-export 過來，方便四張表並排看。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emotion.labels import (  # noqa: E402  F401
    FIVE_CLASS_LABELS,
    NATIVE_LABELS,
    NATIVE_TO_FIVE,
    TEXT_LAYER_LABELS,
)

# ---------------------------------------------------------------------------
# FER（hsemotion / AffectNet 8 類）→ 五類
# ---------------------------------------------------------------------------
# 這是 sourcing/detect/face_timeline.py 用的表。
AFFECT_TO_FIVE = {
    "Happiness": "樂",
    "Sadness": "哀",
    "Anger": "怒",
    "Disgust": "怒",       # 與文字端 to_angry 的處理一致：負向高喚醒
    "Surprise": "驚",
    "Neutral": "中性",
    "Fear": None,          # 五類裡沒有恐懼
    "Contempt": None,      # 輕蔑與憤怒在畫面上差很多，不併
}

AFFECT_LABELS = list(AFFECT_TO_FIVE)

# ---------------------------------------------------------------------------
# 交付用的五個資料夾
# ---------------------------------------------------------------------------
# 2026-08-19 由 Gino 指定的規格。與偵測層的五類差在兩個名字：
#   中性 → 預設   下游拿它當「這個人沒有情緒時的臉」的基準
#   驚   → 喜     驚訝的演出直接當成喜
DELIVERY_LABELS = ["預設", "喜", "怒", "哀", "樂"]

# 偵測層的標籤 → 交付資料夾。做 FER 交叉驗證時要用它把兩邊講成同一種話。
DETECT_TO_DELIVERY = {
    "中性": "預設",
    "驚": "喜",
    "怒": "怒",
    "哀": "哀",
    "樂": "樂",
    "喜": "喜",      # 偵測層產不出來，列著讓這張表是全的
}

# ---------------------------------------------------------------------------
# RAVDESS（檔名第 3 欄的數字代號）→ 交付資料夾
# ---------------------------------------------------------------------------
# 這是 sourcing/actors/prepare_ravdess.py 用的表。
# 演員資料集的好處就在這裡：情緒是拍攝時就決定的，寫在檔名裡，不需要任何偵測。
#
# ⚠️ 08 surprised → 喜 是規格指定的。文字端測過「驚訝」是價性中立的
#    （SMP2020 上 surprise 66.7%、happy 只有 6.7%），但那是文字；
#    演員演出來的 surprised 是明確的挑眉睜眼，當作「喜」在畫面上說得通。
#    真正要留意的是驚喜與驚嚇的差別，這要看素材本身，見 prepare_ravdess.py。
RAVDESS_TO_DELIVERY = {
    "01": "預設",    # neutral
    "02": "預設",    # calm，與 neutral 同一個資料夾
    "03": "樂",
    "04": "哀",
    "05": "怒",
    "06": None,      # fearful，規格指定不抓
    "07": None,      # disgust，規格指定不抓
    "08": "喜",      # surprised
}

# 舊的那張表（08 → 驚、02 棄權）已經被上面的規格取代。
# 保留名字是因為它表達的是「偵測層的五類」，跟交付資料夾不是同一件事。
RAVDESS_TO_FIVE = {
    code: (None if label is None else
           ("中性" if label == "預設" else ("驚" if label == "喜" else label)))
    for code, label in RAVDESS_TO_DELIVERY.items()
}

RAVDESS_CODE_NAMES = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}

# ---------------------------------------------------------------------------
# 偵測層實際產得出來的標籤
# ---------------------------------------------------------------------------
# 需求方要的五類含「喜」，但三套偵測都產不出喜 —— 它們只到「驚」為止，
# 喜／哀 的分家要靠下游拆開驚（見 emotion/labels.py 的說明）。
# 所以三張映射表的值域是這一組，不是 FIVE_CLASS_LABELS。
DETECT_LABELS = list(TEXT_LAYER_LABELS)      # ["中性", "驚", "怒", "哀", "樂"]

# 印表格、跑統計時要走過的完整標籤集合：五類加上還沒拆的「驚」。
# 順序固定，兩次輸出的欄位才對得上。
ALL_LABELS = ["中性", "喜", "怒", "哀", "樂", "驚"]

# ---------------------------------------------------------------------------
# 要去找的四類
# ---------------------------------------------------------------------------
# 中性不列入 —— 每支影片都一堆中性，拿它當達成條件沒有意義。
TARGET = ["哀", "怒", "樂", "驚"]

# 五類與 SMP2020 六類代號的對照。評測那邊的真實標籤是英文六類，
# 要把兩邊的數字放在一起看時需要這張表（eval/configs/models.yaml 的
# joygen5 就是用六類代號寫成的）。
FIVE_TO_SMP = {
    "中性": "neutral",
    "喜": "happy",      # 偵測層產不出來，見 emotion/labels.py 的說明
    "怒": "angry",
    "哀": "sad",
    "樂": "happy",
    "驚": "surprise",   # 尚未拆成喜/哀的那一類
}


def five_classes(table: dict) -> set:
    """一張映射表實際會產出哪些五類標籤（不含棄權）。"""
    return {v for v in table.values() if v}


# 三張表的值域都必須落在 DETECT_LABELS 之內。任何一張表打錯字
#（例如把「樂」寫成「快樂」）都會在載入時就炸掉，而不是等到跑完幾百支影片
# 才發現某一類永遠是 0。
for _name, _table in (("AFFECT_TO_FIVE", AFFECT_TO_FIVE),
                      ("RAVDESS_TO_FIVE", RAVDESS_TO_FIVE),
                      ("NATIVE_TO_FIVE", NATIVE_TO_FIVE)):
    _bad = five_classes(_table) - set(DETECT_LABELS)
    assert not _bad, f"{_name} 出現不在偵測標籤裡的值：{_bad}"

assert set(TARGET) <= set(DETECT_LABELS)
assert set(FIVE_TO_SMP) == set(ALL_LABELS)
assert set(ALL_LABELS) == set(FIVE_CLASS_LABELS) | set(DETECT_LABELS)

# 交付那組同樣要能對得起來：偵測層產得出來的每一類都要有去處，
# 而且去處都要是規格裡的五個資料夾之一。
assert set(DETECT_LABELS) <= set(DETECT_TO_DELIVERY)
assert set(DETECT_TO_DELIVERY.values()) <= set(DELIVERY_LABELS)
assert five_classes(RAVDESS_TO_DELIVERY) <= set(DELIVERY_LABELS)
