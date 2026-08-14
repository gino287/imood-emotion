# 待GINO改寫
"""前處理：把上游送來的文字整理成「可以送進分類器」的樣子。

pipeline 上的位置：模擬上游／STT → **前處理** → BERT 推論 → 輸出封包。

之所以獨立成一支，而不是留在 STT 裡面：STT 是可替換的前置模組，換掉 STT 不該
連前處理一起換掉。前處理是流程上自己的一站。

這裡只做兩件事：
  1. 文字層過濾 —— 擋掉會讓模型「高信心答錯」的垃圾輸入
  2. 簡繁轉換   —— 只有語音辨識來的文字需要，見 prepare() 的說明
"""
import re

from opencc import OpenCC

# 少於這個字數就不送進分類器。
#
# ⚠️ 這個門檻沒有理論依據，是 2026-08-07 看著壓力測試的實測數字挑的：
#    單一字元「我」被判成疑問語調、信心 0.884；只有標點的「。」被判成憤怒語調、
#    信心 0.617。模型對這種輸入照樣給高信心，也就是說「信心低」擋不掉它們，
#    只能在送進去之前就先擋。門檻本身還沒被重新檢視過（見 notes 的待決事項）。
MIN_CHARS = 4

# 純標點的判斷用。中英標點都列，因為語音辨識在靜音段兩種都吐得出來。
_PUNCT = "，。！？；：、「」『』（）〈〉《》…—～·．,.!?;:\"'()\\[\\]{}<>~`@#$%^&*_+=|/\\-"
_PUNCT_ONLY = re.compile(rf"^[\s{_PUNCT}]*$")

# faster-whisper 對中文靜音段的已知幻覺。
#
# 這些字串在真實對話裡幾乎不可能出現，出現就代表辨識器在沒有語音的音訊上，
# 硬湊了訓練語料（大量 YouTube 字幕）裡的常見句。
#
# 後面三條是我們自己製造的幻覺：舊版曾用 initial_prompt 引導繁體輸出，
# 結果辨識器在靜音段把提示詞原封吐回來，還被分類成信心 0.935 的情緒事件。
# 提示詞已經拿掉了（繁體改由下面的 OpenCC 保證），但舊的模型快取可能還會吐，
# 所以清單留著。
HALLUCINATIONS = {
    "謝謝觀看", "謝謝收看", "請不吝點贊訂閱轉發打賞支持明鏡與點點欄目",
    "字幕由Amara.org社群提供", "字幕志願者", "中文字幕由", "請訂閱",
    "谢谢观看", "谢谢收看", "请订阅",
    "以下是繁體中文的對話內容，請以繁體中文轉寫。",
    "請以繁體中文轉寫。", "以下是繁體中文的對話內容",
}

# OpenCC 的轉換器建一次就好（建立要讀字典檔，每句重建會很慢）。
# import 放最上層、物件在第一次用到時才建，兩件事分開。
_converter = None


def to_traditional(text: str) -> str:
    """簡體 → 繁體（台灣用字）。

    用 s2twp 不用 s2t：s2t 只換字形，s2twp 連詞彙也換（軟件→軟體）。
    模型與資料集都是繁體，餵簡體進去會靜默拉低準確率 —— 不會報錯，只是分數變差。
    """
    global _converter
    if _converter is None:
        _converter = OpenCC("s2twp")
    return _converter.convert(text)


def clean(text: str) -> tuple[str, str]:
    """回傳 (整理後的文字, 被擋下來的原因)。

    原因是空字串代表通過；非空代表這段不該送進分類器，字串本身是給人看的說明。
    刻意回傳原因而不是只回 None：略過的時候要能印出「為什麼略過」，
    不然現場看起來就像程式漏掉了幾句。
    """
    text = text.strip()
    if not text:
        return "", "空白"
    if _PUNCT_ONLY.match(text):
        return "", "只有標點"
    if text in HALLUCINATIONS:
        return "", "語音辨識靜音幻覺"
    if len(text) < MIN_CHARS:
        return "", f"過短（{len(text)} 字，門檻 {MIN_CHARS}）"
    return text, ""


def prepare(text: str, traditional: bool = False) -> tuple[str, str]:
    """前處理的單一入口。回傳 (整理後的文字, 被擋下來的原因)。

    traditional 預設 False，只有語音辨識來的文字才要開。
    原因：s2twp 除了字形還會換詞彙，而樣本資料集本來就是繁體，對它做轉換有機會
    改到用詞 → 改到模型輸入 → 讓 200 句 baseline 的數字跟之前量的不能比。
    簡體是上游語音辨識才有的問題，不是所有輸入的問題，所以做成選項而不是預設。

    文字層過濾則兩條路徑都跑：那是防呆，正常的句子不會被它動到。
    """
    if traditional:
        text = to_traditional(text)
    return clean(text)
