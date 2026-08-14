# 待GINO改寫
"""前處理規則（規劃書 v2 §3.3）。

⚠️ 資料來源一律用 refs/BERT_SMP2020-EWECT/data/raw/，不是同層的 data/clean/。
   clean 版經 HanLP CharTable 正規化後把 [惊恐] 改成了《惊恐》，正好毀掉
   方括號表情這個情緒訊號，還產生 10 筆空字串。

處理順序有意義，不要隨意調換：
  1. 全形轉半形  → 讓後面兩步的分隔符判斷只需要處理半形
  2. 移除 @使用者 → 要在話題處理前做，避免 @名稱 裡的 # 被誤判成話題邊界
  3. 話題去符號   → #話題# 保留內文、去掉井號
  4. 空白正規化
"""
import re
from importlib.metadata import version

import opencc

# --- 全形轉半形 -------------------------------------------------------------
# 只處理 U+FF01–U+FF5E（對應 ASCII 0x21–0x7E 的全形版：標點、數字、英文字母）
# 與 U+3000 全形空格。
# 刻意不動 。、「」《》〈〉——這些是 CJK 標點區（U+3000–U+303F），沒有語意等價的
# 半形版本，硬轉會改變文字風貌。參考 repo 的 clean 版就是轉得不完整（5000 筆中
# 4119 筆含全形標點，轉完仍殘留 1946 筆），與其半吊子不如規則講清楚。
_FULLWIDTH_OFFSET = 0xFEE0
_FULLWIDTH_RANGE = (0xFF01, 0xFF5E)


def to_halfwidth(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        if _FULLWIDTH_RANGE[0] <= code <= _FULLWIDTH_RANGE[1]:
            out.append(chr(code - _FULLWIDTH_OFFSET))
        elif code == 0x3000:  # 全形空格
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


# --- @使用者名稱 ------------------------------------------------------------
# 微博的 @提及 沒有結束符號，無法精確切出邊界：
#   '@赵珂宇'                    ← 名稱後直接結束
#   '@寂落乌托邦:交友也是校园'    ← 冒號分隔，好切
#   '@小米看到那个60英寸的电'     ← 名稱是「小米」，後面直接接正文，切不乾淨
# 沿用參考實作（harvesttext）的作法：@ 之後最多吃 6 個字元，遇到分隔符提前停。
# 結尾的 :? 是為了吃掉 "@名稱:" 這種轉發格式的冒號，否則會留下一個孤兒冒號。
# 已知會在第三種情況多刪幾個字，但實測全 5000 筆中只有 39 筆含 @（0.8%），
# 抽樣後預期約 10 筆，影響可控。寧可規則寫死講得清楚，也不要用啟發式猜邊界。
_AT_MENTION = re.compile(r"@[^\s@:,.!?;、，。！？；：]{0,6}:?")

# --- 話題標籤 ---------------------------------------------------------------
# #气死我了# → 气死我了
# 話題文字本身常帶情緒，刪掉整段等於刪掉情緒訊號；井號本身不帶情緒，去掉即可。
_TOPIC = re.compile(r"#([^#]{1,30})#")

_WHITESPACE = re.compile(r"\s+")


def clean(text: str) -> str:
    """套用全部清洗規則，回傳清洗後的簡體文字。

    方括號表情（[心] [泪] [惊恐]）刻意保留，不做任何處理 —— 那是 SMP2020 原始
    語料裡本身就帶情緒訊號的內容，不是雜訊。
    """
    text = to_halfwidth(text.strip())
    text = _AT_MENTION.sub("", text)
    text = _TOPIC.sub(r"\1", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


# --- 簡體 → 台灣正體 --------------------------------------------------------
_converter = None


def to_traditional(text: str) -> str:
    """OpenCC s2twp：不只換字形，還會把「軟件」轉成「軟體」這類台灣慣用詞。

    單純的簡轉繁（s2t）只換字形，對台灣情境的可讀性幫助有限。
    """
    global _converter
    if _converter is None:
        # 轉換器建一次就好（建立要讀字典檔），但 import 一律放最上層
        _converter = opencc.OpenCC("s2twp")
    return _converter.convert(text)


def opencc_version() -> str:
    """記進資料集 meta，之後才追得出當初是哪一版轉的。"""
    try:
        return f"opencc-python-reimplemented {version('opencc-python-reimplemented')}"
    except Exception:
        return "unknown"
