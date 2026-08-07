"""語音轉文字：faster-whisper + 簡轉繁 + 文字衛生檢查。

⚠️ 轉錄耗時**不計入**本專案量測的延遲。這裡量的是「收到文字→分類結果」，
   語音辨識屬上游模組職責，混在一起之後端到端串接會重複計算。
   轉錄時間仍會記錄下來供參考，但與 latency_ms 分開存放。
"""
import re
import time
from dataclasses import dataclass
from pathlib import Path

# ⚠️ 刻意不使用 initial_prompt。
#
# 原本放了一句「以下是繁體中文的對話內容，請以繁體中文轉寫。」想把 Whisper 拉向繁體，
# 但 2026-08-07 實測發現：**Whisper 在靜音或無語音的音訊上會把提示詞原封吐回來**，
# 產生一則信心 0.935 的假情緒事件（「請以繁體中文轉寫。」→ 平淡語氣）。
#
# 提示詞對繁體的幫助本來就只是傾向、不是保證；真正的保證是下面的 OpenCC s2twp。
# 既然收益有限而且引進了一整類幻覺，直接拿掉比事後過濾乾淨。
#
# 下面的清單留著擋其他來源的幻覺，並保留舊提示詞以防有殘留的模型快取仍會吐出它。
_RETIRED_PROMPT = "以下是繁體中文的對話內容，請以繁體中文轉寫。"

# 文字衛生：這些是壓力測試（scripts/stress_fragments.py）實際量出來會出事的輸入。
# 「只有標點」被判為憤怒語調、信心 0.617；單一字元「我」被判為疑問語調、
# 信心 0.884 —— 靜音段若讓它們流進分類器，會產生高信心的錯誤情緒事件。
MIN_CHARS = 4

_PUNCT = "，。！？；：、「」『』（）〈〉《》…—～·．,.!?;:\"'()\\[\\]{}<>~`@#$%^&*_+=|/\\\\-"
_PUNCT_ONLY = re.compile(rf"^[\s{_PUNCT}]*$")

# faster-whisper 對中文靜音段的已知幻覺。這些字串幾乎不會出現在真實對話裡，
# 出現時代表 Whisper 在無語音的音訊上硬湊了字幕語料裡的常見句。
HALLUCINATIONS = {
    "謝謝觀看", "謝謝收看", "請不吝點贊訂閱轉發打賞支持明鏡與點點欄目",
    "字幕由Amara.org社群提供", "字幕志願者", "中文字幕由", "請訂閱",
    "谢谢观看", "谢谢收看", "请订阅",
    # 舊版曾用過的 initial_prompt 及其片段（見上方說明）
    _RETIRED_PROMPT, "請以繁體中文轉寫。", "以下是繁體中文的對話內容",
}


@dataclass
class Transcript:
    text: str                 # 已轉繁體、已清理的文字（被過濾掉時為空字串）
    raw_text: str             # Whisper 的原始輸出，保留供比對繁簡轉換是否生效
    transcribe_ms: float      # 轉錄耗時，**不計入** latency_ms
    skipped_reason: str = ""  # 非空代表這段被過濾，不該送進分類器


class SpeechToText:
    def __init__(self, model_size: str = "small", device: str = "cuda",
                 compute_type: str | None = None, to_traditional: bool = True,
                 vad_filter: bool = False):
        self.model_size = model_size
        self.device = device
        # float16 只在 GPU 上有意義；CPU 用 int8 明顯較快且品質差異可忽略
        self.compute_type = compute_type or ("float16" if device == "cuda" else "int8")
        self.to_traditional = to_traditional
        self.vad_filter = vad_filter
        self.model = None
        self.load_seconds = None
        self._converter = None

    def load(self) -> None:
        import os

        from faster_whisper import WhisperModel

        t0 = time.perf_counter()
        try:
            self.model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        except Exception as exc:
            raise SystemExit(
                f"Whisper 載入失敗：{type(exc).__name__}: {exc}\n"
                f"  模型 = {self.model_size} / 裝置 = {self.device} / 型別 = {self.compute_type}\n"
                f"  HF_HOME = {os.environ.get('HF_HOME', '（未設定）')}\n"
                "  可改用較小的模型：--whisper-model base，或改跑 CPU：--whisper-device cpu"
            )

        if self.to_traditional:
            # OpenCC s2twp：簡體 → 繁體（台灣用字，含詞彙轉換）。
            # 模型與資料集都是繁體，Whisper 吐簡體會靜默拉低準確率。
            from opencc import OpenCC

            self._converter = OpenCC("s2twp")

        self._warmup()
        self.load_seconds = time.perf_counter() - t0

    def _warmup(self) -> None:
        """先跑一次空轉錄，把 CUDA kernel 的初始化成本吃掉。

        實測第一次呼叫要 5 秒（2 秒音訊），之後降到數百毫秒。不暖機的話
        demo 的第一句話會明顯卡住 —— 雖然轉錄耗時不計入延遲數字，
        但現場看到的是「講完話等了五秒才有反應」。
        """
        import numpy as np

        t0 = time.perf_counter()
        silent = np.zeros(16000, dtype="float32")   # 1 秒無聲
        list(self.model.transcribe(silent, language="zh", vad_filter=False, beam_size=1)[0])
        self.warmup_seconds = time.perf_counter() - t0

    def _clean(self, text: str) -> tuple:
        """回傳 (清理後文字, 被過濾的原因)。原因非空代表不該送進分類器。"""
        text = text.strip()
        if not text:
            return "", "空白"
        if _PUNCT_ONLY.match(text):
            return "", "只有標點"
        if text in HALLUCINATIONS:
            return "", "Whisper 靜音幻覺"
        if len(text) < MIN_CHARS:
            # 短到這個程度時模型仍會給出高信心的預測，但那個預測沒有依據
            return "", f"過短（{len(text)} 字，門檻 {MIN_CHARS}）"
        return text, ""

    def transcribe(self, wav_path: Path) -> Transcript:
        t0 = time.perf_counter()
        segments, _info = self.model.transcribe(
            str(wav_path),
            language="zh",
            # 不給 initial_prompt：靜音時會被原封吐回來，見檔案開頭說明。
            # 繁體由下面的 OpenCC 保證，不靠提示詞。
            vad_filter=self.vad_filter,
            beam_size=1,   # demo 要即時；beam_size=1 比預設的 5 快數倍，
                           # 短句上的品質差異在人耳聽來可忽略
        )
        raw = "".join(seg.text for seg in segments).strip()
        elapsed = (time.perf_counter() - t0) * 1000

        converted = self._converter.convert(raw) if self._converter else raw
        cleaned, reason = self._clean(converted)

        return Transcript(
            text=cleaned,
            raw_text=raw,
            transcribe_ms=round(elapsed, 1),
            skipped_reason=reason,
        )
