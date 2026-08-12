"""語音轉文字（音檔 → 文字）。

這是 pipeline 上的**前置模組，可替換**：換掉整個 stt/ 不該影響 BERT 那一端，
所以這支刻意不 import imood_emotion 的任何東西，也不做任何前處理
（簡繁轉換與文字過濾都在 imood_emotion/preprocess.py，那是下一站的事）。

用途縮限在「音檔轉文字驗證」。真實麥克風即時收音已於 2026-08-10 回滾，
理由是把 STT 塞進 BERT 容器會讓只想要 BERT 的人被迫連帶抓 whisper 依賴。

自己有一份 requirements.txt 與 docker/dockerfile.stt，跟 BERT 那邊的依賴分開。

    docker compose -f docker/docker-compose.yml run --rm stt \
        python stt/transcribe.py 某個音檔.wav

⚠️ 轉錄耗時**不計入**本專案量測的延遲。專案量的是「收到文字 → 分類結果」，
   語音辨識屬上游模組職責，混在一起之後端到端串接會重複計算。
"""
import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

# ⚠️ 刻意不使用 initial_prompt。
#
# 原本放了一句「以下是繁體中文的對話內容，請以繁體中文轉寫。」想把 Whisper 拉向繁體，
# 但 2026-08-07 實測發現：**Whisper 在靜音或無語音的音訊上會把提示詞原封吐回來**，
# 產生一則信心 0.935 的假情緒事件（「請以繁體中文轉寫。」→ 平淡語氣）。
#
# 提示詞對繁體的幫助本來就只是傾向、不是保證；真正的保證是下一站前處理的 OpenCC。
# 既然收益有限而且引進了一整類幻覺，直接拿掉比事後過濾乾淨。
# （擋幻覺的清單留在 imood_emotion/preprocess.py，因為那是文字層的事）


@dataclass
class Transcript:
    text: str             # Whisper 的輸出，未經任何清理或簡繁轉換
    transcribe_ms: float  # 轉錄耗時，**不計入** latency_ms


class SpeechToText:
    def __init__(self, model_size: str = "small", device: str = "cuda",
                 compute_type: str | None = None):
        self.model_size = model_size
        self.device = device
        # float16 只在 GPU 上有意義；CPU 用 int8 明顯較快且品質差異可忽略
        self.compute_type = compute_type or ("float16" if device == "cuda" else "int8")
        self.model = None
        self.load_seconds = None
        self.warmup_seconds = None

    def load(self) -> None:
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
                "  可改用較小的模型：--model base，或改跑 CPU：--device cpu"
            )
        self._warmup()
        self.load_seconds = time.perf_counter() - t0

    def _warmup(self) -> None:
        """先跑一次空轉錄，把 CUDA kernel 的初始化成本吃掉。

        實測第一次呼叫要 5 秒（2 秒音訊），之後降到數百毫秒。不暖機的話
        第一支音檔會明顯卡住，看起來像壞掉。
        """
        t0 = time.perf_counter()
        silent = np.zeros(16000, dtype="float32")   # 1 秒無聲
        list(self.model.transcribe(silent, language="zh", vad_filter=False, beam_size=1)[0])
        self.warmup_seconds = time.perf_counter() - t0

    def transcribe(self, wav_path: Path, vad_filter: bool = False) -> Transcript:
        t0 = time.perf_counter()
        segments, _info = self.model.transcribe(
            str(wav_path),
            language="zh",
            vad_filter=vad_filter,
            beam_size=1,   # 要即時；beam_size=1 比預設的 5 快數倍，
                           # 短句上的品質差異在人耳聽來可忽略
        )
        text = "".join(seg.text for seg in segments).strip()
        return Transcript(text=text, transcribe_ms=round((time.perf_counter() - t0) * 1000, 1))


def main():
    p = argparse.ArgumentParser(description="音檔轉文字驗證")
    p.add_argument("wav", type=Path, nargs="+", help="要轉錄的音檔")
    p.add_argument("--model", default="small", help="faster-whisper 模型大小（預設 small）")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    p.add_argument("--vad-filter", action="store_true",
                   help="開啟 faster-whisper 內建的 Silero VAD 過濾靜音段（預設關閉）")
    args = p.parse_args()

    missing = [w for w in args.wav if not w.exists()]
    if missing:
        raise SystemExit(f"找不到音檔：{', '.join(str(m) for m in missing)}")

    stt = SpeechToText(model_size=args.model, device=args.device)
    print(f"載入 Whisper {args.model} → {args.device}")
    stt.load()
    print(f"  載入耗時 {stt.load_seconds:.2f}s（含暖機 {stt.warmup_seconds:.2f}s）\n")

    for wav in args.wav:
        tr = stt.transcribe(wav, vad_filter=args.vad_filter)
        print(f"{wav.name}　（{tr.transcribe_ms:.0f}ms）")
        print(f"  {tr.text or '（沒有辨識到內容）'}")

    print("\n※ 這裡輸出的是 Whisper 原始文字，還沒經過前處理"
          "（簡繁轉換與文字過濾在 imood_emotion/preprocess.py）。")


if __name__ == "__main__":
    main()
