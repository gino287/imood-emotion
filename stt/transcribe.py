# 2026-08-18 更新，待GINO重新審閱
"""
語音轉文字（音檔 → 文字），bert前置模組
主要應用在【Prototype 階段】，模組獨立，未來可替換
此部分僅處理語音轉文字，文字前處理為bert模組內容

耗時獨立計算，不加入bert模組端到端時間計算
"""
import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel


@dataclass
class Segment:
    """Whisper 切出來的一段話，含起訖秒數。

    2026-08-18 新增：JoyGen 素材篩選要的是「這支影片第 42 到 47 秒是生氣的」
    這種切點，只有整段文字沒有時間戳就給不出來。
    """
    start: float          # 起始秒數
    end: float            # 結束秒數
    text: str             # 這一段的文字，未經清理


@dataclass
class Transcript:
    text: str             # Whisper 輸出，未經任何清理或簡繁轉換
    transcribe_ms: float  # 轉錄耗時
    segments: list = None # 逐段的 Segment；原本只回傳整串文字時是 None


class SpeechToText:
    #模型、裝置、讀取時間、暖機時間
    def __init__(self, model_size: str = "small", device: str = "cuda",
                 compute_type: str | None = None):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type or ("float16" if device == "cuda" else "int8")
        self.model = None
        self.load_seconds = None
        self.warmup_seconds = None

    def load(self) -> None:
        t0 = time.perf_counter() #開始時間
        #載入模型
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
        self._warmup()  #暖機
        self.load_seconds = time.perf_counter() - t0 #載入+暖機時間

    def _warmup(self) -> None:
        """
        事先跑一次空的輸入，進行暖機
        """
        t0 = time.perf_counter()
        silent = np.zeros(16000, dtype="float32")   # 1 秒無聲
        list(self.model.transcribe(silent, language="zh", vad_filter=False, beam_size=1)[0])
        #注意此transcribe是whisper自己的函式
        
        self.warmup_seconds = time.perf_counter() - t0

    def transcribe(self, wav_path: Path, vad_filter: bool = False) -> Transcript:
        #推論階段 回傳文字與時間   
        t0 = time.perf_counter()
        segments, _info = self.model.transcribe(
            str(wav_path),
            language="zh",
            vad_filter=vad_filter, #自動偵測靜音
            beam_size=1,   #可嘗試1、3
        )
        # faster-whisper 回傳的是 generator，只能走訪一次。
        # 原本直接 join 掉會把 start/end 一起丟光，所以先收成 list 再各取所需。
        segs = [Segment(start=s.start, end=s.end, text=s.text) for s in segments]
        text = "".join(s.text for s in segs).strip()
        return Transcript(
            text=text,
            transcribe_ms=round((time.perf_counter() - t0) * 1000, 1),
            segments=segs,
        )


def main():
    p = argparse.ArgumentParser(description="音檔轉文字驗證")
    p.add_argument("wav", type=Path, nargs="+", help="要轉錄的音檔")
    p.add_argument("--model", default="small", help="faster-whisper 模型大小（預設 small）")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    p.add_argument("--vad-filter", action="store_true",
                   help="開啟 faster-whisper 內建的 Silero VAD 過濾靜音段（預設關閉）")
    p.add_argument("--segments", action="store_true",
                   help="連同每段的起訖秒數一起印出來")
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
        if args.segments:
            for s in tr.segments or []:
                print(f"    [{s.start:7.2f} → {s.end:7.2f}] {s.text.strip()}")

    print("\n※ 這裡輸出的是 Whisper 原始文字，還沒經過前處理")


if __name__ == "__main__":
    main()
