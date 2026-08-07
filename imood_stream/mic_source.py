"""麥克風來源：輪詢主機寫進來的 wav，轉錄後推進有界佇列。

為什麼從 generator 換成佇列（ADR-005 記錄的到期條件被觸發）：
模擬上游是被動的 —— 消費端沒來拿，它就不吐下一句。真實麥克風不是，
人講話不會等你算完。轉錄與分類必須各自跑在自己的步調上，中間需要一個
會滿、滿了要有明確策略的緩衝。

佇列滿了丟**最舊**的：陪伴情境裡使用者當下講的話比十秒前的更重要。
"""
import queue
import threading
import time
from pathlib import Path

from .source import Utterance

DEFAULT_AUDIO_DIR = Path("_local/audio")
POLL_INTERVAL_SEC = 0.15
IDLE_NOTICE_SEC = 10.0   # 等不到音檔時多久提示一次，避免看起來像卡住


class MicSource:
    """背景執行緒轉錄，主執行緒消費。

    用標準庫的 queue 而非 asyncio：整條路徑都是阻塞式 I/O（讀檔、Whisper、BERT），
    一個生產者執行緒加主執行緒消費就夠，引入事件迴圈只會多一層心智負擔。
    """

    def __init__(self, stt, audio_dir: Path = DEFAULT_AUDIO_DIR, queue_size: int = 8,
                 keep_audio: bool = False, verbose: bool = True):
        self.stt = stt
        self.audio_dir = audio_dir
        self.keep_audio = keep_audio
        self.verbose = verbose

        self.q: queue.Queue = queue.Queue(maxsize=queue_size)
        self.dropped = 0          # 佇列滿而被丟棄的段數
        self.skipped = 0          # 轉錄後被文字衛生檢查擋下的段數
        self.transcribed = 0
        self.transcribe_ms = []

        self._stop = threading.Event()
        self._thread = None
        self._seen = set()
        self._seq = 0
        self._last_yield_at = None

    # ---- 生產端 -----------------------------------------------------------

    def start(self) -> None:
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        # 啟動時把既有檔案標記為已看過：那些是上一輪殘留的，不該被當成新輸入
        self._seen = {p.name for p in self.audio_dir.glob("*.wav")}
        if self._seen and self.verbose:
            print(f"  忽略 {len(self._seen)} 個啟動前就存在的音檔")

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        last_notice = time.time()
        while not self._stop.is_set():
            new = sorted(p for p in self.audio_dir.glob("*.wav") if p.name not in self._seen)
            if not new:
                if self.verbose and time.time() - last_notice > IDLE_NOTICE_SEC:
                    print(f"  （等待音檔中… 請確認主機端的 record_mic.py 正在執行，"
                          f"輸出到 {self.audio_dir}）")
                    last_notice = time.time()
                self._stop.wait(POLL_INTERVAL_SEC)
                continue

            last_notice = time.time()
            for path in new:
                if self._stop.is_set():
                    return
                self._seen.add(path.name)
                self._handle(path)

    def _handle(self, path: Path) -> None:
        try:
            tr = self.stt.transcribe(path)
        except Exception as exc:
            print(f"  ⚠️ 轉錄失敗 {path.name}：{type(exc).__name__}: {exc}")
            return
        finally:
            if not self.keep_audio:
                path.unlink(missing_ok=True)   # 不刪的話長時間錄音會塞爆磁碟

        self.transcribed += 1
        self.transcribe_ms.append(tr.transcribe_ms)

        if tr.skipped_reason:
            self.skipped += 1
            if self.verbose:
                raw = tr.raw_text[:20] if tr.raw_text else "（空）"
                print(f"  · 略過 {path.name}：{tr.skipped_reason}　原始輸出「{raw}」")
            return

        self._seq += 1
        now = time.time()
        gap = 0.0 if self._last_yield_at is None else round(now - self._last_yield_at, 3)
        self._last_yield_at = now

        self._put_drop_oldest(Utterance(
            seq=self._seq,
            text=tr.text,
            gap_sec=gap,
            transcribe_ms=tr.transcribe_ms,
            raw_text=tr.raw_text,
        ))

    def _put_drop_oldest(self, item) -> None:
        """佇列滿時丟掉最舊的一筆，再放入新的。

        只有這一個生產者執行緒會呼叫，所以先 get 再 put 不會有競爭問題。
        """
        try:
            self.q.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            stale = self.q.get_nowait()
            self.dropped += 1
            if self.verbose:
                print(f"  ⚠️ 佇列已滿，丟棄較舊的第 {stale.seq} 段"
                      f"（累計丟棄 {self.dropped}）")
        except queue.Empty:
            pass
        try:
            self.q.put_nowait(item)
        except queue.Full:
            self.dropped += 1   # 極端情況：剛騰出的位置又被填滿

    # ---- 消費端 -----------------------------------------------------------

    def stream(self):
        """逐句 yield。沒有資料時阻塞等待，直到 stop() 被呼叫。"""
        while not self._stop.is_set():
            try:
                yield self.q.get(timeout=0.3)
            except queue.Empty:
                continue

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def stats(self) -> dict:
        import statistics

        out = {
            "audio_chunks_transcribed": self.transcribed,
            "utterances_emitted": self._seq,
            "queue_dropped": self.dropped,
            "text_skipped": self.skipped,
        }
        if self.transcribe_ms:
            # 轉錄耗時獨立記錄：屬上游模組職責，刻意不併入 latency_ms
            out["transcribe_ms_mean"] = round(statistics.mean(self.transcribe_ms), 1)
            out["transcribe_ms_max"] = round(max(self.transcribe_ms), 1)
        return out
