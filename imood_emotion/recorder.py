"""逐句結果落地（.jsonl）與執行環境紀錄（.meta.json）。"""
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from .labels import MODEL_REVISION, SAMPLE_DATASET_REVISION


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


class Recorder:
    """一句一行寫出，不在記憶體裡累積整批。

    串流場景中途可能被中斷（demo 現場尤其），逐行 flush 的話中斷前的結果都還在，
    不會整批丟失。
    """

    def __init__(self, out_path: Path, device: str, model_id: str, samples_path: Path):
        self.out_path = out_path
        self.meta_path = out_path.with_suffix(".meta.json")
        self.device = device
        self.model_id = model_id
        self.samples_path = samples_path
        self.records = []
        self._fh = None

    def __enter__(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.out_path.open("w", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        if self._fh:
            self._fh.close()
        return False

    def write(self, utt, pred, warmup: bool, preprocess_ms: float = 0.0) -> dict:
        # 欄位順序刻意安排：seq / text / pred_label 排前面，掃 jsonl 時
        # 一行的前 80 字元就看得懂；8 鍵的 probs 放後面免得洗版。
        rec = {
            "seq": utt.seq,
            "warmup": warmup,
            "recv_at": now_iso(),
            "gap_sec": utt.gap_sec,
            "text": utt.text,
            "pred_label": pred.label,
            "confidence": pred.confidence,
            # 前處理耗時獨立記錄，**不併入 latency_ms**。latency_ms 的定義是
            # tokenize/forward/post 三段，動這個定義就會讓新舊 baseline 不能比。
            "preprocess_ms": round(preprocess_ms, 3),
            "latency_ms": pred.latency_ms,
            "probs": pred.probs,
            "device": self.device,
            "dataset_label": utt.dataset_label,
        }
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()
        self.records.append(rec)
        return rec

    def write_meta(self, load_seconds: float, extra: dict | None = None) -> None:
        """執行環境指紋。

        CPU 與 GPU 各產一份 jsonl，沒有這個檔案，過幾天就分不出哪份是哪次跑的、
        當時是什麼版本、用的是不是同一份樣本。
        """
        meta = {
            "timestamp": now_iso(),
            "model_id": self.model_id,
            "model_revision": MODEL_REVISION,
            "dataset_revision": SAMPLE_DATASET_REVISION,
            "device": self.device,
            "model_load_seconds": round(load_seconds, 3),
            "samples_file": str(self.samples_path),
            "samples_sha256": file_sha256(self.samples_path),
            "n_records": len(self.records),
            "n_warmup": sum(1 for r in self.records if r["warmup"]),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "in_container": Path("/.dockerenv").exists(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        if self.device.startswith("cuda"):
            meta["torch_peak_allocated_mb"] = round(
                torch.cuda.max_memory_allocated() / 1024 / 1024, 1
            )
            meta["torch_peak_reserved_mb"] = round(
                torch.cuda.max_memory_reserved() / 1024 / 1024, 1
            )
        if extra:
            meta.update(extra)

        self.meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
