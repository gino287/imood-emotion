# 待GINO改寫
"""模型載入、標籤核對、單句推論與計時。"""
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .labels import MODEL_REVISION, NATIVE_LABELS, NATIVE_LABELS_SOURCE

MODEL_ID = "Johnson8187/Chinese-Emotion-Small"
MAX_LENGTH = 160


@dataclass
class Prediction:
    label: str
    confidence: float
    probs: dict          # 完整 8 類機率，不只最高分那一類
    latency_ms: dict     # tokenize / forward / post / total


def resolve_device(requested: str) -> str:
    """把 CLI 的 --device 換成實際要用的裝置。

    指定 cuda 卻沒有 GPU 時直接失敗，不靜默降級：這次評測的重點是 CPU 與 GPU
    的數字對照，靜默降級會產出「device 欄位寫著 cuda、其實跑在 CPU」的紀錄，
    數字看起來完全正常但整份對照是錯的，比直接崩潰難查得多。
    """
    available = torch.cuda.is_available()

    if requested == "cuda":
        if not available:
            raise SystemExit(
                "指定了 --device cuda，但 torch.cuda.is_available() 為 False。\n"
                "  不自動降級成 CPU：那會產生標示錯誤的量測結果。\n"
                "  請改用掛了 GPU 的服務（compose 裡的 app，不是 app-cpu），"
                "或明確指定 --device cpu / --device auto。"
            )
        return "cuda"

    if requested == "auto":
        chosen = "cuda" if available else "cpu"
        print(f"--device auto → 實際使用 {chosen}"
              + ("" if available else "（未偵測到可用的 GPU）"))
        return chosen

    return "cpu"


class EmotionClassifier:
    def __init__(self, device: str, model_id: str = MODEL_ID, max_length: int = MAX_LENGTH):
        self.device = device
        self.model_id = model_id
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.load_seconds = None

    def load(self) -> None:
        t0 = time.perf_counter()
        try:
            # revision 釘住 commit：HF repo 可變，不釘的話不同時間跑到的權重
            # 可能不同，先前量到的數字就失去比較基礎
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, revision=MODEL_REVISION)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, revision=MODEL_REVISION)
        except Exception as exc:
            # 這兩項是載入失敗最常見的原因，一併印出來省一輪來回
            raise SystemExit(
                f"模型載入失敗：{type(exc).__name__}: {exc}\n"
                f"  HF_HOME  = {os.environ.get('HF_HOME', '（未設定）')}\n"
                f"  .env 存在 = {Path('.env').exists()}\n"
                "  模型為公開資源，一般不需 token；若為網路問題請確認容器可連外。"
            )

        self.model.to(self.device)
        self.model.eval()  # 關掉 dropout，否則同一句每次跑出來的分數都不一樣
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - t0

        self._assert_labels()

    def _assert_labels(self) -> None:
        """比對模型 config 的 id2label 與 labels.py 的 NATIVE_LABELS。

        標籤順序對不上是分類任務最典型的靜默錯誤 —— 分數會低得莫名其妙，
        但不會有任何錯誤訊息。

        這個模型的 config 只有 LABEL_0..7 佔位符，模型本身不帶標籤資訊，
        順序只能來自 model card。這種情況只驗得到類別數，就照實說只驗了類別數，
        不要假裝驗過順序。
        """
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        actual = [id2label[i] for i in sorted(id2label)]

        if all(re.fullmatch(r"LABEL_\d+", str(v)) for v in actual):
            if len(actual) != len(NATIVE_LABELS):
                raise SystemExit(
                    f"模型 {self.model_id} 的類別數與設定不符，中止。\n"
                    f"  設定 : {len(NATIVE_LABELS)} 類 {NATIVE_LABELS}\n"
                    f"  模型 : {len(actual)} 類\n"
                )
            print(f"  ⚠️ 模型 config 只有 LABEL_0..{len(actual) - 1} 佔位符，"
                  f"標籤順序無法從模型驗證，只驗到類別數 {len(actual)} 相符")
            print(f"     順序出處：{NATIVE_LABELS_SOURCE}")
            return

        if actual != NATIVE_LABELS:
            raise SystemExit(
                f"模型 {self.model_id} 的標籤與設定不符，中止。\n"
                f"  設定 : {NATIVE_LABELS}\n"
                f"  模型 : {actual}\n"
                "請更新 emotion/labels.py。"
            )
        print(f"  標籤核對通過：模型 config 的 {len(actual)} 類與設定一致")

    def predict(self, text: str) -> Prediction:
        t0 = time.perf_counter()
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        t1 = time.perf_counter()

        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc).logits
        if self.device.startswith("cuda"):
            # 不同步的話量到的是「送出指令」的時間，不是實際運算時間
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        probs = torch.softmax(logits, dim=-1)[0].tolist()
        raw = {label: round(p, 6) for label, p in zip(NATIVE_LABELS, probs)}
        best = max(raw, key=raw.get)
        t3 = time.perf_counter()

        return Prediction(
            label=best,
            confidence=raw[best],
            probs=raw,
            latency_ms={
                "tokenize": round((t1 - t0) * 1000, 3),
                "forward": round((t2 - t1) * 1000, 3),
                "post": round((t3 - t2) * 1000, 3),
                "total": round((t3 - t0) * 1000, 3),
            },
        )
