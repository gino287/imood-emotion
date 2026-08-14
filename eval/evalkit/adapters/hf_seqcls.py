# 待GINO改寫
"""標準 HuggingFace 序列分類轉接器，涵蓋多數候選模型。"""
import re
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base import Adapter, RawPrediction, register


@register("hf_seqcls")
class HFSeqClsAdapter(Adapter):
    def __init__(self, model_cfg, device, max_length):
        super().__init__(model_cfg, device, max_length)
        self.tokenizer = None
        self.model = None
        self.load_seconds = None

    def load(self) -> None:
        t0 = time.perf_counter()
        hf_id = self.cfg["hf_id"]
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_id)
        self.model.to(self.device)
        self.model.eval()  # 關掉 dropout，否則同一句每次跑出來的分數都不一樣
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - t0

        self._assert_labels()

    def _assert_labels(self) -> None:
        """比對模型 config 的 id2label 與設定檔的 native_labels。

        標籤順序對不上是分類任務最典型的靜默錯誤 —— 分數會低得莫名其妙，
        但不會有任何錯誤訊息。這裡直接失敗，比事後 debug 兩小時划算。

        但實務上有兩種情況：
          (a) config 有真實標籤名 → 逐一比對，不符就中止
          (b) config 只有 LABEL_0..N 佔位符（Johnson8187 這系列就是）
              → 模型本身沒帶標籤資訊，順序只能來自 model card。
                這種情況只能驗類別數，並明確警告「順序無法從模型驗證」——
                不要假裝驗過了。
        """
        # 有些 config 從 JSON 讀進來時 key 是字串，統一轉成 int 再依序取
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        actual = [id2label[i] for i in sorted(id2label)]

        placeholder = all(re.fullmatch(r"LABEL_\d+", str(v)) for v in actual)

        if placeholder:
            if len(actual) != len(self.native_labels):
                raise ValueError(
                    f"模型 {self.cfg['hf_id']} 的類別數與設定檔不符，中止。\n"
                    f"  設定檔 native_labels : {len(self.native_labels)} 類 {self.native_labels}\n"
                    f"  模型輸出維度         : {len(actual)} 類\n"
                )
            source = self.cfg.get("native_labels_source", "（設定檔未註明出處）")
            print(f"  ⚠️ 模型 config 只有 LABEL_0..{len(actual) - 1} 佔位符，"
                  f"標籤順序無法從模型驗證，只驗到類別數 {len(actual)} 相符。")
            print(f"     順序出處：{source}")
            return

        if actual != self.native_labels:
            raise ValueError(
                f"模型 {self.cfg['hf_id']} 的標籤與設定檔不符，中止。\n"
                f"  設定檔 native_labels : {self.native_labels}\n"
                f"  模型 config id2label : {actual}\n"
                "請更新 configs/models.yaml 的 native_labels 與 mappings 的鍵。"
            )

    def predict(self, text: str) -> RawPrediction:
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
            torch.cuda.synchronize()  # 不同步的話量到的是「送出指令」的時間，不是實際運算時間
        t2 = time.perf_counter()

        probs = torch.softmax(logits, dim=-1)[0].tolist()
        raw_probs = {label: round(p, 6) for label, p in zip(self.native_labels, probs)}
        best = max(raw_probs, key=raw_probs.get)
        t3 = time.perf_counter()

        return RawPrediction(
            raw_label=best,
            raw_probs=raw_probs,
            confidence=raw_probs[best],
            timings_ms={
                "tokenize": (t1 - t0) * 1000,
                "forward": (t2 - t1) * 1000,
                "post": (t3 - t2) * 1000,
                "total": (t3 - t0) * 1000,
            },
        )

    def predict_batch(self, texts: list) -> list:
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc).logits
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        out = []
        for row in torch.softmax(logits, dim=-1).tolist():
            raw_probs = {label: round(p, 6) for label, p in zip(self.native_labels, row)}
            best = max(raw_probs, key=raw_probs.get)
            out.append(RawPrediction(best, raw_probs, raw_probs[best]))
        return out

    def truncation_count(self, texts: list) -> int:
        """有多少句被 max_length 截斷。字數不等於 token 數，這個要實測。"""
        n = 0
        for text in texts:
            if len(self.tokenizer(text)["input_ids"]) > self.max_length:
                n += 1
        return n

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
