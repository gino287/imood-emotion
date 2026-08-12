"""轉接器介面。

⚠️ 轉接器只負責推論，不做標籤映射（規劃書 v2 §4.1）。
   輸出原生類別與完整機率向量，映射交給報告階段的 mapping.py。
   這樣換映射表不必重跑模型 —— 1200 句 × N 模型 × 2 裝置 × 2 變體，
   重跑的代價高到會讓人懶得試第二種映射。
"""
from dataclasses import dataclass, field


@dataclass
class RawPrediction:
    raw_label: str                  # 模型原生類別（映射前）
    raw_probs: dict                 # {原生類別: 機率}，完整向量
    confidence: float               # 最高機率
    timings_ms: dict = field(default_factory=dict)  # tokenize / forward / post / total


class Adapter:
    """每支轉接器要實作 load() 與 predict()，其餘有預設行為。"""

    def __init__(self, model_cfg: dict, device: str, max_length: int):
        self.cfg = model_cfg
        self.device = device
        self.max_length = max_length
        self.native_labels = list(model_cfg["native_labels"])
        self.load_seconds = None  # 由 load() 填，runner 會拿去記 cold start

    def load(self) -> None:
        raise NotImplementedError

    def predict(self, text: str) -> RawPrediction:
        raise NotImplementedError

    def predict_batch(self, texts: list) -> list:
        """吞吐量參考用。預設退回逐句，轉接器可覆寫成真正的 batch 推論。

        ⚠️ batch 數字不是產品場景（產品是單句即時），報告會另外標注。
        """
        return [self.predict(t) for t in texts]

    def unload(self) -> None:
        pass


_REGISTRY = {}


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls

    return deco


def get_adapter(name: str):
    """從註冊表取一支轉接器。

    註冊是 import 的副作用（@register 裝飾器），所以要先確定所有轉接器模組
    都被 import 過 —— 這件事由 adapters/__init__.py 在最下面統一做。
    """
    if name not in _REGISTRY:
        raise KeyError(f"未知的 adapter '{name}'，目前有：{sorted(_REGISTRY)}")
    return _REGISTRY[name]
