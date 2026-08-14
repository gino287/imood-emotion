# 待GINO改寫
"""模型轉接器。

新增候選模型時，只有推論介面不是標準 HuggingFace
AutoModelForSequenceClassification 的（例如托管在 ModelScope 的模型），
才需要在這裡新增一支；其餘情況改 configs/models.yaml 就夠了。
"""
from .base import Adapter, RawPrediction, get_adapter

# 每支轉接器靠 @register 裝飾器把自己登記進 base 的註冊表，而那要等模組被
# import 過才會發生。所以這裡要逐一列出來 —— 新增轉接器時記得補一行，
# 不然 get_adapter() 會說「未知的 adapter」。
# 放在 from .base 之後：hf_seqcls 反過來要 import base 的 Adapter。
from . import hf_seqcls  # noqa: E402,F401

__all__ = ["Adapter", "RawPrediction", "get_adapter"]
