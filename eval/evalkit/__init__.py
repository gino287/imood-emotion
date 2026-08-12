"""imood BERT 情緒分類 — 評測骨架。

設計原則（規劃書 v2 §4.1）：推論與映射解耦。
runner 只負責產出模型原生輸出與完整機率向量，映射表是報告階段的後處理，
所以換映射、調信心門檻、重算指標都不需要重跑模型。
"""

__all__ = ["dataset", "preprocess", "runner", "mapping", "metrics", "report", "adapters"]
