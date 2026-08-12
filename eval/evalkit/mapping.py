"""映射表套用與三視角切分（規劃書 v2 §4.3(A)）。

映射是報告階段的後處理，不是推論的一部分 —— 同一份 predictions.jsonl
可以零成本套用任意多組映射表，這正是「厭惡語調到底該映到哪」能用數字結案的原因。

映射值為 None 代表棄權：模型結構上答不出這一類。三個視角承接這件事：
  strict  棄權一律算答錯          → 產品實際會拿到的表現
  covered 只算有映射的子集        → 模型有答的部分準不準
  subset  排除模型宣告不具備的標籤 → 公平比較原生類別數不同的模型
"""

ABSTAIN = "__abstain__"


def apply(predictions: list, mapping: dict) -> list:
    """回傳每筆的映射結果，棄權為 None。"""
    out = []
    for p in predictions:
        raw = p["raw_label"]
        if raw not in mapping:
            raise KeyError(
                f"映射表沒有涵蓋原生類別 '{raw}'。"
                "設定檔的 mappings 鍵必須與 native_labels 完全一致。"
            )
        out.append(mapping[raw])
    return out


def build_views(predictions: list, mapping: dict, labels: list, covers: list) -> dict:
    """把 predictions 切成三個視角，各自給出可直接算指標的 (y_true, y_pred, labels)。"""
    mapped = apply(predictions, mapping)
    y_true_all = [p["true_label"] for p in predictions]
    conf_all = [p["confidence"] for p in predictions]

    # --- strict：全部樣本，棄權換成一個永遠不會命中的標籤 --------------------
    strict = {
        "name": "strict",
        "description": "棄權一律算答錯（產品實際會拿到的表現）",
        "y_true": list(y_true_all),
        "y_pred": [m if m is not None else ABSTAIN for m in mapped],
        "confidence": list(conf_all),
        "labels": list(labels),
        "n": len(predictions),
        "coverage": None,
    }

    # --- covered：只留有映射的樣本 -------------------------------------------
    keep = [i for i, m in enumerate(mapped) if m is not None]
    covered = {
        "name": "covered",
        "description": "只計算有映射的子集，另報 coverage",
        "y_true": [y_true_all[i] for i in keep],
        "y_pred": [mapped[i] for i in keep],
        "confidence": [conf_all[i] for i in keep],
        "labels": list(labels),
        "n": len(keep),
        "coverage": len(keep) / len(predictions) if predictions else 0.0,
    }

    # --- subset：只留模型宣告表達得出的標籤 ----------------------------------
    # 若模型沒有原生 fear，true_label 為 fear 的樣本它註定答錯，Macro-F1 會被硬拉低
    # 六分之一。拿這種數字跟有 fear 的模型比並不公平，所以另給一個排除後的視角。
    covers = list(covers)
    keep2 = [i for i, t in enumerate(y_true_all) if t in covers]
    subset = {
        "name": "subset",
        "description": f"只算模型宣告表達得出的 {len(covers)} 類：{', '.join(covers)}",
        "y_true": [y_true_all[i] for i in keep2],
        "y_pred": [mapped[i] if mapped[i] is not None else ABSTAIN for i in keep2],
        "confidence": [conf_all[i] for i in keep2],
        "labels": covers,
        "n": len(keep2),
        "coverage": None,
    }

    return {"strict": strict, "covered": covered, "subset": subset}
