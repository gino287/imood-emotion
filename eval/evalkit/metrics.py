# 待GINO改寫
"""全部指標（規劃書 v2 §4.3）。

刻意只依賴 numpy，不用 scikit-learn：這幾個公式都是教科書等級的短函式，
自己寫可以少一個含 scipy 的重量級依賴（image 小幾百 MB），
而且在沒有完整容器環境時也跑得起來。代價是要自己頂著正確性，
所以每個函式都有手算得出答案的對應檢查：scripts/selftest.py。
"""
from collections import Counter

import numpy as np

from . import mapping as mp
from .mapping import ABSTAIN


# ---------------------------------------------------------------------------
# 基本準確度指標
# ---------------------------------------------------------------------------
def confusion_matrix(y_true: list, y_pred: list, labels: list) -> dict:
    """回傳 {真實標籤: {預測標籤: 次數}}。棄權自成一欄，方便看它吃掉了哪些類。"""
    pred_labels = list(labels)
    if ABSTAIN in y_pred:
        pred_labels = pred_labels + [ABSTAIN]
    matrix = {t: {p: 0 for p in pred_labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1
    return {"rows": labels, "cols": pred_labels, "matrix": matrix}


def per_class(y_true: list, y_pred: list, labels: list) -> dict:
    out = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    return out


def accuracy(y_true: list, y_pred: list) -> float:
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def macro_f1(per_class_scores: dict) -> float:
    if not per_class_scores:
        return 0.0
    return sum(v["f1"] for v in per_class_scores.values()) / len(per_class_scores)


def evaluate_view(view: dict) -> dict:
    """一個視角的完整準確度指標。"""
    pc = per_class(view["y_true"], view["y_pred"], view["labels"])
    correct = [t == p for t, p in zip(view["y_true"], view["y_pred"])]
    return {
        "name": view["name"],
        "description": view["description"],
        "n": view["n"],
        "coverage": round(view["coverage"], 4) if view["coverage"] is not None else None,
        "accuracy": round(accuracy(view["y_true"], view["y_pred"]), 4),
        "macro_f1": round(macro_f1(pc), 4),
        "per_class": pc,
        "confusion": confusion_matrix(view["y_true"], view["y_pred"], view["labels"]),
        "calibration": calibration(view["confidence"], correct),
    }


# ---------------------------------------------------------------------------
# mapping-free 指標（規劃書 v2 §4.3(B)）
# 完全不看映射表，直接從「原生類別 × 六類標籤」的列聯表算。
# 這是「厭惡→fear 到底成不成立」能用數字結案的關鍵。
# ---------------------------------------------------------------------------
def contingency(raw_labels: list, true_labels: list) -> dict:
    table = Counter(zip(raw_labels, true_labels))
    raws = sorted({r for r, _ in table})
    trues = sorted({t for _, t in table})
    return {
        "raw_classes": raws,
        "true_classes": trues,
        "counts": {r: {t: table.get((r, t), 0) for t in trues} for r in raws},
    }


def best_possible_mapping(raw_labels: list, true_labels: list) -> dict:
    """每個原生類別指派給它命中最多的真實標籤（many-to-one），算出準確率天花板。

    與手訂映射的準確率一比，差距就是「映射表猜錯造成的損失」——
    一個數字說完，不需要看圓餅圖辯論。
    """
    ct = contingency(raw_labels, true_labels)
    mapping, hits = {}, 0
    for raw in ct["raw_classes"]:
        row = ct["counts"][raw]
        best = max(row, key=row.get)
        mapping[raw] = best
        hits += row[best]
    n = len(raw_labels)
    return {
        "mapping": mapping,
        "accuracy_upper_bound": round(hits / n, 4) if n else 0.0,
    }


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def normalized_mutual_info(a: list, b: list) -> float:
    """NMI，算術平均正規化（與 sklearn 的 average_method='arithmetic' 一致）。

    完全不看映射，只衡量兩種分類方式的結構相關性。類別數不同時
    accuracy 與 F1 本來就不可比，這是唯一能公平比較
    「8 類模型 vs 28 類模型」的指標。
    """
    a_vals, b_vals = sorted(set(a)), sorted(set(b))
    ai = {v: i for i, v in enumerate(a_vals)}
    bi = {v: i for i, v in enumerate(b_vals)}
    m = np.zeros((len(a_vals), len(b_vals)))
    for x, y in zip(a, b):
        m[ai[x], bi[y]] += 1

    n = m.sum()
    if n == 0:
        return 0.0
    h_a, h_b = _entropy(m.sum(axis=1)), _entropy(m.sum(axis=0))
    if h_a == 0 or h_b == 0:
        return 0.0

    pij = m / n
    pi = pij.sum(axis=1, keepdims=True)
    pj = pij.sum(axis=0, keepdims=True)
    nz = pij > 0
    mi = float((pij[nz] * np.log(pij[nz] / (pi @ pj)[nz])).sum())
    return round(mi / ((h_a + h_b) / 2), 4)


def adjusted_rand_index(a: list, b: list) -> float:
    a_vals, b_vals = sorted(set(a)), sorted(set(b))
    ai = {v: i for i, v in enumerate(a_vals)}
    bi = {v: i for i, v in enumerate(b_vals)}
    m = np.zeros((len(a_vals), len(b_vals)))
    for x, y in zip(a, b):
        m[ai[x], bi[y]] += 1

    def comb2(x):
        return x * (x - 1) / 2

    n = m.sum()
    if n < 2:
        return 0.0
    index = comb2(m).sum()
    sum_a = comb2(m.sum(axis=1)).sum()
    sum_b = comb2(m.sum(axis=0)).sum()
    expected = sum_a * sum_b / comb2(n)
    maximum = (sum_a + sum_b) / 2
    if maximum == expected:
        return 0.0
    return round(float((index - expected) / (maximum - expected)), 4)


def mapping_support(raw_labels: list, true_labels: list, mapping: dict) -> list:
    """每一組映射的雙向條件機率。

    P(true=X | raw=Y)：判成 Y 的句子裡，真的有多少是 X   → 這組映射「準不準」
    P(raw=Y | true=X)：真的是 X 的句子裡，有多少被判成 Y → 這組映射「抓不抓得到」

    前者若只有 15%，這組映射就是純粹湊數，不必再討論。
    """
    ct = contingency(raw_labels, true_labels)
    raw_totals = Counter(raw_labels)
    true_totals = Counter(true_labels)

    rows = []
    for raw, target in sorted(mapping.items()):
        n_raw = raw_totals.get(raw, 0)
        row = ct["counts"].get(raw, {})
        if target is None:
            rows.append({
                "raw": raw, "target": None, "n_raw": n_raw,
                "precision": None, "recall": None,
                "note": "棄權",
            })
            continue
        hit = row.get(target, 0)
        n_true = true_totals.get(target, 0)
        rows.append({
            "raw": raw,
            "target": target,
            "n_raw": n_raw,
            "n_true": n_true,
            "precision": round(hit / n_raw, 4) if n_raw else None,   # P(true=target | raw)
            "recall": round(hit / n_true, 4) if n_true else None,    # P(raw | true=target)
            "hit": hit,
        })
    return rows


def bidirectional_distribution(raw_labels: list, true_labels: list) -> dict:
    """儀表板用的雙向分佈（規劃書 v2 §4.4）。

    方向一：某個六類標籤的句子，被判成哪些原生類別
    方向二：某個原生類別的句子，實際上屬於哪幾個六類標籤
    """
    ct = contingency(raw_labels, true_labels)
    by_true, by_raw = {}, {}

    for t in ct["true_classes"]:
        dist = {r: ct["counts"][r].get(t, 0) for r in ct["raw_classes"]}
        total = sum(dist.values())
        by_true[t] = {
            "total": total,
            "dist": {k: v for k, v in sorted(dist.items(), key=lambda x: -x[1]) if v},
        }

    for r in ct["raw_classes"]:
        dist = {t: v for t, v in ct["counts"][r].items()}
        total = sum(dist.values())
        by_raw[r] = {
            "total": total,
            "dist": {k: v for k, v in sorted(dist.items(), key=lambda x: -x[1]) if v},
        }

    return {"by_true_label": by_true, "by_raw_label": by_raw}


# ---------------------------------------------------------------------------
# 時間指標
# ---------------------------------------------------------------------------
def latency_stats(predictions: list) -> dict:
    """p95 才是「使用者會不會感覺到卡頓」的數字，平均值會被大量短句拉低。"""
    def stats(values):
        arr = np.asarray(values, dtype=float)
        return {
            "mean": round(float(arr.mean()), 3),
            "p50": round(float(np.percentile(arr, 50)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "p99": round(float(np.percentile(arr, 99)), 3),
            "min": round(float(arr.min()), 3),
            "max": round(float(arr.max()), 3),
        }

    out = {}
    for part in ("total", "tokenize", "forward", "post"):
        values = [p["latency_ms"][part] for p in predictions if part in p["latency_ms"]]
        if values:
            out[part] = stats(values)
    out["histogram"] = _histogram([p["latency_ms"]["total"] for p in predictions])
    return out


def _histogram(values: list, bins: int = 30) -> dict:
    arr = np.asarray(values, dtype=float)
    counts, edges = np.histogram(arr, bins=bins)
    return {
        "counts": counts.tolist(),
        "edges": [round(float(e), 3) for e in edges],
    }


# ---------------------------------------------------------------------------
# 信心校準（規劃書 v2 §4.3(D)）
# ---------------------------------------------------------------------------
def calibration(confidences: list, correct: list, bins: int = 10) -> dict:
    """ECE + reliability curve + 門檻覆蓋率表。

    目的很明確：判斷要不要加 confidence threshold。所以直接輸出
    「設 X 門檻可以擋掉多少句、剩下的準確率變多少」這種可決策的形式。
    """
    if not confidences:
        return {"ece": None, "reliability": [], "thresholds": []}

    conf = np.asarray(confidences, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    n = len(conf)

    reliability, ece = [], 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        # 最後一個 bin 含右端點，否則 confidence 剛好 1.0 的樣本會漏掉
        in_bin = (conf > lo) & (conf <= hi) if hi < 1.0 else (conf > lo) & (conf <= 1.0)
        count = int(in_bin.sum())
        if count == 0:
            reliability.append({"lo": round(float(lo), 2), "hi": round(float(hi), 2),
                                "count": 0, "accuracy": None, "avg_confidence": None})
            continue
        acc = float(ok[in_bin].mean())
        avg_conf = float(conf[in_bin].mean())
        ece += count / n * abs(acc - avg_conf)
        reliability.append({
            "lo": round(float(lo), 2), "hi": round(float(hi), 2), "count": count,
            "accuracy": round(acc, 4), "avg_confidence": round(avg_conf, 4),
        })

    thresholds = []
    for th in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        keep = conf >= th
        kept = int(keep.sum())
        thresholds.append({
            "threshold": th,
            "coverage": round(kept / n, 4),
            "kept": kept,
            "accuracy": round(float(ok[keep].mean()), 4) if kept else None,
        })

    return {
        "ece": round(ece, 4),
        "mean_confidence": round(float(conf.mean()), 4),
        "overall_accuracy": round(float(ok.mean()), 4),
        "reliability": reliability,
        "thresholds": thresholds,
    }


# ---------------------------------------------------------------------------
# 彙總
# ---------------------------------------------------------------------------
def evaluate(predictions: list, model_cfg: dict, labels: list, mapping_name: str) -> dict:
    mapping = model_cfg["mappings"][mapping_name]
    views = mp.build_views(predictions, mapping, labels, model_cfg["covers"])

    raw_labels = [p["raw_label"] for p in predictions]
    true_labels = [p["true_label"] for p in predictions]
    best = best_possible_mapping(raw_labels, true_labels)
    strict_acc = accuracy(views["strict"]["y_true"], views["strict"]["y_pred"])

    return {
        "mapping_name": mapping_name,
        "mapping": mapping,
        "views": {name: evaluate_view(v) for name, v in views.items()},
        "mapping_free": {
            "best_possible": best,
            "current_strict_accuracy": round(strict_acc, 4),
            # 這個差距就是「映射表猜錯造成的損失」
            "loss_from_mapping_choice": round(best["accuracy_upper_bound"] - strict_acc, 4),
            "nmi": normalized_mutual_info(raw_labels, true_labels),
            "ari": adjusted_rand_index(raw_labels, true_labels),
            "support": mapping_support(raw_labels, true_labels, mapping),
        },
        "bidirectional": bidirectional_distribution(raw_labels, true_labels),
        "latency": latency_stats(predictions),
    }
