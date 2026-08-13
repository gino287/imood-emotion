"""信心值分析的共用函式：校準、選擇性預測、錯誤偵測、溫度縮放。

⚠️ 本檔案（以及 research/ 底下所有程式）的註解是 AI 起草的，尚未經 Gino 改寫。
   若其中任何邏輯要搬進 imood_emotion/ 或 eval/，依 CLAUDE.md 規則 4 需先自行重寫註解。

只依賴 numpy，理由跟 eval/evalkit/metrics.py 一樣：這些都是短公式，
自己寫可以不用把 scipy/sklearn 拉進 image。
"""
import json
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 不確定度分數（把同一組機率向量換算成不同的「有多不確定」指標）
# ---------------------------------------------------------------------------


def msp(probs: np.ndarray) -> np.ndarray:
    """Maximum Softmax Probability：最大那一格的機率，就是一般講的 confidence。"""
    return probs.max(axis=1)


def margin(probs: np.ndarray) -> np.ndarray:
    """第一名減第二名。兩類打平時 MSP 可能還有 0.5，但 margin 會趨近 0，
    對「模型在猶豫」這件事比 MSP 敏感。"""
    s = np.sort(probs, axis=1)
    return s[:, -1] - s[:, -2]


def neg_entropy(probs: np.ndarray) -> np.ndarray:
    """負熵。熵看的是整個分布有多平，不是只看前兩名，
    對「八類全部都有一點點」這種情況比 margin 敏感。取負號讓「越大越有信心」一致。"""
    p = np.clip(probs, 1e-12, 1.0)
    return (p * np.log(p)).sum(axis=1)


def normalized_neg_entropy(probs: np.ndarray) -> np.ndarray:
    """把熵除以 log(K) 正規化到 0~1，跨不同類別數的模型才比得起來。"""
    k = probs.shape[1]
    return neg_entropy(probs) / math.log(k)


# ---------------------------------------------------------------------------
# 校準（confidence 講的機率準不準）
# ---------------------------------------------------------------------------


def ece_equal_width(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> dict:
    """等寬分箱 ECE（Guo et al. 2017 的原始版本）。

    弱點：分數集中在 0.9~1.0 時，大部分樣本擠在最後一兩個箱，
    前面的箱幾乎是空的，ECE 對 bins 數很敏感。所以下面另外做等質量版。
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    n = len(conf)
    ece, mce, rows = 0.0, 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi) if hi < 1.0 else (conf >= lo) & (conf <= 1.0)
        cnt = int(in_bin.sum())
        if cnt == 0:
            rows.append({"lo": round(lo, 3), "hi": round(hi, 3), "count": 0,
                         "acc": None, "conf": None, "gap": None})
            continue
        acc = float(correct[in_bin].mean())
        avg = float(conf[in_bin].mean())
        gap = abs(acc - avg)
        ece += cnt / n * gap
        mce = max(mce, gap)
        rows.append({"lo": round(lo, 3), "hi": round(hi, 3), "count": cnt,
                     "acc": round(acc, 4), "conf": round(avg, 4),
                     "acc_minus_conf": round(acc - avg, 4)})
    return {"ece": round(ece, 4), "mce": round(mce, 4), "bins": bins, "reliability": rows}


def ece_equal_mass(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> dict:
    """等質量分箱（adaptive ECE）：每箱樣本數一樣多，不會有空箱。
    分數分布極度偏斜時比等寬版可靠。"""
    n = len(conf)
    order = np.argsort(conf)
    c, k = conf[order], correct[order]
    ece, mce, rows = 0.0, 0.0, []
    edges = np.linspace(0, n, bins + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        acc = float(k[a:b].mean())
        avg = float(c[a:b].mean())
        gap = abs(acc - avg)
        ece += (b - a) / n * gap
        mce = max(mce, gap)
        rows.append({"count": b - a, "conf_lo": round(float(c[a]), 4),
                     "conf_hi": round(float(c[b - 1]), 4),
                     "acc": round(acc, 4), "conf": round(avg, 4),
                     "acc_minus_conf": round(acc - avg, 4)})
    return {"ece": round(ece, 4), "mce": round(mce, 4), "bins": bins, "reliability": rows}


def brier_multiclass(probs: np.ndarray, true_idx: np.ndarray) -> float:
    """多類 Brier：整個機率向量跟 one-hot 的平方距離。
    ECE 只看最高分那一格，Brier 看整個分布，兩者一起看比較不會被騙。"""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(true_idx)), true_idx] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def nll(probs: np.ndarray, true_idx: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(true_idx)), true_idx], 1e-12, 1.0)
    return float(-np.log(p).mean())


# ---------------------------------------------------------------------------
# 溫度縮放（唯一一個「不用重訓、只調一個純量」的校準法）
# ---------------------------------------------------------------------------


def probs_to_logits(probs: np.ndarray) -> np.ndarray:
    """我們手上只有存下來的機率，沒有原始 logits。
    softmax 對加常數不變，所以 log(p) 就是一組合法的等價 logits，
    溫度縮放只需要相對值，這樣做不影響結果。"""
    return np.log(np.clip(probs, 1e-12, 1.0))


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, true_idx: np.ndarray,
                    lo: float = 0.05, hi: float = 10.0, iters: int = 60) -> float:
    """用三分搜尋找讓 NLL 最小的 T。

    NLL 對 T 是單峰的，所以三分搜尋夠用，不需要 scipy 的最佳化器。
    T>1 代表原本過度自信（要把分數壓平）；T<1 代表原本不夠自信。
    """
    def loss(T):
        return nll(apply_temperature(logits, T), true_idx)
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if loss(m1) < loss(m2):
            hi = m2
        else:
            lo = m1
    return round((lo + hi) / 2, 4)


# ---------------------------------------------------------------------------
# 選擇性預測（要不要設門檻、設多少）
# ---------------------------------------------------------------------------


def risk_coverage(score: np.ndarray, correct: np.ndarray) -> dict:
    """依分數由高到低排序，逐步放寬覆蓋率，記錄「留下來這批的錯誤率」。

    AURC = 曲線下面積，越小代表分數把對錯排得越開。
    另外算 AURC_oracle（完美排序）與 AURC_random 當上下界，
    單看 AURC 數字沒有尺度感，要有這兩個參考點才知道好壞。
    """
    n = len(score)
    order = np.argsort(-score)
    ok = correct[order].astype(float)
    cum_err = np.cumsum(1.0 - ok)
    cov = np.arange(1, n + 1) / n
    risk = cum_err / np.arange(1, n + 1)
    aurc = float(risk.mean())

    # oracle：所有答對的排前面
    ok_oracle = np.sort(correct.astype(float))[::-1]
    risk_oracle = np.cumsum(1.0 - ok_oracle) / np.arange(1, n + 1)
    base_err = 1.0 - float(correct.mean())

    curve = []
    for target in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        i = max(0, min(n - 1, int(round(target * n)) - 1))
        curve.append({"coverage": target, "n_kept": i + 1,
                      "selective_risk": round(float(risk[i]), 4),
                      "selective_acc": round(1 - float(risk[i]), 4),
                      "score_threshold": round(float(score[order][i]), 4)})
    return {
        "aurc": round(aurc, 4),
        "aurc_oracle": round(float(risk_oracle.mean()), 4),
        "aurc_random": round(base_err, 4),
        "eaurc": round(aurc - float(risk_oracle.mean()), 4),
        "full_coverage_risk": round(base_err, 4),
        "curve": curve,
    }


def threshold_table(score: np.ndarray, correct: np.ndarray, taus=None) -> list:
    """實務上真正要看的表：門檻設 τ → 留下幾成、留下的準確率、被丟掉那批的準確率。

    被丟掉那批的準確率很重要：如果丟掉的那批準確率也不低，
    代表這個門檻在浪費可用的預測，不是在濾掉爛的。
    """
    if taus is None:
        taus = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    n = len(score)
    out = []
    for t in taus:
        keep = score >= t
        k = int(keep.sum())
        out.append({
            "tau": t,
            "coverage": round(k / n, 4),
            "kept": k,
            "kept_acc": round(float(correct[keep].mean()), 4) if k else None,
            "dropped": n - k,
            "dropped_acc": round(float(correct[~keep].mean()), 4) if k < n else None,
        })
    return out


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """用 Mann-Whitney U 直接算 AUROC，不用畫 ROC。

    這裡的用法是「錯誤偵測」：positive = 預測正確，score = 信心。
    AUROC=0.5 表示信心完全分不出對錯；越接近 1 表示越分得開。
    """
    pos = score[positive.astype(bool)]
    neg = score[~positive.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    # 處理同分：同分的名次取平均，否則 AUROC 會被高估
    order = np.argsort(allv)
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    return round(float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 4)


def aupr_error(score: np.ndarray, correct: np.ndarray) -> float:
    """AUPR-Error：把「答錯」當正類、用 -confidence 當分數。
    錯誤率低的時候 AUROC 看起來都很漂亮，AUPR 對不平衡比較誠實。"""
    s = -score
    y = (~correct.astype(bool)).astype(int)
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    total_pos = y.sum()
    if total_pos == 0:
        return float("nan")
    return round(float((prec * y).sum() / total_pos), 4)


# ---------------------------------------------------------------------------
# 讀檔
# ---------------------------------------------------------------------------


def read_jsonl(path) -> list:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def probs_matrix(records: list, key: str, label_order: list) -> np.ndarray:
    return np.array([[r[key][lab] for lab in label_order] for r in records], dtype=float)
