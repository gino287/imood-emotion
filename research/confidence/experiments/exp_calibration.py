"""實驗：這個模型的 confidence 到底代表什麼、能不能拿來做門檻。

三份既有資料，不用重跑模型（機率向量當初都存下來了，這是「儲存策略：存完整
機率分布」那個決策現在開始回本的地方）：

  A. 同領域   _local/out/stream_cpu.jsonl        200 句，原生 8 類，
                                                 與模型同作者的對話資料集
  B. 跨領域   eval/results/.../predictions.jsonl 1200 句 SMP2020 微博，
                                                 6 類，要經過映射
  C. 截斷壓力 _local/stress/fragment_detail.jsonl 200 對（完整句 vs 被切斷的句）

每份都算：分布 → 校準（ECE/Brier/NLL）→ 溫度縮放 → 選擇性預測（風險-覆蓋率）
→ 錯誤偵測（AUROC）。B 另外比較「原生最大機率」與「映射後合併機率」哪個當
confidence 比較準 —— 這件事直接影響給下游的 {ts, text, emotion, confidence}。

    python research/confidence/experiments/exp_calibration.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conflib import (apply_temperature, auroc, aupr_error, brier_multiclass,  # noqa: E402
                     ece_equal_mass, ece_equal_width, fit_temperature, margin,
                     msp, neg_entropy, nll, probs_to_logits, read_jsonl,
                     risk_coverage, threshold_table)

ROOT = Path(__file__).resolve().parents[3]
NATIVE = ["平淡語氣", "關切語調", "開心語調", "憤怒語調", "悲傷語調", "疑問語調", "驚奇語調", "厭惡語調"]


def score_block(probs, correct, true_idx, name):
    """一份資料共通的整套指標。"""
    conf = msp(probs)
    logits = probs_to_logits(probs)

    # 溫度縮放要有獨立的校準集，不然是拿同一批資料自己調自己考
    n = len(probs)
    rng = np.random.default_rng(20260813)
    idx = rng.permutation(n)
    half = n // 2
    cal, test = idx[:half], idx[half:]
    T_fit = fit_temperature(logits[cal], true_idx[cal])
    probs_T = apply_temperature(logits, T_fit)
    conf_T = msp(probs_T)

    out = {
        "name": name,
        "n": int(n),
        "accuracy": round(float(correct.mean()), 4),
        "confidence": {
            "mean": round(float(conf.mean()), 4),
            "p05": round(float(np.percentile(conf, 5)), 4),
            "p25": round(float(np.percentile(conf, 25)), 4),
            "p50": round(float(np.percentile(conf, 50)), 4),
            "p75": round(float(np.percentile(conf, 75)), 4),
            "p95": round(float(np.percentile(conf, 95)), 4),
            "frac_above_0.9": round(float((conf >= 0.9).mean()), 4),
            "frac_above_0.99": round(float((conf >= 0.99).mean()), 4),
            "frac_below_0.5": round(float((conf < 0.5).mean()), 4),
        },
        "gap_conf_minus_acc": round(float(conf.mean() - correct.mean()), 4),
        "calibration": {
            "ece_equal_width_15": ece_equal_width(conf, correct, 15),
            "ece_equal_mass_15": ece_equal_mass(conf, correct, 15),
            "brier": round(brier_multiclass(probs, true_idx), 4),
            "nll": round(nll(probs, true_idx), 4),
        },
        "temperature_scaling": {
            "T_fitted_on_half": T_fit,
            "T_fitted_on_all": fit_temperature(logits, true_idx),
            "heldout": {
                "ece_before": ece_equal_mass(conf[test], correct[test], 10)["ece"],
                "ece_after": ece_equal_mass(conf_T[test], correct[test], 10)["ece"],
                "nll_before": round(nll(probs[test], true_idx[test]), 4),
                "nll_after": round(nll(probs_T[test], true_idx[test]), 4),
                "mean_conf_before": round(float(conf[test].mean()), 4),
                "mean_conf_after": round(float(conf_T[test].mean()), 4),
                "accuracy": round(float(correct[test].mean()), 4),
            },
            "note": "溫度縮放只會單調地壓平/拉尖分數，不改變 argmax，準確率一定不變",
        },
        "selective_prediction": {
            "msp": risk_coverage(conf, correct),
            "threshold_table_msp": threshold_table(conf, correct),
        },
        "error_detection": {
            "auroc_msp": auroc(conf, correct),
            "auroc_margin": auroc(margin(probs), correct),
            "auroc_neg_entropy": auroc(neg_entropy(probs), correct),
            "aupr_error_msp": aupr_error(conf, correct),
            "note": "AUROC=0.5 表示信心完全分不出對錯；正類=預測正確",
        },
    }
    return out


def per_class_block(pred_idx, true_idx, conf, labels):
    """逐類看：這一類的平均信心 vs 這一類的實際準確率。
    整體 ECE 很小不代表每一類都準，可能是高估的類跟低估的類互相抵消。"""
    out = {}
    for i, lab in enumerate(labels):
        m = pred_idx == i
        if m.sum() == 0:
            continue
        acc = float((true_idx[m] == i).mean())
        out[lab] = {
            "n_predicted": int(m.sum()),
            "mean_confidence": round(float(conf[m].mean()), 4),
            "precision": round(acc, 4),
            "conf_minus_acc": round(float(conf[m].mean()) - acc, 4),
        }
    return out


def dataset_a():
    """同領域：模型作者自己的對話資料集，原生 8 類直接比對。"""
    recs = read_jsonl(ROOT / "_local/out/stream_cpu.jsonl")
    recs = [r for r in recs if r.get("dataset_label")]
    probs = np.array([[r["probs"][lab] for lab in NATIVE] for r in recs])
    pred_idx = probs.argmax(axis=1)
    true_idx = np.array([NATIVE.index(r["dataset_label"]) for r in recs])
    correct = (pred_idx == true_idx)
    block = score_block(probs, correct, true_idx, "A_同領域_原生8類_200句")
    block["per_class"] = per_class_block(pred_idx, true_idx, msp(probs), NATIVE)
    return block


def dataset_b(variant="zh_tw", mapping_name="draft"):
    """跨領域：SMP2020 微博 1200 句，模型沒看過這個領域。"""
    cfg = yaml.safe_load((ROOT / "eval/configs/models.yaml").read_text(encoding="utf-8"))
    mapping = cfg["models"][0]["mappings"][mapping_name]
    six = cfg["labels"]

    recs = read_jsonl(ROOT / f"eval/results/johnson-small/cpu/{variant}/predictions.jsonl")
    probs8 = np.array([[r["raw_probs"][lab] for lab in NATIVE] for r in recs])
    raw_pred = [NATIVE[i] for i in probs8.argmax(axis=1)]
    mapped_pred = [mapping.get(p) for p in raw_pred]
    true6 = [r["true_label"] for r in recs]

    # 映射把 8 類壓成 6 類，同一個 6 類目標可能吃到好幾個原生類的機率。
    # 這裡把它們加起來 —— 這才是模型對「這句是 neutral」的真正把握，
    # 原生最大機率只是其中一塊。
    probs6 = np.zeros((len(recs), len(six)))
    for j, nat in enumerate(NATIVE):
        tgt = mapping.get(nat)
        if tgt is None:
            continue
        probs6[:, six.index(tgt)] += probs8[:, j]

    keep = np.array([p is not None for p in mapped_pred])
    true_idx6 = np.array([six.index(t) for t in true6])
    correct = np.array([mp == t for mp, t in zip(mapped_pred, true6)])

    conf_raw = msp(probs8)
    conf_mapped = probs6[np.arange(len(recs)), probs6.argmax(axis=1)]
    # 合併後 argmax 有可能跟「原生 argmax 再映射」不同 —— 這本身是個發現
    agree = np.array([six[i] == mp for i, mp in zip(probs6.argmax(axis=1), mapped_pred)])
    correct_mapped_argmax = np.array(
        [six[i] == t for i, t in zip(probs6.argmax(axis=1), true6)])

    block = score_block(probs8, correct, np.full(len(recs), -1), f"B_跨領域_SMP2020_{variant}_{mapping_name}")
    # 8 類機率配 6 類 gold，Brier/NLL 算不出來，蓋掉避免誤讀
    block["calibration"]["brier"] = None
    block["calibration"]["nll"] = None
    block["temperature_scaling"] = {
        "note": "gold 是 6 類、機率是 8 類，NLL 定義不了，溫度縮放改在下面 mapped 版做"
    }

    logits6 = probs_to_logits(probs6)
    rng = np.random.default_rng(20260813)
    idx = rng.permutation(len(recs))
    cal, test = idx[:len(recs) // 2], idx[len(recs) // 2:]
    T = fit_temperature(logits6[cal], true_idx6[cal])
    probs6_T = apply_temperature(logits6, T)

    block["mapped_confidence"] = {
        "說明": "把映射到同一個 6 類目標的原生機率加總後，再取最大值當 confidence",
        "mean_conf_raw_max": round(float(conf_raw.mean()), 4),
        "mean_conf_mapped_sum": round(float(conf_mapped.mean()), 4),
        "accuracy_via_raw_argmax_then_map": round(float(correct.mean()), 4),
        "accuracy_via_mapped_argmax": round(float(correct_mapped_argmax.mean()), 4),
        "argmax_disagreement_rate": round(float(1 - agree.mean()), 4),
        "ece_raw_max": ece_equal_mass(conf_raw, correct, 15)["ece"],
        "ece_mapped_sum": ece_equal_mass(conf_mapped, correct_mapped_argmax, 15)["ece"],
        "auroc_raw_max": auroc(conf_raw, correct),
        "auroc_mapped_sum": auroc(conf_mapped, correct_mapped_argmax),
        "aurc_raw_max": risk_coverage(conf_raw, correct)["aurc"],
        "aurc_mapped_sum": risk_coverage(conf_mapped, correct_mapped_argmax)["aurc"],
        "brier_mapped": round(brier_multiclass(probs6, true_idx6), 4),
        "nll_mapped": round(nll(probs6, true_idx6), 4),
        "temperature": {
            "T": T,
            "ece_before": ece_equal_mass(msp(probs6[test]), correct_mapped_argmax[test], 10)["ece"],
            "ece_after": ece_equal_mass(msp(probs6_T[test]), correct_mapped_argmax[test], 10)["ece"],
            "nll_before": round(nll(probs6[test], true_idx6[test]), 4),
            "nll_after": round(nll(probs6_T[test], true_idx6[test]), 4),
        },
        "threshold_table_mapped": threshold_table(conf_mapped, correct_mapped_argmax),
    }
    block["per_class_raw"] = per_class_block(
        probs8.argmax(axis=1), np.full(len(recs), -1), conf_raw, NATIVE)
    # 上面那個 true_idx 是假的，precision 沒有意義，只留信心分布
    for v in block["per_class_raw"].values():
        v.pop("precision", None)
        v.pop("conf_minus_acc", None)
    block["per_class_mapped"] = per_class_block(
        probs6.argmax(axis=1), true_idx6, conf_mapped, six)
    block["coverage_of_mapping"] = round(float(keep.mean()), 4)
    return block


def dataset_c():
    """截斷壓力：同一句的完整版 vs 被切斷版，信心怎麼變、類別有沒有翻掉。"""
    recs = read_jsonl(ROOT / "_local/stress/fragment_detail.jsonl")
    full_conf = np.array([r["full_conf"] for r in recs])
    frag_conf = np.array([r["frag_conf"] for r in recs])
    flipped = np.array([r["flipped"] for r in recs])
    keep = np.array([r["keep_ratio"] for r in recs])

    # 核心問題：拿「片段的信心」當訊號，能不能認出這句被切斷了？
    # 正類設成「沒翻掉」，跟前面錯誤偵測的方向一致：AUROC 越高代表越分得開。
    out = {
        "name": "C_截斷壓力_200對",
        "n": len(recs),
        "flip_rate": round(float(flipped.mean()), 4),
        "mean_conf_full": round(float(full_conf.mean()), 4),
        "mean_conf_frag": round(float(frag_conf.mean()), 4),
        "conf_drop_abs": round(float(full_conf.mean() - frag_conf.mean()), 4),
        "conf_drop_pct": round(float(1 - frag_conf.mean() / full_conf.mean()) * 100, 2),
        "mean_conf_frag_when_flipped": round(float(frag_conf[flipped].mean()), 4),
        "mean_conf_frag_when_not_flipped": round(float(frag_conf[~flipped].mean()), 4),
        "auroc_frag_conf_detects_flip": auroc(frag_conf, ~flipped),
        "auroc_conf_delta_detects_flip": auroc(frag_conf - full_conf, ~flipped),
        "threshold_table_flip": threshold_table(frag_conf, ~flipped),
        "high_conf_but_flipped": {
            "conf>=0.9": int(((frag_conf >= 0.9) & flipped).sum()),
            "conf>=0.95": int(((frag_conf >= 0.95) & flipped).sum()),
            "conf>=0.99": int(((frag_conf >= 0.99) & flipped).sum()),
        },
        "by_keep_ratio": {},
    }
    for lo, hi in [(0.0, 0.35), (0.35, 0.55), (0.55, 0.75), (0.75, 1.01)]:
        m = (keep >= lo) & (keep < hi)
        if m.sum() == 0:
            continue
        out["by_keep_ratio"][f"{lo}-{hi}"] = {
            "n": int(m.sum()),
            "flip_rate": round(float(flipped[m].mean()), 4),
            "mean_frag_conf": round(float(frag_conf[m].mean()), 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/confidence/results/calibration.json")
    args = ap.parse_args()

    results = {
        "A_in_domain": dataset_a(),
        "B_cross_domain_zh_tw": dataset_b("zh_tw", "draft"),
        "B_cross_domain_zh_cn": dataset_b("zh_cn", "draft"),
        "B_cross_domain_to_angry": dataset_b("zh_tw", "to_angry"),
        "C_truncation": dataset_c(),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    # numpy 的 int64/float64 不是 json 原生型別，統一轉一次再寫
    def to_py(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(f"不會轉的型別：{type(o)}")

    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=to_py),
                   encoding="utf-8")

    for key in ["A_in_domain", "B_cross_domain_zh_tw"]:
        b = results[key]
        print(f"\n=== {b['name']} ===")
        print(f"  準確率 {b['accuracy']:.3f} / 平均信心 {b['confidence']['mean']:.3f} "
              f"→ 高估 {b['gap_conf_minus_acc']:+.3f}")
        print(f"  ECE(等寬) {b['calibration']['ece_equal_width_15']['ece']} / "
              f"ECE(等質量) {b['calibration']['ece_equal_mass_15']['ece']}")
        print(f"  錯誤偵測 AUROC: msp={b['error_detection']['auroc_msp']} "
              f"margin={b['error_detection']['auroc_margin']} "
              f"negent={b['error_detection']['auroc_neg_entropy']}")
        rc = b["selective_prediction"]["msp"]
        print(f"  AURC {rc['aurc']} (oracle {rc['aurc_oracle']} / 全收 {rc['aurc_random']})")

    c = results["C_truncation"]
    print(f"\n=== {c['name']} ===")
    print(f"  翻類率 {c['flip_rate']:.3f} / 信心只掉 {c['conf_drop_pct']}%")
    print(f"  片段信心偵測翻類的 AUROC = {c['auroc_frag_conf_detects_flip']}")
    print(f"  信心≥0.9 卻翻類的句數：{c['high_conf_but_flipped']['conf>=0.9']}")
    print(f"\n寫出 {out}")


if __name__ == "__main__":
    main()
