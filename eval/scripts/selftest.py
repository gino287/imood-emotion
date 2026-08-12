"""自我檢查：用手算得出答案的小例子驗證前處理、映射與指標。

  python scripts/selftest.py

指標是自己用 numpy 實作的（沒有依賴 scikit-learn），所以更需要這支腳本頂著。
每一個期望值都是手算出來的，註解裡寫了算式。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit import mapping as mp     # noqa: E402
from evalkit import metrics as mt     # noqa: E402
from evalkit import preprocess as pp  # noqa: E402

passed = failed = 0


def check(name, actual, expected, tol=1e-4):
    global passed, failed
    ok = (abs(actual - expected) <= tol) if isinstance(expected, (int, float)) \
        and not isinstance(expected, bool) else (actual == expected)
    if ok:
        passed += 1
        print(f"  ok   {name}")
    else:
        failed += 1
        print(f"  FAIL {name}\n       期望 {expected!r}\n       實際 {actual!r}")


# ---------------------------------------------------------------------------
print("=== 前處理 ===")
# 方括號表情要原封不動保留（clean 版就是在這裡把 [惊恐] 改成《惊恐》）
check("保留方括號表情", pp.clean("开刷泰国恐怖片。[惊恐]"), "开刷泰国恐怖片。[惊恐]")
# 。是 CJK 標點區，刻意不轉半形；，！ 是全形 ASCII 區，要轉
check("全形轉半形", pp.clean("Ａ１２３，好！"), "A123,好!")
check("移除 @使用者（結尾）", pp.clean("道歉@赵珂宇"), "道歉")
check("移除 @使用者（冒號分隔）", pp.clean("@寂落乌托邦:交友也是校园"), "交友也是校园")
check("話題去井號留內文", pp.clean("#气死我了#今天真的"), "气死我了今天真的")
check("空白正規化", pp.clean("  今天　好    累 "), "今天 好 累")

# ---------------------------------------------------------------------------
print("\n=== 準確度指標 ===")
y_true = ["a", "a", "b", "b", "c", "c"]
y_pred = ["a", "b", "b", "b", "c", "a"]
# a: tp=1 fp=1 fn=1 → P=.5  R=.5  F1=.5
# b: tp=2 fp=1 fn=0 → P=2/3 R=1   F1=.8
# c: tp=1 fp=0 fn=1 → P=1   R=.5  F1=.6667
pc = mt.per_class(y_true, y_pred, ["a", "b", "c"])
check("per_class a.f1", pc["a"]["f1"], 0.5)
check("per_class b.f1", pc["b"]["f1"], 0.8)
check("per_class c.f1", pc["c"]["f1"], 0.6667)
check("per_class b.precision", pc["b"]["precision"], 0.6667)
check("accuracy = 4/6", mt.accuracy(y_true, y_pred), 4 / 6)
check("macro_f1 = (.5+.8+.6667)/3", mt.macro_f1(pc), (0.5 + 0.8 + 0.6667) / 3)

conf = mt.confusion_matrix(y_true, y_pred, ["a", "b", "c"])
check("混淆矩陣 a→b", conf["matrix"]["a"]["b"], 1)
check("混淆矩陣 c→a", conf["matrix"]["c"]["a"], 1)

# ---------------------------------------------------------------------------
print("\n=== mapping-free 指標 ===")
# 完全一致的兩種分類 → NMI = 1、ARI = 1
check("NMI 完全一致", mt.normalized_mutual_info(["x", "x", "y", "y"], ["0", "0", "1", "1"]), 1.0)
check("ARI 完全一致", mt.adjusted_rand_index(["x", "x", "y", "y"], ["0", "0", "1", "1"]), 1.0)
# 2x2 全 1 的列聯表 → 互資訊為 0；ARI 的經典負值案例
# index=0, sum_a=sum_b=2, C(4,2)=6, expected=4/6=.6667, max=2 → ARI=(0-.6667)/(2-.6667)=-.5
check("NMI 完全獨立", mt.normalized_mutual_info(["x", "x", "y", "y"], ["0", "1", "0", "1"]), 0.0)
check("ARI 完全獨立", mt.adjusted_rand_index(["x", "x", "y", "y"], ["0", "1", "0", "1"]), -0.5)

# R1 三句(a,a,b) → 指派給 a 命中 2；R2 兩句(a,b) 平手取字典序前者 a 命中 1
# 上界 = (2+1)/5 = 0.6
best = mt.best_possible_mapping(["R1", "R1", "R1", "R2", "R2"], ["a", "a", "b", "a", "b"])
check("最佳映射 R1→a", best["mapping"]["R1"], "a")
check("最佳映射上界 = 3/5", best["accuracy_upper_bound"], 0.6)

# 支持度：判成 R1 的 3 句裡有 2 句真的是 a → precision = 2/3
# 真的是 a 的 3 句裡有 2 句被判成 R1 → recall = 2/3
sup = mt.mapping_support(["R1", "R1", "R1", "R2", "R2"], ["a", "a", "b", "a", "b"],
                         {"R1": "a", "R2": None})
r1 = next(s for s in sup if s["raw"] == "R1")
check("支持度 precision", r1["precision"], 0.6667)
check("支持度 recall", r1["recall"], 0.6667)
check("棄權的支持度為 None", next(s for s in sup if s["raw"] == "R2")["target"], None)

# ---------------------------------------------------------------------------
print("\n=== 三視角切分 ===")
preds = [
    {"true_label": "angry", "raw_label": "憤怒語調", "confidence": 0.9},
    {"true_label": "fear", "raw_label": "厭惡語調", "confidence": 0.8},
    {"true_label": "happy", "raw_label": "開心語調", "confidence": 0.7},
]
mapping = {"憤怒語調": "angry", "厭惡語調": None, "開心語調": "happy"}
views = mp.build_views(preds, mapping, ["happy", "angry", "fear"], ["happy", "angry"])
check("strict 樣本數（全留）", views["strict"]["n"], 3)
check("strict 棄權標記", views["strict"]["y_pred"][1], mp.ABSTAIN)
check("covered 樣本數（去掉棄權）", views["covered"]["n"], 2)
check("covered coverage = 2/3", views["covered"]["coverage"], 2 / 3)
# covers 不含 fear → true_label=fear 的那句被排除
check("subset 樣本數（排除 fear）", views["subset"]["n"], 2)
check("strict accuracy = 2/3", mt.accuracy(views["strict"]["y_true"], views["strict"]["y_pred"]), 2 / 3)
check("covered accuracy = 1.0", mt.accuracy(views["covered"]["y_true"], views["covered"]["y_pred"]), 1.0)

# ---------------------------------------------------------------------------
print("\n=== 信心校準 ===")
# 兩句都落在 (0.8, 0.9] 這格：實際準確率 .5、平均信心 .9 → ECE = |.5-.9| = .4
cal = mt.calibration([0.9, 0.9], [True, False])
check("ECE", cal["ece"], 0.4)
check("平均信心", cal["mean_confidence"], 0.9)
# 門檻 0.9：兩句都保留 → coverage 1.0、準確率 .5
th = next(t for t in cal["thresholds"] if t["threshold"] == 0.9)
check("門檻 0.9 覆蓋率", th["coverage"], 1.0)
check("門檻 0.9 準確率", th["accuracy"], 0.5)
# confidence 剛好 1.0 不能被漏掉（最後一格含右端點）
check("信心 1.0 有進 bin", sum(b["count"] for b in mt.calibration([1.0], [True])["reliability"]), 1)

# ---------------------------------------------------------------------------
print(f"\n{'=' * 40}\n通過 {passed} 項，失敗 {failed} 項")
sys.exit(1 if failed else 0)
