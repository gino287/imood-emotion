# 待GINO改寫
"""重算「原生類別 → 真實標籤」交叉表：8→5 那張映射表的實證依據。

    python sourcing/mapping/crosstab.py                       # johnson-small / zh_cn
    python sourcing/mapping/crosstab.py --model johnson-large --device cuda
    python sourcing/mapping/crosstab.py --five                # 套上映射後的五類交叉表
    python sourcing/mapping/crosstab.py --valence-gate        # 重跑失敗過的價性閘門

**這支在回答什麼問題。**
「憤怒語調到底該對到哪一類」不是靠語感決定的，是去看模型判成憤怒語調的
那些句子，真實標籤實際上是什麼。這支腳本就是把那張表算出來。

吃的是 eval 跑完留下的 predictions.jsonl（每一列有 raw_label、true_label、
raw_probs），不重跑模型，所以是零成本的。換映射表、換門檻、換模型，
重跑這一支就有新數字。

**--five** 是套上映射之後的樣子：可以直接看到「哀」吸進了多少不是 sad 的東西。

**--valence-gate** 保留的是一條走不通的路，留著是為了不要再走第二次：
用 raw_probs 的殘餘機率把「驚奇語調」拆成正向（→喜）與負向（→哀）。
實測兩組的分布幾乎一樣，門檻拉高之後甚至反相關。softmax 底下非勝出類的
機率是雜訊，沒有為「當第二個軸讀」校準過。想自己確認就跑這個旗標。

⚠️ 測試集是 SMP2020-EWECT，**微博書面文字**。產品的實際輸入是口語逐字稿，
   兩者領域不同（疑問語調那一類就是這樣被推翻的）。這裡的數字用來做映射的
   相對比較可以，不要當成產品準確率。口語那份見 eval/scripts/build_spoken_set.py。
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sourcing.common import store  # noqa: E402
from sourcing.mapping import schemes  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "eval" / "results"

# SMP2020 的六類，固定這個順序印，比較兩次輸出時欄位才對得上
SMP_LABELS = ["happy", "angry", "sad", "fear", "surprise", "neutral"]

# 負向的三類，價性閘門用
NEGATIVE_NATIVE = ["憤怒語調", "悲傷語調", "厭惡語調"]


def load(model: str, device: str, variant: str) -> list:
    path = RESULTS / model / device / variant / "predictions.jsonl"
    if not path.exists():
        avail = sorted(p.relative_to(RESULTS).as_posix()
                       for p in RESULTS.glob("*/*/*/predictions.jsonl"))
        raise SystemExit(
            f"找不到 {path}\n"
            + ("目前有這些：\n  " + "\n  ".join(avail) if avail
               else "先跑 python eval/run_eval.py")
        )
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def print_table(rows_by_key: dict, keys: list, title: str) -> None:
    """一列一個預測類別，欄位是真實標籤的百分比，最右邊是該列的樣本數。"""
    print(f"\n{title}")
    header = f"{'':10s}" + "".join(f"{k:>10s}" for k in SMP_LABELS) + f"{'n':>7s}"
    print(header)
    print("-" * len(header))
    for k in keys:
        counter = rows_by_key.get(k)
        if not counter:
            continue
        n = sum(counter.values())
        cells = []
        top = max(counter, key=counter.get) if n else None
        for lab in SMP_LABELS:
            pct = counter.get(lab, 0) / n * 100
            mark = "*" if lab == top and pct > 0 else " "
            cells.append(f"{pct:8.1f}%{mark}"[-10:])
        print(f"{k:10s}" + "".join(cells) + f"{n:7d}")
    print("  （* 是該列最多的真實標籤。看的是「模型判這一類時，實際上是什麼」）")


def native_crosstab(rows: list) -> None:
    by_native = defaultdict(Counter)
    for r in rows:
        by_native[r["raw_label"]][r["true_label"]] += 1
    print_table(by_native, schemes.NATIVE_LABELS, "原生 8 類 → 真實標籤")

    print("\n對映射的意義：")
    for native in schemes.NATIVE_LABELS:
        c = by_native.get(native)
        if not c:
            continue
        n = sum(c.values())
        top, cnt = c.most_common(1)[0]
        mapped = schemes.NATIVE_TO_FIVE[native]
        mapped_txt = mapped if mapped else "（棄權）"
        print(f"  {native:6s} n={n:4d}  最多是 {top:9s} {cnt / n * 100:5.1f}%"
              f"   → 映到 {mapped_txt}")


def five_crosstab(rows: list) -> None:
    """套上 NATIVE_TO_FIVE 之後的樣子。棄權的那些單獨列出來，不併進任何一類。"""
    by_five = defaultdict(Counter)
    abstained = Counter()
    for r in rows:
        five = schemes.NATIVE_TO_FIVE[r["raw_label"]]
        if five is None:
            abstained[r["true_label"]] += 1
            continue
        by_five[five][r["true_label"]] += 1

    print_table(by_five, schemes.ALL_LABELS, "套上 NATIVE_TO_FIVE 後的五類 → 真實標籤")

    n_all = len(rows)
    n_abs = sum(abstained.values())
    print(f"\n棄權 {n_abs} 筆（{n_abs / n_all * 100:.1f}%），"
          f"來自 {'、'.join(k for k, v in schemes.NATIVE_TO_FIVE.items() if not v)}")
    print("  這些句子的真實標籤分布：",
          "、".join(f"{k} {abstained.get(k, 0)}" for k in SMP_LABELS))
    print("  棄權不是浪費 —— 這一類模型本來就答不準，硬塞進五類只會污染那一類。")

    print("\n各五類的樣本數（看有沒有嚴重失衡）：")
    for k in schemes.ALL_LABELS:
        n = sum(by_five.get(k, Counter()).values())
        bar = "#" * int(n / max(1, n_all) * 60)
        note = "  ← 這一層產不出來，要靠下游拆「驚」" if k == "喜" else ""
        print(f"  {k:4s}{n:5d}  {bar}{note}")


def valence_gate(rows: list, threshold: float) -> None:
    """把驚奇語調依 raw_probs 的殘餘機率拆成正向／負向，看兩組分布有沒有差。"""
    pos, neg, mid = Counter(), Counter(), Counter()
    for r in rows:
        if r["raw_label"] != "驚奇語調":
            continue
        probs = r["raw_probs"]
        p_pos = probs.get("開心語調", 0.0)
        p_neg = sum(probs.get(k, 0.0) for k in NEGATIVE_NATIVE)
        if p_pos - p_neg > threshold:
            pos[r["true_label"]] += 1
        elif p_neg - p_pos > threshold:
            neg[r["true_label"]] += 1
        else:
            mid[r["true_label"]] += 1

    print(f"\n價性閘門（門檻 {threshold}）：驚奇語調拆成正向／負向")
    print_table({"正向→喜": pos, "負向→哀": neg, "分不出": mid},
                ["正向→喜", "負向→哀", "分不出"], "拆完之後的分布")

    def pct(c, lab):
        n = sum(c.values())
        return c.get(lab, 0) / n * 100 if n else 0.0

    print(f"\n關鍵比較（閘門有效的話，正向組的 happy 應該明顯高於負向組）：")
    print(f"  正向組 happy {pct(pos, 'happy'):5.1f}%   負向組 happy {pct(neg, 'happy'):5.1f}%")
    print(f"  正向組 sad   {pct(pos, 'sad'):5.1f}%   負向組 sad   {pct(neg, 'sad'):5.1f}%")
    gap = pct(pos, "happy") - pct(neg, "happy")
    if gap < 5:
        print(f"  → 差距只有 {gap:.1f} 個百分點，閘門沒有切開價性。"
              "\n     這就是「喜」不在文字層產出的原因，換門檻也一樣。")
    else:
        print(f"  → 差距 {gap:.1f} 個百分點，值得再看看（之前實測是切不開的）")


def main():
    store.enable_utf8_stdout()
    p = argparse.ArgumentParser(description="重算原生類別與真實標籤的交叉表")
    p.add_argument("--model", default="johnson-small", help="eval/results/ 底下的模型名")
    p.add_argument("--device", default="cpu", help="cpu / cuda")
    p.add_argument("--variant", default="zh_cn", help="zh_cn / zh_tw")
    p.add_argument("--five", action="store_true", help="改看套上映射後的五類交叉表")
    p.add_argument("--valence-gate", action="store_true",
                   help="重跑價性閘門（拆驚奇語調成喜/哀）的實驗")
    p.add_argument("--threshold", type=float, default=0.05, help="價性閘門的門檻")
    args = p.parse_args()

    rows = load(args.model, args.device, args.variant)
    print(f"{args.model} / {args.device} / {args.variant}：{len(rows)} 句")

    if args.valence_gate:
        valence_gate(rows, args.threshold)
    elif args.five:
        five_crosstab(rows)
    else:
        native_crosstab(rows)


if __name__ == "__main__":
    main()
