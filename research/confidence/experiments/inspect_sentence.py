"""互動式看單句的信心分布 —— 拿來培養對這個模型的直覺用的。

一次看一句，把八類機率、margin、熵、以及「截斷之後會怎樣」全部印出來。
比看統計摘要更快建立起「什麼樣的句子會讓模型猶豫」的感覺。

    # 指定句子
    docker compose -f docker/docker-compose.yml run --rm app-cpu \
        python research/confidence/experiments/inspect_sentence.py "你怎麼這麼討厭啊"

    # 不給句子就跑內建的示範組（含刻意設計的模糊句與截斷句）
    docker compose -f docker/docker-compose.yml run --rm app-cpu \
        python research/confidence/experiments/inspect_sentence.py

    # 看同一句被逐字截斷時，預測怎麼跳
    docker compose -f docker/docker-compose.yml run --rm app-cpu \
        python research/confidence/experiments/inspect_sentence.py --prefix "我真的受不了你了"
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from imood_emotion.classifier import EmotionClassifier  # noqa: E402

# 刻意挑的示範句：從「應該很好判斷」排到「人自己也說不準」
DEMO = [
    "我今天超級開心的！",              # 應該很有信心
    "我真的受不了你了",                # 憤怒 or 厭惡？
    "哦，是喔。",                      # 平淡 or 厭惡？人也分不出來
    "你確定嗎",                        # 疑問
    "嗯",                              # 資訊量極低
    "今天禮拜三",                      # 沒有情緒，但模型沒有「無情緒」以外的出口
    "字幕製作人Zither Harp",            # 已知的 Whisper 幻覺字串
    "。。。",                          # 純標點（正常會被前處理擋掉）
]


def entropy_bits(probs) -> float:
    return -sum(p * math.log(p + 1e-12, 2) for p in probs)


def show(clf, text: str) -> None:
    p = clf.predict(text)
    ordered = sorted(p.probs.items(), key=lambda kv: -kv[1])
    top1, top2 = ordered[0][1], ordered[1][1]
    ent = entropy_bits([v for _, v in ordered])

    print(f"\n{'=' * 62}")
    print(f"「{text}」")
    print(f"{'-' * 62}")
    for lab, v in ordered:
        bar = "█" * int(v * 40)
        mark = " ←" if lab == p.label else ""
        print(f"  {lab:8} {v:7.4f} {bar}{mark}")
    print(f"{'-' * 62}")
    print(f"  confidence (MSP) = {p.confidence:.4f}")
    print(f"  margin (第1−第2)  = {top1 - top2:.4f}"
          f"   ← 越小代表模型在前兩名之間猶豫")
    print(f"  entropy          = {ent:.3f} bits / 最大 3.000"
          f"   ← 越大代表整個分布越平")
    print(f"  latency          = {p.latency_ms['total']:.1f} ms")

    # 一句話的體檢結論，讓人不用自己換算
    if p.confidence >= 0.95:
        verdict = "很有把握"
    elif p.confidence >= 0.85:
        verdict = "有把握（但注意 0.90~0.95 這一段實測只有約 78% 準確，見 RESULTS.md）"
    elif top1 - top2 < 0.15:
        verdict = f"在「{ordered[0][0]}」與「{ordered[1][0]}」之間猶豫"
    else:
        verdict = "沒把握"
    print(f"  → {verdict}")


def show_prefixes(clf, text: str) -> None:
    """逐字加長，看預測在哪個字翻掉 —— 截斷不穩定的直觀版。"""
    print(f"\n完整句：「{text}」")
    print(f"{'字數':>4} {'預測':<9}{'信心':>8}  片段")
    prev = None
    for i in range(1, len(text) + 1):
        frag = text[:i]
        p = clf.predict(frag)
        flag = "  ← 翻類" if prev and p.label != prev else ""
        print(f"{i:>4} {p.label:<9}{p.confidence:>8.4f}  {frag}{flag}")
        prev = p.label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="*", help="要看的句子；不給就跑內建示範組")
    ap.add_argument("--prefix", help="對這句做逐字前綴掃描")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    clf = EmotionClassifier(device=args.device)
    clf.load()
    clf.predict("暖機")

    if args.prefix:
        show_prefixes(clf, args.prefix)
        return
    for t in (args.texts or DEMO):
        show(clf, t)


if __name__ == "__main__":
    main()
