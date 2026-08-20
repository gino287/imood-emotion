# 待GINO改寫
"""把四套標籤體系的映射關係印出來，順便驗一致性。

    python sourcing/mapping/show_mapping.py            # 全部
    python sourcing/mapping/show_mapping.py --reverse  # 只看「每個五類吃進了什麼」

不連網、不吃任何資料檔，純粹把 schemes.py 的內容攤開來看。
schemes.py 載入時本身就有斷言，所以這支能跑完就代表三張表沒打錯字。

反向那一欄（--reverse）是實際看的時候最有用的：正向表是「這一類要去哪」，
反向表是「這一類是從哪來的」。譬如「怒」在文字端吃了憤怒語調 + 厭惡語調、
在臉部端吃了 Anger + Disgust —— 兩邊都是兩個來源合成的，
所以「怒」的數字天生比其他類別容易偏高，看報表時要記得這件事。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sourcing.common import store  # noqa: E402
from sourcing.mapping import schemes  # noqa: E402

TABLES = [
    ("文字（BERT 8 類）", "emotion/labels.py", schemes.NATIVE_TO_FIVE),
    ("臉部（AffectNet 8 類）", "sourcing/mapping/schemes.py", schemes.AFFECT_TO_FIVE),
    ("演員（RAVDESS 代號）", "sourcing/mapping/schemes.py", schemes.RAVDESS_TO_FIVE),
]


def label_of(table: dict, key: str) -> str:
    """RAVDESS 那張表的鍵是數字代號，單看 01/02 看不出是什麼，補上英文名。"""
    if table is schemes.RAVDESS_TO_FIVE:
        return f"{key} {schemes.RAVDESS_CODE_NAMES[key]}"
    return key


def show_forward() -> None:
    for title, where, table in TABLES:
        print(f"\n{title}   （定義在 {where}）")
        print("-" * 52)
        for k, v in table.items():
            print(f"  {label_of(table, k):22s} → {v if v else '（棄權）'}")
        produced = schemes.five_classes(table)
        missing = set(schemes.DETECT_LABELS) - produced
        print(f"  產得出來：{'、'.join(sorted(produced))}")
        if missing:
            print(f"  產不出來：{'、'.join(sorted(missing))}")


def show_reverse() -> None:
    print("\n每個五類標籤是從哪些原生類別來的")
    print("=" * 60)
    print(f"{'五類':6s}{'文字（BERT）':26s}{'臉部（AffectNet）':26s}演員（RAVDESS）")
    print("-" * 92)
    for five in schemes.ALL_LABELS:
        cols = []
        for _, _, table in TABLES:
            srcs = [label_of(table, k) for k, v in table.items() if v == five]
            cols.append("+".join(srcs) if srcs else "—")
        mark = "  ← 目標類別" if five in schemes.TARGET else ""
        print(f"{five:6s}{cols[0]:26s}{cols[1]:26s}{cols[2]}{mark}")

    print("\n棄權的原生類別（三套各自）")
    print("-" * 60)
    for title, _, table in TABLES:
        dropped = [label_of(table, k) for k, v in table.items() if not v]
        print(f"  {title}：{'、'.join(dropped) if dropped else '無'}")


def main():
    store.enable_utf8_stdout()
    p = argparse.ArgumentParser(description="印出各套標籤體系與五類的映射關係")
    p.add_argument("--reverse", action="store_true",
                   help="只看反向表（每個五類吃進了哪些原生類別）")
    args = p.parse_args()

    if not args.reverse:
        show_forward()
    show_reverse()

    print("\n三張表的值域都在五類之內（schemes.py 載入時已斷言通過）")
    print("要看這些映射的實證依據：python sourcing/mapping/crosstab.py")


if __name__ == "__main__":
    main()
