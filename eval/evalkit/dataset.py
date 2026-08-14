# 待GINO改寫
"""凍結測試集的讀寫與 hash 核對（規劃書 v2 §3.6）。

為什麼要 hash：跑到第三個模型時，沒有人記得住第一個模型當初跑的是哪一版資料。
核對不過就直接失敗中止 —— 靜靜地跑出一份不能跟前面比較的數字，比報錯糟糕得多。
"""
import hashlib
import json
from pathlib import Path

VARIANT_FIELD = {"zh_cn": "text_zh_cn", "zh_tw": "text_zh_tw"}


class DatasetError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 明確用 \n 而非平台預設換行，否則同一份資料在 Windows 與容器內會算出不同 hash
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_testset(path: Path, expect_sha256=None) -> list:
    if not path.exists():
        raise DatasetError(
            f"找不到測試集：{path}\n先跑一次：python eval/scripts/build_testset.py"
        )

    actual = sha256_of(path)
    if expect_sha256 and actual != expect_sha256:
        raise DatasetError(
            "測試集 hash 與設定檔不符，中止評測。\n"
            f"  設定檔 expect_sha256 : {expect_sha256}\n"
            f"  實際檔案            : {actual}\n"
            "代表這份資料集跟先前跑過的模型不是同一份，比出來的數字不可比。\n"
            "若確定要改用新資料集，請把 configs/models.yaml 的 expect_sha256 更新，"
            "並重跑所有既有模型。"
        )

    rows = read_jsonl(path)
    required = {"id", "true_label", "text_zh_cn", "text_zh_tw"}
    for row in rows[:1]:
        missing = required - set(row)
        if missing:
            raise DatasetError(f"測試集欄位不完整，缺少：{missing}")
    return rows


def text_of(row: dict, variant: str) -> str:
    if variant not in VARIANT_FIELD:
        raise DatasetError(f"未知的文字變體 '{variant}'，可用：{list(VARIANT_FIELD)}")
    return row[VARIANT_FIELD[variant]]
