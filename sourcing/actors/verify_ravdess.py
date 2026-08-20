# 待GINO改寫
"""交叉驗證：用 FER 去看整理好的 RAVDESS 素材，臉部判定跟檔名對不對得上。

    python sourcing/actors/verify_ravdess.py
    python sourcing/actors/verify_ravdess.py --sample-fps 8 --show-all

**為什麼要驗。**
檔名上的情緒是演員被要求演的，不保證他演出來的臉真的長那樣，也不保證
我們的 FER 讀得出來。這兩件事要分開看：

  演員沒演到位      → 素材本身不能用
  FER 讀不出來      → 素材可能還好，是我們的偵測在這個表情上弱

這支腳本只負責把不一致的挑出來，**不自己決定怎麼處理**。

做法：每個檔案抽影格（預設 8 fps，因為 RAVDESS 一段只有 3–4 秒）→ YuNet
偵測人臉 → hsemotion 判 AffectNet 8 類 → 收斂成偵測層五類 → 再換成交付用的
資料夾名（中性→預設、驚→喜），然後跟檔名標的比。

判定用**多數票**而不是單張最高分：單張影格會被眨眼、講話的嘴型帶偏，
一段 3 秒的演出取十幾張、看哪一類最多，才是這段臉「整體看起來是什麼」。
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sourcing.common import paths, store  # noqa: E402
from sourcing.detect import face_timeline as ft  # noqa: E402
from sourcing.mapping.schemes import (  # noqa: E402
    DELIVERY_LABELS,
    DETECT_TO_DELIVERY,
)

OUT_DIR = paths.ACTOR_CLIP_DIR
REPORT = paths.OUT_DIR / "ravdess_fer_check.json"

# 短片段要取密一點。RAVDESS 一段 3–4 秒，2 fps 只有 6、7 張，
# 其中還有開頭結尾的預備動作，多數票會很不穩。
SAMPLE_FPS = 8.0


def judge(samples: list) -> dict:
    """一個檔案的整體判定：有臉的影格裡，哪一個交付標籤最多。"""
    with_face = [s for s in samples if s["affect"]]
    confident = [s for s in with_face
                 if s["label"] and (s["confidence"] or 0) >= ft.MIN_CONF]

    votes = Counter(DETECT_TO_DELIVERY.get(s["label"]) for s in confident)
    affect = Counter(s["affect"] for s in with_face)
    top, n_top = votes.most_common(1)[0] if votes else (None, 0)

    return {
        "verdict": top,
        "votes": dict(votes),
        "affect_dist": dict(affect),
        "n_frames": len(samples),
        "n_face": len(with_face),
        "n_confident": len(confident),
        "share": round(n_top / len(confident), 3) if confident else 0.0,
        "mean_conf": (round(sum(s["confidence"] for s in confident) / len(confident), 3)
                      if confident else 0.0),
    }


def main():
    store.enable_utf8_stdout()
    p = argparse.ArgumentParser(description="用 FER 交叉驗證 RAVDESS 素材")
    p.add_argument("--sample-fps", type=float, default=SAMPLE_FPS)
    p.add_argument("--model", default="enet_b0_8_best_vgaf")
    p.add_argument("--show-all", action="store_true", help="逐檔印出，不只印不一致的")
    args = p.parse_args()

    manifest_path = OUT_DIR / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"找不到 {manifest_path}\n"
                         "先跑 python sourcing/actors/prepare_ravdess.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    fer = ft.make_fer(args.model)
    detector = ft.make_detector()

    rows = []
    print(f"逐檔驗證 {len(manifest)} 個檔案（取樣 {args.sample_fps} fps）\n")
    for i, m in enumerate(manifest, 1):
        path = OUT_DIR / m["file"]
        try:
            samples = ft.scan_video(path, fer, detector, sample_fps=args.sample_fps)
        except Exception as exc:
            print(f"  [{i:3d}] {m['file']} 失敗：{type(exc).__name__}: {exc}")
            continue
        r = judge(samples)
        r.update({"file": m["file"], "actor": m["actor"], "gender": m["gender"],
                  "expected": m["label"], "intensity": m["intensity"],
                  "picked_because": m["picked_because"]})
        r["match"] = (r["verdict"] == m["label"])
        rows.append(r)
        if args.show_all or not r["match"]:
            mark = "OK " if r["match"] else "!! "
            print(f"  [{i:3d}] {mark}{m['actor']} 檔名{r['expected']:2s}"
                  f" → FER {str(r['verdict']):3s}"
                  f"（{r['share'] * 100:3.0f}% 的影格，信心 {r['mean_conf']:.2f}）"
                  f"  {Path(m['file']).name}")

    if not rows:
        raise SystemExit("沒有任何檔案驗證成功")

    # ── 總表：檔名標的（列）× FER 判的（欄） ──────────────────────────
    print("\n\n總表：檔名標的（列）× FER 判的（欄）")
    cols = DELIVERY_LABELS + ["None"]
    corner = "檔名 \\ FER"
    header = f"{corner:12s}" + "".join(f"{c:>7s}" for c in cols) + f"{'吻合率':>9s}"
    print(header)
    print("-" * (len(header) + 8))
    matrix = defaultdict(Counter)
    for r in rows:
        matrix[r["expected"]][str(r["verdict"])] += 1
    for exp in DELIVERY_LABELS:
        c = matrix.get(exp)
        if not c:
            continue
        n = sum(c.values())
        cells = "".join(f"{c.get(col, 0):>7d}" for col in cols)
        print(f"{exp:12s}{cells}{c.get(exp, 0) / n * 100:8.0f}%")

    n_ok = sum(1 for r in rows if r["match"])
    print(f"\n整體吻合 {n_ok}/{len(rows)} = {n_ok / len(rows) * 100:.0f}%")

    # ── 每位演員 ────────────────────────────────────────────────────
    print("\n各演員：")
    by_actor = defaultdict(list)
    for r in rows:
        by_actor[r["actor"]].append(r)
    for actor in sorted(by_actor):
        rs = by_actor[actor]
        ok = sum(1 for r in rs if r["match"])
        bad = sorted({r["expected"] for r in rs if not r["match"]},
                     key=DELIVERY_LABELS.index)
        print(f"  {actor}（{rs[0]['gender']}）{ok}/{len(rs)}"
              + (f"　對不上：{'/'.join(bad)}" if bad else "　全對"))

    # ── 對不上的明細，留給人決定怎麼處理 ─────────────────────────────
    bad_rows = [r for r in rows if not r["match"]]
    if bad_rows:
        print(f"\n\n對不上的 {len(bad_rows)} 個，明細如下（**不自動處理，請人決定**）：")
        for r in bad_rows:
            print(f"\n  {r['file']}")
            print(f"    檔名標 {r['expected']}，FER 判 {r['verdict']}"
                  f"（{r['share'] * 100:.0f}% 的可信影格）")
            print(f"    選它的理由：{r['picked_because']}")
            print(f"    有臉 {r['n_face']}/{r['n_frames']} 張，"
                  f"其中信心夠的 {r['n_confident']} 張")
            print("    五類票數："
                  + "、".join(f"{k}{v}" for k, v in r["votes"].items()))
            print("    AffectNet 原始分布："
                  + "、".join(f"{k}{v}" for k, v in r["affect_dist"].items()))

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "sample_fps": args.sample_fps,
        "model": args.model,
        "n_files": len(rows),
        "n_match": n_ok,
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {REPORT}")


if __name__ == "__main__":
    main()
