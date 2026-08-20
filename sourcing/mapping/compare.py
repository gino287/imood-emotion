# 待GINO改寫
"""同一支影片，文字判的情緒 vs 臉部判的情緒，並排對照。

    python sourcing/mapping/compare.py --uid bili:BV17z4y117cR
    python sourcing/mapping/compare.py --all
    python sourcing/mapping/compare.py --all --matrix     # 只看總表

**這支在回答什麼問題。**
兩條路都收斂到同一組五類，所以可以直接問：**文字說這 5 秒是「怒」，
臉在那 5 秒是什麼？** 這是「文字判不出表情」這個結論的量化版本 ——
之前那個判斷是拿 5 支影片人工看出來的，這支腳本把它變成數字。

做法：拿文字時間軸的每一個片段，去臉部那份的逐張影格（samples）裡撈出
落在同一個時間窗內的判定，看多數票是什麼。

不重跑任何模型，只讀兩邊已經跑完的結果：
    _local/sourcing/timelines/{uid}.json   ← sourcing/detect/text_timeline.py
    _local/sourcing/face/{uid}.json        ← sourcing/detect/face_timeline.py
所以兩邊都跑過的影片才比得出來，--all 會自己挑出交集。

⚠️ 不一致不代表某一邊壞了。兩邊量的本來就是不同的東西（話題的情緒 vs
   臉上的表情），這支腳本的用途是**知道差多少、差在哪**，
   而不是把兩邊調成一致。要拿去給 JoyGen 的素材看的是臉那一邊。
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sourcing.common import paths, store  # noqa: E402
from sourcing.mapping import schemes  # noqa: E402

# 影格判定要有幾張落在這個時間窗內，才拿來跟文字比。
# 太少張的多半是那段沒偵測到臉（側身、轉頭、切畫面），
# 拿一兩張去代表五秒鐘的表情不可靠。
MIN_FRAMES_IN_WINDOW = 4


def load_pair(uid: str):
    t_path, f_path = paths.timeline_path(uid), paths.face_path(uid)
    if not t_path.exists():
        raise SystemExit(f"沒有文字時間軸：{t_path}\n"
                         f"先跑 python sourcing/detect/text_timeline.py --uid {uid}")
    if not f_path.exists():
        raise SystemExit(f"沒有臉部時間軸：{f_path}\n"
                         f"先跑 python sourcing/detect/face_timeline.py --uid {uid}")
    return (json.loads(t_path.read_text(encoding="utf-8")),
            json.loads(f_path.read_text(encoding="utf-8")))


def both_done() -> list:
    """兩邊都跑過的 uid。檔名是 uid 把冒號換成底線來的，這裡反推回去。"""
    if not paths.TIMELINE_DIR.exists() or not paths.FACE_DIR.exists():
        raise SystemExit("兩邊的結果都要有才比得出來（timelines/ 與 face/）")
    t = {p.stem for p in paths.TIMELINE_DIR.glob("*.json")}
    f = {p.stem for p in paths.FACE_DIR.glob("*.json")}
    return sorted(s.replace("_", ":", 1) for s in (t & f))


def face_vote(samples: list, start: float, end: float) -> tuple:
    """時間窗內臉部判定的多數票。回傳 (標籤, 該標籤張數, 窗內總張數)。"""
    inside = [s for s in samples if start <= s["t"] <= end]
    votes = Counter(s["label"] for s in inside if s.get("label"))
    if not votes:
        return None, 0, len(inside)
    label, n = votes.most_common(1)[0]
    return label, n, len(inside)


def compare_one(text: dict, face: dict, verbose: bool) -> Counter:
    """回傳這支影片的 (文字標籤, 臉部標籤) 計數，順便印明細。"""
    samples = face.get("samples", [])
    pairs = Counter()

    print(f"\n{'=' * 74}")
    print(f"{text['uid']}  {text.get('title', '')[:34]}")
    print(f"作者 {text.get('author_name', '')}　時長 {text.get('duration_s', 0)}s"
          f"　臉部偵測率 {face.get('face_rate', 0) * 100:.0f}%")
    print(f"{'=' * 74}")

    print(f"\n{'':6s}{'文字（片段數）':>16s}{'臉部（片段數）':>16s}")
    print("-" * 44)
    for label in schemes.ALL_LABELS:
        tc = text.get("clip_counts", {}).get(label, 0)
        fc = face.get("clip_counts", {}).get(label, 0)
        flag = ""
        if tc and not fc:
            flag = "  ← 文字有、臉沒有"
        elif fc and not tc:
            flag = "  ← 臉有、文字沒有"
        print(f"{label:6s}{tc:>14d}　{fc:>14d}{flag}")

    for label in schemes.ALL_LABELS:
        for clip in text.get("clips", {}).get(label, []):
            vote, n_vote, n_total = face_vote(samples, clip["start"], clip["end"])
            if n_total < MIN_FRAMES_IN_WINDOW:
                pairs[(label, "（那段沒臉）")] += 1
                continue
            pairs[(label, vote or "（判不出）")] += 1
            if verbose:
                mark = "同" if vote == label else "異"
                print(f"  [{mark}] 文字{label} {clip['start']:7.1f}-{clip['end']:6.1f}s"
                      f"  conf {clip['confidence']:.2f}"
                      f"　臉部 {vote or '?'}（{n_vote}/{n_total} 張）"
                      f"　「{(clip.get('text') or '')[:22]}」")

    agree = sum(v for (t, f), v in pairs.items() if t == f)
    total = sum(v for (t, f), v in pairs.items() if not f.startswith("（"))
    if total:
        print(f"\n吻合 {agree}/{total} = {agree / total * 100:.0f}%"
              "（文字片段與同一時間窗的臉部多數票同一類）")
    else:
        print("\n沒有可比的片段（那些時間窗內都偵測不到臉）")
    return pairs


def print_matrix(pairs: Counter) -> None:
    """列＝文字判的，欄＝臉部判的。對角線是兩邊同意的。"""
    face_cols = schemes.ALL_LABELS + ["（那段沒臉）", "（判不出）"]
    print("\n\n總表：文字（列）× 臉部（欄）")
    corner = "文字 \\ 臉"
    header = f"{corner:10s}" + "".join(f"{c:>8s}" for c in face_cols)
    print(header)
    print("-" * len(header))
    for t in schemes.ALL_LABELS:
        row = sum(pairs.get((t, f), 0) for f in face_cols)
        if not row:
            continue
        cells = "".join(f"{pairs.get((t, f), 0):>8d}" for f in face_cols)
        print(f"{t:10s}{cells}   （共 {row}）")

    comparable = sum(v for (t, f), v in pairs.items() if not f.startswith("（"))
    agree = sum(v for (t, f), v in pairs.items() if t == f)
    noface = sum(v for (t, f), v in pairs.items() if f == "（那段沒臉）")
    if comparable:
        print(f"\n可比的片段 {comparable} 個，吻合 {agree} 個"
              f"（{agree / comparable * 100:.0f}%）")
    else:
        print("\n沒有可比的片段")
    print(f"另有 {noface} 個文字片段所在的時間窗偵測不到臉。")
    print("\n怎麼讀這張表：對角線高才代表兩邊在講同一件事。"
          "\n實測是不高的 —— 文字量的是話題的情緒，臉量的是表情，"
          "\n所以最終要交出去的素材以臉那一邊為準。")


def main():
    store.enable_utf8_stdout()
    p = argparse.ArgumentParser(description="文字情緒 vs 臉部情緒的對照")
    p.add_argument("--uid", action="append", default=[], help="指定影片，可重複給")
    p.add_argument("--all", action="store_true", help="所有兩邊都跑過的影片")
    p.add_argument("--matrix", action="store_true", help="只印總表，不印每支的明細")
    p.add_argument("--quiet", action="store_true", help="不印逐片段明細")
    args = p.parse_args()

    uids = args.uid or (both_done() if args.all else [])
    if not uids:
        avail = both_done()
        raise SystemExit(
            "要指定 --uid 或 --all\n"
            + (f"兩邊都跑過的有 {len(avail)} 支：\n  " + "\n  ".join(avail[:20])
               if avail else "目前沒有兩邊都跑過的影片")
        )

    total = Counter()
    for uid in uids:
        text, face = load_pair(uid)
        if args.matrix:
            samples = face.get("samples", [])
            for label in schemes.ALL_LABELS:
                for clip in text.get("clips", {}).get(label, []):
                    vote, _, n_total = face_vote(samples, clip["start"], clip["end"])
                    if n_total < MIN_FRAMES_IN_WINDOW:
                        total[(label, "（那段沒臉）")] += 1
                    else:
                        total[(label, vote or "（判不出）")] += 1
        else:
            total.update(compare_one(text, face, verbose=not args.quiet))

    if len(uids) > 1 or args.matrix:
        print_matrix(total)


if __name__ == "__main__":
    main()
