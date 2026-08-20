# 待GINO改寫
"""用 FER 對 B 站清單做大規模粗掃，找「同一個人身上有幾種情緒」。

    python sourcing/detect/scan_batch.py --plan                    # 只看候選名單與預估時間
    python sourcing/detect/scan_batch.py --probe --limit 40 --sleep 15
    python sourcing/detect/scan_batch.py --full --top 20           # 對粗掃結果好的做完整分析
    python sourcing/detect/scan_batch.py --report                  # 彙整

**這支要回答的問題**：857 個作者裡，有沒有人在自己的影片裡真的出現過
四種目標情緒（喜/怒/哀/樂，不含預設）。之前只拿 5 個人看過，樣本太小。

────────────────────────────────────────────────────────────────────────
兩段式：先粗掃、再細看
────────────────────────────────────────────────────────────────────────

  粗掃（--probe）  在整支影片上挑 8 個時間窗、每個窗抓 45 秒，360p、2 fps
  細看（--full）   整支影片抓下來，480p、2 fps，跑完整的 face_timeline

**粗掃不整支下載，是這支腳本最關鍵的一件事。** 實測這條線路對 B 站
大約只有 200KB/s，一支 52 分鐘的影片光下載就花掉 12 分鐘還會斷線重試 ——
下載時間完全蓋過辨識時間。改成只抓幾個時間窗之後，**成本與影片長度無關**
（不管 10 分鐘還是 50 分鐘，都只抓 6 分鐘的畫面），而且看到的時間點
還是散布在全片，不會只看開頭。

窗內用跟細看一樣的 2 fps，所以「同一種表情連續 3 秒」這個判準在粗掃就能用，
兩段的結果是同一種東西，只是粗掃的覆蓋率低。

粗掃的定位是**寧可多收**：它的工作是不要漏掉，判斷該不該用是細看那一段的事。
反過來說，粗掃看到的是抽樣 —— 某個人在沒被抽到的那 40 分鐘裡笑了，粗掃看不到。
所以粗掃的「有」很可信，「沒有」比較弱，下結論時要留意這個不對稱。

────────────────────────────────────────────────────────────────────────
候選怎麼選：用時長，不用標題
────────────────────────────────────────────────────────────────────────

⚠️ 這裡刻意**不拿標題／題材當篩選門檻**，理由是量出來的：

  1. 片型關鍵字對這批素材幾乎沒作用（857 支只刷掉 6.3%）——
     這份清單本來就是 JoyGen 挑過的單人說話影片。
  2. 更重要的是，「題材的情緒」跟「臉上的表情」實測是兩回事。
     用題材選人，等於把文字那條路失敗的原因原封不動搬進來。

所以門檻只有時長（越長越有機會出現情緒轉折，這是結構性的，不涉及語意判斷）。

但題材分數還是**照樣算出來寫進結果**，只是不當門檻。這樣粗掃跑完之後
可以反過來檢查：題材分數高的人，量到的情緒種類真的比較多嗎？
這個問題目前沒有人有答案，跑完就有了。

────────────────────────────────────────────────────────────────────────
慢慢跑
────────────────────────────────────────────────────────────────────────

每支之間會 sleep（預設 10 秒），而且做完一支就存一支，中斷了重跑會跳過
已經做完的。要跑一整夜就直接開著，不必守著。
"""
import argparse
import json
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sourcing.common import paths, store  # noqa: E402
from sourcing.detect import face_timeline as ft  # noqa: E402
from sourcing.deliver.shortlist import score as topic_score  # noqa: E402
from sourcing.mapping.schemes import DETECT_TO_DELIVERY, TARGET  # noqa: E402

PROBE_DIR = paths.OUT_DIR / "probe"

# ── 粗掃的參數 ──────────────────────────────────────────────────────────
PROBE_WINDOWS = 8        # 在整支影片上取幾個時間窗
PROBE_WINDOW_S = 45      # 每個窗幾秒。8 × 45 = 6 分鐘的畫面，與片長無關
PROBE_FPS = 2.0          # 窗內的取樣率，跟細看一樣，這樣「連續 3 秒」才成立
PROBE_HEIGHT = 360       # 臉還是夠 FER 判，下載量比 480p 少一截
PROBE_MIN_SHARE = 0.02   # 影格層級的「有出現過」門檻（寧可多收）
PROBE_MIN_FRAMES = 5     # 但至少要有 5 張，免得幾百張裡的 2 張雜訊被當成一類

# ── 候選門檻 ────────────────────────────────────────────────────────────
MIN_DURATION = 600       # 10 分鐘以下湊不出多種情緒（實測中位數 295 秒）
MAX_DURATION = 3600      # 一小時以上處理成本太高


def playable_seconds(row: dict) -> int:
    """這支影片**實際抓得到**幾秒，不是 API 說的幾秒。

    B站的分 P 影片（一個 BV 號底下好幾集），`/x/web-interface/view` 回的
    `duration` 是**所有分 P 的總和**，但 yt-dlp 帶 `--no-playlist` 只會抓 P1。
    兩者差很多：實測有一支 5 個分 P 的，API 說 1099 秒，P1 只有 214 秒。

    後果是時間窗會鋪到根本不存在的位置。ffmpeg 對著超過結尾的起點會直接
    `exited with code 222`，實測一支影片 8 個窗爆掉 7 個、整支產不出結果。
    所以候選篩選與時間窗都要用 P1 的長度。
    """
    dur = row.get("duration_s") or 0
    f = paths.raw_path("bilibili", row["vid"])
    if not f.exists():
        return dur
    try:
        v = json.loads(f.read_text(encoding="utf-8"))["view"]["data"]
    except (KeyError, ValueError):
        return dur
    if (v.get("videos") or 1) <= 1:
        return dur
    pages = v.get("pages") or []
    return int(pages[0].get("duration") or dur) if pages else dur


def candidates(min_dur: int, max_dur: int) -> list:
    rows = []
    for r in store.read_jsonl(paths.METADATA):
        if r.get("platform") != "bilibili":
            continue
        # 分 P 影片要用 P1 的長度，不然時間窗會鋪到影片結尾之後
        r["duration_s"] = playable_seconds(r)
        if min_dur <= r["duration_s"] <= max_dur:
            rows.append(r)
    for r in rows:
        s = topic_score(r)
        r["_topic"] = s["narrative_pts"]
        r["_female"] = s["female_pts"]
        r["_likely_female"] = s["likely_female"]
    # 排序只用時長：長的先跑，因為長的樣本量大、結論比較站得住
    rows.sort(key=lambda r: -(r.get("duration_s") or 0))
    return rows


def probe_path(uid: str) -> Path:
    return PROBE_DIR / f"{paths.uid_to_filename(uid)}.json"


# 目標四類用交付的名字（喜/怒/哀/樂），跟規格一致。不含「預設」——
# 每支影片都一堆中性，拿它當達成條件沒有意義。
TARGET_DELIVERY = [DETECT_TO_DELIVERY[k] for k in TARGET]


def windows_for(duration_s: int, n: int, length: int) -> list:
    """把時間窗平均鋪在影片上，前後各留一點。

    開頭常是片頭動畫或還沒開始講，結尾常是道別與訂閱提醒，
    兩端的表情都不代表這個人平常的樣子，所以只鋪在中間那 90%。
    """
    usable = max(0, duration_s - length)
    if usable <= 0:
        return [0.0]
    lo, hi = usable * 0.05, usable * 0.95
    if n == 1:
        return [round(lo, 1)]
    step = (hi - lo) / (n - 1)
    return [round(lo + i * step, 1) for i in range(n)]


def summarize_probe(samples: list, clips: dict) -> dict:
    """粗掃的統計。

    labels_present 用的是**片段**（連續 3 秒），跟細看同一個判準；
    share 是影格層級的比例，拿來看「這個人整體偏哪一種表情」。
    兩個都留著，因為它們回答的問題不一樣。
    """
    with_face = [s for s in samples if s["affect"]]
    confident = [s for s in with_face
                 if s["label"] and (s["confidence"] or 0) >= ft.MIN_CONF]

    votes = Counter(DETECT_TO_DELIVERY.get(s["label"]) for s in confident)
    n = len(confident)
    share = {k: round(v / n, 4) for k, v in votes.items()} if n else {}

    clip_counts = {k: len(v) for k, v in clips.items()}
    present = [k for k in TARGET_DELIVERY if clip_counts.get(k, 0) >= 1]

    # 只有零星影格、湊不成連續片段的那些也記下來 ——「有一點但不成段」
    # 跟「完全沒有」是兩件事，之後要調門檻時會需要這個差別
    frame_only = [k for k in TARGET_DELIVERY
                  if k not in present
                  and votes.get(k, 0) >= PROBE_MIN_FRAMES
                  and share.get(k, 0) >= PROBE_MIN_SHARE]

    return {
        "n_samples": len(samples),
        "n_face": len(with_face),
        "face_rate": round(len(with_face) / len(samples), 3) if samples else 0.0,
        "n_confident": n,
        "affect_dist": dict(Counter(s["affect"] for s in with_face)),
        "counts": dict(votes),
        "share": share,
        "clip_counts": clip_counts,
        "labels_present": present,
        "n_labels_present": len(present),
        "frame_only_labels": frame_only,
    }



def do_probe(rows: list, fer, detector, sleep_s: float,
             n_windows: int, window_s: int) -> None:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows, 1):
        out = probe_path(row["uid"])
        head = (f"[{i}/{len(rows)}] {row['uid']} "
                f"{(row.get('author_name') or '')[:12]} "
                f"{(row.get('duration_s') or 0) // 60}分")
        t0 = time.perf_counter()
        try:
            starts = windows_for(row.get("duration_s") or 0, n_windows, window_s)
            samples, clips, n_win_ok = [], {}, 0
            win_secs = []
            with tempfile.TemporaryDirectory() as tmp:
                for w, start in enumerate(starts):
                    seg = Path(tmp) / f"w{w}.mp4"
                    try:
                        got_s = ft.download_window(row["url"], seg, start, window_s,
                                                   max_height=PROBE_HEIGHT)
                        part = ft.scan_video(seg, fer, detector, sample_fps=PROBE_FPS)
                    except Exception as exc:
                        print(f"    窗 {start:.0f}s 跳過：{type(exc).__name__}: "
                              f"{str(exc)[:80]}")
                        continue
                    n_win_ok += 1
                    win_secs.append(round(got_s, 1))
                    # 片段一個窗一個窗算：跨窗的「連續」是假的，中間隔了好幾分鐘
                    for lab, cs in ft.to_clips(part).items():
                        for c in cs:
                            c["start"] = round(c["start"] + start, 2)
                            c["end"] = round(c["end"] + start, 2)
                        clips.setdefault(DETECT_TO_DELIVERY.get(lab, lab),
                                         []).extend(cs)
                    for sm in part:
                        sm["t"] = round(sm["t"] + start, 2)
                    samples.extend(part)
                    seg.unlink(missing_ok=True)
            if n_win_ok < max(1, len(starts) // 2):
                raise RuntimeError(f"只成功 {n_win_ok}/{len(starts)} 個時間窗")
            rec = {
                "uid": row["uid"], "url": row["url"],
                "title": row.get("title", ""),
                "author_name": row.get("author_name", ""),
                "author_id": row.get("author_id", ""),
                "duration_s": row.get("duration_s", 0),
                "topic_pts": row.get("_topic", 0),
                "female_pts": row.get("_female", 0),
                "likely_female": row.get("_likely_female", False),
                "probe_fps": PROBE_FPS,
                "n_windows": len(starts),
                "n_windows_ok": n_win_ok,
                "window_seconds": window_s,
                # 每個窗實際抓到幾秒。要求 45 秒不代表拿得到 45 秒 ——
                # 這一欄就是為了讓「窗長度不對」再也藏不住，見
                # face_timeline.download_window 的說明。
                "window_seconds_actual": win_secs,
                "clips": clips,
                # 逐張影格也存著。理由同 store.py 的 raw/metadata 分層：
                # 爬一支要三分鐘、重算門檻只要幾毫秒，門檻改了不該再爬一次。
                # 一支約 80KB，198 支也才 16MB。
                "samples": samples,
            }
            rec.update(summarize_probe(samples, clips))
            out.write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            got = "/".join(rec["labels_present"]) or "—"
            print(f"{head}  {time.perf_counter() - t0:5.0f}s  "
                  f"有臉 {rec['face_rate'] * 100:3.0f}%  "
                  f"{rec['n_labels_present']} 種：{got}")
        except Exception as exc:
            print(f"{head}  失敗：{type(exc).__name__}: {str(exc)[:120]}")
            store.append_error("bilibili", row["vid"], f"probe: {type(exc).__name__}")
        if sleep_s:
            time.sleep(sleep_s)


def rescore() -> int:
    """用已經存下來的逐張影格重算片段與門檻，完全不連網。

    跟 text_timeline.py 的 --remap 是同一個道理：判準會改（例如 FER 的
    MIN_CONF 要不要調高），但改判準不該需要重爬一次。
    """
    files = sorted(PROBE_DIR.glob("*.json")) if PROBE_DIR.exists() else []
    n = 0
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        samples = rec.get("samples")
        if not samples:
            print(f"  {f.name} 沒有存影格，跳過（是舊版跑的，要重爬才有）")
            continue
        # 片段要一個窗一個窗重算，用時間差判斷窗的邊界
        clips, window = {}, []
        gap = 5.0 / PROBE_FPS      # 影格間隔超過這個就是跨窗了
        for i, sm in enumerate(samples):
            if window and sm["t"] - window[-1]["t"] > gap:
                _merge_clips(clips, window)
                window = []
            window.append(sm)
        _merge_clips(clips, window)
        rec["clips"] = clips
        rec.update(summarize_probe(samples, clips))
        f.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        n += 1
    return n


def _merge_clips(clips: dict, window: list) -> None:
    """把一個時間窗算出來的片段併進總表。窗內的秒數本來就是絕對時間。"""
    if not window:
        return
    base = window[0]["t"]
    shifted = [dict(s, t=round(s["t"] - base, 2)) for s in window]
    for lab, cs in ft.to_clips(shifted).items():
        for c in cs:
            c["start"] = round(c["start"] + base, 2)
            c["end"] = round(c["end"] + base, 2)
        clips.setdefault(DETECT_TO_DELIVERY.get(lab, lab), []).extend(cs)


def load_probes() -> list:
    if not PROBE_DIR.exists():
        return []
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(PROBE_DIR.glob("*.json"))]


def report() -> None:
    probes = load_probes()
    if not probes:
        raise SystemExit(f"還沒有粗掃結果（{PROBE_DIR}）")

    probes.sort(key=lambda r: (-r["n_labels_present"], -r["duration_s"]))
    print(f"粗掃完成 {len(probes)} 支\n")

    print(f"{'情緒種類':>8s}{'人數':>6s}")
    print("-" * 20)
    dist = Counter(r["n_labels_present"] for r in probes)
    for k in sorted(dist, reverse=True):
        print(f"{k:>8d}{dist[k]:>6d}")

    print(f"\n各類出現過的人數（門檻：≥{PROBE_MIN_SHARE * 100:.0f}% 的可信影格"
          f"且 ≥{PROBE_MIN_FRAMES} 張）")
    print("-" * 40)
    for lab in TARGET_DELIVERY:
        n = sum(1 for r in probes if lab in r["labels_present"])
        print(f"  {lab}  {n:4d} 人  {'#' * int(n / len(probes) * 40)}")

    print("\n前 25 名（依情緒種類、再依時長）")
    print("-" * 96)
    for r in probes[:25]:
        share = "  ".join(f"{k}{r['share'].get(k, 0) * 100:4.1f}%"
                          for k in TARGET_DELIVERY)
        print(f"  {r['n_labels_present']} 種 {share}"
              f"  {r['duration_s'] // 60:3d}分"
              f"  {(r['author_name'] or '')[:10]:12s}{r['title'][:26]}")

    # 題材分數到底有沒有預測力 —— 這是刻意留下來檢驗的
    print("\n題材分數 vs 實測情緒種類（檢驗「用標題選人」有沒有用）")
    print("-" * 56)
    by_topic = defaultdict(list)
    for r in probes:
        bucket = "高（≥6）" if r["topic_pts"] >= 6 else (
            "中（2–5）" if r["topic_pts"] >= 2 else "低（0–1）")
        by_topic[bucket].append(r["n_labels_present"])
    for bucket in ["高（≥6）", "中（2–5）", "低（0–1）"]:
        vals = by_topic.get(bucket)
        if not vals:
            continue
        print(f"  題材分數 {bucket:10s} n={len(vals):3d}  "
              f"平均情緒種類 {sum(vals) / len(vals):.2f}")
    print("  → 三組差不多的話，代表用標題選人沒有幫助，時長才是有效的訊號。")

    n_all4 = sum(1 for r in probes if r["n_labels_present"] == 4)
    print(f"\n四種都出現過的：{n_all4} 人")
    if n_all4:
        for r in probes:
            if r["n_labels_present"] == 4:
                print(f"  {r['uid']}  {r['author_name']}  {r['title'][:30]}")
        print("\n→ 這些人要用 --full 細看，確認是真的片段還是零星影格")
    else:
        print("  （粗掃這一層都沒有，細看只會更少 —— 這個結果本身就是結論）")


def main():
    store.enable_utf8_stdout()
    p = argparse.ArgumentParser(description="FER 大規模粗掃")
    p.add_argument("--plan", action="store_true", help="只印候選名單與預估時間")
    p.add_argument("--probe", action="store_true", help="跑粗掃")
    p.add_argument("--full", action="store_true", help="對粗掃結果好的跑完整分析")
    p.add_argument("--report", action="store_true", help="彙整粗掃結果")
    p.add_argument("--rescore", action="store_true",
                   help="用已存的影格重算門檻與片段，不連網、不重爬")
    p.add_argument("--limit", type=int, help="這一輪最多跑幾支")
    p.add_argument("--top", type=int, default=20, help="--full 時取粗掃前幾名")
    p.add_argument("--sleep", type=float, default=10.0, help="每支之間停幾秒")
    p.add_argument("--windows", type=int, default=PROBE_WINDOWS,
                   help="粗掃在整支影片上取幾個時間窗")
    p.add_argument("--window-seconds", type=int, default=PROBE_WINDOW_S,
                   help="每個時間窗幾秒")
    p.add_argument("--min-duration", type=int, default=MIN_DURATION)
    p.add_argument("--max-duration", type=int, default=MAX_DURATION)
    p.add_argument("--model", default="enet_b0_8_best_vgaf")
    p.add_argument("--refresh", action="store_true", help="已經跑過的也重跑")
    p.add_argument("--uid", action="append", default=[],
                   help="只跑這幾支（可重複）。單獨重跑某一支時用")
    args = p.parse_args()

    if args.rescore:
        n = rescore()
        print(f"重算了 {n} 支（用已存的逐張影格，沒有連網）")
        report()
        return

    if args.report:
        report()
        return

    rows = candidates(args.min_duration, args.max_duration)
    if args.uid:
        rows = [r for r in rows if r["uid"] in set(args.uid)]
        args.refresh = True      # 指名要跑的就是要重跑，不然指名沒有意義

    if args.full:
        probes = load_probes()
        if not probes:
            raise SystemExit("要先跑 --probe")
        probes.sort(key=lambda r: (-r["n_labels_present"], -r["duration_s"]))
        want = {r["uid"] for r in probes[:args.top]}
        rows = [r for r in rows if r["uid"] in want]
        if not args.refresh:
            rows = [r for r in rows if not ft.out_path(r["uid"]).exists()]
        print(f"細看 {len(rows)} 支（粗掃前 {args.top} 名裡還沒細看的）")
        if not rows:
            return
        fer = ft.make_fer(args.model)
        detector = ft.make_detector()
        for i, row in enumerate(rows, 1):
            try:
                rec = ft.process_one(row, fer, detector)
                print(f"[{i}/{len(rows)}] {row['uid']} "
                      f"{rec['n_labels_present']} 種：{'/'.join(rec['labels_present'])}")
            except Exception as exc:
                print(f"[{i}/{len(rows)}] {row['uid']} 失敗："
                      f"{type(exc).__name__}: {str(exc)[:120]}")
            time.sleep(args.sleep)
        return

    if not args.refresh:
        rows = [r for r in rows if not probe_path(r["uid"]).exists()]
    if args.limit:
        rows = rows[:args.limit]

    if args.plan or not args.probe:
        total = len(candidates(args.min_duration, args.max_duration))
        done = len(load_probes())
        mins = sum((r.get("duration_s") or 0) for r in rows) / 60
        per_video_s = args.windows * args.window_seconds
        print(f"候選（{args.min_duration}–{args.max_duration} 秒）：{total} 支")
        print(f"已粗掃：{done} 支　這一輪要跑：{len(rows)} 支")
        print(f"影片總長 {mins:.0f} 分鐘，但粗掃只抓其中"
              f" {args.windows} 個時間窗 × {args.window_seconds} 秒")
        print(f"實際要下載的畫面：{len(rows) * per_video_s / 60:.0f} 分鐘"
              f"（與片長無關），約 {len(rows) * per_video_s * PROBE_FPS:.0f} 張影格要判")
        print("\n前 10 支：")
        for r in rows[:10]:
            print(f"  {r['uid']}  {(r.get('duration_s') or 0) // 60:3d}分  "
                  f"題材{r['_topic']:2d} 女性{r['_female']:2d}  "
                  f"{(r.get('author_name') or '')[:10]:12s}{(r.get('title') or '')[:28]}")
        if not args.probe:
            print("\n要真的跑：加 --probe")
        return

    print(f"這一輪粗掃 {len(rows)} 支，每支之間停 {args.sleep} 秒")
    fer = ft.make_fer(args.model)
    detector = ft.make_detector()
    do_probe(rows, fer, detector, args.sleep,
             args.windows, args.window_seconds)
    print("\n跑完了。彙整：python sourcing/detect/scan_batch.py --report")


if __name__ == "__main__":
    main()
