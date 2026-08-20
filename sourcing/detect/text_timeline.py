# 待GINO改寫
"""L4A：抽音訊 → STT → BERT，產出每支影片的情緒時間軸與可用切點。

    python sourcing/detect/text_timeline.py --uid bili:BV1RD421T7S9
    python sourcing/detect/text_timeline.py --verdict keep --limit 20
    python sourcing/detect/text_timeline.py --report        # 只彙整已跑完的，不重跑

這一步跟現有資產的契合度最高：**要找的東西剛好就是這套模型在做的事。**
把影片的話切成一段一段丟進 BERT，就得到一條情緒的時間軸；
「這個人有沒有多種情緒」就從一個要靠人看片判斷的問題，
變成一行程式：時間軸上有幾種類別出現過。

而且產出的不只是「這支能用」，是**直接給 JoyGen 的切點**
（bili:BV1xxx，怒，00:42–00:47）。這是人工看片給不了的東西。

⚠️ 定位限制，必須先講清楚：
   BERT 讀的是**文字語意**，JoyGen 要的是**臉部表情**。
   人可以面無表情地說「我好生氣」，也可以笑著講難過的事。
   所以這一層是**高召回的粗篩，順便提供時間戳**，不是最終裁決。
   最終要不要用，還是得靠 L4B 的臉部表情辨識或人眼確認。

⚠️ 五類裡的「喜」在這一層產不出來。原生的「驚奇語調」實測是價性中立的
   驚訝（surprise 66.7%、happy 只有 6.7%），拿 raw_probs 做價性閘門也失敗，
   所以文字層只收斂到「驚」，喜/哀 的分家留給下游。詳見 emotion/labels.py。
"""
import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emotion import preprocess  # noqa: E402
from emotion.classifier import EmotionClassifier, resolve_device  # noqa: E402
from emotion.labels import NATIVE_TO_FIVE, TEXT_LAYER_LABELS  # noqa: E402
from sourcing.common import paths, store  # noqa: E402
from stt.transcribe import SpeechToText  # noqa: E402

# 一個片段要能當素材的最低條件。太短的片段抽不出穩定的表情，
# 信心太低的多半是 STT 轉錯字或語意本來就模糊。
MIN_CLIP_S = 2.0
MIN_CONFIDENCE = 0.5

# 一個類別要幾個片段才算「這支影片真的有這種情緒」。
#
# ⚠️ 這個門檻是實測逼出來的，不是憑感覺設的。第一支測試影片
#    （bili:BV1RD421T7S9，3分43秒的口播）109 段裡 78 段是平淡語氣，
#    但「樂/哀/怒」各湊到 1 段、「驚」2 段，於是被判成「五類全齊」。
#    去看逐字稿，那幾段全是中性內容：
#      樂 ← 「和大家聊聊碩士進大專當老師是什麼感受」（conf 0.830）
#      哀 ← 「到現在其實也快乾滿兩個學期了」（conf 0.912）
#    也就是說偽陽性照樣拿得到 0.9 的信心，**光提高信心門檻擋不住**。
#
#    真正有效的判準是「有沒有連續好幾段都是同一種情緒」：真的在生氣的人
#    不會只氣一句話。所以改成要求同一類至少 N 段才算數。
MIN_CLIPS_PER_LABEL = 3


def extract_audio(url: str, dst: Path) -> None:
    """用 yt-dlp 抽音訊。只要 16k 單聲道 wav —— Whisper 內部就是這個取樣率，
    抓更高的品質只是浪費頻寬與轉檔時間。"""
    cmd = [
        "yt-dlp", "-f", "bestaudio", "--no-playlist", "--quiet", "--no-warnings",
        "-x", "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "-o", str(dst.with_suffix(".%(ext)s")),
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists():
        raise RuntimeError(
            f"yt-dlp 抽音訊失敗（returncode={proc.returncode}）：\n"
            f"{(proc.stderr or proc.stdout or '')[-400:]}"
        )


def classify_segments(clf: EmotionClassifier, segments: list) -> list:
    """每一段各跑一次 BERT，套上 8→5 映射。

    刻意逐段跑而不是整篇一次：整篇丟進去只會得到「這支影片整體偏什麼情緒」，
    但我們要的是情緒在時間上的變化，那正好是逐段才看得到的東西。
    """
    out = []
    for seg in segments:
        # traditional=True：Whisper 吐的是簡體，而模型與資料集都是繁體，
        # 不轉會靜默拉低準確率。這正是 preprocess.prepare 存在的理由。
        text, reason = preprocess.prepare(seg.text, traditional=True)
        if reason:
            # 被前處理擋掉的照樣留在時間軸上，只是不給情緒標籤。
            # 直接跳過的話，時間軸上會出現無法解釋的空洞。
            out.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
                "raw_label": None,
                "label": None,
                "confidence": None,
                "skipped": reason,
            })
            continue
        pred = clf.predict(text)
        out.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
            "raw_label": pred.label,
            "label": NATIVE_TO_FIVE[pred.label],   # None 代表棄權（關切語調）
            "confidence": round(pred.confidence, 4),
        })
    return out


def pick_clips(timeline: list, min_clip_s: float, min_conf: float) -> dict:
    """每一類挑出符合條件的片段，就是要交給 JoyGen 的切點。"""
    clips = defaultdict(list)
    for seg in timeline:
        if seg["label"] is None:
            continue
        if seg["end"] - seg["start"] < min_clip_s:
            continue
        if seg["confidence"] < min_conf:
            continue
        clips[seg["label"]].append({
            "start": seg["start"],
            "end": seg["end"],
            "confidence": seg["confidence"],
            "text": seg["text"],
        })
    # 每類信心高的排前面，人工複核時先看最有把握的
    for label in clips:
        clips[label].sort(key=lambda c: -c["confidence"])
    return dict(clips)


def timeline_path(uid: str) -> Path:
    return paths.timeline_path(uid)


def summarize(result: dict, min_clip_s: float, min_conf: float, min_clips: int) -> dict:
    """從 timeline 重算 clips 與各項統計。抽出來是為了 --remap 能重用。"""
    timeline = result["timeline"]
    clips = pick_clips(timeline, min_clip_s, min_conf)

    # 「這支影片真的有這種情緒」的判準：同一類要有夠多段。
    # 只有 1 段的多半是偽陽性 —— 見檔案上方 MIN_CLIPS_PER_LABEL 的說明。
    qualified = sorted(k for k, v in clips.items() if len(v) >= min_clips)

    # 非中性的比例：這支影片到底有多少情緒起伏，還是從頭平到尾。
    # 用來排序候選影片比「湊到幾種情緒」有意義得多。
    labeled = [s for s in timeline if s.get("label")]
    non_neutral = [s for s in labeled if s["label"] != "中性"]

    result["labels_present"] = qualified
    result["n_labels_present"] = len(qualified)
    result["clip_counts"] = {k: len(v) for k, v in sorted(clips.items())}
    result["non_neutral_ratio"] = (
        round(len(non_neutral) / len(labeled), 3) if labeled else 0.0)
    result["clips"] = clips
    return result


def remap(args) -> None:
    """不重跑模型，只用已存的 raw_label 重算五類映射與切點。

    跟 eval/evalkit/mapping.py 開頭講的是同一件事：映射是後處理。
    timeline 裡存了每一段的 raw_label，所以改 NATIVE_TO_FIVE 之後
    不必再抽一次音訊、跑一次 Whisper 與 BERT —— 那是幾十分鐘 vs 幾秒的差別。
    """
    files = sorted(paths.TIMELINE_DIR.glob("*.json")) if paths.TIMELINE_DIR.exists() else []
    if not files:
        raise SystemExit(f"還沒有任何時間軸（{paths.TIMELINE_DIR}）")

    for f in files:
        result = json.loads(f.read_text(encoding="utf-8"))
        for seg in result["timeline"]:
            if seg.get("raw_label"):
                seg["label"] = NATIVE_TO_FIVE[seg["raw_label"]]
        summarize(result, args.min_clip, args.min_confidence, args.min_clips)
        f.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已用目前的 NATIVE_TO_FIVE 重算 {len(files)} 支的映射與切點\n")


def process_one(clf, stt, row: dict, args) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        extract_audio(row["url"], wav)
        tr = stt.transcribe(wav, vad_filter=True)

    timeline = classify_segments(clf, tr.segments or [])

    result = {
        "uid": row["uid"],
        "url": row["url"],
        "title": row.get("title", ""),
        "author_id": row.get("author_id", ""),
        "author_name": row.get("author_name", ""),
        "duration_s": row.get("duration_s"),
        "n_segments": len(timeline),
        "timeline": timeline,
    }
    # 達標的類別才算數；clips 裡沒達標的仍然留著，方便回頭調門檻
    summarize(result, args.min_clip, args.min_confidence, args.min_clips)
    timeline_path(row["uid"]).parent.mkdir(parents=True, exist_ok=True)
    timeline_path(row["uid"]).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def report() -> None:
    """彙整所有跑完的時間軸：哪些人湊得齊幾種情緒。

    這才是要交給 JoyGen 的東西 —— 不是影片清單，是
    person → {情緒: [(影片, 起, 訖)]} 的素材表。
    """
    files = sorted(paths.TIMELINE_DIR.glob("*.json")) if paths.TIMELINE_DIR.exists() else []
    if not files:
        raise SystemExit(f"還沒有任何時間軸，先跑一次（{paths.TIMELINE_DIR}）")

    results = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    by_author = defaultdict(lambda: defaultdict(list))
    names = {}
    for r in results:
        key = r["author_id"] or "(未知)"
        names[key] = r["author_name"]
        # 只收達標的類別。沒達標的片段仍然存在各支影片的 json 裡，
        # 想放寬門檻重看的話直接改參數重跑 --report 即可。
        for label in r.get("labels_present", list(r["clips"])):
            for c in r["clips"].get(label, []):
                by_author[key][label].append({
                    "uid": r["uid"], "url": r["url"],
                    "start": c["start"], "end": c["end"],
                    "confidence": c["confidence"], "text": c["text"],
                })

    print(f"已跑完 {len(results)} 支影片，涉及 {len(by_author)} 位作者\n")

    per_video = Counter(r["n_labels_present"] for r in results)
    print("單支影片達標的情緒種類數：")
    for n in sorted(per_video, reverse=True):
        print(f"  {n} 種 × {per_video[n]:4d} 支")

    # 排序看的是情緒起伏，不是種類數 —— 種類數容易被偽陽性灌水
    ranked_videos = sorted(results, key=lambda r: -r.get("non_neutral_ratio", 0))
    print("\n情緒起伏最大的影片（非中性段落佔比）：")
    for r in ranked_videos[:10]:
        counts = "、".join(f"{k}×{v}" for k, v in r.get("clip_counts", {}).items())
        print(f"  {r.get('non_neutral_ratio', 0) * 100:5.1f}%  "
              f"{r['title'][:26]:28s} [{counts}]")

    ranked = sorted(by_author.items(), key=lambda kv: -len(kv[1]))
    print(f"\n以「人」為單位（這才是 JoyGen 要的）：")
    for key, labels in ranked[:25]:
        got = sorted(labels.keys())
        missing = [c for c in TEXT_LAYER_LABELS if c not in got]
        print(f"  {names.get(key, '')[:18]:20s} {len(got)} 種：{'/'.join(got)}"
              + (f"　（缺 {'/'.join(missing)}）" if missing else "　★ 全齊"))

    out = paths.MATERIAL_TABLE
    out.write_text(json.dumps(
        {names.get(k, k) or k: v for k, v in by_author.items()},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {out}")


def main():
    store.enable_utf8_stdout()
    p = argparse.ArgumentParser(description="L4A 情緒時間軸")
    p.add_argument("--uid", action="append", default=[], help="只跑指定影片，可重複給")
    p.add_argument("--verdict", choices=["keep", "maybe", "drop"],
                   help="只跑 L3 判定為此檔的影片")
    p.add_argument("--limit", type=int, help="最多跑幾支")
    p.add_argument("--device", default="auto", help="cuda / cpu / auto")
    p.add_argument("--whisper", default="small", help="faster-whisper 模型大小")
    p.add_argument("--min-clip", type=float, default=MIN_CLIP_S,
                   help=f"片段最短秒數（預設 {MIN_CLIP_S}）")
    p.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE,
                   help=f"片段最低信心（預設 {MIN_CONFIDENCE}）")
    p.add_argument("--min-clips", type=int, default=MIN_CLIPS_PER_LABEL,
                   help=f"一個類別要幾段才算數（預設 {MIN_CLIPS_PER_LABEL}）。"
                        "設 1 會被偽陽性灌爆，見程式碼上方說明")
    p.add_argument("--report", action="store_true", help="只彙整已跑完的，不重跑")
    p.add_argument("--remap", action="store_true",
                   help="改過 NATIVE_TO_FIVE 之後，用已存的 raw_label 重算映射與切點，"
                        "不重抽音訊也不重跑模型")
    p.add_argument("--refresh", action="store_true", help="已經跑過的也重跑")
    args = p.parse_args()

    if args.remap:
        remap(args)
        report()
        return

    if args.report:
        report()
        return

    rows = store.read_jsonl(paths.METADATA)
    if not rows:
        raise SystemExit(f"找不到 {paths.METADATA}，先跑 L1")

    if args.uid:
        wanted = set(args.uid)
        rows = [r for r in rows if r["uid"] in wanted]
    elif args.verdict:
        verdicts = {r["uid"]: r["verdict"] for r in store.read_jsonl(paths.FILTERED)}
        if not verdicts:
            raise SystemExit(f"找不到 {paths.FILTERED}，先跑 sourcing/collect/coarse_filter.py")
        rows = [r for r in rows if verdicts.get(r["uid"]) == args.verdict]

    if not args.refresh:
        rows = [r for r in rows if not timeline_path(r["uid"]).exists()]
    if args.limit:
        rows = rows[:args.limit]

    if not rows:
        print("沒有待處理的影片，直接彙整")
        report()
        return

    print(f"要處理 {len(rows)} 支影片")

    # 兩個模型都只載入一次。每支影片各載一次的話，光載入就比推論還久。
    device = resolve_device(args.device)
    print(f"載入 Whisper {args.whisper} → {device}")
    stt = SpeechToText(model_size=args.whisper, device=device)
    stt.load()
    print(f"  {stt.load_seconds:.1f}s")

    print(f"載入 BERT → {device}")
    clf = EmotionClassifier(device=device)
    clf.load()
    print(f"  {clf.load_seconds:.1f}s\n")

    ok = failed = 0
    for i, row in enumerate(rows, 1):
        try:
            r = process_one(clf, stt, row, args)
        except Exception as exc:
            failed += 1
            store.append_error("timeline", row["uid"], f"{type(exc).__name__}: {exc}")
            print(f"  [{i}/{len(rows)}] {row['uid']} 失敗：{type(exc).__name__}: {exc}")
            continue
        ok += 1
        counts = "、".join(f"{k}×{v}" for k, v in r["clip_counts"].items())
        print(f"  [{i}/{len(rows)}] {row['uid']} {r['n_segments']:3d} 段，"
              f"達標 {r['n_labels_present']} 種（{'/'.join(r['labels_present']) or '無'}）"
              f"　非中性 {r['non_neutral_ratio'] * 100:.0f}%　[{counts}]"
              f"　{row.get('title', '')[:24]}")

    print(f"\n完成 {ok} 支、失敗 {failed} 支\n")
    report()


if __name__ == "__main__":
    main()
