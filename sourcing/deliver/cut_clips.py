# 待GINO改寫
"""L5：把 FER 找到的片段實際切成影片檔，交給 JoyGen 測試。

    python sourcing/deliver/cut_clips.py --uid bili:BV1aM4y117kD
    python sourcing/deliver/cut_clips.py --all --per-emotion 3
    python sourcing/deliver/cut_clips.py --uid ... --height 720 --pad 0.3

交出去的不能只是「第 42 秒到第 47 秒」這種清單 —— 對方還得自己下載、自己切。
這一支直接產出可以丟進 JoyGen 的檔案，外加一份 manifest 說明每個檔案是誰的
哪一種情緒。

輸出長這樣：
    _local/sourcing/clips/
        四月吨吨_/
            樂/  bili_BV1aM4y117kD_0661.5-0680.0.mp4
            哀/  …
        manifest.json

⚠️ 只切 FER（`sourcing/detect/face_timeline.py`）找到的片段，不用文字那條 ——
   文字判的是話題的情緒，實測與臉對不上。
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sourcing.common import paths, store  # noqa: E402
from sourcing.detect.face_timeline import out_path as face_path  # noqa: E402
from sourcing.mapping.schemes import TARGET  # noqa: E402

CLIP_DIR = paths.CLIP_DIR


def download(url: str, dst: Path, height: int) -> None:
    """抓影片。這裡的畫質要比 FER 那支高 —— 那支只要看得出表情，
    這裡切出來的是要真的拿去生成的素材。**要帶音軌**（FER 那支不用）。

    畫質用 `-S res:N` 排序而不是 `height<=N` 過濾，理由同
    face_timeline.format_args()：直的影片 height 是長邊，用 height 當硬條件
    會把整支影片的格式全排除掉，而 B站沒有合併格式可以退，直接失敗。
    一樣優先 avc1，OpenCV 與多數工具都吃。
    """
    proc = subprocess.run(
        ["yt-dlp", "--no-playlist", "--quiet", "--no-warnings",
         "-f", "bv*[vcodec^=avc1]+ba/b[vcodec^=avc1]/bv*+ba/b",
         "-S", f"res:{height}",
         "--merge-output-format", "mp4", "-o", str(dst), url],
        capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists():
        raise RuntimeError(f"下載失敗：{(proc.stderr or proc.stdout)[-300:]}")


def cut(src: Path, dst: Path, start: float, end: float, pad: float) -> None:
    """切一段出來。

    前後各留一點餘裕（pad）：FER 是逐張判定的，表情的起訖點會被切得很死，
    留一點過渡看起來比較自然，對嘴型模型也比較好處理。

    重新編碼而不是 -c copy：copy 只能從關鍵影格切，實際起點會飄到好幾秒前，
    切出來的片段可能根本不是那個表情。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    s = max(0.0, start - pad)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-ss", f"{s:.2f}", "-to", f"{end + pad:.2f}", "-i", str(src),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-pix_fmt", "yuv420p", "-c:a", "aac", str(dst)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 失敗：{(proc.stderr or '')[-300:]}")


def process(rec: dict, per_emotion: int, height: int, pad: float,
            min_seconds: float) -> list:
    author = (rec.get("author_name") or "unknown").strip() or "unknown"
    # 資料夾名稱不能有路徑分隔或其他會炸掉的字元
    safe_author = "".join(c for c in author if c not in '\\/:*?"<>|').strip() or "unknown"

    picks = []
    for emo in TARGET:
        clips = [c for c in rec["clips"].get(emo, [])
                 if c["end"] - c["start"] >= min_seconds][:per_emotion]
        for c in clips:
            picks.append((emo, c))
    if not picks:
        return []

    made = []
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src.mp4"
        print(f"  下載 {rec['uid']} …")
        download(rec["url"], src, height)
        for emo, c in picks:
            name = f"{rec['uid'].replace(':', '_')}_{c['start']:07.1f}-{c['end']:07.1f}.mp4"
            dst = CLIP_DIR / safe_author / emo / name
            cut(src, dst, c["start"], c["end"], pad)
            made.append({
                "file": str(dst.relative_to(CLIP_DIR)).replace("\\", "/"),
                "author": author,
                "emotion": emo,
                "source_uid": rec["uid"],
                "source_url": rec["url"],
                "source_title": rec.get("title", ""),
                "start": c["start"],
                "end": c["end"],
                "seconds": round(c["end"] - c["start"] + 2 * pad, 2),
                "fer_confidence": c["confidence"],
                "fer_frames": c["frames"],
            })
            print(f"    {emo}  {c['end'] - c['start']:5.1f}s  conf {c['confidence']:.2f}"
                  f"  → {dst.name}")
    return made


def main():
    store.enable_utf8_stdout()
    p = argparse.ArgumentParser(description="L5 切出可交付的情緒片段")
    p.add_argument("--uid", action="append", default=[])
    p.add_argument("--all", action="store_true", help="所有跑過 FER 的影片")
    p.add_argument("--per-emotion", type=int, default=3, help="每種情緒切幾段")
    p.add_argument("--min-seconds", type=float, default=3.0,
                   help="片段最短秒數（太短的對嘴型模型沒用）")
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--pad", type=float, default=0.3, help="前後各留幾秒餘裕")
    args = p.parse_args()

    face_dir = paths.FACE_DIR
    if not face_dir.exists():
        raise SystemExit("還沒有 FER 結果，先跑 sourcing/detect/face_timeline.py")

    if args.all:
        files = sorted(face_dir.glob("*.json"))
    elif args.uid:
        files = [face_path(u) for u in args.uid]
    else:
        raise SystemExit("要指定 --uid 或 --all")

    manifest = []
    for f in files:
        if not f.exists():
            print(f"  跳過（沒有 FER 結果）：{f.name}")
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        try:
            manifest.extend(process(rec, args.per_emotion, args.height,
                                    args.pad, args.min_seconds))
        except Exception as exc:
            print(f"  {rec['uid']} 失敗：{type(exc).__name__}: {exc}")

    if not manifest:
        raise SystemExit("沒有切出任何片段（可能都不到最短秒數）")

    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    (CLIP_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    by_author = {}
    for m in manifest:
        by_author.setdefault(m["author"], set()).add(m["emotion"])
    print(f"\n共 {len(manifest)} 個片段：")
    for a, emos in by_author.items():
        print(f"  {a}：{len(emos)} 種情緒（{'/'.join(sorted(emos))}）")
    print(f"\n→ {CLIP_DIR}")


if __name__ == "__main__":
    main()
