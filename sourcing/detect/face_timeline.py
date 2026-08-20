# 待GINO改寫
"""L4B：直接量臉部表情，完全不看文字。

    python sourcing/detect/face_timeline.py --uid bili:BV1aM4y117kD
    python sourcing/detect/face_timeline.py --from-timelines --limit 8
    python sourcing/detect/face_timeline.py --report

**為什麼不用文字了。**
L4A 是拿逐字稿的語意去猜情緒，實測證明那條路是錯的：彭春花那支影片
（平靜講解榮格心理學）拿到全場最高的 61% 情緒起伏、哀 51 段，
信心高達 0.98，但內容是「他不願意看見和接納現實的險惡」這種
**在講別人的處境**、臉是中性的句子。文字模型量的是「話題的情緒」，
不是「說話者的情緒」，更不是「臉上的表情」。

JoyGen 要的是臉，所以這一支直接看臉：抽影格 → 偵測人臉 → 跑 FER。

⚠️ AffectNet 的 8 類跟我們文字模型的 8 類是**不同體系**，不必也不該硬對齊。
   這裡只把它收斂到同一組五類，讓兩邊的結果可以並排看。

⚠️ 人臉偵測用 YuNet（`make_detector()` 會自己下載 230KB 的 onnx）。
   原本想用不必下載模型的 Haar cascade，但這個 image 是 OpenCV 5.0，
   `cv2.CascadeClassifier` 已經被移除了。

輸出：`_local/sourcing/face/{uid}.json`
      每支影片一檔，內含逐張影格的判定（samples）與連續片段（clips）。
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request      # noqa: F401
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
except ImportError:      # 沒裝的話等到真的要用時才報，import 這個模組本身不該炸
    HSEmotionRecognizer = None

# ⚠️ 上面那個 urllib.request 看起來沒用到，但不能刪。
#    hsemotion-onnx 第一次執行時要下載模型檔，它的程式裡寫了 urllib.request.urlretrieve
#    卻只 import urllib —— 少了子模組的 import，執行時會 AttributeError。
#    在這裡 import urllib.request 會把該屬性掛上 urllib 套件，等於順手補掉它的漏。

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sourcing.common import paths, store  # noqa: E402
from sourcing.mapping.schemes import AFFECT_TO_FIVE, TARGET  # noqa: E402

# AffectNet 8 類 → 五類的那張表放在 sourcing/mapping/schemes.py，
# 跟文字端與演員資料集的映射表擺在一起。要看內容跑：
#   python sourcing/mapping/show_mapping.py

SAMPLE_FPS = 2.0         # 每秒取幾張。表情變化以百毫秒計，2 fps 抓得到轉折
MIN_FACE_PX = 80         # 太小的臉抽不出可靠表情，也不可能拿來當素材
MIN_CONF = 0.45          # FER 的最低信心
MIN_RUN_FRAMES = 6       # 連續幾張同一表情才算一個片段（2fps → 3 秒）
MAX_HEIGHT = 480         # 下載影片的高度上限。臉部裁切用不到更高解析度
MIN_COVERAGE = 0.8       # 實際掃到的長度至少要有 metadata 片長的幾成，不然視為下載不完整
WINDOW_MIN_RATIO = 0.8   # 時間窗實際長度的容許下限（佔要求長度的幾成）
WINDOW_MAX_RATIO = 1.5   # 上限。切點對齊關鍵影格會多一點，但不該多一倍
WINDOW_TRIES = 3         # 長度不對就重抓幾次


def format_args(max_height: int) -> list:
    """挑畫質的參數。三件事都是踩過才知道的。

    1. **不要音軌**：這幾支只看臉，抓音軌純粹浪費頻寬與合併時間。
    2. **一定要 avc1（H.264）**：B站很多影片有 AV1 版本，而這個 image 裡的
       OpenCV 解不了 AV1 —— 而且它不會報錯，只會讀到 0 張影格，
       表面上看起來像「這支影片偵測不到人臉」。
    3. **不能用 `height<=N` 當硬條件**：直的影片（手機拍的）解析度是
       360x480、480x640 這種，height 是長邊。寫 `height<=360` 會把整支影片的
       所有格式都排除掉，yt-dlp 直接回「Requested format is not available」，
       而且 B站的格式全是分離的視訊／音訊軌，連 `b`（最佳合併版）都沒有可以退。
       實測有 25% 的影片是這樣整支失敗的。
       改用 `-S res:N` 排序（res 取的是短邊），要多少畫質給最接近的，
       橫的直的都吃得下，而且**永遠不會沒有格式可選**。
    """
    return [
        "-f", "bv*[vcodec^=avc1]/b[vcodec^=avc1]/bv*/b",
        "-S", f"res:{max_height}",
    ]


def download_video(url: str, dst: Path, max_height: int = MAX_HEIGHT) -> None:
    """抓整支影片的純視訊軌，畫質取最接近 max_height 的。

    max_height 可以再壓低：粗掃只是要知道「這個人有沒有多種表情」，
    360p 的臉還是夠 FER 判，但下載量少一截。
    """
    cmd = [
        "yt-dlp", "--no-playlist", "--quiet", "--no-warnings",
        *format_args(max_height),
        "--merge-output-format", "mp4",
        "-o", str(dst),
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists():
        raise RuntimeError(
            f"yt-dlp 下載失敗（returncode={proc.returncode}）："
            f"{(proc.stderr or proc.stdout or '')[-300:]}"
        )


def ensure_model(model_name: str) -> None:
    """先把 FER 模型放到 hsemotion 會去找的位置，讓它自己那段下載不會被執行。

    要接手的原因有兩個，都是它那邊的問題：
      1. 它組出來的網址是 github.com/.../blob/...?raw=true，實測回 503。
         正確的直連位址是 raw.githubusercontent.com。
      2. 它把模型快取在 ~/.hsemotion。容器的家目錄沒掛載，每跑一次就重抓一次
         （模型十幾 MB）。這裡改成先存在專案的 _local/ 底下再複製過去，
         就只會下載一次。
    """
    cache = paths.MODEL_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / f"{model_name}.onnx"

    if not local.exists():
        url = ("https://raw.githubusercontent.com/HSE-asavchenko/"
               "face-emotion-recognition/main/models/affectnet_emotions/onnx/"
               f"{model_name}.onnx")
        print(f"下載 FER 模型 {model_name} …")
        urllib.request.urlretrieve(url, local)
        print(f"  → {local}（{local.stat().st_size / 1e6:.1f} MB）")

    home_cache = Path.home() / ".hsemotion"
    home_cache.mkdir(parents=True, exist_ok=True)
    dst = home_cache / f"{model_name}.onnx"
    if not dst.exists():
        shutil.copy2(local, dst)


# ⚠️ 走 media.githubusercontent.com 而不是 raw.githubusercontent.com：
#    opencv_zoo 的模型是用 git-lfs 存的，raw 位址只會拿到 0 位元組的空檔，
#    而且不會報錯 —— 要等 OpenCV 解析失敗才發現。media 位址才拿得到真檔案。
YUNET_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
             "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")


def media_duration(path: Path) -> float:
    """用 ffprobe 問這個檔案實際多長。回傳秒數，問不到就回 0。

    不用 OpenCV 的 CAP_PROP_FRAME_COUNT，那個在被切過的檔案上常常是估的。
    """
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True)
    try:
        return float((proc.stdout or "0").strip())
    except ValueError:
        return 0.0


def download_window(url: str, dst: Path, start: float, length: float,
                    max_height: int = MAX_HEIGHT, tries: int = WINDOW_TRIES) -> float:
    """只抓影片中的一小段（第 start 秒起的 length 秒）。回傳實際抓到幾秒。

    粗掃一支 50 分鐘的影片，整支抓下來是 100–200MB，而實測這條線路
    大約只有 200KB/s —— 下載時間會完全蓋過辨識時間。
    改成在整支影片上挑幾個時間窗、每個窗只抓幾十秒，
    **成本就與影片長度無關了**，而且看到的時間點還是散布在全片。

    `--download-sections` 要有 ffmpeg（這個 image 裡有，cut_clips 也在用）。

    ⚠️ 這裡有兩層防呆，兩層都是踩過才加的：

    一、**一定要加 `--force-keyframes-at-cuts`。** 原本沒加，理由是
        「切點滑到最近的關鍵影格對這種用途沒差，還能省下重新編碼」——
        錯得離譜。B站的串流不加這個參數時**根本不遵守指定長度**：
        實測要 45 秒，拿回來的是 7 秒、162 秒、336 秒、931 秒都有。

    二、**抓完要驗實際長度**，對不上就重抓。加了 force-keyframes 也還是會
        間歇性只給前面一小段（單獨跑、沒有搶頻寬時也會發生）。

    長度沒驗到的後果不只是「取樣多寡不一」，而是**時間戳會錯**：
    呼叫端會把 start 加到窗內的秒數上，但超抓的那段畫面根本不在 start 的位置。
    實測有一支 908 秒的影片，粗掃記出了「第 1598 秒」的片段——
    那個時間點在影片裡不存在，拿去切會切到完全不相干的畫面。
    而且窗與窗互相重疊，同一個表情會被重複算成好幾個片段，
    整份統計會虛胖到看起來「這個人四種情緒都有」。
    """
    lo, hi = length * WINDOW_MIN_RATIO, length * WINDOW_MAX_RATIO
    last = 0.0
    for attempt in range(1, tries + 1):
        dst.unlink(missing_ok=True)
        proc = subprocess.run(
            ["yt-dlp", "--no-playlist", "--quiet", "--no-warnings",
             *format_args(max_height), "--merge-output-format", "mp4",
             "--download-sections", f"*{start:.0f}-{start + length:.0f}",
             "--force-keyframes-at-cuts",
             "-o", str(dst), url],
            capture_output=True, text=True)
        if proc.returncode != 0 or not dst.exists():
            if attempt == tries:
                raise RuntimeError(
                    f"yt-dlp 抓時間窗失敗（{start:.0f}s）："
                    f"{(proc.stderr or proc.stdout or '').strip()[-200:]}"
                )
            continue
        last = media_duration(dst)
        if lo <= last <= hi:
            return last
    raise RuntimeError(
        f"時間窗長度不對（{start:.0f}s 起要 {length:.0f} 秒，"
        f"重試 {tries} 次最後拿到 {last:.1f} 秒）。長度不對的窗不能用，"
        "因為呼叫端會把 start 加到窗內秒數上，超抓或少抓都會讓時間戳指錯地方。"
    )


def make_fer(model_name: str):
    """準備好 FER 辨識器。模型檔先自己下載好（見 ensure_model 的說明）。"""
    if HSEmotionRecognizer is None:
        raise SystemExit(
            "沒有安裝 hsemotion-onnx：pip install -r sourcing/requirements.txt"
        )
    ensure_model(model_name)
    return HSEmotionRecognizer(model_name=model_name)


def make_detector():
    """YuNet 人臉偵測器。

    ⚠️ 原本想用 OpenCV 內建的 Haar cascade（不用下載模型），但這個 image 裡是
       OpenCV 5.0，Haar 那套 `cv2.CascadeClassifier` 已經被移除了。
       YuNet 是 OpenCV 5 內建的替代品，準確度也高得多，只是要另外抓一個
       230KB 的 onnx。
    """
    cache = paths.MODEL_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    model = cache / "face_detection_yunet.onnx"
    if not model.exists():
        print("下載 YuNet 人臉偵測模型 …")
        urllib.request.urlretrieve(YUNET_URL, model)
        # 抓到空檔或 LFS 指標檔時要當場失敗，不要留一個壞檔在快取裡 ——
        # 否則下次執行會直接讀那個壞檔，錯誤訊息會指向 OpenCV 而不是下載
        if model.stat().st_size < 100_000:
            size = model.stat().st_size
            model.unlink()
            raise SystemExit(f"YuNet 模型只下載到 {size} 位元組，不像是真的模型檔。"
                             f"\n  網址：{YUNET_URL}")
        print(f"  → {model}（{model.stat().st_size / 1e3:.0f} KB）")
    # 輸入尺寸每張影格再設定，這裡先給個佔位值
    return cv2.FaceDetectorYN.create(str(model), "", (320, 320),
                                     score_threshold=0.7)


def largest_face(detector, frame_bgr):
    """回傳畫面中最大的一張臉的裁切圖，沒偵測到就回 None。

    取最大的那張：口播影片偶爾會拍到背景的人或海報上的臉，
    但說話者一定是畫面中最大的。
    """
    H, W = frame_bgr.shape[:2]
    detector.setInputSize((W, H))
    _, faces = detector.detect(frame_bgr)
    if faces is None or len(faces) == 0:
        return None
    box = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = (int(v) for v in box[:4])
    if w < MIN_FACE_PX or h < MIN_FACE_PX:
        return None
    # 往外擴一點：FER 模型吃的是含額頭與下巴的完整臉，切太緊會掉準確率
    pad = int(0.15 * w)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    return frame_bgr[y0:y1, x0:x1]


def scan_video(video: Path, fer, detector, sample_fps: float = SAMPLE_FPS,
               expect_seconds: float = None) -> list:
    """走過整支影片，回傳每個取樣點的 (秒數, 表情, 信心)。

    sample_fps 可以調：長影片用預設的 2 fps 就夠（要的是連續 3 秒的片段），
    但幾秒鐘的短片段那樣只取得到 6、7 張，這時要調高。

    expect_seconds 是這支影片**應該**有多長（從 metadata 來）。給了就會檢查
    實際掃到的長度，差太多就當場失敗。

    ⚠️ 這個檢查是踩過才加的：下載可能只抓到片頭就結束，而 yt-dlp 回傳成功。
       一支 17 分鐘的影片只抓到 17 秒，掃出來是一份看起來完全正常的結果 ——
       有臉 100%、有片段、有信心值，只是它只代表整支影片的 2%。
       這種「安靜的部分失敗」比整個爆掉危險得多，因為它會混進統計裡。
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"打不開影片：{video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / sample_fps)))

    out = []
    idx = 0
    while True:
        ok = cap.grab()          # grab 不解碼，跳過不要的影格快很多
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                face = largest_face(detector, frame)
                if face is not None and face.size:
                    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    label, scores = fer.predict_emotions(rgb, logits=False)
                    conf = float(np.max(scores))
                    out.append({
                        "t": round(idx / src_fps, 2),
                        "affect": label,
                        "label": AFFECT_TO_FIVE.get(label),
                        "confidence": round(conf, 4),
                    })
                else:
                    out.append({"t": round(idx / src_fps, 2), "affect": None,
                                "label": None, "confidence": None})
        idx += 1
    cap.release()
    # 一張影格都讀不到，多半是編碼解不了（例如 AV1）而不是影片本身有問題。
    # 不擋下來的話會變成「有臉 0%」這種看起來像結論、其實是故障的輸出。
    if not out:
        raise RuntimeError(
            f"讀不到任何影格（{video.name}）。多半是視訊編碼 OpenCV 解不了，"
            "確認 yt-dlp 抓到的是 avc1"
        )
    if expect_seconds:
        covered = out[-1]["t"]
        if covered < expect_seconds * MIN_COVERAGE:
            raise RuntimeError(
                f"只掃到 {covered:.0f} 秒，但這支影片有 {expect_seconds:.0f} 秒"
                f"（{covered / expect_seconds * 100:.0f}%）。下載沒抓完整，"
                "結果不能用。同時跑太多下載會這樣，但單獨跑也會 —— "
                "B站的下載本身就會間歇性地只給前面一小段。"
            )
    return out


def to_clips(samples: list) -> dict:
    """把逐張的判定收成連續片段。

    單張影格的表情不可信（眨眼、講話時的嘴型都會影響），但**連續好幾張
    都是同一種表情**就有意義了。這跟 L4A 要求「同一類至少 3 段」是同一個
    道理，只是這裡是連續性而不是次數。
    """
    clips = defaultdict(list)
    run_label, run_start, run_conf = None, None, []

    def flush(end_t):
        if run_label and run_label != "中性" and len(run_conf) >= MIN_RUN_FRAMES:
            clips[run_label].append({
                "start": run_start,
                "end": end_t,
                "frames": len(run_conf),
                "confidence": round(sum(run_conf) / len(run_conf), 4),
            })

    for s in samples:
        lab = s["label"] if (s["confidence"] or 0) >= MIN_CONF else None
        if lab == run_label:
            run_conf.append(s["confidence"])
            continue
        flush(s["t"])
        run_label, run_start, run_conf = lab, s["t"], (
            [s["confidence"]] if s["confidence"] else [])
    if samples:
        flush(samples[-1]["t"])

    for k in clips:
        clips[k].sort(key=lambda c: (-c["frames"], -c["confidence"]))
    return dict(clips)


def out_path(uid: str) -> Path:
    return paths.face_path(uid)


def process_one(row: dict, fer, detector) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "v.mp4"
        download_video(row["url"], video)
        samples = scan_video(video, fer, detector,
                             expect_seconds=row.get("duration_s"))

    clips = to_clips(samples)
    got = sorted(k for k in clips if k in TARGET)
    with_face = [s for s in samples if s["affect"]]

    result = {
        "uid": row["uid"],
        "url": row["url"],
        "title": row.get("title", ""),
        "author_name": row.get("author_name", ""),
        "author_id": row.get("author_id", ""),
        "duration_s": row.get("duration_s"),
        "n_samples": len(samples),
        "face_rate": round(len(with_face) / len(samples), 3) if samples else 0.0,
        "affect_dist": dict(Counter(s["affect"] for s in with_face)),
        "labels_present": got,
        "n_labels_present": len(got),
        "clip_counts": {k: len(v) for k, v in sorted(clips.items())},
        "clips": clips,
        "samples": samples,
    }
    p = out_path(row["uid"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def report() -> None:
    d = paths.FACE_DIR
    files = sorted(d.glob("*.json")) if d.exists() else []
    if not files:
        raise SystemExit(f"還沒有任何臉部分析結果（{d}）")
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    rows.sort(key=lambda r: (-r["n_labels_present"],
                             -sum(r["clip_counts"].get(k, 0) for k in TARGET)))

    print(f"已分析 {len(rows)} 支\n")
    print(f"{'四類':>4} {'有臉':>5}  {'哀':>3}{'怒':>4}{'樂':>4}{'驚':>4}   作者 / 標題")
    print("-" * 84)
    for r in rows:
        cc = r["clip_counts"]
        print(f"{r['n_labels_present']}/4 {r['face_rate'] * 100:4.0f}%  "
              f"{cc.get('哀', 0):3d}{cc.get('怒', 0):4d}{cc.get('樂', 0):4d}{cc.get('驚', 0):4d}   "
              f"{r['author_name'][:10]:12s} {r['title'][:32]}")

    full = [r for r in rows if r["n_labels_present"] == 4]
    print(f"\n四類都出現過的：{len(full)} 支")


def main():
    store.enable_utf8_stdout()
    p = argparse.ArgumentParser(description="L4B 臉部表情辨識")
    p.add_argument("--uid", action="append", default=[], help="只跑指定影片")
    p.add_argument("--from-timelines", action="store_true",
                   help="跑所有已經做過 L4A 的影片")
    p.add_argument("--limit", type=int)
    p.add_argument("--model", default="enet_b0_8_best_vgaf",
                   help="hsemotion 模型名（要 8 類的）")
    p.add_argument("--refresh", action="store_true", help="已跑過的也重跑")
    p.add_argument("--report", action="store_true", help="只看彙整")
    args = p.parse_args()

    if args.report:
        report()
        return

    meta = {r["uid"]: r for r in store.read_jsonl(paths.METADATA)}
    if args.uid:
        rows = [meta[u] for u in args.uid if u in meta]
    elif args.from_timelines:
        uids = [f.stem.replace("_", ":", 1) for f in paths.TIMELINE_DIR.glob("*.json")]
        rows = [meta[u] for u in uids if u in meta]
    else:
        raise SystemExit("要指定 --uid 或 --from-timelines")

    if not args.refresh:
        rows = [r for r in rows if not out_path(r["uid"]).exists()]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print("沒有待處理的，直接彙整\n")
        report()
        return

    print(f"要處理 {len(rows)} 支，FER 模型 = {args.model}")
    fer = make_fer(args.model)
    detector = make_detector()

    ok = failed = 0
    for i, row in enumerate(rows, 1):
        try:
            r = process_one(row, fer, detector)
        except Exception as exc:
            failed += 1
            store.append_error("face", row["uid"], f"{type(exc).__name__}: {exc}")
            print(f"  [{i}/{len(rows)}] {row['uid']} 失敗：{type(exc).__name__}: {exc}")
            continue
        ok += 1
        cc = r["clip_counts"]
        print(f"  [{i}/{len(rows)}] {row['uid']} 有臉 {r['face_rate'] * 100:.0f}%，"
              f"四類達 {r['n_labels_present']}（"
              + "、".join(f"{k}{cc.get(k, 0)}" for k in TARGET) + "）"
              f"　{row.get('title', '')[:24]}")

    print(f"\n完成 {ok} 支、失敗 {failed} 支\n")
    report()


if __name__ == "__main__":
    main()
