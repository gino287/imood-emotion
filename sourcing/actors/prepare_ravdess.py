# 待GINO改寫
"""把 RAVDESS 的演員影片整理成「同一人 × 五個情緒資料夾」的乾淨小樣本。

    python sourcing/actors/prepare_ravdess.py
    python sourcing/actors/prepare_ravdess.py --per-emotion 1
    python sourcing/actors/prepare_ravdess.py --actors 02 04 08

輸出：
    _local/sourcing/clips_ravdess/
        Actor_02（女）/ 預設/ 喜/ 怒/ 哀/ 樂/     每個資料夾 1–2 個檔案
        manifest.json   每個檔案是誰的哪一種情緒、選它的理由
        LICENSE.txt

**為什麼走這條路。**
野生素材那條路已經量到底了：JoyGen 的清單是一人一支（857 支 = 857 個作者），
單支影片只有一種情緒基調，而且跨同一人的多支影片也不行 ——
四月吨吨_ 四支約 100 分鐘（含「人生中最漫長的七年」這種題材）
仍然只有樂、哀怒驚全是 0。情緒基調是**鏡頭前人設**的屬性。

演員資料集直接繞開這個問題：同一個人把每種情緒都演一遍，是拍出來的。

RAVDESS 的好處是檔名就寫著情緒，不需要任何偵測：
    02-01-06-01-02-01-12.mp4
    │  │  │  │  │  │  └─ 演員 01–24（單數男、雙數女）
    │  │  │  │  │  └──── 重複次數 01/02
    │  │  │  │  └─────── 台詞 01/02
    │  │  │  └────────── 強度 01普通 02強烈
    │  │  └───────────── 情緒 01中性 02平靜 03開心 04悲傷 05憤怒 06恐懼 07厭惡 08驚訝
    │  └──────────────── 聲道 01語音 02歌唱
    └─────────────────── 模態 01視聽 02純視訊 03純音訊

⚠️ **授權**：RAVDESS 是 CC BY-NC-SA 4.0，**非商業用途**。
   拿來做技術測試沒問題，要進產品得另外取得商用授權。這件事要先講清楚。
"""
import argparse
import json
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sourcing.common import paths, store  # noqa: E402
from sourcing.mapping.schemes import (  # noqa: E402
    DELIVERY_LABELS,
    RAVDESS_TO_DELIVERY,
)

OUT_DIR = paths.ACTOR_CLIP_DIR

# 規格：聲道只要語音、模態只要視聽版
WANT_MODALITY = "01"     # 01 視聽（有畫面也有聲音）
WANT_CHANNEL = "01"      # 01 語音（不要唱歌）

# ---------------------------------------------------------------------------
# 同一類有好幾個檔案時，選哪一個
# ---------------------------------------------------------------------------
# 三個維度各自的偏好，排在前面的優先。排序時把「偏好順位」當分數，越小越先取。
#
# 強度：情緒四類一律優先取 02（強烈）。
#   這批是要讓下游一眼看出「這是什麼情緒」的樣本，普通強度的演出有不少接近
#   面無表情，FER 也常判成 Neutral，當素材的識別度不夠。
#   ⚠️ 例外是「預設」：neutral(01) 本身就沒有強度 02，而且我們要的正是
#      「這個人沒有情緒時的臉」，所以那一類反過來優先取普通強度的 neutral，
#      而不是 calm 的強烈版。
#
# 台詞：固定優先 statement 01（"Kids are talking by the door"）。
#   兩句台詞的音節不同會影響嘴型。固定台詞之後，同一位演員五個資料夾之間
#   唯一的變數就是情緒 —— 下游要比對表情或嘴型差異時，這一點很重要。
#
# 重複次數：先 01 再 02。同台詞同強度的第二次重複，是差異最小的第二個樣本，
#   拿來當「每類要兩個檔案」的第二個剛好。
INTENSITY_PREF = ["02", "01"]           # 情緒四類
INTENSITY_PREF_DEFAULT = ["01", "02"]   # 預設那一類
STATEMENT_PREF = ["01", "02"]
REPEAT_PREF = ["01", "02"]

# 「預設」資料夾裡 neutral 優先於 calm：neutral 才是真正的無表情基準，
# calm 是「平靜但有情緒」，兩者放同一個資料夾是規格指定的，但取樣要分先後。
EMOTION_PREF_DEFAULT = ["01", "02"]

# 女演員是雙數編號。7 女 3 男的預設選角，理由見 README：
# 保留已經抓過並用 FER 驗過的 02 與 08，其餘在 01–24 之間平均散開，
# 避免全部集中在編號相近的人（RAVDESS 的錄製是分批進行的）。
DEFAULT_ACTORS = ["02", "04", "08", "12", "16", "20", "24",   # 女
                  "01", "11", "21"]                            # 男


def parse_name(name: str) -> dict:
    """從檔名拆出各欄位。不符合格式的回 None。"""
    parts = Path(name).stem.split("-")
    if len(parts) != 7:
        return None
    modality, channel, emotion, intensity, statement, repeat, actor = parts
    return {
        "modality": modality, "channel": channel, "emotion": emotion,
        "intensity": intensity, "statement": statement,
        "repeat": repeat, "actor": actor,
    }


def rank(meta: dict, label: str) -> tuple:
    """排序用的偏好分數。越小越優先，同分時檔名決定，結果是可重現的。"""
    is_default = label == "預設"
    intensity_pref = INTENSITY_PREF_DEFAULT if is_default else INTENSITY_PREF

    def idx(pref: list, value: str) -> int:
        return pref.index(value) if value in pref else len(pref)

    return (
        # 預設那一類 neutral 先於 calm；其他類只有一個情緒代號，這一項恆為 0
        idx(EMOTION_PREF_DEFAULT, meta["emotion"]) if is_default else 0,
        idx(intensity_pref, meta["intensity"]),
        idx(STATEMENT_PREF, meta["statement"]),
        idx(REPEAT_PREF, meta["repeat"]),
    )


def why(meta: dict, label: str) -> str:
    """把選這個檔案的理由寫進 manifest，之後回頭看不必再推一次。"""
    bits = ["強烈" if meta["intensity"] == "02" else "普通強度"]
    if label == "預設":
        bits.append("neutral" if meta["emotion"] == "01" else "calm")
    bits.append(f"台詞{meta['statement']}")
    bits.append(f"第{meta['repeat']}次")
    return "／".join(bits)


def unzip_all(zip_dir: Path, work: Path, actors: list) -> None:
    zips = sorted(zip_dir.glob("Video_Speech_Actor_*.zip"))
    if actors:
        zips = [z for z in zips if z.stem.split("_")[-1] in actors]
    if not zips:
        raise SystemExit(
            f"{zip_dir} 底下找不到需要的 Video_Speech_Actor_*.zip\n"
            "從 https://zenodo.org/records/1188976 下載（每位演員一個 zip）"
        )
    work.mkdir(parents=True, exist_ok=True)
    for z in zips:
        marker = work / f".{z.stem}.done"
        if marker.exists():
            continue
        # 還在下載的 zip 要當場擋下來，不要讓 zipfile 丟原始的 BadZipFile ——
        # 那個訊息看起來像「檔案壞了」，實際上多半只是還沒下載完。
        # 用「打不打得開」判斷而不是用檔案大小：各演員的 zip 大小本來就不一樣
        # （實測 495–544MB），而 zip 的中央目錄在檔尾，沒下載完就一定打不開。
        if not zipfile.is_zipfile(z):
            size = z.stat().st_size / 1048576
            raise SystemExit(
                f"{z.name} 打不開（目前 {size:.0f}MB）。\n"
                "多半是還在下載中 —— 等它下載完再跑一次。\n"
                "確認方式：python -c \"import zipfile;print(zipfile.is_zipfile('"
                f"{z.as_posix()}'))\""
            )
        print(f"  解壓 {z.name} …")
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if n.endswith(".mp4")]
            zf.extractall(work)
        # 一位演員應該有 120 個檔（60 視聽 + 60 純視訊）。少了就是 zip 不完整，
        # 這時不能寫 .done —— 寫了下次就會跳過，缺的檔案再也補不回來。
        if len(names) < 100:
            raise SystemExit(
                f"{z.name} 只有 {len(names)} 個 mp4，預期 120。zip 可能不完整，"
                "刪掉重新下載。"
            )
        marker.write_text("ok", encoding="utf-8")


def main():
    store.enable_utf8_stdout()
    p = argparse.ArgumentParser(description="整理 RAVDESS 成同一人五種情緒的素材夾")
    p.add_argument("--zip-dir", type=Path, default=paths.REPO_ROOT / "_local" / "ravdess")
    p.add_argument("--actors", nargs="*", default=DEFAULT_ACTORS,
                   help="要收哪幾位演員（兩位數編號）。給空字串代表全部")
    p.add_argument("--per-emotion", type=int, default=2,
                   help="每個情緒資料夾放幾個檔案（預設 2）")
    args = p.parse_args()

    actors = [a for a in args.actors if a]
    work = args.zip_dir / "_extracted"
    unzip_all(args.zip_dir, work, actors)

    files = sorted(work.rglob("*.mp4"))
    print(f"\n解出 {len(files)} 個影片檔")

    skipped = defaultdict(int)
    by_actor = defaultdict(lambda: defaultdict(list))
    for f in files:
        meta = parse_name(f.name)
        if not meta:
            skipped["檔名格式不符"] += 1
            continue
        if actors and meta["actor"] not in actors:
            skipped["不在選角名單"] += 1
            continue
        if meta["modality"] != WANT_MODALITY:
            skipped["非視聽版"] += 1
            continue
        if meta["channel"] != WANT_CHANNEL:
            skipped["唱歌"] += 1
            continue
        label = RAVDESS_TO_DELIVERY.get(meta["emotion"])
        if not label:
            skipped["恐懼／厭惡（規格不抓）"] += 1
            continue
        by_actor[meta["actor"]][label].append((f, meta))

    print("篩掉的：" + "、".join(f"{k} {v}" for k, v in skipped.items()))

    if not by_actor:
        raise SystemExit("沒有符合條件的檔案，檢查 --actors 與 zip 是否都下載了")

    manifest = []
    incomplete = []
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    for actor, emos in sorted(by_actor.items()):
        gender = "女" if int(actor) % 2 == 0 else "男"
        name = f"Actor_{actor}（{gender}）"
        got = []
        for label in DELIVERY_LABELS:
            items = sorted(emos.get(label, []), key=lambda x: (rank(x[1], label), x[0].name))
            items = items[:args.per_emotion]
            if not items:
                continue
            got.append(f"{label}{len(items)}")
            for src, meta in items:
                dst = OUT_DIR / name / label / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                manifest.append({
                    "file": str(dst.relative_to(OUT_DIR)).replace("\\", "/"),
                    "actor": f"Actor_{actor}",
                    "gender": gender,
                    "label": label,
                    "ravdess_emotion_code": meta["emotion"],
                    "intensity": "強烈" if meta["intensity"] == "02" else "普通",
                    "statement": meta["statement"],
                    "repeat": meta["repeat"],
                    "picked_because": why(meta, label),
                    "source": "RAVDESS (CC BY-NC-SA 4.0, 非商業)",
                })
        missing = [k for k in DELIVERY_LABELS if k not in emos]
        if missing:
            incomplete.append((name, missing))
        print(f"  {name}：{'　'.join(got)}"
              + (f"　⚠️ 缺 {'/'.join(missing)}" if missing else ""))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "LICENSE.txt").write_text(
        "素材來源：RAVDESS\n"
        "https://zenodo.org/records/1188976\n"
        "授權：Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)\n"
        "\n"
        "⚠️ 非商業授權。技術測試可用，要用在產品上必須另外取得商用授權。\n"
        "引用：Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of\n"
        "Emotional Speech and Song (RAVDESS). PLoS ONE 13(5): e0196391.\n",
        encoding="utf-8")

    n_actors = len(by_actor)
    n_female = sum(1 for a in by_actor if int(a) % 2 == 0)
    print(f"\n共 {len(manifest)} 個檔案，{n_actors} 位演員"
          f"（女 {n_female}、男 {n_actors - n_female}）")
    if incomplete:
        print("⚠️ 有人的資料夾不齊：")
        for name, missing in incomplete:
            print(f"    {name} 缺 {'/'.join(missing)}")
    else:
        print(f"每位都有完整的五個資料夾：{'／'.join(DELIVERY_LABELS)}")
    print(f"\n→ {OUT_DIR}")
    print("交叉驗證（用 FER 檢查臉部判定跟檔名對不對得上）："
          "\n  python sourcing/actors/verify_ravdess.py")


if __name__ == "__main__":
    main()
