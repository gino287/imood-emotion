# 待GINO改寫
"""所有輸入輸出的位置都在這一個檔案。想知道「某一步的結果存在哪」看這裡就好。

輸出一律放 `_local/sourcing/`，因為那是執行期產物、.gitignore 已排除 `_local/`。
每支腳本都從這裡拿路徑，不自己拼字串 —— 不然改一個資料夾名要改十個檔案，
而且拼錯只會安靜地寫到別的地方去。

底下的常數依照流程順序排，順著讀就是整條線的資料流。
"""
from pathlib import Path

# sourcing/common/paths.py → 往上三層才是專案根目錄
REPO_ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = REPO_ROOT / "_local" / "sourcing"

# ---------------------------------------------------------------------------
# 輸入
# ---------------------------------------------------------------------------
# JoyGen repo Dataset 段落給的兩份裸網址清單（只有網址、零 metadata）
DEFAULT_URL_FILES = [
    REPO_ROOT / "tmp" / "bili_urls.txt",
    REPO_ROOT / "tmp" / "dy_urls.txt",
]

# ---------------------------------------------------------------------------
# collect/：清單與 metadata
# ---------------------------------------------------------------------------
SOURCES = OUT_DIR / "sources.jsonl"        # normalize_urls：正規化後的影片清單
RAW_DIR = OUT_DIR / "raw"                  # fetch_*：API 原始回應，一支一檔、不可變
METADATA = OUT_DIR / "metadata.jsonl"      # fetch_*：抽取後的統一欄位
FETCH_ERRORS = OUT_DIR / "fetch_errors.jsonl"   # fetch_*：失敗紀錄，方便重跑
AUTHORS = OUT_DIR / "authors.jsonl"        # group_authors：以作者分組的排行
CHANNEL_DIR = OUT_DIR / "channels"         # fetch_channel：某位作者的全部作品
FILTERED = OUT_DIR / "filtered.jsonl"      # coarse_filter：粗篩結果（含淘汰理由）
COVER_DIR = OUT_DIR / "covers"             # covers：封面縮圖
SHEET_DIR = OUT_DIR / "sheets"             # covers：封面縮圖牆

# ---------------------------------------------------------------------------
# detect/：情緒時間軸
# ---------------------------------------------------------------------------
TIMELINE_DIR = OUT_DIR / "timelines"       # text_timeline：文字（STT + BERT）
FACE_DIR = OUT_DIR / "face"                # face_timeline：臉部（FER）
MODEL_CACHE = OUT_DIR / "models"           # 下載下來的 FER 與人臉偵測模型檔

# ---------------------------------------------------------------------------
# deliver/：交付物
# ---------------------------------------------------------------------------
SHORTLIST = OUT_DIR / "shortlist.jsonl"    # shortlist：候選影片排序
MATERIAL_TABLE = OUT_DIR / "material_table.json"   # 素材表（人 → 情緒 → 切點）
REVIEW_PAGE = OUT_DIR / "review.html"      # make_review_page：人工看表情的檢視頁
CLIP_DIR = OUT_DIR / "clips"               # cut_clips：野生素材切出來的片段
ACTOR_CLIP_DIR = OUT_DIR / "clips_ravdess"  # prepare_ravdess：演員素材整理結果

# 人工標記 sidecar。與 metadata 分開的理由：篩選規則會改、會重跑，
# 但人工看過的判斷不能被覆蓋掉。
LABELS = OUT_DIR / "labels.jsonl"


def raw_path(platform: str, vid: str) -> Path:
    """某支影片的 API 原始回應位置。續跑就是靠這個檔案存不存在判斷的。"""
    return RAW_DIR / platform / f"{vid}.json"


def uid_to_filename(uid: str) -> str:
    """uid 當檔名用。`bili:BV1xxx` 的冒號在 Windows 不能當檔名，換成底線。"""
    return uid.replace(":", "_")


def timeline_path(uid: str) -> Path:
    return TIMELINE_DIR / f"{uid_to_filename(uid)}.json"


def face_path(uid: str) -> Path:
    return FACE_DIR / f"{uid_to_filename(uid)}.json"
