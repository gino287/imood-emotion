> 待GINO改寫

# sourcing — JoyGen 情緒素材：映射與取材

> ## ⚠️ 實驗中
>
> **這個資料夾還在進行式，尚未整理成可以直接使用的模組。**
> 流程、門檻與判準都還在調整，介面隨時會變，跑出來的中間結果也還在驗證階段。
> 要沿用這裡的任何東西之前，先看 [`ROUTES.md`](ROUTES.md) 確認目前走到哪一步。

要的東西是：**同一個人、五種情緒（中性/喜/怒/哀/樂）都有、正臉單人、有說話。**

JoyGen 只吃「一段影片 + 一段驅動音訊」，情緒是**素材自帶的**，模型不會生成情緒。
所以這整個資料夾在做的事就兩件：**把各家的情緒標籤收斂成我們的五類（映射）**，
以及**去哪裡找到符合條件的素材（取材）**。

離線流程，與線上推論無關。依賴獨立在 `sourcing/requirements.txt`。

- 路線的比較與選擇 → [`ROUTES.md`](ROUTES.md)
- 每個檔案能做什麼 → [最後一節「檔案逐一說明」](#檔案逐一說明)

---

## 資料夾

```
sourcing/
├── common/     路徑與存檔。所有輸出位置集中在 common/paths.py
├── mapping/    ★ 映射：四套標籤體系的對照表與檢查工具（全部不連網、秒級）
├── collect/    取材路線 B 前半：裸網址清單 → 有 metadata 的候選清單
├── detect/     判情緒：文字（BERT）與臉部（FER）兩條並列，可互相對照
├── deliver/    交付物：排序 → 人工檢視頁 → 切成實際 mp4
├── actors/     取材路線 A：演員資料集（RAVDESS／MEAD 這類拍好的素材）
├── ROUTES.md   路線圖：素材從哪來、情緒怎麼判、同一人多情緒怎麼滿足
└── README.md   這一份
```

每個資料夾的 `__init__.py` 都寫了那一層在做什麼、有哪些檔案、踩過哪些坑。
`python -c "import sourcing.detect; print(sourcing.detect.__doc__)"` 就看得到。

---

## 兩條取材路線，現在的狀態

| | 路線 A｜演員資料集 | 路線 B｜自己抓 B 站 |
|---|---|---|
| 資料夾 | `actors/` | `collect/` → `detect/` → `deliver/` |
| 同一人五種情緒 | ✅ 天生滿足 | ❌ 實測最多兩種 |
| 情緒怎麼來 | 檔名就寫著，不需偵測 | 要自己跑 FER |
| 情緒真不真 | 演的（表情清楚好偵測） | 真的 |
| 授權 | ⚠️ 非商業（RAVDESS） | 平台內容，另有版權問題 |
| 交付狀態 | **已交付 40 段（2 位女演員 × 5 情緒 × 4 段）** | 已交付 27 段（5 人，各最多 2 情緒） |

**路線 B 湊不齊的三個原因都是量出來的**，不是推測 —— 見 `collect/__init__.py`
或 [`ROUTES.md`](ROUTES.md#b-自己從平台抓--已驗證湊不齊)。

---

## 怎麼跑

### 環境

```bash
docker compose -f docker/docker-compose.yml build sourcing
DC="docker compose -f docker/docker-compose.yml run --rm sourcing"
```

抖音需要 cookie（公開影片也擋，實測訪客模式 0% 成功率）。放進專案根目錄的 `.env`：

```
DOUYIN_COOKIE=從瀏覽器複製的 document.cookie
BILI_SESSDATA=選填，B站公開影片不需要
```

`mapping/` 底下那幾支不需要容器也不需要網路，本機 python 直接跑就行。

### 最短路徑：只想拿到能交出去的素材

```bash
# 先下載 RAVDESS 演員 zip（女演員是雙數編號 02 04 … 24，一位約 500MB）
curl -L -o _local/ravdess/Video_Speech_Actor_04.zip \
  "https://zenodo.org/records/1188976/files/Video_Speech_Actor_04.zip?download=1"

python sourcing/actors/prepare_ravdess.py --strong-only --per-emotion 4
# → _local/sourcing/clips_ravdess/Actor_XX（女）/情緒/*.mp4
```

### 先理解映射在做什麼（不連網，幾秒鐘）

```bash
python sourcing/mapping/show_mapping.py            # 四套標籤體系怎麼對到五類
python sourcing/mapping/show_mapping.py --reverse  # 每個五類是從哪些原生類別來的
python sourcing/mapping/crosstab.py                # 映射的實證依據（吃 eval 的結果）
python sourcing/mapping/crosstab.py --five         # 套上映射後長什麼樣
python sourcing/mapping/crosstab.py --valence-gate # 走不通的那條（拆驚→喜/哀）
python sourcing/mapping/compare.py --all --matrix  # 文字判的 vs 臉部判的
```

### 路線 B：完整漏斗

成本由低到高，每一層只處理上一層活下來的。**放大之前一定先跑小樣本。**

```bash
# ── collect：清單與 metadata ────────────────────────────────
$DC python sourcing/collect/normalize_urls.py            # → 1188 支（斷言）
$DC python sourcing/collect/fetch_bilibili.py --limit 3  # 小樣本先驗欄位
$DC python sourcing/collect/fetch_bilibili.py            # 全量，約 35 分鐘，會續跑
$DC python sourcing/collect/fetch_douyin.py              # 需要 DOUYIN_COOKIE
$DC python sourcing/collect/group_authors.py             # 以人為單位分組
$DC python sourcing/collect/coarse_filter.py             # 免下載粗篩
$DC python sourcing/collect/covers.py --download --sheet # 封面縮圖牆，肉眼快篩

# ── deliver：先排序，決定接下來跑哪幾支 ─────────────────────
$DC python sourcing/deliver/shortlist.py --female-only --pick 8

# ── detect：判情緒（臉是主力，文字只當對照） ────────────────
$DC python sourcing/detect/face_timeline.py --uid bili:BV1aM4y117kD
$DC python sourcing/detect/face_timeline.py --report
$DC python sourcing/detect/text_timeline.py --verdict keep --limit 5   # 對照組

# ── deliver：產出交付物 ─────────────────────────────────────
$DC python sourcing/deliver/make_review_page.py --min-labels 3
$DC python sourcing/deliver/cut_clips.py --all --per-emotion 3
```

鎖定某個人之後，抓他頻道的其他作品：

```bash
$DC python sourcing/collect/fetch_channel.py --mid 20960244 --merge
```

---

## 輸出檔案在哪

全部在 `_local/sourcing/`（不進版控）。位置定義在 `common/paths.py`。

| 檔案／資料夾 | 誰產的 | 內容 |
|---|---|---|
| `sources.jsonl` | `collect/normalize_urls.py` | 正規化後的影片清單（1188 支） |
| `raw/{platform}/{vid}.json` | `collect/fetch_*.py` | API 原始回應，**不可變**，續跑靠它 |
| `metadata.jsonl` | `collect/fetch_*.py` | 抽取後的統一欄位，一支一列 |
| `fetch_errors.jsonl` | `collect/fetch_*.py` | 失敗紀錄（附加寫入，歷次都留） |
| `authors.jsonl` | `collect/group_authors.py` | 以作者分組的排行 |
| `channels/{mid}.jsonl` | `collect/fetch_channel.py` | 某位作者的全部作品 |
| `filtered.jsonl` | `collect/coarse_filter.py` | 粗篩結果（含淘汰理由） |
| `covers/` `sheets/` | `collect/covers.py` | 封面縮圖與縮圖牆 |
| `timelines/{uid}.json` | `detect/text_timeline.py` | 文字情緒時間軸 + 逐字稿 |
| `face/{uid}.json` | `detect/face_timeline.py` | 臉部情緒時間軸（逐影格 + 連續片段） |
| `models/` | `detect/face_timeline.py` | 下載下來的 FER 與 YuNet 模型檔 |
| `shortlist.jsonl` | `deliver/shortlist.py` | 候選影片排序 |
| `review.html` | `deliver/make_review_page.py` | 人工檢視頁（單一檔案，可直接開） |
| `material_table.json` | `detect/text_timeline.py --report` | 素材表：人 → 情緒 → 切點 |
| `clips/{作者}/{情緒}/*.mp4` | `deliver/cut_clips.py` | 野生素材片段 + manifest.json |
| `clips_ravdess/Actor_XX（女）/{情緒}/*.mp4` | `actors/prepare_ravdess.py` | 演員素材 + manifest + LICENSE |

---

## 設計上的幾個判斷

以下是實作時做過的取捨，寫在這裡是為了之後回頭看得懂為什麼這樣做。

### 為什麼存檔要分 raw 與 metadata 兩層

抓取很貴（1188 支、節流之下要跑二十幾分鐘），抽取很便宜。混在一起的話，
之後想多留一個欄位就得整批重抓。留著 raw，改抽取邏輯只要 `--rebuild`，
不必再碰網路。續跑也是靠 raw 判斷（檔案存在就跳過），所以中斷了直接重跑同一行。

### 真正有效的篩選訊號是「題材」，不是「片型」

`coarse_filter.py` 的片型篩選在這批素材上幾乎沒作用（857 支只刷掉 6.3%），
因為這份清單本來就是 JoyGen 挑過的單人說話影片。

實測有效的是**題材 + 長度**：

- 「經歷過校園暴力和社恐」「我的三段裸辭經歷」這種**人生經歷型敘事**
  → 非中性佔比 40–50%
- 「聊聊我在大專工作半年的感受」這種**平穩經驗分享型口播**
  → 非中性只有 20%，而且那幾段全是偽陽性

### 為什麼關鍵字找的是「片型」不是「情緒」

metadata 看不出情緒，但看得出片型。標題裡寫「開心」「生氣」的影片極少，
但寫「經驗分享」「vlog」「口播」的很多 —— 這些代表「一個人對著鏡頭連續講話」，
正是需要的素材形態。

B 站的 UP 主自訂標籤（`tags`）比標題好用得多，訪客模式也拿得到。
反倒是分區名（`tname`）訪客模式回空字串，只剩數字 `tid`，所以程式不硬編
tid 對照表，而是把分布印出來讓人決定要排除哪些。

### 為什麼封面縮圖牆值得做

封面是 metadata 階段唯一能看到「畫面」的東西，而且一千多張只有幾十 MB。
一頁排 100 張，肉眼掃 30 秒就能刷掉一半「根本沒有人臉／是動畫／是遊戲畫面」
的影片。這件事關鍵字做不到。

### 為什麼判準是「連續性」而不是「信心值」

第一支端到端實測就打臉了「至少 1 段就算數」這個直覺做法：一支 3 分 43 秒的
平穩口播，109 段裡 78 段是平淡語氣，但樂／哀／怒各湊到 1 段、驚 2 段，
於是被判成「五類全齊」。去看逐字稿那幾段全是中性內容
（「到現在其實也快乾滿兩個學期了」被判成哀，信心還有 **0.912**）。

**偽陽性照樣拿得到 0.9 的信心，所以提高信心門檻擋不住。**
有效的判準是「有沒有連續好幾段都是同一種情緒」—— 真的在生氣的人不會只氣一句話。

臉那邊同一個道理：說話張嘴會被 FER 判成「驚訝」（影格層級可以到 26–51%），
但那是散落的單張，連不成 3 秒。

### 為什麼最後只採信臉那一邊

BERT 讀的是**文字語意**，JoyGen 要的是**臉部表情**。人可以面無表情地說
「我好生氣」，也可以笑著講難過的事。拿 5 支人工看過的影片核對：
臉部辨識 5/5 吻合，文字模型至少 2 支嚴重錯誤。

要自己看這個差距：`python sourcing/mapping/compare.py --all --matrix`

### 為什麼 B 站不用 f2

f2 目前只支援抖音／TikTok／Twitter／微博，B 站排在 `0.0.1.8` 還沒實作。
也沒用 yt-dlp 抓 metadata：官方 API 一個請求就給 `owner.mid`（分組主鍵）
與 `cid`（取字幕用），而且 yt-dlp 在 B 站有已知的 412 問題。
yt-dlp 只在需要下載媒體時才用（`detect/` 與 `deliver/cut_clips.py`）。

---

## 檔案逐一說明

每支腳本都可以單獨執行，`--help` 有完整參數。下面按資料夾列。

### common/ — 共用基礎（不能執行，只被 import）

<table>
<tr><th align="left">檔案</th><th align="left">功能</th></tr>
<tr><td><code>paths.py</code></td><td>

**所有輸入輸出的位置。想知道「某一步的結果存在哪」看這個檔案就好。**
常數依流程順序排（collect → detect → deliver），另有
`raw_path()` / `timeline_path()` / `face_path()` 三個小函式。
每支腳本都從這裡拿路徑，不自己拼字串。

</td></tr>
<tr><td><code>store.py</code></td><td>

抓取階段共用的存檔與續跑邏輯。

- `enable_utf8_stdout()` — **每支 CLI 的 `main()` 第一行都要呼叫。**
  這批素材的標題幾乎都是簡體，Windows 終端機是 cp950，碰到「点」「洁」
  會直接 `UnicodeEncodeError` 整支崩掉（不是印成亂碼，是崩掉）
- `read_jsonl` / `write_jsonl` — jsonl 讀寫
- `load_sources()` — 讀 L0 的清單，缺檔時給明確指示
- `has_raw` / `save_raw` / `load_raw` / `iter_raw` — raw 層存取，續跑靠這個
- `merge_metadata()` — 以 uid 為主鍵併回 metadata.jsonl，不會產生重複列
- `append_error()` — 失敗紀錄，附加寫入（同一支重試多次的原因都留著）
- `Throttle` — 兩次請求的最小間隔。用「距離上次多久」而不是固定 sleep，
  因為請求本身就要花時間

</td></tr>
</table>

### mapping/ — 映射（不連網、不吃 GPU、秒級）

想理解整條線在判什麼、為什麼這樣判，從這裡開始最快。

<table>
<tr><th align="left">檔案</th><th align="left">功能與用法</th></tr>
<tr><td><code>schemes.py</code></td><td>

**三張映射表都在這裡**（第四張 `NATIVE_TO_FIVE` 在 `emotion/labels.py`，
因為上線會用到，這裡 re-export 過來讓四張表能並排看）：

- `AFFECT_TO_FIVE` — FER 的 AffectNet 8 類 → 五類
- `RAVDESS_TO_FIVE` — RAVDESS 檔名代號 → 五類
- `FIVE_TO_SMP` — 五類 → SMP2020 六類代號（要跟評測數字對照時用）
- `TARGET` — 要去找的四類（不含中性，每支影片都一堆中性，當條件沒意義）

只放對照關係，不做任何運算。載入時就斷言值域落在五類之內 ——
打錯字當場炸掉，而不是跑完幾百支影片才發現某一類永遠是 0。

</td></tr>
<tr><td><code>show_mapping.py</code></td><td>

把四套體系攤開來印。**要確認「怒到底吃進了什麼」跑這支。**

```bash
python sourcing/mapping/show_mapping.py            # 正向表 + 反向表
python sourcing/mapping/show_mapping.py --reverse  # 只看反向表
```

反向表是實際看的時候最有用的：正向是「這一類要去哪」，反向是「這一類從哪來」。
譬如「怒」在文字端吃了憤怒語調＋厭惡語調、在臉部端吃了 Anger＋Disgust ——
兩邊都是兩個來源合成的，所以「怒」的數字天生比其他類容易偏高。

</td></tr>
<tr><td><code>crosstab.py</code></td><td>

映射表的**實證依據**：拿 eval 跑完的 `predictions.jsonl` 重算
「模型判這一類時，真實標籤實際上是什麼」。不重跑模型，所以是零成本的。

```bash
python sourcing/mapping/crosstab.py                            # 原生 8 類交叉表
python sourcing/mapping/crosstab.py --model johnson-large --device cuda
python sourcing/mapping/crosstab.py --five                     # 套上映射後的五類
python sourcing/mapping/crosstab.py --valence-gate             # 走不通的那條
```

`--five` 可以直接看到「哀」吸進了多少不是 sad 的東西（341 筆預測裡只有 32%
真的是 sad，在吸收 fear 與 angry）。

`--valence-gate` 保留的是一條**走不通**的路，留著是為了不要再走第二次：
用 `raw_probs` 的殘餘機率把「驚奇語調」拆成正向（→喜）與負向（→哀）。
實測兩組分布幾乎一樣，門檻拉高甚至反相關 —— softmax 底下非勝出類的機率
是雜訊，沒有為「當第二個軸讀」校準過。想自己確認就跑這個旗標。

⚠️ 測試集是 SMP2020-EWECT，**微博書面文字**。產品的實際輸入是口語逐字稿，
兩者領域不同（「疑問語調」那一類就是這樣被推翻的）。這裡的數字用來做
映射的相對比較可以，不要當成產品準確率。

</td></tr>
<tr><td><code>compare.py</code></td><td>

同一支影片，**文字判的 vs 臉部判的**，並排對照。
「文字判不出表情」這個結論的量化版本。

```bash
python sourcing/mapping/compare.py --uid bili:BV17z4y117cR   # 逐片段明細
python sourcing/mapping/compare.py --all --matrix            # 只看總表
```

做法：拿文字時間軸的每一個片段，去臉部那份的逐張影格裡撈出落在同一個
時間窗內的判定，看多數票是什麼。不重跑模型，兩邊都跑過的影片才比得出來
（`--all` 會自己挑交集）。

⚠️ 不一致不代表某一邊壞了 —— 兩邊量的本來就是不同的東西。這支的用途是
**知道差多少、差在哪**，不是把兩邊調成一致。

</td></tr>
</table>

### collect/ — 取材路線 B 前半：裸網址 → 候選清單

<table>
<tr><th align="left">檔案</th><th align="left">功能與用法</th></tr>
<tr><td><code>normalize_urls.py</code></td><td>

兩份 txt → `sources.jsonl`。**不連網。** 處理原始清單的三個坑：

1. `bili_urls.txt` 裡混了 10 行抖音，而且 ID 藏在 `modal_id=` 參數裡 ——
   只用 `grep BV` 會安靜地漏掉 9 支
2. 兩份都有重複行（B 站 1002 行含 BV 但只有 1000 支）
3. 網址帶追蹤參數（`spm_id_from`、`vd_source`），不清掉去不了重

總數 1188 寫成**斷言**，對不上就是解析漏了東西。

```bash
python sourcing/collect/normalize_urls.py
python sourcing/collect/normalize_urls.py --dry-run   # 只印統計不寫檔
```

</td></tr>
<tr><td><code>fetch_bilibili.py</code></td><td>

打官方 `/x/web-interface/view`，公開影片免登入，一支一請求（沒有批次端點）。
順便打 `/x/tag/archive/tags` 拿 UP 主自訂標籤 —— 多花一倍請求數是值得的，
訪客模式下分區名回空字串，標籤才是有用的訊號。

```bash
python sourcing/collect/fetch_bilibili.py --limit 3   # 小樣本先驗欄位
python sourcing/collect/fetch_bilibili.py             # 全量約 35 分鐘，會續跑
python sourcing/collect/fetch_bilibili.py --rebuild   # 不連網，只從 raw 重抽欄位
python sourcing/collect/fetch_bilibili.py --no-tags   # 省一半請求數
```

其他參數：`--interval`（請求間隔，預設 1.0 秒）、`--retries`、`--refetch`。

</td></tr>
<tr><td><code>fetch_douyin.py</code></td><td>

走 f2 的 `DouyinHandler`。抖音公開影片也多半要 cookie（放 `.env` 的
`DOUYIN_COOKIE`）。沒 cookie 也能跑，腳本會把成功率印出來 ——
量到實際數字再決定要不要補憑證，不用先猜（實測訪客模式 0%）。

⚠️ 抖音的 `duration` 單位是**毫秒**、B 站是**秒**，這裡統一換算成秒再存。

```bash
python sourcing/collect/fetch_douyin.py --limit 3
python sourcing/collect/fetch_douyin.py --rebuild
```

</td></tr>
<tr><td><code>group_authors.py</code></td><td>

`GROUP BY author_id`，產出候選人物排行。**不連網。**

**這支的結論比它原本的用途重要**：857 支跑出來是 857 個不同作者、零重複，
所以「在清單內找同一個人的多支影片」這條路結構上就不存在。

```bash
python sourcing/collect/group_authors.py
python sourcing/collect/group_authors.py --min-videos 3 --top 20
```

</td></tr>
<tr><td><code>coarse_filter.py</code></td><td>

時長／分區／片型關鍵字粗篩。**完全不連網，改規則重跑零成本。**

判定分三檔而不是二分：`drop`（明確不對）／`keep`（命中片型詞）／
`maybe`（都沒命中，留著但排後面）。留 `maybe` 是因為關鍵字一定有漏，
二分的話漏掉的會被靜默丟棄。

```bash
python sourcing/collect/coarse_filter.py
python sourcing/collect/coarse_filter.py --show drop --show-n 15
python sourcing/collect/coarse_filter.py --exclude-tid 4 --exclude-tid 172
python sourcing/collect/coarse_filter.py --min-duration 15 --max-duration 1800
```

</td></tr>
<tr><td><code>covers.py</code></td><td>

下載封面並拼成縮圖牆。**整條漏斗裡 CP 值最高的一步。**
刻意跟 `coarse_filter.py` 分開：那支不連網、改規則重跑零成本，
混在一起的話每次調關鍵字都得重抓一輪圖。

```bash
python sourcing/collect/covers.py --download --sheet
python sourcing/collect/covers.py --download --sheet --verdict keep
python sourcing/collect/covers.py --sheet --from-shortlist   # 照 shortlist 排序
```

還沒做的進階做法：直接對封面跑人臉偵測，沒有臉的自動淘汰。

</td></tr>
<tr><td><code>fetch_channel.py</code></td><td>

抓某位作者在自己頻道的**全部作品**。B 站 space API 要 WBI 簽名 + buvid3，
比 view 端點麻煩，但公開資料不需要登入。

用途是把清單當成「人的名冊」：臉已經確認可用的人，回他自己的頻道
找缺的那幾種情緒。**⚠️ 這條實測也不通** —— 四月吨吨_ 四支約 100 分鐘
仍然只有樂，哀怒驚全是 0。情緒基調是鏡頭前人設的屬性。

```bash
python sourcing/collect/fetch_channel.py --mid 20960244
python sourcing/collect/fetch_channel.py --mid 20960244 --merge --min-duration 180
```

</td></tr>
</table>

### detect/ — 判情緒（兩種判法並列）

兩支輸出的結構刻意一樣（`labels_present` / `clip_counts` / `clips`），
所以可以直接用 `mapping/compare.py` 並排對照。

<table>
<tr><th align="left">檔案</th><th align="left">功能與用法</th></tr>
<tr><td><code>text_timeline.py</code></td><td>

看**文字**：yt-dlp 抽音訊 → faster-whisper 轉逐字稿 → BERT 逐句判 8 類
→ 套 `NATIVE_TO_FIVE` → 五類時間軸。

```bash
python sourcing/detect/text_timeline.py --uid bili:BV1RD421T7S9
python sourcing/detect/text_timeline.py --verdict keep --limit 20
python sourcing/detect/text_timeline.py --report    # 只彙整已跑完的，不重跑
python sourcing/detect/text_timeline.py --remap     # 換映射／門檻後重算，不重跑模型
```

其他參數：`--min-clips 3`、`--min-confidence 0.5`、`--min-clip 2.0`、
`--device cuda`、`--whisper small`、`--refresh`。

**`--remap` 值得特別記住**：換映射表或調門檻都不必重跑 STT 與 BERT，
時間軸已經存了，重算是秒級的。

⚠️ **不能用來判表情**（見上方「為什麼最後只採信臉那一邊」）。
現在的用途剩兩個：一、產出逐字稿（`eval/scripts/build_spoken_set.py` 吃它）；
二、當「文字判不出表情」的對照組。

</td></tr>
<tr><td><code>scan_batch.py</code></td><td>

★ 大規模粗掃：回答「857 個作者裡有沒有人真的四種情緒都出現過」。
一次跑很多支用這一支，不要自己寫迴圈。

```bash
python sourcing/detect/scan_batch.py --plan                  # 候選名單與預估
python sourcing/detect/scan_batch.py --probe --sleep 8       # 粗掃（可中斷重跑）
python sourcing/detect/scan_batch.py --report                # 彙整
python sourcing/detect/scan_batch.py --full --top 20         # 對前 20 名細看
python sourcing/detect/scan_batch.py --rescore               # 改門檻後重算，不重爬
```

**粗掃不整支下載**，而是在影片上鋪 8 個時間窗、每個窗抓 45 秒（360p、2 fps）。
實測這條線路對 B 站只有約 200KB/s，一支 52 分鐘的影片整支抓要 12 分鐘還會斷線；
改成時間窗之後**成本與片長無關**，一支約 3 分鐘。窗內用跟細看一樣的 2 fps，
所以「連續 3 秒」這個判準在粗掃就成立。

**候選只用時長篩，不用標題。** 片型關鍵字對這批素材幾乎沒作用（只刷掉 6.3%），
而且「題材的情緒」與「臉上的表情」實測是兩回事 —— 用題材選人等於把文字那條路
失敗的原因搬進來。但題材分數照樣算出來寫進結果，`--report` 會列出
「題材分數高的人，量到的情緒種類真的比較多嗎」，讓這個假設可以被檢驗。

逐張影格會存進結果檔（一支約 80KB），所以 `--rescore` 可以在改門檻後
重算片段，完全不必重爬 —— 跟 `store.py` 的 raw/metadata 分層是同一個道理。

⚠️ 粗掃是抽樣：某個人在沒被抽到的時間裡笑了就看不到。
所以粗掃的「有」很可信，「沒有」比較弱。

</td></tr>
<tr><td><code>face_timeline.py</code></td><td>

★ 看**臉**，完全不看文字：yt-dlp 抽 480p 影格（2 fps）→ YuNet 偵測人臉
→ hsemotion（AffectNet 8 類）→ 套 `AFFECT_TO_FIVE` → 五類。
**同一種表情連續 3 秒以上**才算一個片段。

```bash
python sourcing/detect/face_timeline.py --uid bili:BV1aM4y117kD
python sourcing/detect/face_timeline.py --from-timelines --limit 8
python sourcing/detect/face_timeline.py --report
python sourcing/detect/face_timeline.py --uid ... --refresh
```

輸出 `_local/sourcing/face/{uid}.json`，含逐張影格判定（`samples`）
與連續片段（`clips`）。`samples` 是 `compare.py` 能做時間窗對照的原因。

主要常數（改了要重跑）：`SAMPLE_FPS=2.0`、`MIN_FACE_PX=80`、`MIN_CONF=0.45`、
`MIN_RUN_FRAMES=6`（2fps → 3 秒）、`MAX_HEIGHT=480`。

**三個會安靜失敗的坑，都已經處理掉了**（細節在 `detect/__init__.py`）：
AV1 影片會讓 OpenCV 讀到 0 張影格而不報錯、YuNet 模型走 raw.githubusercontent
會抓到 0 位元組、hsemotion 自己的模型下載是壞的。

</td></tr>
</table>

### deliver/ — 交付物

<table>
<tr><th align="left">檔案</th><th align="left">功能與用法</th></tr>
<tr><td><code>shortlist.py</code></td><td>

把候選影片排出優先順序，**決定接下來要跑哪幾支 `detect/`**。
已跑過 detect 的用實測數字，還沒跑過的用 metadata 估分。

```bash
python sourcing/deliver/shortlist.py --top 30
python sourcing/deliver/shortlist.py --female-only --min-duration 600
python sourcing/deliver/shortlist.py --pick 12   # 直接印出可貼的 --uid 參數
```

`--pick` 印出來的東西可以整行貼給 `detect/face_timeline.py`，不用自己抄 uid。

</td></tr>
<tr><td><code>make_review_page.py</code></td><td>

產出**單一 HTML** 的人工檢視頁，封面圖用 data URI 內嵌，丟到哪裡都能開。
每個切點做成 `bilibili.com/video/BVxxx?t=秒數` 的連結，
**點下去直接跳到那一秒**，不必自己拉進度條。

```bash
python sourcing/deliver/make_review_page.py
python sourcing/deliver/make_review_page.py --min-labels 4 --out review.html
```

</td></tr>
<tr><td><code>cut_clips.py</code></td><td>

yt-dlp 下載 + ffmpeg 切段，產出實際的 mp4 與 `manifest.json`
（每個檔案是誰的哪一種情緒、來自哪支影片的第幾秒）。
**只吃 `face_timeline.py` 的結果，不吃文字那條。**

```bash
python sourcing/deliver/cut_clips.py --uid bili:BV1aM4y117kD
python sourcing/deliver/cut_clips.py --all --per-emotion 3
python sourcing/deliver/cut_clips.py --all --height 720 --pad 0.3 --min-seconds 3.0
```

兩個實作上的決定：**重新編碼而不是 `-c copy`**（copy 只能從關鍵影格切，
起點會飄到好幾秒前，切出來可能根本不是那個表情）；**前後各留 `pad` 秒**
（FER 逐張判定，表情起訖點會切得很死，留點過渡對嘴型模型也較好）。

</td></tr>
</table>

### actors/ — 取材路線 A：演員資料集

<table>
<tr><th align="left">檔案</th><th align="left">功能與用法</th></tr>
<tr><td><code>prepare_ravdess.py</code></td><td>

把下載好的 RAVDESS zip 解開，整理成「同一人 × 五個情緒資料夾」的乾淨小樣本，
並產出 `manifest.json` 與 `LICENSE.txt`。**不需要任何偵測** ——
RAVDESS 的檔名就編碼了情緒（欄位對照見 `actors/__init__.py`）。

```bash
python sourcing/actors/prepare_ravdess.py                 # 預設 7 女 3 男
python sourcing/actors/prepare_ravdess.py --actors 02 04 08 --per-emotion 1
```

規格：只收模態 01 + 聲道 01（要有畫面也要有聲音、不唱歌）；
情緒 01/02 → 預設、03 → 樂、04 → 哀、05 → 怒、08 → 喜、06/07 不收。
每個資料夾放 1–2 個檔案，是給下游測試的乾淨小樣本，不是要塞滿。

同一類有好幾個檔案時怎麼挑（常數區塊有完整說明）：
情緒四類取**強烈**（普通強度的演出不少接近面無表情，FER 也常判成 Neutral）、
預設取 **neutral 的普通強度**（那才是無表情基準，不是 calm 的強烈版）、
台詞固定 01、重複先 01 再 02 —— 讓同一位演員五個資料夾之間唯一的變數就是情緒。

解壓用 `.done` 標記檔記住解過哪些，重跑不會重解。

⚠️ **授權 CC BY-NC-SA 4.0，非商業。** 技術測試沒問題，要進產品必須另外
取得商用授權。替代品見 [`ROUTES.md`](ROUTES.md#a-演員拍攝資料集--已實作)。

</td></tr>
<tr><td><code>verify_ravdess.py</code></td><td>

交叉驗證：對整理好的每個檔案跑 FER，確認臉部判定跟檔名標的情緒對得上。

```bash
python sourcing/actors/verify_ravdess.py
python sourcing/actors/verify_ravdess.py --sample-fps 8 --show-all
```

RAVDESS 一段只有 3–4 秒，2 fps 只取得到 6、7 張，所以這裡預設 8 fps。
判定用**多數票**而不是單張最高分 —— 單張會被眨眼與講話的嘴型帶偏。

對不上的會列出明細（五類票數、AffectNet 原始分布、選它的理由），
**但不自動處理**：「演員沒演到位」與「FER 讀不出來」是兩件不同的事，
前者要換素材，後者是我們的偵測在那個表情上弱，要人看過才知道是哪一種。

輸出 `_local/sourcing/ravdess_fer_check.json`。

</td></tr>
</table>

---

## 舊檔名對照（2026-08-19 重構）

原本 15 支腳本全平放在 `sourcing/` 底下，現在依「映射 / 取材」分層。
兩支順便改了名字，讓文字與臉部那組看得出是一對：

| 舊 | 新 |
|---|---|
| `sourcing/emotion_timeline.py` | `sourcing/detect/text_timeline.py` |
| `sourcing/face_emotion.py` | `sourcing/detect/face_timeline.py` |
| `sourcing/paths.py` `store.py` | `sourcing/common/` |
| `sourcing/normalize_urls.py` 等 7 支 | `sourcing/collect/` |
| `sourcing/shortlist.py` `make_review_page.py` `cut_clips.py` | `sourcing/deliver/` |
| `sourcing/prepare_ravdess.py` | `sourcing/actors/` |
| （新增） | `sourcing/mapping/` 四支 |

`face_emotion.py` 裡的 `AFFECT_TO_FIVE`、`prepare_ravdess.py` 裡的 `EMOTION_MAP`
都搬到 `mapping/schemes.py` 集中管理了，原本的 import 名稱不變。
