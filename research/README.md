# research/ — 兩個 open question 的調查與實驗

> 建立於 2026-08-13，對應 `notes/workspace/open_questions.md` 🔴 首要區的兩條：
> **①信心值（confidence）的意義與可參考程度**、**②如何保持 GPU 熱啟動**。

---

## 先看這個：兩個最重要的發現

### ① GPU：問題問偏了，答案是 fp16

一整晚的實驗最後收斂到一個很簡單的結論：
**不用跟降頻對抗，把模型換成半精度就好。**

| | 現況 | fp16 | fp16 + CUDA Graph | （心跳 keep-alive）|
|---|---:|---:|---:|---:|
| 間歇 1.5 秒 p50 | 126.5ms | **61.9ms** | **52.4ms** | 46.3ms |
| 間歇 p95 | 139.1ms | 63.6ms | **53.6ms** | 86.7ms |
| VRAM | 1073MB | **547MB** | **539MB** | 1073MB |
| 額外功耗 / GPU 佔用 | — | **0 / 0** | **0 / 0** | +5.3W / **+16%** |
| 準確率變化 | — | **零**（見下） | — | 0 |

（對照：**CPU 實測 91.5ms / p95 100.6ms**。所以 GPU 現況確實輸給 CPU，
換成 fp16 就贏回來，加 CUDA Graph 後 p95 差距接近兩倍。）

fp16 的準確率代價實測是 **零**：
200 句同領域 **0/200 不一致**；1200 句跨領域 **6/1200 不一致，
而且那 6 句的信心全在 0.25~0.44**（模型本來就在擲硬幣的地方才翻面）。

降頻假設 **完全證實**（閒置 210MHz / 連續 1912MHz），
心跳 **也有效但要付 16% 的 GPU 時間**，而那是要跟 Moshi/JoyGen 搶的。
**fp16 是免費的，還順便把 VRAM 佔用砍一半。**

三個反直覺的否定結果：**便宜的 matmul 心跳完全無效**、
**torch.compile/CUDA Graph 對間歇模式無效**（只救連續模式）、
**TF32 反而讓連續模式慢 33%**。
→ 細節與該做什麼：`gpu-warmup/RESULTS.md`

### ② 信心值：同領域可用、跨領域廢掉，而且一個既有結論要修正

| | 同領域（200 句） | 跨領域（1200 句 SMP2020） |
|---|---:|---:|
| 準確率 | 0.850 | 0.408 |
| 平均信心 | 0.902（高估 **+0.05**）| 0.790（高估 **+0.38**）|
| 錯誤偵測 AUROC | **0.849（可用）** | 0.666（很弱）|

**兩個一分鐘就能自己跑一次的例子**（`confidence/experiments/inspect_sentence.py`）：
- 「哦，是喔。」→ 模型 **98.1% 確定是平淡語氣**（margin 0.97、熵 0.17/3.00 bits），
  但真實對話裡這句多半是敷衍。**信心很高、答案很可能不對，而且任何門檻都擋不住。**
- 「我真的受不了你了」逐字加長 → **前四個字翻了三次類**，信心還高達 0.88；
  完整句答「悲傷語調」信心只有 0.52（一般人會讀成憤怒）。

**要修正的既有結論**：「信心只降 7.7%，所以擋不住截斷」——
這個說法低估了訊號量（翻類的片段平均信心 0.703 vs 沒翻類的 0.887，
AUROC 0.788）。結論仍是「不能用」，但 **理由要換**，
否則跟品靜溝通時很容易被反駁。
→ 正確說法與數字：`confidence/RESULTS.md`

---

## ⚠️ 這個資料夾的公私與版控狀態尚未決定 —— 請 Gino 判斷

依 `CLAUDE.md` 規則 5（不確定算公算私就先問），這裡先不自作主張，列出判斷材料：

| 內容 | 性質 | 建議 |
|---|---|---|
| `*/survey/*.md` | 文獻整理 + 對本專案的推論，含不少「內部判斷」與對決策的批評 | 偏私，像 `notes/` |
| `*/experiments/*.py` | 程式碼 | 偏公，但**註解是 AI 起草的**，依規則 4 需先由 Gino 改寫才能進版控 |
| `*/results/*.jsonl` | 量測數據，**不含資料集原文**（只有延遲/時脈/機率統計） | 可公可私 |
| `research/confidence/results/calibration.json` | 統計摘要，不含原文 | 可公 |

**目前 `.gitignore` 沒有涵蓋 `research/`，所以現狀是「會被 git 追蹤」。**
如果不想現在就進版控，先加一行 `research/` 到 `.gitignore` 最保險。

另外：`notes/pending_review.md` 已附上摘要條目，等 Gino 分類收錄。

---

## 目錄

```
research/
├── confidence/                  ← open question ①
│   ├── survey/                  文獻與觀念整理（讀這個）
│   │   ├── 01_softmax信心值是什麼.md
│   │   ├── 02_校準_ECE與溫度縮放.md
│   │   ├── 03_信心值的用途.md
│   │   ├── 04_套用到我們的模型.md
│   │   ├── 05_比門檻更適合本專案的做法.md   ← 時間平滑／遲滯／成本敏感門檻
│   │   └── 06_如果走微調路線.md             ← focal loss、多語言校準、soft label
│   ├── experiments/
│   │   ├── conflib.py           校準/選擇性預測/錯誤偵測的公式（只依賴 numpy）
│   │   ├── exp_calibration.py   跑三份既有資料的完整分析
│   │   └── inspect_sentence.py  互動式看單句分布（培養直覺用）
│   ├── results/calibration.json
│   └── RESULTS.md               ← 實測數字與解讀（讀這個）
│
└── gpu-warmup/                  ← open question ②
    ├── survey/
    │   ├── 01_為什麼閒置後的GPU會變慢.md
    │   ├── 02_有哪些解法.md
    │   └── 03_對本專案的取捨.md
    ├── experiments/
    │   ├── gpulib.py            NVML 取樣（含 WSL2 的成本陷阱說明）
    │   ├── exp1_gap_sweep.py    間隔 → 時脈 → 延遲（sweep / decay 兩模式）
    │   ├── exp2_keepalive.py    各種心跳做法的效果與代價
    │   ├── exp3_reduce_cost.py  fp16 / torch.compile / CUDA Graph / TF32
    │   ├── exp4_fp16_parity.py  fp16 vs fp32 的預測一致率（換模型時要重跑）
    │   ├── run_all.sh           一次跑完全部（約 55 分鐘）
    │   └── _nvml_cost.py        量 NVML 各查詢自己的成本
    ├── results/                 *.jsonl 逐筆資料 + *.meta.json 摘要 + *.log 執行紀錄
    └── RESULTS.md               ← 實測數字與解讀（讀這個）
```

---

## 建議閱讀順序

**只有 20 分鐘** → 兩份 `RESULTS.md` 的 TL;DR 表格。

**想真的理解** → `confidence/survey/01 → 02 → 03 → 04`，
再 `gpu-warmup/survey/01 → 02 → 03`，最後看兩份 `RESULTS.md`。

**要改程式** → 兩份 `RESULTS.md` 最後的「行動清單」。

---

## 怎麼重跑

**GPU 那一整套（約 55 分鐘）一行就跑得完**：

```bash
docker compose -f docker/docker-compose.yml run --rm app \
    sh research/gpu-warmup/experiments/run_all.sh
```

個別重跑：

```bash
# 信心值分析（CPU，約 5 秒，不重跑模型）
docker compose -f docker/docker-compose.yml run --rm app-cpu \
    python research/confidence/experiments/exp_calibration.py

# GPU 間隔掃描（約 11 分鐘，要獨佔 GPU，跑的時候別開遊戲/其他 CUDA 程式）
docker compose -f docker/docker-compose.yml run --rm app \
    python research/gpu-warmup/experiments/exp1_gap_sweep.py --mode sweep

# GPU 降頻/回升曲線（約 30 秒）
docker compose -f docker/docker-compose.yml run --rm app \
    python research/gpu-warmup/experiments/exp1_gap_sweep.py --mode decay

# 熱啟動手段比較（約 12 分鐘）
docker compose -f docker/docker-compose.yml run --rm app \
    python research/gpu-warmup/experiments/exp2_keepalive.py --gap 1.5

# 降低單次成本：CUDA Graph / torch.compile / fp16（約 15 分鐘）
docker compose -f docker/docker-compose.yml run --rm app \
    python research/gpu-warmup/experiments/exp3_reduce_cost.py

# CPU 對照組（約 11 分鐘）
docker compose -f docker/docker-compose.yml run --rm app-cpu \
    python research/gpu-warmup/experiments/exp1_gap_sweep.py --mode sweep --device cpu

# 互動看單句的信心分布（秒級，培養直覺用）
docker compose -f docker/docker-compose.yml run --rm app-cpu \
    python research/confidence/experiments/inspect_sentence.py "我真的受不了你了"
docker compose -f docker/docker-compose.yml run --rm app-cpu \
    python research/confidence/experiments/inspect_sentence.py --prefix "我真的受不了你了"
```

⚠️ GPU 實驗對「機器上還有沒有別的東西在用 GPU」極度敏感。
重跑時如果數字對不上，第一個要確認的是有沒有別的程式在用卡
（瀏覽器硬體加速也算）。

---

## 這次調查的兩個方法論教訓（值得記住）

1. **量測工具本身的成本要先量。**
   第一版 GPU 實驗把 `nvmlDeviceGetCurrentClocksThrottleReasons`
   放進逐句迴圈，那一個呼叫在 WSL2 要 18ms，
   前後各一次 = 每句偷插 40ms 空檔，**把「連續」變成「間歇」，
   量到的降頻有一部分是工具自己造成的**。

2. **NVML 回報的時脈是落後指標（約 0.5 秒更新一次）。**
   可以看穩態，不能做逐句關聯——否則會得到
   「210 MHz 時延遲 16ms」這種自相矛盾的結果。

3. **4GB 卡上放太多模型，Windows WDDM 不會 OOM，會默默慢五倍。**
   `exp3` 第一版把 6 個變體（約 4.3GB）全部先建好再逐一量，
   量到 eager 連續 91.8ms，但正確答案是 17.5ms。
   超出 VRAM 的部分被搬到系統記憶體，**不報錯**。
   → 改成一次只建一個變體，量完 `del` + `empty_cache()`。

（前兩個都留有反例資料在 `gpu-warmup/results/*_v1_interleaved.*`。）

---

## 三件需要 Gino 決定或動手的事

1. **這個資料夾的公私與版控** —— 見上面那一節。
2. **Windows 電源管理改「偏好最大效能」後重跑 sweep**（5 分鐘）——
   我沒有代為更動系統設定。優先度不高（fp16 已經解決主要問題），
   但如果有效就是零程式碼的額外收益。
3. **問到 Moshi 的 GPU 佔用型態與 VRAM 需求** ——
   這決定「間歇降頻」在真實 pipeline 裡到底存不存在，
   也決定 4GB 夠不夠三個模型一起跑。
