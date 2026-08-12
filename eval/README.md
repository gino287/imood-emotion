# eval — 跨模型評測骨架

比較多個現成中文情緒分類模型在「六類標籤、即時性、準確率」上的表現，作為要不要自己微調的決策依據。

**核心設計**：評測骨架（怎麼測、拿什麼測、算什麼指標）與模型轉接器（各模型的推論邏輯）分離。骨架只建一次，之後每加一個候選模型，多半只要在 `eval/configs/models.yaml` 加一段設定，不動任何一行主程式。

> 這套骨架與 `imood_emotion/`（production 的推論路徑）**刻意解耦，不互相 import**。這邊驗證過的邏輯若要用在 production，用複製的 —— production 不該依賴一個獨立演進的評測模組，兩邊各自維護的代價可以接受。
>
> `eval/data/` 與 `eval/results/` 不進版控：前者是資料集原文（有授權條款、且由固定 seed 的腳本重建），後者的 `predictions.jsonl` 一樣逐句含原文。程式碼本身正常進版控。

---

## 快速開始

全部指令都從專案根目錄下。compose 檔在 `docker/` 底下，所以每一行都要帶 `-f`：

```powershell
docker compose -f docker/docker-compose.yml build

# 自我檢查：確認前處理與指標的實作正確（手算案例，5 秒跑完）
docker compose -f docker/docker-compose.yml run --rm app-cpu python eval/scripts/selftest.py

# 一次性：建立凍結測試集（每類 200 句，共 1200 句）
# 需要 .env 設好 REFS_DIR 指向 SMP2020 原始資料，見 .env.example
docker compose -f docker/docker-compose.yml run --rm app-cpu python eval/scripts/build_testset.py

# 跑評測（GPU，跑完設定檔裡的 devices × text_variants 全部組合）
docker compose -f docker/docker-compose.yml run --rm app python eval/run_eval.py --model johnson-small

# 只跑單一組合 / 冒煙測試
docker compose -f docker/docker-compose.yml run --rm app-cpu python eval/run_eval.py --model johnson-small --device cpu --variant zh_tw --limit 30
```

報告產在 `eval/results/<模型>/<裝置>/<文字變體>/`，用瀏覽器打開裡面的 `dashboard.html`。

其他常用指令：

```powershell
# 列出候選模型
docker compose -f docker/docker-compose.yml run --rm app-cpu python eval/run_eval.py --list
# 資料觀察：標籤分佈、句長、噪音樣態
docker compose -f docker/docker-compose.yml run --rm app-cpu python eval/scripts/explore_dataset.py
```

---

## 檔案地圖

| 檔案 | 說明 |
|------|------|
| `configs/models.yaml` | **唯一要手動維護的設定檔**：候選模型、映射表、執行參數 |
| `scripts/explore_dataset.py` | 資料觀察：標籤分佈、句長、噪音樣態 |
| `scripts/build_testset.py` | 分層抽樣 → 前處理 → 繁簡雙變體 → 凍結成帶 SHA-256 的資料集 |
| `scripts/selftest.py` | 前處理與指標的手算驗證（指標是自己實作的，靠這支頂著正確性） |
| `evalkit/config.py` | 讀設定檔 + 驗證（映射表的鍵與 `native_labels` 對不上會直接失敗） |
| `evalkit/dataset.py` | 載入凍結資料集 + 核對 hash |
| `evalkit/preprocess.py` | 清洗規則與 OpenCC `s2twp` 轉換 |
| `evalkit/adapters/base.py` | 轉接器介面：`load()` / `predict(text)` |
| `evalkit/adapters/hf_seqcls.py` | 標準 HuggingFace 分類模型轉接器（涵蓋多數候選） |
| `evalkit/runner.py` | 推論迴圈、逐句計時、VRAM 量測。**只寫原生輸出，不做映射** |
| `evalkit/mapping.py` | 映射表套用、棄權處理、strict/covered/subset 三視角 |
| `evalkit/metrics.py` | 全部指標計算 |
| `evalkit/report.py` | 產出 `report.md` 與單檔 `dashboard.html` |
| `evalkit/dashboard_template.html` | 儀表板樣板（vanilla JS，資料在產出時內嵌，無 CDN 依賴） |
| `run_eval.py` | CLI 入口 |
| （執行環境） | 與 BERT 那邊共用同一組，見專案根目錄的 `docker/` |

---

## 三步驟工作流

### 1. 建立測試集（一次性）

```powershell
docker compose -f docker/docker-compose.yml run --rm app-cpu python eval/scripts/build_testset.py
```

從 `${REFS_DIR}/BERT_SMP2020-EWECT/data/raw/usual_test_labeled.txt` 分層抽樣，每類 200 句共 1200 句，固定 `seed=20260731`，前處理後產出：

```
eval/data/testset/dataset_v1.jsonl        每筆含 text_zh_cn 與 text_zh_tw 兩種變體
eval/data/testset/dataset_v1.meta.json    來源 hash、seed、各類筆數、資料集自身 SHA-256
```

把 meta 裡的 SHA-256 填回 `eval/configs/models.yaml` 的 `dataset.expect_sha256`，之後每次評測都會核對，確保所有模型跑的是同一份資料。

> **為什麼每類 200 句**：官方 test 集 5,000 句中 fear 只有 210 句，這是抽樣硬上限。每類 200 句時 per-class 指標的 95% 信賴區間約 ±6%；若只抽每類 83 句，區間會放寬到約 ±10%，兩個模型差 8% 也分不出勝負。推論成本幾乎沒差（單模型約 1 分鐘），沒有理由抽少。

### 2. 跑評測

```powershell
docker compose -f docker/docker-compose.yml run --rm app python eval/run_eval.py --model johnson-small
```

逐句推論（batch=1，產品場景），對 `{cuda, cpu} × {zh_cn, zh_tw}` 四種組合各跑一輪，產出 `predictions.jsonl`。

**這一步不做標籤映射** —— 只存模型原生類別與完整機率向量。映射是下一步的後處理，所以之後想換映射表、調信心門檻、重算指標，都不必再花一次推論成本。

### 3. 看報告

```powershell
docker compose -f docker/docker-compose.yml run --rm app-cpu python eval/run_eval.py --model johnson-small --report-only
```

`--report-only` 會沿用既有的 `predictions.jsonl` 重算，**不重跑推論**。改完 `models.yaml` 的映射表後跑這一行就能看到新結果，這是把映射從推論裡拆出來的直接好處。

產出 `report.md`（可直接貼給 leader 的表格）、`metrics.json`、以及 `dashboard.html`（單一檔案、自帶資料、離線可開，也可以直接寄出去）。

儀表板內容：

- **映射方案比較表**：三組映射並列，一眼看出厭惡語調該映到哪
- **雙向類別分佈**：下拉選單切換，方向一看「真正屬於 fear 的句子被判成哪些原生類別」，方向二看「被判成厭惡語調的句子實際上是哪幾類」
- **混淆矩陣**（熱度著色）與 per-class precision / recall / F1，可切換 strict / covered / subset 三視角
- **mapping-free 指標**：最佳映射上界、映射猜錯的損失、NMI/ARI、每組映射的雙向支持度
- **延遲分佈**直方圖與 VRAM 峰值
- **信心校準**：ECE、reliability、門檻覆蓋率表
- **錯誤瀏覽器**：可篩選的逐句表（原文 / 正解 / 原生輸出 / 映射後 / 信心），人工抽查用

---

## 如何新增一個候選模型

多數情況只要改 `eval/configs/models.yaml`，在 `models:` 底下加一段：

```yaml
  - key: johnson-large
    hf_id: Johnson8187/Chinese-Emotion
    adapter: hf_seqcls
    native_labels: [...]              # 對照用，與模型 config 的 id2label 不符會直接失敗
    covers: [happy, angry, sad, surprise, neutral]   # 模型結構上表達得出的標籤
    mappings:
      draft: { 原生類別: 六類標籤, ... }             # 值可為 null 代表棄權
    default_mapping: draft
```

然後 `docker compose -f docker/docker-compose.yml run --rm app python eval/run_eval.py --model johnson-large`，不需要寫任何程式碼。

**只有推論介面不是標準 HuggingFace `AutoModelForSequenceClassification` 的**（例如托管在 ModelScope 的 `iic/nlp_structbert_*`），才需要在 `evalkit/adapters/` 新增一支轉接器，實作 `load()` 與 `predict()` 兩個方法，並在 `evalkit/adapters/__init__.py` 補一行 import（註冊是 import 的副作用，沒 import 就不會被登記進註冊表）。

---

## 報告怎麼讀

### 真正拿來做決策的三個數字

1. **Macro-F1（strict 視角）** — 六類各自 F1 的平均。類別不平衡時比 accuracy 誠實，也是對 leader 講話時用的那個數字。
2. **p95 延遲（batch=1）** — 不是平均值。平均值會被大量短句拉低，p95 才對應「使用者會不會感覺到卡頓」。
3. **VRAM 峰值（nvml 那一欄）** — 直接決定這個模型能不能跟 Moshi、JoyGen 同時常駐在 3050 的 4GB 上。

### 為什麼指標有三個視角

映射表允許把某個原生類別標成 `null`（棄權，代表模型結構上答不出這一類）。三個視角承接這件事：

| 視角 | 定義 | 什麼時候看 |
|---|---|---|
| `strict` | 棄權一律算答錯 | 產品實際會拿到的表現 |
| `covered` | 只算有映射的子集，另報 coverage% | 「模型有答的部分準不準」 |
| `subset` | 排除模型宣告不具備的標籤 | 公平比較原生類別數不同的模型 |

### mapping-free 指標在講什麼

`厭惡語調 → fear` 這種映射到底成不成立，用眼睛看圓餅圖不是可複製的決策依據。報告另外給三個數字：

- **最佳映射上界**：每個原生類別指派給它命中最多的真實標籤，算出這個模型在最理想映射下的準確率天花板。跟手訂映射的準確率一比，差距就是「映射猜錯造成的損失」。
- **NMI / ARI**：完全不看映射，衡量模型原生分類與六類標籤的結構相關性。這是唯一能公平比較「8 類模型 vs 28 類模型」的方式。
- **映射支持度**：`P(真實=fear | 原生=厭惡)` 與 `P(原生=厭惡 | 真實=fear)` 並列。前者若只有 15%，這組映射就是純湊數。

---

## 已知坑

- **不要在 `requirements.txt` 指定 torch**。base image 已內建一組對得上的 CUDA/cuDNN/PyTorch，自己再裝一次必定衝突（07/22 踩過）。
- **HF 模型快取靠 `-v ..:/app` 共用**。`HF_HOME` 設在 `/app/.cache/huggingface`，落在掛載範圍內，主機與容器共用同一份，1.4GB 的模型只下載一次。
- **不要用 `refs/BERT_SMP2020-EWECT/data/clean/`**。那份清洗把 `[惊恐]` 改成了 `《惊恐》`，正好毀掉方括號表情這個情緒訊號，還產生 10 筆空字串。一律從 `data/raw/` 自己前處理。
- **`refs/` 在專案資料夾外面**，容器看不到，而且路徑因人而異。位置寫在 `.env` 的 `REFS_DIR`（範本見 `.env.example`），compose 據此唯讀掛到容器的 `/refs`。沒設定時會改掛空資料夾 `docker/empty/` —— 只有建測試集這一步跑不了，其他功能不受影響。
- **指標沒有用 scikit-learn**，是 `evalkit/metrics.py` 用 numpy 自己實作的（少一個含 scipy 的依賴）。改動那支檔案後務必跑 `scripts/selftest.py`。
- **Johnson8187 系列的 `config.json` 沒有標籤名**，只有 `LABEL_0..7`。標籤順序抄自 model card，程式無法從模型驗證，每次載入會印出警告與出處。分數低得莫名其妙時，第一個懷疑對象就是這個順序。
- **`opencc-python-reimplemented` 有釘版本**。轉換結果會進凍結資料集的 SHA-256，換版本等於換一份測試集，先前模型的數字就不可比了。
- **`dockerfile` 是小寫檔名**。Docker 預設找 `Dockerfile`，所以 compose 裡有明確寫 `dockerfile: docker/dockerfile`。
- **改 `.py` 不用重 build**（compose 有掛 `..:/app`）；**改 `requirements.txt` 或 `dockerfile` 要重 build**。

---

## 硬體備忘

RTX 3050 Laptop / 4GB VRAM。8 月底會有正式實驗室設備，規格未知；最終產品部署在雲端。

VRAM 是這輪評測的硬約束：BERT 這端要與 Moshi、JoyGen 共用同一張卡，而那兩端才是即時性的主要瓶頸。所以評測同時記錄 `torch.cuda.max_memory_allocated()`（只算張量，會低估）與 nvml 讀到的 process 真實佔用（含約 300–600MB 的 CUDA context）—— **後者才是共卡決策要看的數字**。
