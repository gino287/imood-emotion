# 更新紀錄

## 2026-08-07 — 接上真實麥克風輸入

模擬上游換成麥克風即時收音：錄音 → faster-whisper 轉文字 → 既有的分類邏輯。`classifier.py` 的核心未更動。

### 新增

- **`scripts/record_mic.py`**（主機端）：`sounddevice` 錄音、固定秒數切段、寫 wav。含音量表模式（`--check`）與裝置列表。
- **`imood_stream/stt.py`**：faster-whisper 包裝，含 OpenCC 簡轉繁與文字衛生檢查。
- **`imood_stream/mic_source.py`**：輪詢音檔、背景轉錄、有界佇列。
- **`imood_stream/downstream.py`**：給下游模組的精簡封包。
- **`scripts/stress_fragments.py`** 與 **`results/fragment_robustness.md`**：不完整輸入的穩定度測試。
- **`requirements-host.txt`**：主機端依賴（僅 `sounddevice`、`numpy`，不含任何機器學習套件）。
- `run_stream.py` 新增 `--input {simulated,mic}`；模擬串流完整保留，可隨時切回。

### 架構：錄音在主機、辨識與分類在容器

Docker Desktop on Windows 不支援音訊裝置直通（容器內無 `/dev/snd`、無 PulseAudio），麥克風只能在主機讀取。因此主機僅負責錄音並寫出 wav，Whisper 與分類器都留在容器內。wav 以「先寫 `.part` 再原子改名」交接，避免讀到寫入中的檔案。

消費模式由 generator 改為有界佇列（預設 8 段，滿了丟棄最舊的並計數）：真實麥克風是主動送出、不等待消費端的上游，原本的被動式產生器不再適用。

### 切分秒數：4 秒

樣本的中位句長 19 字，以中文語速計約需 4.2 秒說完。切 2~3 秒會使過半句子被截斷；4 秒容納得下中位句，等待時間仍在可接受範圍。

### 不完整輸入的穩定度（200 句隨機非標點截斷）

| | 完整句 | 截斷片段 |
| --- | ---: | ---: |
| 平均信心 | 0.902 | 0.833 |
| 預測類別改變 | — | **29.5%** |
| 執行期例外 | — | 0 次 |

**信心分數不能用來判斷句子是否被截斷**：信心僅下降 7.7%，類別卻改變 29.5%。下游若以 confidence 設門檻，擋不掉被切斷的句子。

邊界輸入的實測：「只有標點」判為憤怒語調（信心 0.617）、單一字元判為疑問語調（信心 0.884）。語音辨識在靜音段常產生這類輸出，故在送入分類器前先於文字層過濾（空白、純標點、少於 4 字、已知辨識幻覺）。

### 繁簡轉換

模型與資料集皆為繁體中文，語音辨識預設輸出簡體，不轉換會靜默降低準確率。以 OpenCC `s2twp` 轉換，屬確定性轉換。

原先另以 `initial_prompt` 引導繁體輸出，實測發現辨識器在無語音的音訊上會將提示詞原樣輸出，並被分類為情緒事件（信心 0.935）。已移除該提示詞，繁體僅由 OpenCC 保證。

### 修正

- **Ctrl+C 時執行環境檔未寫出**：`write_meta` 位於迴圈之後，中斷時不會執行。麥克風模式唯一的結束方式即為中斷，佇列統計會一併遺失。已改為在 `finally` 中回填並寫出。
- **語音辨識未暖機**：首次呼叫耗時 5098ms，其後約 212ms。載入時先跑一次空轉錄。
- **摘要工具誤讀非執行結果**：改以「是否有配對的 `.meta.json`」判斷，不再依賴檔名規則。
- `results/summary.md` 補上「此為模型端延遲，不含語音辨識」的標註。

### 輸出分為兩種

評估用 `_local/out/*.jsonl` 維持原格式（新增 `source` 與 `transcribe_ms` 欄位）；下游用 `_local/downstream/*.jsonl` 為精簡封包（`ts` / `text` / `emotion` / `confidence`），供 Emotion Video Selector 與 JoyGen 參考。

語音辨識耗時記錄於 `transcribe_ms`，**不併入** `latency_ms`：該部分屬上游模組職責，合併計算會在串接完整 pipeline 時重複計入。

---

## 2026-08-06 — 串流 MVP 與第一份 baseline

本 repo 由環境骨架推進到可執行的串流情緒分類 MVP，並產出第一份速度 baseline。

### 新增

- **串流分類 MVP**：`run_stream.py` 與 `imood_stream/`（`source` 假上游／`classifier` 模型與計時／`recorder` 落地）。模擬上游以 0.5~3 秒隨機間隔逐句送入，每句立刻分類，逐句記錄完整 8 類機率分布、四段延遲與執行裝置。
- **`scripts/prepare_samples.py`**：以固定 seed 從 `Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset` 均衡抽樣，並斷言其標籤集合與模型 8 類一致。
- **`scripts/verify_labels.py`**：以混淆矩陣驗證標籤順序（見下）。
- **`scripts/summarize.py`** 與 **`results/summary.md`**：不含資料集原文的速度摘要。
- **`.env.example`**。

### 第一份 baseline（200 句，計入統計 197 句）

| 裝置 | 到達模式 | p50 | p95 |
| --- | --- | ---: | ---: |
| `cpu` | 間隔 0.5~3 秒 | 94.6ms | 116.1ms |
| `cpu` | 連續 | 105.1ms | 124.4ms |
| `cuda` | 間隔 0.5~3 秒 | 134.3ms | 141.4ms |
| `cuda` | 連續 | 17.0ms | 43.9ms |

四種條件的預測結果逐句相同，差異純粹來自延遲。

**主要發現：在模擬實際對話節奏的間隔到達下，GPU 沒有優勢。** GPU 於句間閒置降頻，每句都在時脈未拉起時完成；CPU 方向相反，連續運算使核心無法維持高頻，反而比有間隔時慢 11%。同一個間隔，對 CPU 是休息、對 GPU 是降頻。

**對部署的意涵**：以目前的對話節奏，本環節跑 CPU 即可滿足即時性，可將整張 GPU 讓給 pipeline 上運算量更大的模組。若日後改採批次或高頻輸入，GPU 的優勢才會顯現。

### 標籤順序：由無法驗證改為實證確認

模型的 `config.json` 只有 `LABEL_0..LABEL_7` 佔位符，標籤順序取自 model card 而無法由模型檔案驗證；順序接錯會使分數莫名偏低卻不產生任何錯誤訊息。既然驗不了檔案，改為驗行為：以資料集標註句計算混淆矩陣。

實測 480 句整體對角率 **87.9%**（隨機水準 12.5%），八類對角線全數浮出，**順序確認正確**。已納入 `scripts/verify_labels.py`，對角率低於 50% 即失敗。

同時量到模型本身的弱點：憤怒語調對角率僅 60.0%，主要誤判為悲傷與厭惡；疑問語調 75.0%；其餘六類 88~97%。

### 資料集文件與實際資料不符

`Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset` 的 README 列有「驚訝語調」「恐懼語調」，兩者在實際 data.csv 中皆為 0 筆；實際存在的是「關切語調」與「驚奇語調」，與模型 8 類完全對應。一律以實際資料為準，並於執行時斷言，不為文件錯誤加設轉換層。

### 版本釘選

模型與資料集皆以 commit 釘住 revision。Hugging Face 的 repo 可變且更新無通知，不釘版本則不同時間量到的數字失去比較基礎。`*.meta.json` 一併記錄兩者的 revision。

### 環境公版化

- `env_file` 改為 `required: false`，移除本機專屬掛載 —— 全新 clone 無須額外設定即可 build 與執行（已實測驗證）。
- `.gitignore` 明確劃分：執行期產物與本機設定不進版控；逐句結果含資料集原文，僅摘要進版控。

---

## 2026-08-04 — 重置為 MVP 起點

專案轉向製作 MVP。先前的評測框架與其凍結測試集移出版控，repo 只保留 Docker 開發環境（`dockerfile`／`docker-compose.yml`／`requirements.txt`）。新增 `.gitattributes` 強制 LF，避免在 Linux 容器上踩 CRLF。
