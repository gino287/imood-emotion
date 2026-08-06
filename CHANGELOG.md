# 更新紀錄

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
