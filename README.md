# imood-emotion — BERT 情緒分類模組

imood.ai 情緒陪伴 AI 的情緒分類環節。逐句吃進文字，即時吐出情緒類別與信心分數。

```
麥克風／音檔 ──► STT ──► 前處理 ──► BERT 情緒分類 ──► 輸出封包 ──► 下游
              （可替換）   └───────── 本 repo ─────────┘      Emotion Video
                                                             Selector / JoyGen
```

**即時性是硬性要求** —— 這個環節不能成為 pipeline 的瓶頸，所以每一句的延遲都要量、
CPU 與 GPU 都要有數字。準確率不是目前的優先項目。

目前處於 prototype 階段：模擬上游不規律吐句，逐句即時分類，記錄完整機率分布與延遲。

---

## 快速開始

### 需要什麼

- Docker Desktop（Windows 需 WSL2 後端）
- 要跑 GPU 的話：NVIDIA 驅動 + Container Toolkit
- 不需要在主機裝 Python 或 PyTorch，全部在容器裡跑

### 1. 建置

```bash
docker compose -f docker/docker-compose.yml build
```

> compose 檔放在 `docker/` 底下，所以每個指令都要帶 `-f docker/docker-compose.yml`。

確認環境：

```bash
# torch 版本
docker compose -f docker/docker-compose.yml run --rm app-cpu \
    python -c "import torch; print(torch.__version__)"

# GPU 直通
docker compose -f docker/docker-compose.yml run --rm app \
    python -c "import torch; print(torch.cuda.is_available())"
```

`.env` 不是必要的。目前用到的模型與資料集在 Hugging Face 上都是公開的，
沒有 `.env` 也能 build、也能跑。要填的話複製 `.env.example` 成 `.env`。

### 2. 準備樣本

```bash
docker compose -f docker/docker-compose.yml run --rm app-cpu \
    python scripts/prepare_samples.py --limit 200
```

從 `Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset`（與模型同作者發布）以固定 seed
均衡抽樣，輸出到 `_local/samples.jsonl`。`--limit` 預設 25，想快速試跑可以省略；
要對照已發布的 baseline 數字就用 200。

樣本不進版控 —— 固定 seed 加上釘住的資料集版本，任何人重跑都得到位元組相同的一份，
存腳本比存資料有意義。腳本會印出 SHA-256 供核對。

### 3. 跑起來

```bash
# 間隔到達：模擬上游 0.5~3 秒的不規律停頓（貼近實際使用）
docker compose -f docker/docker-compose.yml run --rm app-cpu \
    python scripts/run_baseline.py --device cpu  --limit 200
docker compose -f docker/docker-compose.yml run --rm app \
    python scripts/run_baseline.py --device cuda --limit 200

# 連續到達：不等待，量模型本身的能力上限
docker compose -f docker/docker-compose.yml run --rm app-cpu \
    python scripts/run_baseline.py --device cpu  --limit 200 --no-delay
docker compose -f docker/docker-compose.yml run --rm app \
    python scripts/run_baseline.py --device cuda --limit 200 --no-delay
```

每收到一句立刻送進模型，不累積等待多句 —— 即時性優先。
間隔到達的兩輪各需約 6 分鐘（等待本身佔掉大部分時間）。

### 4. 產出速度摘要

```bash
docker compose -f docker/docker-compose.yml run --rm app-cpu \
    python checks/summarize.py
```

讀 `_local/out/` 底下所有執行結果，寫出 [`results/summary.md`](results/summary.md)。

### 常用參數

| 參數 | 說明 |
| --- | --- |
| `--device {cpu,cuda,auto}` | 指定 cuda 但 GPU 不可用時直接失敗，不靜默降級成 CPU；`auto` 才會降級 |
| `--limit N` | 處理句數，預設 25；已發布的 baseline 用 200 |
| `--no-delay` | 句間不等待。等待本身不在計時區間內，但會影響 GPU 的時脈狀態（見後段） |
| `--seed N` | 固定句序與間隔，讓兩個裝置跑在相同條件下 |
| `--out PATH` | 輸出路徑，預設 `_local/out/stream_<device>[_nogap].jsonl` |

---

## 輸出格式

### 逐句結果（評估用）

`_local/out/*.jsonl`，一行一句：

```json
{"seq": 4, "warmup": false, "recv_at": "2026-08-06T21:03:11+08:00",
 "gap_sec": 1.82, "text": "……", "pred_label": "開心語調", "confidence": 0.9312,
 "preprocess_ms": 0.02,
 "latency_ms": {"tokenize": 0.31, "forward": 18.4, "post": 0.05, "total": 18.8},
 "probs": {"平淡語氣": 0.0041, "關切語調": 0.0102, "…": 0.0},
 "device": "cuda"}
```

`probs` 是完整 8 類機率分布，不只最高分那一類 —— 下游可能需要次高分或信心門檻，
只存 argmax 會讓那些策略無法評估。

前 3 句標記 `"warmup": true`：CUDA kernel autotune 會使最初幾句明顯偏慢，
統計時排除，但保留在檔案裡以便追溯。

執行環境（torch 版本、模型 id、樣本檔雜湊、時間戳）另存同名的 `.meta.json`。

**逐句結果不進版控**（含資料集原文，且重跑即有）；不含原文的速度摘要放在 `results/`。

### 下游封包

`_local/downstream/*.jsonl`，給 Emotion Video Selector 與 JoyGen 參考：

```json
{"ts": "2026-08-06T21:03:11+08:00", "text": "……", "emotion": "開心語調", "confidence": 0.9312}
```

一行一筆串流寫入，`tail -f` 即可消費。欄位目前是最小可用集合，依實際需求再擴充 ——
加欄位很容易，拿掉已經被依賴的欄位很難。

---

## 專案結構

```
imood_emotion/    前處理、BERT 推論、輸出封包（本 repo 的主體）
stt/              語音轉文字。前置模組、可替換，依賴與 image 都跟 BERT 分開
scripts/          啟動與設定：run_baseline.py / prepare_samples.py
checks/           驗證與分析：verify_labels.py / summarize.py / stress_fragments.py
eval/             跨模型評測骨架（與 production 解耦，見 eval/README.md）
results/          不含資料集原文的摘要，進版控
docker/           dockerfile / dockerfile.stt / docker-compose.yml
```

專案資料夾以 `-v ..:/app` 掛進容器，改 `.py` 不用重 build；只有動到 `requirements.txt`
才需要重跑 `docker compose -f docker/docker-compose.yml build`。
模型快取放在 `.cache/huggingface`（`HF_HOME` 指過去），主機與容器共用同一份，只下載一次。

---
---

# 以下是訓練細節與實驗過程

只想把模型跑起來的話，看到這裡就夠了。

## 模型

`Johnson8187/Chinese-Emotion-Small`（基於 mDeBERTa-v3-base 微調，0.3B 參數），
使用其 8 類原生標籤：

```
平淡語氣  關切語調  開心語調  憤怒語調  悲傷語調  疑問語調  驚奇語調  厭惡語調
```

本階段直接採用原生 8 類，不做 6 類映射；映射方案待 `eval/` 那套評測框架的數字出來後
再定案。

### 版本釘選

模型與資料集都以 commit 釘住 `revision`（見 `imood_emotion/labels.py`）。
Hugging Face 的 repo 是可變的，作者隨時可能更新內容且不會通知；不釘版本的話，
同一份程式在不同時間會拿到不同的權重或資料，先前量到的數字就失去比較基礎。

### 標籤順序的驗證

該模型的 `config.json` 只有 `LABEL_0..LABEL_7` 佔位符，沒有真實標籤名，
**順序無法從模型檔案本身讀出**，只能取自 model card。順序若接錯，分數會低得莫名其妙
卻不會有任何錯誤訊息 —— 是這類任務最典型的靜默錯誤。

既然驗不了檔案，就改成驗行為：

```bash
docker compose -f docker/docker-compose.yml run --rm app-cpu \
    python checks/verify_labels.py
```

拿資料集的標註句跑一輪並印出混淆矩陣。順序正確時對角線會明顯浮出，
接錯時整體對角率會掉到隨機水準（八類為 12.5%）。

**2026-08-06 實測 480 句，整體對角率 87.9%，八類對角線全數浮出，順序確認正確。**
對角率低於 50% 即判定失敗。改動 `NATIVE_LABELS`、換模型或換資料集版本後都應重跑。

同時量到模型本身的弱點：**憤怒語調對角率僅 60.0%**，主要誤判為悲傷與厭惡；
疑問語調 75.0%；其餘六類 88~97%。對情緒陪伴場景，「使用者在生氣」被讀成「在難過」
會選到完全相反的回應，比整體準確率的小數點更值得追蹤。

### 資料集文件與實際資料不符

`Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset` 的 README 列有「驚訝語調」
「恐懼語調」，兩者在實際 `data.csv` 中皆為 0 筆；實際存在的是「關切語調」與「驚奇語調」，
與模型 8 類完全對應。

一律以實際資料為準，不為文件錯誤加設轉換層，並在 `scripts/prepare_samples.py`
每次執行時斷言 —— 資料集是活的，作者哪天更新了內容，這裡會立刻叫出來。

## 前處理

`imood_emotion/preprocess.py`。送進分類器之前擋掉兩類問題：

**文字層過濾**：空白、純標點、少於 4 字、已知的語音辨識幻覺，一律不產生情緒事件。
實測「只有標點」會被判為憤怒語調（信心 0.617）、單一字元會被判為疑問語調（信心 0.884）
—— 模型對這種輸入照樣給高信心，也就是說「信心低」擋不掉它們，只能在送進去之前先擋。

> 4 字這個門檻沒有理論依據，是看著實測數字挑的，還沒被重新檢視過。

**簡繁轉換**：OpenCC `s2twp`（不只換字形，連台灣慣用詞也換）。模型與資料集都是繁體，
餵簡體進去會靜默拉低準確率。

⚠️ 簡繁轉換**只對語音辨識來的文字開啟**，模擬串流預設關閉。s2twp 會改動用詞，
而樣本資料集本來就是繁體，對它做轉換有機會改到模型輸入，讓數字跟已發布的 baseline
不能比。文字層過濾則兩條路徑都跑。

## 速度 baseline

2026-08-06，200 句（計入統計 197 句），RTX 3050 Laptop 4GB：

| 裝置 | 到達模式 | p50 | p95 |
| --- | --- | ---: | ---: |
| `cpu` | 間隔 0.5~3 秒 | 94.6ms | 116.1ms |
| `cpu` | 連續 | 105.1ms | 124.4ms |
| `cuda` | 間隔 0.5~3 秒 | 134.3ms | 141.4ms |
| `cuda` | 連續 | **17.0ms** | 43.9ms |

四種條件的預測結果逐句相同，差異純粹來自延遲。

**兩種到達模式都要測**，因為兩者的差異決定了這個環節該不該佔用 GPU：
間隔到達時 GPU 的延遲反而高於 CPU，而連續到達時 GPU 快一個量級。CPU 方向相反，
連續到達比間隔到達還慢 11%。

合理的解釋是 GPU 在句與句之間閒置降頻，每一句都在時脈尚未拉起時就算完了；
CPU 沒有降頻問題，連續運算反而讓核心無法維持高頻。**同一個間隔，對 CPU 是休息、
對 GPU 是降頻。**

> ⚠️ 「降頻」是推論出來的，用 `--no-delay` 對照組間接驗證，還沒直接用
> `nvidia-smi --query-gpu=clocks.sm` 量過 GPU 時脈。也還沒測鎖時脈／persistence mode
> 能不能把效能救回來，不知道降頻的間隔門檻落在哪裡。
>
> 另外，真實 pipeline 中 Moshi 與 JoyGen 同時運作，GPU 可能根本不會真的閒置，
> 屆時「間歇輸入降頻」這個前提可能整個不成立。
>
> **怎麼維持 GPU 熱啟動、把延遲壓回 17ms 這個量級，是目前的首要研究項目之一。**

完整數字與解釋見 [`results/summary.md`](results/summary.md)。

### 與資料集標註的一致率

`results/summary.md` 另外記了一個一致率 168/197（85%）。

> ⚠️ **這不是準確率。** 樣本僅 197 句、且刻意做成各類均衡（與真實輸入分布不同）。
> 這個數字只能證明「pipeline 接通、標籤沒接錯」，不足以代表模型效能，也不該對外引用。
> 正式的準確率評測是 `eval/` 那套框架的工作範圍。

## 不完整輸入的穩定度

固定秒數切分音訊**必然**會切在句子中間。這支壓力測試在還沒接上麥克風的前提下先模擬
同一件事：把完整句子在非標點位置隨機截斷（保留 30%~90%），比對完整句與片段的預測。

```bash
docker compose -f docker/docker-compose.yml run --rm app-cpu \
    python checks/stress_fragments.py
```

200 句實測（[`results/fragment_robustness.md`](results/fragment_robustness.md)）：

| | 完整句 | 截斷片段 |
| --- | ---: | ---: |
| 平均信心 | 0.902 | 0.833 |
| 預測類別改變 | — | **29.5%** |
| 執行期例外 | — | 0 次 |

> ⚠️ **信心分數不能用來判斷句子是否被截斷。** 信心只下降 7.7%，類別卻改變 29.5%。
> 下游若以 confidence 設門檻過濾，擋不掉被切斷的句子。

信心分數在統計上究竟代表什麼、能不能拿來做下游判斷依據（例如門檻切分），
是目前的另一個首要研究項目。

## 延遲的定義

本專案量測的是**模型端延遲**：從收到一句文字到分類結果產出。

**不含語音辨識**，也不含前處理 —— 前者屬上游模組職責（另記在 `transcribe_ms`），
後者另記在 `preprocess_ms`。混在一起，之後串接完整 pipeline 時會重複計算。

使用者實際感受到的延遲另外還包含：等待切分窗填滿、語音辨識時間，
以及下游 Emotion Video Selector 與 JoyGen 的處理時間。

## 語音轉文字（前置模組）

`stt/`。用途縮限在「音檔轉文字驗證」，不接進上面那條 baseline。

```bash
docker compose -f docker/docker-compose.yml build stt
docker compose -f docker/docker-compose.yml run --rm stt \
    python stt/transcribe.py 某個音檔.wav
```

它是 pipeline 流程圖上的**前置模組、可替換**，所以：

- 有自己的 `stt/requirements.txt` 與 `docker/dockerfile.stt`，
  BERT 那個 image **不含 faster-whisper** —— 只想要情緒分類的人不該被迫連帶抓
  一整套語音辨識依賴。
- 不 import `imood_emotion` 的任何東西，只回傳原始轉錄文字；簡繁轉換與文字過濾
  是下一站（前處理）的事。日後要把它拆成獨立 repo，整個資料夾搬走即可。

刻意不使用 `initial_prompt`：原本放了一句提示詞想把 Whisper 拉向繁體，實測發現
辨識器在靜音段會把提示詞原封吐回來，還被分類成信心 0.935 的假情緒事件。
繁體改由前處理的 OpenCC 保證。

> 真實麥克風即時收音曾於 2026-08-07 實作、08-10 回滾。回滾原因是把 STT 塞進 BERT
> 容器違反容器單一職責，不是功能本身有問題。詳見 [`CHANGELOG.md`](CHANGELOG.md)。

## 跨模型評測骨架

`eval/`。比較多個現成中文情緒分類模型在「六類標籤、即時性、準確率」上的表現，
作為要不要自己微調的決策依據。與 production 的推論路徑刻意解耦、不互相 import。

用法見 [`eval/README.md`](eval/README.md)。

---

更新紀錄見 [`CHANGELOG.md`](CHANGELOG.md)。
