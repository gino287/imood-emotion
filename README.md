# imood AI — BERT 情緒分類模組

imood.ai 情緒陪伴 AI 的情緒分類環節。Pipeline 位置：STT → **BERT 情緒分類** → Emotion Video Selector → JoyGen。

本 repo 目前的內容是串流輸入的 MVP：模擬上游 STT 不規律吐句，逐句即時分類，並記錄速度與完整機率分布，作為 CPU/GPU 的 baseline。

**即時性是硬性要求** —— 這個環節不能成為 pipeline 的瓶頸，所以每一句的延遲都要量、CPU 與 GPU 都要有數字。

## 模型

`Johnson8187/Chinese-Emotion-Small`（基於 mDeBERTa-v3-base 微調，0.3B 參數），使用其 8 類原生標籤：

```
平淡語氣  關切語調  開心語調  憤怒語調  悲傷語調  疑問語調  驚奇語調  厭惡語調
```

本階段直接採用原生 8 類，不做 6 類映射；映射方案待另一套評測框架的數字出來後再定案。

> ⚠️ 該模型的 `config.json` 只有 `LABEL_0..LABEL_7` 佔位符，沒有真實標籤名，
> 因此**標籤順序無法從模型本身驗證**，程式只驗得到「輸出為 8 類」。
> 上面的順序取自 model card 的 `label_mapping`（2026-07-31 查核）。

## 環境

| 檔案 | 用途 |
| --- | --- |
| `dockerfile` | base image `pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime`。這組 CUDA/cuDNN/PyTorch 版本已驗證 GPU 直通可用，不要更換 |
| `docker-compose.yml` | `app`（掛 GPU）與 `app-cpu`（純 CPU 對照組）兩個服務 |
| `requirements.txt` | 套件清單。**不要加 torch/torchvision**，base image 已內建對得上的版本 |
| `.env.example` | 複製成 `.env` 後填寫。目前所用模型與資料集皆為公開，不填也能跑 |

```bash
docker compose build

# 確認環境
docker compose run --rm app-cpu python -c "import torch; print(torch.__version__)"
# 確認 GPU 直通
docker compose run --rm app python -c "import torch; print(torch.cuda.is_available())"
```

專案資料夾以 `-v .:/app` 掛進容器，改 `.py` 不用重 build；只有動到 `requirements.txt` 才需要重跑 `docker compose build`。模型快取放在 `.cache/huggingface`（`HF_HOME` 指過去），主機與容器共用同一份，只下載一次。

## 執行

### 1. 準備樣本

從 `Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset`（與模型同作者發布，標籤與模型 8 類對應）以固定 seed 抽樣：

```bash
docker compose run --rm app-cpu python scripts/prepare_samples.py --limit 25
```

樣本輸出到 `_local/samples.jsonl`，不進版控 —— 該資料集有自身授權條款，且固定 seed 使任何人重跑都得到同一份。腳本會印出 SHA-256 供核對。

### 2. 跑串流分類

```bash
# 間隔到達：模擬 STT 的 0.5~3 秒不規律停頓（貼近實際使用）
docker compose run --rm app-cpu python run_stream.py --device cpu
docker compose run --rm app     python run_stream.py --device cuda

# 連續到達：不等待，量模型本身的能力上限
docker compose run --rm app-cpu python run_stream.py --device cpu  --no-delay
docker compose run --rm app     python run_stream.py --device cuda --no-delay
```

每收到一句立刻送進模型，不累積等待多句 —— 即時性優先。

兩種到達模式都要測：**在間隔到達的條件下，GPU 的延遲反而高於 CPU**（句間閒置導致降頻，每句都在時脈未拉起時完成），而連續到達時 GPU 快一個量級。這決定了這個環節該不該佔用 GPU，數字見 [`results/summary.md`](results/summary.md)。

常用參數：

| 參數 | 說明 |
| --- | --- |
| `--device {cpu,cuda,auto}` | 指定 cuda 但 GPU 不可用時直接失敗，不靜默降級成 CPU；`auto` 才會降級 |
| `--limit N` | 處理句數，預設 25 |
| `--no-delay` | 句間不等待。等待本身不在計時區間內，但會影響 GPU 的時脈狀態（見上） |
| `--seed N` | 固定句序與間隔，讓兩個裝置跑在相同條件下 |
| `--out PATH` | 輸出路徑，預設 `_local/out/stream_<device>[_nogap].jsonl` |

### 3. 產出摘要

```bash
docker compose run --rm app-cpu python scripts/summarize.py
```

## 輸出格式

逐句結果為 `.jsonl`，一行一句：

```json
{"seq": 4, "warmup": false, "recv_at": "2026-08-06T21:03:11+08:00",
 "text": "……", "pred_label": "開心語調", "confidence": 0.9312,
 "latency_ms": {"tokenize": 0.31, "forward": 18.4, "post": 0.05, "total": 18.8},
 "probs": {"平淡語氣": 0.0041, "關切語調": 0.0102, "…": 0.0},
 "device": "cuda"}
```

`probs` 為完整 8 類機率分布，不只最高分那一類 —— 下游的 Emotion Video Selector 可能需要次高分或信心門檻，只存 argmax 會讓那些策略無法評估。

前 3 句標記 `"warmup": true`：CUDA kernel autotune 會使最初幾句明顯偏慢，統計時排除，但保留在檔案裡以便追溯。

執行環境（torch 版本、模型 id、樣本檔雜湊、時間戳）另存同名的 `.meta.json`。

**逐句結果不進版控**（含資料集原文，且重跑即有）；不含原文的速度摘要見 `results/summary.md`。
