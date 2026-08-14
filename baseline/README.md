# baseline

跑 200 句速度 baseline 的整條測試流程。**這個資料夾不上線** —— 接上真實 STT 之後，
底下沒有任何檔案會被呼叫到。

## 執行順序

```bash
# 1. 抽樣：固定 seed 均衡抽 200 句 → _local/samples.jsonl
docker compose -f docker/docker-compose.yml run --rm app-cpu     python baseline/prepare_samples.py --limit 200

# 2. 跑流程：模擬上游逐句吐字 → 前處理 → 推論 → 輸出 → _local/out/
docker compose -f docker/docker-compose.yml run --rm app-cpu     python baseline/run_baseline.py --device cpu  --limit 200
docker compose -f docker/docker-compose.yml run --rm app     python baseline/run_baseline.py --device cuda --limit 200

# 3. 出摘要：讀 _local/out/ → baseline/results/summary.md
docker compose -f docker/docker-compose.yml run --rm app-cpu     python baseline/checks/summarize.py
```

參數說明與 baseline 數字見專案根目錄的 [`README.md`](../README.md)。

## 檔案

| 檔案 | 是什麼 |
| --- | --- |
| `prepare_samples.py` | 入口①：從 HF 資料集抽樣，輸出樣本檔並印出 SHA-256 |
| `run_baseline.py` | 入口②：串起「假上游 → 前處理 → 推論 → 輸出封包」跑完整條流程 |
| `fake_stt.py` | 假上游：把抽好的文字以不規律間隔逐句吐出，不碰語音、不呼叫 `stt/` |
| `recorder.py` | 逐句寫 `.jsonl`，另寫 `.meta.json` 記錄版本與環境指紋 |
| `checks/` | 事後驗證與分析（標籤順序、速度摘要、截斷穩定度） |
| `results/` | 不含資料集原文的摘要，進版控 |

動詞開頭的（`prepare_` / `run_`）是可以直接執行的入口，名詞的是被 import 的零件。

## checks/

| 檔案 | 是什麼 |
| --- | --- |
| `verify_labels.py` | 拿標註句跑混淆矩陣，驗證 8 類標籤順序沒接錯 |
| `summarize.py` | 讀 `_local/out/*.jsonl` 出速度摘要 → `results/summary.md` |
| `stress_fragments.py` | 隨機截斷句子，量不完整輸入的穩定度 → `results/fragment_robustness.md` |

## 為什麼假上游放這裡，不放 emotion/

`emotion/` 只收上線之後真的會被呼叫到的東西。`fake_stt.py` 重播已經抽好的文字樣本、
`recorder.py` 寫的 `dataset_label` 與 `samples_sha256` 只有跑實驗才有意義 ——
兩者都只在測試時用得到，混在 production 套件裡會讓「哪些是上線邏輯」變得看不出來。

`run_baseline.py` 裡「前處理 → 推論 → 輸出封包」的順序，就是上線時 driver 該有的順序；
但它寫死用 `fake_stt` 與 `recorder`，所以它是測試驅動程式，不是上線邏輯。
真正上線時會另外補一支 driver 放在 `emotion/`。
