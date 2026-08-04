# imood AI — Docker 開發環境

imood 情緒辨識的 AI 端。這個 repo 從 2026-08-04 起專心做 MVP，
目前只放「能跑 Docker、能開始寫程式」的最小環境。

## 環境內容

| 檔案 | 用途 |
| --- | --- |
| `dockerfile` | base image `pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime`，07/22 實測 GPU 直通成功的組合，不要換 |
| `docker-compose.yml` | `app`（吃 GPU）與 `app-cpu`（純 CPU 對照組）兩個服務 |
| `requirements.txt` | 套件清單。**不要在這裡放 torch/torchvision**，base image 已內建對得上的版本 |
| `.env` | `HF_TOKEN`，走 `--env-file` 帶進容器，不進版控 |

## 快速開始

```bash
docker compose build

# 確認環境活著
docker compose run --rm app-cpu python -c "import torch; print(torch.__version__)"

# 確認 GPU 直通
docker compose run --rm app python -c "import torch; print(torch.cuda.is_available())"
```

專案資料夾以 `-v .:/app` 掛進容器，改 `.py` 不用重 build；
只有動到 `requirements.txt` 才需要重跑 `docker compose build`。

HF 模型快取放在 `.cache/huggingface`（`HF_HOME` 指過去），
主機與容器共用同一份，1.1GB 的模型只下載一次。

## `_sandbox/`

轉向 MVP 之前的評測骨架（`evalkit/`、`run_eval.py`、`scripts/`、`configs/`、
凍結測試集 `data/`、評測產出 `results/`）整包移到 `_sandbox/`，**不進版控**。

留在本機是為了要用的時候撿得回來，也因為 SMP2020 資料集有自己的授權，
公開 repo 不該轉散布。容器內照樣掛得到：

```bash
docker compose run --rm app-cpu python _sandbox/scripts/selftest.py
docker compose run --rm app python _sandbox/run_eval.py --model johnson-small
```

MVP 需要哪一塊，再從 `_sandbox/` 一份份撿進正式目錄。
