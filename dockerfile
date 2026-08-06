# imood BERT 情緒分類 — 評測骨架執行環境
#
# base image 不要換：這組 CUDA/cuDNN/PyTorch 版本是 07/22 實測 GPU 直通成功的組合，
# 也是 leader 提的「避開 VM 上 CUDA 版本過舊」問題的解法。
FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# PYTHONUNBUFFERED：1200 句跑十幾分鐘，沒有這行看不到即時進度
# PYTHONIOENCODING：容器內 print 中文標籤（憤怒語調…）不亂碼
# HF_HOME：模型快取集中到這個路徑，配合 compose 的 hf-cache volume，
#          large 版模型（約 1.4GB）只會下載一次，換 image 不用重下
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    HF_HOME=/app/.cache/huggingface

# 先只 COPY requirements：只要它沒變，下面這層 pip install 就吃快取，
# 改 .py 不會觸發重裝套件
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# 實際打包哪些檔案由 .dockerignore 決定（_sandbox/、.cache/ 都排除，
# 否則轉向前的凍結資料集與模型快取會被塞進 image，體積爆掉）
COPY . .

# 開發時多半用 -v 掛載 + 在指令列覆蓋 CMD。MVP 程式碼還沒進來，
# 這裡放一行環境自我檢查當安全預設值：直接 docker run 只會印版本，不會誤跑東西
CMD ["python", "-c", "import torch, transformers; print(f'torch {torch.__version__} / cuda {torch.cuda.is_available()} / transformers {transformers.__version__}')"]
