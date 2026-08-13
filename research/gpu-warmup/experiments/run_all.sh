#!/bin/sh
# 一次跑完所有 GPU 實驗（約 55 分鐘）。
#
#   docker compose -f docker/docker-compose.yml run --rm app \
#       sh research/gpu-warmup/experiments/run_all.sh
#
# ⚠️ 刻意不平行，也不要在跑的時候用機器：GPU 實驗對「還有沒有別的東西在用卡」
#    極度敏感，瀏覽器硬體加速就足以汙染數字。
set -x

# 1. 降頻/回升的時間曲線（約 30 秒）
python research/gpu-warmup/experiments/exp1_gap_sweep.py --mode decay

# 2. 間隔 → 時脈 → 延遲的主表（約 11 分鐘）
python research/gpu-warmup/experiments/exp1_gap_sweep.py --mode sweep --n-per-gap 25

# 3. 各種心跳手段（約 12 分鐘）
python research/gpu-warmup/experiments/exp2_keepalive.py --gap 1.5 --n 25

# 4. 降低單次成本：fp16 / torch.compile / CUDA Graph（約 15 分鐘）
python research/gpu-warmup/experiments/exp3_reduce_cost.py --n 25

# 5. fp16 的準確率對照（兩份資料，約 5 分鐘）
python research/gpu-warmup/experiments/exp4_fp16_parity.py
python research/gpu-warmup/experiments/exp4_fp16_parity.py \
  --samples eval/data/testset/dataset_v1.jsonl --text-field text_zh_tw --gold-field true_label \
  --out research/gpu-warmup/results/exp4_fp16_parity_smp1200.json

# 6. TF32 對照（約 3 分鐘）
python research/gpu-warmup/experiments/exp3_reduce_cost.py --n 25 --only tf32 eager --tag tf32

# 7. CPU 對照組（約 11 分鐘）
python research/gpu-warmup/experiments/exp1_gap_sweep.py --mode sweep --device cpu --n-per-gap 25
