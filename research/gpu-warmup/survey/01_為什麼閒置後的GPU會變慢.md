# 01 — 為什麼「停一下再跑」的 GPU 會比 CPU 還慢

> 這一篇解釋機制。做法與實測在 02、03 篇與 `RESULTS.md`。

---

## 1. 現象重述

原始觀察（197 句，p50）：

| 條件 | CPU | GPU |
|---|---:|---:|
| 間隔到達（0.5~3 秒） | 94.6ms | **134.3ms** |
| 連續到達 | 105.1ms | **17.0ms** |

GPU 連續模式快 6 倍，間歇模式反而輸給 CPU。當時的推論是「消費級 GPU 時脈節流」，
但沒有直接量過時脈。**本次 survey 的第一件事就是把這個推論變成量測**
（結果見 `RESULTS.md`，結論：推論正確）。

---

## 2. 機制：GPU 的 DVFS 與 P-state

### 2.1 GPU 不是「開著就全速」

現代 GPU 用 **DVFS（Dynamic Voltage and Frequency Scaling）** 管理功耗：
驅動每秒鐘會調整好幾次時脈與電壓，依據是負載、功耗餘量、溫度。
NVIDIA 把這套叫做 **GPU Boost**。

對應的狀態機叫 **P-state（Performance State）**：
- `P0` = 最高效能狀態
- `P8`/`P12` 等 = 省電狀態

⚠️ 但 **P-state 跟實際時脈不是一回事**。我們這張卡在完全閒置時
`nvidia-smi` 照樣回報 `P0`，時脈卻只有 210 MHz。
**只看 pstate 會誤判，一定要看 `clocks.sm`。**

### 2.2 這張卡的實測數字（RTX 3050 Laptop, 4GB）

```
最大 SM 時脈    2100 MHz
連續負載下      1927~1935 MHz   （42~49W，78~84°C）
完全閒置        210 MHz         （15~16W）
比值            約 9 倍
```

延遲跟時脈大致成反比：**210 MHz 時單句 130ms，1930 MHz 時單句 16ms**，
比值 8 倍，跟時脈比值對得上。

### 2.3 為什麼「間歇輸入」會踩到這個陷阱

BERT 單句推論只需要 **16ms 的 GPU 時間**。如果句子每 1.5 秒來一次：

```
GPU 佔用率 = 16ms / 1500ms ≈ 1%
```

從驅動的角度看，**這張卡 99% 的時間在發呆**，
於是它做了完全正確的事：降頻省電。
問題是每次句子來的時候，卡都剛好在低時脈狀態。

> **關鍵洞察**：這不是 bug，是電源管理正常運作。
> 「省電」與「低延遲」在間歇性小負載下是直接衝突的目標，
> 而消費級 GPU 的預設偏好是省電。

### 2.4 CPU 為什麼相反（連續反而慢 11%）

CPU 也有 turbo boost，但邏輯不同：

- CPU 的 **idle → active 反應時間是微秒等級**（C-state 退出很快），
  不像 GPU 要好幾百毫秒把時脈拉回來。
- CPU turbo 的限制是 **熱與功耗預算**：連續跑會累積熱，turbo 回落。
- 所以 CPU 是「連續 → 變慢」，GPU 是「間歇 → 變慢」，方向剛好相反。

這解釋了原始數據裡 CPU 連續（105.1ms）比間隔（94.6ms）慢 11% 這件事。

---

## 3. 文獻怎麼說

搜到幾篇近期直接講這個現象的論文：

**《The Energy Cost of Execution-Idle in GPU Clusters》**
> 「間歇性的閒置空檔會帶來瞬時的低效，同時增加延遲與變異度。
>  在一個中間區間裡，**閒置時間越長，速度單調下降，run-to-run 變異也越大**。」

跟我們量到的曲線形狀完全一致（見 `RESULTS.md` 的間隔掃描表）。

**《Edge-Inference Governors Need Memory-Clock State》**
> 「DVFS 的切換延遲大多在 5~10ms，而一次完整推論是 60~100ms。」

這句很重要：**切換本身只要 5~10ms，但那是「切一次」的成本。**
真正貴的是 governor 要「決定該切」——它需要觀察到足夠的負載才會升頻，
而單句 16ms 的負載對它來說根本不算負載。

**同一批文獻的另一個觀察**（跟我們的 `nvidia-smi -lgc` 失敗直接相關）：
> 「一旦 CUDA context 建立起來，SM 時脈就會跳到最高 boost 並停在那裡，
>  即使利用率 0%——把一個變動成本變成固定成本。」

⚠️ **這句在我們的環境不成立**：我們的 CUDA context 全程存在
（同一個 process 一直活著），時脈照樣掉到 210 MHz。
這是資料中心卡（Tesla/A100，通常有 persistence mode 且電源策略不同）
與 **消費級筆電卡** 的行為差異。**不能照抄資料中心的經驗。**

---

## 4. 「冷啟動」的四個層次（別搞混）

講 GPU「熱啟動」時，文獻上其實在講四件不同的事，成本差好幾個數量級：

| 層次 | 內容 | 量級 | 我們有沒有這個問題 |
|---|---|---|---|
| L1 容器/程序啟動 | 拉 image、起 process | 秒~分鐘 | ❌ 服務常駐，沒這問題 |
| L2 模型載入 | 權重從硬碟到 VRAM | 我們實測 **8.0 秒** | ❌ 只發生一次 |
| L3 CUDA 初始化 + kernel autotune | 第一次前向特別慢 | 首句 **433ms** vs 穩態 16ms | ❌ 已用 warmup 解決 |
| **L4 時脈降頻** | 閒置後時脈掉，下一次請求變慢 | **每一次間隔都會發生** | ✅ **就是這個** |

**絕大多數網路文章講的「GPU cold start」是 L1~L3**（serverless、scale-to-zero、
model streaming 那一整套），對我們完全沒用——我們的服務不會被關掉。

**我們的問題是 L4，而 L4 的解法完全不同**：
L1~L3 是「事先做好、只做一次」，L4 是「必須持續維持」。

> 這個區分很重要，因為搜「GPU warm start」找到的資料 95% 是 L1~L3。
> 如果照那些做法（預熱容器、預載模型），我們一樣會慢，因為問題根本不在那裡。

---

## 5. 我們環境的特殊限制：WSL2

本機是 Windows 11 + WSL2 + Docker + NVIDIA Container Toolkit。

WSL2 裡的 `nvidia-smi` 與 NVML **不是真正的 Linux 驅動**，
而是把呼叫透過 `/dev/dxg` 轉發給 Windows 驅動的 shim。後果：

| 功能 | WSL2 容器內 | 實測 |
|---|---|---|
| 查詢時脈/功耗/溫度 | ✅ 可用 | 正常 |
| 查 throttle reasons | ⚠️ 可用但 **超慢** | **18.3ms/次**（見下） |
| `nvidia-smi -lgc`（鎖時脈） | ❌ | `Unknown Error` |
| `nvidia-smi -pm 1`（persistence） | 回報成功 | 但在 WSL2 沒有實質意義 |
| `nvmlDeviceGetPowerManagementLimit` | ❌ | `NotSupported` |

**鎖時脈這條路在容器內是死的。** 要鎖只能從 Windows 主機端想辦法（見 02 篇）。

### ⚠️ 一個把我自己坑了一次的量測陷阱

各 NVML 查詢在 WSL2 的成本差兩個數量級（`experiments/_nvml_cost.py` 實測）：

```
clock_sm            0.158 ms
power               0.118 ms
temperature         0.062 ms
utilization         0.507 ms
throttle_reasons   18.283 ms    ← 兇手
```

第一版的量測程式把 `throttle_reasons` 放進逐句迴圈、前後各抓一次，
**等於每句偷偷插入 40ms 的空檔**——把「連續到達」變成「每 40ms 到達一次」，
時脈根本爬不上去。**量到的降頻有一部分是量測工具自己造成的。**

→ 教訓：**在量微秒/毫秒級延遲時，任何 instrumentation 都要先量它自己的成本。**
（v1/v2 的資料留在 `results/` 沒刪，檔名帶 `_v1_interleaved`，正好當反例。）

### ⚠️ 第二個陷阱：NVML 回報的時脈是落後指標

實測（`exp1 decay` 的 recover 階段）：
閒置 12 秒後重新開始連續推論，
**第 3 句開始延遲就回到 16ms，但 NVML 還連續 0.74 秒回報 210 MHz。**

代表 **NVML 的時脈值大約每 0.5 秒才更新一次**。所以：

- ✅ 可以用它看 **穩態**（某個到達率下，時脈長期停在哪）
- ❌ **不能** 用它做「這一句的時脈是多少」的逐句關聯
  ——會得到「210 MHz 時延遲 16ms」這種自相矛盾的結論

---

## 6. 這一篇的結論

1. 現象的機制是 **DVFS 降頻**，不是 CUDA 初始化、不是模型載入。
   實測 210 MHz（閒置）vs 1930 MHz（連續），延遲比 8 倍，對得起來。
2. 單句 16ms 對 1.5 秒間隔而言只有 **1% 佔用率**，驅動判定「閒置」完全合理。
3. 網路上的 "GPU cold start" 資料多半在講 L1~L3（容器/載入/初始化），
   **跟我們的 L4（降頻）不是同一個問題，解法也不同**。
4. **WSL2 容器內鎖不了時脈**，要另尋出路。
5. 量測本身有兩個坑：`throttle_reasons` 貴到會污染實驗；
   NVML 時脈是落後指標，只能看穩態。

---

**下一篇**：`02_有哪些解法.md` —— 硬體端鎖時脈、軟體端心跳、以及降低單次成本三條路。

## 參考

- [The Energy Cost of Execution-Idle in GPU Clusters](https://arxiv.org/pdf/2604.04745)
- [Edge-Inference Governors Need Memory-Clock State](https://arxiv.org/pdf/2606.16106)
- [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) · [Known Limitations with CUDA on WSL 2](https://forums.developer.nvidia.com/t/known-limitations-with-cuda-on-wsl-2/128506)
- [NVIDIA GPU Boost 技術說明](https://www.nvidia.com/en-us/geforce/technologies/gpu-boost/technology)
- [nvidia-smi 手冊](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
- [GPU Power Management: Persistence Mode](https://gigagpu.com/gpu-power-management-persistence-mode/)
