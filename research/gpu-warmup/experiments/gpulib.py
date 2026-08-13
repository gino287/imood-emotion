"""GPU 實驗共用工具：NVML 取樣、樣本讀取、統計。

⚠️ 本檔案（以及 research/ 底下所有程式）的註解是 AI 起草的，尚未經 Gino 改寫。
   若其中任何邏輯要搬進 imood_emotion/ 或 eval/，依 CLAUDE.md 規則 4 需先自行重寫註解。
"""
import json
import platform
import statistics
import time
from pathlib import Path

import pynvml
import torch

# NVML 的 throttle reason 是 bitmask，這裡只列本專案量得到的幾個。
# 0x1 = GpuIdle，正是「閒置降頻」這個假設要驗證的旗標。
THROTTLE_BITS = {
    0x0000000001: "GpuIdle",
    0x0000000002: "ApplicationsClocksSetting",
    0x0000000004: "SwPowerCap",
    0x0000000008: "HwSlowdown",
    0x0000000010: "SyncBoost",
    0x0000000020: "SwThermalSlowdown",
    0x0000000040: "HwThermalSlowdown",
    0x0000000080: "HwPowerBrakeSlowdown",
    0x0000000100: "DisplayClockSetting",
}


def decode_throttle(mask: int) -> list:
    return [name for bit, name in THROTTLE_BITS.items() if mask & bit]


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


class NvmlProbe:
    """包一層 NVML。

    ⚠️ WSL2 底下各個 NVML 查詢的成本差非常多，實測（`_nvml_cost.py`）：

        clock_sm            0.158 ms
        power               0.118 ms
        temperature         0.062 ms
        utilization         0.507 ms
        throttle_reasons   18.283 ms   ← 差兩個數量級

    第一版把 throttle_reasons 放進逐句迴圈，前後各抓一次 = 每句多 40ms 空檔，
    等於偷偷把「連續到達」變成「每 40ms 到達一次」，時脈根本爬不上去
    ——量到的降頻其實有一部分是量測工具自己造成的。
    （WSL2 的 nvidia-smi/NVML 是轉發到 Windows 驅動的 shim，
      每次呼叫都要跨越那層，某些查詢特別貴。）

    所以拆成兩個：
      snapshot()      逐句用，只抓便宜的欄位（約 0.34ms）
      snapshot_full() 偶爾用，含 throttle_reasons
    """

    def __init__(self, index: int = 0):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(index)

    def sm_clock(self) -> int:
        return pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM)

    def snapshot(self) -> dict:
        """逐句迴圈用的輕量版：只有便宜的三個欄位，整包約 0.34ms。"""
        return {
            "sm_mhz": _safe(lambda: pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM), -1),
            "power_w": _safe(lambda: round(pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0, 2), -1),
            "temp_c": _safe(lambda: pynvml.nvmlDeviceGetTemperature(self.h, pynvml.NVML_TEMPERATURE_GPU), -1),
        }

    def snapshot_full(self) -> dict:
        """含 throttle_reasons 與 utilization 的完整版，約 20ms。
        只在條件開始/結束這種不影響量測的地方呼叫。"""
        s = self.snapshot()
        s.update({
            "mem_mhz": _safe(lambda: pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_MEM), -1),
            "util_pct": _safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(self.h).gpu, -1),
            "throttle": decode_throttle(_safe(
                lambda: pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.h), 0)),
        })
        return s

    def static_info(self) -> dict:
        """WSL2 底下有些欄位 NVML 回 NotSupported（例如 power limit），
        一律包 try 讓實驗不要因為抓不到一個附註欄位就整份掛掉。"""
        info = {}
        for key, fn in [
            ("name", lambda: pynvml.nvmlDeviceGetName(self.h)),
            ("driver", lambda: pynvml.nvmlSystemGetDriverVersion()),
            ("max_sm_mhz", lambda: pynvml.nvmlDeviceGetMaxClockInfo(self.h, pynvml.NVML_CLOCK_SM)),
            ("power_limit_w", lambda: round(
                pynvml.nvmlDeviceGetPowerManagementLimit(self.h) / 1000.0, 2)),
            ("persistence_mode", lambda: pynvml.nvmlDeviceGetPersistenceMode(self.h)),
        ]:
            try:
                info[key] = fn()
            except Exception as exc:
                info[key] = f"unsupported: {type(exc).__name__}"
        return info


class NullProbe:
    """--device cpu 時的替身，讓同一支腳本不用為了 CPU 分支多寫一套。
    回傳 -1 而不是 None，這樣下游算統計時型別一致。"""

    def sm_clock(self):
        return -1

    def snapshot(self) -> dict:
        return {"sm_mhz": -1, "power_w": -1, "temp_c": -1}

    def snapshot_full(self) -> dict:
        return {"sm_mhz": -1, "power_w": -1, "temp_c": -1,
                "mem_mhz": -1, "util_pct": -1, "throttle": []}

    def static_info(self) -> dict:
        return {"name": "cpu (no NVML)"}


def load_samples(path: str = "_local/samples.jsonl", n: int = None) -> list:
    texts = [json.loads(line)["text"] for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return texts[:n] if n else texts


def pct(values, q):
    """不靠 numpy 的百分位（線性內插），跟 numpy.percentile 對得起來。"""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 3)


def summarize(values) -> dict:
    if not values:
        return {}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 3),
        "p50": pct(values, 50),
        "p90": pct(values, 90),
        "p95": pct(values, 95),
        "p99": pct(values, 99),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
    }


def env_meta() -> dict:
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
    return meta
