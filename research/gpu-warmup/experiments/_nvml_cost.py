import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pynvml
from gpulib import NvmlProbe
p = NvmlProbe()
def bench(fn, n=200, label=""):
    fn()
    t=time.perf_counter()
    for _ in range(n): fn()
    print(f"{label:<30}{(time.perf_counter()-t)/n*1000:.3f} ms")
h=p.h
bench(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM), label="clock_sm")
bench(lambda: pynvml.nvmlDeviceGetPowerUsage(h), label="power")
bench(lambda: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU), label="temperature")
bench(lambda: pynvml.nvmlDeviceGetUtilizationRates(h), label="utilization")
bench(lambda: pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h), label="throttle_reasons")
bench(lambda: p.snapshot(), label="snapshot() 整包")
