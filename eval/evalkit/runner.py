"""推論迴圈：計時、VRAM 量測、落地原生輸出（規劃書 v2 §4.3(C)）。

⚠️ 這一層完全不做標籤映射。只存模型原生類別與完整機率向量，
   映射、指標、報告都是後續可重跑的離線步驟。
"""
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pynvml
import torch
import transformers
from tqdm import tqdm

from . import dataset as ds
from .adapters import get_adapter


class GpuMemoryProbe:
    """兩個數字都要記（規劃書 v2 §4.3(C)）。

    torch.cuda.max_memory_allocated() 只算張量，會低估；
    nvml 讀到的 process 佔用含 CUDA context（約 300–600MB），
    那才是「能不能跟 Moshi/JoyGen 共用同一張 4GB 卡」要看的數字。

    nvml 沒有內建峰值追蹤，所以採取多點取樣取最大值 —— 這是取樣值不是真峰值，
    報告中據實標注。
    """

    def __init__(self, device: str):
        self.enabled = device.startswith("cuda") and torch.cuda.is_available()
        self.index = torch.cuda.current_device() if self.enabled else None
        self.nvml_peak_mb = None
        self.nvml_note = None
        self._handle = None
        self._pynvml = None

        if not self.enabled:
            return

        torch.cuda.reset_peak_memory_stats()
        try:
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
        except Exception as exc:  # 沒裝 / 沒權限 / WSL 不支援，都降級不中斷評測
            self.nvml_note = f"nvml 不可用，只有 torch 數字：{type(exc).__name__}: {exc}"

    def sample(self) -> None:
        if not self._handle:
            return
        try:
            procs = self._pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
            mine = [p for p in procs if p.pid == os.getpid()]
            if mine and mine[0].usedGpuMemory:
                mb = mine[0].usedGpuMemory / 1024 / 1024
                self.nvml_peak_mb = mb if self.nvml_peak_mb is None else max(self.nvml_peak_mb, mb)
        except Exception as exc:
            self.nvml_note = f"nvml 取樣失敗：{type(exc).__name__}: {exc}"
            self._handle = None

    def result(self) -> dict:
        if not self.enabled:
            return {"device": "cpu", "note": "CPU 執行，不量 VRAM"}
        out = {
            "torch_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
            "torch_peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024 / 1024, 1),
            "nvml_peak_process_mb": round(self.nvml_peak_mb, 1) if self.nvml_peak_mb else None,
            "nvml_sampling": "多點取樣的最大值，非連續追蹤的真峰值",
        }
        if self.nvml_note:
            out["note"] = self.nvml_note
        return out


def environment_fingerprint(device: str) -> dict:
    """三週後回頭看報告時，這些是唯一能確認當初量測條件的依據。"""
    fp = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "in_container": Path("/.dockerenv").exists(),
        "image_tag": os.environ.get("IMAGE_TAG", "unknown"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device,
        "torch_num_threads": torch.get_num_threads(),
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        fp["gpu"] = torch.cuda.get_device_name()
        fp["cuda"] = torch.version.cuda
        fp["gpu_total_mb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, 1
        )
        try:
            pynvml.nvmlInit()
            fp["driver"] = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(fp["driver"], bytes):
                fp["driver"] = fp["driver"].decode()
        except Exception:
            fp["driver"] = "unknown"
    return fp


def run(cfg: dict, model_cfg: dict, device: str, variant: str, out_dir: Path,
        limit=None, skip_throughput=False) -> dict:
    runtime = cfg["runtime"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "指定了 --device cuda 但 torch 看不到 GPU。\n"
            "容器內請用 docker compose run --rm bert-eval（有掛 --gpus），"
            "或改用 --device cpu。"
        )

    torch.set_num_threads(runtime.get("torch_num_threads", 4))

    rows = ds.load_testset(
        Path(cfg["_repo_root"]) / cfg["dataset"]["path"],
        cfg["dataset"].get("expect_sha256"),
    )
    if limit:
        rows = rows[:limit]
    texts = [ds.text_of(r, variant) for r in rows]

    adapter_cls = get_adapter(model_cfg["adapter"])
    adapter = adapter_cls(model_cfg, device, runtime["max_length"])

    probe = GpuMemoryProbe(device)
    print(f"[{model_cfg['key']}] 載入模型（{device} / {variant}）…")
    adapter.load()
    probe.sample()
    print(f"  模型載入耗時 {adapter.load_seconds:.1f}s")

    # --- cold start：載入後的第一次推論，跟穩定期分開記 -----------------------
    t0 = time.perf_counter()
    adapter.predict(texts[0])
    cold_start_ms = (time.perf_counter() - t0) * 1000
    print(f"  首次推論（cold start）{cold_start_ms:.1f}ms")

    # --- warm-up：只跑不計時 ------------------------------------------------
    n_warmup = min(runtime.get("warmup", 20), len(texts))
    for text in texts[:n_warmup]:
        adapter.predict(text)
    probe.sample()

    # --- 正式量測：batch=1 逐句，這是產品場景的數字 --------------------------
    predictions = []
    sample_every = max(1, len(rows) // 20)
    for i, (row, text) in enumerate(tqdm(list(zip(rows, texts)), desc="  推論", unit="句")):
        pred = adapter.predict(text)
        predictions.append({
            "id": row["id"],
            "true_label": row["true_label"],
            "raw_label": pred.raw_label,
            "raw_probs": pred.raw_probs,
            "confidence": round(pred.confidence, 6),
            "latency_ms": {k: round(v, 3) for k, v in pred.timings_ms.items()},
        })
        if i % sample_every == 0:
            probe.sample()
    probe.sample()

    # --- 吞吐量參考值（非產品場景） -----------------------------------------
    throughput = {}
    if not skip_throughput:
        subset = texts[: min(256, len(texts))]
        for bs in runtime.get("batch_sizes_for_throughput", []):
            t0 = time.perf_counter()
            for start in range(0, len(subset), bs):
                adapter.predict_batch(subset[start:start + bs])
            elapsed = time.perf_counter() - t0
            throughput[f"batch_{bs}"] = {
                "sentences": len(subset),
                "seconds": round(elapsed, 3),
                "sentences_per_sec": round(len(subset) / elapsed, 1),
                "ms_per_sentence": round(elapsed / len(subset) * 1000, 3),
            }
            probe.sample()

    truncated = adapter.truncation_count(texts) if hasattr(adapter, "truncation_count") else None

    meta = {
        "model_key": model_cfg["key"],
        "hf_id": model_cfg["hf_id"],
        "device": device,
        "text_variant": variant,
        "n_sentences": len(rows),
        "dataset_sha256": ds.sha256_of(Path(cfg["_repo_root"]) / cfg["dataset"]["path"]),
        "max_length": runtime["max_length"],
        "truncated_sentences": truncated,
        "warmup": n_warmup,
        "model_load_seconds": round(adapter.load_seconds, 3),
        "cold_start_ms": round(cold_start_ms, 3),
        "throughput_reference_only": throughput,
        "vram": probe.result(),
        "environment": environment_fingerprint(device),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ds.write_jsonl(predictions, out_dir / "predictions.jsonl")
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    adapter.unload()

    if truncated:
        print(f"  ⚠️ 有 {truncated} 句超過 max_length={runtime['max_length']} 被截斷")
    print(f"  → {out_dir / 'predictions.jsonl'}")
    return meta
