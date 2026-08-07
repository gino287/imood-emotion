"""主機端錄音：每 N 秒切一段 wav，丟進 _local/audio/ 給容器取用。

    # 先確認麥克風收得到聲音（會即時顯示音量）
    _local/hostenv/Scripts/python scripts/record_mic.py --check

    # 正式錄音（容器端另開一個視窗跑 run_stream.py --input mic）
    _local/hostenv/Scripts/python scripts/record_mic.py

⚠️ 這支腳本跑在**主機**，不是容器裡。
   Docker Desktop on Windows 不支援音訊裝置直通（容器內沒有 /dev/snd、
   沒有 PulseAudio），麥克風只能在主機讀。

⚠️ 依賴刻意只有 sounddevice 與 numpy（見 requirements-host.txt），
   也刻意不 import 任何 imood_stream 模組 —— 主機端不該碰到 ML 依賴。
"""
import argparse
import os
import sys
import time
import wave
from pathlib import Path

# Windows 主控台預設是 cp950，直接 print 中文會變亂碼
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402

SAMPLE_RATE = 16000   # Whisper 的原生取樣率，先在這裡轉好省得後面再重取樣
CHANNELS = 1
DEFAULT_DIR = Path("_local/audio")

# 低於這個 RMS 視為整段沒有人講話，不寫檔。
# 這不是 VAD —— 只是一個振幅門檻，避免把純靜音段送去轉錄
# （Whisper 對靜音容易產生「謝謝觀看」這類幻覺）。
SILENCE_RMS = 0.002


def parse_args():
    p = argparse.ArgumentParser(description="主機端麥克風錄音，切段寫給容器")
    p.add_argument("--chunk-sec", type=float, default=4.0,
                   help="每段秒數（預設 4.0）")
    p.add_argument("--device", type=int, default=None,
                   help="輸入裝置編號，省略則用系統預設；--list-devices 可查")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--list-devices", action="store_true", help="列出可用的輸入裝置後結束")
    p.add_argument("--check", action="store_true",
                   help="音量表模式：即時顯示音量，用來確認麥克風真的收得到聲音")
    p.add_argument("--keep-stale", action="store_true",
                   help="保留 out-dir 裡上一輪殘留的 wav（預設會清掉）")
    p.add_argument("--silence-rms", type=float, default=SILENCE_RMS,
                   help=f"低於此 RMS 的整段視為靜音不寫檔（預設 {SILENCE_RMS}）")
    return p.parse_args()


def list_input_devices() -> None:
    print("可用的輸入裝置：")
    default_in = sd.default.device[0]
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            mark = "  ← 系統預設" if i == default_in else ""
            print(f"  [{i}] {d['name']}"
                  f"  聲道={d['max_input_channels']}"
                  f"  取樣率={int(d['default_samplerate'])}{mark}")


def resolve_device(index):
    """確認裝置存在且可輸入。不靜默退回預設 —— 指定了就要拿到指定的那個。"""
    if index is None:
        return None
    try:
        info = sd.query_devices(index)
    except Exception:
        print(f"找不到裝置編號 {index}。\n", file=sys.stderr)
        list_input_devices()
        raise SystemExit(1)
    if info["max_input_channels"] < 1:
        print(f"裝置 [{index}] {info['name']} 沒有輸入聲道，不能錄音。\n", file=sys.stderr)
        list_input_devices()
        raise SystemExit(1)
    return index


def write_wav_atomic(path: Path, samples: np.ndarray) -> None:
    """先寫 .part 再原子改名。

    容器那端是輪詢資料夾找 *.wav，若直接寫目標檔名，它可能讀到只寫了一半的檔案，
    Whisper 就拿到截斷音訊。這種錯誤只在時序剛好時出現，最難重現，所以從一開始
    就用原子改名擋掉。
    """
    tmp = path.with_suffix(path.suffix + ".part")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    os.replace(tmp, path)   # 同一個檔案系統上是原子操作


def run_check(device) -> None:
    """音量表：靜音時看不出麥克風是壞的還是沒人講話，所以需要這個模式。"""
    print("音量表模式 —— 請對著麥克風說話，Ctrl+C 結束")
    print(f"（RMS 超過 {SILENCE_RMS} 才會被當成有聲音）\n")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            dtype="float32", device=device) as stream:
            while True:
                block, overflowed = stream.read(int(SAMPLE_RATE * 0.2))
                rms = float(np.sqrt((block[:, 0] ** 2).mean()))
                bars = min(50, int(rms * 500))
                state = "有聲音" if rms >= SILENCE_RMS else "靜音  "
                flag = " ⚠️溢位" if overflowed else ""
                print(f"\r  {state} RMS={rms:.5f} |{'█' * bars:<50}|{flag}", end="")
    except KeyboardInterrupt:
        print("\n結束。")


def main():
    args = parse_args()

    if args.list_devices:
        list_input_devices()
        return

    device = resolve_device(args.device)

    if args.check:
        run_check(device)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stale = sorted(args.out_dir.glob("*.wav")) + sorted(args.out_dir.glob("*.wav.part"))
    if stale and not args.keep_stale:
        # 不清的話，容器一啟動就會把上一輪的錄音當成新輸入處理
        for f in stale:
            f.unlink()
        print(f"清掉上一輪殘留的 {len(stale)} 個檔案")

    info = sd.query_devices(device if device is not None else sd.default.device[0])
    print(f"錄音裝置：{info['name']}")
    print(f"每段 {args.chunk_sec} 秒　{SAMPLE_RATE}Hz 單聲道　輸出 → {args.out_dir}")
    print("Ctrl+C 結束。容器端請另開視窗執行 run_stream.py --input mic\n")

    frames = int(SAMPLE_RATE * args.chunk_sec)
    seq, written, skipped = 0, 0, 0

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            dtype="float32", device=device) as stream:
            while True:
                block, overflowed = stream.read(frames)
                seq += 1
                audio = block[:, 0].copy()
                rms = float(np.sqrt((audio ** 2).mean()))
                stamp = time.strftime("%H:%M:%S")

                if rms < args.silence_rms:
                    skipped += 1
                    print(f"  [{seq:>4}] {stamp}  RMS={rms:.5f}  靜音，略過"
                          f"　(已寫 {written} / 略過 {skipped})")
                    continue

                # 檔名用序號補零，容器端照字母序處理就等於時間序
                path = args.out_dir / f"chunk_{seq:06d}.wav"
                write_wav_atomic(path, audio)
                written += 1
                warn = "  ⚠️輸入溢位" if overflowed else ""
                print(f"  [{seq:>4}] {stamp}  RMS={rms:.5f}  → {path.name}"
                      f"　(已寫 {written} / 略過 {skipped}){warn}")
    except KeyboardInterrupt:
        print(f"\n結束。共錄 {seq} 段，寫出 {written}，靜音略過 {skipped}。")
    except Exception as exc:
        raise SystemExit(
            f"\n錄音中斷：{type(exc).__name__}: {exc}\n"
            "  可用 --list-devices 查看裝置，或 --check 確認麥克風收得到聲音。\n"
            "  麥克風若無法使用，容器端可改用 --input simulated 跑模擬串流。"
        )


if __name__ == "__main__":
    main()
