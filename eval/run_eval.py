# 待GINO改寫
"""評測骨架 CLI 入口。

  python run_eval.py --list
  python run_eval.py --model johnson-small
  python run_eval.py --model johnson-small --device cpu --variant zh_tw
  python run_eval.py --model johnson-small --report-only     # 只重算指標與報告，不重跑推論
  python run_eval.py --model johnson-small --limit 30        # 冒煙測試

--report-only 是這套骨架的重點之一：映射是後處理，改完 configs/models.yaml 的
映射表直接重跑報告即可，不必再花一次推論成本（規劃書 v2 §4.1）。
"""
import argparse
import json
import sys
from pathlib import Path

from evalkit import config as cf
from evalkit import dataset as ds
from evalkit import report as rp
from evalkit import runner


def parse_args():
    p = argparse.ArgumentParser(
        description="imood BERT 情緒分類 baseline 評測",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", help="configs/models.yaml 裡的模型 key")
    p.add_argument("--config", help="設定檔路徑（預設 configs/models.yaml）")
    p.add_argument("--device", help="cuda / cpu / all（預設用設定檔的 runtime.devices）")
    p.add_argument("--variant", help="zh_cn / zh_tw / all（預設用設定檔的 runtime.text_variants）")
    p.add_argument("--limit", type=int, help="只跑前 N 句（冒煙測試用）")
    p.add_argument("--report-only", action="store_true",
                   help="不重跑推論，用既有的 predictions.jsonl 重算指標與報告")
    p.add_argument("--no-throughput", action="store_true", help="跳過 batch 吞吐量量測")
    p.add_argument("--list", action="store_true", help="列出設定檔裡的候選模型")
    return p.parse_args()


def resolve(value, fallback):
    if not value:
        return list(fallback)
    if value == "all":
        return list(fallback)
    return [value]


def make_report(cfg, model_cfg, out_dir: Path, variant: str) -> None:
    meta_path = out_dir / "run_meta.json"
    pred_path = out_dir / "predictions.jsonl"
    for path in (pred_path, meta_path):
        if not path.exists():
            raise SystemExit(f"找不到 {path}\n先跑一次推論（去掉 --report-only）")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    predictions = ds.read_jsonl(pred_path)

    rows = ds.load_testset(cf.testset_path(cfg), cfg["dataset"].get("expect_sha256"))
    texts = {r["id"]: ds.text_of(r, variant) for r in rows}

    results = rp.evaluate_all(predictions, model_cfg, cfg["labels"])
    default_mapping = model_cfg.get("default_mapping") or next(iter(model_cfg["mappings"]))
    rp.write_all(out_dir, meta, results, default_mapping, cfg["label_zh"], predictions, texts)

    # 終端機直接秀決策用的數字，不用開檔案
    strict = results[default_mapping]["views"]["strict"]
    mf = results[default_mapping]["mapping_free"]
    print(f"  Macro-F1(strict/{default_mapping}) = {strict['macro_f1']:.4f}"
          f"｜Acc = {strict['accuracy'] * 100:.1f}%"
          f"｜最佳映射上界 = {mf['best_possible']['accuracy_upper_bound'] * 100:.1f}%"
          f"｜NMI = {mf['nmi']}")


def main():
    args = parse_args()
    cfg = cf.load_config(args.config)

    if args.list:
        print(f"設定檔：{cfg['_path']}\n")
        for m in cfg["models"]:
            print(f"  {m['key']:16s} {m['hf_id']}")
            print(f"  {'':16s} adapter={m['adapter']}｜covers={','.join(m['covers'])}"
                  f"｜mappings={','.join(m['mappings'])}")
        return

    if not args.model:
        raise SystemExit("請指定 --model（用 --list 看有哪些）")

    model_cfg = cf.get_model(cfg, args.model)
    devices = resolve(args.device, cfg["runtime"]["devices"])
    variants = resolve(args.variant, cfg["runtime"]["text_variants"])

    for device in devices:
        for variant in variants:
            out_dir = cf.results_dir(cfg, model_cfg["key"], device, variant)
            print(f"\n=== {model_cfg['key']} ｜ {device} ｜ {variant} ===")

            if not args.report_only:
                runner.run(cfg, model_cfg, device, variant, out_dir,
                           limit=args.limit, skip_throughput=args.no_throughput)

            make_report(cfg, model_cfg, out_dir, variant)


if __name__ == "__main__":
    sys.exit(main())
