"""產出 report.md 與 dashboard.html（規劃書 v2 §4.4）。

dashboard.html 是單一檔案、資料內嵌、無任何外部 CDN 依賴：
容器內離線產得出來，產出的檔案可以直接寄給 leader 或隊友，對方雙擊就能開。
"""
import json
from pathlib import Path

from . import metrics as mt

HTML_TEMPLATE_PATH = Path(__file__).parent / "dashboard_template.html"


# ---------------------------------------------------------------------------
# report.md
# ---------------------------------------------------------------------------
def _fmt_pct(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def build_markdown(meta: dict, results: dict, default_mapping: str, label_zh: dict) -> str:
    L = []
    a = L.append

    a(f"# 評測報告 — {meta['model_key']}")
    a("")
    a(f"- 模型：`{meta['hf_id']}`")
    a(f"- 裝置：{meta['device']}｜文字變體：{meta['text_variant']}｜句數：{meta['n_sentences']}")
    a(f"- 測試集 SHA-256：`{meta['dataset_sha256'][:16]}…`")
    env = meta["environment"]
    a(f"- 環境：torch {env['torch']}｜transformers {env['transformers']}"
      f"｜{env.get('gpu', 'CPU')}｜{env['timestamp']}")
    a("")

    # --- 決策用的三個數字 ---------------------------------------------------
    strict = results[default_mapping]["views"]["strict"]
    lat = results[default_mapping]["latency"]["total"]
    vram = meta["vram"]
    a("## 決策用的三個數字")
    a("")
    a("| 指標 | 數值 | 說明 |")
    a("|---|---|---|")
    a(f"| Macro-F1（strict） | **{strict['macro_f1']:.4f}** | 映射 `{default_mapping}`，六類各自 F1 的平均 |")
    a(f"| p95 延遲（batch=1） | **{lat['p95']:.1f} ms** | 不是平均值，這才對應「會不會感覺卡頓」 |")
    nvml = vram.get("nvml_peak_process_mb")
    a(f"| VRAM 峰值（nvml） | **{nvml if nvml else '—'} MB** | 含 CUDA context，共卡決策看這個 |")
    a("")

    # --- 映射方案比較 -------------------------------------------------------
    a("## 映射方案比較")
    a("")
    a("同一份推論結果套用不同映射表重算，成本為零。")
    a("")
    a("| 映射 | strict Acc | strict Macro-F1 | covered Acc | coverage | subset Macro-F1 |")
    a("|---|---|---|---|---|---|")
    for name, res in results.items():
        v = res["views"]
        star = " ⭐" if name == default_mapping else ""
        a(f"| `{name}`{star} | {_fmt_pct(v['strict']['accuracy'])} | {v['strict']['macro_f1']:.4f} "
          f"| {_fmt_pct(v['covered']['accuracy'])} | {_fmt_pct(v['covered']['coverage'])} "
          f"| {v['subset']['macro_f1']:.4f} |")
    a("")

    # --- mapping-free -------------------------------------------------------
    mf = results[default_mapping]["mapping_free"]
    a("## mapping-free 指標（不受映射表猜錯影響）")
    a("")
    a(f"- **最佳映射上界**：{_fmt_pct(mf['best_possible']['accuracy_upper_bound'])}")
    a(f"- 目前映射 `{default_mapping}` 的 strict 準確率：{_fmt_pct(mf['current_strict_accuracy'])}")
    a(f"- **映射猜錯造成的損失**：{_fmt_pct(mf['loss_from_mapping_choice'])}")
    a(f"- NMI：{mf['nmi']}｜ARI：{mf['ari']}（衡量原生分類與六類標籤的結構相關性，不看映射）")
    a("")
    a("最佳映射（模型自己的資料說了算，非人工指定）：")
    a("")
    a("| 原生類別 | 最該映到 | 目前映射 | 一致？ |")
    a("|---|---|---|---|")
    current = results[default_mapping]["mapping"]
    for raw, best in mf["best_possible"]["mapping"].items():
        cur = current.get(raw)
        same = "✅" if cur == best else "⚠️"
        a(f"| {raw} | {best} | {cur if cur else '（棄權）'} | {same} |")
    a("")

    a("### 每組映射的雙向支持度")
    a("")
    a("`precision` = P(真實=目標 | 原生=該類)：判成這類的句子裡，真的有多少是目標標籤")
    a("")
    a("`recall` = P(原生=該類 | 真實=目標)：真的是目標標籤的句子裡，有多少被判成這類")
    a("")
    a("| 原生類別 | 映到 | 該類句數 | precision | recall |")
    a("|---|---|---|---|---|")
    for row in mf["support"]:
        target = row["target"] if row["target"] else "（棄權）"
        a(f"| {row['raw']} | {target} | {row['n_raw']} "
          f"| {_fmt_pct(row.get('precision'))} | {_fmt_pct(row.get('recall'))} |")
    a("")

    # --- per-class ----------------------------------------------------------
    a(f"## 各類別表現（映射 `{default_mapping}`，strict 視角）")
    a("")
    a("| 標籤 | 中文 | precision | recall | F1 | 樣本數 |")
    a("|---|---|---|---|---|---|")
    for label, s in strict["per_class"].items():
        a(f"| {label} | {label_zh.get(label, '')} | {_fmt_pct(s['precision'])} "
          f"| {_fmt_pct(s['recall'])} | {s['f1']:.4f} | {s['support']} |")
    a("")

    # --- 混淆矩陣 -----------------------------------------------------------
    conf = strict["confusion"]
    a("### 混淆矩陣（列＝真實，欄＝預測）")
    a("")
    a("| 真實＼預測 | " + " | ".join(conf["cols"]) + " |")
    a("|---" * (len(conf["cols"]) + 1) + "|")
    for t in conf["rows"]:
        cells = " | ".join(str(conf["matrix"][t][c]) for c in conf["cols"])
        a(f"| **{t}** | {cells} |")
    a("")

    # --- 時間 / 資源 --------------------------------------------------------
    a("## 時間與資源")
    a("")
    a("| 項目 | 數值 |")
    a("|---|---|")
    a(f"| 模型載入（cold start） | {meta['model_load_seconds']:.1f} s |")
    a(f"| 首次推論 | {meta['cold_start_ms']:.1f} ms |")
    lt = results[default_mapping]["latency"]
    a(f"| 單句延遲 mean / p50 / p95 / p99 | "
      f"{lt['total']['mean']:.1f} / {lt['total']['p50']:.1f} / "
      f"{lt['total']['p95']:.1f} / {lt['total']['p99']:.1f} ms |")
    if "tokenize" in lt:
        a(f"| 其中 tokenize / forward / post（mean） | "
          f"{lt['tokenize']['mean']:.2f} / {lt['forward']['mean']:.2f} / "
          f"{lt['post']['mean']:.2f} ms |")
    if vram.get("torch_peak_allocated_mb"):
        a(f"| VRAM torch 張量峰值 | {vram['torch_peak_allocated_mb']} MB（低估，不含 context） |")
        a(f"| VRAM nvml process 峰值 | {nvml if nvml else '—'} MB（含 CUDA context） |")
    if meta.get("truncated_sentences"):
        a(f"| 超過 max_length 被截斷 | {meta['truncated_sentences']} 句 |")
    a("")

    if meta.get("throughput_reference_only"):
        a("### 吞吐量參考（⚠️ 非產品場景，產品是單句即時）")
        a("")
        a("| batch | 句/秒 | 每句 ms |")
        a("|---|---|---|")
        for name, t in meta["throughput_reference_only"].items():
            a(f"| {name} | {t['sentences_per_sec']} | {t['ms_per_sentence']} |")
        a("")

    # --- 信心校準 -----------------------------------------------------------
    cal = strict["calibration"]
    a("## 信心校準")
    a("")
    a(f"ECE = **{cal['ece']}**（越接近 0 代表 softmax 機率越貼近真實準確率）"
      f"｜平均信心 {_fmt_pct(cal['mean_confidence'])}｜實際準確率 {_fmt_pct(cal['overall_accuracy'])}")
    a("")
    a("| 信心門檻 | 覆蓋率 | 保留句數 | 該子集準確率 |")
    a("|---|---|---|---|")
    for row in cal["thresholds"]:
        a(f"| ≥ {row['threshold']} | {_fmt_pct(row['coverage'])} | {row['kept']} "
          f"| {_fmt_pct(row['accuracy'])} |")
    a("")
    a("> 用法：這張表回答「要不要加 confidence threshold」。"
      "例如設 0.8 能擋掉多少句、剩下的準確率升到多少。")
    a("")

    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# dashboard.html
# ---------------------------------------------------------------------------
def build_dashboard(meta: dict, results: dict, default_mapping: str,
                    label_zh: dict, predictions: list, texts: dict) -> str:
    sentences = []
    for p in predictions:
        mapped = {name: res["mapping"].get(p["raw_label"]) for name, res in results.items()}
        sentences.append({
            "id": p["id"],
            "true": p["true_label"],
            "raw": p["raw_label"],
            "conf": p["confidence"],
            "lat": p["latency_ms"]["total"],
            "text": texts.get(p["id"], ""),
            "mapped": mapped,
        })

    payload = {
        "meta": meta,
        "labelZh": label_zh,
        "defaultMapping": default_mapping,
        "results": results,
        "sentences": sentences,
    }

    template = HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.replace(
        "/*__PAYLOAD__*/null",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )


def write_all(out_dir: Path, meta: dict, results: dict, default_mapping: str,
              label_zh: dict, predictions: list, texts: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(
        build_markdown(meta, results, default_mapping, label_zh), encoding="utf-8"
    )
    (out_dir / "dashboard.html").write_text(
        build_dashboard(meta, results, default_mapping, label_zh, predictions, texts),
        encoding="utf-8",
    )

    print(f"  → {out_dir / 'report.md'}")
    print(f"  → {out_dir / 'dashboard.html'}")


def evaluate_all(predictions: list, model_cfg: dict, labels: list) -> dict:
    """對設定檔裡的每一組映射各算一次。成本為零 —— 同一份 predictions 重算而已。"""
    return {
        name: mt.evaluate(predictions, model_cfg, labels, name)
        for name in model_cfg["mappings"]
    }
