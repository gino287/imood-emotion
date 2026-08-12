"""讀取 eval/configs/models.yaml。

設定檔是整個骨架的唯一手動維護點：新增候選模型正常情況下只要在這裡加一段，
不動任何一行程式。

所有路徑都以 eval/ 為基準（REPO_ROOT 從 __file__ 推出來），不看 cwd ——
所以從專案根目錄下 `python eval/run_eval.py` 也找得到設定檔與資料。
"""
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "models.yaml"


class ConfigError(RuntimeError):
    pass


def load_config(path=None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise ConfigError(f"找不到設定檔：{path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    _validate(cfg)
    cfg["_path"] = str(path)
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg


def get_model(cfg: dict, key: str) -> dict:
    for m in cfg["models"]:
        if m["key"] == key:
            return m
    known = ", ".join(m["key"] for m in cfg["models"])
    raise ConfigError(f"設定檔裡沒有 key='{key}' 的模型。目前有：{known}")


def resolve_source(cfg: dict) -> Path:
    """找出 SMP2020 原始資料的實際位置。

    容器內掛在 /refs（compose 那條 ${REFS_DIR}:/refs:ro，路徑由 .env 指定），
    主機直接跑則看相對於 eval/ 的候選路徑。列出候選、挑第一個存在的，
    兩種執行方式都不用改設定。
    """
    for candidate in cfg["dataset"]["source_candidates"]:
        p = Path(os.path.expandvars(candidate))
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        if p.exists():
            return p
    tried = "\n  ".join(cfg["dataset"]["source_candidates"])
    raise ConfigError(
        "找不到 SMP2020 原始資料，試過這些路徑：\n  " + tried +
        "\n（容器內請確認 docker-compose.yml 有掛 ../refs:/refs:ro）"
    )


def testset_path(cfg: dict) -> Path:
    return REPO_ROOT / cfg["dataset"]["path"]


def testset_meta_path(cfg: dict) -> Path:
    return REPO_ROOT / cfg["dataset"]["meta"]


def results_dir(cfg: dict, model_key: str, device: str, variant: str) -> Path:
    return REPO_ROOT / "results" / model_key / device / variant


def _validate(cfg: dict) -> None:
    for field in ("labels", "label_zh", "dataset", "runtime", "models"):
        if field not in cfg:
            raise ConfigError(f"設定檔缺少必要欄位：{field}")

    labels = set(cfg["labels"])
    if set(cfg["label_zh"]) != labels:
        raise ConfigError("label_zh 的鍵與 labels 對不上")

    for field in ("source_candidates", "path", "meta", "seed", "per_class"):
        if field not in cfg["dataset"]:
            raise ConfigError(f"設定檔 dataset 區塊缺少必要欄位：{field}")

    for m in cfg["models"]:
        for field in ("key", "hf_id", "adapter", "native_labels", "covers", "mappings"):
            if field not in m:
                raise ConfigError(f"模型 {m.get('key', '?')} 缺少必要欄位：{field}")

        unknown = set(m["covers"]) - labels
        if unknown:
            raise ConfigError(f"模型 {m['key']} 的 covers 有不存在的標籤：{unknown}")

        native = set(m["native_labels"])
        for name, mapping in m["mappings"].items():
            if set(mapping) != native:
                missing = native - set(mapping)
                extra = set(mapping) - native
                raise ConfigError(
                    f"模型 {m['key']} 的映射表 '{name}' 與 native_labels 對不上"
                    f"（少了 {missing or '無'}／多了 {extra or '無'}）"
                )
            bad = {v for v in mapping.values() if v is not None} - labels
            if bad:
                raise ConfigError(f"模型 {m['key']} 的映射表 '{name}' 映到不存在的標籤：{bad}")

        default = m.get("default_mapping")
        if default and default not in m["mappings"]:
            raise ConfigError(f"模型 {m['key']} 的 default_mapping='{default}' 不在 mappings 裡")
