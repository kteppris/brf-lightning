from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Mapping

import lightning as L
import yaml


def _locate(dotted: str):
    mod, name = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)


def _build(block: Mapping[str, Any]):
    """Instantiate one YAML block (recursively resolves sub-blocks)."""
    cls = _locate(block["class_path"])
    kwargs = {
        k: _build(v) if isinstance(v, Mapping) and "class_path" in v else v
        for k, v in block.get("init_args", {}).items()
    }
    return cls(**kwargs)


def main(cfg_path: Path):
    cfg = yaml.safe_load(cfg_path.read_text())

    if "seed_everything" in cfg:
        L.seed_everything(cfg["seed_everything"], workers=True)

    dm = _build(cfg["data"])
    model = _build(cfg["model"])

    print("\nDatamodule summary")
    dm.setup()
    print(f"  train windows: {len(dm.train_ds)}")
    print(f"  val   windows: {len(dm.val_ds)}")
    print(f"  test  windows: {len(dm.test_ds)}")

    print("\nModel summary")
    print(L.pytorch.utilities.model_summary.ModelSummary(model, max_depth=2))

    return dm, model


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python init_from_cfg.py config.yaml")
    main(Path(sys.argv[1]))
