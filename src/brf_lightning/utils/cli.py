from lightning.pytorch.cli import LightningCLI

from datetime import datetime
from pathlib import Path
import re

DEFAULT_NAME_SENTINELS = {"lightning_logs", None}
DEFAULT_VERSION_SENTINEL = None

def next_numeric_version(root: Path) -> str:
    pat = re.compile(r"version_(\d+)$")
    nums = [int(m.group(1)) for p in root.glob("version_*")
                             if (m := pat.match(p.name))]
    return f"version_{max(nums) + 1 if nums else 0}"

def make_version(root: Path, scheme: str = "auto") -> str:
    if scheme == "timestamp":
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    return next_numeric_version(root) 

class BRFLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--experiment_name", type=str, required=True)
        parser.add_argument("--version_scheme",
                            choices=("auto", "timestamp"), default="auto")

    def before_instantiate_classes(self) -> None:
        cfg = self.config.fit
        exp_name = cfg.experiment_name

        root = Path("results") / exp_name
        version_str = make_version(root, cfg.version_scheme)

        # patch every logger
        for lg in cfg.trainer.logger:
            args = lg.init_args

            # 1) ensure folder name is experiment_name
            for field in ("name", "experiment_name"):
                if hasattr(args, field) and getattr(args, field) in DEFAULT_NAME_SENTINELS:
                    setattr(args, field, exp_name)

            # 2) unify version / run_name when user left them blank
            if hasattr(args, "version") and args.version is DEFAULT_VERSION_SENTINEL:
                args.version = version_str
            if hasattr(args, "run_name") and args.run_name is DEFAULT_VERSION_SENTINEL:
                args.run_name = version_str

            # ensure every logger has a save_dir; fall back to results/
            if hasattr(args, "save_dir") and args.save_dir is None:
                args.save_dir = "results"