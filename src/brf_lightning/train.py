import logging
from lightning.pytorch.cli import LightningCLI

from brf_lightning.models import *
from brf_lightning.data import *
from brf_lightning.defaults import TRAINER_DEFAULTS, CONFIG_DEFAULTS

logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

class ECGLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # 1) globale Variable
        parser.add_argument("--experiment_name", type=str, required=True)

if __name__ == "__main__":
    ECGLightningCLI(
        seed_everything_default=42,
        trainer_defaults=TRAINER_DEFAULTS,
        parser_kwargs=CONFIG_DEFAULTS,
        auto_configure_optimizers=False
    )
