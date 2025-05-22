import logging
from dotenv import load_dotenv

from brf_lightning.models import *
from brf_lightning.data import *
from brf_lightning.defaults import TRAINER_DEFAULTS, CONFIG_DEFAULTS
from brf_lightning.utils.cli import BRFLightningCLI

logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

load_dotenv()

if __name__ == "__main__":
    BRFLightningCLI(
        trainer_defaults=TRAINER_DEFAULTS,
        parser_kwargs=CONFIG_DEFAULTS,
        auto_configure_optimizers=False
    )
