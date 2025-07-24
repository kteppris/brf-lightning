from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary
from lightning.pytorch.profilers import PyTorchProfiler
from jsonargparse import lazy_instance

TRAINER_DEFAULTS = {
    "callbacks": [
        lazy_instance(RichProgressBar), 
        lazy_instance(RichModelSummary, max_depth=3)
],
    "max_epochs": 400,
    # "profiler": {
    #     "class_path": "lightning.pytorch.profilers.PyTorchProfiler", 
    #     "init_args": {
    #         "filename": "profiler_output"
    #     }
    # }
}

CONFIG_DEFAULTS = {
    "fit": {"default_config_files": ["configs/fit_base.yaml", "configs/override_base.yaml"]},
    "parser_mode": "omegaconf"
}