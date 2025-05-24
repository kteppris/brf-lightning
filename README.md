## ✨ What is BRF-Lightning?

> *High-level training utilities and example workflows for Balanced Resonate-and-Fire (BRF) neurons, implemented with PyTorch‑Lightning and powered by the refactored* **[`brf_neurons`](https://github.com/kteppris/brf-neurons)** *package.*

* **BRF-Lightning** is ***not* a re‑implementation of BRF neurons**.
  All low‑level ops and torch layers come from the separate MIT‑licensed library **`brf_neurons`** (a cleaned‑up fork of Adaptive AI Lab’s original code).
* This repo adds a **Lightning‑friendly “shell”** around those layers:

  * ready‑made `LightningModule` wrappers
  * `LightningDataModule`s for curated datasets
  * reproducible experiment configs & CLI entry points
  * notebooks and preprocessing scripts

Keeping these layers separate means:

| Layer                             | PyPI / Git dependency                                         | What it ships                                                    |
| --------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------- |
| `brf_neurons`                     | `pip install git+https://github.com/kteppris/brf-neurons.git` | neuron ops (`BRFCell`, `HRFCell`, …) + thin `nn.Module` wrappers |
| **`brf_lightning`** *(this repo)* | clone & `pip install -e .`                                    | Lightning modules, data modules, YAML configs, notebooks         |

---

## 🏁 Quick start

```bash
# clone experiments repository
git clone https://github.com/kteppris/brf-lightning.git
cd brf-lightning

# create a local env with uv (fast & deterministic)
uv venv --python 3.12
uv pip install -e .
```

`uv` resolves the GitHub dependency for **`brf_neurons`** automatically—no need for an extra `git clone`.

---

## 🗂 Repository structure

```
├── configs
│   ├── ecg
│   │   └── ecg_brf.yaml
│   ├── fit_base.yaml
│   └── override_base.yaml  # Optional, not committed
├── data
└── src
    └── brf_lightning
        ├── data            # Lightning Data Modules
        ├── defaults.py
        ├── models          # Lightning Modules
        ├── train.py
        └── utils
```

---

## 🚀 Run Training

### Basic invocation

Execute the training script with a YAML configuration file and an **experiment name** (required):

```bash
python src/brf_lightning/train.py \
       --config configs/ecg/ecg_brf.yaml \
       --experiment_name my_first_run 
```

* `--config` accepts one or more YAML files; later files **override** earlier ones.
* `--experiment_name` populates the folder name under `results/` and every logger’s *name* field, so that TensorBoard, CSV logs, checkpoints and any extra loggers all land in the same directory. Each yaml config also has a **default experiment** name.

### Inspecting & overriding parameters

* Display the CLI help and a list of *all* configurable flags:

  ```bash
  python src/brf_lightning/train.py fit --help
  ```
* Print the **complete, merged config** (including defaults from every Lightning class) without running any training:

  ```bash
  python src/brf_lightning/train.py fit --print_config
  ```

  The output is a fully‑resolved YAML file you can pipe to disk and edit.  ([lightning.ai](https://lightning.ai/docs/pytorch/stable//cli/lightning_cli_advanced.html?utm_source=chatgpt.com))
* Override individual parameters on the command line using dot notation:

  ```bash
  # change learning rate and number of epochs on the fly
  python src/brf_lightning/train.py \
         --config configs/ecg/ecg_brf.yaml \
         trainer.max_epochs=20 \
         model.optimizer.lr=1e-3
  ```

### What ends up in `results/`

The training script creates a deterministic directory per experiment:

```
results/<experiment_name>/version_<n>/
├── checkpoints/        # *.ckpt files saved by ModelCheckpoint
├── metrics.csv         # CSVLogger
├── events.out...       # TensorBoard logs
├── profiler_output.txt # Profiler results, duration of each component 
└── config.yaml         # Complete config used
```

* `version_<n>` is auto‑incremented; all loggers share **one** value because we patch their `version`/`run_name` when they are `None`.  TensorBoard and CSV accept an explicit `version` argument, preventing them from creating separate folders. ([lightning.ai](https://lightning.ai/docs/pytorch/stable//extensions/generated/lightning.pytorch.loggers.TensorBoardLogger.html?utm_source=chatgpt.com), [lightning.ai](https://lightning.ai/docs/pytorch/stable//extensions/generated/lightning.pytorch.loggers.CSVLogger.html?utm_source=chatgpt.com))
* Checkpoints follow the same path via `Trainer(default_root_dir=…)`. ([lightning.ai](https://lightning.ai/docs/pytorch/stable//common/trainer.html?utm_source=chatgpt.com))
* `MLFlowLogger` maps the same string to `run_name` so that the run inside the MLflow UI matches the directory on disk. ([lightning.ai](https://lightning.ai/docs/pytorch/stable//extensions/generated/lightning.pytorch.loggers.MLFlowLogger.html?utm_source=chatgpt.com))

### Resuming / continuing runs

If you rerun the exact same command, Lightning will detect that `version_<n>` already exists and create `version_<n+1>`. To **resume** from the latest checkpoint instead, add:

```bash
--ckpt_path results/<experiment_name>/version_<n>/checkpoints/last.ckpt
```

---

## 🔬 Publications & citation

If you build on this work, please cite the pioneering BRF papers by Adaptive AI Lab:

```bibtex
@misc{higuchi2024balanced,
  title  = {Balanced Resonate-and-Fire Neurons},
  author = {Saya Higuchi and Sebastian Kairat and Sander M. Bohte and Sebastian Otte},
  year   = {2024},
  eprint = {2402.14603},
  archivePrefix = {arXiv},
  primaryClass  = {cs.NE}
}

@misc{higuchi2024understanding,
  title  = {Understanding the Convergence in Balanced Resonate-and-Fire Neurons},
  author = {Saya Higuchi and Sander M. Bohte and Sebastian Otte},
  year   = {2024},
  eprint = {2406.00389},
  archivePrefix = {arXiv},
  primaryClass  = {cs.NE}
}
```

---

## ⚖️ License & attribution

```
MIT License

Copyright (c) 2024 Adaptive AI Lab
Copyright (c) 2025 Keno Teppris
```

* Core neuron code originates from **Adaptive AI Lab’s [brf-neurons](https://github.com/AdaptiveAILab/brf-neurons)** (MIT).
* That code was refactored into an installable package—**[`kteppris/brf-neurons`](https://github.com/kteppris/brf-neurons)**—and is **imported** (not copied) here.
* All additional training utilities, configs and scripts are © 2025 Keno Teppris under the same MIT terms.

---

## 🙏 Acknowledgements

* Adaptive AI Lab for open‑sourcing the original BRF implementation and datasets.
* The PyTorch‑Lightning community for the training‑loop scaffolding.
* Contributors and users who provide feedback and improvements—issues and PRs are welcome!
