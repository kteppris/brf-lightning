
## ✨ What is BRF-Lightning?

> *High-level training utilities and example workflows for Balanced Resonate-and-Fire (BRF) neurons, implemented with PyTorch-Lightning and powered by the refactored* **[`brf_neurons`](https://github.com/kteppris/brf-neurons)** *package.*

* **BRF-Lightning** is ***not* a re-implementation of BRF neurons**.
  All low-level ops and torch layers come from the separate MIT-licensed library **`brf_neurons`** (a cleaned-up fork of Adaptive AI Lab’s original code).
* This repo adds a **Lightning-friendly “shell”** around those layers:

  * ready-made `LightningModule` wrappers
  * `LightningDataModule`s for curated datasets
  * reproducible experiment configs & CLI entry points
  * notebooks and preprocessing scripts

Keeping these layers separate means:

| Layer                             | PyPI / Git dependency      | What it ships                                                    |
| --------------------------------- | -------------------------- | ---------------------------------------------------------------- |
| `brf_neurons`                     | `pip install https://github.com/kteppris/brf-neurons.git`  | neuron ops (`BRFCell`, `HRFCell`, …) + thin `nn.Module` wrappers |
| **`brf_lightning`** *(this repo)* | clone & `pip install -e .` | Lightning modules, data modules, YAML configs, notebooks         |

---

## 🏁 Quick start

```bash
# clone experiments repository
git clone https://github.com/kteppris/brf-lightning.git
cd brf-lightning

# create a local env with uv (fast & deterministic)
uv venv --python 3.12
```

`uv` resolves the GitHub dependency for **`brf_neurons`** automatically—no need for an extra `git clone`.

---

## 🗂 Repository structure

```
brf-lightning/
├── train.py                  # single entry point
├── configs/
│   ├── ecg/
│   │   ├── brf_small.yaml
│   │   └── brf_large.yaml
│   └── shd/
│       └── rf_baseline.yaml
└── src/brf_lightning/
    ├── data/
    │   ├── ecg.py
    │   └── shd.py
    └── models/
        ├── ecg_brf.py
        └── shd_rfr.py
```

---

## 🔬 Publications & citation

If you build on this work, please cite the pioneering BRF papers by Adaptive AI Lab:

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

* Core neuron code originates from **Adaptive AI Lab’s [brf-neurons](https://github.com/AdaptiveAILab/brf-neurons)** (MIT).
* That code was refactored into an installable package—**[`kteppris/brf-neurons`](https://github.com/kteppris/brf-neurons)**—and is **imported** (not copied) here.
* All additional training utilities, configs and scripts are © 2025 Keno Teppris under the same MIT terms.

---

## 🙏 Acknowledgements

* Adaptive AI Lab for open-sourcing the original BRF implementation and datasets.
* The PyTorch-Lightning community for the training-loop scaffolding.
* Contributors and users who provide feedback and improvements—issues and PRs are welcome!
