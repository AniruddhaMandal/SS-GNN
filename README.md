# SS-GNN

> 🧠 **SS-GNN** — A flexible subgraph-based GNN training framework with a fast **uniform subgraph sampler** for scalable, reproducible graph learning experiments. 

---

## 📖 About

**SS-GNN** is an experimental research framework built for training Graph Neural Networks (GNNs) using **subgraph sampling** and **vanilla architectures**.
It is designed to make running, comparing, and extending GNN experiments simple and reproducible — particularly for graph classification and regression tasks.
It includes a uniform subgraph sampler(`src/ugs_sampler`) due to [Bressan M.](https://arxiv.org/abs/2007.12102)

This project will accompany a **research publication** (📄 *details to be added later*), with plans to include detailed **explanations**, **demonstrations**, and **experimental results** as the work progresses.

---

## ✨ Key Features

* ⚡ **Subgraph Sampling** — Efficient, scalable training via subgraph mini-batching.
* 🧱 **Multiple GNN Architectures** — Vanilla GNNs and subgraph-based variants.
* 🧪 **Task Flexibility** — Supports:

  * Multi-Label Binary Classification
  * Multi-Class Classification
  * Binary Classification
  * Multi-Target Regression
* 🪄 **Model Registry System** — Easily add new models with minimal changes.
* 🧬 **Reproducibility** — Deterministic seeds and standardized experiment structure.
* 🧰 **Extensible for Research** — Clean codebase structured for experiments and papers.
* 📊 **Integrated Logging** — Works with [TensorBoard] and standard metrics.

---

## 🧰 Core Dependencies

* [PyTorch](https://pytorch.org/)
* [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
* [scikit-learn](https://scikit-learn.org/)
* [NumPy](https://numpy.org/)
* `pytest`, `tensorboard`
* `ugs_sampler` *(custom C++ subgraph sampling extension)*

---

## 🛠️ Installation

```bash
# System dependencies
sudo apt install python3.12 python3.12-dev
sudo apt install build-essential

# Virtual environment
virtualenv -p python3.12 venv
source venv/bin/activate

# Python packages
pip install pytest numpy torch torch_geometric scikit-learn tensorboard

# Build and install subgraph sampler
pip install -e src/ugs_sampler --no-build-isolation

# Install gps package with GNN extras
pip install -e src/gps[gnn]
```

---

## 🚀 Running Experiments

Using CLI (recommended):

```bash
gps-run -c configs/ss_gnn/TUDataset/gcn-mutag.json
```

Or directly via Python:

```bash
python main.py --config configs/ss_gnn/TUDataset/gcn-mutag.json
```

With multiple seeds:

```bash
gps-run -c configs/ss_gnn/TUDataset/gcn-mutag.json -m --seeds 42 10 32 29 75
```

Override config in CLI:
```bash
gps-run -c configs/ss_gnn/TUDataset/gcn-mutag.json -o train.epochs=50 model_config.hidden_dim=128
```

📊 The framework automatically averages results and reports mean ± std across seeds and saves it in `experiment_results/'exp_config.name'.txt`

---

## 🧠 Model Registration

1. Add your model in

   ```
   src/gps/gps/model/
   ```
2. Register it in

   ```
   src/gps/gps/model.py
   ```
3. Your model must **return logits only**.

```python
from gps.registry import register_model

@register_model("my_gnn")
class MyGNN(torch.nn.Module):
    ...
    def forward(self, data):
        return logits
```

---

## 🧪 Testing

```bash
pytest -q --config path/to/config-file.json
```

---

## 🧭 Reproducibility & Research

SS-GNN is designed to ensure:

* Consistent data splits and seeds across runs
* Unified experiment tracking
* Config-based control of all hyperparameters and model choices

Future updates will include:

* 📊 Result tables and figures
* 📚 Example experiments from the research paper
* 📘 Detailed explanations and ablation studies

---

## 🏗️ Project Structure

```
SS-GNN/
├── configs/                     # Experiment configs
├── main.py                      # Experiment runner
├── notebooks/                   # Experiment notebooks   
├── LICENSE
├── src/
│   ├── gps/                     # Main Python package
│   │   ├── gps/                
│   │   │   ├── model/           # GNN architectures
│   │   │   ├── utils/           
│   │   │   ├── experiment.py    # Experiment class
│   │   │   └── registry.py      # Model registry
│   │   └── setup.py
│   └── ugs_sampler/             # C++ subgraph sampler
├── tests/                       # Tests for modules
├── tools/                       # Tools for visualizing graphs in 3d
└── README.md
```

---

## 📜 Citation *(placeholder)*

If you use **SS-GNN** in your research, please cite:

```
@inproceedings{YourName2025,
  title     = {SS-GNN: A Flexible Subgraph-based GNN Training Framework},
  author    = {Your Name and Others},
  booktitle = {Conference TBD},
  year      = {2025}
}
```

---

## 🧑‍💻 Contributing

This repository is under active research and development.
If you’d like to contribute (e.g., new sampling strategies, model architectures, or benchmarks), feel free to open an issue or pull request.

---

## 📜 License

Licensed under the [MIT License](./LICENSE).

---

## 📬 Contact

Maintainer: **Aniruddha Mandal**\
Email: `ani96dh@gmail.com` \
GitHub: [Aniruddha Mandal](https://github.com/AniruddhaMandal)