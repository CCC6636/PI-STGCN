# PI-STGCN: a physics-inspired spatiotemporal graph convolutional network for hydrodynamic surrogate modelling

Reference implementation of **PI-STGCN**, the surrogate model described in the manuscript
*"A Hydrodynamic Surrogate Model Based on a Physics-Inspired Spatiotemporal Graph Convolutional Network"*
(Zeming Chen, Rui Bai, Jiaye Li*, Yongjie Li, Mengyao Wang, Wei Jin, Ling Yin), submitted to
*Environmental Modelling & Software*.

PI-STGCN learns the network-wide water-level response of a constrained, pumped-storage river reach from
boundary discharges, replacing minutes-long 2-D shallow-water simulation with second-level inference, while
keeping the spatial interactions interpretable through an explicitly physics-informed adjacency matrix.

---

## Software metadata

| Item | Value |
|---|---|
| Program name | PI-STGCN (files in this repository) |
| Description | Physics-inspired spatiotemporal graph convolutional network for long-sequence hydrodynamic surrogate prediction of water levels in a river-type lower reservoir |
| First release year | 2026 |
| Source code | Python implementation split across the modules listed below |
| Availability | Public GitHub repository: https://github.com/CCC6636/PI-STGCN |
| Copyright authors | Zeming Chen, Rui Bai, Jiaye Li |
| Contact email | lijiaye@dgut.edu.cn |
| Hardware requirements | Standard PC; optional NVIDIA GPU (CUDA) strongly recommended for training |
| Software requirements | Python, PyTorch, NumPy, pandas, scikit-learn, tqdm (see `requirements.txt`) |
| Tested environment | Python 3.13.5; PyTorch 2.10.0+cpu; NumPy 2.3.5; pandas 2.2.3; scikit-learn 1.8.0; tqdm 4.67.3 |
| Language | Python |
| Compiler | Not required (interpreted); CUDA used through PyTorch when available |
| Code executable on | Windows, Linux |
| Memory requirements | All conditions listed in `data_dir` are loaded into RAM before splitting; for the full 48,972-condition dataset, run one subset (or a sampled subset) at a time |
| Other machine requirements | None |
| Operating time | Reference simulation (MIKE 21): 5 min 51 s for 1 h, 18 min 37 s for 5 h. PI-STGCN inference: 0.56 s (1 h) and 2.75 s (5 h), average speed-up ≈ 462× (max 624×) on an Intel Core i9-10900 with 32 GB RAM |
| Publicly testable? | The model pipeline can be tested with the bundled synthetic inputs; the real simulation dataset is available from the corresponding author upon reasonable request (see *Data availability*) |

---

## 1. Problem and modelling idea

Hydrodynamic simulation of complex unsteady flow is computationally demanding, and existing data-driven
surrogates tend to (i) accumulate error over long autoregressive roll-outs and (ii) offer little mechanical
interpretability. PI-STGCN addresses both by

* embedding **explicit hydraulic priors** (directional river-network connectivity and discharge attenuation)
  into the graph construction instead of relying only on loss-level soft constraints;
* predicting **water-level increments** rather than absolute levels, which follows fluid inertia and suppresses
  drift in long sequences;
* coupling a **physics-data dual-driven adjacency matrix** with a state-initialised LSTM decoder, so that the
  learned node-to-node weights can be read back as hydraulic connectivity maps.

Design principles (manuscript §2.1): explicit incorporation of physical priors · incremental temporal
evolution · collaborative spatiotemporal representation.

---

## 2. Architecture and code map

Input per operating condition: initial water-level vector `Z_0 ∈ R^N` (N = 56 nodes) and the boundary-flow
sequence `Q ∈ R^{T×3}`. Output: predicted water-level sequence `Z_1 … Z_T` for all 56 nodes.

```
Q (T x 3) ──► FlowInfluenceModule ──► E (T x 56)          # physical response field, Eq. (1)
                                        │
                         E_1 (first prediction step) ──► FlowEffectEncoder ─┐
                                                                          ▼
Z_0 (56) ──► NodeWiseWaterEncoder ───────────────────────────────► concat X_0
                                                                          │
                                                                          ▼
                     SpatialGraphConv (A_couple) ──► H_spatial (56 x d), Eqs. (5)-(7)
                                                                          │
                                                                          ▼
              (h_0, c_0) = InitProj(MeanPool(H_spatial)) ──► LSTMDecoder
                    each step: [Z_(t-1), E_t] ──► ΔZ_t (Eqs. (8)-(13))
                                                   │
                                                   ▼
                              Z_t = Z_(t-1) + ΔZ_t (autoregressive rollout)
                                                   │
                                                   ▼
              ImprovedPhysicsAwareLoss = L_MAE + γ·L_smooth (no conservation residual)
```

| Manuscript section | Component | File |
|---|---|---|
| §2.2 Flow spatial distribution module, Eq. (1) | `FlowInfluenceModule` | `model_core.py` |
| §2.3 Dual-path state feature encoding, Eqs. (2)–(4) | `NodeWiseWaterEncoder`, `FlowEffectEncoder` | `model_core.py` |
| §2.4 Physics-data dual-driven graph convolution, Eqs. (5)–(7) | `SpatialGraphConv` | `model_core.py` |
| §2.5 Incremental autoregressive decoding, Eqs. (8)–(13) | `ImprovedWaterModel` (orchestration), `LSTMDecoder` | `recursive_model.py` |
| §2.6 Loss implementation | `ImprovedPhysicsAwareLoss` (absolute-level MAE + increment smoothness) | `recursive_model.py` |
| §3.2 / §3.3 data handling, global normalisation, variable-length batching | `data_utils.py` routines | `data_utils.py` |
| §4 autoregressive evaluation (MAE / RMSE / relative error) | `RecursiveInference` | `recursive_inference.py` |
| Optimiser, warm-up, plateau scheduling, checkpointing | `RecursiveTrainer` | `recursive_trainer.py` |
| End-to-end driver (config → load → split → train → test) | `main()` | `main.py` |

The active implementation is the non-commented code in the modules listed above. Some earlier experimental
implementations remain in source files as commented-out code; they are inactive and are not used by the
training or inference pipeline.

---

## 3. Study system and graph definition

The lower reservoir of the **Lianghekou hybrid pumped-storage power station** (Yalong River, Sichuan, China)
was discretised into **56 longitudinal nodes** over a ≈19.6 km narrow, natural river-type reservoir with
abrupt bed sills and a channel constriction between nodes 26–28.

Three flow boundaries are mapped onto the graph (`injection_nodes=[0, 9, 55]`, i.e. 1-based nodes 1, 10 and 56):

| Boundary | Physical meaning | Node (1-based) | Sign convention |
|---|---|---|---|
| `Q1` | Upper-reservoir outflow into the upstream boundary | 1 | generation positive |
| `Q2` | Regulating flow of the four reversible pump-turbine units (inlet/outlet structures) | 10 | pumping negative, generation positive |
| `Q3` | Lower-reservoir outflow at the downstream boundary | 56 | release positive |

The static influence matrix and physical topology matrices are built from the signed `N×N`
`section_distance.csv` matrix. Its raw entries use a magnitude scale of up to 15; the model
preprocessing divides the magnitudes by 15 before constructing the flow and GCN inputs:

The raw matrix uses **source nodes as rows and target nodes as columns**: `M[source, target]`.
The flow-distribution branch retains this layout and uses `|M| / 15` (with the boundary-direction
patches at nodes 1 and 56). For the GCN branch, positive and negative values are split into
downstream and upstream skeletons, respectively, scaled by `/15`, and then transposed to
**target rows / source columns** so that aggregation uses `A @ X`. The GCN diagonals are zeroed
before the self-connection is added inside `SpatialGraphConv`. The boundary-direction patches are
applied to both branches; see `main.py` and `get_directional_physics_matrices`.

---

## 4. Dataset

`48,972` independent operating conditions were generated by automated MIKE 21 (2-D shallow water) simulations
after engineering-feasibility filtering of the exhaustive combination grid:

| Mode | Conditions | Notes |
|---|---|---|
| Idle (non-operation) | 3,136 | `Q2 = 0` |
| Pumping | 8,064 | `Q2` from −625 to −75 m³/s |
| Generation | 37,772 | `Q2` from 75 to 625 m³/s (one-in-five subsampled before splitting) |
| **Total** | **48,972** | initial water level `H0` 2598.0–2605.0 m at 0.5 m steps (15 levels) |

**Expected file layout** (what `main.py` reads):

```
<data_dir>/                  # one CSV per operating condition
    condition_00001.csv      # rows = successive simulation output steps
    condition_00002.csv
    ...
<section_distance.csv>       # N x N signed influence matrix, node names as index/columns
```

Column positions are set in `main.py`: `node_indices='range:7-62'` (56 water-level columns, m) and
`q_indices='4,5,6'` (`Q1`, `Q2`, `Q3`, m³/s). Node names in the CSV header are matched against
`section_distance.csv`; if labels do not match, matrix loading stops with an error rather than substituting an identity prior.

### Smoke test without the MIKE 21 dataset

The repository layout assumed throughout this README is:

```text
pi-stgcn/
├── README.md
├── requirements.txt
├── code/
│   ├── main.py
│   ├── model_core.py
│   ├── recursive_model.py
│   ├── recursive_inference.py
│   ├── recursive_trainer.py
│   └── data_utils.py
├── outputs/                    created automatically for run artifacts
└── Sample/
    ├── README.md
    ├── data/
    ├── section_distance.csv
    ├── generate_sample_data.py
    └── output/                  placeholder directory (contains .gitkeep)
```

The driver configuration is defined inside `main()` in `code/main.py`. Its data and matrix paths are built
from the location of `main.py`, so the command below can be run from the repository root without relying
on the current working directory. The shipped `Sample/` folder contains 12 synthetic conditions, a matching
56 × 56 influence matrix, and a NumPy-only regenerator. These files reproduce the expected input schema but
come from a relaxation toy model; they verify pipeline execution and carry no hydrodynamic meaning.

The default configuration uses `hidden_dim=1024`, `batch_size=8`, `epochs=200`, and `random_seed=42`. For a short smoke test,
edit `hidden_dim`, `batch_size`, and `epochs` in the `config` dictionary inside `main()` in
`code/main.py`. A tested lightweight setting is `hidden_dim=32`, `batch_size=4`, and `epochs=1` (CPU). The configuration shown in this README is descriptive; defining a separate Python
`config` dictionary outside `main.py` does not override the driver's internal configuration.

See `Sample/README.md` for the CSV columns, matrix conventions, and sample-data regeneration instructions.

---

## 5. Normalisation

Global (not per-condition) constants are hard-coded in `data_utils.py` and must match the dataset:

| Constant | Value | Meaning |
|---|---|---|
| `WATER_GLOBAL_MIN` | 2581 | lower bound of water level (m) |
| `WATER_GLOBAL_MAX` | 2613 | upper bound of water level (m) |
| `WATER_GLOBAL_RANGE` | 32 | min-max range used for levels and increments |
| `FLOW_GLOBAL_SCALE` | 1700 | maximum absolute discharge (m³/s) |

Water levels are transformed by `(Z - WATER_GLOBAL_MIN) / WATER_GLOBAL_RANGE`; values within
2581–2613 m map to `[0, 1]`. Flows are divided by `FLOW_GLOBAL_SCALE` (1700 m³/s), preserving
their signs; values within ±1700 m³/s map to `[-1, 1]`. These transformations do not clip out-of-range
values. The training target is the **normalised increment** `ΔZ / WATER_GLOBAL_RANGE` relative to
the previous step (`compute_normalized_deltas_from_abs`). Inference accumulates in normalised space
and denormalises once at the end.

---

## 6. Installation

Clone the repository to a local folder named `pi-stgcn`, then run the environment setup commands
below from that folder:

```bash
git clone https://github.com/CCC6636/PI-STGCN.git pi-stgcn
cd pi-stgcn
python -m venv .venv
```

Activate the environment using the command for your shell:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows Command Prompt
.venv\Scripts\activate.bat
```

Then install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Training works on CPU but is slow; a CUDA device is recommended. The driver automatically selects
`cuda:0` when CUDA is available and otherwise uses the CPU. For GPU training, install a PyTorch build
that matches your CUDA environment by following the official PyTorch installation instructions. The paths for `data_dir`, `save_dir`, and
`influence_matrix_file` are defined inside `main()` in `code/main.py` using the script's absolute location;
edit those entries there when using a different dataset. Run the driver from the repository root as shown
in §7. The default configuration writes normalization statistics, test-condition data, checkpoints, and
evaluation-related artifacts under the repository-level `outputs/` directory, which the driver creates
if needed.

---

## 7. Training

```bash
python code/main.py
```

The default driver reads the included synthetic placeholder conditions from `Sample/data/`
and the signed 56 × 56 influence matrix from `Sample/section_distance.csv`. It writes
artifacts to the repository-level `outputs/` directory. The included demo is for pipeline
verification only; it is not the MIKE 21 training dataset and its metrics are not scientific
performance results.

For a quick smoke test, reduce `epochs`, `hidden_dim`, and `batch_size` in the
`config` dictionary inside `main()` in `code/main.py` before running. The default settings are
intended as a research configuration and may take substantial time.

Each run trains one model on the conditions found in `data_dir`. The driver first reserves 10%
of the conditions for testing (`test_size=0.1`), then splits the remaining 90% into training and
validation sets using `val_ratio=0.111` (approximately 10% of the full dataset). The resulting
overall split is approximately 80:10:10 (train / validation / test); for the 12 shipped sample
conditions, rounding gives 8 / 2 / 2. Both splits use `random_seed=42`. Artifacts written to `save_dir`:

| File | Content |
|---|---|
| `normalization_stats.pkl` | global level/flow normalisation statistics |
| `test_conditions.pkl` | held-out test conditions with their CSV names |
| `best_recursive_model.pth` | weights of the lowest validation loss |
| `recursive_checkpoint_epoch_*.pth` | periodic checkpoints (every 10 epochs) |
| `final_recursive_model.pth` | weights after the last epoch |

Key hyperparameters (defaults in `main.py` / `recursive_trainer.py`):

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| `node_num` | 56 | `hidden_dim` (`d`) | 1024 |
| `time_steps` (initial context) | 1 | `num_layers` (LSTM) | 2 |
| `batch_size` | 8 | `learning_rate` | 5e-4 |
| `epochs` | 200 | `weight_decay` (AdamW) | 1e-4 |
| `random_seed` | 42 | reproducibility seed | shared by Python / NumPy / PyTorch / splits / sampler |
| warm-up | `LinearLR`, 5 epochs, start factor 0.1 | plateau scheduler | `ReduceLROnPlateau`, patience 5, factor 0.5, min 1e-6 |
| gradient clipping | 0.5 | `delta_scale` (β) | learnable, initialised 0.15 |
| `w_up` / `w_down` | learnable, initialised 1.0 / 0.2 | `alpha` (residual amplitude) | learnable, initialised 0.1 |
| loss | absolute-water-level MAE + `smooth_weight` × increment smoothness | `smooth_weight` (γ) | 0.05 |

Training is purely autoregressive (no teacher forcing), matching the long-sequence setting evaluated in the
manuscript.

---

## 8. Inference and evaluation

```python
from pathlib import Path
import sys
import pickle
import glob
import numpy as np
import torch

# Run this example from the repository root: python <script_name>.py
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'code'))

from data_utils import extract_columns_by_indices
from main import load_raw_matrix, get_directional_physics_matrices
from recursive_model import ImprovedWaterModel
from recursive_inference import RecursiveInference


def build_flow_influence_matrix(raw_matrix):
    # Match the flow-branch preprocessing in main.py (source rows / target columns).
    matrix = np.abs(raw_matrix) / 15.0
    node1, node3 = 0, raw_matrix.shape[0] - 1
    for j in range(matrix.shape[1]):
        if raw_matrix[node1, j] < 0 and j != node1:
            matrix[node1, j] = 0.0
        if raw_matrix[node3, j] > 0 and j != node3:
            matrix[node3, j] = 0.0
    return matrix


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_dir = ROOT / 'Sample' / 'data'
matrix_file = ROOT / 'Sample' / 'section_distance.csv'
save_dir = ROOT / 'outputs'

csv_files = sorted(glob.glob(str(data_dir / '*.csv')))
if not csv_files:
    raise FileNotFoundError(f'No condition CSV files found in {data_dir}')
node_columns, _ = extract_columns_by_indices(csv_files[0], 'range:7-62', '4,5,6')
raw_matrix = load_raw_matrix(str(matrix_file), node_columns)
if raw_matrix is None:
    raise ValueError('Could not load the influence matrix or match its node labels.')
up_matrix, down_matrix = get_directional_physics_matrices(raw_matrix)
flow_matrix = build_flow_influence_matrix(raw_matrix)

checkpoint = torch.load(save_dir / 'best_recursive_model.pth', map_location=device)
state_dict = checkpoint['model_state_dict']
# Read the architecture dimensions from the saved weights so this also works
# when the smoke-test hidden_dim / num_layers differ from the defaults.
hidden_dim = state_dict['water_encoder.mlp.2.weight'].shape[0]
num_layers = sum(
    key.startswith('decoder.lstm.weight_ih_l') for key in state_dict
)

model = ImprovedWaterModel(
    node_num=56,
    time_steps=1,
    hidden_dim=hidden_dim,
    up_matrix=up_matrix,
    down_matrix=down_matrix,
    influence_matrix=flow_matrix,
    num_layers=num_layers,
).to(device)
model.load_state_dict(state_dict)
predictor = RecursiveInference(model, device)

# main.py stores each held-out condition as a dict containing `condition_data`.
with open(save_dir / 'test_conditions.pkl', 'rb') as f:
    saved_test_conditions = pickle.load(f)
test_conditions = [item['condition_data'] for item in saved_test_conditions]

report = predictor.evaluate_on_conditions(test_conditions, time_steps=1)
if report is not None:
    print(report['overall_rmse'], report['overall_mae'], report['overall_relative_error'])
```

This example reproduces the matrix preprocessing used by `main.py`, loads the best-validation checkpoint,
and evaluates the held-out conditions saved by that run. It assumes the checkpoint and `test_conditions.pkl`
were generated using the same data, matrix, and model configuration. The sample-data metrics are only a
pipeline smoke test and must not be interpreted as hydrodynamic validation results.

### Reading back the learned physics

```python
A_couple = model.spatial_gcn.get_current_adj()   # (56 x 56), softmax-normalised
```

For the GCN coupling matrix `A_couple`, rows are **target** nodes and columns are **source** nodes;
each entry is the learned influence of the source on the target. This is distinct from the raw
`section_distance.csv` convention (source rows, target columns) and from the flow-distribution
matrix, which retains the raw convention. Plotting `A_couple` gives the connectivity maps of manuscript Fig. 5, where the inlet/outlet
region (nodes 8–12) and the sill/constriction zone (nodes 21–28) dominate.

---

## 9. Paper-reported model variants and results (reference only)

The manuscript reports four experimental variants: PI-STGCN-Base, PI-STGCN-Pump,
PI-STGCN-Gen, and PI-STGCN-Mix. They use the same PI-STGCN architecture but differ
in the operating-condition datasets used for training and evaluation.

**The GitHub demo does not automate these four experiments.** The repository's
`main.py` runs one model on the dataset selected by its configuration. The included
`Sample/` data are synthetic placeholders and are not used to reproduce the manuscript's
numerical results. The results below are reported in the manuscript and must not be
interpreted as results obtained by running the included sample data.

| Variant | Manuscript training data | Manuscript reference |
|---|---|---|
| PI-STGCN-Base | Idle / non-operation conditions | §4.2.1 |
| PI-STGCN-Pump | Pumping conditions | §4.2.2 |
| PI-STGCN-Gen | Generation conditions | §4.2.3 |
| PI-STGCN-Mix | Mixed operating conditions | §4.2.4, §4.3–4.5 |

The manuscript reports a global CRMSD of 0.0125 m for PI-STGCN-Mix and a maximum
individual error of 0.0908 m at the critical node in the specified cross-condition
pumping evaluation. These are manuscript-reported results obtained using the original
MIKE 21 dataset, which is not included in this repository.

## 10. Data availability

The real training data are derived from MIKE 21 simulations of the Lianghekou lower reservoir and are
**not included in this repository**. The data supporting the manuscript findings are available from the
corresponding author upon reasonable request. Users who have access to compatible real data can place the
per-condition CSVs and `section_distance.csv` in the layout of §4 and update `data_dir` and
`influence_matrix_file` in the configuration inside `main()` to use their own data. The `Sample/` directory
contains only synthetic placeholder data generated for software testing and pipeline debugging. These
placeholders are not MIKE 21 results and must not be used to report or validate the scientific metrics in
the manuscript.

## 11. Implementation notes

- The physical influence matrix is required. If it cannot be loaded or its node labels do not
  match the 56 water-level columns, the driver stops with an error rather than silently
  substituting an identity matrix.
- The loss implemented in `ImprovedPhysicsAwareLoss` is absolute-water-level MAE plus a
  weighted temporal smoothness penalty on predicted increments. No explicit mass- or
  momentum-conservation residual is evaluated in the current implementation.
- Training and inference are autoregressive: future measured water levels are used only as
  training/validation targets or evaluation references, never as model inputs during the
  forecast rollout. The rollout uses the initial water level and prescribed future flows.

## 12. License

No software license is currently included. The source code remains subject to the applicable copyright
rights of its holders; public availability on GitHub does not by itself grant general permission to reuse,
modify, or redistribute the code. A formal license may be added after approval by the relevant copyright
holders and institution, and in accordance with any journal requirements.

## 13. Citation

```bibtex
@article{chen2026pistgcn,
  title  = {A Hydrodynamic Surrogate Model Based on a Physics-Inspired Spatiotemporal Graph Convolutional Network},
  author = {Chen, Zeming and Bai, Rui and Li, Jiaye and Li, Yongjie and Wang, Mengyao and Jin, Wei and Yin, Ling},
  journal= {Environmental Modelling \& Software},
  year   = {2026},
  note   = {In review; DOI to be added upon acceptance}
}
```

## 14. Acknowledgements

Supported by the National Key Research and Development Program of China (2023YFC3206700), the Research Project
of Power Construction Corporation of China, Ltd. (DJ-HXGG-2023-01), the Research Project of Power China
Chengdu, Ltd. (CDY-P67525), and the Guangdong Provincial Key Laboratory of Intelligent Disaster Prevention and
Emergency Technologies for Urban Lifeline Engineering (2026B1212210012).

## 15. Contact

Corresponding author: Jiaye Li — `lijiaye@dgut.edu.cn`

For questions about the implementation, please include the operating system, Python/PyTorch versions,
the exact command, and the complete error traceback. Do not send proprietary simulation data unless
sharing has been authorised.
