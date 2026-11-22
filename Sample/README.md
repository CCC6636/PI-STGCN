# PI-STGCN sample inputs (synthetic)

Everything in this folder is a **synthetic placeholder** whose only purpose is to let the code in
[`../code`](../code) run end to end without the MIKE 21 dataset. The files reproduce the exact
input structure that `main.py` expects; the numbers themselves come from a smooth relaxation toy
model, **not** from a hydrodynamic simulation. Losses, MAE, RMSE or speed-ups computed on these
files have no physical meaning and must never be reported as model performance.

The real forcing/response dataset (48,972 MIKE 21 conditions for the Lianghekou hybrid
pumped-storage lower reservoir) is not redistributed here — see *Data availability* in the
repository README.

## Contents

```
Sample/
├── data/                        12 per-condition CSVs, 180 time steps each (~1.4 MB total)
│   ├── condition_idle_01.csv    Q2 = 0          (idle / natural passage)
│   ├── condition_pump_02.csv    Q2 < 0          (pumping)
│   ├── condition_gen_03.csv     Q2 > 0          (generation)
│   └── ...                      4 idle + 4 pumping + 4 generation
├── section_distance.csv         56 x 56 signed influence matrix
├── output/                      placeholder output directory for the sample config (contains .gitkeep)
├── generate_sample_data.py      re-generator (numpy only)
└── README.md
```

## Condition CSV layout

Each row is one output time step (`dt = 60 s`, `Time_s` = elapsed seconds); every condition file
has the same 63-column header. Column positions below are 0-based and match
`config['q_indices'] = '4,5,6'` and `config['node_indices'] = 'range:7-62'`.

| Position | Name | Meaning |
| --- | --- | --- |
| 0 | `Time_s` | time since the start of the condition, 60 s apart (180 steps = 3 h) |
| 1 | `Condition_ID` | file stem, kept for traceability |
| 2 | `H0_m` | initial water level at node 1 (m) |
| 3 | `Case_Type` | `idle` / `pump` / `gen` |
| 4 | `Q1_m3s` | upstream boundary flow, injected at node 1 |
| 5 | `Q2_m3s` | pump-turbine flow at node 10 (negative = pumping) |
| 6 | `Q3_m3s` | downstream boundary flow, injected at node 56 |
| 7 … 62 | `Node_01` … `Node_56` | absolute water level (m) at each longitudinal node |

The three boundary series vary in time rather than sitting at a constant set point: `Q2` ramps from
zero toward its condition-specific signed target using a 20 min exponential time scale (the shipped
files reach absolute values up to about 554 m³/s), while `Q1` and `Q3` vary sinusoidally by ±8 %
around their condition-specific base flows. In the shipped files, `Q1` ranges from about 118 to
380 m³/s and `Q3` from about 159 to 339 m³/s. Each node relaxes
towards the steady state of the instantaneous forcing with a distance-dependent lag, plus a damped
sub-daily oscillation and 2 mm of noise.

Ranges are deliberately kept inside the constants hard-coded in `code/data_utils.py`: levels
2598.6–2604.3 m (bounds 2581–2613 m) and max \|flow\| 554 m³/s (scale 1700 m³/s), so normalised water levels fall in [0, 1], signed flows fall in [-1, 1], and the per-step increments (largest observed: 9.7e-4 normalised) stay well
inside the `Tanh` output range of the decoder head.

## Influence matrix conventions

`section_distance.csv` is stored in **source-row / target-column** form:
`raw[source, target]` describes the signed influence from a source node to a target node.
The two model branches intentionally derive different layouts from this raw matrix:

- **Flow distribution (`FlowInfluenceModule`)** keeps the source-row / target-column layout.
  It selects the rows for injection nodes 1, 10 and 56 to build the three flow-distribution
  vectors over all 56 target nodes.
- **GCN physical adjacency (`SpatialGraphConv`)** splits positive and negative directional
  values and then transposes each matrix to target-row / source-column form. This is required
  by the graph aggregation `A @ X`, so each target node receives features from source nodes.

Do not transpose the raw matrix before both branches are built. The flow and GCN branches have
separate matrix conventions by design. The real research `section_distance.csv` uses the same
source-row / target-column convention as this sample matrix.

Positive values encode downstream influence and negative values encode upstream influence.
The matrix magnitude is scaled by `/15.0`; the diagonal is zero. The labels are written as
`CS_07 (Node_07)` so the parenthetical node names match the condition CSV headers.
A missing or mismatched matrix is treated as an error by the driver; it is not silently replaced
with an identity matrix.

## Running the code on the sample inputs

The sample data are already selected by the default configuration in `code/main.py`. The paths are
constructed relative to the location of `main.py`, not relative to the shell's current working directory.
Run the driver from the repository root:

```bash
python code/main.py
```

The configuration is defined inside `main()` in `code/main.py`. For a short smoke test, edit the
following entries in that in-file `config` dictionary before running:

```python
'hidden_dim': 32,    # default: 1024
'batch_size': 4,     # default: 8
'epochs': 1,         # default: 200
```

This lightweight CPU setting was used to verify the end-to-end sample pipeline. It is only a
software smoke test, not a recommended research-training configuration.

Do not create a separate `config` dictionary in a notebook or another script and expect `main.py` to
read it: the driver uses its own local configuration. By default, model checkpoints and other run
artifacts are saved under the repository-level `outputs/` directory. The `Sample/output/` directory
is included as a placeholder for users who deliberately change `save_dir` to that location; the
training driver creates the configured output directory if it does not exist.

The model is trained autoregressively over the full sequence length. The driver first reserves 10%
of the conditions for testing, then sets aside approximately 10% of the full dataset for validation
from the remaining conditions (overall split approximately 80:10:10). With the 12 shipped files,
rounding gives 8 training, 2 validation, and 2 test conditions. The default driver uses `random_seed=42`
to control Python, NumPy, PyTorch parameter initialisation, data splitting, and training-sample order.
Repeated runs on the same software/hardware stack should therefore follow the same stochastic setup;
exact bitwise identity across different hardware or software versions is not guaranteed. The smaller smoke-test settings reduce training time and are
intended only to verify execution, not to reproduce the research runs.

### Expected console output

With the 12 shipped sample conditions, the initial output should contain messages similar to these
(the device, paths, parameter count, and training logs can vary):

```text
使用设备: cuda:0
找到 12 个CSV文件
节点数量: 56
流量列名: ['Q1_m3s', 'Q2_m3s', 'Q3_m3s']
归一化stats已保存: ...
Test条件已保存: ...
成功加载影响矩阵，形状: (56, 56)
原始矩阵范围: [-15.000, 15.000]
正影响数量: 1540, 负影响数量: 1540
流量影响矩阵（源行/目标列）和 GCN 方向矩阵（目标行/源列）生成成功！
训练工况: 8, 验证工况: 2
数据集划分: Test=2
```

For the shipped 12 conditions, the split is 8 training, 2 validation, and 2 test conditions.
The printed messages are in Chinese because that is what the current Python implementation emits.

## Regenerating or resizing

```bash
cd Sample
python generate_sample_data.py --conditions 24 --steps 120 --seed 7
```

- `--conditions` number of CSVs; keep at least a handful so all three splits get a member
  (12 gives the 8 train / 2 validation / 2 test shown above);
- `--steps` time steps per condition, 60 s apart (must exceed `config['time_steps'] + 1`);
- `--seed` drives `numpy.random.default_rng`, so the same seed and generation arguments produce
  deterministic sample contents.

The generator needs only NumPy and the standard library. Delete `data/*.csv` first if you
generate fewer conditions than before, since stale files would still be picked up by the glob.
