# ParetoPhys-PV

Compact day-ahead photovoltaic forecasting with bounded residual models and multi-objective model selection.

The package learns corrections to an available 24-hour reference forecast. It combines historical power, issue-time features, and a state-conditioned residual adapter. Model selection balances average daytime error, the upper tail of daily errors, and computation per forecast.

## Features

- DLinear, temporal convolution, and compact patch Transformer backbones.
- A bounded residual adapter, daylight masking, and output clipping.
- A discrete model configuration space with deterministic feasibility repair.
- NSGA-II and repaired random search, with configuration caching within each run.
- Pareto-front coverage screening and accuracy, balanced, and efficiency selection rules.
- Group-balanced daytime nMAE, daily-error CVaR, parameter counts, and operation counts.
- Training-only feature scaling, checkpoint inference, and tests for time alignment and feature leakage.

## Installation

Use Python 3.11 or newer. CPU, CUDA, and Apple MPS devices are supported through PyTorch.

```bash
git clone https://github.com/rogel/paretophys-pv.git
cd paretophys-pv
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows, activate the environment with `.venv\Scripts\activate`. To calculate solar descriptors from coordinates, install the optional dependency with `python -m pip install -e ".[solar]"`.

## Quick start

Generate a small toy dataset locally, train a model, and predict its held-out period:

```bash
paretophys demo-data --output data/demo.npz
paretophys train --data data/demo.npz --config configs/compact.yaml \
  --output outputs/compact --epochs 3 --device cpu
paretophys predict --data data/demo.npz \
  --checkpoint outputs/compact/checkpoint.pt \
  --split test --output outputs/test_predictions.csv --device cpu
```

The generated records are for checking the software workflow. They are not observations or a performance benchmark. Datasets, trained weights, and generated outputs are not included in this repository.

Commands can also be run as `python -m paretophys_pv <command>`. Existing checkpoints and result files are not overwritten; choose a new output location for another run.

## Model search

For a short workflow check:

```bash
paretophys search --data data/demo.npz --output outputs/search \
  --population 4 --generations 2 --epochs 2 --device cpu
paretophys select --input outputs/search/evaluations.json \
  --mode screen --count 8 --output outputs/screened
```

The normal defaults are 16 candidates per generation, 8 generations, and up to 12 training epochs per candidate. `--method random` uses repaired independent sampling under the same request budget. Duplicate configurations reuse their evaluation within a run. The supplied bundle determines the training and validation data; the search command does not create a smaller proxy subset automatically.

Search produces evaluation records, a non-dominated front, per-request records, and checkpoints. Coverage screening exports YAML configurations for longer training:

```bash
paretophys train --data data/demo.npz \
  --config outputs/screened/candidate_01.yaml \
  --output outputs/retrained_01 --epochs 40 --device cpu
```

After training all screened configurations, collect their `metrics.json` records into a JSON array. If a configuration has multiple training seeds, average its objectives first. Assign final preferences with:

```bash
paretophys select --input outputs/full_evaluations.json \
  --mode roles --output outputs/selected
```

`accuracy` minimizes validation nMAE. `balanced` minimizes distance to the normalized three-objective ideal. `efficiency` minimizes MACs among configurations within 2% relative nMAE of the accuracy choice. A configuration may serve more than one role. Final roles should use full-training validation records, with the test period reserved for subsequent assessment.

## Bring your own data

Prepare an hourly CSV following [the input schema](docs/data.md), then run:

```bash
paretophys prepare --input data/hourly.csv --output data/sequences.npz
```

Each sample contains 168 historical hours and a 24-hour horizon. Future features are standardized using training rows only. Power and reference curves remain in per-unit capacity. Validation and test targets never enter the loss used for parameter updates.

Data sources:

- [NLR Solar Power Data for Integration Studies](https://www.nlr.gov/grid/solar-power-data)
- [GEFCom2014 publisher-hosted archive](https://ars.els-cdn.com/content/image/1-s2.0-S0169207016000133-mmc1.zip)

See [data preparation and access details](docs/data.md) for timestamp conventions, dataset characteristics, and download routes.

## Package layout

```text
configs/compact.yaml       Example model configuration
src/paretophys_pv/
  data.py                  Canonical hourly input and sequence preparation
  features.py              Origin-anchored feature construction
  time_alignment.py        Hourly interval-center timestamps
  nlr_data.py               Archive naming and deterministic site selection
  models.py                Reference-conditioned neural baselines
  search_space.py          Candidate models, adapter, loss, repair, MAC counts
  evolution.py             Dominance, sorting, crowding, variation, selection
  training.py              Training and checkpoint prediction
  search.py                Search orchestration and within-run caching
  selection.py             Coverage screening and preference rules
  metrics.py               Forecast error metrics
  cli.py                   Command-line interface
tests/                     Numerical, model, data, and selection checks
```

## Tests

```bash
python -m pytest -q
```

See [model and evaluation conventions](docs/design.md) for the residual formulation, feature groups, and metric definitions.
