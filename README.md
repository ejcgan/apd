# APD - Attribution-based Parameter Decomposition
Code used in the paper [Interpretability in Parameter Space: Minimizing
Mechanistic Description Length with
Attribution-based Parameter Decomposition](https://publications.apolloresearch.ai/apd)

Note: previously called Sparse Parameter Decomposition (SPD). The package name will remain as `spd`
for now, but the repository has been renamed to `apd`.

## Installation
From the root of the repository, run one of

```bash
make install-dev  # To install the package, dev requirements and pre-commit hooks
make install  # To just install the package (runs `pip install -e .`)
```

## Usage
Place your wandb information in a .env file. You can use the .env.example file as an example.

The repository consists of several `experiments`, each of which containing scripts to train target
models and run APD.
- `spd/experiments/tms` - Toy model of superposition
- `spd/experiments/resid_mlp` - Toy model of compressed computation and toy model of distributed
  representations
- `spd/experiments/piecewise` - Handcoded gated function model
- `spd/experiments/linear` - Deep linear network without non-linearities (not in paper)

### Train a target model
All experiments apart from `spd/experiments/piecewise` require training a target model. Look for the
`train_*.py` script in the experiment directory. Your trained model will be saved locally and
uploaded to wandb.

### Run APD
APD can be run by executing any of the `*_decomposition.py` scripts defined in the experiment
subdirectories. A config file is required for each experiment, which can be found in the same
directory. For example:
```bash
python spd/experiments/tms/tms_decomposition.py spd/experiments/tms/tms_topk_rank_penalty_config.yaml
```
will run SPD on TMS with the config file `tms_topk_rank_penalty_config.yaml` (which is the main
config file used for the TMS experiments in the paper).

Wandb sweep files are also provided in the experiment subdirectories, and can be run with e.g.:
```bash
wandb sweep spd/experiments/tms/tms_sweep_config.yaml
```

All experiments call the `optimize` function in `spd/run_spd.py`, which contains the main APD logic.

### Analyze results
Experiments contain `*_interp.py` scripts which generate the plots used in the paper.

## Development

Suggested extensions and settings for VSCode/Cursor are provided in `.vscode/`. To use the suggested
settings, copy `.vscode/settings-example.json` to `.vscode/settings.json`.

There are various `make` commands that may be helpful

```bash
make check  # Run pre-commit on all files (i.e. pyright, ruff linter, and ruff formatter)
make type  # Run pyright on all files
make format  # Run ruff linter and formatter on all files
make test  # Run tests that aren't marked `slow`
make test-all  # Run all tests
```