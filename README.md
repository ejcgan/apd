# spd - Sparse Parameter Decomposition

## Installation

From the root of the repository, run one of

```bash
make install-dev  # To install the package, dev requirements and pre-commit hooks
make install  # To just install the package (runs `pip install -e .`)
```

To use the graph viz, install it with your package manager. E.g. `apt install graphviz`

## Development

Suggested extensions and settings for VSCode are provided in `.vscode/`. To use the suggested
settings, copy `.vscode/settings-example.json` to `.vscode/settings.json`.

There are various `make` commands that may be helpful

```bash
make check  # Run pre-commit on all files (i.e. pyright, ruff linter, and ruff formatter)
make type  # Run pyright on all files
make format  # Run ruff linter and formatter on all files
make test  # Run tests that aren't marked `slow`
make test-all  # Run all tests
```

## Usage
Place your wandb information in a .env file. You can use the .env.example file as an example.

### Training a model
Some experiments allow for (and sometimes require) pretraining a model without SPD in order to be
later used in SPD loss terms (e.g. `spd/experiments/linear`). Simply run the `train_*.py` script
in the relevant experiment directories, and use the path of the trained model in your SPD config
file.

### Running SPD
The SPD algorithm can be run by executing any of the `*_decomposition.py` experiments defined in the
subdirectories of `spd/experiments/`. A config file is required for each experiment, which can also
be found in the experiment subdirectories. For example:
```bash
python spd/experiments/tms/tms_decomposition.py spd/experiments/tms/tms_topk_match_param_config.yaml
```
will run SPD on TMS with the config file `tms_topk_match_param_config.yaml`.

Wandb sweep files are also provided in the experiment subdirectories, and can be run with e.g.:
```bash
wandb sweep spd/experiments/tms/tms_topk_match_param_sweep.yaml
```

All experiments call the `optimize` function in `spd/run_spd.py`, which contains the main SPD logic.

**Note, as of writing, the bool_circuits experiment is not currently supported (we had conceptual
issues trying to create a suitable boolean circuit dataset).**
