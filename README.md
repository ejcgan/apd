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
- Place your wandb information in a .env file. You can use the .env.example file as an example.
