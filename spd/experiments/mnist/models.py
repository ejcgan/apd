"""MNIST models for SPD experiments."""

from pathlib import Path
from typing import Any, Literal

import torch
import wandb
import yaml
from jaxtyping import Bool, Float
from pydantic import BaseModel, ConfigDict, PositiveInt
from torch import Tensor, nn
from torch.nn import functional as F
from wandb.apis.public import Run

from spd.hooks import HookedRootModule
from spd.models.base import SPDModel
from spd.models.components import (
    LinearComponent,
)
from spd.run_spd import Config, MNISTTaskConfig
from spd.types import WANDB_PATH_PREFIX, ModelPath
from spd.wandb_utils import download_wandb_file, fetch_latest_wandb_checkpoint, fetch_wandb_run_dir


class MNISTModelPaths(BaseModel):
    """Paths to output files from a MNISTModel training run."""

    mnist_train_config: Path
    checkpoint: Path


class MNISTModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    n_layers: PositiveInt
    input_dim: PositiveInt = 784  # 28x28 images
    hidden_dim: PositiveInt
    output_dim: PositiveInt = 10  # 10 classes
    activation: Literal["relu"] = "relu"
    device: str


class MNISTModel(HookedRootModule):
    """Simple fully-connected ReLU network for MNIST classification."""

    def __init__(self, config: MNISTModelConfig):
        super().__init__()
        self.config = config

        # Create layers
        layers = []

        # Input layer
        layers.append(nn.Linear(config.input_dim, config.hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(config.n_layers - 1):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(config.hidden_dim, config.output_dim))

        self.layers = nn.Sequential(*layers)
        self.setup()

    def forward(
        self, x: Float[Tensor, "batch input_dim"], **_: Any
    ) -> Float[Tensor, "batch output_dim"]:
        # Ensure input is flattened
        x = x.view(-1, self.config.input_dim)
        return self.layers(x)

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> MNISTModelPaths:
        """Download the relevant files from a wandb run."""
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)
        run_dir = fetch_wandb_run_dir(run.id)

        mnist_model_config_path = download_wandb_file(run, run_dir, "mnist_train_config.yaml")

        checkpoint = fetch_latest_wandb_checkpoint(run)
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)
        return MNISTModelPaths(
            mnist_train_config=mnist_model_config_path, checkpoint=checkpoint_path
        )

    @classmethod
    def from_pretrained(cls, path: ModelPath) -> tuple["MNISTModel", dict[str, Any]]:
        """Fetch a pretrained model from wandb or a local path to a checkpoint."""
        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
            paths = cls._download_wandb_files(wandb_path)
        else:
            # `path` should be a local path to a checkpoint
            paths = MNISTModelPaths(
                mnist_train_config=Path(path).parent / "mnist_train_config.yaml",
                checkpoint=Path(path),
            )

        with open(paths.mnist_train_config) as f:
            mnist_train_config_dict = yaml.safe_load(f)

        mnist_config = MNISTModelConfig(**mnist_train_config_dict["mnist_model_config"])
        mnist_model = cls(config=mnist_config)
        params = torch.load(paths.checkpoint, weights_only=True, map_location="cpu")
        mnist_model.load_state_dict(params)

        return mnist_model, mnist_train_config_dict


class MNISTSPDPaths(BaseModel):
    """Paths to output files from a MNISTSPDModel training run."""

    final_config: Path
    mnist_train_config: Path
    checkpoint: Path


class MNISTSPDModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    n_layers: PositiveInt
    input_dim: PositiveInt = 784
    hidden_dim: PositiveInt
    output_dim: PositiveInt = 10
    C: PositiveInt
    m: PositiveInt | None = None
    activation: Literal["relu"] = "relu"
    device: str


class MNISTSPDModel(SPDModel):
    """Decomposed version of the MNIST model for SPD analysis."""

    def __init__(self, config: MNISTSPDModelConfig):
        super().__init__()
        self.config = config
        self.C = config.C
        self.m = min(config.hidden_dim, config.hidden_dim) + 1 if config.m is None else config.m

        # Create decomposed layers
        self.layers = nn.ModuleList()

        # First layer (input -> hidden)
        self.layers.append(
            LinearComponent(
                d_in=config.input_dim,
                d_out=config.hidden_dim,
                n_instances=1,  # Using 1 instance for simplicity
                init_type="xavier_normal",
                init_scale=1.0,
                C=self.C,
                m=self.m,
            )
        )

        # Hidden layers
        for _ in range(config.n_layers - 1):
            self.layers.append(
                LinearComponent(
                    d_in=config.hidden_dim,
                    d_out=config.hidden_dim,
                    n_instances=1,  # Using 1 instance for simplicity
                    init_type="xavier_normal",
                    init_scale=1.0,
                    C=self.C,
                    m=self.m,
                )
            )

        # Output layer
        self.layers.append(
            LinearComponent(
                d_in=config.hidden_dim,
                d_out=config.output_dim,
                n_instances=1,  # Using 1 instance for simplicity
                init_type="xavier_normal",
                init_scale=1.0,
                C=self.C,
                m=self.m,
            )
        )

        self.setup()

    def forward(
        self,
        x: Float[Tensor, "batch input_dim"],
        topk_mask: Bool[Tensor, "batch n_instances C"] | None = None,
    ) -> Float[Tensor, "batch output_dim"]:
        # Ensure input is flattened
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, -1)  # Add instance dimension

        # Apply each layer with activation
        for i, layer in enumerate(self.layers):
            x = layer(x, topk_mask=topk_mask)
            # Apply ReLU after all but the last layer
            if i < len(self.layers) - 1:
                x = F.relu(x)

        # Remove instance dimension
        return x.squeeze(1)

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> MNISTSPDPaths:
        """Download the relevant files from a wandb run."""
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)

        checkpoint = fetch_latest_wandb_checkpoint(run, prefix="spd_model")

        run_dir = fetch_wandb_run_dir(run.id)

        final_config_path = download_wandb_file(run, run_dir, "final_config.yaml")
        mnist_train_config_path = download_wandb_file(run, run_dir, "mnist_train_config.yaml")
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)
        return MNISTSPDPaths(
            final_config=final_config_path,
            mnist_train_config=mnist_train_config_path,
            checkpoint=checkpoint_path,
        )

    @classmethod
    def from_pretrained(cls, path: ModelPath) -> tuple["MNISTSPDModel", Config]:
        """Fetch a pretrained model from wandb or a local path to a checkpoint."""
        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
            paths = cls._download_wandb_files(wandb_path)
        else:
            paths = MNISTSPDPaths(
                final_config=Path(path).parent / "final_config.yaml",
                mnist_train_config=Path(path).parent / "mnist_train_config.yaml",
                checkpoint=Path(path),
            )

        with open(paths.final_config) as f:
            final_config_dict = yaml.safe_load(f)

        spd_config = Config(**final_config_dict)

        with open(paths.mnist_train_config) as f:
            mnist_train_config_dict = yaml.safe_load(f)

        assert isinstance(spd_config.task_config, MNISTTaskConfig)
        mnist_spd_config = MNISTSPDModelConfig(
            **mnist_train_config_dict["mnist_model_config"],
            C=spd_config.C,
            m=spd_config.m,
        )
        model = cls(config=mnist_spd_config)
        params = torch.load(paths.checkpoint, weights_only=True, map_location="cpu")
        model.load_state_dict(params)
        return model, spd_config
