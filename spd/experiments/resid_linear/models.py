import json
from pathlib import Path
from typing import Any

import einops
import torch
import torch.nn.functional as F
import wandb
import yaml
from jaxtyping import Bool, Float
from torch import Tensor, nn
from wandb.apis.public import Run

from spd.models.base import Model, SPDFullRankModel
from spd.models.components import MLP, MLPComponentsFullRank
from spd.run_spd import Config, ResidualLinearConfig
from spd.types import RootPath
from spd.utils import download_wandb_file, init_param_, load_yaml


class ResidualLinearModel(Model):
    def __init__(self, n_features: int, d_embed: int, d_mlp: int, n_layers: int):
        super().__init__()
        self.n_features = n_features
        self.d_embed = d_embed
        self.d_mlp = d_mlp
        self.n_layers = n_layers

        self.W_E = nn.Parameter(torch.empty(n_features, d_embed))
        init_param_(self.W_E)
        # Make each feature have norm 1
        self.W_E.data /= self.W_E.data.norm(dim=1, keepdim=True)

        self.layers = nn.ModuleList(
            [MLP(d_model=d_embed, d_mlp=d_mlp, act_fn=F.gelu) for _ in range(n_layers)]
        )

    def forward(
        self, x: Float[Tensor, "batch n_features"]
    ) -> tuple[
        Float[Tensor, "batch d_embed"],
        dict[str, Float[Tensor, "batch d_embed"] | Float[Tensor, "batch d_mlp"]],
        dict[str, Float[Tensor, "batch d_embed"] | Float[Tensor, "batch d_mlp"]],
    ]:
        layer_pre_acts = {}
        layer_post_acts = {}
        residual = einops.einsum(
            x, self.W_E, "batch n_features, n_features d_embed -> batch d_embed"
        )
        for i, layer in enumerate(self.layers):
            out, pre_acts_i, post_acts_i = layer(residual)
            for k, v in pre_acts_i.items():
                layer_pre_acts[f"layers.{i}.{k}"] = v
            for k, v in post_acts_i.items():
                layer_post_acts[f"layers.{i}.{k}"] = v
            residual = residual + out

        return residual, layer_pre_acts, layer_post_acts

    @classmethod
    def from_pretrained(
        cls, path: str | Path
    ) -> tuple["ResidualLinearModel", dict[str, Any], list[float]]:
        params = torch.load(path, weights_only=True, map_location="cpu")
        with open(Path(path).parent / "target_model_config.yaml") as f:
            config_dict = yaml.safe_load(f)

        with open(Path(path).parent / "label_coeffs.json") as f:
            label_coeffs = json.load(f)

        model = cls(
            n_features=config_dict["n_features"],
            d_embed=config_dict["d_embed"],
            d_mlp=config_dict["d_mlp"],
            n_layers=config_dict["n_layers"],
        )
        model.load_state_dict(params)
        return model, config_dict, label_coeffs

    def all_decomposable_params(
        self,
    ) -> dict[str, Float[Tensor, " d_out"] | Float[Tensor, "d_in d_out"]]:  # bias or weight
        """Dictionary of all parameters which will be decomposed with SPD."""
        params = {}
        for i, mlp in enumerate(self.layers):
            # We transpose because our SPD model uses (input, output) pairs, not (output, input)
            params[f"layers.{i}.input_layer.weight"] = mlp.input_layer.weight.T
            params[f"layers.{i}.input_layer.bias"] = mlp.input_layer.bias
            params[f"layers.{i}.output_layer.weight"] = mlp.output_layer.weight.T
            params[f"layers.{i}.output_layer.bias"] = mlp.output_layer.bias
        return params


class ResidualLinearSPDFullRankModel(SPDFullRankModel):
    def __init__(
        self, n_features: int, d_embed: int, d_mlp: int, n_layers: int, k: int, init_scale: float
    ):
        super().__init__()
        self.n_features = n_features
        self.d_embed = d_embed
        self.d_mlp = d_mlp
        self.n_layers = n_layers
        self.k = k

        self.W_E = nn.Parameter(torch.empty(n_features, d_embed))

        self.layers = nn.ModuleList(
            [
                MLPComponentsFullRank(
                    d_embed=self.d_embed,
                    d_mlp=d_mlp,
                    k=k,
                    init_scale=init_scale,
                    in_bias=True,
                    out_bias=True,
                    act_fn=F.gelu,
                )
                for _ in range(n_layers)
            ]
        )

    def all_subnetwork_params(
        self,
    ) -> dict[str, Float[Tensor, "k d_out"] | Float[Tensor, "k d_in d_out"]]:  # bias or weight
        params = {}
        for i, mlp in enumerate(self.layers):
            params[f"layers.{i}.input_layer.weight"] = mlp.linear1.subnetwork_params
            params[f"layers.{i}.input_layer.bias"] = mlp.linear1.bias
            params[f"layers.{i}.output_layer.weight"] = mlp.linear2.subnetwork_params
            params[f"layers.{i}.output_layer.bias"] = mlp.linear2.bias
        return params

    def all_subnetwork_params_summed(
        self,
    ) -> dict[str, Float[Tensor, "k d_out"] | Float[Tensor, "k d_in d_out"]]:  # bias or weight
        return {p_name: p.sum(dim=0) for p_name, p in self.all_subnetwork_params().items()}

    def forward(
        self, x: Float[Tensor, "batch n_features"], topk_mask: Bool[Tensor, "batch k"] | None = None
    ) -> tuple[
        Float[Tensor, "batch d_embed"],
        dict[str, Float[Tensor, "batch d_embed"] | Float[Tensor, "batch d_mlp"]],
        dict[str, Float[Tensor, "batch k d_embed"]],
    ]:
        """
        Returns:
            x: The output of the model
            layer_acts: A dictionary of activations for each layer in each MLP.
            inner_acts: A dictionary of component activations (just after the A matrix) for each
                layer in each MLP.
        """
        layer_acts = {}
        inner_acts = {}
        residual = einops.einsum(
            x, self.W_E, "batch n_features, n_features d_embed -> batch d_embed"
        )
        for i, layer in enumerate(self.layers):
            layer_out, layer_acts_i, inner_acts_i = layer(residual, topk_mask)
            assert len(layer_acts_i) == len(inner_acts_i) == 2
            residual = residual + layer_out
            layer_acts[f"layers.{i}.input_layer.weight"] = layer_acts_i[0]
            layer_acts[f"layers.{i}.output_layer.weight"] = layer_acts_i[1]
            inner_acts[f"layers.{i}.input_layer.weight"] = inner_acts_i[0]
            inner_acts[f"layers.{i}.output_layer.weight"] = inner_acts_i[1]
        return residual, layer_acts, inner_acts

    @classmethod
    def _load_model(
        cls,
        config_path: Path,
        target_model_config_path: Path,
        checkpoint_path: Path,
        label_coeffs_path: Path,
    ) -> tuple["ResidualLinearSPDFullRankModel", Config, list[float]]:
        """Helper function to load the model from local files."""

        # Load config
        config = Config(**load_yaml(config_path))
        assert isinstance(config.task_config, ResidualLinearConfig)

        # Load target model config
        target_model_config = load_yaml(target_model_config_path)

        # Load checkpoint
        params = torch.load(checkpoint_path, weights_only=True, map_location="cpu")

        # Create model
        model = cls(
            n_features=target_model_config["n_features"],
            d_embed=target_model_config["d_embed"],
            d_mlp=target_model_config["d_mlp"],
            n_layers=target_model_config["n_layers"],
            k=config.task_config.k,
            init_scale=config.task_config.init_scale,
        )
        model.load_state_dict(params)

        # Load label coefficients
        with open(label_coeffs_path) as f:
            label_coeffs = json.load(f)

        return model, config, label_coeffs

    @classmethod
    def from_local_path(
        cls, path: RootPath
    ) -> tuple["ResidualLinearSPDFullRankModel", Config, list[float]]:
        """Instantiate from a checkpoint file."""
        path = Path(path)
        model_dir = path.parent
        return cls._load_model(
            config_path=model_dir / "final_config.yaml",
            target_model_config_path=model_dir / "target_model_config.yaml",
            checkpoint_path=path,
            label_coeffs_path=model_dir / "label_coeffs.json",
        )

    @classmethod
    def from_wandb(
        cls, wandb_project_run_id: str
    ) -> tuple["ResidualLinearSPDFullRankModel", Config, list[float]]:
        """Instantiate ResidualLinearSPDFullRankModel using the latest checkpoint from a wandb run."""
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)

        # Get the latest checkpoint
        checkpoints = [
            file
            for file in run.files()
            if file.name.endswith(".pth") and "target_model" not in file.name
        ]
        if not checkpoints:
            raise ValueError(f"No checkpoint files found in run {wandb_project_run_id}")
        latest_checkpoint_remote = sorted(
            checkpoints, key=lambda x: int(x.name.split(".pth")[0].split("_")[-1])
        )[-1]

        config_path = download_wandb_file(run, "final_config.yaml")
        target_model_config_path = download_wandb_file(run, "target_model_config.yaml")
        label_coeffs_path = download_wandb_file(run, "label_coeffs.json")
        checkpoint_path = download_wandb_file(run, latest_checkpoint_remote.name)

        return cls._load_model(
            config_path=config_path,
            target_model_config_path=target_model_config_path,
            checkpoint_path=checkpoint_path,
            label_coeffs_path=label_coeffs_path,
        )

    def set_subnet_to_zero(
        self, subnet_idx: int
    ) -> dict[str, Float[Tensor, " d_out"] | Float[Tensor, "d_in d_out"]]:  # bias or weight
        stored_vals = {}
        for i, mlp in enumerate(self.layers):
            stored_vals[f"layers.{i}.input_layer.weight"] = (
                mlp.linear1.subnetwork_params[subnet_idx, :, :].detach().clone()
            )
            stored_vals[f"layers.{i}.input_layer.bias"] = (
                mlp.linear1.bias[subnet_idx, :].detach().clone()
            )
            stored_vals[f"layers.{i}.output_layer.weight"] = (
                mlp.linear2.subnetwork_params[subnet_idx, :, :].detach().clone()
            )
            stored_vals[f"layers.{i}.output_layer.bias"] = (
                mlp.linear2.bias[subnet_idx, :].detach().clone()
            )
            mlp.linear1.subnetwork_params[subnet_idx, :, :] = 0.0
            mlp.linear1.bias[subnet_idx, :] = 0.0
            mlp.linear2.subnetwork_params[subnet_idx, :, :] = 0.0
            mlp.linear2.bias[subnet_idx, :] = 0.0
        return stored_vals

    def restore_subnet(
        self,
        subnet_idx: int,
        stored_vals: dict[str, Float[Tensor, " d_out"] | Float[Tensor, "d_in d_out"]],
    ) -> None:
        for i, mlp in enumerate(self.layers):
            mlp.linear1.subnetwork_params[subnet_idx, :, :] = stored_vals[
                f"layers.{i}.input_layer.weight"
            ]
            mlp.linear1.bias[subnet_idx, :] = stored_vals[f"layers.{i}.input_layer.bias"]
            mlp.linear2.subnetwork_params[subnet_idx, :, :] = stored_vals[
                f"layers.{i}.output_layer.weight"
            ]
            mlp.linear2.bias[subnet_idx, :] = stored_vals[f"layers.{i}.output_layer.bias"]
