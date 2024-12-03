import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import einops
import torch
import torch.nn.functional as F
import wandb
import yaml
from jaxtyping import Bool, Float
from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from torch import Tensor, nn
from wandb.apis.public import Run

from spd.models.base import Model, SPDRankPenaltyModel
from spd.models.components import InstancesParamComponentsRankPenalty
from spd.run_spd import Config, ResidualMLPTaskConfig
from spd.types import WANDB_PATH_PREFIX, ModelPath
from spd.utils import init_param_, remove_grad_parallel_to_subnetwork_vecs
from spd.wandb_utils import download_wandb_file, fetch_latest_wandb_checkpoint, fetch_wandb_run_dir


class InstancesMLP(nn.Module):
    def __init__(
        self,
        n_instances: int,
        d_model: int,
        d_mlp: int,
        act_fn: Callable[[Tensor], Tensor],
        in_bias: bool,
        out_bias: bool,
        init_scale: float,
    ):
        super().__init__()
        self.n_instances = n_instances
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.act_fn = act_fn
        self.linear1 = nn.Parameter(torch.empty(n_instances, d_model, d_mlp))
        self.linear2 = nn.Parameter(torch.empty(n_instances, d_mlp, d_model))
        init_param_(self.linear1, scale=init_scale)
        init_param_(self.linear2, scale=init_scale)
        self.bias1 = None
        self.bias2 = None
        if in_bias:
            self.bias1 = nn.Parameter(torch.zeros(n_instances, d_mlp))
        if out_bias:
            self.bias2 = nn.Parameter(torch.zeros(n_instances, d_model))

    def forward(
        self, x: Float[Tensor, "batch n_instances d_model"]
    ) -> tuple[
        Float[Tensor, "batch n_instances d_model"],
        dict[
            str,
            Float[Tensor, "batch n_instances d_model"] | Float[Tensor, "batch n_instances d_mlp"],
        ],
        dict[
            str,
            Float[Tensor, "batch n_instances d_model"] | Float[Tensor, "batch n_instances d_mlp"],
        ],
    ]:
        """Run a forward pass and cache pre and post activations for each parameter.

        Note that we don't need to cache pre activations for the biases. We also don't care about
        the output bias which is always zero.
        """
        out1_pre_act_fn = einops.einsum(
            x,
            self.linear1,
            "batch n_instances d_model, n_instances d_model d_mlp -> batch n_instances d_mlp",
        )
        if self.bias1 is not None:
            out1_pre_act_fn = out1_pre_act_fn + self.bias1
        out1 = self.act_fn(out1_pre_act_fn)
        out2 = einops.einsum(
            out1,
            self.linear2,
            "batch n_instances d_mlp, n_instances d_mlp d_model -> batch n_instances d_model",
        )
        if self.bias2 is not None:
            out2 = out2 + self.bias2

        pre_acts = {
            "linear1": x,
            "bias1": x,
            "linear2": out1,
            "bias2": out1,
        }
        post_acts = {
            "linear1": out1_pre_act_fn,
            "bias1": out1_pre_act_fn,
            "linear2": out2,
            "bias2": out2,
        }
        return out2, pre_acts, post_acts


class InstancesMLPComponentsRankPenalty(nn.Module):
    """A module that contains two linear layers with an activation in between for rank penalty SPD.

    Each linear layer is decomposed into A and B matrices, where the weight matrix W = A @ B.
    The biases are (optionally) part of the "linear" layers, and have a subnetwork dimension.
    """

    def __init__(
        self,
        n_instances: int,
        d_embed: int,
        d_mlp: int,
        k: int,
        init_scale: float,
        act_fn: Callable[[Tensor], Tensor],
        in_bias: bool,
        out_bias: bool,
        m: int | None = None,
    ):
        super().__init__()
        self.act_fn = act_fn
        self.linear1 = InstancesParamComponentsRankPenalty(
            n_instances=n_instances,
            in_dim=d_embed,
            out_dim=d_mlp,
            k=k,
            bias=in_bias,
            init_scale=init_scale,
            m=m,
        )
        self.linear2 = InstancesParamComponentsRankPenalty(
            n_instances=n_instances,
            in_dim=d_mlp,
            out_dim=d_embed,
            k=k,
            bias=out_bias,
            init_scale=init_scale,
            m=m,
        )

    def forward(
        self,
        x: Float[Tensor, "batch n_instances d_embed"],
        topk_mask: Bool[Tensor, "batch n_instances k"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch n_instances d_embed"],
        list[Float[Tensor, "batch n_instances d_embed"] | Float[Tensor, "batch n_instances d_mlp"]],
        list[
            Float[Tensor, "batch n_instances k d_embed"]
            | Float[Tensor, "batch n_instances k d_mlp"]
        ],
    ]:
        """Forward pass through the MLP.

        Args:
            x: Input tensor
            topk_mask: Boolean tensor indicating which subnetworks to keep
        Returns:
            x: The output of the MLP
            layer_acts: The activations at the output of each layer after summing over the
                subnetwork dimension
            inner_acts: The activations at the output of each subnetwork before summing
        """
        layer_acts = []
        inner_acts = []

        # First layer
        x, inner_act = self.linear1(x, topk_mask)
        inner_acts.append(inner_act)
        layer_acts.append(x)
        x = self.act_fn(x)

        # Second layer
        x, inner_act = self.linear2(x, topk_mask)
        inner_acts.append(inner_act)
        layer_acts.append(x)

        return x, layer_acts, inner_acts


class ResidualMLPPaths(BaseModel):
    """Paths to output files from a ResidualMLPModel training run."""

    resid_mlp_train_config: Path
    label_coeffs: Path
    checkpoint: Path


class ResidualMLPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    n_instances: PositiveInt
    n_features: PositiveInt
    d_embed: PositiveInt
    d_mlp: PositiveInt
    n_layers: PositiveInt
    act_fn_name: Literal["gelu", "relu"] = Field(
        description="Defines the activation function in the model. Also used in the labeling "
        "function if label_type is act_plus_resid."
    )
    apply_output_act_fn: bool
    in_bias: bool
    out_bias: bool
    init_scale: float = 1.0


class ResidualMLPModel(Model):
    def __init__(self, config: ResidualMLPConfig):
        super().__init__()
        self.config = config
        self.W_E = nn.Parameter(torch.empty(config.n_instances, config.n_features, config.d_embed))
        init_param_(self.W_E, scale=config.init_scale)
        self.W_U = nn.Parameter(torch.empty(config.n_instances, config.d_embed, config.n_features))
        init_param_(self.W_U, scale=config.init_scale)

        assert config.act_fn_name in ["gelu", "relu"]
        self.act_fn = F.gelu if config.act_fn_name == "gelu" else F.relu
        self.layers = nn.ModuleList(
            [
                InstancesMLP(
                    n_instances=config.n_instances,
                    d_model=config.d_embed,
                    d_mlp=config.d_mlp,
                    act_fn=self.act_fn,
                    in_bias=config.in_bias,
                    out_bias=config.out_bias,
                    init_scale=config.init_scale,
                )
                for _ in range(config.n_layers)
            ]
        )

    def forward(
        self, x: Float[Tensor, "batch n_instances n_features"]
    ) -> tuple[
        Float[Tensor, "batch n_instances d_embed"],
        dict[
            str,
            Float[Tensor, "batch n_instances d_embed"] | Float[Tensor, "batch n_instances d_mlp"],
        ],
        dict[
            str,
            Float[Tensor, "batch n_instances d_embed"] | Float[Tensor, "batch n_instances d_mlp"],
        ],
    ]:
        # Make sure that n_instances are correct to avoid unintended broadcasting
        assert x.shape[1] == self.config.n_instances, "n_instances mismatch"
        assert x.shape[2] == self.config.n_features, "n_features mismatch"
        layer_pre_acts = {}
        layer_post_acts = {}
        residual = einops.einsum(
            x,
            self.W_E,
            "batch n_instances n_features, n_instances n_features d_embed -> batch n_instances d_embed",
        )
        for i, layer in enumerate(self.layers):
            out, pre_acts_i, post_acts_i = layer(residual)
            for k, v in pre_acts_i.items():
                layer_pre_acts[f"layers.{i}.{k}"] = v
            for k, v in post_acts_i.items():
                layer_post_acts[f"layers.{i}.{k}"] = v
            residual = residual + out
        out = einops.einsum(
            residual,
            self.W_U,
            "batch n_instances d_embed, n_instances d_embed n_features -> batch n_instances n_features",
        )
        if self.config.apply_output_act_fn:
            out = self.act_fn(out)
        return out, layer_pre_acts, layer_post_acts

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> ResidualMLPPaths:
        """Download the relevant files from a wandb run."""
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)

        checkpoint = fetch_latest_wandb_checkpoint(run)

        run_dir = fetch_wandb_run_dir(run.id)

        resid_mlp_train_config_path = download_wandb_file(
            run, run_dir, "resid_mlp_train_config.yaml"
        )
        label_coeffs_path = download_wandb_file(run, run_dir, "label_coeffs.json")
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)
        return ResidualMLPPaths(
            resid_mlp_train_config=resid_mlp_train_config_path,
            label_coeffs=label_coeffs_path,
            checkpoint=checkpoint_path,
        )

    @classmethod
    def from_pretrained(
        cls, path: ModelPath
    ) -> tuple["ResidualMLPModel", dict[str, Any], Float[Tensor, "n_instances n_features"]]:
        """Fetch a pretrained model from wandb or a local path to a checkpoint.

        Args:
            path: The path to local checkpoint or wandb project. If a wandb project, format must be
                `wandb:<entity>/<project>/<run_id>` or `wandb:<entity>/<project>/runs/<run_id>`.
                If `api.entity` is set (e.g. via setting WANDB_ENTITY in .env), <entity> can be
                omitted, and if `api.project` is set, <project> can be omitted. If local path,
                assumes that `resid_mlp_train_config.yaml` and `label_coeffs.json` are in the same
                directory as the checkpoint.

        Returns:
            model: The pretrained ResidualMLPModel
            resid_mlp_train_config_dict: The config dict used to train the model (we don't
                instantiate a train config due to circular import issues)
            label_coeffs: The label coefficients used to train the model
        """
        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
            paths = cls._download_wandb_files(wandb_path)
        else:
            # `path` should be a local path to a checkpoint
            paths = ResidualMLPPaths(
                resid_mlp_train_config=Path(path).parent / "resid_mlp_train_config.yaml",
                label_coeffs=Path(path).parent / "label_coeffs.json",
                checkpoint=Path(path),
            )

        with open(paths.resid_mlp_train_config) as f:
            resid_mlp_train_config_dict = yaml.safe_load(f)

        with open(paths.label_coeffs) as f:
            label_coeffs = torch.tensor(json.load(f))

        resid_mlp_config = ResidualMLPConfig(**resid_mlp_train_config_dict["resid_mlp_config"])
        resid_mlp = cls(resid_mlp_config)
        params = torch.load(paths.checkpoint, weights_only=True, map_location="cpu")
        resid_mlp.load_state_dict(params)

        return resid_mlp, resid_mlp_train_config_dict, label_coeffs

    def all_decomposable_params(
        self,
    ) -> dict[
        str, Float[Tensor, "n_instances d_out d_in"] | Float[Tensor, "n_instances d_in d_out"]
    ]:
        """Dictionary of all parameters which will be decomposed with SPD.

        Note that we exclude biases which we never decompose.

        TODO: Decompose embedding matrices if desired.
        """
        params = {}
        for i, mlp in enumerate(self.layers):
            params[f"layers.{i}.linear1"] = mlp.linear1
            params[f"layers.{i}.linear2"] = mlp.linear2
        return params


class ResidualMLPSPDRankPenaltyPaths(BaseModel):
    """Paths to output files from a ResidualMLPSPDRankPenaltyModel training run."""

    final_config: Path
    resid_mlp_train_config: Path
    label_coeffs: Path
    checkpoint: Path


class ResidualMLPSPDRankPenaltyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    n_instances: PositiveInt
    n_features: PositiveInt
    d_embed: PositiveInt
    d_mlp: PositiveInt
    n_layers: PositiveInt
    act_fn_name: Literal["gelu", "relu"]
    apply_output_act_fn: bool
    in_bias: bool
    out_bias: bool
    init_scale: float
    k: PositiveInt
    m: PositiveInt | None = None


class ResidualMLPSPDRankPenaltyModel(SPDRankPenaltyModel):
    def __init__(
        self,
        config: ResidualMLPSPDRankPenaltyConfig,
    ):
        super().__init__()
        self.config = config
        self.n_features = config.n_features  # Currently needed for backward compatibility
        self.n_instances = config.n_instances  # Currently needed for backward compatibility

        assert config.act_fn_name in ["gelu", "relu"]
        self.act_fn = F.gelu if config.act_fn_name == "gelu" else F.relu

        self.W_E = nn.Parameter(torch.empty(config.n_instances, config.n_features, config.d_embed))
        self.W_U = nn.Parameter(torch.empty(config.n_instances, config.d_embed, config.n_features))
        init_param_(self.W_E)
        init_param_(self.W_U)

        self.m = min(config.d_embed, config.d_mlp) if config.m is None else config.m

        self.layers = nn.ModuleList(
            [
                InstancesMLPComponentsRankPenalty(
                    n_instances=config.n_instances,
                    d_embed=config.d_embed,
                    d_mlp=config.d_mlp,
                    k=config.k,
                    init_scale=config.init_scale,
                    in_bias=config.in_bias,
                    out_bias=config.out_bias,
                    act_fn=self.act_fn,
                )
                for _ in range(config.n_layers)
            ]
        )

    def all_subnetwork_params(
        self,
    ) -> dict[str, Float[Tensor, "n_instances k d_in d_out"]]:
        AB_ein_str = "n_instances k d_in m, n_instances k m d_out -> n_instances k d_in d_out"
        params = {}
        for i, mlp in enumerate(self.layers):
            params[f"layers.{i}.linear1"] = einops.einsum(mlp.linear1.A, mlp.linear1.B, AB_ein_str)
            params[f"layers.{i}.linear2"] = einops.einsum(mlp.linear2.A, mlp.linear2.B, AB_ein_str)
        return params

    def all_subnetwork_params_summed(
        self,
    ) -> dict[str, Float[Tensor, "n_instances k d_in d_out"]]:
        return {p_name: p.sum(dim=1) for p_name, p in self.all_subnetwork_params().items()}

    def forward(
        self,
        x: Float[Tensor, "batch n_instances n_features"],
        topk_mask: Bool[Tensor, "batch n_instances k"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch n_instances d_embed"],
        dict[
            str,
            Float[Tensor, "batch n_instances d_embed"] | Float[Tensor, "batch n_instances d_mlp"],
        ],
        dict[
            str,
            Float[Tensor, "batch n_instances k d_embed"]
            | Float[Tensor, "batch n_instances k d_mlp"],
        ],
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
            x,
            self.W_E,
            "batch n_instances n_features, n_instances n_features d_embed -> batch n_instances d_embed",
        )
        for i, layer in enumerate(self.layers):
            layer_out, layer_acts_i, inner_acts_i = layer(residual, topk_mask)
            assert len(layer_acts_i) == len(inner_acts_i) == 2
            residual = residual + layer_out
            layer_acts[f"layers.{i}.linear1"] = layer_acts_i[0]
            layer_acts[f"layers.{i}.linear2"] = layer_acts_i[1]
            inner_acts[f"layers.{i}.linear1"] = inner_acts_i[0]
            inner_acts[f"layers.{i}.linear2"] = inner_acts_i[1]
        out = einops.einsum(
            residual,
            self.W_U,
            "batch n_instances d_embed, n_instances d_embed n_features -> batch n_instances n_features",
        )
        if self.config.apply_output_act_fn:
            out = self.act_fn(out)
        return out, layer_acts, inner_acts

    def set_subnet_to_zero(
        self, subnet_idx: int
    ) -> dict[str, Float[Tensor, "n_instances k d_in m"] | Float[Tensor, "n_instances k m d_out"]]:
        stored_vals = {}
        for i, mlp in enumerate(self.layers):
            stored_vals[f"layers.{i}.linear1.A"] = mlp.linear1.A[subnet_idx].detach().clone()
            stored_vals[f"layers.{i}.linear1.B"] = mlp.linear1.B[subnet_idx].detach().clone()
            stored_vals[f"layers.{i}.linear2.A"] = mlp.linear2.A[subnet_idx].detach().clone()
            stored_vals[f"layers.{i}.linear2.B"] = mlp.linear2.B[subnet_idx].detach().clone()

            mlp.linear1.A.data[subnet_idx] = 0.0
            mlp.linear1.B.data[subnet_idx] = 0.0
            mlp.linear2.A.data[subnet_idx] = 0.0
            mlp.linear2.B.data[subnet_idx] = 0.0
        return stored_vals

    def restore_subnet(
        self,
        subnet_idx: int,
        stored_vals: dict[
            str, Float[Tensor, "n_instances k d_in m"] | Float[Tensor, "n_instances k m d_out"]
        ],
    ) -> None:
        for i, mlp in enumerate(self.layers):
            mlp.linear1.A[subnet_idx].data = stored_vals[f"layers.{i}.linear1.A"]
            mlp.linear1.B[subnet_idx].data = stored_vals[f"layers.{i}.linear1.B"]
            mlp.linear2.A[subnet_idx].data = stored_vals[f"layers.{i}.linear2.A"]
            mlp.linear2.B[subnet_idx].data = stored_vals[f"layers.{i}.linear2.B"]

    def all_As_and_Bs(
        self,
    ) -> dict[
        str, tuple[Float[Tensor, "n_instances k d_in m"], Float[Tensor, "n_instances k m d_out"]]
    ]:
        """Get all A and B matrices for each layer."""
        params = {}
        for i, mlp in enumerate(self.layers):
            params[f"layers.{i}.linear1"] = (mlp.linear1.A, mlp.linear1.B)
            params[f"layers.{i}.linear2"] = (mlp.linear2.A, mlp.linear2.B)
        return params

    def set_matrices_to_unit_norm(self):
        for mlp in self.layers:
            mlp.linear1.A.data /= mlp.linear1.A.data.norm(p=2, dim=-2, keepdim=True)
            mlp.linear2.A.data /= mlp.linear2.A.data.norm(p=2, dim=-2, keepdim=True)

    def fix_normalized_adam_gradients(self):
        for mlp in self.layers:
            assert mlp.linear1.A.grad is not None
            remove_grad_parallel_to_subnetwork_vecs(mlp.linear1.A.data, mlp.linear1.A.grad)
            assert mlp.linear2.A.grad is not None
            remove_grad_parallel_to_subnetwork_vecs(mlp.linear2.A.data, mlp.linear2.A.grad)

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> ResidualMLPSPDRankPenaltyPaths:
        """Download the relevant files from a wandb run."""
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)

        checkpoint = fetch_latest_wandb_checkpoint(run)

        run_dir = fetch_wandb_run_dir(run.id)

        final_config_path = download_wandb_file(run, run_dir, "final_config.yaml")
        resid_mlp_train_config_path = download_wandb_file(
            run, run_dir, "resid_mlp_train_config.yaml"
        )
        label_coeffs_path = download_wandb_file(run, run_dir, "label_coeffs.json")
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)
        return ResidualMLPSPDRankPenaltyPaths(
            final_config=final_config_path,
            resid_mlp_train_config=resid_mlp_train_config_path,
            label_coeffs=label_coeffs_path,
            checkpoint=checkpoint_path,
        )

    @classmethod
    def from_pretrained(
        cls, path: str | Path
    ) -> tuple["ResidualMLPSPDRankPenaltyModel", Config, Float[Tensor, "n_instances n_features"]]:
        """Fetch a pretrained model from wandb or a local path to a checkpoint.

        Args:
            path: The path to local checkpoint or wandb project. If a wandb project, the format
                must be `wandb:entity/project/run_id`. If `api.entity` is set (e.g. via setting
                WANDB_ENTITY in .env), this can be in the form `wandb:project/run_id` and if
                form `wandb:project/run_id` and if `api.project` is set this can just be
                `wandb:run_id`. If local path, assumes that `resid_mlp_train_config.yaml` and
                `label_coeffs.json` are in the same directory as the checkpoint.
        """
        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
            paths = cls._download_wandb_files(wandb_path)
        else:
            paths = ResidualMLPSPDRankPenaltyPaths(
                final_config=Path(path).parent / "final_config.yaml",
                resid_mlp_train_config=Path(path).parent / "resid_mlp_train_config.yaml",
                label_coeffs=Path(path).parent / "label_coeffs.json",
                checkpoint=Path(path),
            )

        with open(paths.final_config) as f:
            final_config_dict = yaml.safe_load(f)
        config = Config(**final_config_dict)

        with open(paths.resid_mlp_train_config) as f:
            resid_mlp_train_config_dict = yaml.safe_load(f)

        with open(paths.label_coeffs) as f:
            label_coeffs = torch.tensor(json.load(f))

        assert isinstance(config.task_config, ResidualMLPTaskConfig)
        resid_mlp_spd_rank_penalty_config = ResidualMLPSPDRankPenaltyConfig(
            **resid_mlp_train_config_dict, k=config.task_config.k, m=config.m
        )
        model = cls(config=resid_mlp_spd_rank_penalty_config)
        params = torch.load(paths.checkpoint, weights_only=True, map_location="cpu")
        model.load_state_dict(params)
        return model, config, label_coeffs
