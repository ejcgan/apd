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
from torch import Tensor, nn
from wandb.apis.public import Run

from spd.models.base import Model, SPDRankPenaltyModel
from spd.run_spd import Config, ResidualMLPConfig
from spd.types import RootPath
from spd.utils import (
    download_wandb_file,
    init_param_,
    load_yaml,
    remove_grad_parallel_to_subnetwork_vecs,
)


class InstancesMLP(nn.Module):
    def __init__(
        self,
        n_instances: int,
        d_model: int,
        d_mlp: int,
        act_fn: Callable[[Tensor], Tensor],
        in_bias: bool,
        out_bias: bool,
    ):
        super().__init__()
        self.n_instances = n_instances
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.act_fn = act_fn
        self.linear1 = nn.Parameter(torch.empty(n_instances, d_model, d_mlp))
        self.linear2 = nn.Parameter(torch.empty(n_instances, d_mlp, d_model))
        init_param_(self.linear1)
        init_param_(self.linear2)
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


class InstancesParamComponentsRankPenalty(nn.Module):
    """A linear layer decomposed into A and B matrices for rank penalty SPD.

    The weight matrix W is decomposed as W = A @ B, where A and B are learned parameters.
    """

    def __init__(
        self,
        n_instances: int,
        in_dim: int,
        out_dim: int,
        k: int,
        bias: bool,
        init_scale: float = 1.0,
        m: int | None = None,
    ):
        super().__init__()
        self.n_instances = n_instances
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.m = min(in_dim, out_dim) if m is None else m

        # Initialize A and B matrices
        self.A = nn.Parameter(torch.empty(n_instances, k, in_dim, self.m))
        self.B = nn.Parameter(torch.empty(n_instances, k, self.m, out_dim))
        self.bias = nn.Parameter(torch.zeros(n_instances, out_dim)) if bias else None

        init_param_(self.A, scale=init_scale)
        init_param_(self.B, scale=init_scale)

    @property
    def subnetwork_params(self) -> Float[Tensor, "n_instances k d_in d_out"]:
        """For compatibility with plotting code."""
        return einops.einsum(
            self.A,
            self.B,
            "n_instances k d_in m, n_instances k m d_out -> n_instances k d_in d_out",
        )

    def forward(
        self,
        x: Float[Tensor, "batch n_instances d_in"],
        topk_mask: Bool[Tensor, "batch n_instances k"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch n_instances d_out"], Float[Tensor, "batch n_instances k d_out"]
    ]:
        """Forward pass through the layer.

        Args:
            x: Input tensor
            topk_mask: Boolean tensor indicating which subnetworks to keep
        Returns:
            output: The summed output across all subnetworks
            inner_acts: The output of each subnetwork before summing
        """
        # First multiply by A to get to intermediate dimension m
        pre_inner_acts = einops.einsum(
            x, self.A, "batch n_instances d_in, n_instances k d_in m -> batch n_instances k m"
        )
        if topk_mask is not None:
            assert topk_mask.shape == pre_inner_acts.shape[:-1]
            pre_inner_acts = einops.einsum(
                pre_inner_acts,
                topk_mask,
                "batch n_instances k m, batch n_instances k -> batch n_instances k m",
            )

        # Then multiply by B to get to output dimension
        inner_acts = einops.einsum(
            pre_inner_acts,
            self.B,
            "batch n_instances k m, n_instances k m d_out -> batch n_instances k d_out",
        )

        if topk_mask is not None:
            inner_acts = einops.einsum(
                inner_acts,
                topk_mask,
                "batch n_instances k d_out, batch n_instances k -> batch n_instances k d_out",
            )

        # Sum over subnetwork dimension
        out = einops.einsum(inner_acts, "batch n_instances k d_out -> batch n_instances d_out")

        # Add the bias if it exists
        if self.bias is not None:
            out += self.bias
        return out, inner_acts


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


class ResidualMLPModel(Model):
    def __init__(
        self,
        n_features: int,
        d_embed: int,
        d_mlp: int,
        n_layers: int,
        n_instances: int,
        act_fn_name: Literal["gelu", "relu"],
        in_bias: bool,
        out_bias: bool,
        apply_output_act_fn: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_embed = d_embed
        self.d_mlp = d_mlp
        self.n_layers = n_layers
        self.n_instances = n_instances
        self.in_bias = in_bias
        self.out_bias = out_bias
        self.act_fn_name = act_fn_name
        self.apply_output_act_fn = apply_output_act_fn
        self.W_E = nn.Parameter(torch.empty(n_instances, n_features, d_embed))
        init_param_(self.W_E)
        self.W_U = nn.Parameter(torch.empty(n_instances, d_embed, n_features))
        init_param_(self.W_U)

        assert act_fn_name in ["gelu", "relu"]
        self.act_fn = F.gelu if act_fn_name == "gelu" else F.relu
        self.layers = nn.ModuleList(
            [
                InstancesMLP(
                    n_instances=n_instances,
                    d_model=d_embed,
                    d_mlp=d_mlp,
                    act_fn=self.act_fn,
                    in_bias=in_bias,
                    out_bias=out_bias,
                )
                for _ in range(n_layers)
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
        assert x.shape[1] == self.n_instances, "n_instances mismatch"
        assert x.shape[2] == self.n_features, "n_features mismatch"
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
        if self.apply_output_act_fn:
            out = self.act_fn(out)
        return out, layer_pre_acts, layer_post_acts

    @classmethod
    def from_pretrained(
        cls, path: str | Path
    ) -> tuple["ResidualMLPModel", dict[str, Any], Float[Tensor, "n_instances n_features"]]:
        params = torch.load(path, weights_only=True, map_location="cpu")
        with open(Path(path).parent / "target_model_config.yaml") as f:
            config_dict = yaml.safe_load(f)

        with open(Path(path).parent / "label_coeffs.json") as f:
            label_coeffs = torch.tensor(json.load(f))

        model = cls(
            n_features=config_dict["n_features"],
            d_embed=config_dict["d_embed"],
            d_mlp=config_dict["d_mlp"],
            n_layers=config_dict["n_layers"],
            n_instances=config_dict["n_instances"],
            act_fn_name=config_dict["act_fn_name"],
            in_bias=config_dict["in_bias"],
            out_bias=config_dict["out_bias"],
        )
        model.load_state_dict(params)
        return model, config_dict, label_coeffs

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


class ResidualMLPSPDRankPenaltyModel(SPDRankPenaltyModel):
    def __init__(
        self,
        n_features: int,
        d_embed: int,
        d_mlp: int,
        n_layers: int,
        n_instances: int,
        k: int,
        init_scale: float,
        act_fn_name: Literal["gelu", "relu"],
        in_bias: bool,
        out_bias: bool,
        apply_output_act_fn: bool = False,
        m: int | None = None,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_embed = d_embed
        self.d_mlp = d_mlp
        self.n_layers = n_layers
        self.n_instances = n_instances
        self.k = k
        self.in_bias = in_bias
        self.out_bias = out_bias
        assert act_fn_name in ["gelu", "relu"]
        self.act_fn = F.gelu if act_fn_name == "gelu" else F.relu
        self.apply_output_act_fn = apply_output_act_fn

        self.W_E = nn.Parameter(torch.empty(n_instances, n_features, d_embed))
        self.W_U = nn.Parameter(torch.empty(n_instances, d_embed, n_features))
        init_param_(self.W_E)
        init_param_(self.W_U)

        self.m = min(d_embed, d_mlp) if m is None else m

        self.layers = nn.ModuleList(
            [
                InstancesMLPComponentsRankPenalty(
                    n_instances=n_instances,
                    d_embed=self.d_embed,
                    d_mlp=d_mlp,
                    k=k,
                    init_scale=init_scale,
                    in_bias=in_bias,
                    out_bias=out_bias,
                    act_fn=self.act_fn,
                )
                for _ in range(n_layers)
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
        if self.apply_output_act_fn:
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

    @classmethod
    def _load_model(
        cls,
        config_path: Path,
        target_model_config_path: Path,
        checkpoint_path: Path,
        label_coeffs_path: Path,
    ) -> tuple["ResidualMLPSPDRankPenaltyModel", Config, list[float]]:
        """Helper function to load the model from local files."""
        # Load config
        config = Config(**load_yaml(config_path))
        assert isinstance(config.task_config, ResidualMLPConfig)

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
            n_instances=target_model_config["n_instances"],
            k=config.task_config.k,
            init_scale=config.task_config.init_scale,
            act_fn_name=target_model_config["act_fn_name"],
            in_bias=target_model_config["in_bias"],
            out_bias=target_model_config["out_bias"],
        )
        model.load_state_dict(params)

        # Load label coefficients
        with open(label_coeffs_path) as f:
            label_coeffs = json.load(f)

        return model, config, label_coeffs

    @classmethod
    def from_local_path(
        cls, path: RootPath
    ) -> tuple["ResidualMLPSPDRankPenaltyModel", Config, list[float]]:
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
    ) -> tuple["ResidualMLPSPDRankPenaltyModel", Config, list[float]]:
        """Instantiate model using the latest checkpoint from a wandb run."""
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
