from pathlib import Path
from typing import Any

import einops
import torch
import wandb
import yaml
from einops import rearrange
from jaxtyping import Bool, Float
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt
from torch import Tensor, nn
from torch.nn import functional as F
from wandb.apis.public import Run

from spd.models.base import Model, SPDFullRankModel, SPDRankPenaltyModel
from spd.models.components import (
    InstancesParamComponentsFullRank,
    InstancesParamComponentsRankPenalty,
)
from spd.run_spd import Config, TMSTaskConfig
from spd.types import WANDB_PATH_PREFIX, ModelPath, RootPath
from spd.utils import remove_grad_parallel_to_subnetwork_vecs
from spd.wandb_utils import download_wandb_file, fetch_latest_wandb_checkpoint, fetch_wandb_run_dir


class TMSModelPaths(BaseModel):
    """Paths to output files from a TMSModel training run."""

    tms_train_config: Path
    checkpoint: Path


class TMSModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    n_instances: PositiveInt
    n_features: PositiveInt
    n_hidden: PositiveInt
    n_hidden_layers: NonNegativeInt
    device: str


class TMSModel(Model):
    def __init__(self, config: TMSModelConfig):
        super().__init__()
        self.config = config

        self.W = nn.Parameter(
            torch.empty(
                (config.n_instances, config.n_features, config.n_hidden), device=config.device
            )
        )
        nn.init.xavier_normal_(self.W)
        self.b_final = nn.Parameter(
            torch.zeros((config.n_instances, config.n_features), device=config.device)
        )

        self.hidden_layers = None
        if config.n_hidden_layers > 0:
            self.hidden_layers = nn.ParameterList()
            for _ in range(config.n_hidden_layers):
                layer = nn.Parameter(
                    torch.empty(
                        (config.n_instances, config.n_hidden, config.n_hidden),
                        device=config.device,
                    )
                )
                nn.init.xavier_normal_(layer)
                self.hidden_layers.append(layer)

    def forward(
        self, features: Float[Tensor, "... n_instances n_features"]
    ) -> tuple[
        Float[Tensor, "... n_instances n_features"],
        dict[
            str,
            Float[Tensor, "... n_instances n_features"] | Float[Tensor, "... n_instances n_hidden"],
        ],
        dict[
            str,
            Float[Tensor, "... n_instances n_features"] | Float[Tensor, "... n_instances n_hidden"],
        ],
    ]:
        # features: [..., instance, n_features]
        # W: [instance, n_features, n_hidden]
        hidden = torch.einsum("...if,ifh->...ih", features, self.W)

        pre_acts = {"W": features}
        post_acts = {"W": hidden}
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                pre_acts[f"hidden_{i}"] = hidden
                hidden = torch.einsum("...ik,ikj->...ij", hidden, layer)
                post_acts[f"hidden_{i}"] = hidden

        out_pre_relu = torch.einsum("...ih,ifh->...if", hidden, self.W) + self.b_final
        out = F.relu(out_pre_relu)

        pre_acts["W_T"] = hidden
        post_acts["W_T"] = out_pre_relu
        return out, pre_acts, post_acts

    def all_decomposable_params(self) -> dict[str, Float[Tensor, "n_instances d_in d_out"]]:
        """Dictionary of all parameters which will be decomposed with SPD."""
        params: dict[str, torch.Tensor] = {"W": self.W, "W_T": rearrange(self.W, "i f h -> i h f")}
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                params[f"hidden_{i}"] = layer
        return params

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> TMSModelPaths:
        """Download the relevant files from a wandb run."""
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)
        run_dir = fetch_wandb_run_dir(run.id)

        tms_model_config_path = download_wandb_file(run, run_dir, "tms_train_config.yaml")

        checkpoint = fetch_latest_wandb_checkpoint(run)
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)
        return TMSModelPaths(tms_train_config=tms_model_config_path, checkpoint=checkpoint_path)

    @classmethod
    def from_pretrained(cls, path: ModelPath) -> tuple["TMSModel", dict[str, Any]]:
        """Fetch a pretrained model from wandb or a local path to a checkpoint.

        Args:
            path: The path to local checkpoint or wandb project. If a wandb project, format must be
                `wandb:<entity>/<project>/<run_id>` or `wandb:<entity>/<project>/runs/<run_id>`.
                If `api.entity` is set (e.g. via setting WANDB_ENTITY in .env), <entity> can be
                omitted, and if `api.project` is set, <project> can be omitted. If local path,
                assumes that `resid_mlp_train_config.yaml` and `label_coeffs.json` are in the same
                directory as the checkpoint.

        Returns:
            model: The pretrained TMSModel
            tms_model_config_dict: The config dict used to train the model (we don't
                instantiate a train config due to circular import issues)
        """
        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
            paths = cls._download_wandb_files(wandb_path)
        else:
            # `path` should be a local path to a checkpoint
            paths = TMSModelPaths(
                tms_train_config=Path(path).parent / "tms_train_config.yaml",
                checkpoint=Path(path),
            )

        with open(paths.tms_train_config) as f:
            tms_train_config_dict = yaml.safe_load(f)

        tms_config = TMSModelConfig(**tms_train_config_dict["tms_model_config"])
        tms = cls(config=tms_config)
        params = torch.load(paths.checkpoint, weights_only=True, map_location="cpu")
        tms.load_state_dict(params)

        return tms, tms_train_config_dict


class TMSSPDFullRankModel(SPDFullRankModel):
    def __init__(
        self,
        n_instances: int,
        n_features: int,
        n_hidden: int,
        n_hidden_layers: int,
        k: int | None,
        bias_val: float,
        device: str = "cuda",
    ):
        super().__init__()
        self.n_instances = n_instances
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_hidden_layers = n_hidden_layers
        self.k = k if k is not None else n_features
        self.bias_val = bias_val

        self.subnetwork_params = nn.Parameter(
            torch.empty((n_instances, self.k, n_features, n_hidden), device=device)
        )

        bias_data = torch.zeros((n_instances, n_features), device=device) + bias_val
        self.b_final = nn.Parameter(bias_data)

        nn.init.xavier_normal_(self.subnetwork_params)

        self.hidden_layers = None
        if n_hidden_layers > 0:
            self.hidden_layers = nn.ModuleList(
                [
                    InstancesParamComponentsFullRank(
                        n_instances=n_instances,
                        in_dim=n_hidden,
                        out_dim=n_hidden,
                        k=self.k,
                        bias=False,
                        init_scale=1.0,
                    )
                    for _ in range(n_hidden_layers)
                ]
            )

    def all_subnetwork_params(
        self,
    ) -> dict[str, Float[Tensor, "n_instances k d_layer_in d_layer_out"]]:
        params: dict[str, Float[Tensor, "n_instances k d_layer_in d_layer_out"]] = {
            "W": self.subnetwork_params,
            "W_T": rearrange(self.subnetwork_params, "i k f h -> i k h f"),
        }
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                assert isinstance(layer, InstancesParamComponentsFullRank)
                params[f"hidden_{i}"] = layer.subnetwork_params
        return params

    def all_subnetwork_params_summed(
        self,
    ) -> dict[str, Float[Tensor, "n_instances d_layer_in d_layer_out"]]:
        """All subnetwork params summed over the subnetwork dimension."""
        summed_params: dict[str, Float[Tensor, "n_instances d_layer_in d_layer_out"]] = {
            p_name: p.sum(dim=-3) for p_name, p in self.all_subnetwork_params().items()
        }
        return summed_params

    def forward(
        self,
        x: Float[Tensor, "batch n_instances n_features"],
        topk_mask: Bool[Tensor, "batch n_instances k"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch n_instances n_features"],
        dict[str, Float[Tensor, "batch n_instances n_features"]],
        dict[str, Float[Tensor, "batch n_instances k n_features"]],
    ]:
        inner_acts = {}
        layer_acts = {}

        inner_acts["W"] = torch.einsum("...if,ikfh->...ikh", x, self.subnetwork_params)
        if topk_mask is not None:
            assert topk_mask.shape == inner_acts["W"].shape[:-1]
            inner_acts["W"] = torch.einsum("...ikh,...ik->...ikh", inner_acts["W"], topk_mask)
        layer_acts["W"] = torch.einsum("...ikh->...ih", inner_acts["W"])
        x = layer_acts["W"]

        # Hidden layers
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                assert isinstance(layer, InstancesParamComponentsFullRank)
                x, inner_acts[f"hidden_{i}"] = layer(x, topk_mask)
                layer_acts[f"hidden_{i}"] = x

        inner_acts["W_T"] = torch.einsum("...ih,ikfh->...ikf", x, self.subnetwork_params)
        if topk_mask is not None:
            assert topk_mask.shape == inner_acts["W_T"].shape[:-1]
            inner_acts["W_T"] = torch.einsum("...ikf,...ik->...ikf", inner_acts["W_T"], topk_mask)
        layer_acts["W_T"] = torch.einsum("...ikf->...if", inner_acts["W_T"]) + self.b_final

        out = F.relu(layer_acts["W_T"])
        return out, layer_acts, inner_acts

    @classmethod
    def from_pretrained(cls, path: str | RootPath) -> "TMSSPDFullRankModel":  # type: ignore
        pass

    def set_subnet_to_zero(
        self, subnet_idx: int
    ) -> dict[str, Float[Tensor, "n_instances n_features n_hidden"]]:
        stored_vals = {"W": self.subnetwork_params.data[:, subnet_idx, :, :].detach().clone()}
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                assert isinstance(layer, InstancesParamComponentsFullRank)
                stored_vals[f"hidden_{i}"] = (
                    layer.subnetwork_params.data[:, subnet_idx, :, :].detach().clone()
                )
                layer.subnetwork_params.data[:, subnet_idx, :, :] = 0.0
        self.subnetwork_params.data[:, subnet_idx, :, :] = 0.0
        return stored_vals

    def restore_subnet(
        self,
        subnet_idx: int,
        stored_vals: dict[str, Float[Tensor, "n_instances n_features n_hidden"]],
    ) -> None:
        self.subnetwork_params.data[:, subnet_idx, :, :] = stored_vals["W"]
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                assert isinstance(layer, InstancesParamComponentsFullRank)
                layer.subnetwork_params.data[:, subnet_idx, :, :] = stored_vals[f"hidden_{i}"]


class TMSSPDRankPenaltyPaths(BaseModel):
    """Paths to output files from a TMSSPDRankPenaltyModel training run."""

    final_config: Path
    tms_train_config: Path
    checkpoint: Path


class TMSSPDRankPenaltyModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    n_instances: PositiveInt
    n_features: PositiveInt
    n_hidden: PositiveInt
    n_hidden_layers: NonNegativeInt
    k: PositiveInt | None = None
    bias_val: float
    device: str
    m: PositiveInt | None = None


class TMSSPDRankPenaltyModel(SPDRankPenaltyModel):
    def __init__(self, config: TMSSPDRankPenaltyModelConfig):
        super().__init__()
        self.config = config
        self.n_instances = config.n_instances  # Required for backwards compatibility
        self.n_features = config.n_features  # Required for backwards compatibility
        self.k = config.k if config.k is not None else config.n_features
        self.bias_val = config.bias_val

        self.m = min(config.n_features, config.n_hidden) + 1 if config.m is None else config.m

        self.A = nn.Parameter(
            torch.empty(
                (config.n_instances, self.k, config.n_features, self.m), device=config.device
            )
        )
        self.B = nn.Parameter(
            torch.empty((config.n_instances, self.k, self.m, config.n_hidden), device=config.device)
        )

        bias_data = (
            torch.zeros((config.n_instances, config.n_features), device=config.device)
            + config.bias_val
        )
        self.b_final = nn.Parameter(bias_data)

        nn.init.xavier_normal_(self.A)
        nn.init.xavier_normal_(self.B)

        self.hidden_layers = None
        if config.n_hidden_layers > 0:
            self.hidden_layers = nn.ModuleList(
                [
                    InstancesParamComponentsRankPenalty(
                        n_instances=config.n_instances,
                        in_dim=config.n_hidden,
                        out_dim=config.n_hidden,
                        k=self.k,
                        bias=False,
                        init_scale=1.0,
                        m=self.m,
                    )
                    for _ in range(config.n_hidden_layers)
                ]
            )

    def all_subnetwork_params(self) -> dict[str, Float[Tensor, "n_instances k d_in d_out"]]:
        """Get all subnetwork parameters."""
        W = einops.einsum(
            self.A,
            self.B,
            "n_instances k n_features m, n_instances k m n_hidden -> n_instances k n_features n_hidden",
        )
        W_T = einops.rearrange(
            W, "n_instances k n_features n_hidden -> n_instances k n_hidden n_features"
        )
        params: dict[str, Float[Tensor, "n_instances k d_in d_out"]] = {"W": W, "W_T": W_T}
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                assert isinstance(layer, InstancesParamComponentsRankPenalty)
                params[f"hidden_{i}"] = einops.einsum(
                    layer.A,
                    layer.B,
                    "n_instances k d_in m, n_instances k m d_out -> n_instances k d_in d_out",
                )
        return params

    def all_subnetwork_params_summed(self) -> dict[str, Float[Tensor, "n_instances d_in d_out"]]:
        """All subnetwork params summed over the subnetwork dimension."""
        summed_params: dict[str, Float[Tensor, "n_instances d_in d_out"]] = {
            p_name: p.sum(dim=1) for p_name, p in self.all_subnetwork_params().items()
        }
        return summed_params

    def forward(
        self,
        x: Float[Tensor, "batch n_instances n_features"],
        topk_mask: Bool[Tensor, "batch n_instances k"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch n_instances n_features"],
        dict[str, Float[Tensor, "batch n_instances n_features"]],
        dict[str, Float[Tensor, "batch n_instances k n_features"]],
    ]:
        layer_acts = {}
        inner_acts = {}

        # First layer/embedding: x -> A -> m dimension -> B -> hidden
        pre_inner_act_0 = torch.einsum("bif,ikfm->bikm", x, self.A)
        if topk_mask is not None:
            assert topk_mask.shape == pre_inner_act_0.shape[:-1]
            pre_inner_act_0 = torch.einsum("bikm,bik->bikm", pre_inner_act_0, topk_mask)
        inner_acts["W"] = torch.einsum("bikm,ikmh->bikh", pre_inner_act_0, self.B)
        layer_acts["W"] = torch.einsum("bikh->bih", inner_acts["W"])
        x = layer_acts["W"]

        # Hidden layers
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                assert isinstance(layer, InstancesParamComponentsRankPenalty)
                x, hidden_inner_act_i = layer(x, topk_mask)
                layer_acts[f"hidden_{i}"] = x
                inner_acts[f"hidden_{i}"] = hidden_inner_act_i

        # Second layer/unembedding: hidden -> B.T -> m dimension -> A.T -> features
        pre_inner_act_1 = torch.einsum("bih,ikmh->bikm", x, self.B)
        if topk_mask is not None:
            assert topk_mask.shape == pre_inner_act_1.shape[:-1]
            pre_inner_act_1 = torch.einsum("bikm,bik->bikm", pre_inner_act_1, topk_mask)
        inner_acts["W_T"] = torch.einsum("bikm,ikfm->bikf", pre_inner_act_1, self.A)
        layer_acts["W_T"] = torch.einsum("bikf->bif", inner_acts["W_T"]) + self.b_final

        out = F.relu(layer_acts["W_T"])
        return out, layer_acts, inner_acts

    def set_subnet_to_zero(
        self, subnet_idx: int
    ) -> dict[str, Float[Tensor, "n_instances in_dim m"] | Float[Tensor, "n_instances m out_dim"]]:
        stored_vals = {
            "A": self.A.data[:, subnet_idx, :, :].detach().clone(),
            "B": self.B.data[:, subnet_idx, :, :].detach().clone(),
        }
        self.A.data[:, subnet_idx, :, :] = 0.0
        self.B.data[:, subnet_idx, :, :] = 0.0
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                assert isinstance(layer, InstancesParamComponentsRankPenalty)
                stored_vals[f"hidden_{i}_A"] = layer.A.data[:, subnet_idx, :, :].detach().clone()
                stored_vals[f"hidden_{i}_B"] = layer.B.data[:, subnet_idx, :, :].detach().clone()
                layer.A.data[:, subnet_idx, :, :] = 0.0
                layer.B.data[:, subnet_idx, :, :] = 0.0

        return stored_vals

    def restore_subnet(
        self,
        subnet_idx: int,
        stored_vals: dict[
            str, Float[Tensor, "n_instances in_dim m"] | Float[Tensor, "n_instances m out_dim"]
        ],
    ) -> None:
        self.A.data[:, subnet_idx, :, :] = stored_vals["A"]
        self.B.data[:, subnet_idx, :, :] = stored_vals["B"]
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                assert isinstance(layer, InstancesParamComponentsRankPenalty)
                layer.A.data[:, subnet_idx, :, :] = stored_vals[f"hidden_{i}_A"]
                layer.B.data[:, subnet_idx, :, :] = stored_vals[f"hidden_{i}_B"]

    def all_As_and_Bs(
        self,
    ) -> dict[
        str, tuple[Float[Tensor, "n_instances k d_in m"], Float[Tensor, "n_instances k m d_out"]]
    ]:
        """Get all A and B matrices. Note that this won't return bias components."""
        params: dict[
            str,
            tuple[Float[Tensor, "n_instances k d_in m"], Float[Tensor, "n_instances k m d_out"]],
        ] = {
            "W": (self.A, self.B),
            "W_T": (
                rearrange(self.B, "i k m h -> i k h m"),
                rearrange(self.A, "i k f m -> i k m f"),
            ),
        }
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                assert isinstance(layer, InstancesParamComponentsRankPenalty)
                params[f"hidden_{i}"] = (layer.A, layer.B)
        return params

    def set_matrices_to_unit_norm(self) -> None:
        """Set the matrices that need to be normalized to unit norm."""
        self.A.data /= self.A.data.norm(p=2, dim=-2, keepdim=True)
        if self.hidden_layers is not None:
            for layer in self.hidden_layers:
                assert isinstance(layer, InstancesParamComponentsRankPenalty)
                layer.A.data /= layer.A.data.norm(p=2, dim=-2, keepdim=True)

    def fix_normalized_adam_gradients(self) -> None:
        """Modify the gradient by subtracting it's component parallel to the activation."""
        assert self.A.grad is not None
        remove_grad_parallel_to_subnetwork_vecs(self.A.data, self.A.grad)
        if self.hidden_layers is not None:
            for layer in self.hidden_layers:
                assert isinstance(layer, InstancesParamComponentsRankPenalty)
                assert layer.A.grad is not None
                remove_grad_parallel_to_subnetwork_vecs(layer.A.data, layer.A.grad)

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> TMSSPDRankPenaltyPaths:
        """Download the relevant files from a wandb run."""
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)

        checkpoint = fetch_latest_wandb_checkpoint(run, prefix="spd_model")

        run_dir = fetch_wandb_run_dir(run.id)

        final_config_path = download_wandb_file(run, run_dir, "final_config.yaml")
        tms_train_config_path = download_wandb_file(run, run_dir, "tms_train_config.yaml")
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)
        return TMSSPDRankPenaltyPaths(
            final_config=final_config_path,
            tms_train_config=tms_train_config_path,
            checkpoint=checkpoint_path,
        )

    @classmethod
    def from_pretrained(cls, path: ModelPath) -> tuple["TMSSPDRankPenaltyModel", Config]:
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
            paths = TMSSPDRankPenaltyPaths(
                final_config=Path(path).parent / "final_config.yaml",
                tms_train_config=Path(path).parent / "tms_train_config.yaml",
                checkpoint=Path(path),
            )

        with open(paths.final_config) as f:
            final_config_dict = yaml.safe_load(f)
        spd_config = Config(**final_config_dict)

        with open(paths.tms_train_config) as f:
            tms_train_config_dict = yaml.safe_load(f)

        assert isinstance(spd_config.task_config, TMSTaskConfig)
        tms_spd_rank_penalty_config = TMSSPDRankPenaltyModelConfig(
            **tms_train_config_dict["tms_model_config"],
            k=spd_config.task_config.k,
            m=spd_config.m,
            bias_val=spd_config.task_config.bias_val,
        )
        model = cls(config=tms_spd_rank_penalty_config)
        params = torch.load(paths.checkpoint, weights_only=True, map_location="cpu")
        model.load_state_dict(params)
        return model, spd_config
