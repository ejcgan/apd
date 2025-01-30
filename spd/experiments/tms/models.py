from pathlib import Path
from typing import Any

import einops
import torch
import wandb
import yaml
from jaxtyping import Bool, Float
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt
from torch import Tensor, nn
from torch.nn import functional as F
from wandb.apis.public import Run

from spd.hooks import HookedRootModule, HookPoint
from spd.models.base import SPDModel
from spd.models.components import Linear, LinearComponent
from spd.run_spd import Config, TMSTaskConfig
from spd.types import WANDB_PATH_PREFIX, ModelPath
from spd.utils import (
    handle_deprecated_config_keys,
    remove_grad_parallel_to_subnetwork_vecs,
    replace_deprecated_param_names,
)
from spd.wandb_utils import download_wandb_file, fetch_latest_wandb_checkpoint, fetch_wandb_run_dir


class TransposedLinear(Linear):
    """Linear layer that uses a transposed weight from another Linear layer.

    We use 'd_in' and 'd_out' to refer to the dimensions of the original Linear layer.
    """

    def __init__(self, original_weight: nn.Parameter):
        # Copy the relevant parts from Linear.__init__. Don't copy operations that will call
        # TransposedLinear.weight.
        nn.Module.__init__(self)
        self.hook_pre = HookPoint()  # (batch ... d_out)
        self.hook_post = HookPoint()  # (batch ... d_in)

        self.register_buffer("original_weight", original_weight, persistent=False)

    @property
    def weight(self) -> Float[Tensor, "n_instances d_out d_in"]:
        return einops.rearrange(
            self.original_weight, "n_instances d_in d_out -> n_instances d_out d_in"
        )


class TransposedLinearComponent(LinearComponent):
    """LinearComponent that uses a transposed weight from another LinearComponent.

    We use 'd_in' and 'd_out' to refer to the dimensions of the original LinearComponent.
    """

    def __init__(self, original_A: nn.Parameter, original_B: nn.Parameter):
        # Copy the relevant parts from LinearComponent.__init__. Don't copy operations that will
        # call TransposedLinear.A or TransposedLinear.B.
        nn.Module.__init__(self)
        self.n_instances, self.k, _, self.m = original_A.shape

        self.hook_pre = HookPoint()  # (batch ... d_out)
        self.hook_component_acts = HookPoint()  # (batch ... k d_in)
        self.hook_post = HookPoint()  # (batch ... d_in)

        self.register_buffer("original_A", original_A, persistent=False)
        self.register_buffer("original_B", original_B, persistent=False)

    @property
    def A(self) -> Float[Tensor, "n_instances k d_out m"]:
        # New A is the transpose of the original B
        return einops.rearrange(
            self.original_B,
            "n_instances k m d_out -> n_instances k d_out m",
        )

    @property
    def B(self) -> Float[Tensor, "n_instances k d_in m"]:
        # New B is the transpose of the original A
        return einops.rearrange(
            self.original_A,
            "n_instances k d_in m -> n_instances k m d_in",
        )

    @property
    def component_weights(self) -> Float[Tensor, "... k d_out d_in"]:
        """A @ B before summing over the subnetwork dimension."""
        return einops.einsum(self.A, self.B, "... k d_out m, ... k m d_in -> ... k d_out d_in")

    @property
    def weight(self) -> Float[Tensor, "... d_out d_in"]:
        """A @ B after summing over the subnetwork dimension."""
        return einops.einsum(self.A, self.B, "... k d_out m, ... k m d_in -> ... d_out d_in")


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


def _tms_forward(
    x: Float[Tensor, "batch n_instances n_features"],
    linear1: Linear | LinearComponent,
    linear2: TransposedLinear | TransposedLinearComponent,
    b_final: Float[Tensor, "n_instances n_features"],
    topk_mask: Bool[Tensor, "batch n_instances k"] | None = None,
    hidden_layers: nn.ModuleList | None = None,
) -> Float[Tensor, "batch n_instances n_features"]:
    """Forward pass used for TMSModel and TMSSPDModel.

    Note that topk_mask is only used for TMSSPDModel.
    """
    hidden = linear1(x, topk_mask=topk_mask)
    if hidden_layers is not None:
        for layer in hidden_layers:
            hidden = layer(hidden, topk_mask=topk_mask)
    out_pre_relu = linear2(hidden, topk_mask=topk_mask) + b_final
    out = F.relu(out_pre_relu)
    return out


class TMSModel(HookedRootModule):
    def __init__(self, config: TMSModelConfig):
        super().__init__()
        self.config = config

        self.linear1 = Linear(
            d_in=config.n_features,
            d_out=config.n_hidden,
            n_instances=config.n_instances,
            init_type="xavier_normal",
        )
        # Use tied weights for the second linear layer
        self.linear2 = TransposedLinear(self.linear1.weight)

        self.b_final = nn.Parameter(torch.zeros((config.n_instances, config.n_features)))

        self.hidden_layers = None
        if config.n_hidden_layers > 0:
            self.hidden_layers = nn.ModuleList()
            for _ in range(config.n_hidden_layers):
                layer = Linear(
                    d_in=config.n_hidden,
                    d_out=config.n_hidden,
                    n_instances=config.n_instances,
                    init_type="xavier_normal",
                )
                self.hidden_layers.append(layer)
        self.setup()

    def forward(
        self, x: Float[Tensor, "... n_instances n_features"], **_: Any
    ) -> Float[Tensor, "... n_instances n_features"]:
        return _tms_forward(
            x=x,
            linear1=self.linear1,
            linear2=self.linear2,
            b_final=self.b_final,
            hidden_layers=self.hidden_layers,
        )

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
        params = replace_deprecated_param_names(params, {"W": "linear1.weight"})
        tms.load_state_dict(params)

        return tms, tms_train_config_dict


class TMSSPDPaths(BaseModel):
    """Paths to output files from a TMSSPDModel training run."""

    final_config: Path
    tms_train_config: Path
    checkpoint: Path


class TMSSPDModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    n_instances: PositiveInt
    n_features: PositiveInt
    n_hidden: PositiveInt
    n_hidden_layers: NonNegativeInt
    k: PositiveInt | None = None
    bias_val: float
    device: str
    m: PositiveInt | None = None


class TMSSPDModel(SPDModel):
    def __init__(self, config: TMSSPDModelConfig):
        super().__init__()
        self.config = config
        self.n_instances = config.n_instances  # Required for backwards compatibility
        self.n_features = config.n_features  # Required for backwards compatibility
        self.k = config.k if config.k is not None else config.n_features
        self.bias_val = config.bias_val

        self.m = min(config.n_features, config.n_hidden) + 1 if config.m is None else config.m

        self.linear1 = LinearComponent(
            d_in=config.n_features,
            d_out=config.n_hidden,
            n_instances=config.n_instances,
            init_type="xavier_normal",
            init_scale=1.0,
            k=self.k,
            m=self.m,
        )
        self.linear2 = TransposedLinearComponent(self.linear1.A, self.linear1.B)

        bias_data = (
            torch.zeros((config.n_instances, config.n_features), device=config.device)
            + config.bias_val
        )
        self.b_final = nn.Parameter(bias_data)

        self.hidden_layers = None
        if config.n_hidden_layers > 0:
            self.hidden_layers = nn.ModuleList(
                [
                    LinearComponent(
                        d_in=config.n_hidden,
                        d_out=config.n_hidden,
                        n_instances=config.n_instances,
                        init_type="xavier_normal",
                        init_scale=1.0,
                        k=self.k,
                        m=self.m,
                    )
                    for _ in range(config.n_hidden_layers)
                ]
            )

        self.setup()

    def all_component_weights(self) -> dict[str, Float[Tensor, "n_instances k d_in d_out"]]:
        """Get all component weights (i.e. A @ B in every layer)."""
        params: dict[str, Float[Tensor, "n_instances k d_in d_out"]] = {}
        params["linear1"] = self.linear1.component_weights
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                assert isinstance(layer, LinearComponent)
                params[f"hidden_layers.{i}"] = layer.component_weights
        params["linear2"] = self.linear2.component_weights
        return params

    def forward(
        self,
        x: Float[Tensor, "batch n_instances n_features"],
        topk_mask: Bool[Tensor, "batch n_instances k"] | None = None,
    ) -> Float[Tensor, "batch n_instances n_features"]:
        return _tms_forward(
            x=x,
            linear1=self.linear1,
            linear2=self.linear2,
            b_final=self.b_final,
            hidden_layers=self.hidden_layers,
            topk_mask=topk_mask,
        )

    def set_subnet_to_zero(
        self, subnet_idx: int
    ) -> dict[str, Float[Tensor, "n_instances in_dim m"] | Float[Tensor, "n_instances m out_dim"]]:
        # Only need to set the values for linear1 to zero, since linear2 references the same params
        stored_vals = {
            "linear1.A": self.linear1.A.data[:, subnet_idx, :, :].detach().clone(),
            "linear1.B": self.linear1.B.data[:, subnet_idx, :, :].detach().clone(),
        }
        self.linear1.A.data[:, subnet_idx, :, :] = 0.0
        self.linear1.B.data[:, subnet_idx, :, :] = 0.0
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                assert isinstance(layer, LinearComponent)
                stored_vals[f"hidden_layers.{i}.A"] = (
                    layer.A.data[:, subnet_idx, :, :].detach().clone()
                )
                stored_vals[f"hidden_layers.{i}.B"] = (
                    layer.B.data[:, subnet_idx, :, :].detach().clone()
                )
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
        self.linear1.A.data[:, subnet_idx, :, :] = stored_vals["linear1.A"]
        self.linear1.B.data[:, subnet_idx, :, :] = stored_vals["linear1.B"]
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                assert isinstance(layer, LinearComponent)
                layer.A.data[:, subnet_idx, :, :] = stored_vals[f"hidden_layers.{i}.A"]
                layer.B.data[:, subnet_idx, :, :] = stored_vals[f"hidden_layers.{i}.B"]

    def all_As_and_Bs(
        self,
    ) -> dict[
        str, tuple[Float[Tensor, "n_instances k d_in m"], Float[Tensor, "n_instances k m d_out"]]
    ]:
        """Get all A and B matrices. Note that this won't return bias components."""
        params = {
            "linear1": (self.linear1.A, self.linear1.B),
            "linear2": (self.linear2.A, self.linear2.B),
        }
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                assert isinstance(layer, LinearComponent)
                params[f"hidden_layers.{i}"] = (layer.A, layer.B)
        return params

    def set_matrices_to_unit_norm(self) -> None:
        """Set the A matrices to unit norm for stability."""
        self.linear1.A.data /= self.linear1.A.data.norm(p=2, dim=-2, keepdim=True)
        if self.hidden_layers is not None:
            for layer in self.hidden_layers:
                assert isinstance(layer, LinearComponent)
                layer.A.data /= layer.A.data.norm(p=2, dim=-2, keepdim=True)

    def fix_normalized_adam_gradients(self) -> None:
        """Modify the gradient by subtracting it's component parallel to the activation."""
        assert self.linear1.A.grad is not None
        remove_grad_parallel_to_subnetwork_vecs(self.linear1.A.data, self.linear1.A.grad)
        if self.hidden_layers is not None:
            for layer in self.hidden_layers:
                assert isinstance(layer, LinearComponent)
                assert layer.A.grad is not None
                remove_grad_parallel_to_subnetwork_vecs(layer.A.data, layer.A.grad)

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> TMSSPDPaths:
        """Download the relevant files from a wandb run."""
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)

        checkpoint = fetch_latest_wandb_checkpoint(run, prefix="spd_model")

        run_dir = fetch_wandb_run_dir(run.id)

        final_config_path = download_wandb_file(run, run_dir, "final_config.yaml")
        tms_train_config_path = download_wandb_file(run, run_dir, "tms_train_config.yaml")
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)
        return TMSSPDPaths(
            final_config=final_config_path,
            tms_train_config=tms_train_config_path,
            checkpoint=checkpoint_path,
        )

    @classmethod
    def from_pretrained(cls, path: ModelPath) -> tuple["TMSSPDModel", Config]:
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
            paths = TMSSPDPaths(
                final_config=Path(path).parent / "final_config.yaml",
                tms_train_config=Path(path).parent / "tms_train_config.yaml",
                checkpoint=Path(path),
            )

        with open(paths.final_config) as f:
            final_config_dict = yaml.safe_load(f)

        final_config_dict = handle_deprecated_config_keys(final_config_dict)
        spd_config = Config(**final_config_dict)

        with open(paths.tms_train_config) as f:
            tms_train_config_dict = yaml.safe_load(f)

        assert isinstance(spd_config.task_config, TMSTaskConfig)
        tms_spd_config = TMSSPDModelConfig(
            **tms_train_config_dict["tms_model_config"],
            k=spd_config.task_config.k,
            m=spd_config.m,
            bias_val=spd_config.task_config.bias_val,
        )
        model = cls(config=tms_spd_config)
        params = torch.load(paths.checkpoint, weights_only=True, map_location="cpu")
        params = replace_deprecated_param_names(params, {"A": "linear1.A", "B": "linear1.B"})
        model.load_state_dict(params)
        return model, spd_config
