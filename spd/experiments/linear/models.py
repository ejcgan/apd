from pathlib import Path

import einops
import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from spd.models.base import Model, SPDFullRankModel, SPDModel
from spd.utils import remove_grad_parallel_to_subnetwork_vecs


class DeepLinearModel(Model):
    def __init__(self, n_features: int, n_layers: int, n_instances: int):
        super().__init__()
        self.n_features = n_features
        self.n_layers = n_layers
        self.n_instances = n_instances
        self.layers = nn.ParameterList(
            [
                nn.Parameter(torch.randn(n_instances, n_features, n_features))
                for _ in range(n_layers)
            ]
        )

        for layer in self.layers:
            nn.init.kaiming_normal_(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.einsum("bif,ifj->bij", x, layer)
        return x

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "DeepLinearModel":
        params = torch.load(path, weights_only=True, map_location="cpu")
        # Get the n_features, n_layers, n_instances from the params
        n_layers = len(params.keys())
        n_features = params["layers.0"].shape[1]
        n_instances = params["layers.0"].shape[0]
        model = cls(n_features, n_layers, n_instances)
        model.load_state_dict(params)
        return model

    def all_decomposable_params(
        self,
    ) -> dict[str, Float[Tensor, "n_instances n_features n_features"]]:
        """Dictionary of all parameters which will be decomposed with SPD."""
        return {f"layer_{i}": layer for i, layer in enumerate(self.layers)}


class DeepLinearParamComponents(nn.Module):
    def __init__(self, n_instances: int, n_features: int, k: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(n_instances, n_features, k))
        self.B = nn.Parameter(torch.empty(n_instances, k, n_features))

    def forward(
        self,
        x: Float[Tensor, "batch n_instances n_features"],
    ) -> tuple[Float[Tensor, "batch n_instances n_features"], Float[Tensor, "batch n_instances k"]]:
        inner_acts = torch.einsum("bif,ifk->bik", x, self.A)
        out = torch.einsum("bik,ikg->big", inner_acts, self.B)
        return out, inner_acts

    def forward_topk(
        self,
        x: Float[Tensor, "batch n_instances n_features"],
        topk_mask: Bool[Tensor, "batch n_instances k"],
    ) -> tuple[Float[Tensor, "batch n_instances n_features"], Float[Tensor, "batch n_instances k"]]:
        """Performs a forward pass using only the top-k subnetwork activations."""
        inner_acts = torch.einsum("bif,ifk->bik", x, self.A)
        inner_acts_topk = inner_acts * topk_mask
        out = torch.einsum("bik,ikg->big", inner_acts_topk, self.B)
        return out, inner_acts_topk


class DeepLinearComponentModel(SPDModel):
    def __init__(
        self,
        n_features: int,
        n_layers: int,
        n_instances: int,
        k: int | None,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_layers = n_layers
        self.n_param_matrices = n_layers
        self.n_instances = n_instances
        self.k = k if k is not None else n_features
        self.layers = nn.ModuleList(
            [
                DeepLinearParamComponents(n_instances=n_instances, n_features=n_features, k=self.k)
                for _ in range(n_layers)
            ]
        )

        for param in self.layers.parameters():
            nn.init.kaiming_normal_(param)

    def all_As_and_Bs(
        self,
    ) -> dict[
        str,
        tuple[Float[Tensor, "n_instances n_features k"], Float[Tensor, "n_instances k n_features"]],
    ]:
        return {f"layer_{i}": (layer.A, layer.B) for i, layer in enumerate(self.layers)}

    def all_subnetwork_params(
        self,
    ) -> dict[str, Float[Tensor, "n_instances k n_features n_features"]]:
        return {
            f"layer_{i}": torch.einsum("ifk,ikh->ikfh", layer.A, layer.B)
            for i, layer in enumerate(self.layers)
        }

    def all_subnetwork_params_summed(
        self,
    ) -> dict[str, Float[Tensor, "n_instances n_features n_features"]]:
        return {
            f"layer_{i}": torch.einsum("ifk,ikh->ifh", layer.A, layer.B)
            for i, layer in enumerate(self.layers)
        }

    def forward(
        self,
        x: Float[Tensor, "batch n_instances n_features"],
    ) -> tuple[
        Float[Tensor, "batch n_instances n_features"],
        dict[str, Float[Tensor, "batch n_instances n_features"]],
        dict[str, Float[Tensor, "batch n_instances k"]],
    ]:
        layer_acts = {}
        inner_acts = {}
        for i, layer in enumerate(self.layers):
            x, inner_act = layer(x)
            layer_acts[f"layer_{i}"] = x
            inner_acts[f"layer_{i}"] = inner_act
        return x, layer_acts, inner_acts

    def forward_topk(
        self,
        x: Float[Tensor, "batch n_instances n_features"],
        topk_mask: Bool[Tensor, "batch n_instances k"],
    ) -> tuple[
        Float[Tensor, "batch n_instances n_features"],
        dict[str, Float[Tensor, "batch n_instances n_features"]],
        dict[str, Float[Tensor, "batch n_instances k"]],
    ]:
        """Performs a forward pass using only the top-k subnetwork activations."""
        layer_acts = {}
        inner_acts_topk = {}
        for i, layer in enumerate(self.layers):
            assert isinstance(layer, DeepLinearParamComponents)
            x, inner_act_topk = layer.forward_topk(x, topk_mask)
            layer_acts[f"layer_{i}"] = x
            inner_acts_topk[f"layer_{i}"] = inner_act_topk
        return x, layer_acts, inner_acts_topk

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "DeepLinearComponentModel":
        params = torch.load(path, weights_only=True, map_location="cpu")
        n_layers = len(params) // 2
        for param in params:
            assert param.startswith("layers.") and param.endswith(("A", "B"))
        n_instances, n_features, k = params["layers.0.A"].shape

        model = cls(n_features=n_features, n_layers=n_layers, n_instances=n_instances, k=k)
        model.load_state_dict(params)
        return model

    def set_matrices_to_unit_norm(self):
        for layer in self.layers:
            layer.A.data /= layer.A.data.norm(p=2, dim=-2, keepdim=True)

    def fix_normalized_adam_gradients(self):
        for layer in self.layers:
            remove_grad_parallel_to_subnetwork_vecs(layer.A.data, layer.A.grad)

    def set_subnet_to_zero(self, subnet_idx: int) -> dict[str, Float[Tensor, "n_instances dim"]]:
        stored_vals = {}
        for i, layer in enumerate(self.layers):
            stored_vals[f"layer_{i}_A"] = layer.A.data[:, :, subnet_idx].detach().clone()
            stored_vals[f"layer_{i}_B"] = layer.B.data[:, subnet_idx, :].detach().clone()
            layer.A.data[:, :, subnet_idx] = 0.0
            layer.B.data[:, subnet_idx, :] = 0.0
        return stored_vals

    def restore_subnet(
        self, subnet_idx: int, stored_vals: dict[str, Float[Tensor, "n_instances dim"]]
    ) -> None:
        for i, layer in enumerate(self.layers):
            layer.A.data[:, :, subnet_idx] = stored_vals[f"layer_{i}_A"]
            layer.B.data[:, subnet_idx, :] = stored_vals[f"layer_{i}_B"]


class DeepLinearParamComponentsFullRank(nn.Module):
    def __init__(self, n_instances: int, n_features: int, k: int):
        super().__init__()
        self.subnetwork_params = nn.Parameter(torch.empty(n_instances, k, n_features, n_features))

    def forward(
        self,
        x: Float[Tensor, "batch n_instances n_features"],
    ) -> tuple[
        Float[Tensor, "batch n_instances n_features"],
        Float[Tensor, "batch n_instances k n_features"],
    ]:
        inner_acts = einops.einsum(
            x,
            self.subnetwork_params,
            "batch n_instances dim1, n_instances k dim1 dim2 -> batch n_instances k dim2",
        )
        out = einops.einsum(inner_acts, "batch n_instances k dim2 -> batch n_instances dim2")
        return out, inner_acts

    def forward_topk(
        self,
        x: Float[Tensor, "batch n_instances n_features"],
        topk_mask: Bool[Tensor, "batch n_instances k"],
    ) -> tuple[
        Float[Tensor, "batch n_instances n_features"],
        Float[Tensor, "batch n_instances k n_features"],
    ]:
        """Performs a forward pass using only the top-k subnetwork activations."""
        inner_acts = einops.einsum(
            x,
            self.subnetwork_params,
            "batch n_instances dim1, n_instances k dim1 dim2 -> batch n_instances k dim2",
        )
        inner_acts_topk = einops.einsum(
            inner_acts,
            topk_mask,
            "batch n_instances k dim2, batch n_instances k -> batch n_instances k dim2",
        )
        out = einops.einsum(inner_acts_topk, "batch n_instances k dim2 -> batch n_instances dim2")
        return out, inner_acts_topk


class DeepLinearComponentFullRankModel(SPDFullRankModel):
    def __init__(
        self,
        n_features: int,
        n_layers: int,
        n_instances: int,
        k: int | None,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_layers = n_layers
        self.n_param_matrices = n_layers
        self.n_instances = n_instances
        self.k = k if k is not None else n_features
        self.layers = nn.ModuleList(
            [
                DeepLinearParamComponentsFullRank(
                    n_instances=n_instances, n_features=n_features, k=self.k
                )
                for _ in range(n_layers)
            ]
        )

        for param in self.layers.parameters():
            nn.init.kaiming_normal_(param)

    def all_subnetwork_params(
        self,
    ) -> dict[str, Float[Tensor, "n_instances k n_features n_features"]]:
        return {f"layer_{i}": layer.subnetwork_params for i, layer in enumerate(self.layers)}

    def all_subnetwork_params_summed(
        self,
    ) -> dict[str, Float[Tensor, "n_instances n_features n_features"]]:
        return {
            f"layer_{i}": layer.subnetwork_params.sum(dim=-3) for i, layer in enumerate(self.layers)
        }

    def forward(
        self,
        x: Float[Tensor, "batch n_instances n_features"],
    ) -> tuple[
        Float[Tensor, "batch n_instances n_features"],
        dict[str, Float[Tensor, "batch n_instances n_features"]],
        dict[str, Float[Tensor, "batch n_instances k n_features"]],
    ]:
        layer_acts = {}
        inner_acts = {}
        for i, layer in enumerate(self.layers):
            x, inner_act = layer(x)
            layer_acts[f"layer_{i}"] = x
            inner_acts[f"layer_{i}"] = inner_act
        return x, layer_acts, inner_acts

    def forward_topk(
        self,
        x: Float[Tensor, "batch n_instances n_features"],
        topk_mask: Bool[Tensor, "batch n_instances k"],
    ) -> tuple[
        Float[Tensor, "batch n_instances n_features"],
        dict[str, Float[Tensor, "batch n_instances n_features"]],
        dict[str, Float[Tensor, "batch n_instances k n_features"]],
    ]:
        """Performs a forward pass using only the top-k subnetwork activations."""
        layer_acts = {}
        inner_acts_topk = {}
        for i, layer in enumerate(self.layers):
            assert isinstance(layer, DeepLinearParamComponentsFullRank)
            x, inner_act_topk = layer.forward_topk(x, topk_mask)
            layer_acts[f"layer_{i}"] = x
            inner_acts_topk[f"layer_{i}"] = inner_act_topk
        return x, layer_acts, inner_acts_topk

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "DeepLinearComponentFullRankModel":
        params = torch.load(path, weights_only=True, map_location="cpu")
        n_layers = len(params) // 2
        for param in params:
            assert param.startswith("layers.") and param.endswith("subnetwork_params")
        n_instances, n_features, k = params["layers.0.subnetwork_params"].shape

        model = cls(n_features=n_features, n_layers=n_layers, n_instances=n_instances, k=k)
        model.load_state_dict(params)
        return model

    def set_subnet_to_zero(
        self, subnet_idx: int
    ) -> dict[str, Float[Tensor, "n_instances d_in d_out"]]:
        stored_vals = {}
        for i, layer in enumerate(self.layers):
            stored_vals[f"layer_{i}"] = layer.subnetwork_params.data[:, subnet_idx].detach().clone()
            layer.subnetwork_params.data[:, subnet_idx, :, :] = 0.0
        return stored_vals

    def restore_subnet(
        self, subnet_idx: int, stored_vals: dict[str, Float[Tensor, "n_instances d_in d_out"]]
    ) -> None:
        for i, layer in enumerate(self.layers):
            layer.subnetwork_params.data[:, subnet_idx, :, :] = stored_vals[f"layer_{i}"]
