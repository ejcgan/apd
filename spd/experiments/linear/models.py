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

    def all_decomposable_params(self) -> list[Float[Tensor, "..."]]:
        """List of all parameters which will be decomposed with SPD."""
        return [layer for layer in self.layers]


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

    def all_As(self) -> list[Float[Tensor, "n_instances n_features k"]]:
        return [layer.A for layer in self.layers]

    def all_Bs(self) -> list[Float[Tensor, "n_instances k n_features"]]:
        return [layer.B for layer in self.layers]

    def forward(
        self,
        x: Float[Tensor, "batch n_instances n_features"],
    ) -> tuple[
        Float[Tensor, "batch n_instances n_features"],
        list[Float[Tensor, "batch n_instances n_features"]],
        list[Float[Tensor, "batch n_instances k"]],
    ]:
        layer_acts = []
        inner_acts = []
        for layer in self.layers:
            x, inner_act = layer(x)
            layer_acts.append(x)
            inner_acts.append(inner_act)
        return x, layer_acts, inner_acts

    def forward_topk(
        self,
        x: Float[Tensor, "batch n_instances n_features"],
        topk_mask: Bool[Tensor, "batch n_instances k"],
    ) -> tuple[
        Float[Tensor, "batch n_instances n_features"],
        list[Float[Tensor, "batch n_instances n_features"]],
        list[Float[Tensor, "batch n_instances k"]],
    ]:
        """Performs a forward pass using only the top-k subnetwork activations."""
        layer_acts = []
        inner_acts_topk = []
        for layer in self.layers:
            assert isinstance(layer, DeepLinearParamComponents)
            x, inner_act_topk = layer.forward_topk(x, topk_mask)
            layer_acts.append(x)
            inner_acts_topk.append(inner_act_topk)
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

    def all_subnetwork_params(self) -> list[Float[Tensor, "k n_features n_features"]]:
        return [layer.subnetwork_params for layer in self.layers]

    def forward(
        self,
        x: Float[Tensor, "batch n_instances n_features"],
    ) -> tuple[
        Float[Tensor, "batch n_instances n_features"],
        list[Float[Tensor, "batch n_instances n_features"]],
        list[Float[Tensor, "batch n_instances k n_features"]],
    ]:
        layer_acts = []
        inner_acts = []
        for layer in self.layers:
            x, inner_act = layer(x)
            layer_acts.append(x)
            inner_acts.append(inner_act)
        return x, layer_acts, inner_acts

    def forward_topk(
        self,
        x: Float[Tensor, "batch n_instances n_features"],
        topk_mask: Bool[Tensor, "batch n_instances k"],
    ) -> tuple[
        Float[Tensor, "batch n_instances n_features"],
        list[Float[Tensor, "batch n_instances n_features"]],
        list[Float[Tensor, "batch n_instances k n_features"]],
    ]:
        """Performs a forward pass using only the top-k subnetwork activations."""
        layer_acts = []
        inner_acts_topk = []
        for layer in self.layers:
            assert isinstance(layer, DeepLinearParamComponentsFullRank)
            x, inner_act_topk = layer.forward_topk(x, topk_mask)
            layer_acts.append(x)
            inner_acts_topk.append(inner_act_topk)
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
