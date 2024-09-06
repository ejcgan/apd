import torch
from einops import rearrange
from jaxtyping import Bool, Float
from torch import Tensor, nn
from torch.nn import functional as F

from spd.models.base import Model, SPDFullRankModel, SPDModel
from spd.types import RootPath
from spd.utils import remove_grad_parallel_to_subnetwork_vecs


class TMSModel(Model):
    def __init__(
        self,
        n_instances: int,
        n_features: int,
        n_hidden: int,
        device: str = "cuda",
    ):
        super().__init__()
        self.n_instances = n_instances
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.W = nn.Parameter(torch.empty((n_instances, n_features, n_hidden), device=device))
        nn.init.xavier_normal_(self.W)
        self.b_final = nn.Parameter(torch.zeros((n_instances, n_features), device=device))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [..., instance, n_features]
        # W: [instance, n_features, n_hidden]
        hidden = torch.einsum("...if,ifh->...ih", features, self.W)
        out = torch.einsum("...ih,ifh->...if", hidden, self.W)
        out = out + self.b_final
        out = F.relu(out)
        return out

    def all_decomposable_params(self) -> list[Float[Tensor, "..."]]:
        """List of all parameters which will be decomposed with SPD."""
        return [self.W, rearrange(self.W, "i f h -> i h f")]


class TMSSPDModel(SPDModel):
    def __init__(
        self,
        n_instances: int,
        n_features: int,
        n_hidden: int,
        k: int | None,
        bias_val: float,
        train_bias: bool,
        device: str = "cuda",
    ):
        super().__init__()
        self.n_instances = n_instances
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.k = k if k is not None else n_features
        self.bias_val = bias_val
        self.train_bias = train_bias

        self.A = nn.Parameter(torch.empty((n_instances, n_features, self.k), device=device))
        self.B = nn.Parameter(torch.empty((n_instances, self.k, n_hidden), device=device))

        bias_data = torch.zeros((n_instances, n_features), device=device) + bias_val
        self.b_final = nn.Parameter(bias_data) if train_bias else bias_data

        nn.init.xavier_normal_(self.A)
        # Fix the first instance to the identity to compare losses
        assert (
            n_features == self.k
        ), "Currently only supports n_features == k if fixing first instance to identity"
        self.A.data[0] = torch.eye(n_features, device=device)
        nn.init.xavier_normal_(self.B)

        self.n_param_matrices = 2  # Two W matrices (even though they're tied)

    def all_As(self) -> list[Float[Tensor, "dim k"]]:
        # Note that A is defined as the matrix which mutliplies the activations
        # to get the inner_acts. In TMS, because we tie the W matrices, our second A matrix
        # is actually the B matrix
        return [self.A, rearrange(self.B, "i k h -> i h k")]

    def all_Bs(self) -> list[Float[Tensor, "k dim"]]:
        return [self.B, rearrange(self.A, "i f k -> i k f")]

    def all_subnetwork_params(self) -> list[Float[Tensor, "n_instances k n_features n_hidden"]]:
        w1 = torch.einsum("ifk,ikh->ikfh", self.A, self.B)
        return [w1, rearrange(w1, "i k f h -> i k h f")]

    def forward(
        self, x: Float[Tensor, "... i f"]
    ) -> tuple[
        Float[Tensor, "... i f"], list[Float[Tensor, "... i f"]], list[Float[Tensor, "... i k"]]
    ]:
        inner_act_0 = torch.einsum("...if,ifk->...ik", x, self.A)
        layer_act_0 = torch.einsum("...ik,ikh->...ih", inner_act_0, self.B)

        inner_act_1 = torch.einsum("...ih,ikh->...ik", layer_act_0, self.B)
        layer_act_1 = torch.einsum("...ik,ifk->...if", inner_act_1, self.A)
        pre_relu = layer_act_1 + self.b_final

        out = F.relu(pre_relu)
        # Can pass layer_act_1 or pre_relu to layer_acts[1] as they're the same for the gradient
        # operations we care about (dout/d(inner_act_1)).
        return out, [layer_act_0, layer_act_1], [inner_act_0, inner_act_1]

    def forward_topk(
        self,
        x: Float[Tensor, "... i f"],
        topk_mask: Bool[Tensor, "... n_instances k"],
    ) -> tuple[
        Float[Tensor, "... i f"],
        list[Float[Tensor, "... i f"]],
        list[Float[Tensor, "... i k"]],
    ]:
        """Performs a forward pass using only the top-k subnetwork activations."""
        inner_act_0 = torch.einsum("...if,ifk->...ik", x, self.A)
        assert topk_mask.shape == inner_act_0.shape
        inner_act_0_topk = inner_act_0 * topk_mask
        layer_act_0 = torch.einsum("...ik,ikh->...ih", inner_act_0_topk, self.B)

        inner_act_1 = torch.einsum("...ih,ikh->...ik", layer_act_0, self.B)
        assert topk_mask.shape == inner_act_1.shape
        inner_act_1_topk = inner_act_1 * topk_mask
        layer_act_1 = torch.einsum("...ik,ifk->...if", inner_act_1_topk, self.A)

        pre_relu = layer_act_1 + self.b_final
        out = F.relu(pre_relu)
        return out, [layer_act_0, layer_act_1], [inner_act_0_topk, inner_act_1_topk]

    @classmethod
    def from_pretrained(cls, path: str | RootPath) -> "TMSSPDModel":  # type: ignore
        pass

    def set_matrices_to_unit_norm(self):
        self.A.data /= self.A.data.norm(p=2, dim=-2, keepdim=True)

    def fix_normalized_adam_gradients(self):
        assert self.A.grad is not None
        remove_grad_parallel_to_subnetwork_vecs(self.A.data, self.A.grad)


class TMSSPDFullRankModel(SPDFullRankModel):
    def __init__(
        self,
        n_instances: int,
        n_features: int,
        n_hidden: int,
        k: int | None,
        bias_val: float,
        train_bias: bool,
        device: str = "cuda",
    ):
        super().__init__()
        self.n_instances = n_instances
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.k = k if k is not None else n_features
        self.bias_val = bias_val
        self.train_bias = train_bias

        self.subnetwork_params = nn.Parameter(
            torch.empty((n_instances, self.k, n_features, n_hidden), device=device)
        )

        bias_data = torch.zeros((n_instances, n_features), device=device) + bias_val
        self.b_final = nn.Parameter(bias_data) if train_bias else bias_data

        nn.init.xavier_normal_(self.subnetwork_params)

        self.n_param_matrices = 2  # Two W matrices (even though they're tied)

    def all_subnetwork_params(self) -> list[Float[Tensor, "n_instances k n_features n_hidden"]]:
        return [self.subnetwork_params, rearrange(self.subnetwork_params, "i k f h -> i k h f")]

    def forward(
        self, x: Float[Tensor, "... n_instances n_features"]
    ) -> tuple[
        Float[Tensor, "... n_instances n_features"],
        list[Float[Tensor, "... n_instances n_features"]],
        list[Float[Tensor, "... n_instances k"]],
    ]:
        inner_act_0 = torch.einsum("...if,ikfh->...ikh", x, self.subnetwork_params)
        layer_act_0 = torch.einsum("...ikh->...ih", inner_act_0)

        inner_act_1 = torch.einsum("...ih,ikfh->...ikf", layer_act_0, self.subnetwork_params)
        layer_act_1 = torch.einsum("...ikf->...if", inner_act_1)
        pre_relu = layer_act_1 + self.b_final

        out = F.relu(pre_relu)
        # Can pass layer_act_1 or pre_relu to layer_acts[1] as they're the same for the gradient
        # operations we care about (dout/d(inner_act_1)).
        return out, [layer_act_0, layer_act_1], [inner_act_0, inner_act_1]

    def forward_topk(
        self,
        x: Float[Tensor, "... n_instances n_features"],
        topk_mask: Bool[Tensor, "... n_instances k"],
    ) -> tuple[
        Float[Tensor, "... n_instances n_features"],
        list[Float[Tensor, "... n_instances n_features"]],
        list[Float[Tensor, "... n_instances k"]],
    ]:
        """Performs a forward pass using only the top-k subnetwork activations."""

        inner_act_0 = torch.einsum("...if,ikfh->...ikh", x, self.subnetwork_params)
        assert topk_mask.shape == inner_act_0.shape[:-1]
        inner_act_0_topk = torch.einsum("...ikh,...ik->...ikh", inner_act_0, topk_mask)
        layer_act_0 = torch.einsum("...ikh->...ih", inner_act_0_topk)

        inner_act_1 = torch.einsum("...ih,ikfh->...ikf", layer_act_0, self.subnetwork_params)
        assert topk_mask.shape == inner_act_1.shape[:-1]
        inner_act_1_topk = torch.einsum("...ikf,...ik->...ikf", inner_act_1, topk_mask)
        layer_act_1 = torch.einsum("...ikf->...if", inner_act_1_topk)

        pre_relu = layer_act_1 + self.b_final
        out = F.relu(pre_relu)
        return out, [layer_act_0, layer_act_1], [inner_act_0_topk, inner_act_1_topk]

    @classmethod
    def from_pretrained(cls, path: str | RootPath) -> "TMSSPDFullRankModel":  # type: ignore
        pass
