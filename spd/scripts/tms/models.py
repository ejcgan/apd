import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from spd.models.base import Model, SPDModel
from spd.types import RootPath


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
        return [self.W, einops.rearrange(self.W, "i f h -> i h f")]


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
        A_1 = einops.rearrange(self.B, "i k h -> i h k")
        return [self.A / self.A.norm(p=2, dim=-2, keepdim=True), A_1]

    def all_Bs(self) -> list[Float[Tensor, "k dim"]]:
        B_1 = einops.rearrange(self.A / self.A.norm(p=2, dim=-2, keepdim=True), "i f k -> i k f")
        return [self.B, B_1]

    def forward(
        self, features: Float[Tensor, "... i f"]
    ) -> tuple[
        Float[Tensor, "... i f"], list[Float[Tensor, "... i f"]], list[Float[Tensor, "... i k"]]
    ]:
        normed_A = self.A / self.A.norm(p=2, dim=-2, keepdim=True)
        h_0 = torch.einsum("...if,ifk->...ik", features, normed_A)
        hidden_0 = torch.einsum("...ik,ikh->...ih", h_0, self.B)

        h_1 = torch.einsum("...ih,ikh->...ik", hidden_0, self.B)
        hidden_1 = torch.einsum("...ik,ifk->...if", h_1, normed_A)
        pre_relu = hidden_1 + self.b_final

        out = F.relu(pre_relu)
        # Can pass hidden_1 or pre_relu to layer_acts[1] as they're the same for the gradient
        # operations we care about (dout/d(h_1)).
        return out, [hidden_0, hidden_1], [h_0, h_1]  # out, layer_acts, inner_acts

    def forward_topk(
        self,
        x: Float[Tensor, "... i f"],
        topk_indices: Int[Tensor, "... topk"],
    ) -> tuple[
        Float[Tensor, "... i f"],
        list[Float[Tensor, "... i f"]],
        list[Float[Tensor, "... i k"]],
    ]:
        normed_A = self.A / self.A.norm(p=2, dim=-2, keepdim=True)
        h_0 = torch.einsum("...if,ifk->...ik", x, normed_A)

        topk_values_0 = h_0.gather(dim=-1, index=topk_indices)
        h_0_topk = torch.zeros_like(h_0)
        h_0_topk.scatter_(dim=-1, index=topk_indices, src=topk_values_0)

        hidden_0 = torch.einsum("...ik,ikh->...ih", h_0_topk, self.B)
        h_1 = torch.einsum("...ih,ikh->...ik", hidden_0, self.B)

        topk_values_1 = h_1.gather(dim=-1, index=topk_indices)
        h_1_topk = torch.zeros_like(h_1)
        h_1_topk.scatter_(dim=-1, index=topk_indices, src=topk_values_1)

        hidden_1 = torch.einsum("...ik,ifk->...if", h_1_topk, normed_A)

        pre_relu = hidden_1 + self.b_final
        out = F.relu(pre_relu)
        return out, [hidden_0, hidden_1], [h_0_topk, h_1_topk]  # out, layer_acts, inner_acts

    @classmethod
    def from_pretrained(cls, path: str | RootPath) -> "TMSSPDModel":  # type: ignore
        pass
