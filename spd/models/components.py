import torch
from jaxtyping import Float
from torch import Tensor, nn


class ParamComponents(nn.Module):
    def __init__(self, dim1: int, dim2: int, k: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(dim1, k))
        self.B = nn.Parameter(torch.empty(k, dim2))

        nn.init.kaiming_normal_(self.A)
        nn.init.kaiming_normal_(self.B)

    def forward(
        self,
        x: Float[Tensor, "... dim1"],
    ) -> tuple[Float[Tensor, "... dim2"], Float[Tensor, "... k"]]:
        normed_A = self.A / self.A.norm(p=2, dim=-2, keepdim=True)
        inner_acts = torch.einsum("bf,fk->bk", x, normed_A)
        out = torch.einsum("bk,kg->bg", inner_acts, self.B)
        return out, inner_acts

    def forward_topk(
        self,
        x: Float[Tensor, "... dim1"],
        topk: int,
        grads: Float[Tensor, "... k"] | None = None,
    ) -> tuple[Float[Tensor, "... dim2"], Float[Tensor, "... k"]]:
        """
        Performs a forward pass using only the top-k components.

        Args:
            x: Input tensor
            topk: Number of top components to keep
            grads: Optional gradients for each component

        Returns:
            out: Output tensor
            inner_acts: Component activations
        """
        normed_A = self.A / self.A.norm(p=2, dim=-2, keepdim=True)
        inner_acts = torch.einsum("bf,fk->bk", x, normed_A)

        if grads is not None:
            topk_indices = (grads * inner_acts).abs().topk(topk, dim=-1).indices
        else:
            topk_indices = inner_acts.abs().topk(topk, dim=-1).indices

        # Get values in inner_acts corresponding to topk_indices
        topk_values = inner_acts.gather(dim=-1, index=topk_indices)
        inner_acts_topk = torch.zeros_like(inner_acts)
        inner_acts_topk.scatter_(dim=-1, index=topk_indices, src=topk_values)
        out = torch.einsum("bk,kg->bg", inner_acts_topk, self.B)

        return out, inner_acts_topk


class MLPComponents(nn.Module):
    """
    A module that contains two linear layers with a ReLU activation in between.

    Note that the first linear layer has a bias that is not decomposed, and the second linear layer
    has no bias.
    """

    def __init__(self, d_embed: int, d_mlp: int, k: int):
        super().__init__()
        self.linear1 = ParamComponents(d_embed, d_mlp, k)
        self.bias1 = nn.Parameter(torch.zeros(d_mlp))
        self.linear2 = ParamComponents(d_mlp, d_embed, k)

    def forward(
        self, x: Float[Tensor, "... d_embed"]
    ) -> tuple[
        Float[Tensor, "... d_embed"],
        list[Float[Tensor, "... d_embed"] | Float[Tensor, "... d_mlp"]],
        list[Float[Tensor, "... k"]],
    ]:
        """
        Returns:
            x: The output of the MLP
            layer_acts: The activations of each linear layer
            inner_acts: The component activations inside each linear layer
        """
        inner_acts = []
        layer_acts = []
        x, inner_acts_linear1 = self.linear1(x)
        x += self.bias1
        inner_acts.append(inner_acts_linear1)
        layer_acts.append(x)

        x, inner_acts_linear2 = self.linear2(torch.nn.functional.relu(x))
        inner_acts.append(inner_acts_linear2)
        layer_acts.append(x)
        return x, layer_acts, inner_acts

    def forward_topk(
        self,
        x: Float[Tensor, "... d_embed"],
        topk: int,
        grads: list[Float[Tensor, "... k"] | None] | None = None,
    ) -> tuple[
        Float[Tensor, "... d_embed"],
        list[Float[Tensor, "... d_embed"] | Float[Tensor, "... d_mlp"]],
        list[Float[Tensor, "... k"]],
    ]:
        """
        Performs a forward pass using only the top-k components for each linear layer.

        Args:
            x: Input tensor
            topk: Number of top components to keep
            grads: Optional list of gradients for each linear layer

        Returns:
            x: The output of the MLP
            layer_acts: The activations of each linear layer
            inner_acts: The component activations inside each linear layer
        """
        inner_acts = []
        layer_acts = []

        # First linear layer
        grad1 = grads[0] if grads is not None else None
        x, inner_acts_linear1 = self.linear1.forward_topk(x, topk, grad1)
        x += self.bias1
        inner_acts.append(inner_acts_linear1)
        layer_acts.append(x)

        x = torch.nn.functional.relu(x)

        # Second linear layer
        grad2 = grads[1] if grads is not None else None
        x, inner_acts_linear2 = self.linear2.forward_topk(x, topk, grad2)
        inner_acts.append(inner_acts_linear2)
        layer_acts.append(x)

        return x, layer_acts, inner_acts
