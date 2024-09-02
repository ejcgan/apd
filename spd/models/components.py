import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from spd.utils import init_param_


class ParamComponents(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        k: int,
        resid_component: nn.Parameter | None,
        resid_dim: int | None,
    ):
        """
        Args:
            in_dim: Input dimension of the parameter to be replaced with AB.
            out_dim: Output dimension of the parameter to be replaced with AB.
            k: Number of subnetworks.
            resid_component: Predefined component matrix of shape (d_resid, k) if A or (k, d_resid)
                if B.
            resid_dim: Dimension in which to use the predefined component.
        """
        super().__init__()

        if resid_component is not None:
            if resid_dim == 0:
                a = resid_component
                b = nn.Parameter(torch.empty(k, out_dim))
            elif resid_dim == 1:
                a = nn.Parameter(torch.empty(in_dim, k))
                b = resid_component
            else:
                raise ValueError("Invalid resid_dim value. Must be 0 or 1.")
        else:
            a = nn.Parameter(torch.empty(in_dim, k))
            b = nn.Parameter(torch.empty(k, out_dim))

        self.A = a
        self.B = b
        init_param_(self.A)
        init_param_(self.B)

    def forward(
        self,
        x: Float[Tensor, "... dim1"],
    ) -> tuple[Float[Tensor, "... dim2"], Float[Tensor, "... k"]]:
        inner_acts = torch.einsum("bf,fk->bk", x, self.A)
        out = torch.einsum("bk,kg->bg", inner_acts, self.B)
        return out, inner_acts

    def forward_topk(
        self,
        x: Float[Tensor, "... dim1"],
        topk_mask: Bool[Tensor, "... k"],
    ) -> tuple[Float[Tensor, "... dim2"], Float[Tensor, "... k"]]:
        """
        Performs a forward pass using only the top-k subnetwork activations.

        Args:
            x: Input tensor
            topk_mask: Boolean tensor indicating which subnetwork activations to keep.

        Returns:
            out: Output tensor
            inner_acts: Subnetwork activations
        """

        inner_acts = torch.einsum("bf,fk->bk", x, self.A)
        inner_acts_topk = inner_acts * topk_mask
        out = torch.einsum("bk,kg->bg", inner_acts_topk, self.B)
        return out, inner_acts_topk


class MLPComponents(nn.Module):
    """
    A module that contains two linear layers with a ReLU activation in between.

    Note that the first linear layer has a bias that is not decomposed, and the second linear layer
    has no bias.
    """

    def __init__(
        self,
        d_embed: int,
        d_mlp: int,
        k: int,
        input_bias: Float[Tensor, " d_mlp"] | None = None,
        input_component: nn.Parameter | None = None,
        output_component: nn.Parameter | None = None,
    ):
        super().__init__()
        self.linear1 = ParamComponents(
            d_embed, d_mlp, k, resid_component=input_component, resid_dim=0
        )
        self.bias1 = nn.Parameter(torch.zeros(d_mlp))
        if input_bias is not None:
            self.bias1.data = input_bias.detach().clone()
        self.linear2 = ParamComponents(
            d_mlp, d_embed, k, resid_component=output_component, resid_dim=1
        )

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
        topk_mask: Bool[Tensor, "... k"],
    ) -> tuple[
        Float[Tensor, "... d_embed"],
        list[Float[Tensor, "... d_embed"] | Float[Tensor, "... d_mlp"]],
        list[Float[Tensor, "... k"]],
    ]:
        """
        Performs a forward pass using only the top-k components for each linear layer.

        Args:
            x: Input tensor
            topk_mask: Boolean tensor indicating which components to keep.
        Returns:
            x: The output of the MLP
            layer_acts: The activations of each linear layer
            inner_acts: The component activations inside each linear layer
        """
        inner_acts = []
        layer_acts = []

        # First linear layer
        x, inner_acts_linear1 = self.linear1.forward_topk(x, topk_mask)
        x += self.bias1
        inner_acts.append(inner_acts_linear1)
        layer_acts.append(x)

        x = torch.nn.functional.relu(x)

        # Second linear layer
        x, inner_acts_linear2 = self.linear2.forward_topk(x, topk_mask)
        inner_acts.append(inner_acts_linear2)
        layer_acts.append(x)

        return x, layer_acts, inner_acts
