import einops
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
        init_scale: float,
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
        init_param_(self.A, init_scale)
        init_param_(self.B, init_scale)

    def forward(
        self,
        x: Float[Tensor, "batch dim1"],
    ) -> tuple[Float[Tensor, "batch dim2"], Float[Tensor, "batch k"]]:
        inner_acts = torch.einsum("bf,fk->bk", x, self.A)
        out = torch.einsum("bk,kg->bg", inner_acts, self.B)
        return out, inner_acts

    def forward_topk(
        self,
        x: Float[Tensor, "batch dim1"],
        topk_mask: Bool[Tensor, "batch k"],
    ) -> tuple[Float[Tensor, "batch dim2"], Float[Tensor, "batch k"]]:
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


class ParamComponentsFullRank(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        k: int,
        bias: bool,
        init_scale: float,
    ):
        super().__init__()

        self.subnetwork_params = nn.Parameter(torch.empty(k, in_dim, out_dim))
        init_param_(self.subnetwork_params, init_scale)

        self.bias = nn.Parameter(torch.zeros(k, out_dim)) if bias else None

    def forward(
        self,
        x: Float[Tensor, "batch dim1"],
    ) -> tuple[Float[Tensor, "batch dim2"], Float[Tensor, "batch k dim2"]]:
        inner_acts = einops.einsum(
            x, self.subnetwork_params, "batch dim1, k dim1 dim2 -> batch k dim2"
        )
        if self.bias is not None:
            inner_acts += self.bias
        out = einops.einsum(inner_acts, "batch k dim2 -> batch dim2")
        return out, inner_acts

    def forward_topk(
        self,
        x: Float[Tensor, "batch dim1"],
        topk_mask: Bool[Tensor, "batch k"],
    ) -> tuple[Float[Tensor, "batch dim2"], Float[Tensor, "batch k dim2"]]:
        """
        Performs a forward pass using only the top-k subnetwork activations.

        Args:
            x: Input tensor
            topk_mask: Boolean tensor indicating which subnetwork activations to keep.

        Returns:
            out: Output tensor
            inner_acts: Subnetwork activations
        """

        inner_acts = einops.einsum(
            x, self.subnetwork_params, "batch dim1, k dim1 dim2 -> batch k dim2"
        )
        if self.bias is not None:
            inner_acts += self.bias
        inner_acts_topk = einops.einsum(
            inner_acts, topk_mask, "batch k dim2, batch k -> batch k dim2"
        )
        out = einops.einsum(inner_acts_topk, "batch k dim2 -> batch dim2")
        return out, inner_acts_topk


class MLPComponents(nn.Module):
    """A module that contains two linear layers with a ReLU activation in between for full rank SPD.

    A bias gets added to the first layer but not the second. The bias does not have a subnetwork
    dimension in this rank 1 case.
    """

    def __init__(
        self,
        d_embed: int,
        d_mlp: int,
        k: int,
        init_scale: float,
        input_bias: Float[Tensor, " d_mlp"] | None = None,
        input_component: nn.Parameter | None = None,
        output_component: nn.Parameter | None = None,
    ):
        super().__init__()

        self.linear1 = ParamComponents(
            in_dim=d_embed,
            out_dim=d_mlp,
            k=k,
            resid_component=input_component,
            resid_dim=0,
            init_scale=init_scale,
        )
        self.linear2 = ParamComponents(
            in_dim=d_mlp,
            out_dim=d_embed,
            k=k,
            resid_component=output_component,
            resid_dim=1,
            init_scale=init_scale,
        )

        self.bias1 = nn.Parameter(torch.zeros(d_mlp))
        if input_bias is not None:
            self.bias1.data[:] = input_bias.detach().clone()

    def forward(
        self, x: Float[Tensor, "... d_embed"]
    ) -> tuple[
        Float[Tensor, "... d_embed"],
        list[Float[Tensor, "... d_embed"] | Float[Tensor, "... d_mlp"]],
        list[Float[Tensor, "... k"]] | list[Float[Tensor, "... k d_embed"]],
    ]:
        """
        Note that "inner_acts" represents the activations after multiplcation by A in the rank 1
        case, and after multiplication by subnetwork_params (but before summing over k) in the
        full-rank case.

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
        list[Float[Tensor, "... k"]] | list[Float[Tensor, "... k d_embed"]],
    ]:
        """
        Performs a forward pass using only the top-k components for each linear layer.

        Note that "inner_acts" represents the activations after multiplcation by A in the rank 1
        case, and after multiplication by subnetwork_params (but before summing over k) in the
        full-rank case.

        Args:
            x: Input tensor
            topk_mask: Boolean tensor indicating which components to keep.
        Returns:
            x: The output of the MLP
            layer_acts: The activations of each linear layer
            inner_acts: The component activations inside each linear layer (i.e. after multiplying
                by A)
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


class MLPComponentsFullRank(nn.Module):
    """A module that contains two linear layers with a ReLU activation in between for full rank SPD.

    The biases are (optionally) part of the "linear" layers, and have a subnetwork dimension in this
    full rank case.
    """

    def __init__(
        self,
        d_embed: int,
        d_mlp: int,
        k: int,
        init_scale: float,
    ):
        super().__init__()
        self.linear1 = ParamComponentsFullRank(
            in_dim=d_embed, out_dim=d_mlp, k=k, bias=True, init_scale=init_scale
        )
        self.linear2 = ParamComponentsFullRank(
            in_dim=d_mlp, out_dim=d_embed, k=k, bias=False, init_scale=init_scale
        )

    def forward(
        self, x: Float[Tensor, "... d_embed"]
    ) -> tuple[
        Float[Tensor, "... d_embed"],
        list[Float[Tensor, "... d_embed"] | Float[Tensor, "... d_mlp"]],
        list[Float[Tensor, "... k"]] | list[Float[Tensor, "... k d_embed"]],
    ]:
        """
        Args:
            x: Input tensor
        Returns:
            x: The output of the MLP
            layer_acts: The activations at the output of each layer after summing over the
                subnetwork dimension.
            inner_acts: The activations at the output of each subnetwork before summing.
        """
        inner_acts = []
        layer_acts = []
        x, inner_acts_linear1 = self.linear1(x)
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
        list[Float[Tensor, "... k"]] | list[Float[Tensor, "... k d_embed"]],
    ]:
        """Performs a forward pass using only the top-k components for each linear layer.

        Args:
            x: Input tensor
            topk_mask: Boolean tensor indicating which components to keep.
        Returns:
            x: The output of the MLP
            layer_acts: The activations at the output of each layer after summing over the
                subnetwork dimension.
            inner_acts: The activations at the output of each subnetwork before summing.
        """
        inner_acts = []
        layer_acts = []

        x, inner_acts_linear1 = self.linear1.forward_topk(x, topk_mask)
        inner_acts.append(inner_acts_linear1)
        layer_acts.append(x)

        x = torch.nn.functional.relu(x)

        x, inner_acts_linear2 = self.linear2.forward_topk(x, topk_mask)
        inner_acts.append(inner_acts_linear2)
        layer_acts.append(x)

        return x, layer_acts, inner_acts
