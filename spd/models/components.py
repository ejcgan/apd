from collections.abc import Callable

import einops
import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from spd.utils import init_param_


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


class InstancesParamComponentsFullRank(nn.Module):
    def __init__(
        self, n_instances: int, in_dim: int, out_dim: int, k: int, bias: bool, init_scale: float
    ):
        super().__init__()
        self.subnetwork_params = nn.Parameter(torch.empty(n_instances, k, in_dim, out_dim))
        init_param_(self.subnetwork_params, init_scale)

        self.bias = nn.Parameter(torch.zeros(n_instances, out_dim)) if bias else None

    def forward(
        self,
        x: Float[Tensor, "batch n_instances d_in"],
        topk_mask: Bool[Tensor, "batch n_instances k"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch n_instances d_out"], Float[Tensor, "batch n_instances k d_out"]
    ]:
        inner_acts = einops.einsum(
            x,
            self.subnetwork_params,
            "batch n_instances d_in, n_instances k d_in d_out -> batch n_instances k d_out",
        )
        if self.bias is not None:
            inner_acts += self.bias

        if topk_mask is not None:
            inner_acts = einops.einsum(
                inner_acts,
                topk_mask,
                "batch n_instances k d_out, batch n_instances k -> batch n_instances k d_out",
            )

        out = einops.einsum(inner_acts, "batch n_instances k d_out -> batch n_instances d_out")
        return out, inner_acts


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
            resid_component: Predefined component matrix of shape (d_embed, k) if A or (k, d_embed)
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
        topk_mask: Bool[Tensor, "batch k"] | None = None,
    ) -> tuple[Float[Tensor, "batch dim2"], Float[Tensor, "batch k"]]:
        inner_acts = einops.einsum(x, self.A, "batch dim1, dim1 k -> batch k")
        if topk_mask is not None:
            inner_acts = einops.einsum(inner_acts, topk_mask, "batch k, batch k -> batch k")
        out = einops.einsum(inner_acts, self.B, "batch k, k dim2 -> batch dim2")
        return out, inner_acts


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
        self, x: Float[Tensor, "batch dim1"], topk_mask: Bool[Tensor, "batch k"] | None = None
    ) -> tuple[Float[Tensor, "batch dim2"], Float[Tensor, "batch k dim2"]]:
        inner_acts = einops.einsum(
            x, self.subnetwork_params, "batch dim1, k dim1 dim2 -> batch k dim2"
        )
        if self.bias is not None:
            inner_acts += self.bias

        if topk_mask is not None:
            inner_acts = einops.einsum(
                inner_acts, topk_mask, "batch k dim2, batch k -> batch k dim2"
            )
        out = einops.einsum(inner_acts, "batch k dim2 -> batch dim2")
        return out, inner_acts


class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        act_fn: Callable[[Tensor], Tensor],
        in_bias: bool,
        out_bias: bool,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.input_layer = nn.Linear(d_model, d_mlp, bias=in_bias)
        self.output_layer = nn.Linear(d_mlp, d_model, bias=out_bias)
        self.act_fn = act_fn

    def forward(
        self, x: Float[Tensor, "... d_model"]
    ) -> tuple[
        Float[Tensor, "... d_model"],
        dict[str, Float[Tensor, "... d_model"] | Float[Tensor, "... d_mlp"] | None],
        dict[str, Float[Tensor, "... d_model"] | Float[Tensor, "... d_mlp"]],
    ]:
        """Run a forward pass and cache pre and post activations for each parameter.

        Note that we don't need to cache pre activations for the biases. We also don't care about
        the output bias which is always zero.
        """
        out1_pre_act_fn = self.input_layer(x)
        out1 = self.act_fn(out1_pre_act_fn)
        out2 = self.output_layer(out1)

        pre_acts = {
            "input_layer.weight": x,
            "input_layer.bias": None,
            "output_layer.weight": out1,
        }
        post_acts = {
            "input_layer.weight": out1_pre_act_fn,
            "input_layer.bias": out1_pre_act_fn,
            "output_layer.weight": out2,
        }
        return out2, pre_acts, post_acts


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

    def forward(
        self, x: Float[Tensor, "... d_embed"], topk_mask: Bool[Tensor, "... k"] | None = None
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
        x, inner_acts_linear1 = self.linear1(x, topk_mask)
        x += self.bias1
        inner_acts.append(inner_acts_linear1)
        layer_acts.append(x)

        x = torch.nn.functional.relu(x)
        x, inner_acts_linear2 = self.linear2(x, topk_mask)
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
        act_fn: Callable[[Tensor], Tensor],
        in_bias: bool,
        out_bias: bool,
    ):
        super().__init__()
        self.act_fn = act_fn
        self.linear1 = ParamComponentsFullRank(
            in_dim=d_embed, out_dim=d_mlp, k=k, bias=in_bias, init_scale=init_scale
        )
        self.linear2 = ParamComponentsFullRank(
            in_dim=d_mlp, out_dim=d_embed, k=k, bias=out_bias, init_scale=init_scale
        )

    def forward(
        self, x: Float[Tensor, "... d_embed"], topk_mask: Bool[Tensor, "... k"] | None = None
    ) -> tuple[
        Float[Tensor, "... d_embed"],
        list[Float[Tensor, "... d_embed"] | Float[Tensor, "... d_mlp"]],
        list[Float[Tensor, "... k"]] | list[Float[Tensor, "... k d_embed"]],
    ]:
        """
        Args:
            x: Input tensor
            topk_mask: Boolean tensor indicating which subnetworks to keep.
        Returns:
            x: The output of the MLP
            layer_acts: The activations at the output of each layer after summing over the
                subnetwork dimension.
            inner_acts: The activations at the output of each subnetwork before summing.
        """
        inner_acts = []
        layer_acts = []
        x, inner_acts_linear1 = self.linear1(x, topk_mask)
        inner_acts.append(inner_acts_linear1)
        layer_acts.append(x)

        x = self.act_fn(x)

        x, inner_acts_linear2 = self.linear2(x, topk_mask)
        inner_acts.append(inner_acts_linear2)
        layer_acts.append(x)
        return x, layer_acts, inner_acts


class ParamComponentsRankPenalty(nn.Module):
    """A linear layer decomposed into A and B matrices for rank penalty SPD.

    The weight matrix W is decomposed as W = A @ B, where A and B are learned parameters.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        k: int,
        bias: bool,
        init_scale: float = 1.0,
        m: int | None = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.m = min(in_dim, out_dim) if m is None else m

        # Initialize A and B matrices
        self.A = nn.Parameter(torch.empty(k, in_dim, self.m))
        self.B = nn.Parameter(torch.empty(k, self.m, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

        init_param_(self.A, scale=init_scale)
        init_param_(self.B, scale=init_scale)

    @property
    def subnetwork_params(self) -> Float[Tensor, "k i j"]:
        """For compatibility with plotting code."""
        return einops.einsum(self.A, self.B, "k i m, k m j -> k i j")

    def forward(
        self, x: Float[Tensor, "batch d_in"], topk_mask: Bool[Tensor, "batch k"] | None = None
    ) -> tuple[Float[Tensor, "batch d_out"], Float[Tensor, "batch k d_out"]]:
        """Forward pass through the layer.

        Args:
            x: Input tensor
            topk_mask: Boolean tensor indicating which subnetworks to keep
        Returns:
            output: The summed output across all subnetworks
            inner_acts: The output of each subnetwork before summing
        """
        # First multiply by A to get to intermediate dimension m
        # pre_inner_acts = torch.einsum("...d,kfm->...km", x, self.A)
        pre_inner_acts = einops.einsum(x, self.A, "batch d_in, k d_in m -> batch k m")
        if topk_mask is not None:
            assert topk_mask.shape == pre_inner_acts.shape[:-1]
            pre_inner_acts = einops.einsum(
                pre_inner_acts, topk_mask, "batch k m, batch k -> batch k m"
            )

        # Then multiply by B to get to output dimension
        inner_acts = einops.einsum(pre_inner_acts, self.B, "batch k m, k m h -> batch k h")

        if topk_mask is not None:
            inner_acts = einops.einsum(inner_acts, topk_mask, "batch k h, batch k -> batch k h")

        # Sum over subnetwork dimension
        out = einops.einsum(inner_acts, "batch k h -> batch h")

        # Add the bias if it exists
        if self.bias is not None:
            out += self.bias
        return out, inner_acts


class MLPComponentsRankPenalty(nn.Module):
    """A module that contains two linear layers with an activation in between for rank penalty SPD.

    Each linear layer is decomposed into A and B matrices, where the weight matrix W = A @ B.
    The biases are (optionally) part of the "linear" layers, and have a subnetwork dimension.
    """

    def __init__(
        self,
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
        self.linear1 = ParamComponentsRankPenalty(
            in_dim=d_embed, out_dim=d_mlp, k=k, bias=in_bias, init_scale=init_scale, m=m
        )
        self.linear2 = ParamComponentsRankPenalty(
            in_dim=d_mlp, out_dim=d_embed, k=k, bias=out_bias, init_scale=init_scale, m=m
        )

    def forward(
        self, x: Float[Tensor, "... d_embed"], topk_mask: Bool[Tensor, "... k"] | None = None
    ) -> tuple[
        Float[Tensor, "... d_embed"],
        list[Float[Tensor, "... d_embed"] | Float[Tensor, "... d_mlp"]],
        list[Float[Tensor, "... k d_embed"] | Float[Tensor, "... k d_mlp"]],
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
