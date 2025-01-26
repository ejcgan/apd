from collections.abc import Callable
from typing import Literal

import einops
import torch
from jaxtyping import Bool, Float
from numpy import ndim
from torch import Tensor, nn

from spd.utils import init_param_


class InstancesParamComponents(nn.Module):
    """A linear layer decomposed into A and B matrices for SPD.

    The weight matrix W is decomposed as W = A @ B, where A and B are learned parameters.
    """

    def __init__(
        self,
        n_instances: int,
        in_dim: int,
        out_dim: int,
        k: int,
        bias: bool,
        init_type: Literal["kaiming_uniform", "xavier_normal"] = "kaiming_uniform",
        init_scale: float = 1.0,
        m: int | None = None,
    ):
        super().__init__()
        self.n_instances = n_instances
        self.in_dim = ndim
        self.out_dim = out_dim
        self.k = k
        self.m = min(in_dim, out_dim) if m is None else m

        # Initialize A and B matrices
        self.A = nn.Parameter(torch.empty(n_instances, k, in_dim, self.m))
        self.B = nn.Parameter(torch.empty(n_instances, k, self.m, out_dim))
        self.bias = nn.Parameter(torch.zeros(n_instances, out_dim)) if bias else None

        init_param_(self.A, scale=init_scale, init_type=init_type)
        init_param_(self.B, scale=init_scale, init_type=init_type)

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
        dict[str, Float[Tensor, "... d_model"] | Float[Tensor, "... d_mlp"]],
        dict[str, Float[Tensor, "... d_model"] | Float[Tensor, "... d_mlp"]],
    ]:
        """Run a forward pass and cache pre and post activations for each parameter.

        Note that we don't need to cache pre activations for the biases. We also don't care about
        the output bias which is always zero.
        """
        out1_pre_act_fn = self.input_layer(x)
        out1 = self.act_fn(out1_pre_act_fn)
        out2 = self.output_layer(out1)

        pre_acts = {"input_layer.weight": x, "output_layer.weight": out1}
        post_acts = {"input_layer.weight": out1_pre_act_fn, "output_layer.weight": out2}
        return out2, pre_acts, post_acts


class ParamComponents(nn.Module):
    """A linear layer decomposed into A and B matrices for SPD.

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


class MLPComponents(nn.Module):
    """A module that contains two linear layers with an activation in between for SPD.

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
        self.linear1 = ParamComponents(
            in_dim=d_embed, out_dim=d_mlp, k=k, bias=in_bias, init_scale=init_scale, m=m
        )
        self.linear2 = ParamComponents(
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
