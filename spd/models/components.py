from typing import Any, Literal

import einops
import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from spd.hooks import HookPoint
from spd.utils import init_param_


class Linear(nn.Module):
    """A linear transformation with an optional n_instances dimension."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_instances: int | None = None,
        init_type: Literal["kaiming_uniform", "xavier_normal"] = "kaiming_uniform",
        init_scale: float = 1.0,
    ):
        super().__init__()
        shape = (n_instances, d_in, d_out) if n_instances is not None else (d_in, d_out)
        self.weight = nn.Parameter(torch.empty(shape))
        init_param_(self.weight, scale=init_scale, init_type=init_type)

        self.hook_pre = HookPoint()  # (batch ... d_in)
        self.hook_post = HookPoint()  # (batch ... d_out)

    def forward(
        self, x: Float[Tensor, "batch ... d_in"], *args: Any, **kwargs: Any
    ) -> Float[Tensor, "batch ... d_out"]:
        x = self.hook_pre(x)
        out = einops.einsum(x, self.weight, "batch ... d_in, ... d_in d_out -> batch ... d_out")
        out = self.hook_post(out)
        return out


class LinearComponent(nn.Module):
    """A linear transformation made from A and B matrices for SPD.

    The weight matrix W is decomposed as W = A @ B, where A and B are learned parameters.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        k: int,
        n_instances: int | None = None,
        init_type: Literal["kaiming_uniform", "xavier_normal"] = "kaiming_uniform",
        init_scale: float = 1.0,
        m: int | None = None,
    ):
        super().__init__()
        self.n_instances = n_instances
        self.k = k
        self.m = min(d_in, d_out) if m is None else m

        # Initialize A and B matrices
        shape_A = (n_instances, k, d_in, self.m) if n_instances is not None else (k, d_in, self.m)
        shape_B = (n_instances, k, self.m, d_out) if n_instances is not None else (k, self.m, d_out)
        self.A = nn.Parameter(torch.empty(shape_A))
        self.B = nn.Parameter(torch.empty(shape_B))
        self.hook_pre = HookPoint()  # (batch ... d_in)
        self.hook_component_acts = HookPoint()  # (batch ... k d_out)
        self.hook_post = HookPoint()  # (batch ... d_out)

        init_param_(self.A, scale=init_scale, init_type=init_type)
        init_param_(self.B, scale=init_scale, init_type=init_type)

    @property
    def component_weights(self) -> Float[Tensor, "... k d_in d_out"]:
        """A @ B before summing over the subnetwork dimension."""
        return einops.einsum(self.A, self.B, "... k d_in m, ... k m d_out -> ... k d_in d_out")

    @property
    def weight(self) -> Float[Tensor, "... d_in d_out"]:
        """A @ B after summing over the subnetwork dimension."""
        return einops.einsum(self.A, self.B, "... k d_in m, ... k m d_out -> ... d_in d_out")

    def forward(
        self,
        x: Float[Tensor, "batch ... d_in"],
        topk_mask: Bool[Tensor, "batch ... k"] | None = None,
    ) -> Float[Tensor, "batch ... d_out"]:
        """Forward pass through A and B matrices which make up the component for this layer.

        Args:
            x: Input tensor
            topk_mask: Boolean tensor indicating which subnetworks to keep
        Returns:
            output: The summed output across all subnetworks
        """
        x = self.hook_pre(x)

        # First multiply by A to get to intermediate dimension m
        inner_acts = einops.einsum(x, self.A, "batch ... d_in, ... k d_in m -> batch ... k m")
        if topk_mask is not None:
            assert topk_mask.shape == inner_acts.shape[:-1]
            inner_acts = einops.einsum(
                inner_acts, topk_mask, "batch ... k m, batch ... k -> batch ... k m"
            )

        # Then multiply by B to get to output dimension
        component_acts = einops.einsum(
            inner_acts, self.B, "batch ... k m, ... k m d_out -> batch ... k d_out"
        )

        if topk_mask is not None:
            component_acts = einops.einsum(
                component_acts, topk_mask, "batch ... k d_out, batch ... k -> batch ... k d_out"
            )
        self.hook_component_acts(component_acts)

        # Sum over subnetwork dimension
        out = einops.einsum(component_acts, "batch ... k d_out -> batch ... d_out")
        out = self.hook_post(out)
        return out
