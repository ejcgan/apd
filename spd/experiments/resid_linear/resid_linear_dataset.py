from typing import Literal

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset


def calc_labels(
    coeffs: Float[Tensor, " n_functions"],
    embed_matrix: Float[Tensor, "n_features d_embed"],
    inputs: Float[Tensor, "batch n_functions"],
) -> Float[Tensor, "batch d_embed"]:
    """Calculate the corresponding labels for the inputs using W_E(gelu(coeffs*x) + x)."""
    weighted_inputs = einops.einsum(
        inputs, coeffs, "batch n_functions, n_functions -> batch n_functions"
    )
    raw_labels = F.gelu(weighted_inputs) + inputs
    embedded_labels = einops.einsum(
        raw_labels, embed_matrix, "batch n_functions, n_functions d_embed -> batch d_embed"
    )
    return embedded_labels


class ResidualLinearDataset(
    Dataset[tuple[Float[Tensor, "batch n_features"], Float[Tensor, "batch d_embed"]]]
):
    def __init__(
        self,
        embed_matrix: Float[Tensor, "n_features d_embed"],
        n_features: int,
        feature_probability: float,
        device: str,
        label_fn_seed: int | None = None,
        label_coeffs: list[float] | None = None,
        data_generation_type: Literal[
            "exactly_one_active", "at_least_zero_active"
        ] = "at_least_zero_active",
    ):
        assert label_coeffs is not None or label_fn_seed is not None
        self.embed_matrix = embed_matrix.to(device)
        self.n_features = n_features
        self.feature_probability = feature_probability
        self.device = device
        self.label_fn_seed = label_fn_seed
        self.data_generation_type = data_generation_type

        if label_coeffs is None:
            # Create random coeffs between [1, 2]
            gen = torch.Generator()
            if self.label_fn_seed is not None:
                gen.manual_seed(self.label_fn_seed)
            self.coeffs = (
                torch.rand(self.embed_matrix.shape[0], generator=gen, device=self.device) + 1
            )
        else:
            self.coeffs = torch.tensor(label_coeffs, device=self.device)

        self.label_fn = lambda inputs: calc_labels(self.coeffs, self.embed_matrix, inputs)

    def __len__(self) -> int:
        return 2**31

    def generate_batch(
        self, batch_size: int
    ) -> tuple[Float[Tensor, "batch n_features"], Float[Tensor, "batch d_embed"]]:
        if self.data_generation_type == "exactly_one_active":
            batch = self._generate_one_feature_active_batch(batch_size)
        elif self.data_generation_type == "at_least_zero_active":
            batch = self._generate_multi_feature_batch(batch_size)
        else:
            raise ValueError(f"Invalid generation type: {self.data_generation_type}")

        labels = self.label_fn(batch)
        return batch, labels

    def _generate_one_feature_active_batch(
        self, batch_size: int
    ) -> Float[Tensor, "batch n_features"]:
        batch = torch.zeros(batch_size, self.n_features, device=self.device)
        active_features = torch.randint(0, self.n_features, (batch_size,), device=self.device)
        # Generate random values in [-1, 1] for active features
        batch[torch.arange(batch_size), active_features] = (
            torch.rand(batch_size, device=self.device) * 2 - 1
        )
        return batch

    def _generate_multi_feature_batch(self, batch_size: int) -> Float[Tensor, "batch n_features"]:
        # Generate random values in [-1, 1] for all features
        batch = torch.rand((batch_size, self.n_features), device=self.device) * 2 - 1
        mask = torch.rand_like(batch) < self.feature_probability
        return batch * mask
