from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset


def plot_A_matrix(x: torch.Tensor, pos_only: bool = False) -> plt.Figure:
    n_instances = x.shape[0]

    fig, axs = plt.subplots(
        1, n_instances, figsize=(2.5 * n_instances, 2), squeeze=False, sharey=True
    )

    cmap = "Blues" if pos_only else "RdBu"
    ims = []
    for i in range(n_instances):
        ax = axs[0, i]
        instance_data = x[i, :, :].detach().cpu().float().numpy()
        max_abs_val = np.abs(instance_data).max()
        vmin = 0 if pos_only else -max_abs_val
        vmax = max_abs_val
        im = ax.matshow(instance_data, vmin=vmin, vmax=vmax, cmap=cmap)
        ims.append(im)
        ax.xaxis.set_ticks_position("bottom")
        if i == 0:
            ax.set_ylabel("k", rotation=0, labelpad=10, va="center")
        else:
            ax.set_yticks([])  # Remove y-axis ticks for all but the first plot
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("n_features")

    plt.subplots_adjust(wspace=0.1, bottom=0.15, top=0.9)
    fig.subplots_adjust(bottom=0.2)

    return fig


class TMSDataset(
    Dataset[tuple[Float[Tensor, "n_instances n_features"], Float[Tensor, "n_instances n_features"]]]
):
    def __init__(
        self,
        n_instances: int,
        n_features: int,
        feature_probability: float,
        device: str,
        data_generation_type: Literal[
            "exactly_one_active", "at_least_zero_active"
        ] = "at_least_zero_active",
    ):
        self.n_instances = n_instances
        self.n_features = n_features
        self.feature_probability = feature_probability
        self.device = device
        self.data_generation_type = data_generation_type

    def __len__(self) -> int:
        return 2**31

    def generate_batch(
        self, batch_size: int
    ) -> tuple[
        Float[Tensor, "batch n_instances n_features"], Float[Tensor, "batch n_instances n_features"]
    ]:
        if self.data_generation_type == "exactly_one_active":
            batch = self._generate_one_feature_active_batch(batch_size)
        elif self.data_generation_type == "at_least_zero_active":
            batch = self._generate_multi_feature_batch(batch_size)
        else:
            raise ValueError(f"Invalid generation type: {self.data_generation_type}")
        return batch, batch.clone().detach()

    def _generate_one_feature_active_batch(
        self, batch_size: int
    ) -> Float[Tensor, "batch n_instances n_features"]:
        """Generate a batch with one feature active per sample and instance."""
        batch = torch.zeros(batch_size, self.n_instances, self.n_features, device=self.device)

        active_features = torch.randint(
            0, self.n_features, (batch_size, self.n_instances), device=self.device
        )
        random_values = torch.rand(batch_size, self.n_instances, 1, device=self.device)
        batch.scatter_(dim=2, index=active_features.unsqueeze(-1), src=random_values)
        return batch

    def _generate_multi_feature_batch(
        self, batch_size: int
    ) -> Float[Tensor, "batch n_instances n_features"]:
        """Generate a batch where each feature activates independently with probability
        `feature_probability`."""
        batch = torch.rand(batch_size, self.n_instances, self.n_features, device=self.device)
        mask = torch.rand_like(batch) < self.feature_probability
        return batch * mask
