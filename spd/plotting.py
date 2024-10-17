from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor
from torch.utils.data import DataLoader

from spd.models.base import SPDFullRankModel, SPDModel
from spd.run_spd import Config
from spd.utils import run_spd_forward_pass


def plot_subnetwork_attributions_statistics(
    topk_mask: Float[Tensor, "batch_size k"],
) -> dict[str, plt.Figure]:
    """Plot a vertical bar chart of the number of active subnetworks over the batch."""
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    assert topk_mask.ndim == 2
    values = topk_mask.sum(dim=1).cpu().detach().numpy()
    bins = list(range(int(values.min().item()), int(values.max().item()) + 2))
    counts, _ = np.histogram(values, bins=bins)
    bars = ax.bar(bins[:-1], counts, align="center", width=0.8)
    ax.set_xticks(bins[:-1])
    ax.set_xticklabels([str(b) for b in bins[:-1]])
    ax.set_title(f"Active subnetworks on current batch (batch_size={topk_mask.shape[0]})")
    ax.set_xlabel("Number of active subnetworks")
    ax.set_ylabel("Count")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    return {"subnetwork_attributions_statistics": fig}


def plot_subnetwork_correlations(
    dataloader: DataLoader[tuple[Float[Tensor, "batch n_inputs"], Any]],
    spd_model: SPDModel | SPDFullRankModel,
    config: Config,
    device: str,
    n_forward_passes: int = 100,
) -> dict[str, plt.Figure]:
    topk_masks = []
    for batch, _ in dataloader:
        batch = batch.to(device=device)
        assert config.topk is not None
        spd_outputs = run_spd_forward_pass(
            spd_model=spd_model,
            target_model=None,
            input_array=batch,
            full_rank=config.full_rank,
            ablation_attributions=config.attribution_type == "ablation",
            batch_topk=config.batch_topk,
            topk=config.topk,
            distil_from_target=config.distil_from_target,
        )
        topk_masks.append(spd_outputs.topk_mask)
        if len(topk_masks) > n_forward_passes:
            break
    topk_masks = torch.cat(topk_masks).float()
    # Calculate correlation matrix
    corr_matrix = torch.corrcoef(topk_masks.T).cpu()
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    im = ax.matshow(corr_matrix)
    ax.xaxis.set_ticks_position("bottom")
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="#EE7777",
                fontsize=8,
            )
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    ax.set_title("Subnetwork Correlation Matrix")
    ax.set_xlabel("Subnetwork")
    ax.set_ylabel("Subnetwork")
    return {"subnetwork_correlation_matrix": fig}
