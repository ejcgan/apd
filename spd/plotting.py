from typing import Any

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor
from torch.utils.data import DataLoader

from spd.models.base import Model, SPDFullRankModel, SPDRankPenaltyModel
from spd.run_spd import Config
from spd.utils import calc_topk_mask, calculate_attributions


def plot_subnetwork_attributions_statistics(
    topk_mask: Float[Tensor, "batch_size n_instances k"],
) -> dict[str, plt.Figure]:
    """Plot vertical bar charts of the number of active subnetworks over the batch for each instance."""
    batch_size = topk_mask.shape[0]
    if topk_mask.ndim == 2:
        n_instances = 1
        topk_mask = einops.repeat(topk_mask, "batch k -> batch n_instances k", n_instances=1)
    else:
        n_instances = topk_mask.shape[1]

    fig, axs = plt.subplots(
        ncols=n_instances, nrows=1, figsize=(5 * n_instances, 5), constrained_layout=True
    )

    axs = np.array([axs]) if n_instances == 1 else np.array(axs)
    for i, ax in enumerate(axs):
        values = topk_mask[:, i].sum(dim=1).cpu().detach().numpy()
        bins = list(range(int(values.min().item()), int(values.max().item()) + 2))
        counts, _ = np.histogram(values, bins=bins)
        bars = ax.bar(bins[:-1], counts, align="center", width=0.8)
        ax.set_xticks(bins[:-1])
        ax.set_xticklabels([str(b) for b in bins[:-1]])

        # Only add y-label to first subplot
        if i == 0:
            ax.set_ylabel("Count")

        ax.set_xlabel("Number of active subnetworks")
        ax.set_title(f"Instance {i+1}")

        # Add value annotations on top of each bar
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

    fig.suptitle(f"Active subnetworks on current batch (batch_size={batch_size})")
    return {"subnetwork_attributions_statistics": fig}


def plot_subnetwork_correlations(
    dataloader: DataLoader[
        tuple[Float[Tensor, "batch n_inputs"] | Float[Tensor, "batch n_instances? n_inputs"], Any]
    ],
    target_model: Model,
    spd_model: SPDFullRankModel | SPDRankPenaltyModel,
    config: Config,
    device: str,
    n_forward_passes: int = 100,
) -> dict[str, plt.Figure]:
    topk_masks = []
    for batch, _ in dataloader:
        batch = batch.to(device=device)
        assert config.topk is not None

        target_out, pre_acts, post_acts = target_model(batch)
        # Get the topk mask
        model_output_spd, layer_acts, inner_acts = spd_model(batch)
        attribution_scores = calculate_attributions(
            model=spd_model,
            batch=batch,
            out=model_output_spd,
            target_out=target_out,
            pre_acts=pre_acts,
            post_acts=post_acts,
            inner_acts=inner_acts,
            attribution_type=config.attribution_type,
        )

        # We always assume the final subnetwork is the one we want to distil
        topk_attrs = (
            attribution_scores[..., :-1] if config.distil_from_target else attribution_scores
        )
        topk_mask = calc_topk_mask(topk_attrs, config.topk, batch_topk=config.batch_topk)

        topk_masks.append(topk_mask)
        if len(topk_masks) > n_forward_passes:
            break
    topk_masks = torch.cat(topk_masks).float()

    if hasattr(spd_model, "n_instances"):
        n_instances = spd_model.n_instances
    else:
        n_instances = 1
        topk_masks = einops.repeat(topk_masks, "batch k -> batch n_instances k", n_instances=1)

    fig, axs = plt.subplots(
        ncols=n_instances, nrows=1, figsize=(5 * n_instances, 5), constrained_layout=True
    )

    axs = np.array([axs]) if n_instances == 1 else np.array(axs)
    im, ax = None, None
    for i, ax in enumerate(axs):
        # Calculate correlation matrix
        corr_matrix = torch.corrcoef(topk_masks[:, i].T).cpu()

        im = ax.matshow(corr_matrix)
        ax.xaxis.set_ticks_position("bottom")
        for l in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                ax.text(
                    j,
                    l,
                    f"{corr_matrix[l, j]:.2f}",
                    ha="center",
                    va="center",
                    color="#EE7777",
                    fontsize=8,
                )
    if (im is not None) and (ax is not None):
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        ax.set_title("Subnetwork Correlation Matrix")
        ax.set_xlabel("Subnetwork")
        ax.set_ylabel("Subnetwork")
    return {"subnetwork_correlation_matrix": fig}
