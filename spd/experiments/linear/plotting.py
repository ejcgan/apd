from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from matplotlib.colors import CenteredNorm
from torch import Tensor
from tqdm import tqdm

from spd.experiments.linear.models import (
    DeepLinearComponentFullRankModel,
    DeepLinearComponentModel,
)
from spd.utils import (
    calc_grad_attributions_full_rank_per_layer,
    calc_grad_attributions_rank_one_per_layer,
    permute_to_identity,
)


def _collect_permuted_subnetwork_attributions(
    model: DeepLinearComponentModel | DeepLinearComponentFullRankModel,
    device: str,
) -> tuple[
    Float[Tensor, "batch n_instances n_features"], list[Float[Tensor, "batch n_instances k"]]
]:
    """
    Collect subnetwork attributions and permute them for visualization.

    This function creates a test batch using an identity matrix, passes it through the model,
    and collects the attributions, and then permutes them to align with the identity.

    Args:
        model (DeepLinearComponentModel | DeepLinearComponentFullRankModel): The model to collect
            attributions on.
        device (str): The device to run computations on.

    Returns:
        - The input test batch (identity matrix expanded over instance dimension).
        - A list of permuted attributions for each layer.
    """
    test_batch = einops.repeat(
        torch.eye(model.n_features, device=device),
        "batch n_features -> batch n_instances n_features",
        n_instances=model.n_instances,
    )

    out, test_layer_acts, test_inner_acts = model(test_batch)
    if isinstance(model, DeepLinearComponentModel):
        layer_attributions = calc_grad_attributions_rank_one_per_layer(
            out=out, inner_acts=test_inner_acts
        )
    else:
        assert isinstance(model, DeepLinearComponentFullRankModel)
        layer_attributions = calc_grad_attributions_full_rank_per_layer(
            out=out, inner_acts=test_inner_acts, layer_acts=test_layer_acts
        )

    test_attributions_permuted = []
    for layer in range(model.n_layers):
        test_attributions_layer_permuted = []
        for i in range(model.n_instances):
            test_attributions_layer_permuted.append(
                permute_to_identity(layer_attributions[layer][:, i, :].abs())
            )
        test_attributions_permuted.append(torch.stack(test_attributions_layer_permuted, dim=1))

    return test_batch, test_attributions_permuted


def plot_subnetwork_grad_attributions_fn(
    batch: Float[Tensor, "batch n_instances n_features"],
    attributions: list[Float[Tensor, "batch n_instances k"]],
    step: int | None = None,
    n_instances: int | None = None,
) -> plt.Figure:
    """Plot the gradient attributions per layer, as well as the raw input batch.

    The first row of the plot is a matrix of shape (batch, n_features) where a unique features is
    active in each element of the batch.
    Subsequent rows are attributions for each layer of shape (batch, k).
    Each column is a new instance.
    """
    n_layers = len(attributions)
    assert n_instances is None or n_instances <= batch.shape[1]
    n_instances = batch.shape[1] if n_instances is None else n_instances

    fig, axs = plt.subplots(
        n_layers + 1,
        n_instances,
        figsize=(2.5 * n_instances, 2.5 * (n_layers + 1)),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )

    cmap = "Blues"

    for i in range(n_instances):
        ax = axs[0, i]
        data = batch[:, i, :].detach().cpu().float().numpy()
        ax.matshow(data, vmin=0, vmax=np.max(data), cmap=cmap)

        if n_instances > 1:
            ax.set_title(f"Instance {i}")
        if i == 0:
            ax.set_ylabel("Inputs")
        elif i == n_instances - 1:
            ax.set_ylabel("batch_idx", rotation=-90, va="bottom", labelpad=15)
            ax.yaxis.set_label_position("right")

        # Set an xlabel for each plot
        ax.set_xlabel("n_features")

        ax.set_xticks([])
        ax.set_yticks([])

    # Add the attributions
    for layer in range(n_layers):
        for i in range(n_instances):
            ax = axs[layer + 1, i]
            instance_data = attributions[layer][:, i, :].abs().detach().cpu().float().numpy()
            ax.matshow(instance_data, vmin=0, vmax=np.max(instance_data), cmap=cmap)

            if i == 0:
                ax.set_ylabel(f"Layer {layer}")
            elif i == n_instances - 1:
                ax.set_ylabel("batch_idx", rotation=-90, va="bottom", labelpad=15)
                ax.yaxis.set_label_position("right")

            if layer == n_layers - 1:
                ax.set_xlabel("k")

            ax.set_xticks([])
            ax.set_yticks([])

    title_text = "Subnet Gradient Attributions"
    if step is not None:
        title_text += f" (Step {step})"
    fig.suptitle(title_text)
    return fig


def plot_multiple_subnetwork_params(
    model: DeepLinearComponentModel | DeepLinearComponentFullRankModel,
    step: int | None = None,
    n_instances: int | None = None,
) -> plt.Figure:
    """Plot each subnetwork parameter matrix."""
    all_params = model.all_subnetwork_params()
    # Each param (of which there are n_layers): [n_instances, k, n_features, n_features]
    n_params = len(all_params)
    assert n_params >= 1

    param_n_instances, k, dim1, dim2 = all_params[0].shape

    assert n_instances is None or n_instances <= param_n_instances
    n_instances = param_n_instances if n_instances is None else n_instances

    fig, axs = plt.subplots(
        n_instances * n_params,
        k,
        figsize=(1 * k, 1 * n_instances * n_params),
        constrained_layout=True,
    )

    for inst_idx in range(n_instances):
        for param_idx in range(n_params):
            for subnet_idx in range(k):
                col_idx = subnet_idx
                row_idx = param_idx + inst_idx * n_params

                ax = axs[row_idx, col_idx]  # type: ignore
                param = all_params[param_idx][inst_idx, subnet_idx].detach().cpu().numpy()
                ax.matshow(param, cmap="RdBu", norm=CenteredNorm())
                ax.set_xticks([])
                ax.set_yticks([])

                if col_idx == 0:
                    label = f"Layer {param_idx}"
                    if n_instances > 1:
                        label = f"Inst {inst_idx} {label}"
                    ax.set_ylabel(label, rotation=0, ha="right", va="center")

                if row_idx == n_instances * n_params - 1:
                    ax.set_xlabel(f"Subnet {subnet_idx}", rotation=0, ha="center", va="top")

    title_text = "Subnet Parameters"
    if step is not None:
        title_text += f" (Step {step})"
    fig.suptitle(title_text)
    return fig


def make_linear_plots(
    model: DeepLinearComponentModel | DeepLinearComponentFullRankModel,
    step: int | None,
    out_dir: Path | None,
    device: str,
    n_instances: int | None = None,
    **_,
) -> dict[str, plt.Figure]:
    test_batch, test_attributions = _collect_permuted_subnetwork_attributions(model, device)

    act_fig = plot_subnetwork_grad_attributions_fn(
        batch=test_batch, attributions=test_attributions, step=step, n_instances=n_instances
    )
    if out_dir is not None:
        filename = (
            f"layer_grad_attributions_{step}.png"
            if step is not None
            else "layer_grad_attributions.png"
        )
        act_fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
        tqdm.write(f"Saved layer_grad_attributions to {out_dir / filename}")
    plt.close(act_fig)

    param_fig = plot_multiple_subnetwork_params(model=model, step=step, n_instances=n_instances)
    if out_dir is not None:
        filename = f"subnetwork_params_{step}.png" if step is not None else "subnetwork_params.png"
        param_fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
        tqdm.write(f"Saved subnetwork_params to {out_dir / filename}")
    plt.close(param_fig)

    return {"layer_attributions": act_fig, "subnetwork_params": param_fig}
