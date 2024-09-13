import json

import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from spd.run_spd import (
    Config,
)
from spd.utils import REPO_ROOT


def plot_vectors(
    subnet: Float[Tensor, "n_instances n_subnets n_features n_hidden"],
    n_instances: int | None = None,
) -> plt.Figure:
    """2D polygon plot of each subnetwork.

    Adapted from
    https://colab.research.google.com/github/anthropics/toy-models-of-superposition/blob/main/toy_models.ipynb.
    """
    n_data_instances, n_subnets, n_features, n_hidden = subnet.shape
    if n_instances is None:
        n_instances = n_data_instances
    else:
        assert (
            n_instances <= n_data_instances
        ), "n_instances must be less than or equal to n_data_instances"

    sel = range(n_instances)

    # Use different colors for each subnetwork if there's only one instance
    color_vals = np.linspace(0, 1, n_features) if n_instances == 1 else np.zeros(n_features)
    colors = plt.cm.viridis(color_vals)  # type: ignore

    fig, axs = plt.subplots(len(sel), n_subnets + 1, figsize=(2 * (n_subnets + 1), 2 * (len(sel))))
    axs = np.atleast_2d(np.array(axs))
    for j in range(n_subnets + 1):
        for i, ax in enumerate(axs[:, j]):
            if j == 0:
                # First, plot the addition of the subnetworks
                arr = subnet[i].sum(dim=0).cpu().detach().numpy()
            else:
                # Plot the jth subnet
                arr = subnet[i, j - 1].cpu().detach().numpy()

            # Plot each feature with its unique color
            for k in range(n_features):
                ax.scatter(arr[k, 0], arr[k, 1], color=colors[k])
                ax.add_collection(
                    mc.LineCollection([[(0, 0), (arr[k, 0], arr[k, 1])]], colors=[colors[k]])
                )

            ax.set_aspect("equal")
            z = 1.5
            ax.set_facecolor("#FCFBF8")
            ax.set_xlim((-z, z))
            ax.set_ylim((-z, z))
            ax.tick_params(left=True, right=False, labelleft=False, labelbottom=False, bottom=True)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["bottom", "left"]:
                ax.spines[spine].set_position("center")

            if i == len(sel) - 1:
                label = "Sum of subnets" if j == 0 else f"Subnet {j-1}"
                ax.set_xlabel(label, rotation=0, ha="center", labelpad=60)
            if j == 0 and n_instances > 1:
                ax.set_ylabel(f"Instance {i}", rotation=90, ha="center", labelpad=60)

    return fig


if __name__ == "__main__":
    pretrained_path = REPO_ROOT / "spd/experiments/tms/demo_spd_model/model_30000.pth"

    with open(pretrained_path.parent / "config.json") as f:
        config_dict = json.load(f)
        config = Config(**config_dict)

    assert config.full_rank, "This script only works for full rank models"
    subnet = torch.load(pretrained_path, map_location="cpu")["subnetwork_params"]
    fig = plot_vectors(subnet, n_instances=1)
    fig.savefig(pretrained_path.parent / "polygon_diagram.png", bbox_inches="tight", dpi=200)
    print(f"Saved figure to {pretrained_path.parent / 'polygon_diagram.png'}")
