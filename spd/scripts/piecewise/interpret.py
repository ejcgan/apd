# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from jaxtyping import Float
from torch import Tensor

from spd.models.piecewise_models import (
    PiecewiseFunctionSPDTransformer,
    PiecewiseFunctionTransformer,
)
from spd.run_spd import Config, PiecewiseConfig
from spd.scripts.piecewise.trig_functions import create_trig_function
from spd.utils import calc_attributions


def make_plot(pretrained_path: Path, title: str, plot_all: bool = False) -> None:
    with open(pretrained_path.parent / "config.json") as f:
        config = Config(**json.load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(pretrained_path.parent / "function_params.json") as f:
        function_params = json.load(f)
    functions = [create_trig_function(*param) for param in function_params]

    assert isinstance(config.task_config, PiecewiseConfig)
    hardcoded_model = PiecewiseFunctionTransformer.from_handcoded(
        functions=functions,
        neurons_per_function=config.task_config.neurons_per_function,
        n_layers=config.task_config.n_layers,
        range_min=config.task_config.range_min,
        range_max=config.task_config.range_max,
        seed=config.seed,
    ).to(device)
    hardcoded_model.eval()

    model = PiecewiseFunctionSPDTransformer(
        n_inputs=hardcoded_model.n_inputs,
        d_mlp=hardcoded_model.d_mlp,
        n_layers=hardcoded_model.n_layers,
        k=config.task_config.k,
        d_embed=hardcoded_model.d_embed,
    )
    model.load_state_dict(torch.load(pretrained_path, weights_only=True, map_location="cpu"))
    model.to(device)

    hardcoded_weights = hardcoded_model.all_decomposable_params()
    param_match_loss = torch.zeros(1, device=device)
    for i, (A, B) in enumerate(zip(model.all_As(), model.all_Bs(), strict=True)):
        normed_A = A / A.norm(p=2, dim=-2, keepdim=True)
        AB = torch.einsum("...fk,...kg->...fg", normed_A, B)
        param_match_loss = param_match_loss + ((AB - hardcoded_weights[i]) ** 2).mean(dim=(-2, -1))
    param_match_loss = param_match_loss / model.n_param_matrices

    # Step 1: Create a batch of inputs with different control bits active
    x_val = torch.tensor(2.5)
    batch_size = len(functions)
    true_labels = torch.tensor([f(x_val) for f in functions])

    x = torch.zeros(batch_size, hardcoded_model.n_inputs).to(device)
    x[:, 0] = x_val.item()
    x[torch.arange(batch_size), torch.arange(1, batch_size + 1)] = 1

    # Step 2: Forward pass on the spd model
    out, layer_acts, inner_acts = model(x)
    # Also check the hardcoded model
    out_hardcoded = hardcoded_model(x)

    # Step 3: Get attribution scores by doing a backward pass on the spd model
    attribution_scores = calc_attributions(out, inner_acts)

    print(f"Attribution scores: {attribution_scores}")
    # Plot a matshow of the attribution scores
    # Each row should have it's own color scale
    # Normalize each row to have mean 0 and std 1
    attribution_scores_normed = (
        attribution_scores - attribution_scores.mean(dim=1, keepdim=True)
    ) / attribution_scores.std(dim=1, keepdim=True)

    # Find the max absolute value in the attribution scores
    # max_abs_value = attribution_scores_normed.abs().max()
    max_abs_value = attribution_scores.abs().max()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    [ax, ax2] = axes  # type: ignore
    # matshow
    fig.suptitle(title)
    im = ax.matshow(
        # attribution_scores_normed.detach().cpu().numpy(),
        attribution_scores.detach().cpu().numpy(),
        cmap="coolwarm",
        vmin=-max_abs_value,
        vmax=max_abs_value,
    )
    # ylabel should be the function index
    ax.set_ylabel("Function index")
    ax.set_xlabel("subnetwork index")
    # Add title saying it's the attribution scores
    # ax.set_title("Attribution scores (20k steps)")
    ax.set_title("Attribution scores raw")
    # Use coolwarm colormap
    assert ax.figure is not None
    cbar = ax.figure.colorbar(im)
    cbar.ax.set_ylabel("Attribution score", rotation=-90, va="bottom")

    def plot_component(
        x: Float[Tensor, "dim1 dim2"],
        ylabel: str,
        xlabel: str,
        title: str,
        save_path: Path,
        ax: plt.Axes | None = None,
    ) -> None:
        ax = ax or plt.subplots()[1]
        max_abs_value = x.abs().max()
        im = ax.matshow(
            x.detach().cpu().numpy(),
            cmap="coolwarm",
            vmin=-max_abs_value,
            vmax=max_abs_value,
        )
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        assert ax.figure is not None
        cbar = ax.figure.colorbar(im)
        cbar.ax.set_ylabel(title, rotation=-90, va="bottom")
        plt.savefig(save_path)
        plt.show()

    plot_component(
        x=model.input_component / model.input_component.norm(p=2, dim=-2, keepdim=True),
        ylabel="Input index",
        xlabel="subnetwork index",
        title="Normed input component",
        save_path=Path("out/normed_A_matrix.png"),
        ax=ax2,
    )

    fig.savefig(f"out/attribution_scores/{title}.png")

    if plot_all:
        print(f"Param match loss: {param_match_loss}")
        print(f"Input: {x_val}, True labels: {true_labels}")
        print(f"out: {out}")
        print(f"out_hardcoded: {out_hardcoded}")
        print(f"Attribution scores: {attribution_scores}")
        plot_component(
            x=model.output_component.T,
            ylabel="Output index",
            xlabel="subnetwork index",
            title="Output component",
            save_path=Path("out/normed_B_matrix.png"),
        )

        # Our topk outputs should be similar to the true labels
        # Get the top 4 attribution abs values for each function
        top_k_indices = attribution_scores.abs().topk(4, dim=-1).indices

        # Do a forward_topk pass
        out_topk, layer_acts_topk, inner_acts_topk = model.forward_topk(x, top_k_indices)
        print(f"Top-k output: {out_topk}")
        # Print the labels
        print(f"Top-k labels: {true_labels}")


# %%

if __name__ == "__main__":
    pretrained_paths = Path("out/").rglob("reproduce_good*/model_19999.pth")
    for pretrained_path in pretrained_paths:
        pretrained_path = Path(
            "/data/stefan_heimersheim/projects/SPD/spd/spd/scripts/piecewise/" / pretrained_path
        )
        print(f"Processing {pretrained_path}")
        try:
            make_plot(pretrained_path, title=pretrained_path.parent.name)
        except Exception as e:
            print(f"Error processing {pretrained_path}: {e}")
