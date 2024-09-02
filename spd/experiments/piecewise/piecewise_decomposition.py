"""Linear decomposition script."""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import torch
import wandb
from jaxtyping import Float
from matplotlib.colors import CenteredNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor
from tqdm import tqdm

from spd.experiments.piecewise.models import (
    PiecewiseFunctionSPDTransformer,
    PiecewiseFunctionTransformer,
)
from spd.experiments.piecewise.piecewise_dataset import PiecewiseDataset
from spd.experiments.piecewise.trig_functions import generate_trig_functions
from spd.log import logger
from spd.run_spd import Config, PiecewiseConfig, calc_recon_mse, optimize
from spd.utils import (
    BatchedDataLoader,
    calc_attributions,
    init_wandb,
    load_config,
    save_config_to_wandb,
    set_seed,
)

wandb.require("core")


def plot_components(
    model: PiecewiseFunctionSPDTransformer,
    device: str,
    topk: float | None,
    step: int,
    out_dir: Path | None,
    batch_topk: bool,
) -> plt.Figure:
    # Get number of layers
    n_layers = model.n_layers

    # Create a batch of inputs with different control bits active
    x_val = torch.tensor(2.5, device=device)
    batch_size = model.n_inputs - 1  # Assuming first input is for x_val and rest are control bits
    x = torch.zeros(batch_size, model.n_inputs, device=device)
    x[:, 0] = x_val
    x[torch.arange(batch_size), torch.arange(1, batch_size + 1)] = 1

    # Forward pass
    out, layer_acts, inner_acts = model(x)

    # Calculate attribution scores
    attribution_scores = calc_attributions(out, inner_acts)
    n_functions = attribution_scores.shape[0]

    # Create figure with subplots
    fig, axes_ = plt.subplots(2 * n_layers, 4, figsize=(40, 10), constrained_layout=True)
    axes: np.ndarray[plt.Axes] = axes_  # type: ignore
    plt.suptitle(f"Subnetwork Analysis (Step {step})")

    # Plot attribution scores
    im1 = axes[0, 0].matshow(
        attribution_scores.detach().cpu().numpy(),
        cmap="coolwarm",
        norm=CenteredNorm(),
    )
    axes[0, 0].set_yticks(range(n_functions))
    axes[0, 0].set_yticklabels(range(1, n_functions + 1))
    axes[0, 0].set_ylabel("Function index")
    axes[0, 0].set_xlabel("Subnetwork index")
    axes[0, 0].set_title("Raw attribution Scores")
    divider = make_axes_locatable(axes[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax, format=tkr.FormatStrFormatter("%.0f"))

    # Plot normalized attribution scores
    attribution_scores_normed = attribution_scores / attribution_scores.std(dim=1, keepdim=True)
    im2 = axes[0, 1].matshow(
        attribution_scores_normed.detach().cpu().numpy(),
        cmap="coolwarm",
        norm=CenteredNorm(),
    )
    axes[0, 1].set_yticks(range(n_functions))
    axes[0, 1].set_yticklabels(range(1, n_functions + 1))
    axes[0, 1].set_ylabel("Function index")
    axes[0, 1].set_xlabel("Subnetwork index")
    axes[0, 1].set_title("Normalized attribution Scores")
    divider = make_axes_locatable(axes[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax, format=tkr.FormatStrFormatter("%.0f"))

    assert len(model.all_As()) == len(model.all_Bs()), "A and B matrices must have the same length"
    assert len(model.all_As()) % 2 == 0, "A and B matrices must have an even length (MLP in + out)"
    assert len(model.all_As()) // 2 == n_layers, "Number of A and B matrices must be 2*n_layers"

    As = model.all_As()
    Bs = model.all_Bs()
    ABs = [torch.einsum("...fk,...kg->...fg", As[i], Bs[i]) for i in range(len(As))]
    for n in range(n_layers):
        s_row = 2 * n  # the row where we put the small plots
        l_row = s_row + 1  # the row where we put the large plots
        assert n_layers == 1, "Current implementation only supports 1 layer"

        # Plot A of W_in
        im1 = axes[s_row, 2].matshow(
            As[2 * n].detach().cpu().numpy(), cmap="coolwarm", norm=CenteredNorm()
        )
        axes[s_row, 2].set_ylabel("Embedding index")
        axes[s_row, 2].set_xlabel("Subnetwork index")
        axes[s_row, 2].set_title(f"A (W_in, layer {n})")
        divider = make_axes_locatable(axes[s_row, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax, format=tkr.FormatStrFormatter("%.1f"))

        # Plot B of W_out
        im2 = axes[s_row, 3].matshow(
            Bs[2 * n + 1].T.detach().cpu().numpy(), cmap="coolwarm", norm=CenteredNorm()
        )
        axes[s_row, 3].set_ylabel("Embedding index")
        axes[s_row, 3].set_xlabel("Subnetwork index")
        axes[s_row, 3].set_title(f"B (W_out, layer {n})")
        divider = make_axes_locatable(axes[s_row, 3])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2, cax=cax, format=tkr.FormatStrFormatter("%.1f"))

        # Plot B of W_in in 2nd row
        im3 = axes[l_row, 2].matshow(
            Bs[2 * n].detach().cpu().numpy(), cmap="coolwarm", norm=CenteredNorm()
        )
        axes[l_row, 2].set_ylabel("Subnetwork index")
        axes[l_row, 2].set_xlabel("Neuron index")
        axes[l_row, 2].set_title(f"B (W_in, layer {n})")
        divider = make_axes_locatable(axes[l_row, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im3, cax=cax, format=tkr.FormatStrFormatter("%.2f"))

        # Plot A of W_out in 2nd row
        im4 = axes[l_row, 3].matshow(
            As[2 * n + 1].T.detach().cpu().numpy(), cmap="coolwarm", norm=CenteredNorm()
        )
        axes[l_row, 3].set_ylabel("Subnetwork index")
        axes[l_row, 3].set_xlabel("Neuron index")
        axes[l_row, 3].set_title(f"A (W_out, layer {n})")
        divider = make_axes_locatable(axes[l_row, 3])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im4, cax=cax, format=tkr.FormatStrFormatter("%.2f"))

        # Plot AB product in 2nd row pos 1 and 2
        im5 = axes[l_row, 0].matshow(
            ABs[n].detach().cpu().numpy(), cmap="coolwarm", norm=CenteredNorm()
        )
        axes[l_row, 0].set_ylabel("Embedding index")
        axes[l_row, 0].set_xlabel("Neuron index")
        axes[l_row, 0].set_title(f"AB Product (W_in, layer {n})")
        divider = make_axes_locatable(axes[l_row, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im5, cax=cax, format=tkr.FormatStrFormatter("%.2f"))

        im6 = axes[l_row, 1].matshow(
            ABs[n + 1].detach().cpu().numpy().T, cmap="coolwarm", norm=CenteredNorm()
        )
        axes[l_row, 1].set_ylabel("Embedding index")
        axes[l_row, 1].set_xlabel("Neuron index")
        axes[l_row, 1].set_title(f"AB Product Transposed (W_out.T, layer {n})")
        divider = make_axes_locatable(axes[l_row, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im6, cax=cax, format=tkr.FormatStrFormatter("%.2f"))

    if out_dir:
        fig.savefig(out_dir / f"subnetwork_analysis_{step}.png")
        plt.close(fig)
        tqdm.write(f"Saved subnetwork analysis to {out_dir / f'subnetwork_analysis_{step}.png'}\n")

    return fig


def get_run_name(config: Config) -> str:
    """Generate a run name based on the config."""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        assert isinstance(config.task_config, PiecewiseConfig)
        run_suffix = (
            f"lay{config.task_config.n_layers}_"
            f"lr{config.lr}_"
            f"p{config.pnorm}_"
            f"topk{config.topk}_"
            f"topkrecon{config.topk_recon_coeff}_"
            f"lpsp{config.lp_sparsity_coeff}_"
            f"topkl2_{config.topk_l2_coeff}_"
            f"bs{config.batch_size}"
        )
        if config.task_config.handcoded_AB:
            run_suffix += "_hAB"
    return config.wandb_run_name_prefix + run_suffix


def get_model_and_dataloader(
    config: Config,
    device: str,
    out_dir: Path | None = None,
) -> tuple[
    PiecewiseFunctionTransformer,
    PiecewiseFunctionSPDTransformer,
    BatchedDataLoader[tuple[Float[Tensor, " n_inputs"], Float[Tensor, ""]]],
    BatchedDataLoader[tuple[Float[Tensor, " n_inputs"], Float[Tensor, ""]]],
]:
    """Set up the piecewise models and dataset."""
    assert isinstance(config.task_config, PiecewiseConfig)
    functions, function_params = generate_trig_functions(config.task_config.n_functions)

    if out_dir:
        with open(out_dir / "function_params.json", "w") as f:
            json.dump(function_params, f, indent=4)
        logger.info(f"Saved function params to {out_dir / 'function_params.json'}")

    piecewise_model = PiecewiseFunctionTransformer.from_handcoded(
        functions=functions,
        neurons_per_function=config.task_config.neurons_per_function,
        n_layers=config.task_config.n_layers,
        range_min=config.task_config.range_min,
        range_max=config.task_config.range_max,
        seed=config.seed,
        simple_bias=config.task_config.simple_bias,
    ).to(device)
    piecewise_model.eval()

    input_biases = [
        piecewise_model.mlps[i].input_layer.bias.detach().clone()
        for i in range(piecewise_model.n_layers)
    ]
    piecewise_model_spd = PiecewiseFunctionSPDTransformer(
        n_inputs=piecewise_model.n_inputs,
        d_mlp=piecewise_model.d_mlp,
        n_layers=piecewise_model.n_layers,
        k=config.task_config.k,
        input_biases=input_biases,
    )
    if config.task_config.handcoded_AB:
        logger.info("Setting handcoded A and B matrices (!)")
        piecewise_model_spd.set_handcoded_AB(piecewise_model)
    piecewise_model_spd.to(device)

    # Set requires_grad to False for all embeddings and all input biases
    for i in range(piecewise_model_spd.n_layers):
        piecewise_model_spd.mlps[i].bias1.requires_grad_(False)
    piecewise_model_spd.W_E.requires_grad_(False)
    piecewise_model_spd.W_U.requires_grad_(False)

    dataset = PiecewiseDataset(
        n_inputs=piecewise_model.n_inputs,
        functions=functions,
        feature_probability=config.task_config.feature_probability,
        range_min=config.task_config.range_min,
        range_max=config.task_config.range_max,
        batch_size=config.batch_size,
        return_labels=False,
    )
    dataloader = BatchedDataLoader(dataset)

    test_dataset = PiecewiseDataset(
        n_inputs=piecewise_model.n_inputs,
        functions=functions,
        feature_probability=config.task_config.feature_probability,
        range_min=config.task_config.range_min,
        range_max=config.task_config.range_max,
        batch_size=config.batch_size,
        return_labels=True,
    )
    test_dataloader = BatchedDataLoader(test_dataset)

    return piecewise_model, piecewise_model_spd, dataloader, test_dataloader


def main(
    config_path_or_obj: Path | str | Config, sweep_config_path: Path | str | None = None
) -> None:
    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, sweep_config_path)
        save_config_to_wandb(config)

    set_seed(config.seed)
    logger.info(config)

    run_name = get_run_name(config)
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    assert isinstance(config.task_config, PiecewiseConfig)
    assert config.task_config.k is not None

    out_dir = Path(__file__).parent / "out" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    piecewise_model, piecewise_model_spd, dataloader, test_dataloader = get_model_and_dataloader(
        config, device, out_dir
    )

    # Evaluate the hardcoded model on 5 batches to get the labels
    n_batches = 5
    loss = 0

    for i, (batch, labels) in enumerate(test_dataloader):
        if i >= n_batches:
            break
        hardcoded_out = piecewise_model(batch.to(device))
        loss += calc_recon_mse(hardcoded_out, labels.to(device))
    loss /= n_batches
    logger.info(f"Loss of hardcoded model on 5 batches: {loss}")

    optimize(
        model=piecewise_model_spd,
        config=config,
        out_dir=out_dir,
        device=device,
        pretrained_model=piecewise_model,
        dataloader=dataloader,
        plot_results_fn=plot_components,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
