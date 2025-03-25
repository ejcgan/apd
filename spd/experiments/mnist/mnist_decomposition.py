"""Run SPD on MNIST models."""

from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from spd.experiments.mnist.models import (
    MNISTModel,
    MNISTModelConfig,
    MNISTSPDModel,
    MNISTSPDModelConfig,
)
from spd.log import logger
from spd.run_spd import Config, MNISTTaskConfig, get_common_run_name_suffix, optimize
from spd.utils import load_config, set_seed
from spd.wandb_utils import init_wandb

wandb.require("core")


def get_run_name(config: Config, mnist_model_config: MNISTModelConfig) -> str:
    """Generate a run name based on the config."""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        run_suffix = get_common_run_name_suffix(config)
        run_suffix += f"layers{mnist_model_config.n_layers}_"
        run_suffix += f"hidden{mnist_model_config.hidden_dim}"
    return config.wandb_run_name_prefix + run_suffix


def plot_weight_matrices(
    model: MNISTSPDModel, layer_idx: int, step: int, out_dir: Path
) -> plt.Figure:
    """Plot the component weight matrices for a specific layer."""
    layer = model.layers[layer_idx]
    component_weights = torch.einsum("cam,cmb->cab", layer.A, layer.B)

    # component_weights shape: [C, input_dim, output_dim]
    C, input_dim, output_dim = component_weights.shape

    # Create a grid of subplots
    n_cols = min(5, C)  # Limit columns to 5
    n_rows = (C + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = np.array(axes).flatten()

    # Create a normalization based on the entire matrix set
    vmin = component_weights.min().item()
    vmax = component_weights.max().item()

    # Plot each component
    for c in range(C):
        ax = axes[c]
        im = ax.matshow(
            component_weights[c].detach().cpu().numpy(), cmap="RdBu_r", vmin=vmin, vmax=vmax
        )
        ax.set_title(f"Component {c}")
        ax.axis("off")

    # Hide any unused subplots
    for i in range(C, len(axes)):
        axes[i].axis("off")

    # Add a colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # Set the figure title
    fig.suptitle(f"Layer {layer_idx} Component Weights (Step {step})")

    # Save the figure
    filename = f"layer{layer_idx}_component_weights_s{step}.png"
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return fig


def plot_attribution_heatmap(
    attributions: Float[Tensor, "batch C"], layer_idx: int, step: int, out_dir: Path
) -> plt.Figure:
    """Plot the attribution heatmap for a layer."""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot heatmap
    im = ax.matshow(attributions.detach().cpu().numpy(), aspect="auto", cmap="viridis")

    # Add colorbar
    fig.colorbar(im, ax=ax, label="Attribution Score")

    # Set labels and title
    ax.set_xlabel("Component")
    ax.set_ylabel("Batch Sample")
    ax.set_title(f"Layer {layer_idx} Attribution Scores (Step {step})")

    # Set x and y ticks
    ax.set_xticks(np.arange(attributions.shape[1]))
    # Only show a subset of y ticks if there are many samples
    if attributions.shape[0] > 20:
        y_ticks = np.linspace(0, attributions.shape[0] - 1, 10, dtype=int)
        ax.set_yticks(y_ticks)

    # Save figure
    filename = f"layer{layer_idx}_attribution_heatmap_s{step}.png"
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return fig


def plot_digit_reconstructions(
    original_digit: Tensor, reconstructions: list[Tensor], step: int, out_dir: Path
) -> plt.Figure:
    """Plot the original digit and its reconstructions using different components."""
    n_images = 1 + len(reconstructions)  # Original + reconstructions

    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 2, 2))

    # Plot original digit
    axes[0].imshow(original_digit.reshape(28, 28).detach().cpu().numpy(), cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Plot reconstructions
    for i, recon in enumerate(reconstructions):
        axes[i + 1].imshow(recon.reshape(28, 28).detach().cpu().numpy(), cmap="gray")
        axes[i + 1].set_title(f"Top-{i + 1}")
        axes[i + 1].axis("off")

    # Save figure
    filename = f"digit_reconstructions_s{step}.png"
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return fig


def make_plots(
    model: MNISTSPDModel,
    target_model: MNISTModel,
    step: int,
    out_dir: Path,
    device: str,
    config: Config,
    topk_mask: Float[Tensor, "batch n_instances C"] | None,
    batch: Float[Tensor, "batch input_dim"],
    **_,
) -> dict[str, plt.Figure]:
    """Create plots for MNIST SPD model analysis."""
    plots = {}

    # Plot component weights for each layer
    for layer_idx in range(len(model.layers)):
        layer_key = f"layer{layer_idx}_component_weights"
        plots[layer_key] = plot_weight_matrices(model, layer_idx, step, out_dir)

    if topk_mask is not None:
        # Flatten the instance dimension since we're using 1 instance
        topk_mask = topk_mask.squeeze(1)

        # For a subset of the batch, calculate attributions per layer
        n_samples = min(10, batch.shape[0])
        batch_subset = batch[:n_samples]

        # Run forward pass with gradient tracking to get attributions
        batch_subset.requires_grad_(True)
        target_output = target_model(batch_subset)

        # Get component attributions for each layer
        for layer_idx in range(len(model.layers)):
            with torch.no_grad():
                # This is a simplified approach - in practice you would use the attribution
                # calculation from run_spd.py, but adapted for per-layer analysis
                model_output = model(batch_subset, topk_mask=topk_mask[:n_samples])
                # Just use the magnitude of the layer parameters for this example
                layer = model.layers[layer_idx]
                component_weights = torch.einsum("cam,cmb->cab", layer.A, layer.B)
                attributions = component_weights.norm(dim=(1, 2))
                attributions = attributions.expand(n_samples, -1)  # Expand to batch dimension

            # Plot attribution heatmap
            attr_key = f"layer{layer_idx}_attribution_heatmap"
            plots[attr_key] = plot_attribution_heatmap(attributions, layer_idx, step, out_dir)

    # For visualization, reconstruct a sample digit using different numbers of components
    if step % 5 == 0 and step > 0:  # Do this less frequently
        # Select a single digit for visualization
        sample_digit = batch[0:1]

        # Get reconstructions using top-k components
        reconstructions = []
        for k in range(1, min(5, config.C) + 1):  # Use top 1, 2, 3, 4, 5 components
            # Create a mask that only uses top k components
            with torch.no_grad():
                # This is simplified - you would actually want to determine the top-k
                # components for this specific digit
                temp_mask = torch.zeros(1, 1, config.C, device=device, dtype=torch.bool)
                temp_mask[0, 0, :k] = True  # Just use first k components
                recon = model(sample_digit, topk_mask=temp_mask)
                reconstructions.append(recon.squeeze())

        plots["digit_reconstructions"] = plot_digit_reconstructions(
            sample_digit.squeeze(), reconstructions, step, out_dir
        )

    return plots


def save_target_model_info(
    save_to_wandb: bool,
    out_dir: Path,
    mnist_model: MNISTModel,
    mnist_model_train_config_dict: dict[str, Any],
) -> None:
    """Save target model information to disk and wandb."""
    torch.save(mnist_model.state_dict(), out_dir / "mnist_model.pth")

    with open(out_dir / "mnist_train_config.yaml", "w") as f:
        yaml.dump(mnist_model_train_config_dict, f, indent=2)

    if save_to_wandb:
        wandb.save(str(out_dir / "mnist_model.pth"), base_path=out_dir, policy="now")
        wandb.save(str(out_dir / "mnist_train_config.yaml"), base_path=out_dir, policy="now")


def main(
    config_path_or_obj: Path | str | Config, sweep_config_path: Path | str | None = None
) -> None:
    """Run SPD decomposition on a trained MNIST model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, sweep_config_path)

    task_config = config.task_config
    assert isinstance(task_config, MNISTTaskConfig)

    set_seed(config.seed)
    logger.info(config)

    # Load the pretrained target model
    target_model, target_model_train_config_dict = MNISTModel.from_pretrained(
        task_config.pretrained_model_path
    )
    target_model = target_model.to(device)
    target_model.eval()

    # Set up run name and output directory
    run_name = get_run_name(config=config, mnist_model_config=target_model.config)
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = Path(__file__).parent / "out" / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save configs
    with open(out_dir / "final_config.yaml", "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    if config.wandb_project:
        wandb.save(str(out_dir / "final_config.yaml"), base_path=out_dir, policy="now")

    # Save target model info
    save_target_model_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        mnist_model=target_model,
        mnist_model_train_config_dict=target_model_train_config_dict,
    )

    # Create SPD model
    mnist_spd_model_config = MNISTSPDModelConfig(
        **target_model.config.model_dump(mode="json"),
        C=config.C,
        m=config.m,
    )
    model = MNISTSPDModel(config=mnist_spd_model_config)

    # Prepare MNIST data for SPD optimization
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST("data", train=False, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # List of layer names to match parameters
    param_names = [f"layers.{i}" for i in range(len(model.layers))]

    # Run SPD optimization
    optimize(
        model=model,
        config=config,
        device=device,
        dataloader=dataloader,
        target_model=target_model,
        param_names=param_names,
        out_dir=out_dir,
        plot_results_fn=make_plots,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
