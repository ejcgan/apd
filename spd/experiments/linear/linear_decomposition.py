"""Linear decomposition script."""

from pathlib import Path

import einops
import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from jaxtyping import Float
from matplotlib.colors import CenteredNorm
from torch import Tensor
from tqdm import tqdm

from spd.experiments.linear.linear_dataset import DeepLinearDataset
from spd.experiments.linear.models import (
    DeepLinearComponentFullRankModel,
    DeepLinearComponentModel,
    DeepLinearModel,
)
from spd.log import logger
from spd.run_spd import Config, DeepLinearConfig, optimize
from spd.utils import (
    DatasetGeneratedDataLoader,
    calc_attributions_full_rank_per_layer,
    calc_attributions_rank_one_per_layer,
    init_wandb,
    load_config,
    permute_to_identity,
    save_config_to_wandb,
    set_seed,
)

wandb.require("core")


def get_run_name(config: Config) -> str:
    """Generate a run name based on the config."""
    run_suffix = ""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        assert isinstance(config.task_config, DeepLinearConfig)
        if config.pnorm is not None:
            run_suffix += f"p{config.pnorm:.2e}_"
        if config.lp_sparsity_coeff is not None:
            run_suffix += f"lpsp{config.lp_sparsity_coeff:.2e}_"
        if config.topk is not None:
            run_suffix += f"topk{config.topk:.2e}_"
        if config.topk_recon_coeff is not None:
            run_suffix += f"topkrecon{config.topk_recon_coeff:.2e}_"
        if config.topk_l2_coeff is not None:
            run_suffix += f"topkl2_{config.topk_l2_coeff:.2e}_"
        run_suffix += f"lr{config.lr:.2e}_"
        run_suffix += f"bs{config.batch_size}_"
        run_suffix += f"ft{config.task_config.n_features}_"
        run_suffix += f"hid{config.task_config.n_layers}"
    return config.wandb_run_name_prefix + run_suffix


def _plot_multiple_subnetwork_params(
    model: DeepLinearComponentModel | DeepLinearComponentFullRankModel, step: int
) -> plt.Figure:
    """Plot each subnetwork parameter matrix."""
    all_params = model.all_subnetwork_params()
    # Each param (of which there are n_layers): [n_instances, k, n_features, n_features]
    n_params = len(all_params)
    assert n_params >= 1

    n_instances, k, dim1, dim2 = all_params[0].shape

    fig, axs = plt.subplots(
        k,
        n_instances * n_params,
        # + n_params - 1 to account for the space between the subplots
        figsize=(2 * n_instances * n_params + n_params - 1, 2 * k),
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
    )

    for inst_idx in range(n_instances):
        for param_idx in range(n_params):
            for k_idx in range(k):
                row_idx = k_idx
                column_idx = param_idx + inst_idx * n_params
                ax = axs[row_idx, column_idx]  # type: ignore
                param = all_params[param_idx][inst_idx, k_idx].detach().cpu().numpy()
                ax.matshow(param, cmap="RdBu", norm=CenteredNorm())
                ax.set_xticks([])
                ax.set_yticks([])

                if column_idx == 0:
                    ax.set_ylabel(f"k={k_idx}", rotation=0, ha="right", va="center")
                if k_idx == k - 1:
                    ax.set_xlabel(f"Inst {inst_idx} Param {param_idx}", rotation=0, ha="center")

    fig.suptitle(f"Subnetwork Parameters (Step {step})")
    return fig


def _plot_subnetwork_attributions_fn(
    batch: Float[Tensor, "batch n_instances n_features"],
    attributions: list[Float[Tensor, "batch n_instances k"]],
) -> plt.Figure:
    """Plot the attributions for the first batch_elements in the batch.

    The first row is the raw batch information, the following rows are the attributions per layer.
    """
    n_layers = len(attributions)
    n_instances = batch.shape[1]

    fig, axs = plt.subplots(
        n_layers + 1,
        n_instances,
        figsize=(2.5 * n_instances, 2.5 * (n_layers + 1)),
        squeeze=False,
        sharey=True,
    )

    cmap = "Blues"
    # Add the batch data
    for i in range(n_instances):
        ax = axs[0, i]
        data = batch[:, i, :].detach().cpu().float().numpy()
        ax.matshow(data, vmin=0, vmax=np.max(data), cmap=cmap)

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

    plt.tight_layout()
    return fig


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
        layer_attributions = calc_attributions_rank_one_per_layer(
            out=out, inner_acts=test_inner_acts
        )
    else:
        assert isinstance(model, DeepLinearComponentFullRankModel)
        layer_attributions = calc_attributions_full_rank_per_layer(
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


def make_linear_plots(
    model: DeepLinearComponentModel,
    step: int,
    out_dir: Path | None,
    device: str,
    **_,
) -> dict[str, plt.Figure]:
    test_batch, test_attributions = _collect_permuted_subnetwork_attributions(model, device)

    act_fig = _plot_subnetwork_attributions_fn(batch=test_batch, attributions=test_attributions)
    if out_dir is not None:
        act_fig.savefig(out_dir / f"layer_attributions_{step}.png")
    plt.close(act_fig)

    param_fig = _plot_multiple_subnetwork_params(model, step)
    if out_dir is not None:
        param_fig.savefig(out_dir / f"subnetwork_params_{step}.png", dpi=300, bbox_inches="tight")
    plt.close(param_fig)

    if out_dir is not None:
        tqdm.write(f"Saved layer_attributions to {out_dir / f'layer_attributions_{step}.png'}")
        tqdm.write(f"Saved subnetwork_params to {out_dir / f'subnetwork_params_{step}.png'}")
    return {"layer_attributions": act_fig, "subnetwork_params": param_fig}


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
    out_dir = Path(__file__).parent / "out" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    assert isinstance(config.task_config, DeepLinearConfig)

    if config.task_config.pretrained_model_path is not None:
        dl_model = DeepLinearModel.from_pretrained(config.task_config.pretrained_model_path).to(
            device
        )
        assert (
            config.task_config.n_features is None
            and config.task_config.n_layers is None
            and config.task_config.n_instances is None
        ), "n_features, n_layers, and n_instances must not be set if pretrained_model_path is set"
        n_features = dl_model.n_features
        n_layers = dl_model.n_layers
        n_instances = dl_model.n_instances
    else:
        assert config.out_recon_coeff is not None, "Only out recon loss allows no pretrained model"
        dl_model = None
        n_features = config.task_config.n_features
        n_layers = config.task_config.n_layers
        n_instances = config.task_config.n_instances
        assert (
            n_features is not None and n_layers is not None and n_instances is not None
        ), "n_features, n_layers, and n_instances must be set"

    if config.full_rank:
        dlc_model = DeepLinearComponentFullRankModel(
            n_features=n_features,
            n_layers=n_layers,
            n_instances=n_instances,
            k=config.task_config.k,
        ).to(device)
    else:
        dlc_model = DeepLinearComponentModel(
            n_features=n_features,
            n_layers=n_layers,
            n_instances=n_instances,
            k=config.task_config.k,
        ).to(device)

    dataset = DeepLinearDataset(n_features, n_instances)
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimize(
        model=dlc_model,
        config=config,
        out_dir=out_dir,
        device=device,
        dataloader=dataloader,
        pretrained_model=dl_model,
        plot_results_fn=make_linear_plots,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
