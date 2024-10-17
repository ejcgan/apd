"""Residual Linear decomposition script."""

import json
from functools import partial
from pathlib import Path
from typing import Any

import fire
import matplotlib.pyplot as plt
import torch
import wandb
import yaml
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from spd.experiments.resid_linear.models import ResidualLinearModel, ResidualLinearSPDFullRankModel
from spd.experiments.resid_linear.resid_linear_dataset import (
    ResidualLinearDataset,
)
from spd.log import logger
from spd.plotting import plot_subnetwork_attributions_statistics, plot_subnetwork_correlations
from spd.run_spd import Config, ResidualLinearConfig, optimize
from spd.utils import (
    DatasetGeneratedDataLoader,
    calc_attributions_full_rank,
    calc_attributions_rank_one,
    init_wandb,
    load_config,
    save_config_to_wandb,
    set_seed,
)

wandb.require("core")


def get_run_name(config: Config, n_features: int, n_layers: int, d_resid: int, d_mlp: int) -> str:
    """Generate a run name based on the config."""
    run_suffix = ""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        assert isinstance(config.task_config, ResidualLinearConfig)
        run_suffix = f"seed{config.seed}_"
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
        run_suffix += f"ft{n_features}_lay{n_layers}_resid{d_resid}_mlp{d_mlp}"
    return config.wandb_run_name_prefix + run_suffix


def _collect_subnetwork_attributions(
    model: ResidualLinearSPDFullRankModel, device: str, full_rank: bool
) -> Float[Tensor, "batch k"]:
    """
    Collect subnetwork attributions.

    This function creates a test batch using an identity matrix, passes it through the model,
    and collects the attributions.

    Args:
        model (ResidualLinearSPDFullRankModel): The model to collect attributions on.
        device (str): The device to run computations on.
        full_rank (bool): Whether the model is full rank or rank one.
    Returns:
        The attribution scores.
    """
    test_batch = torch.eye(model.n_features, device=device)

    out, test_layer_acts, test_inner_acts = model(test_batch)

    if full_rank:
        attribution_scores = calc_attributions_full_rank(
            out=out, inner_acts=test_inner_acts, layer_acts=test_layer_acts
        )
    else:
        attribution_scores = calc_attributions_rank_one(
            out=out, inner_acts_vals=list(test_inner_acts.values())
        )
    return attribution_scores


def plot_subnetwork_attributions(
    attribution_scores: Float[Tensor, "batch k"],
    out_dir: Path | None,
    step: int,
) -> dict[str, plt.Figure]:
    """Plot subnetwork attributions."""
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    im = ax.matshow(attribution_scores.detach().cpu().numpy(), aspect="auto", cmap="Reds")
    ax.set_xlabel("Subnetwork Index")
    ax.set_ylabel("Batch Index")
    ax.set_title("Subnetwork Attributions")

    # Annotate each cell with the numeric value
    for i in range(attribution_scores.shape[0]):
        for j in range(attribution_scores.shape[1]):
            ax.text(
                j,
                i,
                f"{attribution_scores[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )
    plt.colorbar(im)
    if out_dir:
        fig.savefig(out_dir / f"subnetwork_attributions_s{step}.png", dpi=200)
    return {"subnetwork_attributions": fig}


def resid_linear_plot_results_fn(
    model: ResidualLinearSPDFullRankModel,
    step: int,
    out_dir: Path | None,
    device: str,
    config: Config,
    topk_mask: Float[Tensor, " batch_size k"] | None,
    dataloader: DatasetGeneratedDataLoader[
        tuple[Float[Tensor, "batch n_features"], Float[Tensor, "batch d_embed"]]
    ]
    | None = None,
    **_,
) -> dict[str, plt.Figure]:
    assert isinstance(config.task_config, ResidualLinearConfig)
    fig_dict = {}

    attribution_scores = _collect_subnetwork_attributions(model, device, config.full_rank)
    fig_dict_attributions = plot_subnetwork_attributions(attribution_scores, out_dir, step)
    fig_dict.update(fig_dict_attributions)

    if config.topk is not None:
        if dataloader is not None and config.task_config.k > 1:
            fig_dict_correlations = plot_subnetwork_correlations(
                dataloader=dataloader,
                spd_model=model,
                config=config,
                device=device,
            )
            fig_dict.update(fig_dict_correlations)

        assert topk_mask is not None
        fig_dict_attributions = plot_subnetwork_attributions_statistics(topk_mask=topk_mask)
        fig_dict.update(fig_dict_attributions)

    # Save plots to files
    if out_dir:
        for k, v in fig_dict.items():
            out_file = out_dir / f"{k}_s{step}.png"
            v.savefig(out_file, dpi=200)
            tqdm.write(f"Saved plot to {out_file}")
    return fig_dict


def save_target_model_info(
    save_to_wandb: bool,
    out_dir: Path,
    target_model: ResidualLinearModel,
    target_model_config_dict: dict[str, Any],
    label_coeffs: list[float],
) -> None:
    torch.save(target_model.state_dict(), out_dir / "target_model.pth")

    with open(out_dir / "target_model_config.yaml", "w") as f:
        yaml.dump(target_model_config_dict, f, indent=2)

    with open(out_dir / "label_coeffs.json", "w") as f:
        json.dump(label_coeffs, f, indent=2)

    if save_to_wandb:
        wandb.save(str(out_dir / "target_model.pth"), base_path=out_dir)
        wandb.save(str(out_dir / "target_model_config.yaml"), base_path=out_dir)
        wandb.save(str(out_dir / "label_coeffs.json"), base_path=out_dir)


def main(
    config_path_or_obj: Path | str | Config, sweep_config_path: Path | str | None = None
) -> None:
    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, sweep_config_path)
        save_config_to_wandb(config)

    set_seed(config.seed)
    logger.info(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    assert isinstance(config.task_config, ResidualLinearConfig)

    target_model, target_model_config, label_coeffs = ResidualLinearModel.from_pretrained(
        config.task_config.pretrained_model_path
    )
    target_model = target_model.to(device)

    run_name = get_run_name(
        config,
        n_features=target_model.n_features,
        n_layers=target_model.n_layers,
        d_resid=target_model.d_embed,
        d_mlp=target_model.d_mlp,
    )
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name
    out_dir = Path(__file__).parent / "out" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "final_config.yaml", "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)

    save_target_model_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        target_model=target_model,
        target_model_config_dict=target_model_config,
        label_coeffs=label_coeffs,
    )

    model = ResidualLinearSPDFullRankModel(
        n_features=target_model.n_features,
        d_embed=target_model.d_embed,
        d_mlp=target_model.d_mlp,
        n_layers=target_model.n_layers,
        k=config.task_config.k,
        init_scale=config.task_config.init_scale,
    ).to(device)

    # Use the target_model's embedding matrix and don't train it further
    model.W_E.data[:, :] = target_model.W_E.data.detach().clone()
    model.W_E.requires_grad = False

    param_map = {}
    for i in range(target_model.n_layers):
        # Map from pretrained model's `all_decomposable_params` to the SPD models'
        # `all_subnetwork_params_summed`.
        param_map[f"layers.{i}.input_layer.weight"] = f"layers.{i}.input_layer.weight"
        param_map[f"layers.{i}.input_layer.bias"] = f"layers.{i}.input_layer.bias"
        param_map[f"layers.{i}.output_layer.weight"] = f"layers.{i}.output_layer.weight"
        param_map[f"layers.{i}.output_layer.bias"] = f"layers.{i}.output_layer.bias"

    dataset = ResidualLinearDataset(
        embed_matrix=model.W_E,
        n_features=model.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        label_coeffs=label_coeffs,
        one_feature_active=config.task_config.one_feature_active,
    )

    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    plot_results_fn = partial(resid_linear_plot_results_fn, dataloader=dataloader)
    optimize(
        model=model,
        config=config,
        device=device,
        dataloader=dataloader,
        pretrained_model=target_model,
        param_map=param_map,
        out_dir=out_dir,
        plot_results_fn=plot_results_fn,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
