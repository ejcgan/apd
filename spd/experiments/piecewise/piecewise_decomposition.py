"""Linear decomposition script."""

import json
from functools import partial
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import torch
import wandb
import yaml
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from spd.experiments.piecewise.models import (
    PiecewiseFunctionSPDFullRankTransformer,
    PiecewiseFunctionSPDRankPenaltyTransformer,
    PiecewiseFunctionSPDTransformer,
    PiecewiseFunctionTransformer,
)
from spd.experiments.piecewise.piecewise_dataset import PiecewiseDataset
from spd.experiments.piecewise.plotting import (
    plot_components_fullrank,
    plot_model_functions,
    plot_piecewise_network,
)
from spd.experiments.piecewise.trig_functions import generate_trig_functions
from spd.log import logger
from spd.plotting import plot_subnetwork_attributions_statistics, plot_subnetwork_correlations
from spd.run_spd import (
    Config,
    PiecewiseConfig,
    calc_recon_mse,
    get_common_run_name_suffix,
    optimize,
)
from spd.utils import BatchedDataLoader, load_config, set_seed
from spd.wandb_utils import init_wandb

wandb.require("core")


def piecewise_plot_results_fn(
    model: PiecewiseFunctionSPDFullRankTransformer | PiecewiseFunctionSPDRankPenaltyTransformer,
    target_model: PiecewiseFunctionTransformer,
    step: int,
    out_dir: Path | None,
    device: str,
    config: Config,
    topk_mask: Float[Tensor, " batch_size k"] | None,
    dataloader: BatchedDataLoader[tuple[Float[Tensor, " n_inputs"], Float[Tensor, ""]]]
    | None = None,
    **_,
) -> dict[str, plt.Figure]:
    assert isinstance(config.task_config, PiecewiseConfig)
    slow_images = config.slow_images
    fig_dict = {}
    # Plot functions
    if config.topk is not None:
        fig_dict_functions = plot_model_functions(
            spd_model=model,
            target_model=target_model,
            attribution_type=config.attribution_type,
            device=device,
            start=config.task_config.range_min,
            stop=config.task_config.range_max,
            print_info=False,
            distil_from_target=config.distil_from_target,
        )
        fig_dict.update(fig_dict_functions)
        fig_dict_network = plot_piecewise_network(model)
        fig_dict.update(fig_dict_network)

    if config.topk is not None:
        if dataloader is not None:
            # Plot correlations
            fig_dict_correlations = plot_subnetwork_correlations(
                dataloader=dataloader,
                target_model=target_model,
                spd_model=model,
                config=config,
                device=device,
            )
            fig_dict.update(fig_dict_correlations)

        assert topk_mask is not None
        # Plot subnet attribution statistics
        fig_dict_attributions = plot_subnetwork_attributions_statistics(topk_mask=topk_mask)
        fig_dict.update(fig_dict_attributions)

    # Plot components
    if config.task_config.n_layers == 1:
        fig_dict_components = plot_components_fullrank(
            model=model, step=step, out_dir=out_dir, slow_images=slow_images
        )
        fig_dict.update(fig_dict_components)
    else:
        tqdm.write("Skipping component plots for >1 layer models")
    # Save plots to files
    if out_dir:
        for k, v in fig_dict.items():
            out_file = out_dir / f"{k}_s{step}.png"
            v.savefig(out_file, dpi=200)
            tqdm.write(f"Saved plot to {out_file}")
    return fig_dict


def get_run_name(config: Config) -> str:
    """Generate a run name based on the config."""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        assert isinstance(config.task_config, PiecewiseConfig)
        run_suffix = get_common_run_name_suffix(config)
        if config.task_config.target_seed is not None:
            run_suffix += f"target-seed{config.task_config.target_seed}_"
        if config.task_config.handcoded_AB:
            run_suffix += "hAB_"
        run_suffix += f"lay{config.task_config.n_layers}"

    return config.wandb_run_name_prefix + run_suffix


def get_model_and_dataloader(
    config: Config,
    device: str,
    out_dir: Path | None = None,
) -> tuple[
    PiecewiseFunctionTransformer,
    PiecewiseFunctionSPDTransformer
    | PiecewiseFunctionSPDFullRankTransformer
    | PiecewiseFunctionSPDRankPenaltyTransformer,
    BatchedDataLoader[tuple[Float[Tensor, " n_inputs"], Float[Tensor, ""]]],
    BatchedDataLoader[tuple[Float[Tensor, " n_inputs"], Float[Tensor, ""]]],
]:
    """Set up the piecewise models and dataset."""
    assert isinstance(config.task_config, PiecewiseConfig)
    target_seed = (
        config.task_config.target_seed
        if config.task_config.target_seed is not None
        else config.seed
    )
    # Set seed for function generation and handcoded parameter setting
    set_seed(target_seed)
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

    # Assert that the output bias is 0
    for i in range(piecewise_model.n_layers):
        assert torch.allclose(
            piecewise_model.mlps[i].output_layer.bias,
            torch.zeros_like(piecewise_model.mlps[i].output_layer.bias),
        )

    set_seed(config.seed)

    # Initialize the SPD model
    if config.spd_type == "full_rank":
        piecewise_model_spd = PiecewiseFunctionSPDFullRankTransformer(
            n_inputs=piecewise_model.n_inputs,
            d_mlp=piecewise_model.d_mlp,
            n_layers=piecewise_model.n_layers,
            k=config.task_config.k,
            init_scale=config.task_config.init_scale,
        )
    elif config.spd_type == "rank_penalty":
        piecewise_model_spd = PiecewiseFunctionSPDRankPenaltyTransformer(
            n_inputs=piecewise_model.n_inputs,
            d_mlp=piecewise_model.d_mlp,
            n_layers=piecewise_model.n_layers,
            k=config.task_config.k,
            init_scale=config.task_config.init_scale,
            m=config.m,
        )
    else:
        raise ValueError(f"Unknown/unsupported SPD type: {config.spd_type}")

    if config.distil_from_target:
        assert config.spd_type == "full_rank", "Distillation only supported for full rank"
        piecewise_model_spd.set_subnet_to_target(piecewise_model)

    # Copy the biases (never decomposed)
    for i in range(piecewise_model_spd.n_layers):
        # Copy input biases from model & set requires_grad=False
        piecewise_model_spd.mlps[i].linear1.bias.data[:] = (
            piecewise_model.mlps[i].input_layer.bias.data.detach().clone()
        )
        piecewise_model_spd.mlps[i].linear1.bias.requires_grad_(False)
        # Make sure that there is no output bias
        assert piecewise_model_spd.mlps[i].linear2.bias is None

    # Handcoded the parameters if requested
    if config.task_config.handcoded_AB:
        if config.task_config.n_layers > 1:
            raise ValueError(
                "Handcoded AB not supported for >1 layer models due to an unsolved "
                "bug in the W_out matrices (noticed in full_rank, unsure about others)"
            )
        logger.info("Setting handcoded A and B matrices (!)")

        # Create a rank-one handcoded model & copy its SPD weights
        rank_one_spd_model = PiecewiseFunctionSPDTransformer(
            n_inputs=piecewise_model.n_inputs,
            d_mlp=piecewise_model.d_mlp,
            n_layers=piecewise_model.n_layers,
            k=config.task_config.k,
            init_scale=config.task_config.init_scale,
        )
        rank_one_spd_model.set_handcoded_spd_params(piecewise_model)
        piecewise_model_spd.set_handcoded_spd_params(rank_one_spd_model)

    piecewise_model_spd.to(device)

    piecewise_model_spd.W_E.requires_grad_(False)
    piecewise_model_spd.W_U.requires_grad_(False)

    train_dataset_seed = (
        config.task_config.dataset_seed
        if config.task_config.dataset_seed is not None
        else config.seed
    )
    dataset = PiecewiseDataset(
        n_inputs=piecewise_model.n_inputs,
        functions=functions,
        feature_probability=config.task_config.feature_probability,
        range_min=config.task_config.range_min,
        range_max=config.task_config.range_max,
        batch_size=config.batch_size,
        return_labels=False,
        dataset_seed=train_dataset_seed,
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
        dataset_seed=train_dataset_seed + 1,
    )
    test_dataloader = BatchedDataLoader(test_dataset)

    return piecewise_model, piecewise_model_spd, dataloader, test_dataloader


def main(
    config_path_or_obj: Path | str | Config, sweep_config_path: Path | str | None = None
) -> None:
    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, sweep_config_path)

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

    with open(out_dir / "final_config.yaml", "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    if config.wandb_project:
        wandb.save(str(out_dir / "final_config.yaml"), base_path=out_dir)

    piecewise_model, piecewise_model_spd, dataloader, test_dataloader = get_model_and_dataloader(
        config, device, out_dir
    )

    # Evaluate the hardcoded model on 5 batches to get the labels
    n_batches = 5
    loss = 0

    for i, (batch, labels) in enumerate(test_dataloader):
        if i >= n_batches:
            break
        hardcoded_out, _, _ = piecewise_model(batch.to(device))
        loss += calc_recon_mse(hardcoded_out, labels.to(device))
    loss /= n_batches
    logger.info(f"Loss of hardcoded model on 5 batches: {loss}")

    plot_results_fn = partial(
        piecewise_plot_results_fn,
        dataloader=test_dataloader,
    )

    # Map from pretrained model's `all_decomposable_params` to the SPD models'
    # `all_subnetwork_params_summed`.
    param_map = {}
    for i in range(piecewise_model_spd.n_layers):
        param_map[f"mlp_{i}.input_layer.weight"] = f"mlp_{i}.input_layer.weight"
        param_map[f"mlp_{i}.output_layer.weight"] = f"mlp_{i}.output_layer.weight"

    assert isinstance(
        piecewise_model_spd,
        PiecewiseFunctionSPDFullRankTransformer | PiecewiseFunctionSPDRankPenaltyTransformer,
    )
    optimize(
        model=piecewise_model_spd,
        config=config,
        out_dir=out_dir,
        device=device,
        pretrained_model=piecewise_model,
        param_map=param_map,
        dataloader=dataloader,
        plot_results_fn=plot_results_fn,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
