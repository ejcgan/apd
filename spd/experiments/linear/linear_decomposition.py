"""Linear decomposition script."""

from pathlib import Path

import fire
import torch
import wandb

from spd.experiments.linear.linear_dataset import DeepLinearDataset
from spd.experiments.linear.models import (
    DeepLinearComponentFullRankModel,
    DeepLinearComponentModel,
    DeepLinearModel,
)
from spd.experiments.linear.plotting import (
    make_linear_plots,
)
from spd.log import logger
from spd.run_spd import Config, DeepLinearConfig, optimize
from spd.utils import (
    DatasetGeneratedDataLoader,
    init_wandb,
    load_config,
    save_config_to_wandb,
    set_seed,
)

wandb.require("core")


def get_run_name(config: Config, n_features: int) -> str:
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
        if config.topk_act_recon_coeff is not None:
            run_suffix += f"topkactrecon_{config.topk_act_recon_coeff:.2e}_"
        run_suffix += f"lr{config.lr:.2e}_"
        run_suffix += f"bs{config.batch_size}_"
        run_suffix += f"ft{n_features}"
    return config.wandb_run_name_prefix + run_suffix


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

    run_name = get_run_name(config, n_features)
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name
    out_dir = Path(__file__).parent / "out" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

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

    param_map = None
    if config.task_config.pretrained_model_path:
        # Map from pretrained model's `all_decomposable_params` to the SPD models'
        # `all_subnetwork_params_summed`.
        param_map = {f"layer_{i}": f"layer_{i}" for i in range(n_layers)}

    dataset = DeepLinearDataset(n_features, n_instances)
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimize(
        model=dlc_model,
        config=config,
        out_dir=out_dir,
        device=device,
        dataloader=dataloader,
        pretrained_model=dl_model,
        param_map=param_map,
        plot_results_fn=make_linear_plots,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
