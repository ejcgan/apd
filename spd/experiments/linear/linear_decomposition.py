"""Linear decomposition script."""

from datetime import datetime
from pathlib import Path

import fire
import torch
import wandb
import yaml

from spd.experiments.linear.linear_dataset import DeepLinearDataset
from spd.experiments.linear.models import (
    DeepLinearComponentFullRankModel,
    DeepLinearModel,
)
from spd.experiments.linear.plotting import (
    make_linear_plots,
)
from spd.log import logger
from spd.run_spd import Config, DeepLinearConfig, get_common_run_name_suffix, optimize
from spd.utils import (
    DatasetGeneratedDataLoader,
    load_config,
    set_seed,
)
from spd.wandb_utils import init_wandb

wandb.require("core")


def get_run_name(config: Config, n_features: int) -> str:
    """Generate a run name based on the config."""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        run_suffix = get_common_run_name_suffix(config)
        run_suffix += f"ft{n_features}"
    return config.wandb_run_name_prefix + run_suffix


def main(
    config_path_or_obj: Path | str | Config, sweep_config_path: Path | str | None = None
) -> None:
    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, sweep_config_path)

    set_seed(config.seed)
    logger.info(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    assert isinstance(config.task_config, DeepLinearConfig)

    dl_model = DeepLinearModel.from_pretrained(config.task_config.pretrained_model_path).to(device)
    assert (
        config.task_config.n_features is None
        and config.task_config.n_layers is None
        and config.task_config.n_instances is None
    ), "n_features, n_layers, and n_instances must not be set if pretrained_model_path is set"
    n_features = dl_model.n_features
    n_layers = dl_model.n_layers
    n_instances = dl_model.n_instances

    run_name = get_run_name(config, n_features)
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = Path(__file__).parent / "out" / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "final_config.yaml", "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    if config.wandb_project:
        wandb.save(str(out_dir / "final_config.yaml"), base_path=out_dir, policy="now")

    if config.spd_type == "full_rank":
        dlc_model = DeepLinearComponentFullRankModel(
            n_features=n_features,
            n_layers=n_layers,
            n_instances=n_instances,
            k=config.task_config.k,
        ).to(device)
    else:
        raise ValueError(f"Unknown/unsupported SPD type: {config.spd_type}")

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
