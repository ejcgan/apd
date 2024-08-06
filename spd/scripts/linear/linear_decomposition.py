"""Linear decomposition script."""

import time
from pathlib import Path
from tempfile import TemporaryDirectory

import fire
import torch
import wandb
import yaml

from spd.log import logger
from spd.models.linear_models import DeepLinearComponentModel, DeepLinearModel
from spd.run_spd import Config, DeepLinearModelConfig, optimize
from spd.scripts.linear.linear_dataset import DeepLinearDataLoader, DeepLinearDataset
from spd.utils import (
    init_wandb,
    load_config,
    set_seed,
)

wandb.require("core")


def get_run_name(config: Config) -> str:
    """Generate a run name based on the config."""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        run_suffix = (
            f"sp{config.max_sparsity_coeff}_"
            f"lr{config.lr}_"
            f"p{config.pnorm}_"
            f"topk{config.topk}_"
            f"bs{config.batch_size}_"
        )
    return config.wandb_run_name_prefix + run_suffix


def main(
    config_path_or_obj: Path | str | Config, sweep_config_path: Path | str | None = None
) -> None:
    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, sweep_config_path)
        # Save the config to wandb
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "final_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config.model_dump(mode="json"), f, indent=2)
            wandb.save(str(config_path), policy="now", base_path=tmp_dir)
            # Unfortunately wandb.save is async, so we need to wait for it to finish before
            # continuing, and wandb python api provides no way to do this.
            # TODO: Find a better way to do this.
            time.sleep(1)

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
    model_config = config.torch_model_config
    assert isinstance(model_config, DeepLinearModelConfig)

    if model_config.pretrained_model_path is not None:
        dl_model = DeepLinearModel.from_pretrained(model_config.pretrained_model_path).to(device)
        assert (
            model_config.n_features is None
            and model_config.n_layers is None
            and model_config.n_instances is None
        ), "n_features, n_layers, and n_instances must not be set if pretrained_model_path is set"
        n_features = dl_model.n_features
        n_layers = dl_model.n_layers
        n_instances = dl_model.n_instances
    else:
        assert config.loss_type == "behavioral", "Only behavioral loss allows no pretrained model"
        dl_model = None
        n_features = model_config.n_features
        n_layers = model_config.n_layers
        n_instances = model_config.n_instances
        assert (
            n_features is not None and n_layers is not None and n_instances is not None
        ), "n_features, n_layers, and n_instances must be set"

    dlc_model = DeepLinearComponentModel(
        n_features=n_features, n_layers=n_layers, n_instances=n_instances, k=model_config.k
    ).to(device)

    dataset = DeepLinearDataset(n_features, n_instances)
    dataloader = DeepLinearDataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimize(
        model=dlc_model,
        config=config,
        out_dir=out_dir,
        device=device,
        dataloader=dataloader,
        pretrained_model=dl_model,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
